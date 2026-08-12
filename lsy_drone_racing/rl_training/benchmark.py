"""Measure fused JAX PPO throughput and enforce the PyTorch speedup gate."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
from flax.training.train_state import TrainState

from lsy_drone_racing.rl_training.algorithm import build_optimizer, make_ppo_update
from lsy_drone_racing.rl_training.config import get_profile
from lsy_drone_racing.rl_training.environment import initialize_runner, make_rollout_collector
from lsy_drone_racing.rl_training.runtime import (
    PROJECT_ROOT,
    create_environment,
    create_model,
    initialize_parameters,
)
from lsy_drone_racing.rl_training.training import make_training_step


class GpuSampler:
    """Sample NVIDIA utilization and memory without touching the training device state."""

    def __init__(self) -> None:
        """Initialize an idle sampler."""
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.splitlines()[0]
                utilization, memory = (float(value.strip()) for value in output.split(","))
                self.samples.append((utilization, memory))
            except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
                return

    def __enter__(self) -> GpuSampler:
        """Start background sampling."""
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Stop background sampling."""
        self._stop.set()
        self._thread.join(timeout=1.0)

    def summary(self) -> dict[str, float]:
        """Return median utilization and peak allocated memory."""
        if not self.samples:
            return {"gpu_utilization_median_percent": math.nan, "gpu_memory_peak_mib": math.nan}
        samples = np.asarray(self.samples)
        return {
            "gpu_utilization_median_percent": float(np.median(samples[:, 0])),
            "gpu_memory_peak_mib": float(np.max(samples[:, 1])),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("fast", "safe_fast"), default="fast")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=96)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--update-epochs", type=int)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--pytorch-baseline-sps", type=float, default=11_627.0)
    parser.add_argument("--pytorch-report", type=Path)
    parser.add_argument("--baseline-label", default="historical-25ws-main-average")
    parser.add_argument("--minimum-speedup", type=float, default=2.0)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "saves" / "jax_ppo" / "benchmark.json"
    )
    return parser


def main() -> None:
    """Run isolated rollout/optimizer timings followed by end-to-end steady timing."""
    args = _parser().parse_args()
    baseline_sps = args.pytorch_baseline_sps
    baseline_label = args.baseline_label
    if args.pytorch_report is not None:
        baseline_report = json.loads(args.pytorch_report.read_text(encoding="utf-8"))
        baseline_sps = float(baseline_report["end_to_end_sps"])
        baseline_label = str(baseline_report.get("implementation", args.pytorch_report))
    config = get_profile(args.profile).with_overrides(
        device=args.device,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        total_timesteps=max(args.timesteps, args.num_envs * args.num_steps),
    )
    env = create_environment(config)
    runner = initialize_runner(env, config)
    model = create_model(env, config)
    params = initialize_parameters(model, runner.observation, config.seed)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=build_optimizer(config))
    collector = make_rollout_collector(env, model, config)
    updater = make_ppo_update(config)
    rollout_compile_start = time.perf_counter()
    rollout_runner, rollout, _ = collector(runner, state.params)
    jax.block_until_ready(rollout.reward)
    rollout_compile_seconds = time.perf_counter() - rollout_compile_start
    optimizer_compile_start = time.perf_counter()
    optimizer_state, _ = updater(state, rollout, jax.random.key(config.seed))
    jax.block_until_ready(optimizer_state.step)
    optimizer_compile_seconds = time.perf_counter() - optimizer_compile_start
    runner, state = rollout_runner, optimizer_state
    training_step = make_training_step(collector, updater)

    first_end_to_end_start = time.perf_counter()
    first_result = training_step(runner, state)
    jax.block_until_ready(first_result[1].step)
    first_end_to_end_seconds = time.perf_counter() - first_end_to_end_start
    runner, state = first_result[:2]

    repeats = max(2, math.ceil(args.timesteps / config.batch_size))
    with GpuSampler() as sampler:
        rollout_start = time.perf_counter()
        for _ in range(repeats):
            rollout_runner, rollout, _ = collector(rollout_runner, state.params)
        jax.block_until_ready(rollout.reward)
        rollout_seconds = time.perf_counter() - rollout_start

        optimizer_start = time.perf_counter()
        for index in range(repeats):
            optimizer_state, _ = updater(
                optimizer_state, rollout, jax.random.key(config.seed + index)
            )
        jax.block_until_ready(optimizer_state.step)
        optimizer_seconds = time.perf_counter() - optimizer_start

        end_to_end_start = time.perf_counter()
        for _ in range(repeats):
            runner, state, _, _ = training_step(runner, state)
        jax.block_until_ready(state.step)
        end_to_end_seconds = time.perf_counter() - end_to_end_start

    measured_steps = repeats * config.batch_size
    end_to_end_sps = measured_steps / end_to_end_seconds
    speedup = end_to_end_sps / baseline_sps
    report = {
        "config": asdict(config),
        "first_end_to_end_update_seconds": first_end_to_end_seconds,
        "xla_rollout_compile_seconds": rollout_compile_seconds,
        "xla_optimizer_compile_seconds": optimizer_compile_seconds,
        "measured_environment_steps": measured_steps,
        "rollout_sps": measured_steps / rollout_seconds,
        "optimizer_sample_sps": measured_steps / optimizer_seconds,
        "optimizer_epoch_sample_sps": measured_steps * config.update_epochs / optimizer_seconds,
        "end_to_end_sps": end_to_end_sps,
        "pytorch_baseline_sps": baseline_sps,
        "baseline_label": baseline_label,
        "speedup": speedup,
        "minimum_speedup": args.minimum_speedup,
        "performance_gate_passed": speedup >= args.minimum_speedup,
        "estimated_500m_training_hours_excluding_validation": 500_000_000 / end_to_end_sps / 3600.0,
        **sampler.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    env.close()
    if not report["performance_gate_passed"]:
        raise SystemExit(
            f"JAX speedup {speedup:.2f}x is below the required {args.minimum_speedup:.2f}x; "
            "do not start long training."
        )


if __name__ == "__main__":
    main()
