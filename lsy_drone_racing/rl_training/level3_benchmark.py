"""Benchmark fused Level 3 JAX PPO and enforce its same-environment speed gate."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import jax

from lsy_drone_racing.rl_training.benchmark import GpuSampler
from lsy_drone_racing.rl_training.level3_algorithm import (
    Level3TrainState,
    build_level3_optimizer,
    make_level3_ppo_update,
)
from lsy_drone_racing.rl_training.level3_config import get_level3_profile
from lsy_drone_racing.rl_training.level3_environment import (
    initialize_level3_runner,
    make_level3_rollout_collector,
)
from lsy_drone_racing.rl_training.level3_runtime import (
    create_level3_environment,
    create_level3_model,
    initialize_level3_parameters,
)
from lsy_drone_racing.rl_training.level3_training import make_level3_training_step
from lsy_drone_racing.rl_training.runtime import PROJECT_ROOT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("level3_v2", "level3_v2_segment"), default="level3_v2"
    )
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--pytorch-report", required=True, type=Path)
    parser.add_argument("--minimum-speedup", type=float, default=2.0)
    parser.add_argument("--minimum-sps", type=float, default=171_000.0)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Measure compile, rollout, optimizer, and end-to-end steady throughput separately."""
    args = _parser().parse_args()
    baseline_report = json.loads(args.pytorch_report.read_text(encoding="utf-8"))
    baseline_sps = float(baseline_report["end_to_end_sps"])
    config = get_level3_profile(args.profile).with_overrides(
        device=args.device,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        total_timesteps=max(args.timesteps, args.num_envs * args.num_steps),
    )
    env = create_level3_environment(config)
    runner = initialize_level3_runner(env, config)
    model = create_level3_model(env, config)
    params = initialize_level3_parameters(
        model, runner.actor_observation, runner.critic_observation, config.seed
    )
    state = Level3TrainState.create(
        apply_fn=model.apply, params=params, tx=build_level3_optimizer(config)
    )
    collector = make_level3_rollout_collector(env, model, config)
    updater = make_level3_ppo_update(config)

    rollout_compile_start = time.perf_counter()
    rollout_runner, rollout, _ = collector(runner, state.params, 0.0)
    jax.block_until_ready(rollout.reward)
    rollout_compile_seconds = time.perf_counter() - rollout_compile_start
    optimizer_compile_start = time.perf_counter()
    optimizer_state, _ = updater(state, rollout, jax.random.key(config.seed))
    jax.block_until_ready(optimizer_state.step)
    optimizer_compile_seconds = time.perf_counter() - optimizer_compile_start
    runner, state = rollout_runner, optimizer_state
    training_step = make_level3_training_step(collector, updater)

    first_start = time.perf_counter()
    first_result = training_step(runner, state, 0.0)
    jax.block_until_ready(first_result[1].step)
    first_end_to_end_seconds = time.perf_counter() - first_start
    runner, state = first_result[:2]
    # The production updater donates train-state buffers. Isolated optimizer timing therefore
    # needs its own state tree so it cannot invalidate the state used by the end-to-end section.
    optimizer_state = jax.tree.map(lambda value: value.copy(), state)

    repeats = max(2, math.ceil(args.timesteps / config.batch_size))
    with GpuSampler() as sampler:
        rollout_start = time.perf_counter()
        for _ in range(repeats):
            rollout_runner, rollout, _ = collector(rollout_runner, state.params, 0.0)
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
            runner, state, _, _ = training_step(runner, state, 0.0)
        jax.block_until_ready(state.step)
        end_to_end_seconds = time.perf_counter() - end_to_end_start

    measured_steps = repeats * config.batch_size
    end_to_end_sps = measured_steps / end_to_end_seconds
    speedup = end_to_end_sps / baseline_sps
    report = {
        "implementation": f"fused-{args.profile.replace('_', '-')}-jax-ppo",
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
        "baseline_label": baseline_report.get("implementation", str(args.pytorch_report)),
        "speedup": speedup,
        "minimum_speedup": args.minimum_speedup,
        "minimum_end_to_end_sps": args.minimum_sps,
        "performance_gate_passed": (
            speedup >= args.minimum_speedup and end_to_end_sps >= args.minimum_sps
        ),
        "estimated_500m_training_hours_excluding_validation": (
            500_000_000 / end_to_end_sps / 3600.0
        ),
        **sampler.summary(),
    }
    default_name = (
        "level3_v2_final_jax_10m.json"
        if args.profile == "level3_v2"
        else "level3_v2_segment_jax_10m.json"
    )
    output = args.output or PROJECT_ROOT / "saves" / "jax_ppo" / "performance" / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    env.close()
    if not report["performance_gate_passed"]:
        raise SystemExit(
            f"Level 3 JAX performance ({end_to_end_sps:,.0f} SPS, {speedup:.2f}x) is below "
            f"the {args.minimum_sps:,.0f} SPS / {args.minimum_speedup:.2f}x gates; "
            "do not start long training."
        )


if __name__ == "__main__":
    main()
