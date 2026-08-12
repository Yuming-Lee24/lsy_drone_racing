"""High-throughput end-to-end JAX PPO training entry point."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
from flax.training.train_state import TrainState

from lsy_drone_racing.rl_training.algorithm import build_optimizer, make_ppo_update
from lsy_drone_racing.rl_training.checkpoint import (
    POLICY_FILE,
    load_policy,
    restore_checkpoint,
    save_checkpoint,
)
from lsy_drone_racing.rl_training.config import get_profile
from lsy_drone_racing.rl_training.environment import (
    initialize_runner,
    make_rollout_collector,
    summarize_rollout,
)
from lsy_drone_racing.rl_training.evaluation import evaluate_params
from lsy_drone_racing.rl_training.runtime import (
    PROJECT_ROOT,
    create_environment,
    create_model,
    initialize_parameters,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lsy_drone_racing.rl_training.config import PPOConfig


def make_training_step(
    collector: Callable, ppo_update: Callable
) -> Callable[[Any, TrainState], tuple[Any, TrainState, dict, dict]]:
    """Chain the two compiled device stages without an intervening host synchronization.

    The rollout and optimizer are intentionally separate XLA executables. Wrapping both in another
    ``jit`` causes XLA to inline a very large simulation-and-backpropagation graph, increasing
    memory pressure and reducing throughput on consumer GPUs.
    """
    summarize = jax.jit(summarize_rollout)

    def training_step(runner: Any, train_state: TrainState) -> tuple[Any, TrainState, dict, dict]:
        runner, rollout, rollout_metrics = collector(runner, train_state.params)
        update_key, next_key = jax.random.split(runner.rng)
        runner = runner.replace(rng=next_key)
        train_state, optimizer_metrics = ppo_update(train_state, rollout, update_key)
        return runner, train_state, summarize(rollout_metrics), optimizer_metrics

    return training_step


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    host = jax.device_get(metrics)
    return {key: float(value) for key, value in host.items()}


def _write_metric(path: Path, metric: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metric, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("fast", "safe_fast"), default="fast")
    parser.add_argument("--config-file")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--update-epochs", type=int)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--disable-validation", action="store_true")
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Keep training to the configured budget after the evaluation target is reached.",
    )
    return parser


def _configuration(args: argparse.Namespace) -> PPOConfig:
    return get_profile(args.profile).with_overrides(
        config_file=args.config_file,
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        total_timesteps=args.total_timesteps,
    )


def _validation(
    params: Any,
    config: PPOConfig,
    episodes: int,
    output: Path,
    total_steps: int,
    *,
    seed: int | None = None,
) -> tuple[dict[str, Any], bool]:
    evaluation_seed = 20_000 + total_steps if seed is None else seed
    result = evaluate_params(params, config, episodes=episodes, seed=evaluation_seed)
    metric = {"total_steps": total_steps, **asdict(result)}
    _write_metric(output / "validation.jsonl", metric)
    meets_target = (
        result.success_rate >= config.success_target
        and result.mean_success_time <= config.mean_time_target
    )
    return metric, meets_target


def train(config: PPOConfig, args: argparse.Namespace) -> None:
    """Compile and run the device-resident PPO loop."""
    output = args.output or PROJECT_ROOT / "saves" / "jax_ppo" / config.profile
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    env = create_environment(config)
    runner = initialize_runner(env, config)
    model = create_model(env, config)
    params = initialize_parameters(model, runner.observation, config.seed)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=build_optimizer(config))

    if args.resume is not None and args.warm_start is not None:
        raise ValueError("--resume and --warm-start are mutually exclusive")
    warm_start = args.warm_start
    if warm_start is None and config.profile == "safe_fast" and args.resume is None:
        candidate = PROJECT_ROOT / "saves" / "jax_ppo" / "fast" / "best"
        if not (candidate / POLICY_FILE).exists():
            raise FileNotFoundError(
                f"safe_fast requires the best fast policy; expected {candidate / POLICY_FILE}"
            )
        warm_start = candidate
    if warm_start is not None:
        train_state = train_state.replace(params=load_policy(warm_start, train_state.params))
        print(f"Warm-started policy from {warm_start}; optimizer was reset.")

    update = 0
    total_steps = 0
    if args.resume is not None:
        train_state, runner, update, total_steps = restore_checkpoint(
            args.resume, train_state, runner
        )
        print(f"Resumed {args.resume} at update={update}, total_steps={total_steps}.")

    collector = make_rollout_collector(env, model, config)
    ppo_update = make_ppo_update(config)
    training_step = make_training_step(collector, ppo_update)
    maximum_updates = config.num_updates
    if args.max_updates is not None:
        maximum_updates = min(maximum_updates, update + args.max_updates)
    if update >= maximum_updates:
        save_checkpoint(output / "latest", train_state, runner, update, total_steps, config)
        env.close()
        print(f"No updates requested; checkpoint preserved at {output / 'latest'}.")
        return

    print(
        f"Compiling rollout and PPO stages on {jax.devices(config.device)[0]} "
        f"for {config.num_envs} envs × {config.num_steps} steps..."
    )
    compile_start = time.perf_counter()
    runner, train_state, rollout_metrics, optimizer_metrics = training_step(runner, train_state)
    jax.block_until_ready(train_state.step)
    compile_seconds = time.perf_counter() - compile_start
    update += 1
    total_steps += config.batch_size
    print(f"XLA compile + first update: {compile_seconds:.2f}s (excluded from steady SPS).")
    _write_metric(
        output / "performance.jsonl",
        {
            "kind": "compile_and_first_update",
            "seconds": compile_seconds,
            "batch_size": config.batch_size,
            "total_steps": total_steps,
        },
    )

    interval_start = time.perf_counter()
    interval_steps = 0
    next_checkpoint = ((total_steps // config.checkpoint_interval) + 1) * config.checkpoint_interval
    next_validation = ((total_steps // config.validation_interval) + 1) * config.validation_interval
    best_success = -1.0
    best_time = float("inf")
    stopped_early = False

    try:
        while update < maximum_updates:
            runner, train_state, rollout_metrics, optimizer_metrics = training_step(
                runner, train_state
            )
            update += 1
            total_steps += config.batch_size
            interval_steps += config.batch_size

            if update % args.log_interval == 0 or update == maximum_updates:
                jax.block_until_ready(train_state.step)
                now = time.perf_counter()
                elapsed = now - interval_start
                sps = interval_steps / elapsed if elapsed else float("inf")
                metric = {
                    "kind": "steady",
                    "update": update,
                    "total_steps": total_steps,
                    "seconds": elapsed,
                    "end_to_end_sps": sps,
                    **{
                        f"rollout/{key}": value
                        for key, value in _scalar_metrics(rollout_metrics).items()
                    },
                    **{
                        f"optimizer/{key}": value
                        for key, value in _scalar_metrics(optimizer_metrics).items()
                    },
                }
                _write_metric(output / "metrics.jsonl", metric)
                print(
                    f"step={total_steps:,} sps={sps:,.0f} "
                    f"success={metric['rollout/success_rate']:.3f} "
                    f"reward={metric['rollout/episode_reward']:.2f}"
                )
                interval_start = now
                interval_steps = 0

            if total_steps >= next_checkpoint:
                save_checkpoint(output / "latest", train_state, runner, update, total_steps, config)
                next_checkpoint += config.checkpoint_interval
                interval_start = time.perf_counter()
                interval_steps = 0

            if not args.disable_validation and total_steps >= next_validation:
                validation, meets_target = _validation(
                    train_state.params, config, config.validation_episodes, output, total_steps
                )
                print(
                    f"validation success={validation['success_rate']:.3f} "
                    f"mean_success_time={validation['mean_success_time']:.3f}s"
                )
                is_better = validation["success_rate"] > best_success or (
                    validation["success_rate"] == best_success
                    and validation["mean_success_time"] < best_time
                )
                if is_better:
                    best_success = validation["success_rate"]
                    best_time = validation["mean_success_time"]
                    save_checkpoint(
                        output / "best", train_state, runner, update, total_steps, config
                    )
                next_validation += config.validation_interval
                if meets_target and not args.disable_early_stopping:
                    final, _ = _validation(
                        train_state.params,
                        config,
                        config.final_evaluation_episodes,
                        output,
                        total_steps,
                        seed=30_000,
                    )
                    stopped_early = (
                        final["success_rate"] >= config.success_target
                        and final["mean_success_time"] <= config.mean_time_target
                    )
                    if stopped_early:
                        print("Final 500-scenario target passed; stopping early.")
                        break
                interval_start = time.perf_counter()
                interval_steps = 0
    except KeyboardInterrupt:
        print("Interrupted; saving an exact continuation checkpoint.")
    finally:
        save_checkpoint(output / "latest", train_state, runner, update, total_steps, config)
        env.close()

    if args.disable_early_stopping and update >= maximum_updates:
        final, _ = _validation(
            train_state.params,
            config,
            config.final_evaluation_episodes,
            output,
            total_steps,
            seed=30_000,
        )
        print(
            f"final success={final['success_rate']:.3f} "
            f"mean_success_time={final['mean_success_time']:.3f}s"
        )

    if stopped_early:
        status = "target_reached"
    elif update >= maximum_updates:
        status = "completed"
    else:
        status = "stopped"
    print(
        f"Training {status} at {total_steps:,} environment steps. Checkpoint: {output / 'latest'}"
    )


def main() -> None:
    """Parse command-line arguments and start training."""
    args = _parser().parse_args()
    train(_configuration(args), args)


if __name__ == "__main__":
    main()
