"""High-throughput reliability-first Level 3 JAX PPO training."""

from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax

from lsy_drone_racing.rl_training.level3_algorithm import (
    Level3TrainState,
    build_level3_optimizer,
    make_level3_ppo_update,
)
from lsy_drone_racing.rl_training.level3_checkpoint import (
    load_level3_policy,
    restore_level3_checkpoint,
    save_level3_checkpoint,
)
from lsy_drone_racing.rl_training.level3_config import LEVEL3_PROFILE_NAMES, get_level3_profile
from lsy_drone_racing.rl_training.level3_environment import (
    initialize_level3_runner,
    make_level3_rollout_collector,
    summarize_level3_rollout,
)
from lsy_drone_racing.rl_training.level3_evaluation import evaluate_level3_params
from lsy_drone_racing.rl_training.level3_runtime import (
    create_level3_environment,
    create_level3_model,
    initialize_level3_parameters,
)
from lsy_drone_racing.rl_training.runtime import PROJECT_ROOT

if TYPE_CHECKING:
    from collections.abc import Callable

    from lsy_drone_racing.rl_training.level3_config import Level3PPOConfig


def make_level3_training_step(
    collector: Callable, ppo_update: Callable
) -> Callable[[Any, Level3TrainState, float], tuple[Any, Level3TrainState, dict, dict]]:
    """Keep rollout and optimizer as separate XLA executables without a host sync between them."""
    summarize = jax.jit(summarize_level3_rollout)

    def training_step(
        runner: Any, train_state: Level3TrainState, curriculum_progress: float
    ) -> tuple[Any, Level3TrainState, dict, dict]:
        runner, rollout, rollout_metrics = collector(
            runner, train_state.params, curriculum_progress
        )
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


def level3_validation_rank(metric: dict[str, Any]) -> tuple[int, int, int, float]:
    """Return the frozen lexicographic checkpoint-selection score."""
    return (
        int(metric["successes"]),
        int(metric["total_gates_passed"]),
        -int(metric["collisions"]),
        float(metric["mean_episode_return"]),
    )


def _existing_validation_ranks(output: Path) -> dict[str, tuple[int, int, int, float] | None]:
    """Recover best-selection state from the append-only validation log on exact resume."""
    ranks: dict[str, tuple[int, int, int, float] | None] = {"raw": None, "ema": None}
    path = output / "validation.jsonl"
    if not path.exists():
        return ranks
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            metric = json.loads(line)
            label = {"current_raw": "raw", "current_ema": "ema"}.get(metric.get("checkpoint"))
            if label is None:
                continue
            rank = level3_validation_rank(metric)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if ranks[label] is None or rank > ranks[label]:
            ranks[label] = rank
    return ranks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=LEVEL3_PROFILE_NAMES, default="level3_v2")
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
    parser.add_argument(
        "--initialize-policy",
        type=Path,
        help="Initialize V2 parameters from a policy while resetting optimizer/environment state.",
    )
    parser.add_argument(
        "--initialize-policy-ema",
        action="store_true",
        help="Load EMA instead of raw parameters from --initialize-policy.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--disable-validation", action="store_true")
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument(
        "--skip-final-evaluation",
        action="store_true",
        help="Used only by short pilots; full-budget runs must retain final evaluation.",
    )
    return parser


def _configuration(args: argparse.Namespace) -> Level3PPOConfig:
    return get_level3_profile(args.profile).with_overrides(
        config_file=args.config_file,
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        total_timesteps=args.total_timesteps,
        validation_episodes=args.validation_episodes,
    )


def _validation(
    params: Any,
    config: Level3PPOConfig,
    output: Path,
    total_steps: int,
    *,
    label: str,
    episodes: int | None = None,
    seed: int = 20_000,
) -> dict[str, Any]:
    result = evaluate_level3_params(
        params,
        config,
        episodes=config.validation_episodes if episodes is None else episodes,
        seed=seed,
    )
    metric = {"checkpoint": label, "total_steps": total_steps, **asdict(result)}
    _write_metric(output / "validation.jsonl", metric)
    return metric


def train_level3(config: Level3PPOConfig, args: argparse.Namespace) -> None:
    """Compile and run the no-early-stop Level 3 PPO experiment."""
    if args.resume is not None and args.initialize_policy is not None:
        raise ValueError("--resume and --initialize-policy are mutually exclusive")
    if args.initialize_policy_ema and args.initialize_policy is None:
        raise ValueError("--initialize-policy-ema requires --initialize-policy")
    if config.segment_reset.enabled and args.initialize_policy is not None:
        raise ValueError(
            f"{config.profile} must start from random parameters or exact-resume itself"
        )
    output = args.output or PROJECT_ROOT / "saves" / "jax_ppo" / config.profile
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    env = create_level3_environment(config)
    runner = initialize_level3_runner(env, config)
    model = create_level3_model(env, config)
    params = initialize_level3_parameters(
        model, runner.actor_observation, runner.critic_observation, config.seed
    )
    train_state = Level3TrainState.create(
        apply_fn=model.apply, params=params, tx=build_level3_optimizer(config)
    )
    update = 0
    total_steps = 0
    if args.resume is not None:
        train_state, runner, update, total_steps = restore_level3_checkpoint(
            args.resume, train_state, runner, config
        )
        print(f"Resumed {args.resume} at update={update}, total_steps={total_steps}.")
    elif args.initialize_policy is not None:
        params = load_level3_policy(
            args.initialize_policy, train_state.params, ema=args.initialize_policy_ema
        )
        train_state = Level3TrainState.create(
            apply_fn=model.apply, params=params, tx=build_level3_optimizer(config)
        )
        source = "EMA" if args.initialize_policy_ema else "raw"
        print(
            f"Initialized from {source} policy {args.initialize_policy}; "
            "optimizer, EMA, PRNG, environment, and curriculum were reset."
        )

    collector = make_level3_rollout_collector(env, model, config)
    ppo_update = make_level3_ppo_update(config)
    training_step = make_level3_training_step(collector, ppo_update)
    maximum_updates = config.num_updates
    if args.max_updates is not None:
        maximum_updates = min(maximum_updates, update + args.max_updates)
    if update >= maximum_updates:
        save_level3_checkpoint(output / "latest", train_state, runner, update, total_steps, config)
        env.close()
        print(f"No updates requested; checkpoint preserved at {output / 'latest'}.")
        return

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signum}; saving after the current PPO update.", flush=True)

    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    print(
        f"Compiling Level 3 rollout and PPO on {jax.devices(config.device)[0]} for "
        f"{config.num_envs} envs × {config.num_steps} steps..."
    )
    compile_start = time.perf_counter()
    runner, train_state, rollout_metrics, optimizer_metrics = training_step(
        runner, train_state, total_steps / config.curriculum_timesteps
    )
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
    best_rank = _existing_validation_ranks(output)

    try:
        while update < maximum_updates and not stop_requested:
            runner, train_state, rollout_metrics, optimizer_metrics = training_step(
                runner, train_state, total_steps / config.curriculum_timesteps
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
                    "curriculum_progress": min(total_steps / config.curriculum_timesteps, 1.0),
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
                save_level3_checkpoint(
                    output / "latest", train_state, runner, update, total_steps, config
                )
                next_checkpoint += config.checkpoint_interval
                interval_start = time.perf_counter()
                interval_steps = 0

            if not args.disable_validation and total_steps >= next_validation:
                for parameter_label, validation_params in (
                    ("raw", train_state.params),
                    ("ema", train_state.ema_params),
                ):
                    validation = _validation(
                        validation_params,
                        config,
                        output,
                        total_steps,
                        label=f"current_{parameter_label}",
                    )
                    print(
                        f"validation {parameter_label} "
                        f"success={validation['success_rate']:.3f} "
                        f"CI=[{validation['success_rate_ci_low']:.3f}, "
                        f"{validation['success_rate_ci_high']:.3f}]"
                    )
                    rank = level3_validation_rank(validation)
                    if best_rank[parameter_label] is None or rank > best_rank[parameter_label]:
                        best_rank[parameter_label] = rank
                        save_level3_checkpoint(
                            output / f"best_{parameter_label}",
                            train_state,
                            runner,
                            update,
                            total_steps,
                            config,
                        )
                next_validation += config.validation_interval
                interval_start = time.perf_counter()
                interval_steps = 0
    finally:
        save_level3_checkpoint(output / "latest", train_state, runner, update, total_steps, config)
        env.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)

    completed = update >= maximum_updates
    if completed and not args.skip_final_evaluation:
        for parameter_label, evaluation_params in (
            ("latest_raw", train_state.params),
            ("latest_ema", train_state.ema_params),
        ):
            result = _validation(
                evaluation_params,
                config,
                output,
                total_steps,
                label=parameter_label,
                episodes=config.final_evaluation_episodes,
                seed=30_000,
            )
            print(
                f"final {parameter_label} success={result['success_rate']:.3f} "
                f"CI-low={result['success_rate_ci_low']:.3f}"
            )
        for parameter_label in ("raw", "ema"):
            best_path = output / f"best_{parameter_label}"
            if best_path.exists():
                best_params = load_level3_policy(
                    best_path, train_state.params, ema=parameter_label == "ema"
                )
                _validation(
                    best_params,
                    config,
                    output,
                    total_steps,
                    label=f"best_{parameter_label}",
                    episodes=config.final_evaluation_episodes,
                    seed=30_000,
                )

    status = "completed" if completed else "stopped"
    print(f"Level 3 training {status} at {total_steps:,} steps. Checkpoint: {output / 'latest'}")


def main() -> None:
    """Parse the Level 3 CLI and run training."""
    args = _parser().parse_args()
    train_level3(_configuration(args), args)


if __name__ == "__main__":
    main()
