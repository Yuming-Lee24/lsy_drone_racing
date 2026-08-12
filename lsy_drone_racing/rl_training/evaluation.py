"""Fixed-seed deterministic evaluation for JAX PPO policies."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from lsy_drone_racing.rl_training.checkpoint import load_policy
from lsy_drone_racing.rl_training.config import get_profile
from lsy_drone_racing.rl_training.environment import initialize_runner, make_evaluation_collector
from lsy_drone_racing.rl_training.runtime import (
    create_environment,
    create_model,
    initialize_parameters,
)

if TYPE_CHECKING:
    from lsy_drone_racing.rl_training.config import PPOConfig


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics from exactly one episode per evaluation world."""

    episodes: int
    successes: int
    success_rate: float
    success_rate_ci_low: float
    success_rate_ci_high: float
    mean_success_time: float
    mean_episode_return: float
    collision_rate: float
    truncation_rate: float


def wilson_interval(
    successes: int, episodes: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial rate."""
    if episodes == 0:
        return math.nan, math.nan
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    centre = (proportion + z * z / (2.0 * episodes)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / episodes + z * z / (4.0 * episodes**2))
        / denominator
    )
    return centre - half_width, centre + half_width


def evaluate_params(
    params: Any, config: PPOConfig, *, episodes: int, batch_size: int = 100, seed: int = 10_000
) -> EvaluationResult:
    """Evaluate a policy deterministically on fixed, reproducible scenario batches."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    successes: list[bool] = []
    collisions: list[bool] = []
    truncations: list[bool] = []
    lengths: list[int] = []
    returns: list[float] = []
    frequency = 0
    collectors: dict[int, tuple[Any, Any, Any]] = {}

    for start in range(0, episodes, batch_size):
        count = min(batch_size, episodes - start)
        eval_config = config.with_overrides(
            num_envs=count,
            num_steps=config.max_episode_steps,
            num_minibatches=1,
            update_epochs=1,
            total_timesteps=count * config.max_episode_steps,
        )
        if count not in collectors:
            env = create_environment(eval_config, seed=seed + start)
            model = create_model(env, eval_config)
            collect = make_evaluation_collector(env, model, eval_config)
            collectors[count] = (env, model, collect)
        env, _, collect = collectors[count]
        frequency = env.settings.freq
        runner = initialize_runner(env, eval_config, seed=seed + start)
        _, metrics = collect(runner, params)
        metrics = jax.device_get(metrics)
        done = np.asarray(metrics.done)
        if not np.all(done.any(axis=0)):
            raise RuntimeError(
                "Evaluation rollout ended before every world reached a terminal state"
            )
        first = np.argmax(done, axis=0)
        world = np.arange(count)
        batch_success = np.asarray(metrics.success)[first, world]
        batch_collision = np.asarray(metrics.collision)[first, world]
        batch_length = np.asarray(metrics.episode_length)[first, world]
        components = np.asarray(metrics.episode_components)[first, world]
        successes.extend(batch_success.tolist())
        collisions.extend(batch_collision.tolist())
        truncations.extend((~batch_success & ~batch_collision).tolist())
        lengths.extend(batch_length.tolist())
        returns.extend(components.sum(axis=-1).tolist())

    for env, _, _ in collectors.values():
        env.close()

    successes_array = np.asarray(successes, dtype=bool)
    lengths_array = np.asarray(lengths)
    successful_times = lengths_array[successes_array] / frequency
    success_count = int(successes_array.sum())
    ci_low, ci_high = wilson_interval(success_count, episodes)
    return EvaluationResult(
        episodes=episodes,
        successes=success_count,
        success_rate=success_count / episodes,
        success_rate_ci_low=ci_low,
        success_rate_ci_high=ci_high,
        mean_success_time=float(successful_times.mean()) if success_count else math.inf,
        mean_episode_return=float(np.mean(returns)),
        collision_rate=float(np.mean(collisions)),
        truncation_rate=float(np.mean(truncations)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--profile", choices=("fast", "safe_fast"), default="fast")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--config-file")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Run checkpoint evaluation from the command line."""
    args = _parser().parse_args()
    config = get_profile(args.profile).with_overrides(
        device=args.device, config_file=args.config_file
    )
    template_config = config.with_overrides(
        num_envs=1, num_steps=1, num_minibatches=1, update_epochs=1, total_timesteps=1
    )
    env = create_environment(template_config)
    runner = initialize_runner(env, template_config)
    model = create_model(env, template_config)
    template = initialize_parameters(model, runner.observation, config.seed)
    params = load_policy(args.checkpoint, template)
    env.close()
    result = evaluate_params(
        params, config, episodes=args.episodes, batch_size=args.batch_size, seed=args.seed
    )
    report = json.dumps(asdict(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
