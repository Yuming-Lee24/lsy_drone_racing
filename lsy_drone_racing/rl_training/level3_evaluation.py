"""Deterministic fixed-seed evaluation for Level 3 V2 policies."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from lsy_drone_racing.rl_training.evaluation import wilson_interval
from lsy_drone_racing.rl_training.level3_checkpoint import load_level3_policy
from lsy_drone_racing.rl_training.level3_config import LEVEL3_PROFILE_NAMES, get_level3_profile
from lsy_drone_racing.rl_training.level3_environment import (
    initialize_level3_runner,
    make_level3_evaluation_collector,
)
from lsy_drone_racing.rl_training.level3_runtime import (
    create_level3_environment,
    create_level3_model,
    initialize_level3_parameters,
)

if TYPE_CHECKING:
    from lsy_drone_racing.rl_training.level3_config import Level3PPOConfig


@dataclass(frozen=True)
class Level3EvaluationResult:
    """Metrics from exactly one original-Level-3 episode per fixed seed."""

    episodes: int
    successes: int
    collisions: int
    success_rate: float
    success_rate_ci_low: float
    success_rate_ci_high: float
    mean_success_time: float
    median_success_time: float
    p95_success_time: float
    mean_episode_return: float
    collision_rate: float
    truncation_rate: float
    gate_collision_rate: float
    obstacle_collision_rate: float
    ground_collision_rate: float
    boundary_collision_rate: float
    failures_by_gate: dict[str, int]
    total_gates_passed: int
    mean_gates_passed: float
    mean_min_gate_center_distance: float
    mean_min_gate_plane_distance: float
    mean_min_lateral_error_near_plane: float
    reached_gate_plane_neighborhood_rate: float
    safe_gate_prediction_rate: float
    current_gate_ever_visited_rate: float
    current_gate_pass_given_visit_rate: float
    mean_first_visit_gate_distance: float
    mean_first_visit_lateral_error: float
    mean_first_visit_correction_distance: float
    mean_unsafe_near_plane_stall_time: float
    mean_route_potential_reduction: float
    mean_terminal_speed: float


def evaluate_level3_params(
    params: Any,
    config: Level3PPOConfig,
    *,
    episodes: int,
    batch_size: int = 100,
    seed: int = 30_000,
) -> Level3EvaluationResult:
    """Evaluate a deterministic Actor on full-difficulty independently seeded tracks."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    successes: list[bool] = []
    collisions: list[bool] = []
    truncations: list[bool] = []
    gate_collisions: list[bool] = []
    obstacle_collisions: list[bool] = []
    ground_collisions: list[bool] = []
    boundary_collisions: list[bool] = []
    failed_gates: list[int] = []
    lengths: list[int] = []
    returns: list[float] = []
    min_gate_center_distances: list[float] = []
    min_gate_plane_distances: list[float] = []
    min_lateral_errors_near_plane: list[float] = []
    reached_gate_plane_neighborhoods: list[bool] = []
    safe_gate_predictions: list[bool] = []
    current_gate_ever_visited: list[bool] = []
    current_gate_passed_after_visit: list[bool] = []
    first_visit_gate_distances: list[float] = []
    first_visit_lateral_errors: list[float] = []
    first_visit_correction_distances: list[float] = []
    unsafe_near_plane_stall_times: list[float] = []
    route_potential_reductions: list[float] = []
    terminal_speeds: list[float] = []
    frequency = 0
    collectors: dict[int, tuple[Any, Any]] = {}

    for start in range(0, episodes, batch_size):
        count = min(batch_size, episodes - start)
        eval_config = config.with_overrides(
            curriculum=False,
            num_envs=count,
            num_steps=config.max_episode_steps,
            num_minibatches=1,
            update_epochs=1,
            total_timesteps=count * config.max_episode_steps,
        )
        if count not in collectors:
            env = create_level3_environment(eval_config, seed=seed, curriculum=False)
            model = create_level3_model(env, eval_config)
            collect = make_level3_evaluation_collector(env, model, eval_config)
            collectors[count] = (env, collect)
        env, collect = collectors[count]
        runner = initialize_level3_runner(env, eval_config, seed=seed + start, mixed_segment=False)
        _, metrics = collect(runner, params)
        metrics = jax.device_get(metrics)
        frequency = env.settings.freq
        num_passes = int(env.data.gate_sequence.shape[0])

        done = np.asarray(metrics.done)
        if not np.all(done.any(axis=0)):
            raise RuntimeError("Evaluation ended before every world reached one terminal state")
        first = np.argmax(done, axis=0)
        world = np.arange(count)
        batch_success = np.asarray(metrics.success)[first, world]
        batch_collision = np.asarray(metrics.collision)[first, world]
        successes.extend(batch_success.tolist())
        collisions.extend(batch_collision.tolist())
        truncations.extend(np.asarray(metrics.timeout)[first, world].tolist())
        gate_collisions.extend(np.asarray(metrics.gate_collision)[first, world].tolist())
        obstacle_collisions.extend(np.asarray(metrics.obstacle_collision)[first, world].tolist())
        ground_collisions.extend(np.asarray(metrics.ground_collision)[first, world].tolist())
        boundary_collisions.extend(np.asarray(metrics.boundary_collision)[first, world].tolist())
        failed_gates.extend(np.asarray(metrics.failed_gate)[first, world].tolist())
        lengths.extend(np.asarray(metrics.episode_length)[first, world].tolist())
        components = np.asarray(metrics.episode_components)[first, world]
        returns.extend(components.sum(axis=-1).tolist())

        step_index = np.arange(done.shape[0])[:, None]
        active = step_index <= first[None]
        gate_position = np.asarray(metrics.gate_position)
        centre_distance = np.linalg.norm(gate_position, axis=-1)
        plane_distance = np.abs(gate_position[..., 0])
        lateral_error = np.linalg.norm(gate_position[..., 1:], axis=-1)
        near_plane = active & (plane_distance <= 0.20)
        masked_centre = np.where(active, centre_distance, np.inf)
        masked_plane = np.where(active, plane_distance, np.inf)
        masked_lateral = np.where(near_plane, lateral_error, np.inf)
        min_gate_center_distances.extend(masked_centre.min(axis=0).tolist())
        min_gate_plane_distances.extend(masked_plane.min(axis=0).tolist())
        lateral_minimum = masked_lateral.min(axis=0)
        reached = np.isfinite(lateral_minimum)
        reached_gate_plane_neighborhoods.extend(reached.tolist())
        min_lateral_errors_near_plane.extend(lateral_minimum[reached].tolist())
        safe_prediction = np.asarray(metrics.gate_prediction_safe) & active
        safe_gate_predictions.extend(safe_prediction.any(axis=0).tolist())
        visited = np.asarray(metrics.current_gate_visited) & active
        ever_visited = visited.any(axis=0)
        current_gate_ever_visited.extend(ever_visited.tolist())
        terminal_progress = np.asarray(metrics.final_progress)[first, world]
        passed_first_gate = terminal_progress >= 1
        current_gate_passed_after_visit.extend((ever_visited & passed_first_gate).tolist())
        first_visit = np.argmax(visited, axis=0)
        visited_world = np.flatnonzero(ever_visited)
        if visited_world.size:
            visited_step = first_visit[visited_world]
            first_position = gate_position[visited_step, visited_world]
            first_visit_gate_distances.extend(np.linalg.norm(first_position, axis=-1).tolist())
            first_visit_lateral_errors.extend(
                np.linalg.norm(first_position[..., 1:], axis=-1).tolist()
            )
            first_visit_correction_distances.extend(
                np.asarray(metrics.gate_correction_distance)[visited_step, visited_world].tolist()
            )
        unsafe_near_plane_stall_times.extend(
            (
                np.sum(np.asarray(metrics.unsafe_near_plane_stall) & active, axis=0)
                / env.settings.freq
            ).tolist()
        )
        route_potential = np.asarray(metrics.route_potential)
        active_potential = np.where(active, route_potential, np.inf)
        route_potential_reductions.extend(
            (route_potential[0] - active_potential.min(axis=0)).tolist()
        )
        terminal_speeds.extend(np.asarray(metrics.speed)[first, world].tolist())

    for env, _ in collectors.values():
        env.close()

    successes_array = np.asarray(successes, dtype=bool)
    lengths_array = np.asarray(lengths)
    successful_times = lengths_array[successes_array] / frequency
    success_count = int(successes_array.sum())
    ci_low, ci_high = wilson_interval(success_count, episodes)
    failed_gate_array = np.asarray(failed_gates, dtype=int)
    failures_by_gate = {
        f"before_gate_{gate + 1}": int(np.sum((~successes_array) & (failed_gate_array == gate)))
        for gate in range(num_passes)
    }
    gates_passed = np.where(successes_array, num_passes, failed_gate_array)
    visited_array = np.asarray(current_gate_ever_visited, dtype=bool)
    passed_after_visit_array = np.asarray(current_gate_passed_after_visit, dtype=bool)
    return Level3EvaluationResult(
        episodes=episodes,
        successes=success_count,
        collisions=int(np.sum(collisions)),
        success_rate=success_count / episodes,
        success_rate_ci_low=ci_low,
        success_rate_ci_high=ci_high,
        mean_success_time=float(successful_times.mean()) if success_count else math.inf,
        median_success_time=float(np.median(successful_times)) if success_count else math.inf,
        p95_success_time=float(np.percentile(successful_times, 95)) if success_count else math.inf,
        mean_episode_return=float(np.mean(returns)),
        collision_rate=float(np.mean(collisions)),
        truncation_rate=float(np.mean(truncations)),
        gate_collision_rate=float(np.mean(gate_collisions)),
        obstacle_collision_rate=float(np.mean(obstacle_collisions)),
        ground_collision_rate=float(np.mean(ground_collisions)),
        boundary_collision_rate=float(np.mean(boundary_collisions)),
        failures_by_gate=failures_by_gate,
        total_gates_passed=int(gates_passed.sum()),
        mean_gates_passed=float(gates_passed.mean()),
        mean_min_gate_center_distance=float(np.mean(min_gate_center_distances)),
        mean_min_gate_plane_distance=float(np.mean(min_gate_plane_distances)),
        mean_min_lateral_error_near_plane=(
            float(np.mean(min_lateral_errors_near_plane))
            if min_lateral_errors_near_plane
            else math.inf
        ),
        reached_gate_plane_neighborhood_rate=float(np.mean(reached_gate_plane_neighborhoods)),
        safe_gate_prediction_rate=float(np.mean(safe_gate_predictions)),
        current_gate_ever_visited_rate=float(np.mean(visited_array)),
        current_gate_pass_given_visit_rate=(
            float(np.sum(passed_after_visit_array) / np.sum(visited_array))
            if np.any(visited_array)
            else 0.0
        ),
        mean_first_visit_gate_distance=(
            float(np.mean(first_visit_gate_distances)) if first_visit_gate_distances else math.inf
        ),
        mean_first_visit_lateral_error=(
            float(np.mean(first_visit_lateral_errors)) if first_visit_lateral_errors else math.inf
        ),
        mean_first_visit_correction_distance=(
            float(np.mean(first_visit_correction_distances))
            if first_visit_correction_distances
            else math.inf
        ),
        mean_unsafe_near_plane_stall_time=float(np.mean(unsafe_near_plane_stall_times)),
        mean_route_potential_reduction=float(np.mean(route_potential_reductions)),
        mean_terminal_speed=float(np.mean(terminal_speeds)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--profile", choices=LEVEL3_PROFILE_NAMES, default="level3_v2")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--config-file")
    parser.add_argument("--raw", action="store_true", help="Evaluate raw instead of EMA parameters")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    """Load one V2 checkpoint and print the fixed-seed report."""
    args = _parser().parse_args()
    config = get_level3_profile(args.profile).with_overrides(
        device=args.device, config_file=args.config_file
    )
    template_config = config.with_overrides(
        curriculum=False,
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=1,
    )
    env = create_level3_environment(template_config, curriculum=False)
    runner = initialize_level3_runner(env, template_config, mixed_segment=False)
    model = create_level3_model(env, template_config)
    template = initialize_level3_parameters(
        model, runner.actor_observation, runner.critic_observation, config.seed
    )
    params = load_level3_policy(args.checkpoint, template, ema=not args.raw)
    env.close()
    result = evaluate_level3_params(
        params, config, episodes=args.episodes, batch_size=args.batch_size, seed=args.seed
    )
    report = json.dumps(asdict(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
