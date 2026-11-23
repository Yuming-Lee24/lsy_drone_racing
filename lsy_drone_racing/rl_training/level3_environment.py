"""Fused functional Level 3 adapter for structured JAX PPO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from flax import struct

from lsy_drone_racing.envs.race_core import (
    _mark_drones_for_reset,
    _update_disabled_drones,
    _update_target_gates,
    _update_visited_objects,
    terminated,
    truncated,
)
from lsy_drone_racing.rl_training.environment import scale_action
from lsy_drone_racing.rl_training.level3_algorithm import Level3RolloutBatch
from lsy_drone_racing.rl_training.level3_level_replay import (
    initialize_level_replay,
    sample_level_replay,
    update_level_replay,
)
from lsy_drone_racing.rl_training.level3_observations import (
    ActorObservation,
    CriticObservation,
    Level3ObservationMemory,
    advance_level3_memory,
    build_actor_observation,
    build_critic_observation,
    current_gate_geometry,
    initialize_level3_memory,
    mask_actor_observation,
)
from lsy_drone_racing.rl_training.level3_rewards import (
    LEVEL3_REWARD_COMPONENTS,
    Level3RewardState,
    initialize_level3_reward_state,
    level3_reward,
    reward_route_potential,
)
from lsy_drone_racing.rl_training.level3_runtime import belief_observation, true_observation
from lsy_drone_racing.rl_training.level3_segment_reset import (
    CORRIDOR_START,
    FULL_START,
    POST_GATE_START,
    PRE_GATE_START,
    apply_level3_segment_reset,
)
from lsy_drone_racing.rl_training.networks import (
    deterministic_action,
    sample_action,
    squashed_log_probability,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
    from lsy_drone_racing.envs.race_core import EnvData
    from lsy_drone_racing.rl_training.level3_config import Level3PPOConfig
    from lsy_drone_racing.rl_training.level3_networks import Level3ActorCritic


@struct.dataclass
class Level3RunnerState:
    """Every dynamic input carried through a Level 3 rollout scan."""

    environment_data: Any
    belief_observation: dict[str, Array]
    true_observation: dict[str, Array]
    observation_memory: Level3ObservationMemory
    reward_state: Level3RewardState
    actor_observation: ActorObservation
    critic_observation: CriticObservation
    episode_start_progress: Array
    episode_start_kind: Array
    rng: Array
    level_replay: Any


@struct.dataclass
class Level3RolloutMetrics:
    """Sparse completed-episode metrics and on-device curriculum telemetry."""

    done: Array
    success: Array
    collision: Array
    timeout: Array
    gate_collision: Array
    obstacle_collision: Array
    ground_collision: Array
    boundary_collision: Array
    failed_gate: Array
    episode_start_progress: Array
    episode_start_kind: Array
    final_progress: Array
    target_gate_passed: Array
    difficulty: Array
    gate_position: Array
    gate_velocity: Array
    gate_clearance: Array
    current_gate_visited: Array
    gate_correction_distance: Array
    unsafe_near_plane_stall: Array
    gate_prediction_safe: Array
    route_potential: Array
    speed: Array
    episode_components: Array
    episode_length: Array
    replayed: Array


@struct.dataclass
class Level3RolloutOutput:
    """PPO transition plus its monitoring values."""

    batch: Level3RolloutBatch
    metrics: Level3RolloutMetrics


@struct.dataclass
class Level3EnvironmentTransition:
    """One policy-independent Level 3 transition."""

    next_runner: Level3RunnerState
    terminal_actor_observation: ActorObservation
    terminal_critic_observation: CriticObservation
    reward: Array
    terminated: Array
    truncated: Array
    metrics: Level3RolloutMetrics


def _critic_features(
    env: VecDroneRaceEnv,
    data: EnvData,
    belief: dict[str, Array],
    actor: ActorObservation,
    memory: Level3ObservationMemory,
    config: Level3PPOConfig,
) -> CriticObservation:
    default = env.sim.default_data
    nominal_mass = default.params.mass.reshape((default.core.n_worlds, -1))[0, 0]
    nominal_inertia = jnp.diagonal(default.params.J[0, 0])
    return build_critic_observation(
        belief,
        actor,
        data.gates_pos,
        data.gates_quat,
        data.obstacles_pos,
        data.nominal_gates_pos,
        data.nominal_gates_quat,
        data.nominal_obstacles_pos,
        data.sim_data.params.mass,
        nominal_mass,
        data.sim_data.params.J,
        nominal_inertia,
        data.sim_data.states.force,
        data.difficulty,
        memory,
        data.pos_limit_low,
        data.pos_limit_high,
        config.geometry,
    )


def _actor_features(
    data: EnvData,
    belief: dict[str, Array],
    memory: Level3ObservationMemory,
    config: Level3PPOConfig,
) -> ActorObservation:
    actor = build_actor_observation(
        belief,
        memory,
        data.steps,
        config.max_episode_steps,
        data.pos_limit_low,
        data.pos_limit_high,
        config.geometry,
    )
    return mask_actor_observation(actor, config.actor_input_mask)


def initialize_level3_runner(
    env: VecDroneRaceEnv,
    config: Level3PPOConfig,
    seed: int | None = None,
    *,
    mixed_segment: bool | None = None,
) -> Level3RunnerState:
    """Reset Level 3 at curriculum progress zero and build both observation streams."""
    seed = config.seed if seed is None else seed
    mixed_segment = config.segment_reset.enabled if mixed_segment is None else mixed_segment
    if (
        config.segment_reset.enabled
        and env.settings.freq != config.segment_reset.environment_frequency
    ):
        raise ValueError(
            "level3_v2_segment elapsed-time initialization requires the configured 50 Hz "
            "environment frequency"
        )
    initial_data = env.data.replace(
        curriculum_progress=jnp.asarray(0.0 if config.curriculum else 1.0, dtype=jnp.float32)
    )
    data, _ = env._reset(initial_data, seed=seed)
    original_reset_belief = belief_observation(data)
    data, episode_start_progress, episode_start_kind = apply_level3_segment_reset(
        data, config, mixed_segment=mixed_segment
    )
    transformed_start = episode_start_kind != FULL_START
    belief = belief_observation(data)
    actual = true_observation(data)
    # Reset observations are the only nominal-map source guaranteed to exist in deployment. If an
    # object already starts inside sensor range, its exact pose is a valid anchor and its visited
    # flag still makes that fact explicit.
    memory = initialize_level3_memory(
        belief,
        history_length=config.history_length,
        nominal_gates_pos=jnp.where(
            transformed_start[:, None, None],
            data.nominal_gates_pos,
            original_reset_belief["gates_pos"],
        ),
        nominal_gates_quat=jnp.where(
            transformed_start[:, None, None],
            data.nominal_gates_quat,
            original_reset_belief["gates_quat"],
        ),
        nominal_obstacles_pos=jnp.where(
            transformed_start[:, None, None],
            data.nominal_obstacles_pos,
            original_reset_belief["obstacles_pos"],
        ),
    )
    actor = _actor_features(data, belief, memory, config)
    critic = _critic_features(env, data, belief, actor, memory, config)
    return Level3RunnerState(
        environment_data=data,
        belief_observation=belief,
        true_observation=actual,
        observation_memory=memory,
        reward_state=initialize_level3_reward_state(actual, config.geometry, config.reward),
        actor_observation=actor,
        critic_observation=critic,
        episode_start_progress=episode_start_progress,
        episode_start_kind=episode_start_kind,
        rng=jax.random.key(seed),
        level_replay=initialize_level_replay(data, config.level_replay, seed),
    )


def _build_level3_environment_step(
    env: VecDroneRaceEnv,
    config: Level3PPOConfig,
    *,
    mixed_segment: bool,
    level_replay: bool = False,
) -> Callable[[Level3RunnerState, Array], Level3EnvironmentTransition]:
    """Build the pure transition used inside rollout and evaluation scans."""
    apply_action = env.build_apply_action_fn()
    classify_contacts = env.build_contact_classification_fn()
    step_simulation = env.sim.build_step_fn()
    reset_environment = env._reset
    simulation_steps = env.sim.freq // env.settings.freq
    action_low = jnp.asarray(env.single_action_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(env.single_action_space.high, dtype=jnp.float32)

    def environment_step(
        data: EnvData, normalized_action: Array
    ) -> tuple[EnvData, dict[str, Array], dict[str, Array], Array, Array, Array, Array]:
        physical_action = scale_action(normalized_action, action_low, action_high)
        data = apply_action(physical_action, data)
        data = data.replace(sim_data=step_simulation(data.sim_data, simulation_steps))
        any_contact, gate_contact, obstacle_contact = classify_contacts(data)
        data = _update_disabled_drones(data, any_contact)
        data = _update_visited_objects(data)
        data = _update_target_gates(data)
        data = _mark_drones_for_reset(data)
        data = data.replace(steps=data.steps + 1)
        belief = belief_observation(data)
        actual = true_observation(data)
        return (
            data,
            belief,
            actual,
            terminated(data)[:, 0],
            truncated(data, config.max_episode_steps)[:, 0],
            gate_contact[:, 0],
            obstacle_contact[:, 0],
        )

    def reset_done(
        data: EnvData,
        done: Array,
        episode_start_progress: Array,
        episode_start_kind: Array,
        memory: Level3ObservationMemory,
        replay_state: Any,
    ) -> tuple[EnvData, Array, Array, Array, Array, Array, Any]:
        def perform_reset(value: tuple) -> tuple[EnvData, Array, Array, Array, Array, Array, Any]:
            current_data, reset_mask, current_start, current_kind, current_memory, replay = value
            reset_data, _ = reset_environment(current_data, seed=None, mask=reset_mask)
            if level_replay:
                reset_data, replay = sample_level_replay(
                    replay, reset_data, reset_mask, config.level_replay
                )
            original_reset_belief = belief_observation(reset_data)
            reset_data, reset_start, reset_kind = apply_level3_segment_reset(
                reset_data, config, reset_mask, mixed_segment=mixed_segment
            )
            segment_mask = (reset_kind != FULL_START)[:, None, None]
            reset_nominal_gates_pos = jnp.where(
                segment_mask, reset_data.nominal_gates_pos, original_reset_belief["gates_pos"]
            )
            reset_nominal_gates_quat = jnp.where(
                segment_mask, reset_data.nominal_gates_quat, original_reset_belief["gates_quat"]
            )
            reset_nominal_obstacles_pos = jnp.where(
                segment_mask,
                reset_data.nominal_obstacles_pos,
                original_reset_belief["obstacles_pos"],
            )
            return (
                reset_data,
                jnp.where(reset_mask, reset_start, current_start),
                jnp.where(reset_mask, reset_kind, current_kind),
                jnp.where(
                    reset_mask[:, None, None],
                    reset_nominal_gates_pos,
                    current_memory.nominal_gates_pos,
                ),
                jnp.where(
                    reset_mask[:, None, None],
                    reset_nominal_gates_quat,
                    current_memory.nominal_gates_quat,
                ),
                jnp.where(
                    reset_mask[:, None, None],
                    reset_nominal_obstacles_pos,
                    current_memory.nominal_obstacles_pos,
                ),
                replay,
            )

        return jax.lax.cond(
            done.any(),
            perform_reset,
            lambda value: (
                value[0],
                value[2],
                value[3],
                value[4].nominal_gates_pos,
                value[4].nominal_gates_quat,
                value[4].nominal_obstacles_pos,
                value[5],
            ),
            (data, done, episode_start_progress, episode_start_kind, memory, replay_state),
        )

    def transition(carry: Level3RunnerState, action: Array) -> Level3EnvironmentTransition:
        (
            terminal_data,
            terminal_belief,
            terminal_actual,
            base_terminated,
            base_truncated,
            gate_contact,
            obstacle_contact,
        ) = environment_step(carry.environment_data, action)
        sequence_length = terminal_belief["gate_sequence"].shape[-1]
        success = terminal_belief["n_gates_passed"] >= sequence_length
        done = base_terminated | base_truncated | success
        replay_state = carry.level_replay
        if level_replay:
            replay_state = update_level_replay(
                replay_state,
                terminal_data,
                done,
                terminal_belief["n_gates_passed"],
                sequence_length,
                config.level_replay,
            )
        (
            next_data,
            next_episode_start_progress,
            next_episode_start_kind,
            reset_nominal_gates_pos,
            reset_nominal_gates_quat,
            reset_nominal_obstacles_pos,
            next_replay_state,
        ) = reset_done(
            terminal_data,
            done,
            carry.episode_start_progress,
            carry.episode_start_kind,
            carry.observation_memory,
            replay_state,
        )
        next_belief = belief_observation(next_data)
        next_actual = true_observation(next_data)
        next_memory, terminal_memory = advance_level3_memory(
            carry.observation_memory,
            terminal_belief,
            next_belief,
            action,
            done,
            reset_nominal_gates_pos=reset_nominal_gates_pos,
            reset_nominal_gates_quat=reset_nominal_gates_quat,
            reset_nominal_obstacles_pos=reset_nominal_obstacles_pos,
        )
        terminal_actor = _actor_features(terminal_data, terminal_belief, terminal_memory, config)
        terminal_critic = _critic_features(
            env, terminal_data, terminal_belief, terminal_actor, terminal_memory, config
        )
        next_reward_state, reward_transition = level3_reward(
            carry.reward_state,
            terminal_actual,
            next_actual,
            terminal_memory,
            action,
            base_terminated,
            base_truncated,
            terminal_data.pos_limit_low,
            terminal_data.pos_limit_high,
            config.reward,
            config.geometry,
        )
        next_actor = _actor_features(next_data, next_belief, next_memory, config)
        next_critic = _critic_features(env, next_data, next_belief, next_actor, next_memory, config)
        gate_geometry = current_gate_geometry(terminal_actual, config.geometry)
        gate_index = jnp.clip(
            terminal_actual["n_gates_passed"], 0, terminal_actual["gate_sequence"].shape[1] - 1
        )
        physical_gate = jnp.take_along_axis(
            terminal_actual["gate_sequence"], gate_index[:, None], axis=1
        ).squeeze(1)
        world = jnp.arange(physical_gate.shape[0])
        gate_correction_distance = jnp.linalg.norm(
            terminal_data.gates_pos[world, physical_gate]
            - terminal_memory.nominal_gates_pos[world, physical_gate],
            axis=-1,
        )
        unsafe_near_plane_stall = (
            (jnp.abs(gate_geometry.position[:, 0]) <= 0.20)
            & (
                (jnp.abs(gate_geometry.position[:, 1]) > config.geometry.gate_safe_half_width)
                | (jnp.abs(gate_geometry.position[:, 2]) > config.geometry.gate_safe_half_height)
            )
            & (jnp.linalg.norm(terminal_actual["vel"], axis=-1) <= 0.20)
        )
        return Level3EnvironmentTransition(
            next_runner=Level3RunnerState(
                environment_data=next_data,
                belief_observation=next_belief,
                true_observation=next_actual,
                observation_memory=next_memory,
                reward_state=next_reward_state,
                actor_observation=next_actor,
                critic_observation=next_critic,
                episode_start_progress=next_episode_start_progress,
                episode_start_kind=next_episode_start_kind,
                rng=carry.rng,
                level_replay=next_replay_state,
            ),
            terminal_actor_observation=terminal_actor,
            terminal_critic_observation=terminal_critic,
            reward=reward_transition.reward,
            terminated=reward_transition.terminated,
            truncated=reward_transition.truncated,
            metrics=Level3RolloutMetrics(
                done=reward_transition.done,
                success=reward_transition.success,
                collision=reward_transition.collision,
                timeout=reward_transition.timeout,
                gate_collision=reward_transition.collision & gate_contact,
                obstacle_collision=reward_transition.collision & obstacle_contact,
                ground_collision=(
                    reward_transition.collision
                    & (terminal_actual["pos"][:, 2] < terminal_data.pos_limit_low[2])
                ),
                boundary_collision=(
                    reward_transition.collision
                    & ~gate_contact
                    & ~obstacle_contact
                    & ~(terminal_actual["pos"][:, 2] < terminal_data.pos_limit_low[2])
                ),
                failed_gate=jnp.where(
                    reward_transition.done,
                    terminal_actual["n_gates_passed"],
                    -jnp.ones_like(terminal_actual["n_gates_passed"]),
                ),
                episode_start_progress=jnp.where(
                    reward_transition.done, carry.episode_start_progress, -1
                ),
                episode_start_kind=jnp.where(reward_transition.done, carry.episode_start_kind, -1),
                final_progress=jnp.where(
                    reward_transition.done,
                    terminal_actual["n_gates_passed"],
                    -jnp.ones_like(terminal_actual["n_gates_passed"]),
                ),
                target_gate_passed=(
                    reward_transition.done
                    & (terminal_actual["n_gates_passed"] > carry.episode_start_progress)
                ),
                difficulty=jnp.where(reward_transition.done, terminal_data.difficulty, 0.0),
                gate_position=gate_geometry.position,
                gate_velocity=gate_geometry.velocity,
                gate_clearance=gate_geometry.clearance,
                current_gate_visited=gate_geometry.visited,
                gate_correction_distance=gate_correction_distance,
                unsafe_near_plane_stall=unsafe_near_plane_stall,
                gate_prediction_safe=(
                    gate_geometry.prediction_valid & (gate_geometry.clearance > 0.0)
                ),
                route_potential=reward_route_potential(
                    terminal_actual, config.geometry, config.reward
                ),
                speed=jnp.linalg.norm(terminal_actual["vel"], axis=-1),
                episode_components=reward_transition.completed_components,
                episode_length=reward_transition.completed_length,
                replayed=reward_transition.done & (carry.level_replay.episode_slot >= 0),
            ),
        )

    return transition


def make_level3_functional_step(
    env: VecDroneRaceEnv, config: Level3PPOConfig
) -> Callable[[Level3RunnerState, Array], Level3EnvironmentTransition]:
    """Compile a policy-free Level 3 transition for mathematical integration tests."""
    return jax.jit(
        _build_level3_environment_step(
            env,
            config,
            mixed_segment=config.segment_reset.enabled,
            level_replay=config.level_replay.enabled,
        )
    )


def make_level3_rollout_collector(
    env: VecDroneRaceEnv,
    model: Level3ActorCritic,
    config: Level3PPOConfig,
    *,
    deterministic: bool = False,
) -> Callable[[Level3RunnerState, Any, Array], tuple[Level3RunnerState, Any, Any]]:
    """Compile policy, simulation, observation, reward, reset, and bootstrap in one scan."""
    environment_transition = _build_level3_environment_step(
        env,
        config,
        mixed_segment=config.segment_reset.enabled,
        level_replay=config.level_replay.enabled,
    )

    def collect(
        runner: Level3RunnerState, params: Any, curriculum_progress: Array
    ) -> tuple[Level3RunnerState, Level3RolloutBatch, Level3RolloutMetrics]:
        runner = runner.replace(
            environment_data=runner.environment_data.replace(
                curriculum_progress=jnp.clip(curriculum_progress, 0.0, 1.0)
            )
        )

        def rollout_step(
            carry: Level3RunnerState, _: None
        ) -> tuple[Level3RunnerState, Level3RolloutOutput]:
            rng, action_key = jax.random.split(carry.rng)
            mean, log_std, value = model.apply(
                {"params": params}, carry.actor_observation, carry.critic_observation
            )
            if deterministic:
                raw_action = mean
                action = deterministic_action(mean)
                log_probability = squashed_log_probability(raw_action, mean, log_std)
            else:
                action, raw_action, log_probability = sample_action(mean, log_std, action_key)
            transition = environment_transition(carry, action)
            bootstrap_value = model.apply(
                {"params": params},
                transition.terminal_actor_observation,
                transition.terminal_critic_observation,
                method=model.critic,
            )
            return transition.next_runner.replace(rng=rng), Level3RolloutOutput(
                batch=Level3RolloutBatch(
                    actor_observation=carry.actor_observation,
                    critic_observation=carry.critic_observation,
                    raw_action=raw_action,
                    log_probability=log_probability,
                    value=value,
                    reward=transition.reward,
                    bootstrap_value=bootstrap_value,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                ),
                metrics=transition.metrics,
            )

        runner, output = jax.lax.scan(rollout_step, runner, None, length=config.num_steps)
        return runner, output.batch, output.metrics

    return jax.jit(collect)


def make_level3_evaluation_collector(
    env: VecDroneRaceEnv, model: Level3ActorCritic, config: Level3PPOConfig
) -> Callable[[Level3RunnerState, Any], tuple[Level3RunnerState, Level3RolloutMetrics]]:
    """Compile deterministic Actor-only evaluation without materializing PPO samples."""
    # Evaluation is always an original full-start race, even when loading a segment profile.
    environment_transition = _build_level3_environment_step(env, config, mixed_segment=False)

    def collect(
        runner: Level3RunnerState, params: Any
    ) -> tuple[Level3RunnerState, Level3RolloutMetrics]:
        runner = runner.replace(
            environment_data=runner.environment_data.replace(
                curriculum_progress=jnp.asarray(1.0, dtype=jnp.float32)
            )
        )

        def rollout_step(
            carry: Level3RunnerState, _: None
        ) -> tuple[Level3RunnerState, Level3RolloutMetrics]:
            mean, _ = model.apply({"params": params}, carry.actor_observation, method=model.actor)
            transition = environment_transition(carry, deterministic_action(mean))
            return transition.next_runner, transition.metrics

        return jax.lax.scan(rollout_step, runner, None, length=config.num_steps)

    return jax.jit(collect)


def summarize_level3_rollout(metrics: Level3RolloutMetrics) -> dict[str, Array]:
    """Aggregate only stable scalar metrics before any device-to-host transfer."""
    episode_count = jnp.sum(metrics.done)
    denominator = jnp.maximum(episode_count, 1)
    component_sums = jnp.sum(metrics.episode_components, axis=(0, 1))
    summary = {
        "episodes": episode_count,
        "success_rate": jnp.sum(metrics.success) / denominator,
        "collision_rate": jnp.sum(metrics.collision) / denominator,
        "timeout_rate": jnp.sum(metrics.timeout) / denominator,
        "gate_collision_rate": jnp.sum(metrics.gate_collision) / denominator,
        "obstacle_collision_rate": jnp.sum(metrics.obstacle_collision) / denominator,
        "ground_collision_rate": jnp.sum(metrics.ground_collision) / denominator,
        "boundary_collision_rate": jnp.sum(metrics.boundary_collision) / denominator,
        "episode_length": jnp.sum(metrics.episode_length) / denominator,
        "episode_reward": jnp.sum(component_sums) / denominator,
        "completed_difficulty": jnp.sum(metrics.difficulty) / denominator,
        "current_gate_visited_step_rate": jnp.mean(metrics.current_gate_visited),
        "unsafe_near_plane_stall_step_rate": jnp.mean(metrics.unsafe_near_plane_stall),
        "level_replay/replayed_fraction": jnp.sum(metrics.replayed) / denominator,
        "level_replay/fresh_success_rate": jnp.sum(metrics.success & ~metrics.replayed)
        / jnp.maximum(jnp.sum(metrics.done & ~metrics.replayed), 1),
    }
    summary.update(
        {
            f"reward/{name}": value / denominator
            for name, value in zip(LEVEL3_REWARD_COMPONENTS, component_sums, strict=True)
        }
    )
    for start in range(4):
        start_mask = metrics.done & (metrics.episode_start_progress == start)
        start_count = jnp.sum(start_mask)
        start_denominator = jnp.maximum(start_count, 1)
        summary.update(
            {
                f"segment/start_{start}_episodes": start_count,
                f"segment/start_{start}_target_gate_pass_rate": (
                    jnp.sum(metrics.target_gate_passed & start_mask) / start_denominator
                ),
                f"segment/start_{start}_remaining_track_success_rate": (
                    jnp.sum(metrics.success & start_mask) / start_denominator
                ),
                f"segment/start_{start}_collision_rate": (
                    jnp.sum(metrics.collision & start_mask) / start_denominator
                ),
                f"segment/start_{start}_timeout_rate": (
                    jnp.sum(metrics.timeout & start_mask) / start_denominator
                ),
                f"segment/start_{start}_final_progress": (
                    jnp.sum(jnp.where(start_mask, metrics.final_progress, 0)) / start_denominator
                ),
            }
        )
    for kind, name in (
        (FULL_START, "full"),
        (POST_GATE_START, "post_gate"),
        (PRE_GATE_START, "pre_gate"),
        (CORRIDOR_START, "corridor"),
    ):
        kind_mask = metrics.done & (metrics.episode_start_kind == kind)
        kind_count = jnp.sum(kind_mask)
        kind_denominator = jnp.maximum(kind_count, 1)
        summary.update(
            {
                f"start_kind/{name}_episodes": kind_count,
                f"start_kind/{name}_target_gate_pass_rate": (
                    jnp.sum(metrics.target_gate_passed & kind_mask) / kind_denominator
                ),
                f"start_kind/{name}_remaining_track_success_rate": (
                    jnp.sum(metrics.success & kind_mask) / kind_denominator
                ),
                f"start_kind/{name}_collision_rate": (
                    jnp.sum(metrics.collision & kind_mask) / kind_denominator
                ),
                f"start_kind/{name}_timeout_rate": (
                    jnp.sum(metrics.timeout & kind_mask) / kind_denominator
                ),
                f"start_kind/{name}_final_progress": (
                    jnp.sum(jnp.where(kind_mask, metrics.final_progress, 0)) / kind_denominator
                ),
            }
        )
    return summary
