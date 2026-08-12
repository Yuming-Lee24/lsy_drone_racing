"""Functional, fused adapter between RaceCoreEnv and the JAX PPO agent."""

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
    obs,
    terminated,
    truncated,
)
from lsy_drone_racing.rl_training.algorithm import RolloutBatch
from lsy_drone_racing.rl_training.networks import (
    deterministic_action,
    sample_action,
    squashed_log_probability,
)
from lsy_drone_racing.rl_training.observations import (
    ObservationState,
    advance_observation_state,
    initialize_observation_state,
    racing_observation,
    squeeze_single_drone_observation,
)
from lsy_drone_racing.rl_training.rewards import (
    REWARD_COMPONENTS,
    RewardState,
    initialize_reward_state,
    racing_reward,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
    from lsy_drone_racing.envs.race_core import EnvData
    from lsy_drone_racing.rl_training.config import PPOConfig
    from lsy_drone_racing.rl_training.networks import ActorCritic


@struct.dataclass
class RunnerState:
    """All dynamic data carried through a compiled rollout."""

    environment_data: Any
    raw_observation: dict[str, Array]
    observation_state: ObservationState
    reward_state: RewardState
    observation: Array
    rng: Array


@struct.dataclass
class RolloutMetrics:
    """Per-step episode metrics emitted alongside a rollout."""

    done: Array
    success: Array
    collision: Array
    episode_components: Array
    episode_length: Array


@struct.dataclass
class RolloutOutput:
    """PPO samples plus monitoring values from a rollout."""

    batch: RolloutBatch
    metrics: RolloutMetrics


@struct.dataclass
class EnvironmentTransition:
    """One policy-free transition including the terminal value observation."""

    next_runner: RunnerState
    terminal_observation: Array
    reward: Array
    terminated: Array
    truncated: Array
    metrics: RolloutMetrics


def scale_action(action: Array, low: Array, high: Array) -> Array:
    """Map normalized actions to the simulator action space on device."""
    scale = (high - low) / 2.0
    mean = (high + low) / 2.0
    return jnp.clip(action, -1.0, 1.0) * scale + mean


def hover_action_bias(low: Array, high: Array, hover_thrust: float) -> tuple[float, ...]:
    """Return pre-tanh actor biases corresponding to level attitude and hover thrust."""
    normalized = (2.0 * jnp.asarray(hover_thrust) - high[-1] - low[-1]) / (high[-1] - low[-1])
    thrust_bias = jnp.arctanh(jnp.clip(normalized, -0.95, 0.95))
    return (0.0, 0.0, 0.0, float(thrust_bias))


def initialize_runner(
    env: VecDroneRaceEnv, config: PPOConfig, seed: int | None = None
) -> RunnerState:
    """Reset a vector environment and build device-side observation/reward state."""
    seed = config.seed if seed is None else seed
    environment_data, (raw_observation, _) = env._reset(env.data, seed=seed)
    raw_observation = squeeze_single_drone_observation(raw_observation)
    observation_state = initialize_observation_state(raw_observation, config.history_length)
    reward_state = initialize_reward_state(raw_observation)
    observation = racing_observation(raw_observation, observation_state, config.curriculum_stage)
    return RunnerState(
        environment_data=environment_data,
        raw_observation=raw_observation,
        observation_state=observation_state,
        reward_state=reward_state,
        observation=observation,
        rng=jax.random.key(seed),
    )


def _build_environment_step(
    env: VecDroneRaceEnv, config: PPOConfig
) -> Callable[[RunnerState, Array], EnvironmentTransition]:
    """Build one pure transition shared by JAX and the local PyTorch reference."""
    apply_action = env.build_apply_action_fn()
    check_contacts = env.build_contact_check_fn()
    step_simulation = env.sim.build_step_fn()
    reset_environment = env._reset
    simulation_steps = env.sim.freq // env.settings.freq
    action_low = jnp.asarray(env.single_action_space.low, dtype=jnp.float32)
    action_high = jnp.asarray(env.single_action_space.high, dtype=jnp.float32)

    def environment_step(
        data: EnvData, normalized_action: Array
    ) -> tuple[EnvData, dict[str, Array], Array, Array]:
        physical_action = scale_action(normalized_action, action_low, action_high)
        data = apply_action(physical_action, data)
        simulation_data = step_simulation(data.sim_data, simulation_steps)
        data = data.replace(sim_data=simulation_data)
        data = _update_disabled_drones(data, check_contacts(data))
        data = _update_visited_objects(data)
        data = _update_target_gates(data)
        data = _mark_drones_for_reset(data)
        data = data.replace(steps=data.steps + 1)
        raw_observation = squeeze_single_drone_observation(obs(data))
        environment_terminated = terminated(data)[:, 0]
        environment_truncated = truncated(data, config.max_episode_steps)[:, 0]
        return data, raw_observation, environment_terminated, environment_truncated

    def reset_done(data: EnvData, done: Array) -> tuple[EnvData, dict[str, Array]]:
        def perform_reset(value: tuple[EnvData, Array]) -> tuple[EnvData, dict[str, Array]]:
            current_data, reset_mask = value
            reset_data, (reset_observation, _) = reset_environment(
                current_data, seed=None, mask=reset_mask
            )
            return reset_data, squeeze_single_drone_observation(reset_observation)

        def skip_reset(value: tuple[EnvData, Array]) -> tuple[EnvData, dict[str, Array]]:
            current_data, _ = value
            return current_data, squeeze_single_drone_observation(obs(current_data))

        return jax.lax.cond(done.any(), perform_reset, skip_reset, (data, done))

    def transition(carry: RunnerState, action: Array) -> EnvironmentTransition:
        terminal_data, terminal_observation, base_terminated, base_truncated = environment_step(
            carry.environment_data, action
        )
        sequence_length = terminal_observation["gate_sequence"].shape[-1]
        success = terminal_observation["n_gates_passed"] >= sequence_length
        done = base_terminated | base_truncated | success
        next_data, next_raw_observation = reset_done(terminal_data, done)
        next_observation_state, active_observation_state = advance_observation_state(
            carry.observation_state, carry.raw_observation, next_raw_observation, action, done
        )
        terminal_vector = racing_observation(
            terminal_observation, active_observation_state, config.curriculum_stage
        )
        next_reward_state, reward_transition = racing_reward(
            carry.reward_state,
            terminal_observation,
            next_raw_observation,
            action,
            base_terminated,
            base_truncated,
            env.settings.freq,
            config.reward,
        )
        next_observation = racing_observation(
            next_raw_observation, next_observation_state, config.curriculum_stage
        )
        next_runner = RunnerState(
            environment_data=next_data,
            raw_observation=next_raw_observation,
            observation_state=next_observation_state,
            reward_state=next_reward_state,
            observation=next_observation,
            rng=carry.rng,
        )
        return EnvironmentTransition(
            next_runner=next_runner,
            terminal_observation=terminal_vector,
            reward=reward_transition.reward,
            terminated=reward_transition.terminated,
            truncated=reward_transition.truncated,
            metrics=RolloutMetrics(
                done=reward_transition.done,
                success=reward_transition.success,
                collision=reward_transition.collision,
                episode_components=reward_transition.completed_components,
                episode_length=reward_transition.completed_length,
            ),
        )

    return transition


def make_functional_step(
    env: VecDroneRaceEnv, config: PPOConfig
) -> Callable[[RunnerState, Array], EnvironmentTransition]:
    """Compile a policy-free transition for the local PyTorch A/B reference."""
    return jax.jit(_build_environment_step(env, config))


def make_rollout_collector(
    env: VecDroneRaceEnv, model: ActorCritic, config: PPOConfig, *, deterministic: bool = False
) -> Callable[[RunnerState, Any], tuple[RunnerState, RolloutBatch, RolloutMetrics]]:
    """Build a compiled rollout containing policy, simulation, reward, and reset logic."""
    environment_transition = _build_environment_step(env, config)

    def collect(
        runner: RunnerState, params: Any
    ) -> tuple[RunnerState, RolloutBatch, RolloutMetrics]:
        def rollout_step(carry: RunnerState, _: None) -> tuple[RunnerState, RolloutOutput]:
            rng, action_key = jax.random.split(carry.rng)
            mean, log_std, value = model.apply({"params": params}, carry.observation)
            if deterministic:
                raw_action = mean
                action = deterministic_action(mean)
                log_probability = squashed_log_probability(raw_action, mean, log_std)
            else:
                action, raw_action, log_probability = sample_action(mean, log_std, action_key)
            transition = environment_transition(carry, action)
            _, _, bootstrap_value = model.apply({"params": params}, transition.terminal_observation)
            next_runner = transition.next_runner.replace(rng=rng)
            output = RolloutOutput(
                batch=RolloutBatch(
                    observation=carry.observation,
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
            return next_runner, output

        runner, output = jax.lax.scan(rollout_step, runner, None, length=config.num_steps)
        return runner, output.batch, output.metrics

    return jax.jit(collect)


def make_evaluation_collector(
    env: VecDroneRaceEnv, model: ActorCritic, config: PPOConfig
) -> Callable[[RunnerState, Any], tuple[RunnerState, RolloutMetrics]]:
    """Compile a deterministic rollout that emits only terminal evaluation metrics."""
    environment_transition = _build_environment_step(env, config)

    def collect(runner: RunnerState, params: Any) -> tuple[RunnerState, RolloutMetrics]:
        def rollout_step(carry: RunnerState, _: None) -> tuple[RunnerState, RolloutMetrics]:
            mean, _, _ = model.apply({"params": params}, carry.observation)
            transition = environment_transition(carry, deterministic_action(mean))
            return transition.next_runner, transition.metrics

        return jax.lax.scan(rollout_step, runner, None, length=config.num_steps)

    return jax.jit(collect)


def summarize_rollout(metrics: RolloutMetrics) -> dict[str, Array]:
    """Aggregate episode metrics without transferring per-world arrays to the host."""
    episode_count = jnp.sum(metrics.done)
    denominator = jnp.maximum(episode_count, 1)
    component_sums = jnp.sum(metrics.episode_components, axis=(0, 1))
    summary = {
        "episodes": episode_count,
        "success_rate": jnp.sum(metrics.success) / denominator,
        "collision_rate": jnp.sum(metrics.collision) / denominator,
        "episode_length": jnp.sum(metrics.episode_length) / denominator,
        "episode_reward": jnp.sum(component_sums) / denominator,
    }
    summary.update(
        {
            f"reward/{name}": value / denominator
            for name, value in zip(REWARD_COMPONENTS, component_sums, strict=True)
        }
    )
    return summary
