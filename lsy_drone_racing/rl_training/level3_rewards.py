"""Reliability-first Level 3 reward and episode accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import struct

from lsy_drone_racing.rl_training.level3_observations import (
    Level3ObservationMemory,
    current_gate_geometry,
    obstacle_geometry,
    route_potential,
)

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.rl_training.level3_config import Level3RewardCoefficients, SafetyGeometry

LEVEL3_REWARD_COMPONENTS = (
    "route_progress",
    "gate",
    "finish",
    "safe_gate_speed",
    "gate_risk",
    "obstacle_risk",
    "boundary_risk",
    "action_delta",
    "collision",
    "timeout",
)


@struct.dataclass
class Level3RewardState:
    """Per-world V2 potential and completed-episode accumulators."""

    last_route_potential: Array
    last_action: Array
    last_progress: Array
    episode_components: Array
    episode_length: Array


@struct.dataclass
class Level3RewardTransition:
    """One reward output including unambiguous terminal classifications."""

    reward: Array
    terminated: Array
    truncated: Array
    done: Array
    success: Array
    collision: Array
    timeout: Array
    completed_components: Array
    completed_length: Array


def initialize_level3_reward_state(
    true_observation: dict[str, Array], geometry: SafetyGeometry
) -> Level3RewardState:
    """Initialize a reward potential from actual reset geometry."""
    num_envs = true_observation["pos"].shape[0]
    return Level3RewardState(
        last_route_potential=route_potential(true_observation, geometry),
        last_action=jnp.zeros((num_envs, 4), dtype=jnp.float32),
        last_progress=true_observation["n_gates_passed"],
        episode_components=jnp.zeros((num_envs, len(LEVEL3_REWARD_COMPONENTS)), dtype=jnp.float32),
        episode_length=jnp.zeros(num_envs, dtype=jnp.int32),
    )


def level3_reward(
    state: Level3RewardState,
    terminal_true_observation: dict[str, Array],
    reset_true_observation: dict[str, Array],
    observation_memory: Level3ObservationMemory,
    action: Array,
    environment_terminated: Array,
    environment_truncated: Array,
    pos_limit_low: Array,
    pos_limit_high: Array,
    coefficients: Level3RewardCoefficients,
    geometry: SafetyGeometry,
) -> tuple[Level3RewardState, Level3RewardTransition]:
    """Compute bounded geometric shaping and reset all completed-world state."""
    progress = terminal_true_observation["n_gates_passed"]
    sequence_length = terminal_true_observation["gate_sequence"].shape[-1]
    success = progress >= sequence_length
    collision = environment_terminated & ~success
    terminated = collision | success
    timeout = environment_truncated & ~terminated
    truncated = timeout
    done = terminated | truncated

    potential = route_potential(terminal_true_observation, geometry)
    gate_changed = progress != state.last_progress
    route_delta = jnp.clip(
        state.last_route_potential - potential,
        -coefficients.max_route_progress_per_step,
        coefficients.max_route_progress_per_step,
    )
    route_component = jnp.where(gate_changed | success, 0.0, route_delta)
    route_component *= coefficients.route_progress
    passed_count = jnp.maximum(progress - state.last_progress, 0).astype(jnp.float32)
    gate_component = passed_count * coefficients.gate
    finish_component = success.astype(jnp.float32) * coefficients.finish

    gate = current_gate_geometry(terminal_true_observation, geometry)
    safe_prediction = gate.prediction_valid & (gate.clearance > 0.0) & ~success
    safe_gate_speed_component = (
        coefficients.safe_gate_speed
        * jnp.clip(gate.velocity[:, 0], 0.0, 2.0)
        * safe_prediction.astype(jnp.float32)
    )
    gate_margin_risk = jnp.clip(
        (geometry.gate_risk_buffer - gate.clearance) / geometry.gate_risk_buffer, 0.0, 1.0
    )
    gate_risk = jnp.where(gate.prediction_valid & ~success, gate_margin_risk**2, 0.0)
    gate_risk_component = -coefficients.gate_risk * gate_risk

    obstacles = obstacle_geometry(terminal_true_observation, observation_memory, geometry)
    obstacle_risk_component = -coefficients.obstacle_risk * jnp.max(obstacles.risk, axis=-1)

    lower_clearance = terminal_true_observation["pos"] - pos_limit_low
    upper_clearance = pos_limit_high - terminal_true_observation["pos"]
    boundary_clearance = jnp.min(
        jnp.concatenate((lower_clearance, upper_clearance), axis=-1), axis=-1
    )
    boundary_risk = jnp.clip(
        (geometry.boundary_risk_buffer - boundary_clearance) / geometry.boundary_risk_buffer,
        0.0,
        1.0,
    )
    boundary_risk_component = -coefficients.boundary_risk * boundary_risk**2
    action_delta_component = -coefficients.action_delta * jnp.sum(
        (action - state.last_action) ** 2, axis=-1
    )
    collision_component = -collision.astype(jnp.float32) * (
        coefficients.collision + coefficients.failure_per_gate * progress.astype(jnp.float32)
    )
    timeout_component = -timeout.astype(jnp.float32) * (
        coefficients.timeout + coefficients.failure_per_gate * progress.astype(jnp.float32)
    )
    components = jnp.stack(
        (
            route_component,
            gate_component,
            finish_component,
            safe_gate_speed_component,
            gate_risk_component,
            obstacle_risk_component,
            boundary_risk_component,
            action_delta_component,
            collision_component,
            timeout_component,
        ),
        axis=-1,
    ).astype(jnp.float32)
    reward = jnp.sum(components, axis=-1)
    episode_components = state.episode_components + components
    episode_length = state.episode_length + 1
    completed_components = jnp.where(done[:, None], episode_components, 0.0)
    completed_length = jnp.where(done, episode_length, 0)

    reset_potential = route_potential(reset_true_observation, geometry)
    next_state = Level3RewardState(
        last_route_potential=jnp.where(done, reset_potential, potential),
        last_action=jnp.where(done[:, None], 0.0, action),
        last_progress=jnp.where(done, reset_true_observation["n_gates_passed"], progress),
        episode_components=jnp.where(done[:, None], 0.0, episode_components),
        episode_length=jnp.where(done, 0, episode_length),
    )
    return next_state, Level3RewardTransition(
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        done=done,
        success=success & done,
        collision=collision & done,
        timeout=timeout & done,
        completed_components=completed_components,
        completed_length=completed_length,
    )
