"""Device-side racing reward and episode accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import struct

from lsy_drone_racing.rl_training.observations import (
    quaternion_to_matrix,
    target_gate_directions,
    target_gate_indices,
)

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.rl_training.config import RewardCoefficients

REWARD_COMPONENTS = (
    "progress",
    "gate",
    "finish",
    "align",
    "time",
    "ground",
    "collision",
    "smooth",
    "spin",
    "angle",
)


@struct.dataclass
class RewardState:
    """Per-environment potential and episode accumulators."""

    last_distance: Array
    last_action: Array
    last_progress: Array
    episode_components: Array
    episode_length: Array


@struct.dataclass
class RewardTransition:
    """Reward and terminal information emitted by a single environment step."""

    reward: Array
    terminated: Array
    truncated: Array
    done: Array
    success: Array
    collision: Array
    completed_components: Array
    completed_length: Array


def _distance_to_target(obs: dict[str, Array]) -> Array:
    gate_id = target_gate_indices(obs)
    target = obs["gates_pos"][jnp.arange(gate_id.shape[0]), gate_id]
    return jnp.linalg.norm(obs["pos"] - target, axis=-1)


def initialize_reward_state(obs: dict[str, Array]) -> RewardState:
    """Initialize potential-based reward state after an environment reset."""
    num_envs = obs["pos"].shape[0]
    return RewardState(
        last_distance=_distance_to_target(obs),
        last_action=jnp.zeros((num_envs, 4), dtype=jnp.float32),
        last_progress=obs["n_gates_passed"],
        episode_components=jnp.zeros((num_envs, len(REWARD_COMPONENTS)), dtype=jnp.float32),
        episode_length=jnp.zeros(num_envs, dtype=jnp.int32),
    )


def _alignment_reward(obs: dict[str, Array], finished: Array) -> Array:
    gate_id = target_gate_indices(obs)
    quaternion = obs["gates_quat"][jnp.arange(gate_id.shape[0]), gate_id]
    normal = quaternion_to_matrix(quaternion)[..., :, 0]
    direction = target_gate_directions(obs).astype(jnp.float32)
    alignment = jnp.sum(obs["vel"] * normal * direction[:, None], axis=-1) * 0.5
    return jnp.where(finished, 0.0, alignment)


def _roll_pitch_norm(quaternion: Array) -> Array:
    x, y, z, w = jnp.moveaxis(quaternion, -1, 0)
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = jnp.arcsin(jnp.clip(2 * (w * y - z * x), -1.0, 1.0))
    return jnp.sqrt(roll * roll + pitch * pitch)


def racing_reward(
    state: RewardState,
    terminal_obs: dict[str, Array],
    reset_obs: dict[str, Array],
    action: Array,
    environment_terminated: Array,
    environment_truncated: Array,
    frequency: int,
    coefficients: RewardCoefficients,
) -> tuple[RewardState, RewardTransition]:
    """Compute shaped reward and reset-aware state for a vectorized transition."""
    progress = terminal_obs["n_gates_passed"]
    sequence_length = terminal_obs["gate_sequence"].shape[-1]
    success = progress >= sequence_length
    collision = environment_terminated & ~success
    terminated = collision | success
    truncated = environment_truncated & ~terminated
    done = terminated | truncated

    distance = _distance_to_target(terminal_obs)
    gate_changed = progress != state.last_progress
    progress_delta = jnp.clip(
        state.last_distance - distance,
        -coefficients.max_progress_speed / frequency,
        coefficients.max_progress_speed / frequency,
    )
    progress_reward = jnp.where(gate_changed | success, 0.0, progress_delta) * coefficients.progress
    passed = progress > state.last_progress
    gate_reward = jnp.where(
        passed, (state.last_progress.astype(jnp.float32) + 1.0) * coefficients.gate, 0.0
    )
    finish_reward = jnp.where(success, coefficients.finish, 0.0)
    align_reward = _alignment_reward(terminal_obs, success) * coefficients.align
    time_penalty = jnp.where(success, 0.0, coefficients.time)
    ground_penalty = jnp.where(
        (terminal_obs["pos"][:, 2] < 0.1) & ~success, coefficients.ground, 0.0
    )
    collision_penalty = jnp.where(collision, coefficients.collision, 0.0)
    smooth_penalty = coefficients.smooth * jnp.sum((action - state.last_action) ** 2, axis=-1)
    spin_penalty = coefficients.spin * jnp.sum(terminal_obs["ang_vel"] ** 2, axis=-1)
    angle_penalty = coefficients.angle * _roll_pitch_norm(terminal_obs["quat"])
    components = jnp.stack(
        (
            progress_reward,
            gate_reward,
            finish_reward,
            align_reward,
            -time_penalty,
            -ground_penalty,
            -collision_penalty,
            -smooth_penalty,
            -spin_penalty,
            -angle_penalty,
        ),
        axis=-1,
    ).astype(jnp.float32)
    reward = jnp.sum(components, axis=-1)
    episode_components = state.episode_components + components
    episode_length = state.episode_length + 1
    completed_components = jnp.where(done[:, None], episode_components, 0.0)
    completed_length = jnp.where(done, episode_length, 0)

    reset_distance = _distance_to_target(reset_obs)
    next_state = RewardState(
        last_distance=jnp.where(done, reset_distance, distance),
        last_action=jnp.where(done[:, None], 0.0, action),
        last_progress=jnp.where(done, reset_obs["n_gates_passed"], progress),
        episode_components=jnp.where(done[:, None], 0.0, episode_components),
        episode_length=jnp.where(done, 0, episode_length),
    )
    transition = RewardTransition(
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        done=done,
        success=success & done,
        collision=collision & done,
        completed_components=completed_components,
        completed_length=completed_length,
    )
    return next_state, transition
