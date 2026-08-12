"""Pure JAX observation processing shared by training, evaluation, and deployment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct

if TYPE_CHECKING:
    from jax import Array

BASE_OBSERVATION_DIM = 56
HISTORY_STATE_DIM = 16
ACTION_DIM = 4
MAX_OBSTACLES = 4
GATE_CORNERS_LOCAL = jnp.asarray(
    [[0.0, -0.2, 0.2], [0.0, 0.2, 0.2], [0.0, 0.2, -0.2], [0.0, -0.2, -0.2]], dtype=jnp.float32
)


@struct.dataclass
class ObservationState:
    """History required to construct a Markov observation."""

    previous_action: Array
    history: Array


def squeeze_single_drone_observation(obs: dict[str, Array]) -> dict[str, Array]:
    """Remove the singleton drone dimension from a RaceCore observation."""
    return {key: value[:, 0] for key, value in obs.items()}


def quaternion_to_matrix(quaternion: Array) -> Array:
    """Convert scipy-order ``[x, y, z, w]`` quaternions to rotation matrices."""
    quaternion = quaternion / jnp.maximum(jnp.linalg.norm(quaternion, axis=-1, keepdims=True), 1e-8)
    x, y, z, w = jnp.moveaxis(quaternion, -1, 0)
    return jnp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape((*quaternion.shape[:-1], 3, 3))


def target_gate_indices(obs: dict[str, Array], offset: int = 0) -> Array:
    """Return physical gate IDs for the current or a future gate-sequence entry."""
    sequence = obs["gate_sequence"]
    progress = jnp.clip(obs["n_gates_passed"] + offset, 0, sequence.shape[-1] - 1)
    return jnp.take_along_axis(sequence, progress[:, None], axis=1).squeeze(1)


def target_gate_directions(obs: dict[str, Array], offset: int = 0) -> Array:
    """Return the configured passing direction for a current or future sequence entry."""
    sequence_direction = obs["gate_sequence_direction"]
    progress = jnp.clip(obs["n_gates_passed"] + offset, 0, sequence_direction.shape[-1] - 1)
    return jnp.take_along_axis(sequence_direction, progress[:, None], axis=1).squeeze(1)


def _gather(values: Array, indices: Array) -> Array:
    return values[jnp.arange(values.shape[0]), indices]


def _basic_history_state(obs: dict[str, Array]) -> Array:
    rotation = quaternion_to_matrix(obs["quat"])
    return jnp.concatenate(
        (obs["pos"][:, 2:3], rotation.reshape((rotation.shape[0], 9)), obs["vel"], obs["ang_vel"]),
        axis=-1,
    ).astype(jnp.float32)


def initialize_observation_state(obs: dict[str, Array], history_length: int) -> ObservationState:
    """Initialize action and history memory from a reset observation."""
    basic = _basic_history_state(obs)
    history = jnp.repeat(basic[:, None, :], history_length, axis=1)
    return ObservationState(
        previous_action=jnp.zeros((basic.shape[0], ACTION_DIM), dtype=jnp.float32), history=history
    )


def advance_observation_state(
    state: ObservationState,
    current_obs: dict[str, Array],
    next_obs: dict[str, Array],
    action: Array,
    done: Array,
) -> tuple[ObservationState, ObservationState]:
    """Advance history and return both reset-aware and uninterrupted terminal states.

    The uninterrupted state is used to value a time-limit terminal observation. The reset-aware
    state is the memory consumed by the next policy step.
    """
    if state.history.shape[1]:
        current = _basic_history_state(current_obs)
        active_history = jnp.concatenate((state.history[:, 1:], current[:, None]), axis=1)
        reset = _basic_history_state(next_obs)
        reset_history = jnp.repeat(reset[:, None, :], state.history.shape[1], axis=1)
        next_history = jnp.where(done[:, None, None], reset_history, active_history)
    else:
        active_history = state.history
        next_history = state.history
    active_state = ObservationState(previous_action=action, history=active_history)
    next_state = ObservationState(
        previous_action=jnp.where(done[:, None], 0.0, action), history=next_history
    )
    return next_state, active_state


def _gate_corners_body(
    gate_position: Array,
    gate_quaternion: Array,
    direction: Array,
    drone_position: Array,
    drone_rotation: Array,
) -> Array:
    gate_rotation = quaternion_to_matrix(gate_quaternion)
    # Mirroring the ordered local Y coordinates preserves the physical square while encoding which
    # face of a reversible gate is the entrance. This keeps the legacy 88-D layout unchanged.
    directed_corners = (
        GATE_CORNERS_LOCAL[None]
        * jnp.stack((jnp.ones_like(direction), direction, jnp.ones_like(direction)), axis=-1)[
            :, None
        ]
    )
    corners_world = jnp.einsum("nij,nkj->nki", gate_rotation, directed_corners)
    relative_world = corners_world + gate_position[:, None] - drone_position[:, None]
    corners_body = jnp.einsum("nji,nkj->nki", drone_rotation, relative_world)
    return corners_body.reshape((drone_position.shape[0], 12))


def _nearest_obstacles_body(obs: dict[str, Array], drone_rotation: Array) -> Array:
    obstacles = obs["obstacles_pos"]
    num_envs, num_obstacles = obstacles.shape[:2]
    if num_obstacles == 0:
        return jnp.full((num_envs, MAX_OBSTACLES * 3), 10.0, dtype=jnp.float32)
    drone_position = obs["pos"]
    effective_z = jnp.minimum(obstacles[..., 2], drone_position[:, None, 2])
    effective_position = obstacles.at[..., 2].set(effective_z)
    relative_world = effective_position - drone_position[:, None]
    relative_body = jnp.einsum("nji,nkj->nki", drone_rotation, relative_world)
    distances = jnp.linalg.norm(relative_body, axis=-1)
    if num_obstacles < MAX_OBSTACLES:
        padding = MAX_OBSTACLES - num_obstacles
        relative_body = jnp.pad(relative_body, ((0, 0), (0, padding), (0, 0)), constant_values=10.0)
        distances = jnp.pad(distances, ((0, 0), (0, padding)), constant_values=jnp.inf)
    indices = jnp.argsort(distances, axis=1)[:, :MAX_OBSTACLES]
    nearest = jnp.take_along_axis(relative_body, indices[..., None], axis=1)
    return nearest.reshape((num_envs, MAX_OBSTACLES * 3))


def racing_observation(
    obs: dict[str, Array], state: ObservationState, curriculum_stage: int = 2
) -> Array:
    """Construct the fixed 88-dimensional end-to-end racing observation."""
    drone_rotation = quaternion_to_matrix(obs["quat"])
    velocity_body = jnp.einsum("nji,nj->ni", drone_rotation, obs["vel"])
    gate_1 = target_gate_indices(obs)
    gate_2 = target_gate_indices(obs, offset=1)
    gate_1_direction = target_gate_directions(obs)
    gate_2_direction = target_gate_directions(obs, offset=1)
    gate_1_corners = _gate_corners_body(
        _gather(obs["gates_pos"], gate_1),
        _gather(obs["gates_quat"], gate_1),
        gate_1_direction,
        obs["pos"],
        drone_rotation,
    )
    gate_2_corners = _gate_corners_body(
        _gather(obs["gates_pos"], gate_2),
        _gather(obs["gates_quat"], gate_2),
        gate_2_direction,
        obs["pos"],
        drone_rotation,
    )
    obstacles = _nearest_obstacles_body(obs, drone_rotation)
    if curriculum_stage == 0:
        gate_2_corners = gate_1_corners
        obstacles = jnp.full_like(obstacles, 10.0)
    elif curriculum_stage == 1:
        obstacles = jnp.full_like(obstacles, 10.0)
    vector = jnp.concatenate(
        (
            obs["pos"][:, 2:3],
            velocity_body,
            obs["ang_vel"],
            drone_rotation.reshape((drone_rotation.shape[0], 9)),
            gate_1_corners,
            gate_2_corners,
            state.previous_action,
            obstacles,
            state.history.reshape((state.history.shape[0], -1)),
        ),
        axis=-1,
    )
    return vector.astype(jnp.float32)


def observation_dimension(history_length: int) -> int:
    """Return the flat observation size for a history length."""
    return BASE_OBSERVATION_DIM + history_length * HISTORY_STATE_DIM


racing_observation_jit = jax.jit(racing_observation, static_argnames="curriculum_stage")
