"""Structured, deployment-safe Level 3 observations and privileged critic features."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import struct

from lsy_drone_racing.rl_training.observations import quaternion_to_matrix

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.rl_training.level3_config import SafetyGeometry

EGO_DIM = 28
HISTORY_LENGTH = 4
HISTORY_FRAME_DIM = 16
GATE_TOKEN_DIM = 23
OBSTACLE_TOKEN_DIM = 15
DIRECT_GEOMETRY_DIM = 31
PRIVILEGED_DYNAMICS_DIM = 8
ACTION_DIM = 4
MAX_OBSTACLES = 4


@struct.dataclass
class Level3ObservationMemory:
    """Deterministic belief-map anchors and short dynamics history."""

    nominal_gates_pos: Array
    nominal_gates_quat: Array
    nominal_obstacles_pos: Array
    previous_action: Array
    previous_body_velocity: Array
    previous_angular_velocity: Array
    history: Array


@struct.dataclass
class ActorObservation:
    """Deployment-safe structured policy input."""

    ego: Array
    history: Array
    gate_tokens: Array
    obstacle_tokens: Array
    direct_geometry: Array


@struct.dataclass
class CriticObservation:
    """Privileged value input; never consumed by the deployed actor."""

    true_gate_tokens: Array
    true_obstacle_tokens: Array
    true_direct_geometry: Array
    dynamics: Array


@struct.dataclass
class GateGeometry:
    """Directed current-gate geometry shared by observations and rewards."""

    position: Array
    velocity: Array
    time_to_plane: Array
    hit_y: Array
    hit_z: Array
    clearance_y: Array
    clearance_z: Array
    clearance: Array
    prediction_valid: Array
    correct_side: Array
    approach_body: Array
    exit_body: Array
    turn_sin: Array
    turn_cos: Array
    visited: Array


@struct.dataclass
class ObstacleGeometry:
    """Per-obstacle risk geometry in actor-belief coordinates."""

    relative_body: Array
    correction_body: Array
    relative_velocity_body: Array
    visited: Array
    corridor_distance: Array
    time_to_closest: Array
    closest_clearance: Array
    risk: Array
    valid: Array


def _body_velocity(obs: dict[str, Array]) -> Array:
    rotation = quaternion_to_matrix(obs["quat"])
    return jnp.einsum("nji,nj->ni", rotation, obs["vel"])


def _history_frame(
    obs: dict[str, Array], action: Array, previous_body_velocity: Array, previous_ang_vel: Array
) -> tuple[Array, Array]:
    body_velocity = _body_velocity(obs)
    angular_velocity = obs["ang_vel"]
    frame = jnp.concatenate(
        (
            jnp.clip(body_velocity / 4.0, -2.0, 2.0),
            jnp.clip(angular_velocity / 10.0, -2.0, 2.0),
            action,
            jnp.clip((body_velocity - previous_body_velocity) / 2.0, -2.0, 2.0),
            jnp.clip((angular_velocity - previous_ang_vel) / 5.0, -2.0, 2.0),
        ),
        axis=-1,
    ).astype(jnp.float32)
    return frame, body_velocity


def initialize_level3_memory(
    obs: dict[str, Array],
    *,
    history_length: int = HISTORY_LENGTH,
    nominal_gates_pos: Array | None = None,
    nominal_gates_quat: Array | None = None,
    nominal_obstacles_pos: Array | None = None,
) -> Level3ObservationMemory:
    """Initialize a belief map from the deployment-visible reset observation."""
    num_envs = obs["pos"].shape[0]
    zeros_action = jnp.zeros((num_envs, ACTION_DIM), dtype=jnp.float32)
    body_velocity = _body_velocity(obs)
    frame, _ = _history_frame(obs, zeros_action, body_velocity, obs["ang_vel"])
    return Level3ObservationMemory(
        nominal_gates_pos=obs["gates_pos"] if nominal_gates_pos is None else nominal_gates_pos,
        nominal_gates_quat=obs["gates_quat"] if nominal_gates_quat is None else nominal_gates_quat,
        nominal_obstacles_pos=(
            obs["obstacles_pos"] if nominal_obstacles_pos is None else nominal_obstacles_pos
        ),
        previous_action=zeros_action,
        previous_body_velocity=body_velocity,
        previous_angular_velocity=obs["ang_vel"],
        history=jnp.repeat(frame[:, None], history_length, axis=1),
    )


def advance_level3_memory(
    memory: Level3ObservationMemory,
    terminal_obs: dict[str, Array],
    reset_obs: dict[str, Array],
    action: Array,
    done: Array,
    *,
    reset_nominal_gates_pos: Array | None = None,
    reset_nominal_gates_quat: Array | None = None,
    reset_nominal_obstacles_pos: Array | None = None,
) -> tuple[Level3ObservationMemory, Level3ObservationMemory]:
    """Advance memory and return reset-aware and uninterrupted terminal versions."""
    frame, body_velocity = _history_frame(
        terminal_obs, action, memory.previous_body_velocity, memory.previous_angular_velocity
    )
    terminal_memory = memory.replace(
        previous_action=action,
        previous_body_velocity=body_velocity,
        previous_angular_velocity=terminal_obs["ang_vel"],
        history=jnp.concatenate((memory.history[:, 1:], frame[:, None]), axis=1),
    )
    reset_memory = initialize_level3_memory(
        reset_obs,
        history_length=memory.history.shape[1],
        nominal_gates_pos=reset_nominal_gates_pos,
        nominal_gates_quat=reset_nominal_gates_quat,
        nominal_obstacles_pos=reset_nominal_obstacles_pos,
    )
    next_memory = jax_tree_where(done, reset_memory, terminal_memory)
    return next_memory, terminal_memory


def jax_tree_where(
    mask: Array, reset: Level3ObservationMemory, active: Level3ObservationMemory
) -> Level3ObservationMemory:
    """Select reset memory for completed worlds without host-side branching."""
    import jax

    def select(reset_value: Array, active_value: Array) -> Array:
        shape = (mask.shape[0],) + (1,) * (reset_value.ndim - 1)
        return jnp.where(mask.reshape(shape), reset_value, active_value)

    return jax.tree.map(select, reset, active)


def _gather_sequence(values: Array, sequence: Array) -> Array:
    return jnp.take_along_axis(values, sequence[..., None], axis=1)


def _current_sequence_value(
    values: Array, sequence: Array, progress: Array, offset: int = 0
) -> Array:
    sequence_index = jnp.clip(progress + offset, 0, sequence.shape[1] - 1)
    physical_index = jnp.take_along_axis(sequence, sequence_index[:, None], axis=1).squeeze(1)
    return values[jnp.arange(values.shape[0]), physical_index]


def _current_sequence_direction(directions: Array, progress: Array) -> Array:
    """Gather a direction by sequence position rather than physical gate ID."""
    sequence_index = jnp.clip(progress, 0, directions.shape[1] - 1)
    return jnp.take_along_axis(directions, sequence_index[:, None], axis=1).squeeze(1)


def _rotation_vector(nominal_quat: Array, belief_quat: Array) -> Array:
    nominal = quaternion_to_matrix(nominal_quat)
    belief = quaternion_to_matrix(belief_quat)
    relative = jnp.einsum("...ji,...jk->...ik", nominal, belief)
    vee = 0.5 * jnp.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        axis=-1,
    )
    cosine = jnp.clip((jnp.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    angle = jnp.arccos(cosine)
    sine = jnp.sin(angle)
    scale = jnp.where(jnp.abs(sine) > 1e-4, angle / sine, 1.0)
    return vee * scale[..., None]


def _sequence_gate_features(obs: dict[str, Array], memory: Level3ObservationMemory) -> Array:
    rotation = quaternion_to_matrix(obs["quat"])
    sequence = obs["gate_sequence"]
    directions = obs["gate_sequence_direction"].astype(jnp.float32)
    centres = _gather_sequence(obs["gates_pos"], sequence)
    quaternions = _gather_sequence(obs["gates_quat"], sequence)
    nominal_centres = _gather_sequence(memory.nominal_gates_pos, sequence)
    nominal_quaternions = _gather_sequence(memory.nominal_gates_quat, sequence)
    visited = jnp.take_along_axis(obs["gates_visited"], sequence, axis=1)
    gate_rotation = quaternion_to_matrix(quaternions)
    relative_world = centres - obs["pos"][:, None]
    relative_body = jnp.einsum("nji,nkj->nki", rotation, relative_world)
    normal_body = jnp.einsum("nji,nkj->nki", rotation, gate_rotation[..., :, 0])
    horizontal_body = jnp.einsum("nji,nkj->nki", rotation, gate_rotation[..., :, 1])
    vertical_body = jnp.einsum("nji,nkj->nki", rotation, gate_rotation[..., :, 2])
    normal_body *= directions[..., None]
    horizontal_body *= directions[..., None]
    correction_body = jnp.einsum("nji,nkj->nki", rotation, centres - nominal_centres)
    correction_rotation = _rotation_vector(nominal_quaternions, quaternions)
    correction_rotation_body = jnp.einsum("nji,nkj->nki", rotation, correction_rotation)
    sequence_index = jnp.arange(sequence.shape[1])[None]
    progress = obs["n_gates_passed"][:, None]
    passed = sequence_index < progress
    current = sequence_index == progress
    future = sequence_index > progress
    offset = jnp.clip((sequence_index - progress) / sequence.shape[1], -1.0, 1.0)
    return jnp.concatenate(
        (
            jnp.clip(relative_body / 3.0, -2.0, 2.0),
            normal_body,
            horizontal_body,
            vertical_body,
            jnp.clip(correction_body / 0.20, -2.0, 2.0),
            jnp.clip(correction_rotation_body / 0.25, -2.0, 2.0),
            visited[..., None].astype(jnp.float32),
            passed[..., None].astype(jnp.float32),
            current[..., None].astype(jnp.float32),
            future[..., None].astype(jnp.float32),
            offset[..., None].astype(jnp.float32),
        ),
        axis=-1,
    ).astype(jnp.float32)


def current_gate_geometry(obs: dict[str, Array], geometry: SafetyGeometry) -> GateGeometry:
    """Compute current-gate planning features in a directed gate frame."""
    progress = obs["n_gates_passed"]
    sequence = obs["gate_sequence"]
    directions = obs["gate_sequence_direction"].astype(jnp.float32)
    centre = _current_sequence_value(obs["gates_pos"], sequence, progress)
    quat = _current_sequence_value(obs["gates_quat"], sequence, progress)
    direction = _current_sequence_direction(directions, progress)
    gate_rotation = quaternion_to_matrix(quat)
    normal = gate_rotation[..., :, 0] * direction[:, None]
    horizontal = gate_rotation[..., :, 1] * direction[:, None]
    vertical = gate_rotation[..., :, 2]
    relative_world = obs["pos"] - centre
    position = jnp.stack(
        (
            jnp.sum(relative_world * normal, axis=-1),
            jnp.sum(relative_world * horizontal, axis=-1),
            jnp.sum(relative_world * vertical, axis=-1),
        ),
        axis=-1,
    )
    velocity = jnp.stack(
        (
            jnp.sum(obs["vel"] * normal, axis=-1),
            jnp.sum(obs["vel"] * horizontal, axis=-1),
            jnp.sum(obs["vel"] * vertical, axis=-1),
        ),
        axis=-1,
    )
    approaching = (position[:, 0] < 0.0) & (velocity[:, 0] > 0.05)
    raw_time = -position[:, 0] / jnp.maximum(velocity[:, 0], 0.05)
    prediction_valid = approaching & (raw_time <= geometry.gate_prediction_horizon)
    time_to_plane = jnp.where(prediction_valid, raw_time, geometry.gate_prediction_horizon)
    hit_y = jnp.where(prediction_valid, position[:, 1] + velocity[:, 1] * raw_time, 0.0)
    hit_z = jnp.where(prediction_valid, position[:, 2] + velocity[:, 2] * raw_time, 0.0)
    clearance_y = geometry.gate_safe_half_width - jnp.abs(hit_y)
    clearance_z = geometry.gate_safe_half_height - jnp.abs(hit_z)
    clearance = jnp.minimum(clearance_y, clearance_z)
    rotation = quaternion_to_matrix(obs["quat"])
    approach_world = centre - normal * geometry.approach_distance
    exit_world = centre + normal * geometry.exit_distance
    approach_body = jnp.einsum("nji,nj->ni", rotation, approach_world - obs["pos"])
    exit_body = jnp.einsum("nji,nj->ni", rotation, exit_world - obs["pos"])
    next_centre = _current_sequence_value(obs["gates_pos"], sequence, progress, offset=1)
    next_vector = next_centre - exit_world
    current_xy = normal[:, :2]
    next_xy = next_vector[:, :2]
    current_xy /= jnp.maximum(jnp.linalg.norm(current_xy, axis=-1, keepdims=True), 1e-6)
    next_xy /= jnp.maximum(jnp.linalg.norm(next_xy, axis=-1, keepdims=True), 1e-6)
    turn_cos = jnp.sum(current_xy * next_xy, axis=-1)
    turn_sin = current_xy[:, 0] * next_xy[:, 1] - current_xy[:, 1] * next_xy[:, 0]
    gate_index = jnp.clip(progress, 0, sequence.shape[1] - 1)
    physical_index = jnp.take_along_axis(sequence, gate_index[:, None], axis=1).squeeze(1)
    visited = obs["gates_visited"][jnp.arange(progress.shape[0]), physical_index]
    return GateGeometry(
        position=position,
        velocity=velocity,
        time_to_plane=time_to_plane,
        hit_y=hit_y,
        hit_z=hit_z,
        clearance_y=clearance_y,
        clearance_z=clearance_z,
        clearance=clearance,
        prediction_valid=prediction_valid,
        correct_side=position[:, 0] < 0.0,
        approach_body=approach_body,
        exit_body=exit_body,
        turn_sin=turn_sin,
        turn_cos=turn_cos,
        visited=visited,
    )


def route_potential(
    obs: dict[str, Array], geometry: SafetyGeometry, *, alignment_temperature: float | None = None
) -> Array:
    """Return a smooth remaining directed route length for the current gate.

    V2 advances route phases from signed plane distance alone. V3 additionally gates those phase
    transitions on opening alignment, so approaching the plane outside the aperture keeps the
    pre-gate centre waypoint active instead of exposing an exit-point shortcut.
    """
    gate = current_gate_geometry(obs, geometry)
    position = gate.position
    approach = jnp.asarray([-geometry.approach_distance, 0.0, 0.0])
    exit_point = jnp.asarray([geometry.exit_distance, 0.0, 0.0])
    to_approach = jnp.linalg.norm(position - approach, axis=-1)
    to_centre = jnp.linalg.norm(position, axis=-1)
    to_exit = jnp.linalg.norm(position - exit_point, axis=-1)
    first_blend = jax_sigmoid(
        (position[:, 0] + geometry.approach_distance) / geometry.route_blend_distance
    )
    second_blend = jax_sigmoid(position[:, 0] / geometry.route_blend_distance)
    if alignment_temperature is not None:
        opening_clearance = jnp.minimum(
            geometry.gate_safe_half_width - jnp.abs(position[:, 1]),
            geometry.gate_safe_half_height - jnp.abs(position[:, 2]),
        )
        alignment = jax_sigmoid(opening_clearance / alignment_temperature)
        first_blend *= alignment
        second_blend *= alignment
    before = to_approach + geometry.approach_distance + geometry.exit_distance
    crossing = to_centre + geometry.exit_distance
    return (1.0 - first_blend) * before + first_blend * (
        (1.0 - second_blend) * crossing + second_blend * to_exit
    )


def jax_sigmoid(value: Array) -> Array:
    """Numerically stable sigmoid without importing a neural-network namespace."""
    return 0.5 * (jnp.tanh(value / 2.0) + 1.0)


def obstacle_geometry(
    obs: dict[str, Array], memory: Level3ObservationMemory, geometry: SafetyGeometry
) -> ObstacleGeometry:
    """Compute belief-based static-obstacle risk features."""
    rotation = quaternion_to_matrix(obs["quat"])
    obstacles = obs["obstacles_pos"]
    nominal = memory.nominal_obstacles_pos
    if obstacles.shape[1] > MAX_OBSTACLES:
        raise ValueError(f"Level 3 V2 supports at most {MAX_OBSTACLES} obstacles")
    padding = MAX_OBSTACLES - obstacles.shape[1]
    valid = jnp.concatenate(
        (
            jnp.ones(obstacles.shape[:2], dtype=bool),
            jnp.zeros((obstacles.shape[0], padding), dtype=bool),
        ),
        axis=1,
    )
    obstacles = jnp.concatenate(
        (obstacles, jnp.broadcast_to(obs["pos"][:, None], (obs["pos"].shape[0], padding, 3))),
        axis=1,
    )
    nominal = jnp.concatenate(
        (nominal, jnp.broadcast_to(obs["pos"][:, None], (obs["pos"].shape[0], padding, 3))), axis=1
    )
    visited = jnp.concatenate(
        (obs["obstacles_visited"], jnp.zeros((obs["pos"].shape[0], padding), dtype=bool)), axis=1
    )
    effective_z = jnp.clip(obs["pos"][:, None, 2], 0.0, obstacles[..., 2])
    effective = obstacles.at[..., 2].set(effective_z)
    nominal_effective = nominal.at[..., 2].set(
        jnp.clip(obs["pos"][:, None, 2], 0.0, nominal[..., 2])
    )
    relative_world = effective - obs["pos"][:, None]
    relative_body = jnp.einsum("nji,nkj->nki", rotation, relative_world)
    correction_body = jnp.einsum("nji,nkj->nki", rotation, effective - nominal_effective)
    relative_velocity_world = jnp.broadcast_to(-obs["vel"][:, None], relative_world.shape)
    relative_velocity_body = jnp.einsum("nji,nkj->nki", rotation, relative_velocity_world)
    velocity_squared = jnp.sum(relative_velocity_world**2, axis=-1)
    time_to_closest = jnp.clip(
        -jnp.sum(relative_world * relative_velocity_world, axis=-1)
        / jnp.maximum(velocity_squared, 1e-6),
        0.0,
        geometry.obstacle_prediction_horizon,
    )
    closest = relative_world + relative_velocity_world * time_to_closest[..., None]
    closest_clearance = jnp.linalg.norm(closest, axis=-1) - geometry.obstacle_safe_radius

    gate = current_gate_geometry(obs, geometry)
    approach_world = obs["pos"] + jnp.einsum("nij,nj->ni", rotation, gate.approach_body)
    segment = approach_world - obs["pos"]
    segment_squared = jnp.sum(segment**2, axis=-1, keepdims=True)
    projection = jnp.clip(
        jnp.sum((effective - obs["pos"][:, None]) * segment[:, None], axis=-1)
        / jnp.maximum(segment_squared, 1e-6),
        0.0,
        1.0,
    )
    closest_corridor = obs["pos"][:, None] + projection[..., None] * segment[:, None]
    corridor_distance = jnp.linalg.norm(effective - closest_corridor, axis=-1)
    normalized_risk = jnp.clip(
        (geometry.obstacle_risk_buffer - closest_clearance) / geometry.obstacle_risk_buffer,
        0.0,
        1.0,
    )
    urgency = jnp.exp(-time_to_closest / 0.35)
    risk = normalized_risk**2 * urgency * valid.astype(jnp.float32)
    return ObstacleGeometry(
        relative_body=relative_body,
        correction_body=correction_body,
        relative_velocity_body=relative_velocity_body,
        visited=visited,
        corridor_distance=corridor_distance,
        time_to_closest=time_to_closest,
        closest_clearance=closest_clearance,
        risk=risk,
        valid=valid,
    )


def _obstacle_tokens(obstacles: ObstacleGeometry, geometry: SafetyGeometry) -> Array:
    return jnp.concatenate(
        (
            jnp.clip(obstacles.relative_body / 3.0, -2.0, 2.0),
            jnp.clip(obstacles.correction_body / 0.20, -2.0, 2.0),
            jnp.clip(obstacles.relative_velocity_body / 4.0, -2.0, 2.0),
            obstacles.visited[..., None].astype(jnp.float32),
            jnp.clip(obstacles.corridor_distance[..., None] / 1.0, 0.0, 2.0),
            (obstacles.time_to_closest / geometry.obstacle_prediction_horizon)[..., None],
            jnp.clip(obstacles.closest_clearance[..., None] / 0.5, -2.0, 2.0),
            obstacles.risk[..., None],
            obstacles.valid[..., None].astype(jnp.float32),
        ),
        axis=-1,
    ).astype(jnp.float32)


def _direct_geometry(
    gate: GateGeometry,
    obstacles: ObstacleGeometry,
    boundary_minimum: Array,
    geometry: SafetyGeometry,
) -> Array:
    risk_index = jnp.argmax(obstacles.risk, axis=1)
    row = jnp.arange(risk_index.shape[0])
    risk_relative = obstacles.relative_body[row, risk_index]
    risk_time = obstacles.time_to_closest[row, risk_index]
    risk_clearance = obstacles.closest_clearance[row, risk_index]
    risk_score = obstacles.risk[row, risk_index]
    risk_valid = obstacles.valid[row, risk_index]
    return jnp.concatenate(
        (
            jnp.clip(gate.position / jnp.asarray([3.0, 0.5, 0.5]), -2.0, 2.0),
            jnp.clip(gate.velocity / 4.0, -2.0, 2.0),
            (gate.time_to_plane / geometry.gate_prediction_horizon)[:, None],
            jnp.clip(jnp.stack((gate.hit_y, gate.hit_z), axis=-1) / 0.3, -2.0, 2.0),
            jnp.clip(
                jnp.stack((gate.clearance_y, gate.clearance_z, gate.clearance), axis=-1) / 0.10,
                -2.0,
                2.0,
            ),
            gate.prediction_valid[:, None].astype(jnp.float32),
            gate.correct_side[:, None].astype(jnp.float32),
            jnp.clip(gate.approach_body / 3.0, -2.0, 2.0),
            jnp.clip(gate.exit_body / 3.0, -2.0, 2.0),
            jnp.stack((gate.turn_sin, gate.turn_cos), axis=-1),
            gate.visited[:, None].astype(jnp.float32),
            jnp.clip(risk_relative / 3.0, -2.0, 2.0),
            (risk_time / geometry.obstacle_prediction_horizon)[:, None],
            jnp.clip(risk_clearance[:, None] / 0.5, -2.0, 2.0),
            risk_score[:, None],
            risk_valid[:, None].astype(jnp.float32),
            jnp.clip(boundary_minimum[:, None] / geometry.boundary_risk_buffer, -2.0, 2.0),
        ),
        axis=-1,
    ).astype(jnp.float32)


ACTOR_INPUT_ABLATIONS = ("history", "direct_geometry", "obstacles", "far_gates", "visited")


def _zero_columns(values: Array, columns: tuple[int, ...]) -> Array:
    return values.at[..., jnp.asarray(columns)].set(0.0)


def mask_actor_observation(actor: ActorObservation, ablations: tuple[str, ...]) -> ActorObservation:
    """Zero the named actor inputs for ablation studies without changing shapes.

    ``obstacles`` also removes the obstacle-risk entries of the direct geometry, ``far_gates``
    keeps only the current and next gate tokens, and ``visited`` removes visited flags together
    with the true-minus-nominal corrections that would reveal them.
    """
    unknown = set(ablations) - set(ACTOR_INPUT_ABLATIONS)
    if unknown:
        raise ValueError(f"Unknown actor input ablations: {sorted(unknown)}")
    if "history" in ablations:
        actor = actor.replace(history=jnp.zeros_like(actor.history))
    if "direct_geometry" in ablations:
        actor = actor.replace(direct_geometry=jnp.zeros_like(actor.direct_geometry))
    if "obstacles" in ablations:
        actor = actor.replace(
            obstacle_tokens=jnp.zeros_like(actor.obstacle_tokens),
            direct_geometry=_zero_columns(actor.direct_geometry, tuple(range(23, 30))),
        )
    if "far_gates" in ablations:
        tokens = actor.gate_tokens
        near = (tokens[..., 20] > 0.5) | (jnp.abs(tokens[..., 22] - 1.0 / tokens.shape[-2]) < 1e-4)
        actor = actor.replace(gate_tokens=jnp.where(near[..., None], tokens, 0.0))
    if "visited" in ablations:
        actor = actor.replace(
            gate_tokens=_zero_columns(actor.gate_tokens, tuple(range(12, 19))),
            obstacle_tokens=_zero_columns(actor.obstacle_tokens, (3, 4, 5, 9)),
            direct_geometry=_zero_columns(actor.direct_geometry, (22,)),
        )
    return actor


def build_actor_observation(
    obs: dict[str, Array],
    memory: Level3ObservationMemory,
    steps: Array,
    max_episode_steps: int,
    pos_limit_low: Array,
    pos_limit_high: Array,
    geometry: SafetyGeometry,
) -> ActorObservation:
    """Build the exact deployment-visible V2 actor observation."""
    rotation = quaternion_to_matrix(obs["quat"])
    body_velocity = jnp.einsum("nji,nj->ni", rotation, obs["vel"])
    lower = obs["pos"] - pos_limit_low
    upper = pos_limit_high - obs["pos"]
    spans = pos_limit_high - pos_limit_low
    boundary = jnp.concatenate((lower / spans, upper / spans), axis=-1)
    progress = obs["n_gates_passed"] / obs["gate_sequence"].shape[1]
    remaining = 1.0 - steps.astype(jnp.float32) / max_episode_steps
    ego = jnp.concatenate(
        (
            obs["pos"][:, 2:3] / pos_limit_high[2],
            jnp.clip(body_velocity / 4.0, -2.0, 2.0),
            jnp.clip(obs["ang_vel"] / 10.0, -2.0, 2.0),
            rotation.reshape((rotation.shape[0], 9)),
            jnp.clip(boundary, -1.0, 2.0),
            progress[:, None],
            remaining[:, None],
            memory.previous_action,
        ),
        axis=-1,
    ).astype(jnp.float32)
    gate = current_gate_geometry(obs, geometry)
    obstacles = obstacle_geometry(obs, memory, geometry)
    boundary_minimum = jnp.min(jnp.concatenate((lower, upper), axis=-1), axis=-1)
    return ActorObservation(
        ego=ego,
        history=memory.history,
        gate_tokens=_sequence_gate_features(obs, memory),
        obstacle_tokens=_obstacle_tokens(obstacles, geometry),
        direct_geometry=_direct_geometry(gate, obstacles, boundary_minimum, geometry),
    )


def build_critic_observation(
    belief_obs: dict[str, Array],
    actor_observation: ActorObservation,
    actual_gates_pos: Array,
    actual_gates_quat: Array,
    actual_obstacles_pos: Array,
    nominal_gates_pos: Array,
    nominal_gates_quat: Array,
    nominal_obstacles_pos: Array,
    mass: Array,
    nominal_mass: Array,
    inertia: Array,
    nominal_inertia: Array,
    external_force: Array,
    difficulty: Array,
    memory: Level3ObservationMemory,
    pos_limit_low: Array,
    pos_limit_high: Array,
    geometry: SafetyGeometry,
) -> CriticObservation:
    """Build privileged features while preserving the actor's separate belief input."""
    true_obs = {
        **belief_obs,
        "gates_pos": actual_gates_pos,
        "gates_quat": actual_gates_quat,
        "obstacles_pos": actual_obstacles_pos,
        "gates_visited": jnp.ones_like(belief_obs["gates_visited"], dtype=bool),
        "obstacles_visited": jnp.ones_like(belief_obs["obstacles_visited"], dtype=bool),
    }
    true_memory = memory.replace(
        nominal_gates_pos=nominal_gates_pos,
        nominal_gates_quat=nominal_gates_quat,
        nominal_obstacles_pos=nominal_obstacles_pos,
    )
    true_gate = current_gate_geometry(true_obs, geometry)
    true_obstacles = obstacle_geometry(true_obs, true_memory, geometry)
    lower = belief_obs["pos"] - pos_limit_low
    upper = pos_limit_high - belief_obs["pos"]
    boundary_minimum = jnp.min(jnp.concatenate((lower, upper), axis=-1), axis=-1)
    mass_ratio = mass.reshape((mass.shape[0], -1))[:, :1] / nominal_mass
    inertia_diagonal = jnp.diagonal(inertia, axis1=-2, axis2=-1).reshape((inertia.shape[0], -1))
    inertia_ratio = inertia_diagonal[:, :3] / nominal_inertia
    force = external_force.reshape((external_force.shape[0], -1))[:, :3] / 0.1
    dynamics = jnp.concatenate(
        (mass_ratio, inertia_ratio, jnp.clip(force, -2.0, 2.0), difficulty[:, None]), axis=-1
    ).astype(jnp.float32)
    del actor_observation  # Kept explicit in the API to document critic conditioning at call sites.
    return CriticObservation(
        true_gate_tokens=_sequence_gate_features(true_obs, true_memory),
        true_obstacle_tokens=_obstacle_tokens(true_obstacles, geometry),
        true_direct_geometry=_direct_geometry(
            true_gate, true_obstacles, boundary_minimum, geometry
        ),
        dynamics=dynamics,
    )


def validate_observation_shapes(
    actor: ActorObservation, critic: CriticObservation, history_length: int = HISTORY_LENGTH
) -> None:
    """Raise a clear error when code and the frozen V2 interface diverge."""
    expected_actor = {
        "ego": EGO_DIM,
        "history": HISTORY_FRAME_DIM,
        "gate_tokens": GATE_TOKEN_DIM,
        "obstacle_tokens": OBSTACLE_TOKEN_DIM,
        "direct_geometry": DIRECT_GEOMETRY_DIM,
    }
    for name, width in expected_actor.items():
        if getattr(actor, name).shape[-1] != width:
            raise ValueError(f"Actor {name} width changed; expected {width}")
    if actor.history.shape[-2] != history_length:
        raise ValueError(f"Actor history length changed; expected {history_length}")
    if critic.true_gate_tokens.shape[-1] != GATE_TOKEN_DIM:
        raise ValueError("Critic gate-token width changed")
    if critic.true_obstacle_tokens.shape[-1] != OBSTACLE_TOKEN_DIM:
        raise ValueError("Critic obstacle-token width changed")
    if critic.true_direct_geometry.shape[-1] != DIRECT_GEOMETRY_DIM:
        raise ValueError("Critic direct-geometry width changed")
    if critic.dynamics.shape[-1] != PRIVILEGED_DYNAMICS_DIM:
        raise ValueError("Critic dynamics width changed")
