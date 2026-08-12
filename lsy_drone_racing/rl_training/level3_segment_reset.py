"""Pure-JAX mixed gate/segment starts for Level 3 training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from lsy_drone_racing.rl_training.observations import quaternion_to_matrix

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.envs.race_core import EnvData
    from lsy_drone_racing.rl_training.level3_config import Level3PPOConfig

FULL_START = 0
POST_GATE_START = 1
PRE_GATE_START = 2


def _normalize(value: Array) -> Array:
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


def _euler_to_quaternion(roll: Array, pitch: Array, yaw: Array) -> Array:
    """Convert XYZ Euler angles to scipy-order ``[x, y, z, w]`` quaternions."""
    half_roll, half_pitch, half_yaw = roll / 2.0, pitch / 2.0, yaw / 2.0
    cr, sr = jnp.cos(half_roll), jnp.sin(half_roll)
    cp, sp = jnp.cos(half_pitch), jnp.sin(half_pitch)
    cy, sy = jnp.cos(half_yaw), jnp.sin(half_yaw)
    return jnp.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    ).astype(jnp.float32)


def sample_segment_start_progress(
    key: Array, probabilities: tuple[float, ...], count: int
) -> Array:
    """Sample full/segment start IDs ``0..3`` from the frozen profile probabilities."""
    logits = jnp.log(jnp.asarray(probabilities, dtype=jnp.float32))
    return jax.random.categorical(key, logits, shape=(count,)).astype(jnp.int32)


def mixed_start_probabilities(progress: Array, config: Level3PPOConfig) -> Array:
    """Interpolate ``[full, post-gate, pre-gate]`` probabilities on the fixed 500M clock."""
    segment = config.segment_reset
    knots = segment.mixed_progress_knots
    rows = jnp.asarray(
        (
            segment.mixed_start_probabilities_0,
            segment.mixed_start_probabilities_1,
            segment.mixed_start_probabilities_2,
            segment.mixed_start_probabilities_3,
        ),
        dtype=jnp.float32,
    )

    def interpolate(left: Array, right: Array, start: float, end: float) -> Array:
        fraction = jnp.clip((progress - start) / (end - start), 0.0, 1.0)
        return left + fraction * (right - left)

    stage_0 = interpolate(rows[0], rows[1], 0.0, knots[0])
    stage_1 = interpolate(rows[1], rows[2], knots[0], knots[1])
    stage_2 = interpolate(rows[2], rows[3], knots[1], knots[2])
    return jnp.where(
        progress < knots[0],
        stage_0,
        jnp.where(progress < knots[1], stage_1, jnp.where(progress < knots[2], stage_2, rows[3])),
    )


def sample_mixed_start_kind(
    key: Array, progress: Array, config: Level3PPOConfig, count: int
) -> Array:
    """Sample full/post/pre start kinds from the on-device curriculum distribution."""
    probabilities = mixed_start_probabilities(progress, config)
    return jax.random.categorical(key, jnp.log(probabilities), shape=(count,)).astype(jnp.int32)


def _ordered_track(data: EnvData) -> tuple[Array, Array, Array]:
    sequence = data.gate_sequence
    positions = jnp.take_along_axis(
        data.gates_pos, jnp.broadcast_to(sequence[None, :, None], data.gates_pos.shape), axis=1
    )
    quaternions = jnp.take_along_axis(
        data.gates_quat, jnp.broadcast_to(sequence[None, :, None], data.gates_quat.shape), axis=1
    )
    directions = jnp.broadcast_to(
        data.gate_sequence_direction[None].astype(jnp.float32), positions.shape[:2]
    )
    return positions, quaternions, directions


def _candidate_is_safe(data: EnvData, candidate: Array, config: Level3PPOConfig) -> Array:
    geometry = config.geometry
    segment = config.segment_reset
    boundary_safe = jnp.all(
        (candidate >= data.pos_limit_low + geometry.drone_radius)
        & (candidate <= data.pos_limit_high - geometry.drone_radius),
        axis=-1,
    )

    gate_rotation = quaternion_to_matrix(data.gates_quat)
    gate_relative = candidate[:, None] - data.gates_pos
    gate_local = jnp.einsum("ngij,ngi->ngj", gate_rotation, gate_relative)
    near_gate_plane = (
        jnp.abs(gate_local[..., 0]) <= geometry.drone_radius + segment.gate_plane_half_thickness
    )
    within_gate_extent = jnp.all(
        jnp.abs(gate_local[..., 1:]) <= geometry.gate_inner_half_width + segment.gate_outer_buffer,
        axis=-1,
    )
    within_opening = jnp.all(jnp.abs(gate_local[..., 1:]) <= geometry.gate_safe_half_width, axis=-1)
    gate_safe = ~jnp.any(near_gate_plane & within_gate_extent & ~within_opening, axis=-1)

    obstacle_xy = candidate[:, None, :2] - data.obstacles_pos[..., :2]
    obstacle_safe = jnp.all(
        jnp.linalg.norm(obstacle_xy, axis=-1) >= geometry.obstacle_safe_radius, axis=-1
    )
    return boundary_safe & gate_safe & obstacle_safe


def apply_level3_segment_reset(
    data: EnvData, config: Level3PPOConfig, mask: Array | None = None, *, mixed_segment: bool
) -> tuple[EnvData, Array, Array]:
    """Apply a training-only segment transform after the ordinary RaceCore reset.

    The ordinary reset remains responsible for the random track, uncertainty, dynamics and clean
    controller state. This transform only replaces the reset state for worlds sampled with a
    non-zero post-gate start progress. The diagnostic ``pre_gate`` mode instead transforms every
    world to the entrance side of its sampled target gate. If all three candidates are unsafe,
    that world safely falls back to the original start and reports start progress zero. The third
    return value is the start kind (0 full, 1 post-gate, 2 pre-gate), which also identifies
    transformed worlds when a pre-gate gate-1 start has legitimate progress zero.
    """
    count = data.sim_data.core.n_worlds
    reset_mask = jnp.ones((count,), dtype=bool) if mask is None else mask
    if not mixed_segment or not config.segment_reset.enabled:
        zeros = jnp.zeros((count,), dtype=jnp.int32)
        return data, zeros, zeros

    segment = config.segment_reset
    keys = jax.random.split(data.sim_data.core.rng_key, 14)
    next_key = keys[0]
    pre_distance_min = segment.entrance_distance_min
    pre_distance_max = segment.entrance_distance_max
    pre_lateral_limit = segment.entrance_lateral_offset
    pre_vertical_limit = segment.entrance_vertical_offset
    pre_speed_min = segment.entrance_speed_min
    pre_speed_max = segment.entrance_speed_max
    pre_velocity_perturbation = segment.entrance_velocity_perturbation
    if segment.start_mode == "mixed_curriculum":
        sampled_kind = sample_mixed_start_kind(keys[1], data.curriculum_progress, config, count)
        pre_start = sample_segment_start_progress(keys[2], segment.start_probabilities, count)
        post_start = 1 + jax.random.categorical(
            keys[3], jnp.zeros((3,), dtype=jnp.float32), shape=(count,)
        ).astype(jnp.int32)
        sampled_start = jnp.where(
            sampled_kind == PRE_GATE_START,
            pre_start,
            jnp.where(sampled_kind == POST_GATE_START, post_start, 0),
        )
        geometry_fraction = jnp.clip(
            data.curriculum_progress / segment.mixed_geometry_progress_end, 0.0, 1.0
        )

        def mix(initial: float, final: float) -> Array:
            return initial + geometry_fraction * (final - initial)

        pre_distance_min = mix(segment.entrance_distance_min, segment.mixed_entrance_distance_min)
        pre_distance_max = mix(segment.entrance_distance_max, segment.mixed_entrance_distance_max)
        pre_lateral_limit = mix(segment.entrance_lateral_offset, segment.mixed_entrance_offset)
        pre_vertical_limit = mix(segment.entrance_vertical_offset, segment.mixed_entrance_offset)
        pre_speed_min = mix(segment.entrance_speed_min, segment.mixed_entrance_speed_min)
        pre_speed_max = mix(segment.entrance_speed_max, segment.mixed_entrance_speed_max)
        pre_velocity_perturbation = mix(
            segment.entrance_velocity_perturbation, segment.mixed_entrance_velocity_perturbation
        )
    elif segment.start_mode == "pre_gate":
        sampled_kind = jnp.full((count,), PRE_GATE_START, dtype=jnp.int32)
        sampled_start = sample_segment_start_progress(keys[2], segment.start_probabilities, count)
    else:
        sampled_start = sample_segment_start_progress(keys[2], segment.start_probabilities, count)
        sampled_kind = jnp.where(sampled_start > 0, POST_GATE_START, FULL_START).astype(jnp.int32)
    pre_gate = sampled_kind == PRE_GATE_START
    distance_min = jnp.where(pre_gate, pre_distance_min, segment.exit_distance_min)
    distance_max = jnp.where(pre_gate, pre_distance_max, segment.exit_distance_max)
    position_lateral_limit = jnp.where(pre_gate, pre_lateral_limit, segment.lateral_offset)
    position_vertical_limit = jnp.where(pre_gate, pre_vertical_limit, segment.vertical_offset)
    speed_min = jnp.where(pre_gate, pre_speed_min, segment.speed_min)
    speed_max = jnp.where(pre_gate, pre_speed_max, segment.speed_max)
    velocity_perturbation = jnp.where(
        pre_gate, pre_velocity_perturbation, segment.velocity_perturbation
    )
    reset_distance = jax.random.uniform(keys[4], (count,), minval=distance_min, maxval=distance_max)
    lateral = jax.random.uniform(
        keys[5], (count,), minval=-position_lateral_limit, maxval=position_lateral_limit
    )
    vertical = jax.random.uniform(
        keys[6], (count,), minval=-position_vertical_limit, maxval=position_vertical_limit
    )
    speed = jax.random.uniform(keys[7], (count,), minval=speed_min, maxval=speed_max)
    beta = jax.random.uniform(keys[8], (count,), maxval=segment.direction_blend_max)
    velocity_lateral = jax.random.uniform(
        keys[9], (count,), minval=-velocity_perturbation, maxval=velocity_perturbation
    )
    velocity_vertical = jax.random.uniform(
        keys[10], (count,), minval=-velocity_perturbation, maxval=velocity_perturbation
    )
    attitude_noise = jax.random.uniform(keys[11], (count, 3), minval=-1.0, maxval=1.0)
    angular_velocity = jax.random.uniform(
        keys[12], (count, 3), minval=-segment.angular_velocity, maxval=segment.angular_velocity
    )
    elapsed_scale = jax.random.uniform(
        keys[13], (count,), minval=segment.elapsed_scale_min, maxval=segment.elapsed_scale_max
    )

    ordered_pos, ordered_quat, ordered_direction = _ordered_track(data)
    sequence_length = ordered_pos.shape[1]
    world = jnp.arange(count)
    pre_target_index = jnp.clip(sampled_start, 0, sequence_length - 1)
    post_target_index = jnp.clip(sampled_start, 1, sequence_length - 1)
    pre_frame_index = pre_target_index
    post_frame_index = post_target_index - 1

    pre_center = ordered_pos[world, pre_frame_index]
    pre_rotation = quaternion_to_matrix(ordered_quat[world, pre_frame_index])
    pre_direction = ordered_direction[world, pre_frame_index]
    pre_normal = pre_rotation[..., :, 0] * pre_direction[:, None]
    pre_horizontal = pre_rotation[..., :, 1] * pre_direction[:, None]
    pre_vertical = pre_rotation[..., :, 2]
    pre_candidate = (
        pre_center
        - reset_distance[:, None] * pre_normal
        + lateral[:, None] * pre_horizontal
        + vertical[:, None] * pre_vertical
    )
    if segment.start_mode == "mixed_curriculum":
        fallback_pre_distance_1 = jnp.maximum(0.14, 0.75 * distance_min)
        fallback_pre_distance_2 = jnp.maximum(0.12, 0.50 * distance_min)
    else:
        fallback_pre_distance_1 = segment.fallback_entrance_distances[0]
        fallback_pre_distance_2 = segment.fallback_entrance_distances[1]
    pre_fallback_1 = pre_center - jnp.asarray(fallback_pre_distance_1)[..., None] * pre_normal
    pre_fallback_2 = pre_center - jnp.asarray(fallback_pre_distance_2)[..., None] * pre_normal

    post_center = ordered_pos[world, post_frame_index]
    post_rotation = quaternion_to_matrix(ordered_quat[world, post_frame_index])
    post_direction = ordered_direction[world, post_frame_index]
    post_normal = post_rotation[..., :, 0] * post_direction[:, None]
    post_horizontal = post_rotation[..., :, 1] * post_direction[:, None]
    post_vertical = post_rotation[..., :, 2]
    post_candidate = (
        post_center
        + reset_distance[:, None] * post_normal
        + lateral[:, None] * post_horizontal
        + vertical[:, None] * post_vertical
    )
    post_fallback_1 = post_center + segment.fallback_exit_distances[0] * post_normal
    post_fallback_2 = post_center + segment.fallback_exit_distances[1] * post_normal

    target_index = jnp.where(pre_gate, pre_target_index, post_target_index)
    frame_index = jnp.where(pre_gate, pre_frame_index, post_frame_index)
    normal = jnp.where(pre_gate[:, None], pre_normal, post_normal)
    horizontal = jnp.where(pre_gate[:, None], pre_horizontal, post_horizontal)
    vertical_axis = jnp.where(pre_gate[:, None], pre_vertical, post_vertical)
    random_candidate = jnp.where(pre_gate[:, None], pre_candidate, post_candidate)
    fallback_1 = jnp.where(pre_gate[:, None], pre_fallback_1, post_fallback_1)
    fallback_2 = jnp.where(pre_gate[:, None], pre_fallback_2, post_fallback_2)
    safe_random = _candidate_is_safe(data, random_candidate, config)
    safe_fallback_1 = _candidate_is_safe(data, fallback_1, config)
    safe_fallback_2 = _candidate_is_safe(data, fallback_2, config)
    candidate = jnp.where(
        safe_random[:, None],
        random_candidate,
        jnp.where(safe_fallback_1[:, None], fallback_1, fallback_2),
    )
    candidate_valid = safe_random | safe_fallback_1 | safe_fallback_2
    eligible = sampled_kind != FULL_START
    segment_mask = reset_mask & eligible & candidate_valid
    effective_start = jnp.where(segment_mask, sampled_start, 0).astype(jnp.int32)
    effective_kind = jnp.where(segment_mask, sampled_kind, FULL_START).astype(jnp.int32)

    target_center = ordered_pos[world, target_index]
    next_direction = _normalize(target_center - candidate)
    post_travel_direction = _normalize(
        (1.0 - beta[:, None]) * normal + beta[:, None] * next_direction
    )
    travel_direction = jnp.where(pre_gate[:, None], next_direction, post_travel_direction)
    velocity = (
        speed[:, None] * travel_direction
        + velocity_lateral[:, None] * horizontal
        + velocity_vertical[:, None] * vertical_axis
    ).astype(jnp.float32)
    yaw = jnp.arctan2(velocity[:, 1], velocity[:, 0]) + (
        attitude_noise[:, 2] * segment.yaw_perturbation
    )
    quaternion = _euler_to_quaternion(
        attitude_noise[:, 0] * segment.roll_pitch_perturbation,
        attitude_noise[:, 1] * segment.roll_pitch_perturbation,
        yaw,
    )

    initial_position = data.sim_data.states.pos[:, 0]
    first_leg = jnp.linalg.norm(ordered_pos[:, 0] - initial_position, axis=-1)
    between_gates = jnp.linalg.norm(ordered_pos[:, 1:] - ordered_pos[:, :-1], axis=-1)
    cumulative_route = jnp.cumsum(
        jnp.concatenate((first_leg[:, None], between_gates), axis=1), axis=1
    )
    pre_travelled_distance = jnp.maximum(
        cumulative_route[world, target_index] - reset_distance, 0.0
    )
    post_travelled_distance = cumulative_route[world, frame_index]
    travelled_distance = jnp.where(pre_gate, pre_travelled_distance, post_travelled_distance)
    elapsed_seconds = (
        segment.elapsed_base_seconds + travelled_distance / segment.route_speed_estimate
    ) * elapsed_scale
    elapsed_steps = jnp.rint(elapsed_seconds * segment.environment_frequency).astype(jnp.int32)
    elapsed_steps = jnp.minimum(
        elapsed_steps, config.max_episode_steps - segment.minimum_remaining_steps
    )
    elapsed_steps = jnp.where(segment_mask, elapsed_steps, 0)

    state_mask = segment_mask[:, None, None]
    scalar_state_mask = segment_mask[:, None]
    states = data.sim_data.states.replace(
        pos=jnp.where(state_mask, candidate[:, None], data.sim_data.states.pos),
        quat=jnp.where(state_mask, quaternion[:, None], data.sim_data.states.quat),
        vel=jnp.where(state_mask, velocity[:, None], data.sim_data.states.vel),
        ang_vel=jnp.where(state_mask, angular_velocity[:, None], data.sim_data.states.ang_vel),
        force=jnp.where(state_mask, 0.0, data.sim_data.states.force),
        torque=jnp.where(state_mask, 0.0, data.sim_data.states.torque),
    )
    derivatives = data.sim_data.states_deriv.replace(
        vel=jnp.where(state_mask, velocity[:, None], data.sim_data.states_deriv.vel),
        ang_vel=jnp.where(
            state_mask, angular_velocity[:, None], data.sim_data.states_deriv.ang_vel
        ),
        acc=jnp.where(state_mask, 0.0, data.sim_data.states_deriv.acc),
        ang_acc=jnp.where(state_mask, 0.0, data.sim_data.states_deriv.ang_acc),
        rotor_acc=jnp.where(segment_mask[:, None, None], 0.0, data.sim_data.states_deriv.rotor_acc),
    )
    controls = data.sim_data.controls
    if controls.attitude is not None:
        controls = controls.replace(
            attitude=controls.attitude.replace(
                r_int_error=jnp.where(state_mask, 0.0, controls.attitude.r_int_error),
                last_ang_vel=jnp.where(state_mask, 0.0, controls.attitude.last_ang_vel),
            )
        )
    simulation_steps_per_control_step = int(
        data.sim_data.core.freq // segment.environment_frequency
    )
    core = data.sim_data.core.replace(
        rng_key=next_key,
        steps=jnp.where(
            scalar_state_mask,
            elapsed_steps[:, None] * simulation_steps_per_control_step,
            data.sim_data.core.steps,
        ),
        mjx_synced=jnp.asarray(False),
    )
    sim_data = data.sim_data.replace(
        states=states, states_deriv=derivatives, controls=controls, core=core
    )

    physical_gate = jnp.arange(data.gates_pos.shape[1])[None, None, :]
    passed_in_sequence = jnp.arange(sequence_length)[None, :, None] < effective_start[:, None, None]
    sequence_gate = data.gate_sequence[None, :, None] == physical_gate
    passed_gate = jnp.any(passed_in_sequence & sequence_gate, axis=1)
    gate_in_sensor_range = (
        jnp.linalg.norm(candidate[:, None, :2] - data.gates_pos[..., :2], axis=-1)
        < data.sensor_range[0]
    )
    obstacle_in_sensor_range = (
        jnp.linalg.norm(candidate[:, None, :2] - data.obstacles_pos[..., :2], axis=-1)
        < data.sensor_range[0]
    )
    segment_gate_visited = passed_gate | gate_in_sensor_range
    segment_obstacle_visited = obstacle_in_sensor_range

    return (
        data.replace(
            sim_data=sim_data,
            n_gates_passed=jnp.where(
                segment_mask[:, None], effective_start[:, None], data.n_gates_passed
            ),
            last_drone_pos=jnp.where(state_mask, candidate[:, None], data.last_drone_pos),
            takeoff_pos=jnp.where(state_mask, candidate[:, None], data.takeoff_pos),
            steps=jnp.where(segment_mask, elapsed_steps, data.steps),
            gates_visited=jnp.where(
                segment_mask[:, None, None], segment_gate_visited[:, None], data.gates_visited
            ),
            obstacles_visited=jnp.where(
                segment_mask[:, None, None],
                segment_obstacle_visited[:, None],
                data.obstacles_visited,
            ),
            disabled_drones=jnp.where(segment_mask[:, None], False, data.disabled_drones),
            marked_for_reset=jnp.where(segment_mask, False, data.marked_for_reset),
        ),
        effective_start,
        effective_kind,
    )
