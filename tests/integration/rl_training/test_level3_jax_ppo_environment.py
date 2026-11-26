from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.rl_training.environment import scale_action
from lsy_drone_racing.rl_training.level3_algorithm import (
    Level3TrainState,
    build_level3_optimizer,
    make_level3_ppo_update,
)
from lsy_drone_racing.rl_training.level3_checkpoint import (
    restore_level3_checkpoint,
    save_level3_checkpoint,
)
from lsy_drone_racing.rl_training.level3_config import YAW_ACTION_INDEX, get_level3_profile
from lsy_drone_racing.rl_training.level3_environment import (
    initialize_level3_runner,
    make_level3_evaluation_collector,
    make_level3_functional_step,
    make_level3_rollout_collector,
)
from lsy_drone_racing.rl_training.level3_observations import (
    DIRECT_GEOMETRY_DIM,
    EGO_DIM,
    validate_observation_shapes,
)
from lsy_drone_racing.rl_training.level3_runtime import (
    create_level3_environment,
    create_level3_model,
    initialize_level3_parameters,
)
from lsy_drone_racing.rl_training.level3_segment_reset import (
    CORRIDOR_START,
    FULL_START,
    POST_GATE_START,
    PRE_GATE_START,
    _candidate_is_safe,
    apply_level3_segment_reset,
)
from lsy_drone_racing.rl_training.observations import quaternion_to_matrix

if TYPE_CHECKING:
    from pathlib import Path


def run_level3_smoke(
    device: str,
    num_envs: int,
    max_episode_steps: int = 1500,
    profile: str = "level3_v2",
    curriculum_progress: float = 0.0,
) -> tuple:
    """Compile the complete structured rollout and one PPO update."""
    config = get_level3_profile(profile).with_overrides(
        device=device,
        num_envs=num_envs,
        num_steps=2,
        num_minibatches=8 if num_envs >= 8 else 1,
        update_epochs=1,
        total_timesteps=num_envs * 2,
        max_episode_steps=max_episode_steps,
    )
    env = create_level3_environment(config)
    runner = initialize_level3_runner(env, config)
    validate_observation_shapes(
        runner.actor_observation, runner.critic_observation, config.history_length
    )
    model = create_level3_model(env, config)
    params = initialize_level3_parameters(
        model, runner.actor_observation, runner.critic_observation, config.seed
    )
    state = Level3TrainState.create(
        apply_fn=model.apply, params=params, tx=build_level3_optimizer(config)
    )
    runner, rollout, metrics = make_level3_rollout_collector(env, model, config)(
        runner, state.params, jnp.asarray(curriculum_progress)
    )
    state, update_metrics = make_level3_ppo_update(config)(state, rollout, jax.random.key(99))
    jax.block_until_ready(state.step)
    env.close()
    return runner, rollout, metrics, state, update_metrics


@pytest.mark.integration
def test_level3_cpu_rollout_mask_reset_and_full_update():
    runner, rollout, metrics, state, update_metrics = run_level3_smoke("cpu", 2, 1)

    assert rollout.actor_observation.ego.shape == (2, 2, EGO_DIM)
    assert rollout.actor_observation.direct_geometry.shape == (2, 2, DIRECT_GEOMETRY_DIM)
    assert metrics.done.shape == (2, 2)
    assert bool(jnp.all(metrics.done))
    assert bool(jnp.all(runner.environment_data.steps == 0))
    assert int(state.step) == 1
    assert np.all(np.isfinite(rollout.reward))
    assert np.isfinite(float(update_metrics["loss"]))
    assert np.all(np.isfinite(runner.actor_observation.ego))
    np.testing.assert_allclose(runner.observation_memory.previous_action, 0.0)


@pytest.mark.integration
def test_level3_gpu_1024_world_rollout_and_full_update():
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX GPU backend is unavailable")
    runner, rollout, metrics, state, _ = run_level3_smoke(
        "gpu", 1024, profile="level3_v3_route_corridor"
    )

    assert rollout.actor_observation.ego.shape == (2, 1024, EGO_DIM)
    assert metrics.done.shape == (2, 1024)
    assert int(state.step) == 8
    assert bool(jnp.all(jnp.isfinite(rollout.reward)))
    num_passes = runner.environment_data.gate_sequence.shape[0]
    assert set(np.asarray(runner.episode_start_progress).tolist()) == set(range(num_passes))
    assert set(np.asarray(runner.episode_start_kind).tolist()) == {
        FULL_START,
        PRE_GATE_START,
        CORRIDOR_START,
    }


@pytest.mark.integration
def test_level3_segment_state_geometry_visited_elapsed_and_initializers():
    config = get_level3_profile("level3_v2_segment").with_overrides(
        device="cpu",
        num_envs=256,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=512,
    )
    env = create_level3_environment(config)
    runner = initialize_level3_runner(env, config, seed=123)
    data = runner.environment_data
    start = np.asarray(runner.episode_start_progress)
    segment = start > 0
    world = jnp.arange(config.num_envs)
    num_passes = data.gate_sequence.shape[0]
    previous_sequence_index = jnp.clip(runner.episode_start_progress - 1, 0, num_passes - 1)
    previous_gate_id = data.gate_sequence[previous_sequence_index]
    previous_center = data.gates_pos[world, previous_gate_id]
    previous_rotation = quaternion_to_matrix(data.gates_quat[world, previous_gate_id])
    previous_direction = data.gate_sequence_direction[previous_sequence_index]
    normal = previous_rotation[..., :, 0] * previous_direction[:, None]
    horizontal = previous_rotation[..., :, 1] * previous_direction[:, None]
    vertical = previous_rotation[..., :, 2]
    relative = data.sim_data.states.pos[:, 0] - previous_center
    directed_exit = jnp.sum(relative * normal, axis=-1)
    lateral = jnp.sum(relative * horizontal, axis=-1)
    height = jnp.sum(relative * vertical, axis=-1)

    assert set(start.tolist()) == set(range(num_passes))
    assert np.all((np.asarray(directed_exit)[segment] >= 0.119))
    assert np.all((np.asarray(directed_exit)[segment] <= 0.351))
    assert np.all(np.abs(np.asarray(lateral)[segment]) <= 0.041)
    assert np.all(np.abs(np.asarray(height)[segment]) <= 0.041)
    assert np.all(
        np.linalg.norm(np.asarray(data.sim_data.states.vel[:, 0])[segment], axis=-1) < 1.0
    )
    assert np.all(np.abs(np.asarray(data.sim_data.states.ang_vel[:, 0])[segment]) <= 0.5)
    assert np.all(np.asarray(data.steps)[segment] <= 1100)
    np.testing.assert_array_equal(np.asarray(data.steps)[~segment], 0)
    quaternion = np.asarray(data.sim_data.states.quat[:, 0])[segment]
    x, y, z, w = np.moveaxis(quaternion, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    velocity_yaw = np.arctan2(
        np.asarray(data.sim_data.states.vel[:, 0, 1])[segment],
        np.asarray(data.sim_data.states.vel[:, 0, 0])[segment],
    )
    yaw_error = np.arctan2(np.sin(yaw - velocity_yaw), np.cos(yaw - velocity_yaw))
    assert np.all(np.abs(roll) <= config.segment_reset.roll_pitch_perturbation + 1e-5)
    assert np.all(np.abs(pitch) <= config.segment_reset.roll_pitch_perturbation + 1e-5)
    assert np.all(np.abs(yaw_error) <= config.segment_reset.yaw_perturbation + 1e-5)
    simulation_ratio = env.sim.freq // env.settings.freq
    np.testing.assert_array_equal(
        np.asarray(data.sim_data.core.steps[:, 0]), data.steps * simulation_ratio
    )
    assert np.all(
        np.asarray(data.sim_data.states.pos[:, 0])[segment]
        >= np.asarray(data.pos_limit_low + config.geometry.drone_radius)
    )
    assert np.all(
        np.asarray(data.sim_data.states.pos[:, 0])[segment]
        <= np.asarray(data.pos_limit_high - config.geometry.drone_radius)
    )
    obstacle_distance = np.linalg.norm(
        np.asarray(data.sim_data.states.pos[:, 0, None, :2] - data.obstacles_pos[..., :2]), axis=-1
    )
    assert np.all(obstacle_distance[segment] >= config.geometry.obstacle_safe_radius)

    first_gate_id = int(data.gate_sequence[0])
    first_rotation = quaternion_to_matrix(data.gates_quat[:, first_gate_id])
    frame_candidate = data.gates_pos[:, first_gate_id] + 0.15 * first_rotation[..., :, 1]
    assert not bool(jnp.any(_candidate_is_safe(data, frame_candidate, config)))
    boundary_candidate = jnp.broadcast_to(data.pos_limit_low, frame_candidate.shape)
    assert not bool(jnp.any(_candidate_is_safe(data, boundary_candidate, config)))
    assert not bool(jnp.any(_candidate_is_safe(data, data.obstacles_pos[:, 0], config)))

    sequence = np.asarray(data.gate_sequence)
    passed_order = np.arange(num_passes)[None, :] < start[:, None]
    passed_physical = np.stack(
        [
            (passed_order & (sequence == gate)).any(axis=1)
            for gate in range(data.gates_pos.shape[1])
        ],
        axis=1,
    )
    gate_sensor = np.linalg.norm(
        np.asarray(data.sim_data.states.pos[:, 0, None, :2] - data.gates_pos[..., :2]), axis=-1
    ) < float(data.sensor_range[0])
    obstacle_sensor = obstacle_distance < float(data.sensor_range[0])
    np.testing.assert_array_equal(
        np.asarray(data.gates_visited[:, 0]), passed_physical | gate_sensor
    )
    np.testing.assert_array_equal(np.asarray(data.obstacles_visited[:, 0]), obstacle_sensor)
    np.testing.assert_allclose(
        runner.observation_memory.nominal_gates_pos[segment], data.nominal_gates_pos[segment]
    )
    np.testing.assert_allclose(
        runner.observation_memory.nominal_obstacles_pos[segment],
        data.nominal_obstacles_pos[segment],
    )
    np.testing.assert_array_equal(runner.reward_state.last_progress, runner.episode_start_progress)
    np.testing.assert_allclose(runner.reward_state.episode_components, 0.0)
    np.testing.assert_allclose(runner.observation_memory.previous_action, 0.0)
    env.close()


@pytest.mark.integration
def test_level3_segment_reverse_gate_full_start_and_evaluation_reset():
    config = get_level3_profile("level3_v2_segment").with_overrides(
        device="cpu",
        num_envs=8,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=16,
    )
    env = create_level3_environment(config)
    num_passes = env.data.gate_sequence.shape[0]
    forced = config.with_overrides(
        segment_reset=replace(
            config.segment_reset,
            start_probabilities=tuple(float(i == 1) for i in range(num_passes)),
        )
    )
    env.close()
    env = create_level3_environment(forced)
    reset_data, _ = env._reset(env.data, seed=7)
    positive, positive_start, positive_kind = apply_level3_segment_reset(
        reset_data, forced, mixed_segment=True
    )
    reverse_source = reset_data.replace(
        gate_sequence_direction=reset_data.gate_sequence_direction.at[0].set(-1)
    )
    reverse, reverse_start, reverse_kind = apply_level3_segment_reset(
        reverse_source, forced, mixed_segment=True
    )
    gate_id = int(reset_data.gate_sequence[0])
    rotation = quaternion_to_matrix(reset_data.gates_quat[:, gate_id])
    physical_normal = rotation[..., :, 0]
    positive_delta = positive.sim_data.states.pos[:, 0] - reset_data.gates_pos[:, gate_id]
    reverse_delta = reverse.sim_data.states.pos[:, 0] - reset_data.gates_pos[:, gate_id]
    assert bool(jnp.all(jnp.sum(positive_delta * physical_normal, axis=-1) > 0.0))
    assert bool(jnp.all(jnp.sum(reverse_delta * physical_normal, axis=-1) < 0.0))
    np.testing.assert_array_equal(positive_start, 1)
    np.testing.assert_array_equal(reverse_start, 1)
    np.testing.assert_array_equal(positive_kind, POST_GATE_START)
    np.testing.assert_array_equal(reverse_kind, POST_GATE_START)

    normal_reset, _ = env._reset(
        env.data.replace(curriculum_progress=jnp.asarray(0.0, dtype=jnp.float32)), seed=99
    )
    full_runner = initialize_level3_runner(env, forced, seed=99, mixed_segment=False)
    for actual, expected in zip(
        jax.tree.leaves(full_runner.environment_data), jax.tree.leaves(normal_reset), strict=True
    ):
        if jax.dtypes.issubdtype(actual.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                jax.random.key_data(actual), jax.random.key_data(expected)
            )
        else:
            np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(full_runner.episode_start_progress, 0)
    np.testing.assert_array_equal(full_runner.episode_start_kind, FULL_START)

    mixed_runner = initialize_level3_runner(env, forced, seed=101)
    mixed_runner = mixed_runner.replace(
        environment_data=mixed_runner.environment_data.replace(
            steps=jnp.full((forced.num_envs,), forced.max_episode_steps - 1)
        )
    )
    model = create_level3_model(env, forced)
    params = initialize_level3_parameters(
        model, mixed_runner.actor_observation, mixed_runner.critic_observation, forced.seed
    )
    evaluated_runner, metrics = make_level3_evaluation_collector(env, model, forced)(
        mixed_runner, params
    )
    assert bool(jnp.all(metrics.done[0]))
    np.testing.assert_array_equal(evaluated_runner.episode_start_progress, 0)
    np.testing.assert_array_equal(evaluated_runner.episode_start_kind, FULL_START)
    env.close()


@pytest.mark.integration
def test_level3_pregate_starts_on_entrance_side_and_handles_reverse_gate():
    config = get_level3_profile("level3_v3_crossing_seed").with_overrides(
        device="cpu",
        num_envs=64,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=128,
    )
    env = create_level3_environment(config)
    reset_data, _ = env._reset(env.data, seed=707)
    pregate, start, start_kind = apply_level3_segment_reset(reset_data, config, mixed_segment=True)
    np.testing.assert_array_equal(start_kind, PRE_GATE_START)

    world = jnp.arange(config.num_envs)
    gate_ids = reset_data.gate_sequence[start]
    center = reset_data.gates_pos[world, gate_ids]
    rotation = quaternion_to_matrix(reset_data.gates_quat[world, gate_ids])
    direction = reset_data.gate_sequence_direction[start].astype(jnp.float32)
    directed_normal = rotation[..., :, 0] * direction[:, None]
    delta = pregate.sim_data.states.pos[:, 0] - center
    velocity = pregate.sim_data.states.vel[:, 0]
    signed_distance = jnp.sum(delta * directed_normal, axis=-1)
    forward_speed = jnp.sum(velocity * directed_normal, axis=-1)
    assert bool(jnp.all(signed_distance < -0.11))
    assert bool(jnp.all(signed_distance >= -config.segment_reset.entrance_distance_max - 1e-5))
    assert bool(jnp.all(forward_speed > 0.0))

    sequence = np.asarray(reset_data.gate_sequence)
    passed_order = np.arange(sequence.shape[0])[None, :] < np.asarray(start)[:, None]
    passed_physical = np.stack(
        [
            (passed_order & (sequence == gate)).any(axis=1)
            for gate in range(pregate.gates_pos.shape[1])
        ],
        axis=1,
    )
    gate_sensor = np.linalg.norm(
        np.asarray(pregate.sim_data.states.pos[:, 0, None, :2] - pregate.gates_pos[..., :2]),
        axis=-1,
    ) < float(pregate.sensor_range[0])
    np.testing.assert_array_equal(
        np.asarray(pregate.gates_visited[:, 0]), passed_physical | gate_sensor
    )

    forced = config.with_overrides(
        segment_reset=replace(
            config.segment_reset,
            start_probabilities=tuple(float(i == 0) for i in range(sequence.shape[0])),
        )
    )
    reverse_source = reset_data.replace(
        gate_sequence_direction=reset_data.gate_sequence_direction.at[0].set(-1)
    )
    reverse, reverse_start, reverse_kind = apply_level3_segment_reset(
        reverse_source, forced, mixed_segment=True
    )
    first_gate_id = int(reset_data.gate_sequence[0])
    first_rotation = quaternion_to_matrix(reset_data.gates_quat[:, first_gate_id])
    physical_normal = first_rotation[..., :, 0]
    reverse_delta = reverse.sim_data.states.pos[:, 0] - reset_data.gates_pos[:, first_gate_id]
    reverse_velocity = reverse.sim_data.states.vel[:, 0]
    assert bool(jnp.all(jnp.sum(reverse_delta * physical_normal, axis=-1) > 0.0))
    assert bool(jnp.all(jnp.sum(reverse_velocity * physical_normal, axis=-1) < 0.0))
    np.testing.assert_array_equal(reverse_start, 0)
    np.testing.assert_array_equal(reverse_kind, PRE_GATE_START)

    runner = initialize_level3_runner(env, config, seed=707)
    start_zero = np.asarray(runner.episode_start_progress) == 0
    assert start_zero.any()
    np.testing.assert_allclose(
        runner.observation_memory.nominal_gates_pos[start_zero],
        runner.environment_data.nominal_gates_pos[start_zero],
    )
    np.testing.assert_allclose(
        runner.observation_memory.nominal_obstacles_pos[start_zero],
        runner.environment_data.nominal_obstacles_pos[start_zero],
    )
    env.close()


@pytest.mark.integration
def test_level3_mixed_curriculum_combines_full_post_and_progressive_pregate_starts():
    config = get_level3_profile("level3_v3_mixed_curriculum").with_overrides(
        device="cpu",
        num_envs=512,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=1024,
    )
    env = create_level3_environment(config)

    def reset_at(progress: float, seed: int) -> tuple:
        source, _ = env._reset(
            env.data.replace(curriculum_progress=jnp.asarray(progress, dtype=jnp.float32)),
            seed=seed,
        )
        mixed, start, kind = apply_level3_segment_reset(source, config, mixed_segment=True)
        return source, mixed, np.asarray(start), np.asarray(kind)

    source, mixed, start, kind = reset_at(0.0, 808)
    frequencies = np.bincount(kind, minlength=3) / kind.size
    np.testing.assert_allclose(frequencies, (0.10, 0.20, 0.70), atol=0.05)
    np.testing.assert_array_equal(start[kind == FULL_START], 0)
    last_pass = mixed.gate_sequence.shape[0] - 1
    assert np.all(
        (start[kind == POST_GATE_START] >= 1) & (start[kind == POST_GATE_START] <= last_pass)
    )
    np.testing.assert_allclose(
        mixed.sim_data.states.pos[np.asarray(kind == FULL_START), 0],
        source.sim_data.states.pos[np.asarray(kind == FULL_START), 0],
    )

    world = jnp.arange(config.num_envs)
    pre_target = jnp.clip(jnp.asarray(start), 0, last_pass)
    pre_gate_id = mixed.gate_sequence[pre_target]
    pre_rotation = quaternion_to_matrix(mixed.gates_quat[world, pre_gate_id])
    pre_direction = mixed.gate_sequence_direction[pre_target]
    pre_normal = pre_rotation[..., :, 0] * pre_direction[:, None]
    pre_center = mixed.gates_pos[world, pre_gate_id]
    pre_signed = jnp.sum((mixed.sim_data.states.pos[:, 0] - pre_center) * pre_normal, axis=-1)
    pre_mask = kind == PRE_GATE_START
    assert np.all(np.asarray(pre_signed)[pre_mask] < -0.11)
    assert np.all(np.asarray(pre_signed)[pre_mask] >= -0.181)

    _, wide, wide_start, wide_kind = reset_at(0.40, 909)
    wide_world = jnp.arange(config.num_envs)
    wide_target = jnp.clip(jnp.asarray(wide_start), 0, wide.gate_sequence.shape[0] - 1)
    wide_gate_id = wide.gate_sequence[wide_target]
    wide_rotation = quaternion_to_matrix(wide.gates_quat[wide_world, wide_gate_id])
    wide_direction = wide.gate_sequence_direction[wide_target]
    wide_normal = wide_rotation[..., :, 0] * wide_direction[:, None]
    wide_center = wide.gates_pos[wide_world, wide_gate_id]
    wide_signed = jnp.sum((wide.sim_data.states.pos[:, 0] - wide_center) * wide_normal, axis=-1)
    wide_pre_mask = wide_kind == PRE_GATE_START
    assert np.all(np.asarray(wide_signed)[wide_pre_mask] < -0.29)
    assert np.all(np.asarray(wide_signed)[wide_pre_mask] >= -1.001)
    wide_frequencies = np.bincount(wide_kind, minlength=3) / wide_kind.size
    np.testing.assert_allclose(wide_frequencies, (0.45, 0.30, 0.25), atol=0.05)
    env.close()


@pytest.mark.integration
def test_level3_route_corridor_spans_source_to_gate_entrance():
    config = get_level3_profile("level3_v3_route_corridor").with_overrides(
        device="cpu",
        num_envs=1024,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=2048,
    )
    env = create_level3_environment(config)

    def reset_at(progress: float, seed: int) -> tuple:
        source, _ = env._reset(
            env.data.replace(curriculum_progress=jnp.asarray(progress, dtype=jnp.float32)),
            seed=seed,
        )
        reset, start, kind = apply_level3_segment_reset(source, config, mixed_segment=True)
        return source, reset, np.asarray(start), np.asarray(kind)

    source, reset, start, kind = reset_at(0.0, 1001)
    frequencies = np.bincount(kind, minlength=4) / kind.size
    np.testing.assert_allclose(frequencies, (0.10, 0.0, 0.30, 0.60), atol=0.04)
    np.testing.assert_array_equal(start[kind == FULL_START], 0)
    last_pass = reset.gate_sequence.shape[0] - 1
    assert np.all(
        (start[kind == CORRIDOR_START] >= 0) & (start[kind == CORRIDOR_START] <= last_pass)
    )
    np.testing.assert_array_equal(
        np.asarray(reset.n_gates_passed[:, 0])[kind == CORRIDOR_START],
        start[kind == CORRIDOR_START],
    )
    np.testing.assert_allclose(
        reset.sim_data.states.pos[kind == FULL_START, 0],
        source.sim_data.states.pos[kind == FULL_START, 0],
    )
    assert bool(
        jnp.all(
            _candidate_is_safe(reset, reset.sim_data.states.pos[:, 0], config)
            | jnp.asarray(kind == FULL_START)
        )
    )

    world = jnp.arange(config.num_envs)
    target = jnp.asarray(start)
    target_gate_id = reset.gate_sequence[target]
    target_rotation = quaternion_to_matrix(reset.gates_quat[world, target_gate_id])
    target_direction = reset.gate_sequence_direction[target]
    target_normal = target_rotation[..., :, 0] * target_direction[:, None]
    target_center = reset.gates_pos[world, target_gate_id]
    target_signed = jnp.sum(
        (reset.sim_data.states.pos[:, 0] - target_center) * target_normal, axis=-1
    )
    corridor_mask = kind == CORRIDOR_START
    assert np.all(np.asarray(target_signed)[corridor_mask] < -0.10)
    corridor_speed = np.linalg.norm(
        np.asarray(reset.sim_data.states.vel[:, 0])[corridor_mask], axis=-1
    )
    assert np.all((corridor_speed > 0.55) & (corridor_speed < 0.95))

    late_source, late_reset, late_start, late_kind = reset_at(0.10, 1002)
    late_mask = late_kind == CORRIDOR_START
    late_world = np.arange(config.num_envs)
    previous = np.clip(late_start - 1, 0, late_reset.gate_sequence.shape[0] - 1)
    previous_gate_id = np.asarray(late_reset.gate_sequence)[previous]
    previous_rotation = quaternion_to_matrix(
        late_reset.gates_quat[jnp.asarray(late_world), jnp.asarray(previous_gate_id)]
    )
    previous_direction = np.asarray(late_reset.gate_sequence_direction)[previous]
    previous_normal = np.asarray(previous_rotation[..., :, 0]) * previous_direction[:, None]
    previous_center = np.asarray(late_reset.gates_pos)[late_world, previous_gate_id]
    previous_exit = previous_center + (
        config.segment_reset.corridor_source_exit_distance * previous_normal
    )
    route_source = np.where(
        (late_start == 0)[:, None], np.asarray(late_source.sim_data.states.pos[:, 0]), previous_exit
    )
    late_target_id = np.asarray(late_reset.gate_sequence)[late_start]
    late_target = np.asarray(late_reset.gates_pos)[late_world, late_target_id]
    source_distance = np.linalg.norm(late_target - route_source, axis=-1)
    sampled_distance = np.linalg.norm(
        np.asarray(late_reset.sim_data.states.pos[:, 0]) - route_source, axis=-1
    )
    normalized_distance = sampled_distance / np.maximum(source_distance, 1e-6)
    assert np.any(normalized_distance[late_mask] < 0.25)
    env.close()


@pytest.mark.integration
def test_level3_segment_masked_reset_and_exact_runner_resume(tmp_path: Path):
    config = get_level3_profile("level3_v2_segment").with_overrides(
        device="cpu",
        num_envs=8,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=16,
    )
    env = create_level3_environment(config)
    runner = initialize_level3_runner(env, config, seed=303)
    original_start = runner.episode_start_progress
    original_kind = runner.episode_start_kind
    forced_done = jnp.arange(config.num_envs) % 2 == 0
    runner = runner.replace(
        environment_data=runner.environment_data.replace(
            steps=jnp.where(
                forced_done, config.max_episode_steps - 1, runner.environment_data.steps
            )
        )
    )
    action = jnp.full((config.num_envs, 4), 0.25, dtype=jnp.float32)
    transition = make_level3_functional_step(env, config)(runner, action)
    assert bool(jnp.all(transition.metrics.done[forced_done]))
    active = ~transition.metrics.done
    np.testing.assert_array_equal(
        transition.next_runner.episode_start_progress[active], original_start[active]
    )
    np.testing.assert_array_equal(
        transition.next_runner.episode_start_kind[active], original_kind[active]
    )
    np.testing.assert_allclose(
        transition.next_runner.observation_memory.previous_action[transition.metrics.done], 0.0
    )
    np.testing.assert_allclose(
        transition.next_runner.observation_memory.previous_action[active], action[active]
    )

    model = create_level3_model(env, config)
    params = initialize_level3_parameters(
        model,
        transition.next_runner.actor_observation,
        transition.next_runner.critic_observation,
        config.seed,
    )
    state = Level3TrainState.create(
        apply_fn=model.apply, params=params, tx=build_level3_optimizer(config)
    )
    save_level3_checkpoint(
        tmp_path, state, transition.next_runner, update=4, total_steps=524_288, config=config
    )
    _, restored_runner, update, total_steps = restore_level3_checkpoint(
        tmp_path, state, runner, config
    )
    assert update == 4
    assert total_steps == 524_288
    np.testing.assert_array_equal(
        restored_runner.episode_start_progress, transition.next_runner.episode_start_progress
    )
    np.testing.assert_array_equal(
        restored_runner.episode_start_kind, transition.next_runner.episode_start_kind
    )
    np.testing.assert_array_equal(
        jax.random.key_data(restored_runner.rng), jax.random.key_data(transition.next_runner.rng)
    )
    env.close()


@pytest.mark.integration
def test_level3_level_replay_scores_inserts_and_replays_failed_layouts():
    from lsy_drone_racing.rl_training.level3_config import LevelReplayConfig
    from lsy_drone_racing.rl_training.level3_level_replay import (
        sample_level_replay,
        update_level_replay,
    )

    replay_config = LevelReplayConfig(enabled=True, buffer_size=8, replay_probability=1.0)
    config = get_level3_profile("level3_v18_plr").with_overrides(
        device="cpu", num_envs=4, num_steps=2, total_timesteps=8, level_replay=replay_config
    )
    env = create_level3_environment(config)
    runner = initialize_level3_runner(env, config)
    state, data = runner.level_replay, runner.environment_data
    assert not state.valid.any() and (state.episode_slot == -1).all()

    done = jnp.ones(4, dtype=bool)
    state = update_level_replay(state, data, done, jnp.asarray([0, 2, 4, 4]), 4, replay_config)
    assert int(state.valid.sum()) == 4
    np.testing.assert_allclose(np.sort(np.asarray(state.score[state.valid])), [0, 0, 0.5, 1])
    slot = int(jnp.argmax(state.score))
    np.testing.assert_allclose(state.tracks["gates_pos"][slot], data.gates_pos[0])
    np.testing.assert_allclose(
        state.tracks["nominal_obstacles_pos"][slot], data.nominal_obstacles_pos[0]
    )

    state = update_level_replay(state, data, done, jnp.asarray([1, 1, 1, 4]), 4, replay_config)
    assert int(state.valid.sum()) == 8
    state = update_level_replay(state, data, done, jnp.asarray([3, 3, 4, 4]), 4, replay_config)
    assert int(state.valid.sum()) == 8
    assert float(state.score.min()) == pytest.approx(0.0)
    assert int((state.score == 0.75).sum()) == 3

    reset_data, _ = env._reset(data, seed=3)
    replayed_data, sampled = sample_level_replay(state, reset_data, done, replay_config)
    slots = np.asarray(sampled.episode_slot)
    assert (slots >= 0).all() and np.asarray(sampled.valid)[slots].all()
    np.testing.assert_allclose(replayed_data.gates_pos, state.tracks["gates_pos"][slots])
    np.testing.assert_allclose(replayed_data.gates_quat, state.tracks["gates_quat"][slots])
    drone_xy = replayed_data.sim_data.states.pos[:, :, None, :2]
    expected_visited = (
        jnp.linalg.norm(drone_xy - replayed_data.gates_pos[:, None, :, :2], axis=-1)
        < replayed_data.sensor_range
    )
    np.testing.assert_array_equal(replayed_data.gates_visited, expected_visited)
    assert (np.asarray(sampled.last_sampled)[slots] == int(state.clock)).all()

    before = np.asarray(sampled.score)[slots[0]]
    finished = jnp.asarray([True, False, False, False])
    rescored = update_level_replay(
        sampled, replayed_data, finished, jnp.full(4, 4), 4, replay_config
    )
    assert float(rescored.score[slots[0]]) == pytest.approx(0.5 * before)

    no_replay = replace(replay_config, replay_probability=0.0)
    kept_data, kept = sample_level_replay(state, reset_data, done, no_replay)
    np.testing.assert_allclose(kept_data.gates_pos, reset_data.gates_pos)
    assert (kept.episode_slot == -1).all()
    env.close()


@pytest.mark.integration
def test_level3_level_replay_cpu_rollout_and_update():
    _, _, metrics, state, update_metrics = run_level3_smoke(
        "cpu", 4, max_episode_steps=1, profile="level3_v18_plr"
    )
    assert bool(jnp.all(metrics.done))
    assert np.isfinite(float(update_metrics["loss"]))
    assert int(state.step) > 0


@pytest.mark.integration
def test_level3_full_sensor_range_exposes_true_track_from_reset():
    config = get_level3_profile("level3_v21_fullinfo").with_overrides(
        device="cpu", num_envs=4, num_steps=2, num_minibatches=1, total_timesteps=8
    )
    env = create_level3_environment(config)
    runner = initialize_level3_runner(env, config)
    data = runner.environment_data
    assert bool(jnp.all(data.gates_visited)) and bool(jnp.all(data.obstacles_visited))
    np.testing.assert_allclose(runner.belief_observation["gates_pos"], data.gates_pos)
    np.testing.assert_allclose(runner.belief_observation["obstacles_pos"], data.obstacles_pos)
    np.testing.assert_allclose(runner.observation_memory.nominal_gates_pos, data.gates_pos)
    assert runner.observation_memory.history.shape[1] == 25
    default = get_level3_profile("level3_v20_memory")
    assert default.sensor_range is None
    env.close()


@pytest.mark.integration
def test_level3_yaw_anneal_scales_stored_yaw_and_locks_evaluation():
    config = get_level3_profile("level3_v22_yaw_lock").with_overrides(
        device="cpu", num_envs=4, num_steps=2, num_minibatches=1, total_timesteps=8
    )
    env = create_level3_environment(config)
    runner = initialize_level3_runner(env, config, seed=11)
    action = jnp.full((config.num_envs, 4), 0.5, dtype=jnp.float32)

    def at_progress(progress: float) -> object:
        return runner.replace(
            environment_data=runner.environment_data.replace(
                curriculum_progress=jnp.asarray(progress, dtype=jnp.float32)
            )
        )

    hard = config.with_overrides(yaw_anneal_timesteps=0)
    for step_config, progress, scale in (
        (config, 0.0, 1.0),
        (config, 60e6 / config.curriculum_timesteps, 0.5),
        (config, 1.0, 0.0),
        (hard, 0.0, 0.0),
    ):
        transition = make_level3_functional_step(env, step_config)(at_progress(progress), action)
        active = ~transition.metrics.done
        expected = action.at[:, YAW_ACTION_INDEX].multiply(scale)
        np.testing.assert_allclose(
            transition.next_runner.observation_memory.previous_action[active],
            expected[active],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            transition.next_runner.reward_state.last_action[active], expected[active], atol=1e-6
        )
        np.testing.assert_allclose(transition.metrics.yaw_authority, scale, atol=1e-6)
    low = jnp.asarray(env.single_action_space.low)
    high = jnp.asarray(env.single_action_space.high)
    locked_action = action.at[:, YAW_ACTION_INDEX].set(0.0)
    assert float(scale_action(locked_action, low, high)[0, YAW_ACTION_INDEX]) == 0.0

    model = create_level3_model(env, config)
    params = initialize_level3_parameters(
        model, runner.actor_observation, runner.critic_observation, config.seed
    )
    final, metrics = make_level3_evaluation_collector(env, model, config)(at_progress(0.0), params)
    np.testing.assert_array_equal(metrics.yaw_authority, 0.0)
    np.testing.assert_array_equal(
        final.observation_memory.previous_action[:, YAW_ACTION_INDEX], 0.0
    )
    env.close()

    _, rollout, metrics, state, update_metrics = run_level3_smoke(
        "cpu", 2, 1, profile="level3_v22_yaw_lock"
    )
    assert rollout.yaw_weight.shape == rollout.reward.shape
    np.testing.assert_array_equal(rollout.yaw_weight, 1.0)
    np.testing.assert_array_equal(metrics.yaw_authority, 1.0)
    assert int(state.step) == 1
    assert np.isfinite(float(update_metrics["loss"]))
    _, rollout, metrics, _, update_metrics = run_level3_smoke(
        "cpu", 2, 1, profile="level3_v22_yaw_lock", curriculum_progress=0.5
    )
    np.testing.assert_array_equal(rollout.yaw_weight, 0.0)
    np.testing.assert_array_equal(metrics.yaw_authority, 0.0)
    assert np.isfinite(float(update_metrics["loss"]))
    assert np.all(np.isfinite(rollout.log_probability))
    _, rollout, metrics, _, _ = run_level3_smoke("cpu", 2, 1, profile="level3_v20_memory")
    assert rollout.yaw_weight is None
    np.testing.assert_array_equal(metrics.yaw_authority, 1.0)
