from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.rl_training.level3_algorithm import (
    Level3TrainState,
    build_level3_optimizer,
    make_level3_ppo_update,
)
from lsy_drone_racing.rl_training.level3_checkpoint import (
    restore_level3_checkpoint,
    save_level3_checkpoint,
)
from lsy_drone_racing.rl_training.level3_config import get_level3_profile
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
    _candidate_is_safe,
    apply_level3_segment_reset,
)
from lsy_drone_racing.rl_training.observations import quaternion_to_matrix

if TYPE_CHECKING:
    from pathlib import Path


def run_level3_smoke(
    device: str, num_envs: int, max_episode_steps: int = 1500, profile: str = "level3_v2"
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
    validate_observation_shapes(runner.actor_observation, runner.critic_observation)
    model = create_level3_model(env, config)
    params = initialize_level3_parameters(
        model, runner.actor_observation, runner.critic_observation, config.seed
    )
    state = Level3TrainState.create(
        apply_fn=model.apply, params=params, tx=build_level3_optimizer(config)
    )
    runner, rollout, metrics = make_level3_rollout_collector(env, model, config)(
        runner, state.params, jnp.asarray(0.0)
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
    runner, rollout, metrics, state, _ = run_level3_smoke("gpu", 1024, profile="level3_v2_segment")

    assert rollout.actor_observation.ego.shape == (2, 1024, EGO_DIM)
    assert metrics.done.shape == (2, 1024)
    assert int(state.step) == 8
    assert bool(jnp.all(jnp.isfinite(rollout.reward)))
    assert set(np.asarray(runner.episode_start_progress).tolist()) == {0, 1, 2, 3}


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
    previous_sequence_index = jnp.clip(runner.episode_start_progress - 1, 0, 3)
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

    assert set(start.tolist()) == {0, 1, 2, 3}
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

    passed_order = np.arange(4)[None, :] < start[:, None]
    passed_physical = np.zeros((config.num_envs, 4), dtype=bool)
    passed_physical[:, np.asarray(data.gate_sequence)] = passed_order
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
    forced = config.with_overrides(
        segment_reset=replace(config.segment_reset, start_probabilities=(0.0, 1.0, 0.0, 0.0))
    )
    env = create_level3_environment(forced)
    reset_data, _ = env._reset(env.data, seed=7)
    positive, positive_start = apply_level3_segment_reset(reset_data, forced, mixed_segment=True)
    reverse_source = reset_data.replace(
        gate_sequence_direction=reset_data.gate_sequence_direction.at[0].set(-1)
    )
    reverse, reverse_start = apply_level3_segment_reset(reverse_source, forced, mixed_segment=True)
    gate_id = int(reset_data.gate_sequence[0])
    rotation = quaternion_to_matrix(reset_data.gates_quat[:, gate_id])
    physical_normal = rotation[..., :, 0]
    positive_delta = positive.sim_data.states.pos[:, 0] - reset_data.gates_pos[:, gate_id]
    reverse_delta = reverse.sim_data.states.pos[:, 0] - reset_data.gates_pos[:, gate_id]
    assert bool(jnp.all(jnp.sum(positive_delta * physical_normal, axis=-1) > 0.0))
    assert bool(jnp.all(jnp.sum(reverse_delta * physical_normal, axis=-1) < 0.0))
    np.testing.assert_array_equal(positive_start, 1)
    np.testing.assert_array_equal(reverse_start, 1)

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
        jax.random.key_data(restored_runner.rng), jax.random.key_data(transition.next_runner.rng)
    )
    env.close()
