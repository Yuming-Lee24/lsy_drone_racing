from __future__ import annotations

import json
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import serialization
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from lsy_drone_racing.control.attitude_jax_ppo import Controller
from lsy_drone_racing.rl_training.algorithm import (
    RolloutBatch,
    build_optimizer,
    generalized_advantage_estimate,
    make_ppo_update,
)
from lsy_drone_racing.rl_training.checkpoint import (
    METADATA_FILE,
    POLICY_FILE,
    restore_checkpoint,
    save_checkpoint,
)
from lsy_drone_racing.rl_training.config import RewardCoefficients, get_profile
from lsy_drone_racing.rl_training.environment import RunnerState
from lsy_drone_racing.rl_training.evaluation import wilson_interval
from lsy_drone_racing.rl_training.networks import (
    ActorCritic,
    sample_action,
    squashed_log_probability,
)
from lsy_drone_racing.rl_training.observations import (
    advance_observation_state,
    initialize_observation_state,
    observation_dimension,
    racing_observation,
    target_gate_directions,
    target_gate_indices,
)
from lsy_drone_racing.rl_training.rewards import initialize_reward_state, racing_reward

if TYPE_CHECKING:
    from pathlib import Path


def synthetic_observation(num_envs: int = 2) -> dict[str, jax.Array]:
    """Build a small RaceCore-compatible observation tree."""
    identity = jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    gate_positions = jnp.asarray(
        [[0.0, 0.0, 0.7], [1.0, 0.0, 1.2], [2.0, 0.0, 0.7], [3.0, 0.0, 1.2]], dtype=jnp.float32
    )
    obstacle_positions = jnp.asarray(
        [[0.5, 0.5, 1.55], [1.5, -0.5, 1.55], [2.5, 0.5, 1.55], [3.5, -0.5, 1.55]],
        dtype=jnp.float32,
    )
    return {
        "pos": jnp.tile(jnp.asarray([[0.0, 0.0, 0.2]], dtype=jnp.float32), (num_envs, 1)),
        "quat": jnp.tile(identity[None], (num_envs, 1)),
        "vel": jnp.zeros((num_envs, 3), dtype=jnp.float32),
        "ang_vel": jnp.zeros((num_envs, 3), dtype=jnp.float32),
        "n_gates_passed": jnp.zeros(num_envs, dtype=jnp.int32),
        "gate_sequence": jnp.tile(jnp.asarray([[0, 2, 1, 3]], dtype=jnp.int32), (num_envs, 1)),
        "gate_sequence_direction": jnp.tile(
            jnp.asarray([[1, -1, 1, 1]], dtype=jnp.int32), (num_envs, 1)
        ),
        "gates_pos": jnp.tile(gate_positions[None], (num_envs, 1, 1)),
        "gates_quat": jnp.tile(identity[None, None], (num_envs, 4, 1)),
        "obstacles_pos": jnp.tile(obstacle_positions[None], (num_envs, 1, 1)),
    }


@pytest.mark.unit
def test_observation_layout_gate_sequence_and_reverse_direction():
    obs = synthetic_observation()
    obs = {**obs, "n_gates_passed": jnp.asarray([0, 1], dtype=jnp.int32)}
    state = initialize_observation_state(obs, history_length=2)
    vector = racing_observation(obs, state)

    assert vector.shape == (2, 88)
    assert observation_dimension(2) == 88
    np.testing.assert_array_equal(target_gate_indices(obs), np.asarray([0, 2]))
    np.testing.assert_array_equal(target_gate_indices(obs, 1), np.asarray([2, 1]))
    np.testing.assert_array_equal(target_gate_directions(obs), np.asarray([1, -1]))
    forward_only = {**obs, "gate_sequence_direction": jnp.ones_like(obs["gate_sequence_direction"])}
    forward_vector = racing_observation(forward_only, initialize_observation_state(forward_only, 2))
    assert not np.array_equal(vector[1, 16:28], forward_vector[1, 16:28])


@pytest.mark.unit
def test_history_is_reset_only_for_done_worlds():
    current = synthetic_observation()
    next_obs = {**current, "pos": current["pos"].at[:, 2].set(jnp.asarray([0.3, 0.8]))}
    state = initialize_observation_state(current, history_length=2)
    action = jnp.ones((2, 4), dtype=jnp.float32) * 0.25
    next_state, terminal_state = advance_observation_state(
        state, current, next_obs, action, jnp.asarray([True, False])
    )

    np.testing.assert_allclose(next_state.previous_action[0], 0.0)
    np.testing.assert_allclose(next_state.previous_action[1], action[1])
    np.testing.assert_allclose(terminal_state.previous_action, action)
    assert np.allclose(next_state.history[0, :, 0], 0.3)
    assert np.allclose(next_state.history[1, :, 0], np.asarray([0.2, 0.2]))


@pytest.mark.unit
def test_reward_components_and_terminal_types():
    initial = synthetic_observation()
    state = initialize_reward_state(initial)
    terminal = {**initial, "n_gates_passed": jnp.asarray([1, 4], dtype=jnp.int32)}
    reset = synthetic_observation()
    next_state, transition = racing_reward(
        state,
        terminal,
        reset,
        jnp.zeros((2, 4)),
        jnp.asarray([False, False]),
        jnp.asarray([True, False]),
        50,
        RewardCoefficients(),
    )

    assert bool(transition.truncated[0])
    assert not bool(transition.terminated[0])
    assert bool(transition.success[1])
    assert bool(transition.terminated[1])
    assert float(transition.completed_components[0, 1]) == pytest.approx(20.0)
    assert float(transition.completed_components[1, 2]) == pytest.approx(100.0)
    np.testing.assert_array_equal(next_state.episode_length, np.asarray([0, 0]))

    collision_state = initialize_reward_state(synthetic_observation(1))
    _, collision = racing_reward(
        collision_state,
        synthetic_observation(1),
        synthetic_observation(1),
        jnp.zeros((1, 4)),
        jnp.asarray([True]),
        jnp.asarray([False]),
        50,
        RewardCoefficients(),
    )
    assert bool(collision.collision[0])
    assert float(collision.completed_components[0, 6]) == pytest.approx(-20.0)


@pytest.mark.unit
def test_squashed_gaussian_is_bounded_and_log_probability_consistent():
    mean = jnp.zeros((128, 4))
    log_std = jnp.full((128, 4), -0.5)
    action, raw_action, log_probability = sample_action(mean, log_std, jax.random.key(7))

    assert np.all(np.abs(action) < 1.0)
    np.testing.assert_allclose(
        log_probability, squashed_log_probability(raw_action, mean, log_std), rtol=1e-6
    )
    assert np.all(np.isfinite(log_probability))


@pytest.mark.unit
def test_gae_bootstraps_truncation_but_does_not_cross_episode_boundary():
    reward = jnp.asarray([[1.0], [1.0]])
    value = jnp.asarray([[1.0], [2.0]])
    bootstrap = jnp.asarray([[2.0], [10.0]])
    terminated = jnp.asarray([[False], [False]])
    truncated = jnp.asarray([[False], [True]])
    advantage, _ = generalized_advantage_estimate(
        reward, value, bootstrap, terminated, truncated, gamma=0.9, gae_lambda=0.95
    )
    np.testing.assert_allclose(advantage[:, 0], np.asarray([8.64, 8.0]), rtol=1e-6)

    terminal_advantage, _ = generalized_advantage_estimate(
        reward,
        value,
        bootstrap,
        jnp.asarray([[False], [True]]),
        jnp.asarray([[False], [False]]),
        gamma=0.9,
        gae_lambda=0.95,
    )
    assert float(terminal_advantage[1, 0]) == pytest.approx(-1.0)


@pytest.mark.unit
def test_ppo_update_changes_parameters():
    config = get_profile("fast").with_overrides(
        device="cpu",
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=4,
        hidden_dim=16,
    )
    model = ActorCritic(hidden_dim=16)
    observation = jnp.zeros((4, 88), dtype=jnp.float32)
    params = model.init(jax.random.key(0), observation)["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=build_optimizer(config))
    mean, log_std, value = model.apply({"params": params}, observation)
    raw_action = jnp.ones((4, 4), dtype=jnp.float32) * 0.1
    rollout = RolloutBatch(
        observation=observation.reshape((2, 2, 88)),
        raw_action=raw_action.reshape((2, 2, 4)),
        log_probability=squashed_log_probability(raw_action, mean, log_std).reshape((2, 2)),
        value=value.reshape((2, 2)),
        reward=jnp.ones((2, 2)),
        bootstrap_value=jnp.zeros((2, 2)),
        terminated=jnp.zeros((2, 2), dtype=bool),
        truncated=jnp.zeros((2, 2), dtype=bool),
    )
    updated, metrics = make_ppo_update(config)(state, rollout, jax.random.key(1))

    changed = [
        not np.array_equal(before, after)
        for before, after in zip(jax.tree.leaves(state.params), jax.tree.leaves(updated.params))
    ]
    assert any(changed)
    assert np.isfinite(float(metrics["loss"]))


@pytest.mark.unit
def test_checkpoint_restores_typed_prng_and_optimizer(tmp_path: Path):
    config = get_profile("fast").with_overrides(
        device="cpu",
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=1,
        hidden_dim=8,
    )
    model = ActorCritic(hidden_dim=8)
    params = model.init(jax.random.key(0), jnp.zeros((1, 88)))["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(1e-3))
    obs = synthetic_observation(1)
    observation_state = initialize_observation_state(obs, 2)
    runner = RunnerState(
        environment_data={"key": jax.random.key(11)},
        raw_observation=obs,
        observation_state=observation_state,
        reward_state=initialize_reward_state(obs),
        observation=racing_observation(obs, observation_state),
        rng=jax.random.key(12),
    )
    save_checkpoint(tmp_path, state, runner, 3, 123, config)
    restored, restored_runner, update, total_steps = restore_checkpoint(tmp_path, state, runner)

    assert update == 3
    assert total_steps == 123
    assert jax.dtypes.issubdtype(restored_runner.rng.dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(
        jax.random.key_data(restored_runner.environment_data["key"]), np.asarray([0, 11])
    )
    np.testing.assert_array_equal(
        jax.tree.leaves(restored.params)[0], jax.tree.leaves(state.params)[0]
    )


@pytest.mark.unit
def test_controller_is_deterministic_and_uses_training_observation(tmp_path: Path):
    obs = {key: np.asarray(value[0]) for key, value in synthetic_observation(1).items()}
    hardware_mass = 0.04338
    low = jnp.asarray([-np.pi / 2, -np.pi / 2, -np.pi / 2, 0.08545052])
    high = jnp.asarray([np.pi / 2, np.pi / 2, np.pi / 2, 0.8])
    normalized_hover = (2 * hardware_mass * 9.81 - high[-1] - low[-1]) / (high[-1] - low[-1])
    model = ActorCritic(action_bias=(0.0, 0.0, 0.0, float(jnp.arctanh(normalized_hover))))
    state = initialize_observation_state(_batch := synthetic_observation(1), 2)
    params = model.init(jax.random.key(42), racing_observation(_batch, state))["params"]
    (tmp_path / POLICY_FILE).write_bytes(serialization.to_bytes(params))
    metadata = {
        "config": {"history_length": 2, "curriculum_stage": 2, "hidden_dim": 256, "seed": 42}
    }
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata))
    config = ConfigDict(
        {
            "sim": {"drone": "cf21B_500"},
            "env": {"freq": 50},
            "controller": {"checkpoint": str(tmp_path)},
        }
    )
    controller = Controller(obs, {}, config)
    first = controller.compute_control(obs)
    controller.episode_reset()
    second = controller.compute_control(obs)

    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)
    assert first.shape == (4,)
    assert np.all(first >= np.asarray(low))
    assert np.all(first <= np.asarray(high))


@pytest.mark.unit
def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(325, 500)
    assert low < 0.65 < high
