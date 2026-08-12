from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from ml_collections import ConfigDict

from lsy_drone_racing.control.attitude_jax_ppo_level3 import Controller
from lsy_drone_racing.envs.race_core import _sample_curriculum_difficulty
from lsy_drone_racing.rl_training.checkpoint import METADATA_FILE, POLICY_FILE
from lsy_drone_racing.rl_training.level3_algorithm import (
    Level3RolloutBatch,
    Level3TrainState,
    build_level3_optimizer,
    level3_generalized_advantage_estimate,
    make_level3_ppo_update,
)
from lsy_drone_racing.rl_training.level3_checkpoint import (
    LEVEL3_FORMAT_VERSION,
    LEVEL3_OBSERVATION_VERSION,
    load_level3_policy,
    restore_level3_checkpoint,
    save_level3_checkpoint,
    validate_level3_metadata,
)
from lsy_drone_racing.rl_training.level3_config import get_level3_profile
from lsy_drone_racing.rl_training.level3_networks import Level3ActorCritic
from lsy_drone_racing.rl_training.level3_observations import (
    DIRECT_GEOMETRY_DIM,
    EGO_DIM,
    GATE_TOKEN_DIM,
    HISTORY_FRAME_DIM,
    HISTORY_LENGTH,
    OBSTACLE_TOKEN_DIM,
    PRIVILEGED_DYNAMICS_DIM,
    ActorObservation,
    CriticObservation,
    advance_level3_memory,
    build_actor_observation,
    current_gate_geometry,
    initialize_level3_memory,
    route_potential,
    validate_observation_shapes,
)
from lsy_drone_racing.rl_training.level3_rewards import (
    LEVEL3_REWARD_COMPONENTS,
    initialize_level3_reward_state,
    level3_reward,
)
from lsy_drone_racing.rl_training.level3_segment_reset import (
    mixed_start_probabilities,
    sample_mixed_start_kind,
    sample_segment_start_progress,
)
from lsy_drone_racing.rl_training.level3_training import (
    _existing_validation_ranks,
    level3_validation_rank,
)
from lsy_drone_racing.rl_training.networks import squashed_log_probability


def synthetic_level3_observation(num_envs: int = 2) -> dict[str, jax.Array]:
    """Build a deployment-compatible four-gate/four-obstacle batch."""
    identity = jnp.asarray([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    gates = jnp.asarray(
        [[1.0, 0.0, 0.7], [2.0, 0.0, 1.2], [3.0, 0.0, 0.7], [4.0, 0.0, 1.2]], dtype=jnp.float32
    )
    obstacles = jnp.asarray(
        [[2.0, 2.0, 1.55], [2.5, -2.0, 1.55], [3.0, 2.0, 1.55], [3.5, -2.0, 1.55]],
        dtype=jnp.float32,
    )
    return {
        "pos": jnp.tile(jnp.asarray([[0.0, 0.0, 0.7]], dtype=jnp.float32), (num_envs, 1)),
        "quat": jnp.tile(identity[None], (num_envs, 1)),
        "vel": jnp.zeros((num_envs, 3), dtype=jnp.float32),
        "ang_vel": jnp.zeros((num_envs, 3), dtype=jnp.float32),
        "n_gates_passed": jnp.zeros(num_envs, dtype=jnp.int32),
        "gate_sequence": jnp.tile(jnp.asarray([[0, 2, 1, 3]], dtype=jnp.int32), (num_envs, 1)),
        "gate_sequence_direction": jnp.tile(
            jnp.asarray([[1, -1, 1, 1]], dtype=jnp.int32), (num_envs, 1)
        ),
        "gates_pos": jnp.tile(gates[None], (num_envs, 1, 1)),
        "gates_quat": jnp.tile(identity[None, None], (num_envs, 4, 1)),
        "gates_visited": jnp.zeros((num_envs, 4), dtype=bool),
        "obstacles_pos": jnp.tile(obstacles[None], (num_envs, 1, 1)),
        "obstacles_visited": jnp.zeros((num_envs, 4), dtype=bool),
    }


def structured_observations(batch: int = 4) -> tuple[ActorObservation, CriticObservation]:
    """Return shape-correct zero structures for optimizer tests."""
    actor = ActorObservation(
        ego=jnp.zeros((batch, EGO_DIM), dtype=jnp.float32),
        history=jnp.zeros((batch, HISTORY_LENGTH, HISTORY_FRAME_DIM), dtype=jnp.float32),
        gate_tokens=jnp.zeros((batch, 4, GATE_TOKEN_DIM), dtype=jnp.float32),
        obstacle_tokens=jnp.ones((batch, 4, OBSTACLE_TOKEN_DIM), dtype=jnp.float32),
        direct_geometry=jnp.zeros((batch, DIRECT_GEOMETRY_DIM), dtype=jnp.float32),
    )
    critic = CriticObservation(
        true_gate_tokens=jnp.zeros((batch, 4, GATE_TOKEN_DIM), dtype=jnp.float32),
        true_obstacle_tokens=jnp.ones((batch, 4, OBSTACLE_TOKEN_DIM), dtype=jnp.float32),
        true_direct_geometry=jnp.zeros((batch, DIRECT_GEOMETRY_DIM), dtype=jnp.float32),
        dynamics=jnp.zeros((batch, PRIVILEGED_DYNAMICS_DIM), dtype=jnp.float32),
    )
    return actor, critic


@pytest.mark.unit
def test_level3_observation_layout_visited_flags_and_reverse_gate_geometry():
    config = get_level3_profile()
    obs = synthetic_level3_observation()
    obs = {
        **obs,
        "n_gates_passed": jnp.asarray([0, 1], dtype=jnp.int32),
        "gates_visited": obs["gates_visited"].at[1, 2].set(True),
    }
    memory = initialize_level3_memory(obs)
    actor = build_actor_observation(
        obs,
        memory,
        jnp.asarray([0, 7]),
        config.max_episode_steps,
        jnp.asarray([-3.0, -3.0, 0.0]),
        jnp.asarray([3.0, 3.0, 2.5]),
        config.geometry,
    )
    _, critic = structured_observations(2)
    validate_observation_shapes(actor, critic)

    assert actor.ego.shape == (2, EGO_DIM)
    assert actor.history.shape == (2, HISTORY_LENGTH, HISTORY_FRAME_DIM)
    assert actor.gate_tokens.shape == (2, 4, GATE_TOKEN_DIM)
    assert actor.obstacle_tokens.shape == (2, 4, OBSTACLE_TOKEN_DIM)
    assert actor.direct_geometry.shape == (2, DIRECT_GEOMETRY_DIM)
    assert float(actor.gate_tokens[1, 1, 18]) == 1.0
    geometry = current_gate_geometry(obs, config.geometry)
    assert float(geometry.position[0, 0]) < 0.0
    assert float(geometry.position[1, 0]) > 0.0
    assert np.all(np.isfinite(np.asarray(jax.tree.leaves(actor)[0])))

    two_obstacles = {
        **obs,
        "obstacles_pos": obs["obstacles_pos"][:, :2],
        "obstacles_visited": obs["obstacles_visited"][:, :2],
    }
    padded_memory = initialize_level3_memory(two_obstacles)
    padded = build_actor_observation(
        two_obstacles,
        padded_memory,
        jnp.zeros((2,), dtype=jnp.int32),
        config.max_episode_steps,
        jnp.asarray([-3.0, -3.0, 0.0]),
        jnp.asarray([3.0, 3.0, 2.5]),
        config.geometry,
    )
    assert padded.obstacle_tokens.shape == (2, 4, OBSTACLE_TOKEN_DIM)
    np.testing.assert_array_equal(padded.obstacle_tokens[:, 2:, -1], 0.0)


@pytest.mark.unit
def test_crossing_prediction_clearance_and_route_potential_telescopes():
    geometry = get_level3_profile().geometry
    obs = synthetic_level3_observation(1)
    obs = {**obs, "vel": jnp.asarray([[1.0, 0.0, 0.0]]), "pos": jnp.asarray([[0.0, 0.0, 0.7]])}
    gate = current_gate_geometry(obs, geometry)
    assert bool(gate.prediction_valid[0])
    assert float(gate.time_to_plane[0]) == pytest.approx(1.0)
    assert float(gate.clearance[0]) == pytest.approx(0.1)

    middle = {**obs, "pos": jnp.asarray([[0.05, 0.0, 0.7]])}
    end = {**obs, "pos": jnp.asarray([[0.10, 0.0, 0.7]])}
    potentials = [float(route_potential(value, geometry)[0]) for value in (obs, middle, end)]
    deltas = (potentials[0] - potentials[1]) + (potentials[1] - potentials[2])
    assert deltas == pytest.approx(potentials[0] - potentials[2], abs=1e-6)


@pytest.mark.unit
def test_reward_v3_route_stays_on_approach_phase_until_aperture_is_aligned():
    config = get_level3_profile("level3_v3_reward")
    observation = synthetic_level3_observation(1)
    aligned_approach = {**observation, "pos": jnp.asarray([[0.5, 0.0, 0.7]], dtype=jnp.float32)}
    unsafe_plane = {**observation, "pos": jnp.asarray([[1.0, 0.7, 0.7]], dtype=jnp.float32)}
    safer_plane = {**observation, "pos": jnp.asarray([[1.0, 0.6, 0.7]], dtype=jnp.float32)}
    temperature = config.reward.alignment_temperature
    aligned_potential = route_potential(
        aligned_approach, config.geometry, alignment_temperature=temperature
    )
    unsafe_potential = route_potential(
        unsafe_plane, config.geometry, alignment_temperature=temperature
    )
    safer_potential = route_potential(
        safer_plane, config.geometry, alignment_temperature=temperature
    )

    assert float(unsafe_potential[0]) > float(aligned_potential[0])
    assert float(safer_potential[0]) < float(unsafe_potential[0])
    assert config.reward.route_progress == 12.0
    assert config.reward.gate == 30.0


@pytest.mark.unit
def test_history_reset_is_masked_and_obstacle_pooling_is_permutation_invariant():
    obs = synthetic_level3_observation()
    memory = initialize_level3_memory(obs)
    terminal = {**obs, "vel": jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])}
    action = jnp.full((2, 4), 0.25)
    next_memory, terminal_memory = advance_level3_memory(
        memory, terminal, obs, action, jnp.asarray([True, False])
    )
    np.testing.assert_allclose(next_memory.previous_action[0], 0.0)
    np.testing.assert_allclose(next_memory.previous_action[1], action[1])
    np.testing.assert_allclose(terminal_memory.previous_action, action)

    actor, critic = structured_observations(2)
    model = Level3ActorCritic(entity_dim=16, attention_heads=4, fusion_dim=32)
    params = model.init(jax.random.key(1), actor, critic)["params"]
    mean, _ = model.apply({"params": params}, actor, method=model.actor)
    permuted = actor.replace(obstacle_tokens=actor.obstacle_tokens[:, ::-1])
    permuted_mean, _ = model.apply({"params": params}, permuted, method=model.actor)
    np.testing.assert_allclose(mean, permuted_mean, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_reward_only_rewards_safe_crossing_speed_and_penalizes_late_failure_more():
    config = get_level3_profile()
    obs = synthetic_level3_observation(1)
    unsafe = {**obs, "pos": jnp.asarray([[0.0, 0.15, 0.7]]), "vel": jnp.asarray([[1.0, 0.0, 0.0]])}
    state = initialize_level3_reward_state(unsafe, config.geometry)
    unsafe_state_after, transition = level3_reward(
        state,
        unsafe,
        obs,
        initialize_level3_memory(unsafe),
        jnp.zeros((1, 4)),
        jnp.asarray([False]),
        jnp.asarray([False]),
        jnp.asarray([-3.0, -3.0, 0.0]),
        jnp.asarray([3.0, 3.0, 2.5]),
        config.reward,
        config.geometry,
    )
    assert float(transition.reward[0]) < 0.0
    safe_speed_index = LEVEL3_REWARD_COMPONENTS.index("safe_gate_speed")
    assert float(unsafe_state_after.episode_components[0, safe_speed_index]) == 0.0
    assert float(transition.completed_components[0].sum()) == 0.0

    safe = {**obs, "pos": jnp.asarray([[0.0, 0.0, 0.7]]), "vel": jnp.asarray([[1.0, 0.0, 0.0]])}
    safe_state = initialize_level3_reward_state(safe, config.geometry)
    safe_state_after, safe_transition = level3_reward(
        safe_state,
        safe,
        obs,
        initialize_level3_memory(safe),
        jnp.zeros((1, 4)),
        jnp.asarray([False]),
        jnp.asarray([False]),
        jnp.asarray([-3.0, -3.0, 0.0]),
        jnp.asarray([3.0, 3.0, 2.5]),
        config.reward,
        config.geometry,
    )
    assert float(safe_transition.reward[0]) > 0.0
    assert float(safe_state_after.episode_components[0, safe_speed_index]) > 0.0

    late = {**obs, "n_gates_passed": jnp.asarray([3])}
    late_state = initialize_level3_reward_state(late, config.geometry)
    _, collision = level3_reward(
        late_state,
        late,
        obs,
        initialize_level3_memory(late),
        jnp.zeros((1, 4)),
        jnp.asarray([True]),
        jnp.asarray([False]),
        jnp.asarray([-3.0, -3.0, 0.0]),
        jnp.asarray([3.0, 3.0, 2.5]),
        config.reward,
        config.geometry,
    )
    collision_index = LEVEL3_REWARD_COMPONENTS.index("collision")
    assert float(collision.completed_components[0, collision_index]) == pytest.approx(-180.0)


@pytest.mark.unit
def test_reward_v3_penalizes_stationary_unsafe_plane_without_prediction_validity():
    config = get_level3_profile("level3_v3_reward")
    observation = synthetic_level3_observation(1)
    unsafe = {
        **observation,
        "pos": jnp.asarray([[1.0, 0.7, 0.7]], dtype=jnp.float32),
        "vel": jnp.zeros((1, 3), dtype=jnp.float32),
    }
    state = initialize_level3_reward_state(unsafe, config.geometry, config.reward)
    next_state, transition = level3_reward(
        state,
        unsafe,
        unsafe,
        initialize_level3_memory(unsafe),
        jnp.zeros((1, 4), dtype=jnp.float32),
        jnp.asarray([False]),
        jnp.asarray([False]),
        jnp.asarray([-3.0, -3.0, 0.0]),
        jnp.asarray([3.0, 3.0, 2.5]),
        config.reward,
        config.geometry,
    )
    risk_index = LEVEL3_REWARD_COMPONENTS.index("gate_risk")

    assert float(next_state.episode_components[0, risk_index]) <= -0.39
    assert float(transition.reward[0]) < 0.0


@pytest.mark.unit
def test_level3_gae_truncation_bootstrap_and_ppo_parameter_update():
    advantage, _ = level3_generalized_advantage_estimate(
        jnp.asarray([[1.0], [1.0]]),
        jnp.asarray([[1.0], [2.0]]),
        jnp.asarray([[2.0], [10.0]]),
        jnp.asarray([[False], [False]]),
        jnp.asarray([[False], [True]]),
        gamma=0.9,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(advantage[:, 0], np.asarray([8.64, 8.0]), rtol=1e-6)

    config = get_level3_profile().with_overrides(
        device="cpu",
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=4,
        entity_dim=16,
        fusion_dim=32,
    )
    actor, critic = structured_observations(4)
    model = Level3ActorCritic(entity_dim=16, attention_heads=4, fusion_dim=32)
    params = model.init(jax.random.key(2), actor, critic)["params"]
    state = Level3TrainState.create(
        apply_fn=model.apply, params=params, tx=build_level3_optimizer(config)
    )
    mean, log_std, value = model.apply({"params": params}, actor, critic)
    raw_action = jnp.full((4, 4), 0.1)
    rollout = Level3RolloutBatch(
        actor_observation=jax.tree.map(lambda x: x.reshape((2, 2, *x.shape[1:])), actor),
        critic_observation=jax.tree.map(lambda x: x.reshape((2, 2, *x.shape[1:])), critic),
        raw_action=raw_action.reshape((2, 2, 4)),
        log_probability=squashed_log_probability(raw_action, mean, log_std).reshape((2, 2)),
        value=value.reshape((2, 2)),
        reward=jnp.ones((2, 2)),
        bootstrap_value=jnp.zeros((2, 2)),
        terminated=jnp.zeros((2, 2), dtype=bool),
        truncated=jnp.zeros((2, 2), dtype=bool),
    )
    before_params = jax.device_get(state.params)
    updated, metrics = make_level3_ppo_update(config)(state, rollout, jax.random.key(3))
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(
            jax.tree.leaves(before_params), jax.tree.leaves(updated.params), strict=True
        )
    )
    assert np.isfinite(float(metrics["loss"]))
    assert float(metrics["epochs_executed"]) == 1.0


@pytest.mark.unit
def test_curriculum_schedule_has_expected_mixture_means():
    samples = 100_000
    early = _sample_curriculum_difficulty(jnp.asarray(0.0), jax.random.key(1), samples)
    middle = _sample_curriculum_difficulty(jnp.asarray(0.3), jax.random.key(2), samples)
    late = _sample_curriculum_difficulty(jnp.asarray(0.6), jax.random.key(3), samples)
    full = _sample_curriculum_difficulty(jnp.asarray(0.8), jax.random.key(4), samples)
    assert float(jnp.mean(early)) == pytest.approx(0.40, abs=0.01)
    assert float(jnp.mean(middle)) == pytest.approx(0.6425, abs=0.01)
    assert float(jnp.mean(late)) == pytest.approx(0.885, abs=0.01)
    np.testing.assert_array_equal(full, np.ones(samples, dtype=np.float32))


@pytest.mark.unit
def test_segment_start_sampling_and_frozen_profile_defaults():
    config = get_level3_profile("level3_v2_segment")
    pregate = get_level3_profile("level3_v3_pregate")
    crossing_seed = get_level3_profile("level3_v3_crossing_seed")
    route_corridor = get_level3_profile("level3_v3_route_corridor")
    samples = sample_segment_start_progress(
        jax.random.key(123), config.segment_reset.start_probabilities, 100_000
    )
    frequencies = np.bincount(np.asarray(samples), minlength=4) / samples.size

    np.testing.assert_allclose(frequencies, 0.25, atol=0.01)
    assert config.segment_reset.enabled
    assert config.profile == "level3_v2_segment"
    assert config.segment_reset.start_mode == "post_gate"
    assert pregate.profile == "level3_v3_pregate"
    assert pregate.segment_reset.start_mode == "pre_gate"
    assert pregate.reward.route_progress == 12.0
    assert crossing_seed.profile == "level3_v3_crossing_seed"
    assert crossing_seed.segment_reset.entrance_distance_min == 0.12
    assert crossing_seed.segment_reset.entrance_distance_max == 0.18
    assert crossing_seed.segment_reset.entrance_speed_min == 0.60
    assert crossing_seed.segment_reset.entrance_speed_max == 0.90
    assert route_corridor.profile == "level3_v3_route_corridor"
    assert route_corridor.segment_reset.start_mode == "route_corridor"
    assert route_corridor.segment_reset.corridor_fraction_min_start == 0.65
    assert route_corridor.segment_reset.corridor_fraction_min_end == 0.0
    assert route_corridor.allow_policy_initialization
    assert config.total_timesteps == 500_000_000
    assert config.curriculum_timesteps == 500_000_000
    assert config.num_updates == 3814


@pytest.mark.unit
def test_mixed_start_curriculum_probabilities_and_sampling():
    config = get_level3_profile("level3_v3_mixed_curriculum")
    expected = (
        (0.0, (0.10, 0.20, 0.70)),
        (0.10, (0.20, 0.25, 0.55)),
        (0.40, (0.45, 0.30, 0.25)),
        (0.80, (0.70, 0.15, 0.15)),
        (1.00, (0.70, 0.15, 0.15)),
    )
    for progress, probabilities in expected:
        np.testing.assert_allclose(
            mixed_start_probabilities(jnp.asarray(progress), config), probabilities, atol=1e-6
        )

    samples = sample_mixed_start_kind(jax.random.key(321), jnp.asarray(0.0), config, 100_000)
    frequencies = np.bincount(np.asarray(samples), minlength=3) / samples.size
    np.testing.assert_allclose(frequencies, (0.10, 0.20, 0.70), atol=0.01)
    assert config.segment_reset.start_mode == "mixed_curriculum"
    assert config.allow_policy_initialization


@pytest.mark.unit
def test_route_corridor_curriculum_probabilities_and_sampling():
    config = get_level3_profile("level3_v3_route_corridor")
    expected = (
        (0.0, (0.10, 0.60, 0.30)),
        (0.10, (0.20, 0.65, 0.15)),
        (0.40, (0.45, 0.50, 0.05)),
        (0.80, (0.70, 0.30, 0.0)),
    )
    for progress, probabilities in expected:
        np.testing.assert_allclose(
            mixed_start_probabilities(jnp.asarray(progress), config), probabilities, atol=1e-6
        )

    samples = sample_mixed_start_kind(jax.random.key(654), jnp.asarray(0.0), config, 100_000)
    frequencies = np.bincount(np.asarray(samples), minlength=4) / samples.size
    np.testing.assert_allclose(frequencies, (0.10, 0.0, 0.30, 0.60), atol=0.01)


@pytest.mark.unit
def test_level3_validation_rank_is_lexicographic():
    base = {
        "successes": 400,
        "total_gates_passed": 1800,
        "collisions": 70,
        "mean_episode_return": 50.0,
    }
    more_success = {**base, "successes": 401, "total_gates_passed": 0}
    more_gates = {**base, "total_gates_passed": 1801, "collisions": 500}
    fewer_collisions = {**base, "collisions": 69, "mean_episode_return": -100.0}
    more_return = {**base, "mean_episode_return": 51.0}

    assert level3_validation_rank(more_success) > level3_validation_rank(base)
    assert level3_validation_rank(more_gates) > level3_validation_rank(base)
    assert level3_validation_rank(fewer_collisions) > level3_validation_rank(base)
    assert level3_validation_rank(more_return) > level3_validation_rank(base)


@pytest.mark.unit
def test_level3_validation_rank_survives_exact_resume(tmp_path: Path):
    records = (
        {
            "checkpoint": "current_raw",
            "successes": 40,
            "total_gates_passed": 180,
            "collisions": 8,
            "mean_episode_return": 50.0,
        },
        {
            "checkpoint": "current_raw",
            "successes": 41,
            "total_gates_passed": 170,
            "collisions": 9,
            "mean_episode_return": 49.0,
        },
        {
            "checkpoint": "current_ema",
            "successes": 39,
            "total_gates_passed": 185,
            "collisions": 7,
            "mean_episode_return": 52.0,
        },
        {"checkpoint": "current_raw", "success_rate": 0.5},
    )
    (tmp_path / "validation.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    ranks = _existing_validation_ranks(tmp_path)
    assert ranks["raw"] == level3_validation_rank(records[1])
    assert ranks["ema"] == level3_validation_rank(records[2])


@pytest.mark.unit
def test_level3_production_path_does_not_import_torch():
    package = Path(__file__).parents[3] / "lsy_drone_racing" / "rl_training"
    production = (
        "level3_algorithm.py",
        "level3_checkpoint.py",
        "level3_config.py",
        "level3_environment.py",
        "level3_evaluation.py",
        "level3_networks.py",
        "level3_observations.py",
        "level3_rewards.py",
        "level3_runtime.py",
        "level3_segment_reset.py",
        "level3_training.py",
    )
    for filename in production:
        source = (package / filename).read_text(encoding="utf-8")
        assert "import torch" not in source
        assert "from torch" not in source


@pytest.mark.unit
def test_checkpoint_version_rejection_and_level3_controller_determinism(tmp_path: Path):
    (tmp_path / METADATA_FILE).write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="V1 policies cannot be loaded"):
        validate_level3_metadata(tmp_path)

    obs_batch = synthetic_level3_observation(1)
    obs = {key: np.asarray(value[0]) for key, value in obs_batch.items()}
    config = get_level3_profile().with_overrides(entity_dim=16, fusion_dim=32)
    memory = initialize_level3_memory(obs_batch)
    actor = build_actor_observation(
        obs_batch,
        memory,
        jnp.zeros((1,), dtype=jnp.int32),
        config.max_episode_steps,
        jnp.asarray([-3.0, -3.0, 0.0]),
        jnp.asarray([3.0, 3.0, 2.5]),
        config.geometry,
    )
    _, critic = structured_observations(1)
    hardware_mass = 0.04338
    low = jnp.asarray([-np.pi / 2, -np.pi / 2, -np.pi / 2, 0.08545052])
    high = jnp.asarray([np.pi / 2, np.pi / 2, np.pi / 2, 0.8])
    normalized_hover = (2 * hardware_mass * 9.81 - high[-1] - low[-1]) / (high[-1] - low[-1])
    model = Level3ActorCritic(
        entity_dim=16,
        attention_heads=4,
        fusion_dim=32,
        action_bias=(0.0, 0.0, 0.0, float(jnp.arctanh(normalized_hover))),
    )
    params = model.init(jax.random.key(42), actor, critic)["params"]
    (tmp_path / POLICY_FILE).write_bytes(serialization.to_bytes(params))
    (tmp_path / "ema_policy.msgpack").write_bytes(serialization.to_bytes(params))
    metadata = {
        "format_version": LEVEL3_FORMAT_VERSION,
        "observation_version": LEVEL3_OBSERVATION_VERSION,
        "config": {
            "max_episode_steps": 1500,
            "entity_dim": 16,
            "attention_heads": 4,
            "fusion_dim": 32,
            "seed": 42,
            "geometry": {},
        },
    }
    (tmp_path / METADATA_FILE).write_text(json.dumps(metadata), encoding="utf-8")
    controller_config = ConfigDict(
        {
            "sim": {"drone": "cf21B_500"},
            "env": {"freq": 50},
            "controller": {"checkpoint": str(tmp_path)},
        }
    )
    controller = Controller(obs, {}, controller_config)
    first = controller.compute_control(obs)
    expected_mean, _ = model.apply({"params": params}, actor, method=model.actor)
    expected_normalized = np.tanh(np.asarray(expected_mean[0]))
    expected_physical = expected_normalized * np.asarray((high - low) / 2.0) + np.asarray(
        (high + low) / 2.0
    )
    np.testing.assert_allclose(first, expected_physical, rtol=1e-6, atol=1e-6)
    controller.episode_reset()
    second = controller.compute_control(obs)
    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)
    assert first.shape == (4,)
    assert np.all(first >= np.asarray(low))
    assert np.all(first <= np.asarray(high))


@pytest.mark.unit
def test_level3_checkpoint_exactly_restores_optimizer_ema_and_typed_key(tmp_path: Path):
    config = get_level3_profile().with_overrides(
        device="cpu",
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        total_timesteps=1,
        curriculum_timesteps=1,
        entity_dim=16,
        fusion_dim=32,
    )
    actor, critic = structured_observations(1)
    model = Level3ActorCritic(entity_dim=16, attention_heads=4, fusion_dim=32)
    params = model.init(jax.random.key(8), actor, critic)["params"]
    state = Level3TrainState.create(
        apply_fn=model.apply, params=params, tx=build_level3_optimizer(config)
    )
    runner = {"rng": jax.random.key(9), "value": jnp.asarray([3.0])}
    save_level3_checkpoint(tmp_path, state, runner, 7, 123, config)
    restored, restored_runner, update, total_steps = restore_level3_checkpoint(
        tmp_path, state, runner, config
    )

    assert update == 7
    assert total_steps == 123
    assert jax.dtypes.issubdtype(restored_runner["rng"].dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(
        jax.tree.leaves(restored.params)[0], jax.tree.leaves(state.params)[0]
    )
    np.testing.assert_array_equal(
        jax.tree.leaves(restored.ema_params)[0], jax.tree.leaves(state.ema_params)[0]
    )

    metadata_path = tmp_path / METADATA_FILE
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = 2
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    policy = load_level3_policy(tmp_path, state.params)
    np.testing.assert_array_equal(jax.tree.leaves(policy)[0], jax.tree.leaves(state.params)[0])
    with pytest.raises(ValueError, match="policy-only compatible"):
        restore_level3_checkpoint(tmp_path, state, runner, config)

    metadata["format_version"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    policy = load_level3_policy(tmp_path, state.params)
    np.testing.assert_array_equal(jax.tree.leaves(policy)[0], jax.tree.leaves(state.params)[0])
    with pytest.raises(ValueError, match="policy-only compatible"):
        restore_level3_checkpoint(tmp_path, state, runner, config)
