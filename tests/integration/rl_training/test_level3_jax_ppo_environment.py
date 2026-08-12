from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lsy_drone_racing.rl_training.level3_algorithm import (
    Level3TrainState,
    build_level3_optimizer,
    make_level3_ppo_update,
)
from lsy_drone_racing.rl_training.level3_config import get_level3_profile
from lsy_drone_racing.rl_training.level3_environment import (
    initialize_level3_runner,
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


def run_level3_smoke(device: str, num_envs: int, max_episode_steps: int = 1500) -> tuple:
    """Compile the complete structured rollout and one PPO update."""
    config = get_level3_profile().with_overrides(
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
    _, rollout, metrics, state, _ = run_level3_smoke("gpu", 1024)

    assert rollout.actor_observation.ego.shape == (2, 1024, EGO_DIM)
    assert metrics.done.shape == (2, 1024)
    assert int(state.step) == 8
    assert bool(jnp.all(jnp.isfinite(rollout.reward)))
