from __future__ import annotations

from importlib.metadata import version

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.training.train_state import TrainState
from packaging.version import Version

from lsy_drone_racing.rl_training.algorithm import build_optimizer, make_ppo_update
from lsy_drone_racing.rl_training.config import get_profile
from lsy_drone_racing.rl_training.environment import initialize_runner, make_rollout_collector
from lsy_drone_racing.rl_training.runtime import (
    create_environment,
    create_model,
    initialize_parameters,
)


def run_fused_smoke(device: str, num_envs: int, max_episode_steps: int = 1500) -> tuple:
    """Compile rollout and one PPO update on the requested backend."""
    config = get_profile("fast").with_overrides(
        device=device,
        num_envs=num_envs,
        num_steps=2,
        num_minibatches=8 if num_envs >= 8 else 1,
        update_epochs=1,
        total_timesteps=num_envs * 2,
        max_episode_steps=max_episode_steps,
    )
    env = create_environment(config)
    runner = initialize_runner(env, config)
    model = create_model(env, config)
    params = initialize_parameters(model, runner.observation, config.seed)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=build_optimizer(config))
    runner, rollout, metrics = make_rollout_collector(env, model, config)(runner, state.params)
    state, update_metrics = make_ppo_update(config)(state, rollout, jax.random.key(99))
    jax.block_until_ready(state.step)
    env.close()
    return runner, rollout, metrics, state, update_metrics


@pytest.mark.integration
def test_cpu_functional_rollout_mask_reset_and_full_update():
    runner, rollout, metrics, state, update_metrics = run_fused_smoke("cpu", 2, 1)

    assert rollout.observation.shape == (2, 2, 88)
    assert metrics.done.shape == (2, 2)
    assert bool(jnp.all(metrics.done))
    assert bool(jnp.all(runner.environment_data.steps == 0))
    assert int(state.step) == 1
    assert np.all(np.isfinite(rollout.reward))
    assert np.isfinite(float(update_metrics["loss"]))
    assert runner.observation.shape == (2, 88)


@pytest.mark.integration
def test_crazyflow_040_or_newer_within_supported_minor():
    installed = Version(version("crazyflow"))
    assert Version("0.4.0") <= installed < Version("0.5.0")


@pytest.mark.integration
def test_gpu_1024_world_rollout_and_full_update():
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX GPU backend is unavailable")
    _, rollout, _, state, _ = run_fused_smoke("gpu", 1024)

    assert rollout.observation.shape == (2, 1024, 88)
    assert int(state.step) == 8
    assert bool(jnp.all(jnp.isfinite(rollout.reward)))
