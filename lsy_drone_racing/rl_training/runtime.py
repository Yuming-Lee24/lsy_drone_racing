"""Shared construction helpers for JAX PPO training and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.rl_training.environment import hover_action_bias
from lsy_drone_racing.rl_training.networks import ActorCritic
from lsy_drone_racing.utils.utils import load_config

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.rl_training.config import PPOConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_race_config(config_file: str) -> Path:
    """Resolve a race TOML relative to the repository's config directory."""
    path = Path(config_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / "config" / path
    if not path.exists():
        raise FileNotFoundError(f"Race configuration does not exist: {path}")
    return path


def create_environment(
    config: PPOConfig, *, num_envs: int | None = None, seed: int | None = None
) -> VecDroneRaceEnv:
    """Create the unwrapped functional racing environment used by the hot path."""
    race_config = load_config(resolve_race_config(config.config_file))
    race_config.sim.render = False
    return VecDroneRaceEnv(
        num_envs=config.num_envs if num_envs is None else num_envs,
        freq=race_config.env.freq,
        sim_config=race_config.sim,
        track=race_config.env.track,
        sensor_range=race_config.env.sensor_range,
        control_mode="attitude",
        disturbances=race_config.env.get("disturbances"),
        randomizations=race_config.env.get("randomizations"),
        seed=config.seed if seed is None else seed,
        max_episode_steps=config.max_episode_steps,
        device=config.device,
    )


def create_model(env: VecDroneRaceEnv, config: PPOConfig) -> ActorCritic:
    """Create an actor-critic with a physical hover point at initialization."""
    low = jnp.asarray(env.single_action_space.low, dtype=jnp.float32)
    high = jnp.asarray(env.single_action_space.high, dtype=jnp.float32)
    nominal_mass = float(np.asarray(env.drone_mass).reshape(-1)[0])
    return ActorCritic(
        hidden_dim=config.hidden_dim, action_bias=hover_action_bias(low, high, nominal_mass * 9.81)
    )


def initialize_parameters(model: ActorCritic, observation: Array, seed: int) -> dict:
    """Initialize model parameters deterministically."""
    import jax

    return model.init(jax.random.key(seed), observation)["params"]
