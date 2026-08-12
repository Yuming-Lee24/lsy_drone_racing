"""Construction helpers shared by Level 3 training, evaluation, and deployment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.race_core import obs
from lsy_drone_racing.rl_training.environment import hover_action_bias
from lsy_drone_racing.rl_training.level3_networks import Level3ActorCritic
from lsy_drone_racing.rl_training.observations import squeeze_single_drone_observation
from lsy_drone_racing.rl_training.runtime import resolve_race_config
from lsy_drone_racing.utils.utils import load_config

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.envs.race_core import EnvData
    from lsy_drone_racing.rl_training.level3_config import Level3PPOConfig
    from lsy_drone_racing.rl_training.level3_observations import ActorObservation, CriticObservation


def create_level3_environment(
    config: Level3PPOConfig,
    *,
    num_envs: int | None = None,
    seed: int | None = None,
    curriculum: bool | None = None,
) -> VecDroneRaceEnv:
    """Create the original Level 3 track in direct-attitude functional mode."""
    race_config = load_config(resolve_race_config(config.config_file))
    race_config.sim.render = False
    use_curriculum = config.curriculum if curriculum is None else curriculum
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
        curriculum=use_curriculum,
    )


def create_level3_model(env: VecDroneRaceEnv, config: Level3PPOConfig) -> Level3ActorCritic:
    """Create the structured model with the physical hover-thrust output bias."""
    low = jnp.asarray(env.single_action_space.low, dtype=jnp.float32)
    high = jnp.asarray(env.single_action_space.high, dtype=jnp.float32)
    nominal_mass = float(np.asarray(env.drone_mass).reshape(-1)[0])
    return Level3ActorCritic(
        entity_dim=config.entity_dim,
        attention_heads=config.attention_heads,
        fusion_dim=config.fusion_dim,
        action_bias=hover_action_bias(low, high, nominal_mass * 9.81),
    )


def initialize_level3_parameters(
    model: Level3ActorCritic,
    actor_observation: ActorObservation,
    critic_observation: CriticObservation,
    seed: int,
) -> dict:
    """Initialize asymmetric model parameters deterministically."""
    return model.init(jax.random.key(seed), actor_observation, critic_observation)["params"]


def belief_observation(data: EnvData) -> dict[str, Array]:
    """Return the mixed nominal/visited observation visible to a deployed policy."""
    return squeeze_single_drone_observation(obs(data))


def true_observation(data: EnvData) -> dict[str, Array]:
    """Return actual current geometry for critic, reward, and diagnostics only."""
    observation = belief_observation(data)
    return {
        **observation,
        "gates_pos": data.gates_pos,
        "gates_quat": data.gates_quat,
        "obstacles_pos": data.obstacles_pos,
    }
