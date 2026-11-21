"""Deterministic deployment controller for the structured Level 3 JAX PPO policy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.drones import load_params as load_hardware_params

from lsy_drone_racing.control import Controller as BaseController
from lsy_drone_racing.rl_training.checkpoint import METADATA_FILE
from lsy_drone_racing.rl_training.environment import hover_action_bias, scale_action
from lsy_drone_racing.rl_training.level3_checkpoint import load_level3_policy
from lsy_drone_racing.rl_training.level3_config import SafetyGeometry
from lsy_drone_racing.rl_training.level3_networks import Level3ActorCritic
from lsy_drone_racing.rl_training.level3_observations import (
    DIRECT_GEOMETRY_DIM,
    GATE_TOKEN_DIM,
    OBSTACLE_TOKEN_DIM,
    PRIVILEGED_DYNAMICS_DIM,
    CriticObservation,
    advance_level3_memory,
    build_actor_observation,
    initialize_level3_memory,
)
from lsy_drone_racing.rl_training.networks import deterministic_action
from lsy_drone_racing.rl_training.runtime import PROJECT_ROOT

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _batch_observation(obs: dict[str, Any]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value)[None] for key, value in obs.items()}


def _critic_template(batch_size: int) -> CriticObservation:
    return CriticObservation(
        true_gate_tokens=jnp.zeros((batch_size, 4, GATE_TOKEN_DIM), dtype=jnp.float32),
        true_obstacle_tokens=jnp.zeros((batch_size, 4, OBSTACLE_TOKEN_DIM), dtype=jnp.float32),
        true_direct_geometry=jnp.zeros((batch_size, DIRECT_GEOMETRY_DIM), dtype=jnp.float32),
        dynamics=jnp.zeros((batch_size, PRIVILEGED_DYNAMICS_DIM), dtype=jnp.float32),
    )


class Controller(BaseController):
    """Run only the deployment-safe Actor with training-identical memory and normalization."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Validate a V2 checkpoint and initialize the configured-length belief state."""
        super().__init__(obs, info, config)
        configured = config.controller.get("checkpoint", None)
        checkpoint = os.environ.get("LSY_LEVEL3_JAX_PPO_CHECKPOINT", configured)
        if checkpoint is None:
            checkpoint_path = PROJECT_ROOT / "saves" / "jax_ppo" / "level3_v2" / "best_ema"
        else:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Level 3 JAX PPO checkpoint not found at {checkpoint_path}. Set "
                "LSY_LEVEL3_JAX_PPO_CHECKPOINT or controller.checkpoint."
            )

        metadata_path = (
            checkpoint_path / METADATA_FILE
            if checkpoint_path.is_dir()
            else checkpoint_path.parent / METADATA_FILE
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training_config = metadata["config"]
        self.max_episode_steps = int(training_config.get("max_episode_steps", 1500))
        self.history_length = int(training_config.get("history_length", 4))
        self.geometry = SafetyGeometry(**training_config.get("geometry", {}))
        self.pos_limit_low = jnp.asarray([-3.0, -3.0, 0.0], dtype=jnp.float32)
        self.pos_limit_high = jnp.asarray([3.0, 3.0, 2.5], dtype=jnp.float32)

        hardware = load_hardware_params(config.sim.drone)
        self.action_low = jnp.asarray(
            [-np.pi / 2, -np.pi / 2, -np.pi / 2, hardware["thrust_min"] * 4], dtype=jnp.float32
        )
        self.action_high = jnp.asarray(
            [np.pi / 2, np.pi / 2, np.pi / 2, hardware["thrust_max"] * 4], dtype=jnp.float32
        )
        model = Level3ActorCritic(
            entity_dim=int(training_config.get("entity_dim", 128)),
            attention_heads=int(training_config.get("attention_heads", 4)),
            fusion_dim=int(training_config.get("fusion_dim", 256)),
            action_bias=hover_action_bias(
                self.action_low, self.action_high, float(hardware["mass"] * 9.81)
            ),
        )
        initial = _batch_observation(obs)
        self.observation_memory = initialize_level3_memory(
            initial, history_length=self.history_length
        )
        actor = build_actor_observation(
            initial,
            self.observation_memory,
            jnp.zeros((1,), dtype=jnp.int32),
            self.max_episode_steps,
            self.pos_limit_low,
            self.pos_limit_high,
            self.geometry,
        )
        template = model.init(
            jax.random.key(int(training_config.get("seed", 42))), actor, _critic_template(1)
        )["params"]
        use_raw = bool(config.controller.get("raw_parameters", False))
        self.params = load_level3_policy(checkpoint_path, template, ema=not use_raw)
        self._actor = jax.jit(
            lambda params, observation: model.apply(
                {"params": params}, observation, method=model.actor
            )
        )
        self._last_observation: dict[str, jax.Array] | None = None
        self._last_action: jax.Array | None = None
        self._steps = 0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return deterministic ``[roll, pitch, yaw, collective thrust]`` control."""
        current = _batch_observation(obs)
        if self._last_observation is None:
            self.observation_memory = initialize_level3_memory(
                current, history_length=self.history_length
            )
        else:
            self.observation_memory, _ = advance_level3_memory(
                self.observation_memory, current, current, self._last_action, jnp.asarray([False])
            )
        actor_observation = build_actor_observation(
            current,
            self.observation_memory,
            jnp.asarray([self._steps], dtype=jnp.int32),
            self.max_episode_steps,
            self.pos_limit_low,
            self.pos_limit_high,
            self.geometry,
        )
        mean, _ = self._actor(self.params, actor_observation)
        normalized_action = deterministic_action(mean)[0]
        physical_action = scale_action(normalized_action, self.action_low, self.action_high)
        self._last_observation = current
        self._last_action = normalized_action[None]
        self._steps += 1
        return np.asarray(physical_action, dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Let the competition environment own all terminal decisions."""
        return False

    def episode_reset(self):
        """Clear temporal state before a new independently seeded race."""
        self._last_observation = None
        self._last_action = None
        self._steps = 0
