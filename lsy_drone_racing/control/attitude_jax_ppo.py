"""Deterministic attitude controller backed by a trained JAX PPO policy."""

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
from lsy_drone_racing.rl_training.checkpoint import METADATA_FILE, load_policy
from lsy_drone_racing.rl_training.environment import hover_action_bias, scale_action
from lsy_drone_racing.rl_training.networks import ActorCritic, deterministic_action
from lsy_drone_racing.rl_training.observations import (
    advance_observation_state,
    initialize_observation_state,
    racing_observation,
)
from lsy_drone_racing.rl_training.runtime import PROJECT_ROOT, initialize_parameters

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _batch_observation(obs: dict[str, Any]) -> dict[str, jax.Array]:
    return {key: jnp.asarray(value)[None] for key, value in obs.items()}


class Controller(BaseController):
    """Run the same bounded Flax policy and 88-dimensional observation path used in training."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Load a local policy checkpoint and initialize observation history."""
        super().__init__(obs, info, config)
        configured = config.controller.get("checkpoint", None)
        checkpoint = os.environ.get("LSY_JAX_PPO_CHECKPOINT", configured)
        if checkpoint is None:
            checkpoint_path = PROJECT_ROOT / "saves" / "jax_ppo" / "fast" / "best"
        else:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"JAX PPO checkpoint not found at {checkpoint_path}. Set "
                "LSY_JAX_PPO_CHECKPOINT or controller.checkpoint."
            )

        metadata_path = (
            checkpoint_path / METADATA_FILE
            if checkpoint_path.is_dir()
            else checkpoint_path.parent / METADATA_FILE
        )
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        training_config = metadata.get("config", {})
        self.history_length = int(training_config.get("history_length", 2))
        self.curriculum_stage = int(training_config.get("curriculum_stage", 2))
        hidden_dim = int(training_config.get("hidden_dim", 256))

        hardware = load_hardware_params(config.sim.drone)
        self.action_low = jnp.asarray(
            [-np.pi / 2, -np.pi / 2, -np.pi / 2, hardware["thrust_min"] * 4], dtype=jnp.float32
        )
        self.action_high = jnp.asarray(
            [np.pi / 2, np.pi / 2, np.pi / 2, hardware["thrust_max"] * 4], dtype=jnp.float32
        )
        model = ActorCritic(
            hidden_dim=hidden_dim,
            action_bias=hover_action_bias(
                self.action_low, self.action_high, float(hardware["mass"] * 9.81)
            ),
        )
        initial_obs = _batch_observation(obs)
        self.observation_state = initialize_observation_state(initial_obs, self.history_length)
        vector = racing_observation(initial_obs, self.observation_state, self.curriculum_stage)
        template = initialize_parameters(model, vector, int(training_config.get("seed", 42)))
        self.params = load_policy(checkpoint_path, template)
        self._apply = jax.jit(model.apply)
        self._last_observation: dict[str, jax.Array] | None = None
        self._last_action: jax.Array | None = None

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return deterministic ``[roll, pitch, yaw, collective thrust]`` control."""
        current = _batch_observation(obs)
        if self._last_observation is None:
            self.observation_state = initialize_observation_state(current, self.history_length)
        else:
            self.observation_state, _ = advance_observation_state(
                self.observation_state,
                self._last_observation,
                current,
                self._last_action,
                jnp.asarray([False]),
            )
        vector = racing_observation(current, self.observation_state, self.curriculum_stage)
        mean, _, _ = self._apply({"params": self.params}, vector)
        normalized_action = deterministic_action(mean)[0]
        physical_action = scale_action(normalized_action, self.action_low, self.action_high)
        self._last_observation = current
        self._last_action = normalized_action[None]
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
        """Let the environment decide when the race is complete."""
        return False

    def episode_reset(self):
        """Reset temporal observation state before the next episode."""
        self._last_observation = None
        self._last_action = None
