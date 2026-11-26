"""Deterministic deployment controller for the structured Level 3 JAX PPO policy.

Each control step (memory update, observation, Actor, action scaling) is one jit call on the CPU,
compiled in ``__init__``; single-sample inference is faster there than on a GPU.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import load_params as load_dynamics_params

from lsy_drone_racing.control import Controller as BaseController
from lsy_drone_racing.rl_training.checkpoint import METADATA_FILE
from lsy_drone_racing.rl_training.environment import hover_action_bias, scale_action
from lsy_drone_racing.rl_training.level3_checkpoint import (
    checkpoint_gate_sequence,
    load_level3_policy,
)
from lsy_drone_racing.rl_training.level3_config import YAW_ACTION_INDEX, SafetyGeometry
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
    mask_actor_observation,
)
from lsy_drone_racing.rl_training.networks import deterministic_action
from lsy_drone_racing.rl_training.runtime import PROJECT_ROOT

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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
        """Validate a V2 checkpoint, load it on the CPU and compile the control step."""
        super().__init__(obs, info, config)
        self._device = jax.devices("cpu")[0]
        with jax.default_device(self._device):
            self._setup(obs, config)

    def _setup(self, obs: dict[str, NDArray[np.floating]], config: dict):
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
        trained_sequence = checkpoint_gate_sequence(metadata)
        track_sequence = tuple(int(gate) for gate in np.asarray(obs["gate_sequence"]).ravel())
        if trained_sequence != track_sequence:
            logger.warning(
                "Checkpoint was trained on gate order %s, but this track uses %s",
                trained_sequence,
                track_sequence,
            )
        self.max_episode_steps = int(training_config.get("max_episode_steps", 1500))
        self.history_length = int(training_config.get("history_length", 4))
        self.actor_input_mask = tuple(training_config.get("actor_input_mask", ()))
        self.lock_yaw = bool(training_config.get("lock_yaw", False))
        self.geometry = SafetyGeometry(**training_config.get("geometry", {}))
        self.pos_limit_low = jnp.asarray([-3.0, -3.0, 0.0], dtype=jnp.float32)
        self.pos_limit_high = jnp.asarray([3.0, 3.0, 2.5], dtype=jnp.float32)

        hardware = load_dynamics_params(config.sim.dynamics, config.sim.drone)
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
        self.params = jax.device_put(
            load_level3_policy(checkpoint_path, template, ema=not use_raw), self._device
        )

        def policy(params: Any, memory: Any, current: dict, steps: jax.Array) -> tuple:
            actor_observation = build_actor_observation(
                current,
                memory,
                steps,
                self.max_episode_steps,
                self.pos_limit_low,
                self.pos_limit_high,
                self.geometry,
            )
            actor_observation = mask_actor_observation(actor_observation, self.actor_input_mask)
            mean, _ = model.apply({"params": params}, actor_observation, method=model.actor)
            normalized_action = deterministic_action(mean)[0]
            if self.lock_yaw:
                normalized_action = normalized_action.at[YAW_ACTION_INDEX].set(0.0)
            physical_action = scale_action(normalized_action, self.action_low, self.action_high)
            return memory, normalized_action[None], physical_action

        def first_step(params: Any, current: dict, steps: jax.Array) -> tuple:
            memory = initialize_level3_memory(current, history_length=self.history_length)
            return policy(params, memory, current, steps)

        def next_step(
            params: Any, memory: Any, current: dict, last_action: jax.Array, steps: jax.Array
        ) -> tuple:
            memory, _ = advance_level3_memory(
                memory, current, current, last_action, jnp.asarray([False])
            )
            return policy(params, memory, current, steps)

        self._first_step = jax.jit(first_step)
        self._next_step = jax.jit(next_step)
        steps = jnp.zeros((1,), dtype=jnp.int32)
        memory, last_action, _ = self._first_step(self.params, initial, steps)
        jax.block_until_ready(self._next_step(self.params, memory, initial, last_action, steps))
        self._last_action: jax.Array | None = None
        self._steps = 0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return deterministic ``[roll, pitch, yaw, collective thrust]`` control.

        A ``lock_yaw`` checkpoint always commands 0 rad yaw, exactly as during training.
        """
        with jax.default_device(self._device):
            current = _batch_observation(obs)
            steps = jnp.asarray([self._steps], dtype=jnp.int32)
            if self._last_action is None:
                memory, last_action, physical_action = self._first_step(self.params, current, steps)
            else:
                memory, last_action, physical_action = self._next_step(
                    self.params, self.observation_memory, current, self._last_action, steps
                )
        self.observation_memory = memory
        self._last_action = last_action
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
        self._last_action = None
        self._steps = 0
