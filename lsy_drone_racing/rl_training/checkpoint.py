"""Local, resumable checkpoints for the JAX PPO pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from flax import serialization

from lsy_drone_racing.rl_training.observations import observation_dimension

if TYPE_CHECKING:
    from pathlib import Path

    from flax.training.train_state import TrainState

    from lsy_drone_racing.rl_training.config import PPOConfig
    from lsy_drone_racing.rl_training.environment import RunnerState

CHECKPOINT_FILE = "checkpoint.msgpack"
POLICY_FILE = "policy.msgpack"
METADATA_FILE = "metadata.json"


def _metadata(config: PPOConfig, total_steps: int, update: int) -> dict[str, Any]:
    return {
        "format_version": 1,
        "crazyflow_version": version("crazyflow"),
        "total_steps": int(total_steps),
        "update": int(update),
        "observation_dimension": observation_dimension(config.history_length),
        "config": asdict(config),
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _is_typed_prng_key(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    return dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)


def _encode_prng_keys(tree: Any) -> Any:
    """Convert JAX typed keys to serializable uint32 key data."""
    return jax.tree.map(
        lambda value: jax.random.key_data(value) if _is_typed_prng_key(value) else value, tree
    )


def _restore_prng_keys(encoded: Any, template: Any) -> Any:
    """Restore typed-key leaves based on a live template tree."""
    return jax.tree.map(
        lambda value, target: (
            jax.random.wrap_key_data(value) if _is_typed_prng_key(target) else value
        ),
        encoded,
        template,
    )


def save_checkpoint(
    directory: Path,
    train_state: TrainState,
    runner: RunnerState,
    update: int,
    total_steps: int,
    config: PPOConfig,
) -> None:
    """Save optimizer, policy, environment state, and PRNG for exact continuation."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": train_state.params,
        "opt_state": train_state.opt_state,
        "step": train_state.step,
        "runner": runner,
        "update": jnp.asarray(update, dtype=jnp.int32),
        "total_steps": jnp.asarray(total_steps, dtype=jnp.int32),
    }
    _atomic_write(directory / CHECKPOINT_FILE, serialization.to_bytes(_encode_prng_keys(payload)))
    _atomic_write(directory / POLICY_FILE, serialization.to_bytes(train_state.params))
    metadata = json.dumps(_metadata(config, total_steps, update), indent=2, sort_keys=True)
    _atomic_write(directory / METADATA_FILE, metadata.encode())


def restore_checkpoint(
    path: Path, train_state: TrainState, runner: RunnerState
) -> tuple[TrainState, RunnerState, int, int]:
    """Restore a full checkpoint using live structures as serialization templates."""
    checkpoint_file = path / CHECKPOINT_FILE if path.is_dir() else path
    target = {
        "params": train_state.params,
        "opt_state": train_state.opt_state,
        "step": train_state.step,
        "runner": runner,
        "update": jnp.asarray(0, dtype=jnp.int32),
        "total_steps": jnp.asarray(0, dtype=jnp.int32),
    }
    encoded_target = _encode_prng_keys(target)
    encoded_payload = serialization.from_bytes(encoded_target, checkpoint_file.read_bytes())
    payload = _restore_prng_keys(encoded_payload, target)
    restored = train_state.replace(
        params=payload["params"], opt_state=payload["opt_state"], step=payload["step"]
    )
    return restored, payload["runner"], int(payload["update"]), int(payload["total_steps"])


def load_policy(path: Path, target: Any) -> Any:
    """Load policy parameters from either a checkpoint directory or a policy file."""
    policy_file = path / POLICY_FILE if path.is_dir() else path
    return serialization.from_bytes(target, policy_file.read_bytes())
