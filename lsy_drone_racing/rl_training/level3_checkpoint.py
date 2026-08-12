"""Versioned and exactly resumable Level 3 V2 checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from flax import serialization

from lsy_drone_racing.rl_training.checkpoint import CHECKPOINT_FILE, METADATA_FILE, POLICY_FILE

if TYPE_CHECKING:
    from pathlib import Path

    from lsy_drone_racing.rl_training.level3_algorithm import Level3TrainState
    from lsy_drone_racing.rl_training.level3_config import Level3PPOConfig
    from lsy_drone_racing.rl_training.level3_environment import Level3RunnerState

EMA_POLICY_FILE = "ema_policy.msgpack"
LEVEL3_FORMAT_VERSION = 4
LEVEL3_OBSERVATION_VERSION = 2
LEVEL3_POLICY_FORMAT_VERSIONS = frozenset((2, 3, LEVEL3_FORMAT_VERSION))


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _is_typed_prng_key(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    return dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)


def _encode_prng_keys(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: jax.random.key_data(value) if _is_typed_prng_key(value) else value, tree
    )


def _restore_prng_keys(encoded: Any, template: Any) -> Any:
    return jax.tree.map(
        lambda value, target: (
            jax.random.wrap_key_data(value) if _is_typed_prng_key(target) else value
        ),
        encoded,
        template,
    )


def _metadata(config: Level3PPOConfig, total_steps: int, update: int) -> dict[str, Any]:
    return {
        "format_version": LEVEL3_FORMAT_VERSION,
        "observation_version": LEVEL3_OBSERVATION_VERSION,
        "model": "level3_structured_asymmetric_actor_critic",
        "crazyflow_version": version("crazyflow"),
        "total_steps": int(total_steps),
        "update": int(update),
        "config": asdict(config),
    }


def validate_level3_metadata(
    path: Path, config: Level3PPOConfig | None = None, *, exact_resume: bool = False
) -> dict[str, Any]:
    """Validate policy compatibility and, when requested, exact-resume compatibility."""
    directory = path if path.is_dir() else path.parent
    metadata_path = directory / METADATA_FILE
    if not metadata_path.exists():
        raise ValueError(f"Level 3 V2 checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    format_version = metadata.get("format_version")
    if format_version not in LEVEL3_POLICY_FORMAT_VERSIONS:
        raise ValueError(
            "Incompatible checkpoint format: Level 3 V2 requires format_version 2, 3, or 4; "
            "V1 policies cannot be loaded into the structured model."
        )
    if metadata.get("observation_version") != LEVEL3_OBSERVATION_VERSION:
        raise ValueError("Incompatible checkpoint observation version; expected Level 3 V2")
    if exact_resume and format_version != LEVEL3_FORMAT_VERSION:
        raise ValueError(
            "Exact resume requires Level 3 checkpoint format_version=4 because the runner now "
            "contains episode_start_kind; format_version 2/3 remains policy-only compatible."
        )
    expected_config = json.loads(json.dumps(asdict(config))) if config is not None else None
    if expected_config is not None and metadata.get("config") != expected_config:
        raise ValueError(
            "Exact resume requires the same Level 3 configuration, including rollout shapes, "
            "optimizer schedule, curriculum budget, and random seed."
        )
    return metadata


def save_level3_checkpoint(
    directory: Path,
    train_state: Level3TrainState,
    runner: Level3RunnerState,
    update: int,
    total_steps: int,
    config: Level3PPOConfig,
) -> None:
    """Save raw/EMA policy, optimizer, environment, curriculum, history, and PRNG."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": train_state.params,
        "ema_params": train_state.ema_params,
        "opt_state": train_state.opt_state,
        "step": train_state.step,
        "runner": runner,
        "update": jnp.asarray(update, dtype=jnp.int32),
        "total_steps": jnp.asarray(total_steps, dtype=jnp.int32),
    }
    _atomic_write(directory / CHECKPOINT_FILE, serialization.to_bytes(_encode_prng_keys(payload)))
    _atomic_write(directory / POLICY_FILE, serialization.to_bytes(train_state.params))
    _atomic_write(directory / EMA_POLICY_FILE, serialization.to_bytes(train_state.ema_params))
    metadata = json.dumps(_metadata(config, total_steps, update), indent=2, sort_keys=True)
    _atomic_write(directory / METADATA_FILE, metadata.encode())


def restore_level3_checkpoint(
    path: Path,
    train_state: Level3TrainState,
    runner: Level3RunnerState,
    config: Level3PPOConfig | None = None,
) -> tuple[Level3TrainState, Level3RunnerState, int, int]:
    """Restore an exact V2 continuation using live device structures as templates."""
    validate_level3_metadata(path, config, exact_resume=True)
    checkpoint_file = path / CHECKPOINT_FILE if path.is_dir() else path
    target = {
        "params": train_state.params,
        "ema_params": train_state.ema_params,
        "opt_state": train_state.opt_state,
        "step": train_state.step,
        "runner": runner,
        "update": jnp.asarray(0, dtype=jnp.int32),
        "total_steps": jnp.asarray(0, dtype=jnp.int32),
    }
    encoded = serialization.from_bytes(_encode_prng_keys(target), checkpoint_file.read_bytes())
    payload = _restore_prng_keys(encoded, target)
    state = train_state.replace(
        params=payload["params"],
        ema_params=payload["ema_params"],
        opt_state=payload["opt_state"],
        step=payload["step"],
    )
    return state, payload["runner"], int(payload["update"]), int(payload["total_steps"])


def load_level3_policy(path: Path, target: Any, *, ema: bool = False) -> Any:
    """Load one validated raw or EMA V2 policy."""
    validate_level3_metadata(path)
    filename = EMA_POLICY_FILE if ema else POLICY_FILE
    policy_file = path / filename if path.is_dir() else path
    return serialization.from_bytes(target, policy_file.read_bytes())
