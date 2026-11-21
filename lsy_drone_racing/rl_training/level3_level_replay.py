"""Training-only prioritized replay of failed Level 3 track layouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct

from lsy_drone_racing.envs.race_core import _reset_env_data

if TYPE_CHECKING:
    from jax import Array

    from lsy_drone_racing.envs.race_core import EnvData
    from lsy_drone_racing.rl_training.level3_config import LevelReplayConfig

TRACK_FIELDS = (
    "gates_pos",
    "gates_quat",
    "obstacles_pos",
    "nominal_gates_pos",
    "nominal_gates_quat",
    "nominal_obstacles_pos",
)


@struct.dataclass
class LevelReplayState:
    """Fixed-size on-device level buffer and the buffer slot of each running episode."""

    tracks: dict[str, Array]
    score: Array
    last_sampled: Array
    valid: Array
    clock: Array
    episode_slot: Array
    rng: Array


def initialize_level_replay(
    data: EnvData, config: LevelReplayConfig, seed: int
) -> LevelReplayState:
    """Create an empty buffer shaped like one world's track layout."""
    size = config.buffer_size if config.enabled else 1
    num_envs = data.gates_pos.shape[0]
    return LevelReplayState(
        tracks={
            name: jnp.zeros((size, *getattr(data, name).shape[1:]), getattr(data, name).dtype)
            for name in TRACK_FIELDS
        },
        score=jnp.zeros(size, dtype=jnp.float32),
        last_sampled=jnp.zeros(size, dtype=jnp.int32),
        valid=jnp.zeros(size, dtype=bool),
        clock=jnp.asarray(0, dtype=jnp.int32),
        episode_slot=-jnp.ones(num_envs, dtype=jnp.int32),
        rng=jax.random.key(seed + 7_919),
    )


def update_level_replay(
    state: LevelReplayState,
    terminal_data: EnvData,
    done: Array,
    final_progress: Array,
    sequence_length: int,
    config: LevelReplayConfig,
) -> LevelReplayState:
    """Score finished episodes and insert fresh layouts over the lowest-scoring slots."""
    size = state.score.shape[0]
    new_score = 1.0 - final_progress.astype(jnp.float32) / sequence_length
    replayed = done & (state.episode_slot >= 0)
    replay_index = jnp.where(replayed, state.episode_slot, size)
    old_score = state.score[jnp.clip(state.episode_slot, 0, size - 1)]
    blended = (1.0 - config.score_ema) * old_score + config.score_ema * new_score
    score = state.score.at[replay_index].set(blended, mode="drop")

    fresh = done & (state.episode_slot < 0)
    eviction_key = jnp.where(state.valid, score, -1.0)
    candidates = jnp.argsort(eviction_key)[: done.shape[0]]
    rank = jnp.cumsum(fresh) - 1
    target = candidates[jnp.clip(rank, 0, candidates.shape[0] - 1)]
    insert = fresh & (new_score > eviction_key[target])
    insert_index = jnp.where(insert, target, size)
    tracks = {
        name: buffer.at[insert_index].set(getattr(terminal_data, name), mode="drop")
        for name, buffer in state.tracks.items()
    }
    return state.replace(
        tracks=tracks,
        score=score.at[insert_index].set(new_score, mode="drop"),
        last_sampled=state.last_sampled.at[insert_index].set(state.clock, mode="drop"),
        valid=state.valid.at[insert_index].set(True, mode="drop"),
    )


def sample_level_replay(
    state: LevelReplayState, data: EnvData, reset_mask: Array, config: LevelReplayConfig
) -> tuple[EnvData, LevelReplayState]:
    """Replace a fraction of freshly reset layouts with prioritized buffered layouts."""
    rng, choice_key, slot_key = jax.random.split(state.rng, 3)
    num_envs = reset_mask.shape[0]
    priority = jnp.where(state.valid, (state.score + 1e-3) ** (1.0 / config.temperature), 0.0)
    staleness = jnp.where(state.valid, (state.clock - state.last_sampled).astype(jnp.float32), 0.0)
    probability = (1.0 - config.staleness_coef) * priority / jnp.maximum(
        priority.sum(), 1e-12
    ) + config.staleness_coef * staleness / jnp.maximum(staleness.sum(), 1e-12)
    slot = jax.random.categorical(
        slot_key, jnp.log(jnp.maximum(probability, 1e-30)), shape=(num_envs,)
    ).astype(jnp.int32)
    replay = (
        reset_mask
        & state.valid.any()
        & (jax.random.uniform(choice_key, (num_envs,)) < config.replay_probability)
    )
    track_mask = replay[:, None, None]
    data = data.replace(
        **{
            name: jnp.where(track_mask, buffer[slot], getattr(data, name))
            for name, buffer in state.tracks.items()
        }
    )
    data = _reset_env_data(data, replay)
    return data, state.replace(
        last_sampled=state.last_sampled.at[jnp.where(replay, slot, state.score.shape[0])].set(
            state.clock, mode="drop"
        ),
        clock=state.clock + 1,
        episode_slot=jnp.where(reset_mask, jnp.where(replay, slot, -1), state.episode_slot),
        rng=rng,
    )
