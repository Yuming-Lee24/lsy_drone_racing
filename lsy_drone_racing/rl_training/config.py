"""Configuration profiles for the end-to-end JAX PPO agent."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class RewardCoefficients:
    """Coefficients used by the racing reward."""

    progress: float = 60.0
    gate: float = 20.0
    finish: float = 100.0
    time: float = 0.0
    align: float = 0.5
    collision: float = 20.0
    smooth: float = 0.0
    spin: float = 0.0
    angle: float = 0.0
    ground: float = 0.1
    max_progress_speed: float = 1.5


@dataclass(frozen=True)
class PPOConfig:
    """Static PPO and environment configuration.

    Shape-changing fields are intentionally immutable. A training function is compiled once for a
    profile, so changing one of these fields requires a new compilation.
    """

    profile: str = "fast"
    config_file: str = "level2.toml"
    seed: int = 42
    device: str = "gpu"
    num_envs: int = 1024
    num_steps: int = 96
    total_timesteps: int = 500_000_000
    num_minibatches: int = 8
    update_epochs: int = 12
    learning_rate: float = 1e-4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    entropy_coef: float = 0.005
    value_coef: float = 0.6
    max_grad_norm: float = 0.4
    normalize_advantage: bool = True
    hidden_dim: int = 256
    history_length: int = 2
    curriculum_stage: int = 2
    max_episode_steps: int = 1500
    checkpoint_interval: int = 10_000_000
    validation_interval: int = 10_000_000
    validation_episodes: int = 100
    final_evaluation_episodes: int = 500
    success_target: float = 0.65
    mean_time_target: float = 3.4
    reward: RewardCoefficients = RewardCoefficients()

    @property
    def batch_size(self) -> int:
        """Number of samples in one rollout."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Number of samples in one optimizer minibatch."""
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        """Maximum number of PPO updates."""
        return self.total_timesteps // self.batch_size

    def validate(self) -> None:
        """Validate shape and range invariants before compiling."""
        if self.num_envs <= 0 or self.num_steps <= 0:
            raise ValueError("num_envs and num_steps must be positive")
        if self.batch_size % self.num_minibatches:
            raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
        if self.history_length < 0:
            raise ValueError("history_length must be non-negative")
        if self.curriculum_stage not in (0, 1, 2):
            raise ValueError("curriculum_stage must be 0, 1, or 2")

    def with_overrides(self, **overrides: Any) -> PPOConfig:
        """Return a validated copy with CLI overrides applied."""
        values = {key: value for key, value in overrides.items() if value is not None}
        config = replace(self, **values)
        config.validate()
        return config


_FAST = PPOConfig()
_SAFE_FAST = replace(
    _FAST,
    profile="safe_fast",
    success_target=0.79,
    mean_time_target=4.0,
    reward=replace(_FAST.reward, angle=1.0),
)


def get_profile(name: str) -> PPOConfig:
    """Return a named immutable training profile."""
    profiles = {"fast": _FAST, "safe_fast": _SAFE_FAST}
    try:
        config = profiles[name]
    except KeyError as error:
        raise ValueError(f"Unknown PPO profile {name!r}; choose from {tuple(profiles)}") from error
    config.validate()
    return config
