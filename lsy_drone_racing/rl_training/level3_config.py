"""Static configuration for the versioned Level 3 JAX PPO policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class SafetyGeometry:
    """Physical clearances derived from the Crazyflow collision model.

    The cf21B collision sphere radius is 0.086 m. The gate collision boxes begin 0.20 m
    from the gate centre, leaving 0.114 m for the sphere centre. We reserve another
    0.014 m for robustness and therefore target a 0.10 m centre clearance.
    """

    drone_radius: float = 0.086
    gate_inner_half_width: float = 0.20
    gate_safe_half_width: float = 0.10
    gate_safe_half_height: float = 0.10
    gate_risk_buffer: float = 0.04
    obstacle_radius: float = 0.015
    obstacle_safe_radius: float = 0.13
    obstacle_risk_buffer: float = 0.12
    boundary_risk_buffer: float = 0.20
    approach_distance: float = 0.50
    exit_distance: float = 0.35
    gate_prediction_horizon: float = 1.0
    obstacle_prediction_horizon: float = 1.0
    route_blend_distance: float = 0.12


@dataclass(frozen=True)
class Level3RewardCoefficients:
    """Reliability-first Level 3 reward coefficients."""

    route_progress: float = 30.0
    max_route_progress_per_step: float = 0.20
    gate: float = 10.0
    finish: float = 100.0
    safe_gate_speed: float = 0.30
    collision: float = 90.0
    timeout: float = 60.0
    failure_per_gate: float = 30.0
    gate_risk: float = 0.15
    obstacle_risk: float = 0.10
    boundary_risk: float = 0.05
    action_delta: float = 0.0


@dataclass(frozen=True)
class Level3V3RewardCoefficients(Level3RewardCoefficients):
    """Aperture-gated reward that cannot be harvested by parking outside a gate."""

    route_progress: float = 12.0
    gate: float = 30.0
    alignment_temperature: float = 0.04
    near_plane_distance: float = 0.30
    unsafe_near_plane: float = 0.30
    unsafe_stall: float = 0.10
    stall_speed: float = 0.20


@dataclass(frozen=True)
class SegmentResetConfig:
    """Static mixed-start distribution used only by Level 3 training collectors."""

    enabled: bool = False
    start_mode: str = "post_gate"
    start_probabilities: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    exit_distance_min: float = 0.25
    exit_distance_max: float = 0.35
    fallback_exit_distances: tuple[float, float] = (0.20, 0.12)
    lateral_offset: float = 0.04
    vertical_offset: float = 0.04
    speed_min: float = 0.35
    speed_max: float = 0.85
    direction_blend_max: float = 0.35
    velocity_perturbation: float = 0.08
    yaw_perturbation: float = 0.20
    roll_pitch_perturbation: float = 0.10
    angular_velocity: float = 0.50
    entrance_distance_min: float = 0.45
    entrance_distance_max: float = 0.65
    fallback_entrance_distances: tuple[float, float] = (0.35, 0.25)
    entrance_lateral_offset: float = 0.03
    entrance_vertical_offset: float = 0.03
    entrance_speed_min: float = 0.20
    entrance_speed_max: float = 0.45
    entrance_velocity_perturbation: float = 0.04
    route_speed_estimate: float = 0.75
    elapsed_base_seconds: float = 1.0
    elapsed_scale_min: float = 0.8
    elapsed_scale_max: float = 1.2
    environment_frequency: float = 50.0
    minimum_remaining_steps: int = 400
    gate_plane_half_thickness: float = 0.02
    gate_outer_buffer: float = 0.10


@dataclass(frozen=True)
class Level3PPOConfig:
    """Shape-stable Level 3 training configuration."""

    profile: str = "level3_v2"
    observation_version: int = 2
    config_file: str = "level3.toml"
    seed: int = 42
    device: str = "gpu"
    num_envs: int = 1024
    num_steps: int = 128
    total_timesteps: int = 500_000_000
    curriculum_timesteps: int = 500_000_000
    num_minibatches: int = 8
    update_epochs: int = 4
    learning_rate_start: float = 5e-5
    learning_rate_end: float = 1e-5
    gamma: float = 0.995
    gae_lambda: float = 0.97
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    entropy_coef: float = 0.003
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    target_kl: float = 0.01
    ema_decay: float = 0.99
    entity_dim: int = 128
    attention_heads: int = 4
    fusion_dim: int = 256
    history_length: int = 4
    max_episode_steps: int = 1500
    checkpoint_interval: int = 10_000_000
    validation_interval: int = 10_000_000
    validation_episodes: int = 500
    final_evaluation_episodes: int = 5_000
    success_target: float = 0.90
    curriculum: bool = True
    geometry: SafetyGeometry = SafetyGeometry()
    reward: Level3RewardCoefficients = Level3RewardCoefficients()
    segment_reset: SegmentResetConfig = SegmentResetConfig()

    @property
    def batch_size(self) -> int:
        """Number of transitions in a rollout."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Number of transitions in one PPO minibatch."""
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        """Number of complete updates that fit in the configured budget."""
        return self.total_timesteps // self.batch_size

    @property
    def maximum_optimizer_steps(self) -> int:
        """Upper bound used by the learning-rate schedule."""
        return self.num_updates * self.update_epochs * self.num_minibatches

    def validate(self) -> None:
        """Validate static shape and numerical invariants."""
        if self.observation_version != 2:
            raise ValueError("Level 3 PPO requires observation_version=2")
        if self.num_envs <= 0 or self.num_steps <= 0:
            raise ValueError("num_envs and num_steps must be positive")
        if self.curriculum_timesteps <= 0:
            raise ValueError("curriculum_timesteps must be positive")
        if self.batch_size % self.num_minibatches:
            raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
        if self.history_length != 4:
            raise ValueError("Level 3 V2 has a fixed four-step history")
        if self.entity_dim % self.attention_heads:
            raise ValueError("entity_dim must be divisible by attention_heads")
        if not 0.0 < self.target_kl:
            raise ValueError("target_kl must be positive")
        probabilities = self.segment_reset.start_probabilities
        if len(probabilities) != 4 or any(value < 0.0 for value in probabilities):
            raise ValueError("segment start_probabilities must contain four non-negative values")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("segment start_probabilities must sum to one")
        if self.segment_reset.enabled:
            if self.segment_reset.start_mode not in {"post_gate", "pre_gate"}:
                raise ValueError("segment start_mode must be 'post_gate' or 'pre_gate'")
            if self.max_episode_steps <= self.segment_reset.minimum_remaining_steps:
                raise ValueError("segment reset requires more than minimum_remaining_steps")
            if not 0.0 < self.segment_reset.route_speed_estimate:
                raise ValueError("segment route_speed_estimate must be positive")
            if not 0.0 < self.segment_reset.environment_frequency:
                raise ValueError("segment environment_frequency must be positive")
            if not 0.0 <= self.segment_reset.direction_blend_max <= 1.0:
                raise ValueError("segment direction_blend_max must be in [0, 1]")

    def with_overrides(self, **overrides: Any) -> Level3PPOConfig:
        """Return a validated configuration copy."""
        values = {key: value for key, value in overrides.items() if value is not None}
        config = replace(self, **values)
        config.validate()
        return config


LEVEL3_PROFILE_NAMES = (
    "level3_v2",
    "level3_v2_segment",
    "level3_v3_reward",
    "level3_v3_pregate",
    "level3_v3_crossing_seed",
)


def get_level3_profile(name: str = "level3_v2") -> Level3PPOConfig:
    """Return the frozen reliability-first Level 3 profile."""
    if name == "level3_v2":
        config = Level3PPOConfig()
    elif name == "level3_v2_segment":
        config = Level3PPOConfig(
            profile="level3_v2_segment", segment_reset=SegmentResetConfig(enabled=True)
        )
    elif name == "level3_v3_reward":
        config = Level3PPOConfig(
            profile="level3_v3_reward",
            reward=Level3V3RewardCoefficients(),
            segment_reset=SegmentResetConfig(enabled=True),
        )
    elif name == "level3_v3_pregate":
        config = Level3PPOConfig(
            profile="level3_v3_pregate",
            reward=Level3V3RewardCoefficients(),
            segment_reset=SegmentResetConfig(enabled=True, start_mode="pre_gate"),
        )
    elif name == "level3_v3_crossing_seed":
        config = Level3PPOConfig(
            profile="level3_v3_crossing_seed",
            reward=Level3V3RewardCoefficients(),
            segment_reset=SegmentResetConfig(
                enabled=True,
                start_mode="pre_gate",
                entrance_distance_min=0.12,
                entrance_distance_max=0.18,
                fallback_entrance_distances=(0.14, 0.12),
                entrance_lateral_offset=0.015,
                entrance_vertical_offset=0.015,
                entrance_speed_min=0.60,
                entrance_speed_max=0.90,
                entrance_velocity_perturbation=0.02,
            ),
        )
    else:
        raise ValueError(f"Unknown Level 3 profile; expected one of {LEVEL3_PROFILE_NAMES}")
    config.validate()
    return config
