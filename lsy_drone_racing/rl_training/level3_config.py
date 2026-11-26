"""Static configuration for the versioned Level 3 JAX PPO policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

YAW_ACTION_INDEX = 2


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
class Level3V4RewardCoefficients(Level3RewardCoefficients):
    """Commitment reward: hovering costs time, and passing a gate is never net-negative."""

    gate: float = 100.0
    safe_gate_speed: float = 0.0
    failure_per_gate: float = 0.0
    alignment_temperature: float = 0.04
    time: float = 0.03
    gate_pass_speed: float = 5.0
    timeout_terminal: bool = True


@dataclass(frozen=True)
class Level3V5RewardCoefficients(Level3V4RewardCoefficients):
    """V4 rebalanced toward completion: costlier crashes, larger finish, no speed bonus."""

    finish: float = 300.0
    collision: float = 150.0
    gate_risk: float = 0.40
    obstacle_risk: float = 0.30
    gate_pass_speed: float = 0.0


@dataclass(frozen=True)
class Level3V6RewardCoefficients(Level3V5RewardCoefficients):
    """V5 with the summed reward scaled so value targets stay O(1) for PPO."""

    reward_scale: float = 0.02


@dataclass(frozen=True)
class Level3V7RewardCoefficients(Level3V6RewardCoefficients):
    """V6 trading lap time for margin: costlier crashes, obstacle risk, and a soft speed limit."""

    collision: float = 250.0
    obstacle_risk: float = 1.0
    time: float = 0.01
    overspeed: float = 2.0
    speed_limit: float = 1.6


@dataclass(frozen=True)
class Level3V8RewardCoefficients(Level3V7RewardCoefficients):
    """V7 plus an early horizontal keep-away field around obstacle poles."""

    obstacle_clearance: float = 2.0
    obstacle_clearance_radius: float = 0.35


@dataclass(frozen=True)
class Level3V9RewardCoefficients(Level3V8RewardCoefficients):
    """V8 with a binding speed limit and a pole keep-away field that outweighs progress."""

    overspeed: float = 8.0
    speed_limit: float = 1.5
    obstacle_clearance: float = 10.0
    obstacle_clearance_radius: float = 0.40


@dataclass(frozen=True)
class Level3V10RewardCoefficients(Level3V9RewardCoefficients):
    """V9 plus a margin around unseen nominal poles that covers their worst-case shift."""

    nominal_clearance: float = 5.0
    nominal_clearance_radius: float = 0.45


@dataclass(frozen=True)
class Level3V13RewardCoefficients(Level3V10RewardCoefficients):
    """V10 with the soft speed limit raised to 2.2 m/s and a larger time penalty."""

    speed_limit: float = 2.2
    time: float = 0.03


@dataclass(frozen=True)
class Level3V14RewardCoefficients(Level3V13RewardCoefficients):
    """V13 with the soft speed limit raised to 3.0 m/s and a larger time penalty."""

    speed_limit: float = 3.0
    time: float = 0.2


@dataclass(frozen=True)
class Level3V15RewardCoefficients(Level3V14RewardCoefficients):
    """V14 with the soft speed limit raised to 4.0 m/s."""

    speed_limit: float = 4.0
    time: float = 0.3


@dataclass(frozen=True)
class Level3V16RewardCoefficients(Level3V15RewardCoefficients):
    """V15 with the soft speed limit raised to 5.0 m/s and a lap-time-dominant time penalty."""

    speed_limit: float = 5.0
    time: float = 0.8


@dataclass(frozen=True)
class Level3V17RewardCoefficients(Level3V16RewardCoefficients):
    """V16 with a time penalty large enough to trade a small success loss for lap time."""

    speed_limit: float = 6.0
    time: float = 2.0


@dataclass(frozen=True)
class Level3V18RewardCoefficients(Level3V17RewardCoefficients):
    """V17 with a smaller time penalty to recover reliability at sprint speed."""

    time: float = 1.0


@dataclass(frozen=True)
class Level3V19RewardCoefficients(Level3V18RewardCoefficients):
    """V18 with weaker gate-centering risk and a tighter pole keep-away for faster lines."""

    gate_risk: float = 0.20
    obstacle_clearance: float = 5.0
    obstacle_clearance_radius: float = 0.30


@dataclass(frozen=True)
class LevelReplayConfig:
    """Training-only prioritized replay of failed track layouts."""

    enabled: bool = False
    buffer_size: int = 4096
    replay_probability: float = 0.5
    temperature: float = 0.5
    staleness_coef: float = 0.1
    score_ema: float = 0.5


@dataclass(frozen=True)
class SegmentResetConfig:
    """Static mixed-start distribution used only by Level 3 training collectors."""

    enabled: bool = False
    start_mode: str = "post_gate"
    start_probabilities: tuple[float, ...] = ()
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
    mixed_progress_knots: tuple[float, float, float] = (0.10, 0.40, 0.80)
    mixed_start_probabilities_0: tuple[float, float, float] = (0.10, 0.20, 0.70)
    mixed_start_probabilities_1: tuple[float, float, float] = (0.20, 0.25, 0.55)
    mixed_start_probabilities_2: tuple[float, float, float] = (0.45, 0.30, 0.25)
    mixed_start_probabilities_3: tuple[float, float, float] = (0.70, 0.15, 0.15)
    mixed_geometry_progress_end: float = 0.40
    mixed_entrance_distance_min: float = 0.60
    mixed_entrance_distance_max: float = 1.00
    mixed_entrance_offset: float = 0.12
    mixed_entrance_speed_min: float = 0.15
    mixed_entrance_speed_max: float = 0.50
    mixed_entrance_velocity_perturbation: float = 0.08
    corridor_geometry_progress_end: float = 0.10
    corridor_fraction_min_start: float = 0.65
    corridor_fraction_min_end: float = 0.0
    corridor_source_exit_distance: float = 0.25
    corridor_entrance_distance_min_start: float = 0.12
    corridor_entrance_distance_max_start: float = 0.18
    corridor_entrance_distance_min_end: float = 0.30
    corridor_entrance_distance_max_end: float = 0.45
    corridor_min_entrance_depth: float = 0.10
    corridor_offset_start: float = 0.015
    corridor_offset_end: float = 0.08
    corridor_speed_min_start: float = 0.60
    corridor_speed_max_start: float = 0.90
    corridor_speed_min_end: float = 0.25
    corridor_speed_max_end: float = 0.70
    corridor_velocity_perturbation_start: float = 0.02
    corridor_velocity_perturbation_end: float = 0.08
    corridor_yaw_perturbation_start: float = 0.10
    corridor_yaw_perturbation_end: float = 0.35
    corridor_roll_pitch_start: float = 0.05
    corridor_roll_pitch_end: float = 0.12
    corridor_angular_velocity_start: float = 0.20
    corridor_angular_velocity_end: float = 0.60


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
    level_replay: LevelReplayConfig = LevelReplayConfig()
    sensor_range: float | None = None
    actor_input_mask: tuple[str, ...] = ()
    allow_policy_initialization: bool = False
    lock_yaw: bool = False
    yaw_anneal_timesteps: int = 0

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
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        if self.yaw_anneal_timesteps < 0:
            raise ValueError("yaw_anneal_timesteps must be non-negative")
        if self.yaw_anneal_timesteps and not self.lock_yaw:
            raise ValueError("yaw_anneal_timesteps requires lock_yaw")
        if self.yaw_anneal_timesteps > self.curriculum_timesteps:
            raise ValueError("yaw_anneal_timesteps must not exceed curriculum_timesteps")
        if self.entity_dim % self.attention_heads:
            raise ValueError("entity_dim must be divisible by attention_heads")
        if not 0.0 < self.target_kl:
            raise ValueError("target_kl must be positive")
        probabilities = self.segment_reset.start_probabilities
        if any(value < 0.0 for value in probabilities):
            raise ValueError("segment start_probabilities must be non-negative")
        if probabilities and abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("segment start_probabilities must sum to one")
        if self.segment_reset.enabled:
            if self.segment_reset.start_mode not in {
                "post_gate",
                "pre_gate",
                "mixed_curriculum",
                "route_corridor",
            }:
                raise ValueError(
                    "segment start_mode must be 'post_gate', 'pre_gate', 'mixed_curriculum', "
                    "or 'route_corridor'"
                )
            if self.max_episode_steps <= self.segment_reset.minimum_remaining_steps:
                raise ValueError("segment reset requires more than minimum_remaining_steps")
            if not 0.0 < self.segment_reset.route_speed_estimate:
                raise ValueError("segment route_speed_estimate must be positive")
            if not 0.0 < self.segment_reset.environment_frequency:
                raise ValueError("segment environment_frequency must be positive")
            if not 0.0 <= self.segment_reset.direction_blend_max <= 1.0:
                raise ValueError("segment direction_blend_max must be in [0, 1]")
            knots = self.segment_reset.mixed_progress_knots
            if not 0.0 < knots[0] < knots[1] < knots[2] <= 1.0:
                raise ValueError("mixed_progress_knots must be strictly increasing in (0, 1]")
            probability_rows = (
                self.segment_reset.mixed_start_probabilities_0,
                self.segment_reset.mixed_start_probabilities_1,
                self.segment_reset.mixed_start_probabilities_2,
                self.segment_reset.mixed_start_probabilities_3,
            )
            for row in probability_rows:
                if len(row) != 3 or any(value < 0.0 for value in row):
                    raise ValueError(
                        "mixed start probabilities must have three non-negative values"
                    )
                if abs(sum(row) - 1.0) > 1e-6:
                    raise ValueError("mixed start probabilities must sum to one")
            if not 0.0 < self.segment_reset.mixed_geometry_progress_end <= 1.0:
                raise ValueError("mixed_geometry_progress_end must be in (0, 1]")
            if not 0.0 < self.segment_reset.corridor_geometry_progress_end <= 1.0:
                raise ValueError("corridor_geometry_progress_end must be in (0, 1]")
            if not (
                0.0
                <= self.segment_reset.corridor_fraction_min_end
                <= self.segment_reset.corridor_fraction_min_start
                <= 1.0
            ):
                raise ValueError("corridor fraction bounds must satisfy 0 <= end <= start <= 1")

    def with_overrides(self, **overrides: Any) -> Level3PPOConfig:
        """Return a validated configuration copy."""
        values = {key: value for key, value in overrides.items() if value is not None}
        config = replace(self, **values)
        config.validate()
        return config


LEVEL3_OBSERVATION_ABLATIONS = {
    "level3_abl_no_history": ("history",),
    "level3_abl_no_direct_geometry": ("direct_geometry",),
    "level3_abl_no_obstacles": ("obstacles",),
    "level3_abl_near_gates": ("far_gates",),
    "level3_abl_no_visited": ("visited",),
}
LEVEL3_SCRATCH_REWARD_ABLATIONS = {
    "level3_abl_gate10": {"gate": 10.0},
    "level3_abl_failure_per_gate": {"failure_per_gate": 30.0},
    "level3_abl_timeout_bootstrap": {"timeout_terminal": False},
    "level3_abl_no_time": {"time": 0.0},
    "level3_abl_v2_reward": {
        "gate": 10.0,
        "safe_gate_speed": 0.30,
        "failure_per_gate": 30.0,
        "alignment_temperature": None,
        "time": 0.0,
        "gate_pass_speed": 0.0,
        "timeout_terminal": False,
    },
}
LEVEL3_FINETUNE_REWARD_ABLATIONS = {
    "level3_abl_ft_base": {},
    "level3_abl_ft_no_route_progress": {"route_progress": 0.0},
    "level3_abl_ft_no_gate_risk": {"gate_risk": 0.0},
    "level3_abl_ft_no_obstacle_risk": {"obstacle_risk": 0.0},
    "level3_abl_ft_no_obstacle_clearance": {"obstacle_clearance": 0.0, "nominal_clearance": 0.0},
    "level3_abl_ft_no_overspeed": {"overspeed": 0.0},
    "level3_abl_ft_no_time": {"time": 0.0},
}

LEVEL3_PROFILE_NAMES = (
    "level3_v2",
    "level3_v2_segment",
    "level3_v2_explore",
    "level3_v4_commit",
    "level3_v5_safe",
    "level3_v6_scaled",
    "level3_v7_margin",
    "level3_v8_clearance",
    "level3_v9_slow",
    "level3_v10_full",
    "level3_v11_explore",
    "level3_v12_refine",
    "level3_v13_fast",
    "level3_v14_faster",
    "level3_v15_race",
    "level3_v16_lap",
    "level3_v17_sprint",
    "level3_v18_refine",
    "level3_v18_plr",
    "level3_v19_tight",
    "level3_v20_memory",
    "level3_v21_fullinfo",
    "level3_v22_yaw_lock",
    "level3_v22_yaw_lock_hard",
    "level3_abl_base",
    "level3_v3_reward",
    "level3_v3_pregate",
    "level3_v3_crossing_seed",
    "level3_v3_mixed_curriculum",
    "level3_v3_route_corridor",
    *LEVEL3_OBSERVATION_ABLATIONS,
    *LEVEL3_SCRATCH_REWARD_ABLATIONS,
    *LEVEL3_FINETUNE_REWARD_ABLATIONS,
)


def get_level3_profile(name: str = "level3_v2") -> Level3PPOConfig:
    """Return the frozen reliability-first Level 3 profile."""
    if name == "level3_v2":
        config = Level3PPOConfig()
    elif name == "level3_v2_segment":
        config = Level3PPOConfig(
            profile="level3_v2_segment", segment_reset=SegmentResetConfig(enabled=True)
        )
    elif name == "level3_v2_explore":
        config = Level3PPOConfig(
            profile="level3_v2_explore",
            learning_rate_start=3e-4,
            learning_rate_end=3e-5,
            entropy_coef=0.02,
            target_kl=0.03,
            reward=replace(Level3RewardCoefficients(), gate=100.0),
        )
    elif name == "level3_v4_commit":
        config = Level3PPOConfig(
            profile="level3_v4_commit",
            learning_rate_start=3e-4,
            learning_rate_end=3e-5,
            target_kl=0.03,
            reward=Level3V4RewardCoefficients(),
        )
    elif name == "level3_v5_safe":
        config = Level3PPOConfig(
            profile="level3_v5_safe",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            target_kl=0.02,
            reward=Level3V5RewardCoefficients(),
        )
    elif name == "level3_v6_scaled":
        config = Level3PPOConfig(
            profile="level3_v6_scaled",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            target_kl=0.02,
            max_grad_norm=2.0,
            reward=Level3V6RewardCoefficients(),
        )
    elif name == "level3_v7_margin":
        config = Level3PPOConfig(
            profile="level3_v7_margin",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            target_kl=0.02,
            max_grad_norm=2.0,
            reward=Level3V7RewardCoefficients(),
        )
    elif name == "level3_v8_clearance":
        config = Level3PPOConfig(
            profile="level3_v8_clearance",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            target_kl=0.02,
            max_grad_norm=2.0,
            reward=Level3V8RewardCoefficients(),
        )
    elif name == "level3_v9_slow":
        config = Level3PPOConfig(
            profile="level3_v9_slow",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            target_kl=0.02,
            max_grad_norm=2.0,
            reward=Level3V9RewardCoefficients(),
        )
    elif name == "level3_v10_full":
        config = Level3PPOConfig(
            profile="level3_v10_full",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=Level3V10RewardCoefficients(),
        )
    elif name == "level3_v11_explore":
        config = Level3PPOConfig(
            profile="level3_v11_explore",
            learning_rate_start=2e-4,
            learning_rate_end=3e-5,
            entropy_coef=0.01,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=Level3V10RewardCoefficients(),
        )
    elif name == "level3_v12_refine":
        config = Level3PPOConfig(
            profile="level3_v12_refine",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            entropy_coef=0.001,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=Level3V10RewardCoefficients(),
        )
    elif name == "level3_v13_fast":
        config = Level3PPOConfig(
            profile="level3_v13_fast",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=Level3V13RewardCoefficients(),
        )
    elif name == "level3_v14_faster":
        config = Level3PPOConfig(
            profile="level3_v14_faster",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=Level3V14RewardCoefficients(),
        )
    elif name == "level3_v15_race":
        config = Level3PPOConfig(
            profile="level3_v15_race",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=Level3V15RewardCoefficients(),
        )
    elif name == "level3_v16_lap":
        config = Level3PPOConfig(
            profile="level3_v16_lap",
            learning_rate_start=1e-4,
            learning_rate_end=2e-5,
            gamma=0.997,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=Level3V16RewardCoefficients(),
        )
    elif name == "level3_v17_sprint":
        config = Level3PPOConfig(
            profile="level3_v17_sprint",
            learning_rate_start=2e-4,
            learning_rate_end=2e-5,
            entropy_coef=0.001,
            gamma=0.998,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=Level3V17RewardCoefficients(),
        )
    elif name in ("level3_v18_refine", "level3_v18_plr"):
        config = Level3PPOConfig(
            profile=name,
            learning_rate_start=5e-5,
            learning_rate_end=1e-5,
            entropy_coef=0.001,
            gamma=0.998,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=Level3V18RewardCoefficients(),
            level_replay=LevelReplayConfig(enabled=name == "level3_v18_plr"),
        )
    elif name in (
        "level3_v19_tight",
        "level3_v20_memory",
        "level3_v21_fullinfo",
        "level3_v22_yaw_lock",
        "level3_v22_yaw_lock_hard",
    ):
        config = Level3PPOConfig(
            profile=name,
            history_length=4 if name == "level3_v19_tight" else 25,
            sensor_range=10.0 if name == "level3_v21_fullinfo" else None,
            lock_yaw=name.startswith("level3_v22_yaw_lock"),
            yaw_anneal_timesteps=120_000_000 if name == "level3_v22_yaw_lock" else 0,
            learning_rate_start=5e-5,
            learning_rate_end=1e-5,
            entropy_coef=0.001,
            gamma=0.998,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=Level3V19RewardCoefficients(),
        )
    elif (
        name == "level3_abl_base"
        or name in LEVEL3_OBSERVATION_ABLATIONS
        or name in LEVEL3_SCRATCH_REWARD_ABLATIONS
    ):
        config = Level3PPOConfig(
            profile=name,
            learning_rate_start=3e-4,
            learning_rate_end=3e-5,
            target_kl=0.03,
            reward=replace(
                Level3V4RewardCoefficients(), **LEVEL3_SCRATCH_REWARD_ABLATIONS.get(name, {})
            ),
            actor_input_mask=LEVEL3_OBSERVATION_ABLATIONS.get(name, ()),
        )
    elif name in LEVEL3_FINETUNE_REWARD_ABLATIONS:
        config = Level3PPOConfig(
            profile=name,
            history_length=25,
            learning_rate_start=5e-5,
            learning_rate_end=1e-5,
            entropy_coef=0.001,
            gamma=0.998,
            target_kl=0.02,
            max_grad_norm=2.0,
            curriculum=False,
            reward=replace(Level3V19RewardCoefficients(), **LEVEL3_FINETUNE_REWARD_ABLATIONS[name]),
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
    elif name == "level3_v3_mixed_curriculum":
        config = Level3PPOConfig(
            profile="level3_v3_mixed_curriculum",
            reward=Level3V3RewardCoefficients(),
            segment_reset=SegmentResetConfig(
                enabled=True,
                start_mode="mixed_curriculum",
                entrance_distance_min=0.12,
                entrance_distance_max=0.18,
                fallback_entrance_distances=(0.14, 0.12),
                entrance_lateral_offset=0.015,
                entrance_vertical_offset=0.015,
                entrance_speed_min=0.60,
                entrance_speed_max=0.90,
                entrance_velocity_perturbation=0.02,
            ),
            allow_policy_initialization=True,
        )
    elif name == "level3_v3_route_corridor":
        config = Level3PPOConfig(
            profile="level3_v3_route_corridor",
            reward=Level3V3RewardCoefficients(),
            segment_reset=SegmentResetConfig(
                enabled=True,
                start_mode="route_corridor",
                entrance_distance_min=0.12,
                entrance_distance_max=0.18,
                fallback_entrance_distances=(0.14, 0.12),
                entrance_lateral_offset=0.015,
                entrance_vertical_offset=0.015,
                entrance_speed_min=0.60,
                entrance_speed_max=0.90,
                entrance_velocity_perturbation=0.02,
                mixed_start_probabilities_0=(0.10, 0.60, 0.30),
                mixed_start_probabilities_1=(0.20, 0.65, 0.15),
                mixed_start_probabilities_2=(0.45, 0.50, 0.05),
                mixed_start_probabilities_3=(0.70, 0.30, 0.0),
            ),
            allow_policy_initialization=True,
        )
    else:
        raise ValueError(f"Unknown Level 3 profile; expected one of {LEVEL3_PROFILE_NAMES}")
    config.validate()
    return config
