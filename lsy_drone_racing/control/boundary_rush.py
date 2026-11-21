"""Test controller that deliberately flies out of the safety limits at full speed.

With `mode = "thrust_test"` it instead measures the real maximum thrust: a short vertical burst at
full PWM from hover, after which the drone hovers again and the controller finishes. The vertical
acceleration during the burst (see `scripts/analyze_flight.py`) gives T_max = m (a + g).

Use it to test how the drone recovers once it leaves the safety box (`env.track.safety_limits`):
leaving the box terminates the episode, and `RealRaceCoreEnv.close` then runs the RL brake (if
`deploy.brake_rl = true`) followed by `Crazyflie.return_to_start`. `scripts/sim_boundary_rush.py`
compares both recoveries in simulation from the same state.

Phases:
    takeoff  Minimum-jerk climb from the start position to `height` (default 1 m).
    hover    Hold the takeoff point.
    rush     Saturate the thrust along `direction` (the drone tilts as far as the thrust budget
             and MAX_TILT allow, while holding the height and the line). With `max_speed`, the
             acceleration stops once that speed is reached.
    burst    (thrust_test) Level attitude at full thrust for `burst_duration`, then hover again
             for BURST_HOVER seconds and finish (return_to_start lands the drone).

Optional config section (all keys optional):

    [boundary_rush]
    height = 1          # Takeoff height in m
    direction = [-1, 0]   # Horizontal rush direction (normalized)
    max_speed = 0.0       # Speed cap in m/s, 0 means no cap (accelerate until leaving the box)
    thrust_frac = 0.9     # Fraction of the maximum collective thrust used during the rush
    mode = "rush"         # "rush" or "thrust_test"
    burst_duration = 0.15 # Full-thrust burst in s (thrust_test). With T/W 2.8 the drone climbs
                          # about 0.6 m, so keep 1 m of headroom above `height`

Usage:
    python scripts/deploy.py --controller boundary_rush.py --record
    python scripts/sim_boundary_rush.py
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.dynamics import load_params as load_dynamics_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

G = 9.81
MAX_TILT = np.deg2rad(60.0)
TAKEOFF_DURATION = 3.0
HOVER_DURATION = 2.0
RUSH_TIMEOUT = 3.0
RUSH_ACC = 100.0  # Requested along-track acceleration; always saturated by the thrust budget
BURST_HOVER = 3.0  # Hover time after the thrust-test burst in s


def acc_to_attitude(
    a_des: NDArray, yaw: float, mass: float, thrust_max: float
) -> tuple[NDArray, float]:
    """Map a desired world acceleration to [roll, pitch, yaw] in rad and the collective thrust in N.

    The vertical force has priority. The horizontal force uses the remaining thrust budget and is
    limited to MAX_TILT.
    """
    f = mass * (np.asarray(a_des, dtype=np.float64) + np.array([0.0, 0.0, G]))
    fz = float(np.clip(f[2], 0.2 * mass * G, thrust_max))
    fxy_max = min(np.sqrt(max(thrust_max**2 - fz**2, 0.0)), fz * np.tan(MAX_TILT))
    fxy = f[:2] * min(1.0, fxy_max / max(float(np.linalg.norm(f[:2])), 1e-9))
    f = np.array([fxy[0], fxy[1], fz])
    z_axis = f / np.linalg.norm(f)
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_axis = np.cross(z_axis, x_c)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    rpy = R.from_matrix(np.column_stack([x_axis, y_axis, z_axis])).as_euler("xyz")
    return rpy, float(np.linalg.norm(f))


def min_jerk(p0: NDArray, p1: NDArray, t: float, T: float) -> tuple[NDArray, NDArray, NDArray]:
    """Position, velocity and acceleration of a rest-to-rest minimum-jerk segment."""
    s = np.clip(t / T, 0.0, 1.0)
    d = p1 - p0
    p = p0 + d * (10 * s**3 - 15 * s**4 + 6 * s**5)
    v = d * (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    a = d * (60 * s - 180 * s**2 + 120 * s**3) / T**2
    if t >= T:
        v, a = np.zeros(3), np.zeros(3)
    return p, v, a


class BoundaryRush(Controller):
    """Take off, hover, then accelerate at full thrust until the drone leaves the safety box."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the controller from the optional `[boundary_rush]` config section."""
        super().__init__(obs, info, config)
        assert config.env.control_mode == "attitude", "boundary_rush requires attitude control"
        self.freq = config.env.freq
        params = load_dynamics_params(config.sim.dynamics, config.sim.drone)
        self.mass = float(params["mass"])
        cfg = config.get("boundary_rush", None) or {}
        self.height = float(cfg.get("height", 1.0))
        direction = np.asarray(cfg.get("direction", [-1.0, 0.0]), dtype=np.float64)
        self.dir = np.array([*(direction / np.linalg.norm(direction)), 0.0])
        self.max_speed = float(cfg.get("max_speed", 0.0))
        thrust_frac = float(cfg.get("thrust_frac", 0.9))
        self.thrust_max = 4 * float(params["thrust_max"]) * thrust_frac
        self.thrust_full = 4 * float(params["thrust_max"])  # Maps to pwm_max on the drone
        self.mode = str(cfg.get("mode", "rush"))
        assert self.mode in ("rush", "thrust_test"), f"Unknown boundary_rush mode {self.mode}"
        self.burst_duration = float(cfg.get("burst_duration", 0.15))

        self.start = np.asarray(obs["pos"], dtype=np.float64).copy()
        self.hover_pos = self.start.copy()
        self.hover_pos[2] = self.height
        # Position gains in N/m (Mellinger-like, as in attitude_controller.py), converted to m/s^2
        self.kp = np.array([0.4, 0.4, 1.25]) / self.mass
        self.kd = np.array([0.2, 0.2, 0.4]) / self.mass
        self.ki = np.array([0.05, 0.05, 0.05]) / self.mass
        self.i_error = np.zeros(3)
        self.tick = 0
        self.finished = False
        self.rush_logged = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return [roll, pitch, yaw, collective thrust]."""
        t = self.tick / self.freq
        pos = np.asarray(obs["pos"], dtype=np.float64)
        vel = np.asarray(obs["vel"], dtype=np.float64)
        t_rush = t - TAKEOFF_DURATION - HOVER_DURATION
        if self.mode == "thrust_test" and 0 <= t_rush < self.burst_duration:
            if not self.rush_logged:
                logger.info(f"Thrust burst started at {pos.round(2)}")
                self.rush_logged = True
            return np.array([0.0, 0.0, 0.0, self.thrust_full], dtype=np.float32)
        if t_rush < 0 or self.mode == "thrust_test":  # Track the hover reference
            if t_rush > self.burst_duration + BURST_HOVER:
                self.finished = True
            p, v, a = min_jerk(self.start, self.hover_pos, t, TAKEOFF_DURATION)
            e = p - pos
            self.i_error = np.clip(self.i_error + e / self.freq, -0.4, 0.4)
            a_des = a + self.kp * e + self.kd * (v - vel) + self.ki * self.i_error
        else:
            if not self.rush_logged:
                logger.info(f"Rush started at {pos.round(2)} towards {self.dir[:2].round(2)}")
                self.rush_logged = True
            if t_rush > RUSH_TIMEOUT:
                self.finished = True
            # Hold the height and the line through the hover point, push along the direction
            e = self.hover_pos - pos
            e_lat = e - self.dir * (e @ self.dir)
            v_lat = vel - self.dir * (vel @ self.dir)
            a_des = self.kp * e_lat - self.kd * v_lat
            a_des[2] = self.kp[2] * e[2] - self.kd[2] * vel[2] + self.ki[2] * self.i_error[2]
            a_along = RUSH_ACC
            if self.max_speed > 0:
                a_along = np.clip(4.0 * (self.max_speed - vel @ self.dir), -RUSH_ACC, RUSH_ACC)
            a_des += self.dir * a_along
        rpy, thrust = acc_to_attitude(a_des, 0.0, self.mass, self.thrust_max)
        return np.array([*rpy, thrust], dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the clock. Finishes only if the drone stayed in the box until the timeout."""
        self.tick += 1
        return self.finished

    def episode_callback(self):
        """Reset the internal state."""
        self.tick = 0
        self.i_error[:] = 0
        self.finished = False
        self.rush_logged = False
