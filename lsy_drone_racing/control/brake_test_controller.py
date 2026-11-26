"""Controller that hovers, then dashes along `DASH_DIRECTION` until the drone leaves the limits.

It exists to test the RL brake on the real drone: once the drone crosses
`env.track.safety_limits`, the environment terminates and `RealRaceCoreEnv.close` hands the drone to
the brake policy. The controller never reports itself finished, so leaving the limits is the only
regular way the run ends.

Warning:
    The drone must reach `DASH_VEL` before it crosses the limit: the acceleration phase takes
    `DASH_VEL**2 / (2 * DASH_ACC)` meters. Leave several meters of free space beyond the limit for
    the brake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

HOVER_HEIGHT = 0.8  # m
TAKEOFF_DURATION = 2.0  # s, from the start position to the hover position
HOVER_DURATION = 3.0  # s, hold at the hover position before the dash
DASH_DIRECTION = np.array([0.0, -1.0, 0.0])  # Unit vector in the world frame
DASH_VEL = 6.0  # m/s
DASH_ACC = 10.0  # m/s^2, constant until DASH_VEL is reached


class BrakeTestController(Controller):
    """State controller that hovers at `HOVER_HEIGHT` and then flies along `DASH_DIRECTION`."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Plan the takeoff from the initial position.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._hover_pos = np.array([obs["pos"][0], obs["pos"][1], HOVER_HEIGHT])
        self._takeoff_spline = CubicSpline(
            [0.0, TAKEOFF_DURATION], np.stack([obs["pos"], self._hover_pos]), bc_type="clamped"
        )
        self._dash_start_time = TAKEOFF_DURATION + HOVER_DURATION
        self._dash_acc_duration = DASH_VEL / DASH_ACC
        self._tick = 0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, qx, qy, qz, qw, wx, wy, wz] as a
            numpy array.
        """
        t = self._tick / self._freq
        des_pos, des_vel, des_acc = self._hover_pos.copy(), np.zeros(3), np.zeros(3)
        if t < TAKEOFF_DURATION:
            des_pos = self._takeoff_spline(t)
            des_vel = self._takeoff_spline(t, 1)
        elif t >= self._dash_start_time:
            dash_time = t - self._dash_start_time
            if dash_time < self._dash_acc_duration:
                des_pos += 0.5 * DASH_ACC * dash_time**2 * DASH_DIRECTION
                des_vel = DASH_ACC * dash_time * DASH_DIRECTION
                des_acc = DASH_ACC * DASH_DIRECTION
            else:
                acc_distance = 0.5 * DASH_ACC * self._dash_acc_duration**2
                cruise_distance = DASH_VEL * (dash_time - self._dash_acc_duration)
                des_pos += (acc_distance + cruise_distance) * DASH_DIRECTION
                des_vel = DASH_VEL * DASH_DIRECTION
        des_quat = np.array([0.0, 0.0, 0.0, 1.0])
        return np.concatenate((des_pos, des_vel, des_acc, des_quat, np.zeros(3)), dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the time step counter.

        Returns:
            Always False: the run ends when the environment terminates at the safety limits.
        """
        self._tick += 1
        return False

    def episode_callback(self):
        """Reset the time step counter."""
        self._tick = 0
