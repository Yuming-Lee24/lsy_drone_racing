"""RL brake policy that stops the drone after the race and holds hover.

The real environment hands the drone to this policy when a deployment ends (track finished, safety
limits left or controller finished), before the high-level commander returns it to the start.

`BrakeRL` runs on the CPU with numpy only and loads the actor weights in `brake_rl.npz`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_TILT_CMD = np.pi / 3
MIN_HEIGHT = 0.10  # Training floor: lower counts as a crash
STOP_VEL = 0.1
STOP_ANG_VEL = 1.0
STOP_COS_TILT = float(np.cos(np.deg2rad(5.0)))
WEIGHTS = Path(__file__).parent / "brake_rl.npz"


def quat2mat(quat: NDArray) -> NDArray:
    """Rotation matrix from a scalar-last quaternion."""
    x, y, z, w = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class BrakeRL:
    """RL brake policy.

    Call `reset` once at takeover, then `compute_control` at 50 Hz until `stopped` returns True.
    """

    def __init__(self, drone_params: dict):
        """Load the actor weights.

        Args:
            drone_params: Crazyflow drone parameters (needs mass, thrust_min, thrust_max).
        """
        w = np.load(WEIGHTS)
        self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(3)]
        self.thrust_min = float(drone_params["thrust_min"]) * 4
        self.thrust_max = float(drone_params["thrust_max"]) * 4
        hover = float(drone_params["mass"]) * 9.81
        self.hover_act = 2 * (hover - self.thrust_min) / (self.thrust_max - self.thrust_min) - 1
        self.yaw0 = 0.0
        self.p0 = np.zeros(3)
        self.d_max = 0.0
        self.act1 = self.act2 = np.array([0.0, 0.0, self.hover_act])

    def reset(self, obs: dict[str, NDArray]) -> None:
        """Latch the takeover point and heading and clear the action history."""
        rot = quat2mat(np.asarray(obs["quat"], dtype=np.float64))
        self.yaw0 = float(np.arctan2(rot[1, 0], rot[0, 0]))
        self.p0 = np.asarray(obs["pos"], dtype=np.float64).copy()
        self.d_max = 0.0
        self.act1 = self.act2 = np.array([0.0, 0.0, self.hover_act])

    def compute_control(self, obs: dict[str, NDArray]) -> tuple[NDArray, float]:
        """Return the attitude command [roll, pitch, yaw] in rad and the collective thrust in N."""
        pos = np.asarray(obs["pos"], dtype=np.float64)
        rot = quat2mat(np.asarray(obs["quat"], dtype=np.float64))
        self.d_max = max(self.d_max, float(np.linalg.norm(pos[:2] - self.p0[:2])))
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        rz_t = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
        x = np.concatenate(
            [
                rz_t @ np.asarray(obs["vel"], dtype=np.float64),
                (rz_t @ rot).ravel(),
                np.asarray(obs["ang_vel"], dtype=np.float64),
                pos[2:3],
                rz_t @ (pos - self.p0),
                [self.d_max],
                self.act1,
                self.act2,
            ]
        )
        for W, b in self.layers:  # Every layer of the actor, including the output, is tanh
            x = np.tanh(W @ x + b)
        act = np.clip(x, -1.0, 1.0)
        self.act2, self.act1 = self.act1, act
        thrust = (act[2] + 1) / 2 * (self.thrust_max - self.thrust_min) + self.thrust_min
        rpy = np.array([act[0] * MAX_TILT_CMD, act[1] * MAX_TILT_CMD, self.yaw0])
        return rpy, float(thrust)

    @staticmethod
    def stopped(obs: dict[str, NDArray]) -> bool:
        """Whether the drone is slow, barely rotating and nearly level."""
        cos_tilt = quat2mat(np.asarray(obs["quat"], dtype=np.float64))[2, 2]
        return bool(
            np.linalg.norm(obs["vel"]) < STOP_VEL
            and np.linalg.norm(obs["ang_vel"]) < STOP_ANG_VEL
            and cos_tilt > STOP_COS_TILT
        )
