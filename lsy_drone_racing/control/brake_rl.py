"""RL emergency-brake policy that stops a fast drone after the last gate and holds hover.

The observation builder is shared between training (`train_brake_rl.py`, JAX) and deployment
(`BrakeRL`, numpy) so both see identical inputs. All quantities are expressed in the heading frame
(world frame rotated by the current yaw), which makes the policy invariant to position and yaw.

Observation (26): heading-frame velocity (3), heading-frame rotation matrix (9), body angular
velocity (3), height (1), heading-frame offset from the takeover point (3), maximum horizontal
excursion so far (1), last two normalized actions (6).
Action (3): normalized roll, pitch in [-1, 1] -> +-MAX_TILT_CMD and collective thrust in [-1, 1] ->
[thrust_min, thrust_max]. Yaw is held at its value when braking starts.

`BrakeRL` runs without torch (the deploy environment has none): export a trained checkpoint with
    pixi run -e gpu-tests python lsy_drone_racing/control/brake_rl.py <ckpt> [out.npz]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

N_OBS = 26
MAX_TILT_CMD = np.pi / 3
MIN_HEIGHT = 0.5
STOP_VEL = 0.1
STOP_ANG_VEL = 1.0
STOP_COS_TILT = float(np.cos(np.deg2rad(5.0)))
DEFAULT_WEIGHTS = Path(__file__).parent / "brake_rl.npz"


def quat2mat(quat: Any, xp: Any = np) -> Any:
    """Rotation matrices from scalar-last quaternions of shape (..., 4)."""
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return xp.stack([xp.stack(r, axis=-1) for r in rows], axis=-2)


def brake_obs(
    pos: Any,
    quat: Any,
    vel: Any,
    ang_vel: Any,
    p0: Any,
    d_max: Any,
    act1: Any,
    act2: Any,
    xp: Any = np,
) -> Any:
    """Build the heading-frame observation. Works on batched numpy or JAX arrays."""
    rot = quat2mat(quat, xp)
    yaw = xp.arctan2(rot[..., 1, 0], rot[..., 0, 0])
    c, s = xp.cos(yaw), xp.sin(yaw)
    zero, one = xp.zeros_like(c), xp.ones_like(c)
    rz_t = xp.stack(
        [xp.stack([c, s, zero], -1), xp.stack([-s, c, zero], -1), xp.stack([zero, zero, one], -1)],
        axis=-2,
    )
    vel_h = xp.einsum("...ij,...j->...i", rz_t, vel)
    rot_h = xp.einsum("...ij,...jk->...ik", rz_t, rot).reshape(rot.shape[:-2] + (9,))
    rel_h = xp.einsum("...ij,...j->...i", rz_t, pos - p0)
    d_max = xp.asarray(d_max)[..., None]
    return xp.concatenate([vel_h, rot_h, ang_vel, pos[..., 2:3], rel_h, d_max, act1, act2], axis=-1)


def brake_obs_jax(
    pos: Any, quat: Any, vel: Any, ang_vel: Any, p0: Any, d_max: Any, act1: Any, act2: Any
) -> Any:
    """JAX version of `brake_obs` for the training environment."""
    import jax.numpy as jp

    return brake_obs(pos, quat, vel, ang_vel, p0, d_max, act1, act2, xp=jp)


def is_stopped(quat: Any, vel: Any, ang_vel: Any, xp: Any = np) -> Any:
    """Hand-off condition: slow, barely rotating and nearly level."""
    cos_tilt = quat2mat(quat, xp)[..., 2, 2]
    return (
        (xp.linalg.norm(vel, axis=-1) < STOP_VEL)
        & (xp.linalg.norm(ang_vel, axis=-1) < STOP_ANG_VEL)
        & (cos_tilt > STOP_COS_TILT)
    )


class BrakeRL:
    """Deployable brake policy (numpy only).

    Call `reset` once at takeover, then `compute_control` at the training frequency (50 Hz) until
    `stopped` returns True.
    """

    def __init__(self, drone_params: dict, weights: Path = DEFAULT_WEIGHTS):
        """Load the exported actor weights.

        Args:
            drone_params: Crazyflow drone parameters (needs mass, thrust_min, thrust_max).
            weights: `.npz` file written by `export_weights`.
        """
        w = np.load(weights)
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
        self.d_max = max(self.d_max, float(np.linalg.norm(pos[:2] - self.p0[:2])))
        x = brake_obs(
            pos,
            np.asarray(obs["quat"], dtype=np.float64),
            np.asarray(obs["vel"], dtype=np.float64),
            np.asarray(obs["ang_vel"], dtype=np.float64),
            self.p0,
            self.d_max,
            self.act1,
            self.act2,
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
        """Whether the drone can be handed to the high-level commander."""
        return bool(is_stopped(np.asarray(obs["quat"]), np.asarray(obs["vel"]), obs["ang_vel"]))


def export_weights(ckpt: str | Path, out: str | Path = DEFAULT_WEIGHTS) -> None:
    """Export the actor of a `BrakeAgent` checkpoint to a torch-free `.npz` file."""
    import torch

    sd = torch.load(ckpt, map_location="cpu")
    arrays = {}
    for i, layer in enumerate((0, 2, 4)):
        arrays[f"W{i}"] = sd[f"actor_mean.0.{layer}.weight"].numpy().astype(np.float64)
        arrays[f"b{i}"] = sd[f"actor_mean.0.{layer}.bias"].numpy().astype(np.float64)
    np.savez(out, **arrays)


if __name__ == "__main__":
    import fire

    fire.Fire(export_weights)
