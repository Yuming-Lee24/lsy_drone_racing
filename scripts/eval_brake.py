"""Evaluate brake controllers in simulation and compare the RL policy against baselines.

Controllers:
    rl          Trained policy: the actor of an `.npz` checkpoint from `train_brake_rl.py` (default:
                the deployed `brake_rl.npz`).
    analytic    Velocity-feedback controller that tilts the thrust vector against the velocity,
                saturated at full thrust (near bang-bang), holding the start height.
    goto_vel0   Approximation of the current `Crazyflie.return_to_start` braking: a 7th-order
                polynomial reference to a point 1 m ahead at 1.71 m height over 3 s, tracked by a
                Mellinger-like PD position controller. The reference starts at the current velocity.
    goto_zero   Same, but the reference starts at zero velocity (if the high-level planner ignores
                the drone's velocity at hand-over).

All controllers run in the same randomized `BrakeEnv` episodes. Metrics per episode: time until the
drone is stopped (`is_stopped` for 0.2 s), maximum horizontal excursion from the takeover point,
height drop and failures by cause (below MIN_HEIGHT, above 2 m, flipped, not stopped in the
horizon). `z_min`/`z_max` override the takeover height range of the training distribution.
Every controller sees identical episodes (same initial states, randomization and sensor noise) and
the baselines act on the same noisy position/velocity as the policy.

Usage:
    pixi run -e gpu python scripts/eval_brake.py --ckpt runs/brake/brake_final.npz
    pixi run -e gpu python scripts/eval_brake.py --z_min 0.4 --z_max 0.95 --controllers rl
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import fire
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.dynamics import load_params

from lsy_drone_racing.control.brake_rl import DEFAULT_WEIGHTS, MAX_TILT_CMD, MIN_HEIGHT
from lsy_drone_racing.control.train_brake_rl import MAX_HEIGHT, Args, BrakeEnv

G = 9.81
GOTO_DIST = 1.0
GOTO_DURATION = 3.0
GOTO_HEIGHT = 1.2 + 0.36 + 0.15  # return_height in real_race_env for level3 (tallest gate 1.2 m)
STOP_HOLD_STEPS = 10
Array = jp.ndarray


def septic_coeffs(p0: Array, v0: Array, p1: Array, T: float) -> Array:
    """Coefficients (..., 8) of x(t) with x(0)=p0, x'(0)=v0, x(T)=p1 and zero acc/jerk/end vel."""
    t = np.array(T)
    A = np.zeros((8, 8))
    for k in range(4):  # derivatives at 0
        A[k, k] = math.factorial(k)
    for k in range(4):  # derivatives at T
        for i in range(k, 8):
            A[4 + k, i] = np.prod(np.arange(i - k + 1, i + 1)) * t ** (i - k)
    A_inv = jp.asarray(np.linalg.inv(A), dtype=jp.float32)
    zeros = jp.zeros_like(p0)
    b = jp.stack([p0, v0, zeros, zeros, p1, zeros, zeros, zeros], axis=-1)
    return jp.einsum("ij,...j->...i", A_inv, b)


def eval_poly(c: Array, t: float) -> tuple[Array, Array, Array]:
    """Position, velocity and acceleration of the polynomial at time t."""
    i = np.arange(8)
    p = jp.sum(c * t**i, axis=-1)
    v = jp.sum(c[..., 1:] * i[1:] * t ** (i[1:] - 1), axis=-1)
    a = jp.sum(c[..., 2:] * i[2:] * (i[2:] - 1) * t ** (i[2:] - 2), axis=-1)
    return p, v, a


def acc_to_action(a_des: Array, yaw0: Array, mass: float, t_min: float, t_max: float) -> Array:
    """Map a desired world acceleration to a normalized [roll, pitch, thrust] action.

    Thrust is saturated with priority on the vertical component, tilt at MAX_TILT_CMD.
    """
    f = mass * (a_des + jp.array([0.0, 0.0, G]))
    fz = jp.clip(f[:, 2], 0.2 * mass * G, 0.98 * t_max)
    fxy = f[:, :2]
    fxy_max = jp.minimum(
        jp.sqrt(jp.maximum((0.98 * t_max) ** 2 - fz**2, 0.0)), fz * jp.tan(MAX_TILT_CMD)
    )
    n = jp.linalg.norm(fxy, axis=-1)
    fxy = fxy * jp.minimum(1.0, fxy_max / jp.maximum(n, 1e-9))[:, None]
    c, s = jp.cos(yaw0), jp.sin(yaw0)
    fh = jp.stack([c * fxy[:, 0] + s * fxy[:, 1], -s * fxy[:, 0] + c * fxy[:, 1], fz], axis=-1)
    T = jp.linalg.norm(fh, axis=-1)
    zb = fh / T[:, None]
    pitch = jp.arctan2(zb[:, 0], zb[:, 2])
    roll = jp.arcsin(jp.clip(-zb[:, 1], -1.0, 1.0))
    thrust = 2 * (jp.clip(T, t_min, t_max) - t_min) / (t_max - t_min) - 1
    return jp.stack([roll / MAX_TILT_CMD, pitch / MAX_TILT_CMD, thrust], axis=-1)


class Analytic:
    """Saturated velocity feedback: a_xy = -kv * v_xy, PD on height around the start height."""

    def __init__(self, env: BrakeEnv, mass: float, kv: float = 6.0):
        """Init."""
        self.env, self.mass, self.kv = env, mass, kv

    def reset(self) -> None:
        """Nothing to reset."""

    def __call__(self, obs: Array, step: int) -> Array:
        """Action."""
        s = self.env.state() | self.env.noisy
        v, z, z0 = s["vel"], s["pos"][:, 2], s["p0"][:, 2]
        a = jp.concatenate(
            [-self.kv * v[:, :2], (6.0 * (z0 - z) - 4.0 * v[:, 2])[:, None]], axis=-1
        )
        return acc_to_action(a, s["yaw0"], self.mass, self.env.thrust_min, self.env.thrust_max)


class GoTo:
    """PD + feedforward tracking of the high-level `go_to` braking reference (approximation)."""

    def __init__(self, env: BrakeEnv, mass: float, zero_vel: bool):
        """Init.

        Gains follow the Crazyflie Mellinger defaults (kp_xy 0.4, kd_xy 0.2, kp_z 1.25, kd_z 0.4 in
        N/m) divided by the mass.
        """
        self.env, self.mass, self.zero_vel = env, mass, zero_vel
        self.kp = jp.array([0.4, 0.4, 1.25]) / mass
        self.kd = jp.array([0.2, 0.2, 0.4]) / mass

    def reset(self) -> None:
        """Plan the reference from the current state."""
        s = self.env.state() | self.env.noisy
        p0, v0 = s["pos"], s["vel"]
        d = v0 / jp.maximum(jp.linalg.norm(v0, axis=-1, keepdims=True), 1e-6)
        p1 = (p0 + d * GOTO_DIST).at[:, 2].set(GOTO_HEIGHT)
        v_start = jp.zeros_like(v0) if self.zero_vel else v0
        self.coeffs = septic_coeffs(p0, v_start, p1, GOTO_DURATION)

    def __call__(self, obs: Array, step: int) -> Array:
        """Action."""
        s = self.env.state() | self.env.noisy
        t = min(step / self.env.freq, GOTO_DURATION)
        p, v, a = eval_poly(self.coeffs, t)
        if t >= GOTO_DURATION:
            v, a = jp.zeros_like(v), jp.zeros_like(a)
        a_des = a + self.kp * (p - s["pos"]) + self.kd * (v - s["vel"])
        return acc_to_action(a_des, s["yaw0"], self.mass, self.env.thrust_min, self.env.thrust_max)


class Policy:
    """Deterministic RL policy from the actor weights (W0..b2) of an `.npz` checkpoint."""

    def __init__(self, ckpt: Path):
        """Init."""
        w = np.load(ckpt)
        self.layers = [
            (jp.asarray(w[f"W{i}"], jp.float32), jp.asarray(w[f"b{i}"])) for i in range(3)
        ]
        self.act = jax.jit(self._act)

    def _act(self, layers: list, x: Array) -> Array:
        for W, b in layers:  # Every layer, including the output, is tanh
            x = jp.tanh(x @ W.T + b)
        return x

    def reset(self) -> None:
        """Nothing to reset."""

    def __call__(self, obs: Array, step: int) -> Array:
        """Action."""
        return self.act(self.layers, obs)


def rollout(env: BrakeEnv, ctrl, horizon: int, seed: int) -> dict[str, np.ndarray]:  # noqa: ANN001
    """Run one synchronized batch of episodes (no autoreset within the horizon).

    Both the simulation rng and the env rng (thrust gain, delay, sensor noise) are reseeded, so
    every controller sees identical episodes.
    """
    env.key = jax.device_put(jax.random.key(seed), env.device)
    obs, _ = env.reset(seed=seed)
    ctrl.reset()
    n = env.num_envs
    s0 = {key: np.asarray(v) for key, v in env.state().items()}
    speed0 = np.linalg.norm(s0["vel"], axis=-1)
    alive = np.ones(n, bool)
    cause = np.full(n, "", dtype=object)
    run = np.zeros(n, int)
    t_stop = np.full(n, np.inf)
    d_max = np.zeros(n)
    z_min = np.full(n, np.inf)
    traj = []
    for k in range(horizon):
        obs, _, term, _, info = env.step(ctrl(obs, k))
        s = {key: np.asarray(v) for key, v in info["state"].items()}
        stopped = np.asarray(info["stopped"])
        new_fail = alive & np.asarray(term)
        z = s["pos"][:, 2]
        cause[new_fail] = np.where(
            z[new_fail] < MIN_HEIGHT, "floor", np.where(z[new_fail] > MAX_HEIGHT, "ceiling", "flip")
        )
        alive &= ~new_fail
        d = np.linalg.norm(s["pos"][:, :2] - s["p0"][:, :2], axis=-1)
        d_max = np.where(alive, np.maximum(d_max, d), d_max)
        z_min = np.where(alive, np.minimum(z_min, z), z_min)
        run = np.where(stopped & alive, run + 1, 0)
        new = np.isinf(t_stop) & (run >= STOP_HOLD_STEPS)
        t_stop[new] = (k + 2 - STOP_HOLD_STEPS) / env.freq
        traj.append(np.concatenate([s["pos"][:8], s["vel"][:8]], axis=-1))
    cause[alive & np.isinf(t_stop)] = "not_stopped"
    ok = cause == ""
    return {
        "speed0": speed0,
        "z0": s0["pos"][:, 2],
        "vz0": s0["vel"][:, 2],
        "ok": ok,
        "cause": cause.astype(str),
        "t_stop": np.where(ok, t_stop, np.inf),
        "d_max": np.where(ok, d_max, np.inf),
        "z_drop": s["p0"][:, 2] - z_min,
        "traj": np.stack(traj, 1),
    }


def stats(r: dict[str, np.ndarray], m: np.ndarray, common: np.ndarray) -> dict:
    """Statistics over the episodes in mask `m`.

    `*_all` metrics count failed episodes as infinitely bad (so a median is finite only while the
    success rate is above 50 %). `*_common` metrics use only episodes every controller completed.
    """
    c = m & common

    def q(x: np.ndarray, p: float) -> float | None:
        v = float(np.percentile(x, p)) if x.size else None
        return None if v is None or not np.isfinite(v) else v

    return {
        "n": int(m.sum()),
        "success": float(r["ok"][m].mean()) if m.any() else None,
        **{
            f"fail_{k}": float((r["cause"][m] == k).mean()) if m.any() else None
            for k in ("floor", "ceiling", "flip", "not_stopped")
        },
        "t_stop_med_all": q(r["t_stop"][m], 50),
        "d_max_med_all": q(r["d_max"][m], 50),
        "n_common": int(c.sum()),
        "t_stop_med_common": q(r["t_stop"][c], 50),
        "t_stop_p95_common": q(r["t_stop"][c], 95),
        "d_max_med_common": q(r["d_max"][c], 50),
        "d_max_p95_common": q(r["d_max"][c], 95),
        "z_drop_p95_common": q(r["z_drop"][c], 95),
    }


def main(
    ckpt: str = str(DEFAULT_WEIGHTS),
    dynamics: str = "first_principles",
    n: int = 4096,
    horizon_s: float = 4.0,
    seed: int = 12345,
    randomize: bool = True,
    controllers: str = "rl,analytic,goto_vel0,goto_zero",
    kv: float = 6.0,
    z_min: float | None = None,
    z_max: float | None = None,
    out: str = "runs/brake_eval",
) -> None:
    """Evaluate the selected controllers on identical episode batches and write a JSON report."""
    args = Args(dynamics=dynamics, max_episode_time=horizon_s + 1.0, seed=seed)
    if z_min is not None:
        args = replace(args, z_min=z_min)
    if z_max is not None:
        args = replace(args, z_max=z_max)
    if not randomize:
        args = replace(
            args,
            mass_delta=0.0,
            thrust_gain_delta=0.0,
            max_delay=0,
            pos_noise=0.0,
            vel_noise=0.0,
            disturbance_force=0.0,
        )
    env = BrakeEnv(args, n, device="gpu" if jax.default_backend() == "gpu" else "cpu")
    env.record = True
    mass = float(load_params(dynamics, args.drone)["mass"])  # nominal, true mass is randomized
    makers = {
        "rl": lambda: Policy(Path(ckpt)),
        "analytic": lambda: Analytic(env, mass, kv),
        "goto_vel0": lambda: GoTo(env, mass, zero_vel=False),
        "goto_zero": lambda: GoTo(env, mass, zero_vel=True),
    }
    names = controllers.split(",") if isinstance(controllers, str) else list(controllers)
    raw = {k: rollout(env, makers[k](), int(horizon_s * env.freq), seed) for k in names}
    common = np.logical_and.reduce([r["ok"] for r in raw.values()])
    bins = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.01]
    report = {"n_common": int(common.sum())}
    for k, r in raw.items():
        report[k] = {
            "overall": stats(r, np.ones_like(common), common),
            "bins": {
                f"{lo:.0f}-{hi:.0f}": stats(r, (r["speed0"] >= lo) & (r["speed0"] < hi), common)
                for lo, hi in zip(bins[:-1], bins[1:])
            },
        }
        print(k, json.dumps(report[k]["overall"]), flush=True)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{dynamics}_{'rand' if randomize else 'nominal'}_z{args.z_min:g}-{args.z_max:g}"
    (out_dir / f"report_{tag}.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        out_dir / f"raw_{tag}.npz", **{f"{k}__{m}": v for k, r in raw.items() for m, v in r.items()}
    )


if __name__ == "__main__":
    fire.Fire(main)
