"""Compare return-to-start with and without the RL brake after the drone leaves the safety box.

The drone takes off to 1.7 m, hovers, then accelerates at full thrust until it crosses the safety
limits (`lsy_drone_racing/control/boundary_rush.py`). From that exact state, the simulation is
replayed twice, mirroring `RealRaceCoreEnv.close`:

    return_home  `Crazyflie.return_to_start` alone: go_to 1 m further along the velocity at the
                 return height (3 s), go_to the start at the return height (5 s), descend (3 s).
    rl_brake     RL brake policy (`BrakeRL`, at most 3 s, same hand-off rules as `_brake`), then
                 `return_to_start` with `breaking_distance = 0` if it stopped the drone.

The firmware high-level commander is approximated: each go_to is a 7th-order polynomial from the
current reference (the first one starts at the measured position and velocity, as the firmware
takes over from the current state) tracked by a Mellinger-like PD position controller at 50 Hz.
Treat its numbers as an estimate of the real return_to_start, not as an exact replica.

Usage:
    python scripts/sim_boundary_rush.py
    python scripts/sim_boundary_rush.py --direction "[0, 1]" --max_speed 3 --slowdown 2
    python scripts/sim_boundary_rush.py --render False --plot plots/boundary_rush.png
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np
from crazyflow.dynamics import load_params
from crazyflow.sim.visualize import draw_line, draw_points
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.control.boundary_rush import BoundaryRush, acc_to_attitude
from lsy_drone_racing.control.brake_rl import MIN_HEIGHT, BrakeRL
from lsy_drone_racing.utils import load_config

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

COLORS = {
    "rush": [1.0, 1.0, 1.0, 1.0],
    "return_home": [0.2, 0.4, 1.0, 1.0],
    "rl_brake": [1.0, 0.1, 0.1, 1.0],
}
BREAK_DURATION, RETURN_DURATION, LAND_DURATION = 3.0, 5.0, 3.0
BRAKE_MAX_DURATION = 3.0
STOP_SPEED = 0.2  # m/s, horizontal speed counted as "stopped" in the comparison


def septic(p0: NDArray, v0: NDArray, p1: NDArray, T: float) -> NDArray:
    """Coefficients (3, 8): x(0)=p0, x'(0)=v0, x(T)=p1, rest-to-rest higher derivatives."""
    A = np.zeros((8, 8))
    for k in range(4):
        A[k, k] = math.factorial(k)
        for i in range(k, 8):
            A[4 + k, i] = np.prod(np.arange(i - k + 1, i + 1)) * T ** (i - k)
    z = np.zeros(3)
    b = np.stack([p0, v0, z, z, p1, z, z, z], axis=-1)
    return np.linalg.solve(A, b.T).T


def eval_poly(c: NDArray, t: float) -> tuple[NDArray, NDArray, NDArray]:
    """Position, velocity and acceleration of the polynomial at time t."""
    i = np.arange(8)
    p = c @ t**i
    v = c[:, 1:] @ (i[1:] * t ** (i[1:] - 1))
    a = c[:, 2:] @ (i[2:] * (i[2:] - 1) * t ** (i[2:] - 2))
    return p, v, a


class Recorder:
    """Steps the raw simulation with attitude commands and records the trajectory."""

    def __init__(self, sim, freq: int, n_substeps: int, render: bool, slowdown: float, color):  # noqa: ANN001
        """Init."""
        self.sim, self.freq, self.n_substeps = sim, freq, n_substeps
        self.render, self.slowdown, self.color = render, slowdown, np.array(color)
        self.t, self.pos, self.vel, self.tilt, self.phase = [], [], [], [], []
        self.background: list[tuple[NDArray, NDArray]] = []

    def state(self) -> dict[str, NDArray]:
        """Current state of drone 0 in world 0."""
        s = self.sim.data.states
        keys = ("pos", "quat", "vel", "ang_vel")
        return {k: np.asarray(getattr(s, k)[0, 0], dtype=np.float64) for k in keys}

    def step(self, rpy: NDArray, thrust: float, phase: str) -> dict[str, NDArray]:
        """Apply one 50 Hz attitude command and record the resulting state."""
        t_loop = time.perf_counter()
        cmd = np.array([*rpy, thrust], dtype=np.float32).reshape(1, 1, 4)
        self.sim.attitude_control(cmd)
        self.sim.step(self.n_substeps)
        s = self.state()
        self.t.append(len(self.t) / self.freq)
        self.pos.append(s["pos"])
        self.vel.append(s["vel"])
        q = s["quat"]
        self.tilt.append(np.degrees(np.arccos(np.clip(1 - 2 * (q[0] ** 2 + q[1] ** 2), -1, 1))))
        self.phase.append(phase)
        if self.render:
            for pts, rgba in self.background:
                draw_line(self.sim, pts, rgba=rgba)
            if len(self.pos) > 1:
                draw_line(self.sim, np.array(self.pos[::2] + [s["pos"]]), rgba=self.color)
            self.sim.render()
            time.sleep(max(0.0, self.slowdown / self.freq - (time.perf_counter() - t_loop)))
        return s

    def arrays(self) -> dict[str, NDArray]:
        """Recorded trajectory as arrays."""
        return {
            "t": np.array(self.t),
            "pos": np.array(self.pos),
            "vel": np.array(self.vel),
            "tilt": np.array(self.tilt),
            "phase": np.array(self.phase),
        }


def return_to_start(
    rec: Recorder,
    return_pos: NDArray,
    return_height: float,
    breaking_distance: float,
    mass: float,
    thrust_max: float,
    start_with_vel: bool,
) -> None:
    """Emulate `Crazyflie.return_to_start`: high-level go_to segments + onboard position control."""
    kp = np.array([0.4, 0.4, 1.25]) / mass
    kd = np.array([0.2, 0.2, 0.4]) / mass
    s = rec.state()
    vel_norm = np.linalg.norm(s["vel"])
    break_pos = s["pos"].copy()
    if vel_norm > 1e-6:
        break_pos += s["vel"] / vel_norm * breaking_distance
    break_pos[2] = return_height
    home = np.array([return_pos[0], return_pos[1], return_height])
    land = np.array([return_pos[0], return_pos[1], 0.05])
    p_ref, v_ref = s["pos"], (s["vel"] if start_with_vel else np.zeros(3))
    segments = [("rth_break", break_pos, BREAK_DURATION), ("rth_return", home, RETURN_DURATION)]
    segments.append(("rth_land", land, LAND_DURATION))
    for phase, goal, T in segments:
        c = septic(p_ref, v_ref, goal, T)
        for k in range(int(T * rec.freq)):
            t = (k + 1) / rec.freq
            p, v, a = eval_poly(c, min(t, T))
            a_des = a + kp * (p - s["pos"]) + kd * (v - s["vel"])
            rpy, thrust = acc_to_attitude(a_des, 0.0, mass, thrust_max)
            s = rec.step(rpy, thrust, phase)
        p_ref, v_ref = goal, np.zeros(3)


def rl_brake(rec: Recorder, brake: BrakeRL) -> bool:
    """Run the brake policy with the same hand-off rules as `RealRaceCoreEnv._brake`."""
    s = rec.state()
    brake.reset(s)
    for _ in range(int(BRAKE_MAX_DURATION * rec.freq)):
        if brake.stopped(s):
            return True
        if s["pos"][2] < MIN_HEIGHT - 0.2:
            logger.warning("Brake policy fell below its height limit, handing over")
            return False
        rpy, thrust = brake.compute_control(s)
        s = rec.step(rpy, thrust, "rl_brake")
    return False


def outside_dist(pos: NDArray, low: NDArray, high: NDArray) -> NDArray:
    """Distance of each position outside the box (0 inside)."""
    return np.linalg.norm(np.maximum(0, low - pos) + np.maximum(0, pos - high), axis=-1)


def summarize(name: str, tr: dict, low: NDArray, high: NDArray, home: NDArray) -> dict:
    """Comparison metrics of one recovery."""
    out = outside_dist(tr["pos"], low, high)
    speed_xy = np.linalg.norm(tr["vel"][:, :2], axis=-1)
    stopped = np.flatnonzero(speed_xy < STOP_SPEED)
    inside = np.flatnonzero(out == 0)
    return {
        "name": name,
        "max_outside": float(out.max()),
        "t_stop": float(tr["t"][stopped[0]]) if len(stopped) else np.nan,
        "t_inside": float(tr["t"][inside[0]]) if len(inside) else np.nan,
        "t_outside_total": float((out > 0).sum() / 50.0),
        "z_min": float(tr["pos"][:, 2].min()),
        "z_max": float(tr["pos"][:, 2].max()),
        "tilt_max": float(tr["tilt"].max()),
        "land_err": float(np.linalg.norm(tr["pos"][-1, :2] - home[:2])),
    }


def plot(rush: dict, results: dict, low: NDArray, high: NDArray, home: NDArray, path: str) -> None:
    """Top view, speed, height and distance outside the box over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"return_home": "tab:blue", "rl_brake": "tab:red"}
    labels = {"return_home": "return_to_start only", "rl_brake": "RL brake + return_to_start"}
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    a.add_patch(
        plt.Rectangle(
            low[:2], *(high[:2] - low[:2]), fill=False, ec="k", ls="--", label="safety box"
        )
    )
    a.plot(rush["pos"][:, 0], rush["pos"][:, 1], c="gray", label="rush")
    exit_p = rush["pos"][-1]
    a.plot(*exit_p[:2], "kx", ms=10, label="box exit")
    a.plot(*home[:2], "g*", ms=14, label="start / home")
    for name, (tr, _) in results.items():
        a.plot(tr["pos"][:, 0], tr["pos"][:, 1], c=colors[name], label=labels[name])
        if (tr["phase"] == "rl_brake").any():
            last = np.flatnonzero(tr["phase"] == "rl_brake")[-1]
            a.plot(
                *tr["pos"][last, :2], "o", c=colors[name], mfc="none", ms=10, label="brake hand-off"
            )
    a.set_aspect("equal")
    a.set_xlabel("x [m]")
    a.set_ylabel("y [m]")
    a.set_title("Top view")
    a.legend(fontsize=8, loc="best")
    t_rush = rush["t"] - rush["t"][-1]
    rush_speed = np.linalg.norm(rush["vel"], axis=-1)
    for a, ylabel in zip(
        (ax[0, 1], ax[1, 0], ax[1, 1]), ("speed [m/s]", "z [m]", "outside box [m]")
    ):
        a.axvline(0, c="k", ls=":", lw=1)
        a.set_xlabel("time since box exit [s]")
        a.set_ylabel(ylabel)
        a.grid(alpha=0.3)
    ax[0, 1].plot(t_rush, rush_speed, c="gray")
    ax[1, 0].plot(t_rush, rush["pos"][:, 2], c="gray")
    for name, (tr, _) in results.items():
        ax[0, 1].plot(
            tr["t"], np.linalg.norm(tr["vel"], axis=-1), c=colors[name], label=labels[name]
        )
        ax[1, 0].plot(tr["t"], tr["pos"][:, 2], c=colors[name])
        ax[1, 1].plot(tr["t"], outside_dist(tr["pos"], low, high), c=colors[name])
    ax[1, 0].axhline(MIN_HEIGHT, c="k", ls="--", lw=1)
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_title(f"Speed (exit speed {rush_speed[-1]:.2f} m/s)")
    ax[1, 0].set_title("Height (dashed: RL brake min height)")
    ax[1, 1].set_title("Distance outside the safety box")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"Saved plot to {path}")


def main(
    config: str = "level2.toml",
    direction: list[float] | None = None,
    max_speed: float | None = None,
    height: float | None = None,
    thrust_frac: float | None = None,
    render: bool = True,
    slowdown: float = 1.0,
    clean: bool = True,
    goto_start_vel: bool = True,
    seed: int = 0,
    plot_path: str = "plots/boundary_rush.png",
):
    """Fly out of the safety box and compare both recoveries from the identical exit state.

    Args:
        config: Config file in `config/`. Its track, safety limits and drone are used.
        direction: Horizontal rush direction, default [-1, 0] (or `[boundary_rush]` in the config).
        max_speed: Speed cap in m/s for the rush, 0 for full thrust until leaving the box.
        height: Takeoff height in m.
        thrust_frac: Fraction of the maximum collective thrust used during the rush.

        `direction`, `max_speed`, `height` and `thrust_frac` default to the config's
        `[boundary_rush]` section, then to the controller defaults.
        render: Show the MuJoCo viewer (white rush, blue return_to_start, red RL brake).
        slowdown: Playback slowdown in the viewer.
        clean: Disable the config's disturbances and randomizations.
        goto_start_vel: The first emulated go_to starts at the measured velocity (firmware takes
            over from the current state). False starts it at zero velocity.
        seed: Environment seed.
        plot_path: Where to save the comparison plot.
    """
    cfg = load_config(Path(__file__).parents[1] / "config" / config)
    cfg.env.control_mode = "attitude"
    cfg.sim.render = render
    cfg.env.seed = seed
    rush_cfg = dict(cfg.get("boundary_rush", None) or {})
    overrides = {"height": height, "max_speed": max_speed, "thrust_frac": thrust_frac}
    rush_cfg.update({k: v for k, v in overrides.items() if v is not None})
    if direction is not None:
        rush_cfg["direction"] = list(direction)
    cfg.boundary_rush = rush_cfg
    if clean:
        cfg.env.disturbances = None
        cfg.env.randomizations = None
    env = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=seed,
    )
    env = JaxToNumpy(env)
    raw = env.unwrapped
    freq = cfg.env.freq
    obs, info = env.reset(seed=seed)
    ctrl = BoundaryRush(obs, info, cfg)
    # The sim env uses fixed +-3 m limits, the real env the config's safety limits: check those here
    low = np.asarray(cfg.env.track.safety_limits.pos_limit_low, dtype=np.float64)
    high = np.asarray(cfg.env.track.safety_limits.pos_limit_high, dtype=np.float64)
    home = np.asarray(cfg.env.track.drones[0]["pos"], dtype=np.float64)
    return_height = max(g["pos"][2] for g in cfg.env.track.gates) + 0.36 + 0.15

    # Phase 1: takeoff, hover and rush until the drone leaves the safety box (as RealRaceCoreEnv)
    rush = {"t": [], "pos": [], "vel": []}
    left_box = terminated = finished = taken_off = False
    k = 0
    while not (left_box or terminated or finished):
        t_loop = time.perf_counter()
        action = ctrl.compute_control(obs, info)
        obs, _, terminated, truncated, info = env.step(action)
        finished = ctrl.step_callback(action, obs, 0.0, terminated, truncated, info)
        pos = np.asarray(obs["pos"], dtype=np.float64)
        rush["t"].append(k / freq)
        rush["pos"].append(pos)
        rush["vel"].append(np.asarray(obs["vel"], dtype=np.float64))
        taken_off |= pos[2] > 0.1  # The sim drone rests slightly below the z limit on the ground
        left_box = taken_off and bool(np.any(pos < low) or np.any(pos > high))
        k += 1
        if render:
            env.render()
            time.sleep(max(0.0, slowdown / freq - (time.perf_counter() - t_loop)))
    rush = {key: np.array(v) for key, v in rush.items()}
    if not left_box:
        reason = "crashed" if terminated else "did not leave the safety box within the rush timeout"
        print(f"The drone {reason} at {rush['pos'][-1].round(2)}.")
        env.close()
        return

    # From here on, the recoveries command the raw simulation directly without the race logic
    sim = raw.sim
    snapshot = raw.data.sim_data  # The env keeps its own copy of the sim state, not sim.data
    n_substeps = int(snapshot.core.freq) // freq
    params = load_params(cfg.sim.dynamics, cfg.sim.drone)
    mass, thrust_max = float(params["mass"]), 4 * float(params["thrust_max"])
    exit_pos, exit_vel = rush["pos"][-1], rush["vel"][-1]
    print(
        f"Left the box at {exit_pos.round(2)} with {np.linalg.norm(exit_vel):.2f} m/s "
        f"(vel {exit_vel.round(2)}), {rush['t'][-1] - 5.0:.2f} s after the rush started"
    )

    # Phase 2: both recoveries from the identical state
    rush_line = (rush["pos"][::2], np.array(COLORS["rush"]))
    results = {}
    for name in ("return_home", "rl_brake"):
        sim.data = snapshot
        rec = Recorder(sim, freq, n_substeps, render, slowdown, COLORS[name])
        rec.background = [rush_line]
        stopped = False
        if name == "rl_brake":
            stopped = rl_brake(rec, BrakeRL(params))
        return_to_start(
            rec, home, return_height, 0.0 if stopped else 1.0, mass, thrust_max, goto_start_vel
        )
        tr = rec.arrays()
        m = summarize(name, tr, low, high, home)
        m["brake"] = ("stopped" if stopped else "handed over") if name == "rl_brake" else "-"
        results[name] = (tr, m)
        if render:
            draw_points(sim, exit_pos[None], rgba=np.array([1.0, 1.0, 0.0, 1.0]), size=0.03)
            time.sleep(1.0)

    hdr = (
        f"{'recovery':12s} {'max outside':>11s} {'t stop':>7s} {'t back in':>9s} "
        f"{'t outside':>9s} {'z min':>6s} {'z max':>6s} {'tilt max':>8s} {'land err':>8s}  brake"
    )
    print(hdr)
    print("-" * len(hdr))
    for _, m in results.values():
        print(
            f"{m['name']:12s} {m['max_outside']:9.2f} m {m['t_stop']:5.2f} s "
            f"{m['t_inside']:7.2f} s "
            f"{m['t_outside_total']:7.2f} s {m['z_min']:6.2f} {m['z_max']:6.2f} "
            f"{m['tilt_max']:6.0f} deg {m['land_err']:6.2f} m  {m['brake']}"
        )
    print(f"(t stop: first time the horizontal speed is below {STOP_SPEED} m/s; times since exit)")
    plot(rush, results, low, high, home, plot_path)
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    fire.Fire(main)
