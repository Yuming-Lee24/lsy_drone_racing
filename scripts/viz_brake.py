"""Watch brake controllers in the MuJoCo viewer, or record them to a GIF.

Each episode draws one takeover state (see `train_brake_rl.Args`) and runs every selected
controller from that identical state, one after the other. The RL policy uses the exported `.npz`
weights, i.e. exactly what `BrakeRL` deploys. Trails: rl red, analytic green, goto_vel0 blue,
goto_zero yellow; the white point marks the takeover position. The camera follows the drone from
the side, perpendicular to the takeover velocity.
Controllers are described in `eval_brake.py`.

Usage:
    pixi run -e gpu python scripts/viz_brake.py --speed 6
    pixi run -e gpu python scripts/viz_brake.py --controllers rl,goto_vel0 --episodes 3
    MUJOCO_GL=egl pixi run -e gpu python scripts/viz_brake.py --gif runs/brake.gif
"""

from __future__ import annotations

import time
from dataclasses import replace

import fire
import jax
import jax.numpy as jp
import numpy as np
from crazyflow.dynamics import load_params
from crazyflow.sim.visualize import draw_line, draw_points
from eval_brake import STOP_HOLD_STEPS, Analytic, GoTo

from lsy_drone_racing.control.brake_rl import DEFAULT_WEIGHTS
from lsy_drone_racing.control.train_brake_rl import Args, BrakeEnv

COLORS = {
    "rl": [1.0, 0.1, 0.1, 1.0],
    "analytic": [0.1, 0.8, 0.1, 1.0],
    "goto_vel0": [0.2, 0.4, 1.0, 1.0],
    "goto_zero": [1.0, 0.85, 0.1, 1.0],
}
CAM = {"distance": 3.0, "elevation": -15.0, "azimuth": 90.0}


class NpzPolicy:
    """Deterministic actor from the exported `.npz` weights (same math as `BrakeRL`)."""

    def __init__(self, weights: str):
        """Load the weights."""
        w = np.load(weights)
        self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(3)]

    def reset(self) -> None:
        """Nothing to reset."""

    def __call__(self, obs: jax.Array, step: int) -> jax.Array:
        """Action."""
        x = np.asarray(obs, dtype=np.float64)
        for W, b in self.layers:
            x = np.tanh(x @ W.T + b)
        return jp.asarray(x, dtype=jp.float32)


def main(
    controllers: str = "rl,analytic,goto_vel0",
    speed: float | None = None,
    episodes: int = 5,
    dynamics: str = "first_principles",
    randomize: bool = False,
    seed: int = 0,
    weights: str = str(DEFAULT_WEIGHTS),
    horizon_s: float = 4.0,
    slowdown: float = 1.0,
    follow: bool = True,
    gif: str | None = None,
) -> None:
    """Run the controllers episode by episode and render them.

    Args:
        controllers: Comma-separated subset of rl, analytic, goto_vel0, goto_zero.
        speed: Fixed takeover speed in m/s. None samples it from the training range.
        episodes: Number of takeover states.
        dynamics: Crazyflow dynamics model.
        randomize: Enable the training domain randomization (mass, thrust gain, delay, noise,
            disturbances).
        seed: Seed of the first episode.
        weights: Exported policy weights.
        horizon_s: Episode length in seconds.
        slowdown: Playback slowdown factor in the viewer (2 = half speed).
        follow: Let the camera follow the drone from the side.
        gif: Render offscreen to this GIF instead of opening the viewer.
    """
    args = Args(dynamics=dynamics, max_episode_time=horizon_s + 1.0, seed=seed)
    if speed is not None:
        args = replace(args, speed_min=speed, speed_max=speed)
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
    env = BrakeEnv(args, 1, device="cpu")
    env.record = True
    mass = float(load_params(dynamics, args.drone)["mass"])
    makers = {
        "rl": lambda: NpzPolicy(weights),
        "analytic": lambda: Analytic(env, mass),
        "goto_vel0": lambda: GoTo(env, mass, zero_vel=False),
        "goto_zero": lambda: GoTo(env, mass, zero_vel=True),
    }
    names = controllers.split(",") if isinstance(controllers, str) else list(controllers)
    mode = "rgb_array" if gif else "human"
    frames = []
    dt = 1.0 / env.freq
    for ep in range(episodes):
        for name in names:
            ctrl = makers[name]()
            env.key = jax.device_put(jax.random.key(seed + ep), env.device)
            obs, _ = env.reset(seed=seed + ep)
            ctrl.reset()
            s = env.state()
            p0 = np.asarray(s["p0"][0])
            v0 = np.asarray(s["vel"][0])
            speed0 = float(np.linalg.norm(v0))
            azimuth = float(np.degrees(np.arctan2(v0[1], v0[0]))) + 90.0
            trail = [np.asarray(s["pos"][0])]
            run, t_stop, outcome = 0, None, "not stopped"
            for k in range(int(horizon_s * env.freq)):
                t_start = time.perf_counter()
                obs, _, term, trunc, info = env.step(ctrl(obs, k))
                if bool(term[0]):
                    outcome = "crashed (floor/ceiling/flip)"
                    break
                pos = np.asarray(info["state"]["pos"][0])
                trail.append(pos)
                run = run + 1 if bool(info["stopped"][0]) else 0
                if t_stop is None and run >= STOP_HOLD_STEPS:
                    t_stop = (k + 2 - STOP_HOLD_STEPS) * dt
                    outcome = "stopped"
                draw_line(env.sim, np.array(trail[::2] + [pos]), rgba=np.array(COLORS[name]))
                draw_points(env.sim, p0[None], rgba=np.array([1.0, 1.0, 1.0, 1.0]), size=0.03)
                if follow and env.sim.viewer is not None:
                    env.sim.viewer.viewer.cam.lookat[:] = pos
                    env.sim.viewer.viewer.cam.azimuth = azimuth
                frame = env.sim.render(mode=mode, cam_config=CAM, width=960, height=540)
                if gif:
                    frames.append(frame)
                else:
                    time.sleep(max(0.0, dt * slowdown - (time.perf_counter() - t_start)))
            d_max = max(float(np.linalg.norm(p[:2] - p0[:2])) for p in trail)
            t_txt = f"{t_stop:.2f} s" if t_stop is not None else "-"
            print(
                f"ep {ep} {name:9s} v0 {speed0:.1f} m/s  {outcome:28s} t_stop {t_txt:7s} "
                f"d_max {d_max:.2f} m  z_min {min(p[2] for p in trail):.2f} m"
            )
    if gif:
        import imageio.v3 as iio

        iio.imwrite(gif, frames[::2], duration=2 * dt * 1000, loop=0)
        print(f"Saved {gif}")
    env.close()


if __name__ == "__main__":
    fire.Fire(main)
