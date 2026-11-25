"""Record state-action trajectories for fitting a residual dynamics model.

Runs a controller exactly like ``deploy.py`` (``--mode real``) or ``sim.py`` (``--mode sim``) and
saves, for every control step k, the observed drone state, the command sent, and timing, so that
``(state[k], action[k]) -> state[k + 1]`` pairs can be replayed in simulation.

Usage:

    python scripts/record_flight.py --config level3.toml --controller attitude_jax_ppo_level3.py
    python scripts/record_flight.py --mode sim --n_runs 3 --config level3.toml \
        --controller attitude_jax_ppo_level3.py

Each episode is written to ``saves/flight_logs/<timestamp>_<mode>_<episode>/`` as ``flight.npz``
(arrays below, ``state`` arrays have one more entry than ``action``) and ``meta.json``.

    time         (N+1,)    seconds since the first observation (wall clock on the real drone,
                           step index / freq in simulation)
    pos          (N+1, 3)  world position
    quat         (N+1, 4)  orientation, xyzw
    vel          (N+1, 3)  world linear velocity
    ang_vel      (N+1, 3)  angular velocity
    n_gates_passed (N+1,)   index into ``gate_sequence`` (saved in meta.json) of the target pass
    action       (N, A)    command sent to the drone (after optional excitation noise)
    policy_action (N, A)   command produced by the controller
    compute_time (N,)      controller computation time in seconds
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import numpy as np

from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parents[1]
STATE_KEYS = ("pos", "quat", "vel", "ang_vel", "n_gates_passed")


class FlightRecorder:
    """Accumulate one episode in memory and write it once at the end."""

    def __init__(self, sim_freq: float | None = None):
        """Start an empty episode; simulated episodes are timestamped by step index."""
        self.start = time.perf_counter()
        self.sim_freq = sim_freq
        self.columns: dict[str, list] = {key: [] for key in (*STATE_KEYS, "time")}
        self.columns.update(action=[], policy_action=[], compute_time=[])

    def add_state(self, obs: dict):
        """Store one observation and its timestamp."""
        if self.sim_freq is None:
            self.columns["time"].append(time.perf_counter() - self.start)
        else:
            self.columns["time"].append(len(self.columns["time"]) / self.sim_freq)
        for key in STATE_KEYS:
            self.columns[key].append(np.asarray(obs[key]))

    def add_action(self, action: np.ndarray, policy_action: np.ndarray, compute_time: float):
        """Store the command sent after the most recent observation."""
        self.columns["action"].append(np.asarray(action, dtype=np.float32))
        self.columns["policy_action"].append(np.asarray(policy_action, dtype=np.float32))
        self.columns["compute_time"].append(compute_time)

    def save(self, directory: Path, meta: dict):
        """Write the episode arrays and metadata."""
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "flight.npz", **{k: np.asarray(v) for k, v in self.columns.items()}
        )
        meta = {**meta, "steps": len(self.columns["action"])}
        (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        logger.info(f"Saved {meta['steps']} steps to {directory}")


def _excite(action: np.ndarray, rng: np.random.Generator, std: float, attitude: bool) -> np.ndarray:
    if std <= 0.0 or not attitude:
        return action
    noisy = action.copy()
    noisy[:2] += rng.normal(0.0, std, size=2)
    return noisy


def _make_env(mode: str, config: ConfigDict) -> gymnasium.Env:
    if mode == "real":
        import rclpy

        rclpy.init()
        return gymnasium.make(
            "RealDroneRacing-v0",
            drones=config.deploy.drones,
            freq=config.env.freq,
            track=config.env.track,
            randomizations=config.env.randomizations,
            dynamics=config.sim.dynamics,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
        )
    from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

    config.sim.render = False
    return JaxToNumpy(
        gymnasium.make(
            config.env.id,
            freq=config.env.freq,
            sim_config=config.sim,
            sensor_range=config.env.sensor_range,
            control_mode=config.env.control_mode,
            track=config.env.track,
            disturbances=config.env.get("disturbances"),
            randomizations=config.env.get("randomizations"),
            seed=config.env.seed,
        )
    )


def main(
    config: str = "level3.toml",
    controller: str | None = None,
    mode: str = "real",
    n_runs: int = 1,
    action_noise: float = 0.0,
    output: str | None = None,
    seed: int = 0,
):
    """Fly the controller and record every control step.

    Args:
        config: Competition configuration in `config/`.
        controller: Controller file in `lsy_drone_racing/control/`; defaults to the config's.
        mode: "real" to fly the physical drone, "sim" to record the simulator instead.
        n_runs: Number of episodes (sim only; real flights record one episode per call).
        action_noise: Std of Gaussian noise in rad added to commanded roll and pitch, to excite
            dynamics the policy alone rarely visits. Keep 0 unless the flight area is safe.
        output: Base directory; defaults to `saves/flight_logs`.
        seed: Seed of the excitation noise.
    """
    if mode not in ("real", "sim"):
        raise ValueError("mode must be 'real' or 'sim'")
    config_name = config
    config = load_config(ROOT / "config" / config)
    if controller is not None:
        config.controller.file = controller
    controller_cls = load_controller(ROOT / "lsy_drone_racing/control" / config.controller.file)
    attitude = config.env.control_mode == "attitude"
    rng = np.random.default_rng(seed)
    base = Path(output) if output else ROOT / "saves" / "flight_logs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = _make_env(mode, config)
    try:
        for episode in range(n_runs if mode == "sim" else 1):
            if mode == "real":
                obs, info = env.reset(seed=config.env.seed, options=config.deploy)
                drone = config.deploy.drones[0]["drone"]
            else:
                obs, info = env.reset()
                drone = config.sim.drone
            ctrl = controller_cls(obs, info, config)
            recorder = FlightRecorder(None if mode == "real" else config.env.freq)
            outcome = "running"
            try:
                while True:
                    t_loop = time.perf_counter()
                    if mode == "real":
                        obs, info = env.unwrapped.obs(), env.unwrapped.info()
                        obs = {k: v[0] for k, v in obs.items()}
                    recorder.add_state(obs)
                    policy_action = np.asarray(ctrl.compute_control(obs, info))
                    compute_time = time.perf_counter() - t_loop
                    action = _excite(policy_action, rng, action_noise, attitude)
                    recorder.add_action(action, policy_action, compute_time)
                    obs, reward, terminated, truncated, info = env.step(action)
                    finished = ctrl.step_callback(action, obs, reward, terminated, truncated, info)
                    if terminated or truncated or finished:
                        outcome = (
                            "terminated" if terminated else "truncated" if truncated else "finished"
                        )
                        break
                    dt = time.perf_counter() - t_loop
                    if mode == "real" and dt < 1 / config.env.freq:
                        time.sleep(1 / config.env.freq - dt)
            except KeyboardInterrupt:
                outcome = "interrupted"
                raise
            finally:
                if mode == "real":
                    obs = {k: v[0] for k, v in env.unwrapped.obs().items()}
                recorder.add_state(obs)
                recorder.save(
                    base / f"{stamp}_{mode}_{episode:03d}",
                    {
                        "mode": mode,
                        "config": config_name,
                        "controller": config.controller.file,
                        "drone": drone,
                        "freq": config.env.freq,
                        "control_mode": config.env.control_mode,
                        "action_noise": action_noise,
                        "seed": seed,
                        "outcome": outcome,
                        "gates_passed": int(np.asarray(obs["n_gates_passed"])),
                        "gate_sequence": np.asarray(obs["gate_sequence"]).tolist(),
                        "gates_pos": np.asarray(obs["gates_pos"]).tolist(),
                        "obstacles_pos": np.asarray(obs["obstacles_pos"]).tolist(),
                        "started": stamp,
                    },
                )
    finally:
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("jax").setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    fire.Fire(main)
