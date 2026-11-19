"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

With `--brake_rl`, the RL brake policy (`lsy_drone_racing/control/brake_rl.py`) takes over after the
last gate and stops the drone, as `deploy.brake_rl` does on the real drone. It requires
`control_mode = "attitude"`.

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from crazyflow.dynamics import load_params
from crazyflow.sim.visualize import draw_line
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.control.brake_rl import MIN_HEIGHT, BrakeRL
from lsy_drone_racing.utils import load_config, load_controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv
    from lsy_drone_racing.envs.race_core import EnvData


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
    brake_rl: bool = False,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        render: Enable/disable rendering the simulation.
        brake_rl: Stop the drone with the RL brake policy after it finished the track.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    if brake_rl:
        assert config.env.control_mode == "attitude", "brake_rl requires control_mode = 'attitude'"
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
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
    env = JaxToNumpy(env)

    ep_times = []
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq

            t_loop = time.perf_counter()
            action = controller.compute_control(obs, info)
            control_time = time.perf_counter() - t_loop

            data_before_step = env.unwrapped.data
            obs, reward, terminated, truncated, info = env.step(action)
            # Update the controller internal state and models.
            t_callback = time.perf_counter()
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            if (exc := time.perf_counter() - t_callback + control_time - 1 / config.env.freq) > 0:
                logger.warning(f"Controller execution time exceeded loop frequency by {exc:.3f}s.")
            # Add up reward, collisions
            if terminated or truncated or controller_finished:
                break
            if config.sim.render:  # Render the sim if selected.
                if ((i * fps) % config.env.freq) < fps:
                    controller.render_callback(env.unwrapped.sim)
                    env.render()
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, curr_time)
        controller.episode_reset()
        finished = obs["n_gates_passed"] == obs["gate_sequence"].shape[0]
        ep_times.append(curr_time if finished else None)
        if brake_rl and finished:
            run_brake(env.unwrapped, data_before_step, action, config)

    # Close the environment
    env.close()
    return ep_times


def run_brake(
    env: DroneRaceEnv, data: EnvData, action: NDArray, config: ConfigDict, duration: float = 4.0
):
    """Stop the drone with the RL brake policy after it finished the track.

    The env disables the drone on the finishing step, so the step is replayed on the raw simulation
    without the race logic, and the brake policy then controls the simulation directly.
    """
    sim, freq = env.sim, config.env.freq
    n_substeps = int(data.sim_data.core.freq) // freq
    sim.data = env.build_apply_action_fn()(jp.asarray(action), data).sim_data
    sim.step(n_substeps)

    def state() -> dict[str, NDArray]:
        s = sim.data.states
        keys = ("pos", "quat", "vel", "ang_vel")
        return {k: np.asarray(getattr(s, k)[0, 0], dtype=np.float64) for k in keys}

    brake = BrakeRL(load_params(config.sim.dynamics, config.sim.drone))
    s = state()
    brake.reset(s)
    p0, speed0 = s["pos"].copy(), float(np.linalg.norm(s["vel"]))
    lim_low, lim_high = np.asarray(data.pos_limit_low), np.asarray(data.pos_limit_high)
    trail, t_stop, left_limits = [p0], None, False
    for k in range(int(duration * freq)):
        t_loop = time.perf_counter()
        if t_stop is None and brake.stopped(s):
            t_stop = k / freq
        rpy, thrust = brake.compute_control(s)
        sim.attitude_control(np.array([*rpy, thrust], dtype=np.float32).reshape(1, 1, 4))
        sim.step(n_substeps)
        s = state()
        trail.append(s["pos"])
        left_limits |= bool(np.any(s["pos"] < lim_low) or np.any(s["pos"] > lim_high))
        if config.sim.render:
            draw_line(sim, np.array(trail), rgba=np.array([1.0, 0.1, 0.1, 1.0]))
            sim.render()
            time.sleep(max(0.0, 1 / freq - (time.perf_counter() - t_loop)))
    trail = np.array(trail)
    logger.info(
        f"Brake: speed at takeover {speed0:.2f} m/s, "
        f"time to stop {f'{t_stop:.2f} s' if t_stop is not None else 'not stopped'}, "
        f"braking distance {np.max(np.linalg.norm(trail[:, :2] - p0[:2], axis=-1)):.2f} m, "
        f"min height {trail[:, 2].min():.2f} m (limit {MIN_HEIGHT} m), "
        f"{'LEFT' if left_limits else 'stayed within'} the safety limits"
    )


def log_episode_stats(obs: dict, curr_time: float):
    """Log the statistics of a single episode."""
    finished = obs["n_gates_passed"] == obs["gate_sequence"].shape[0]
    logger.info(
        f"Flight time (s): {curr_time}\n"
        f"Finished: {finished}\n"
        f"Gates passed: {obs['n_gates_passed']}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
