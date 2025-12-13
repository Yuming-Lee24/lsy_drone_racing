import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
from lsy_drone_racing.utils import load_config
from drone_models.core import load_params

from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from drone_controllers.mellinger.params import ForceTorqueParams

config_path = Path(__file__).parents[2] / "config" / "level0.toml"
print(config_path)
config = load_config(config_path)

env = VecDroneRaceEnv(
    num_envs=1,
    freq=config.env.freq,
    sim_config=config.sim,
    track=config.env.track,
    sensor_range=config.env.sensor_range,
    control_mode=config.env.control_mode, 
    seed=42,
    device="cpu"
)
low = env.single_action_space.low
high = env.single_action_space.high
print("\n" + "="*50)

thrust_min_a = low[3]
thrust_max_a = high[3]
print(f"Thrust Low from VecDroneRaceEnv:  {thrust_min_a}")
print(f"Thrust High from VecDroneRaceEnv: {thrust_max_a}")
print("\n" + "="*50)

# get action high and low as in control/attitude_rl.py does
drone_params = load_params(config.sim.physics, config.sim.drone_model)
thrust_min_b = drone_params["thrust_min"] * 4  # min total thrust
thrust_max_b = drone_params["thrust_max"] * 4  # max total thrust
print(f"Thrust Low:  {thrust_min_b}")
print(f"Thrust High: {thrust_max_b}")
print("\n" + "="*50)

#if I use the loading method from build_action_space, I get:
params = ForceTorqueParams.load("cf21B_500")
thrust_min, thrust_max = params.thrust_min * 4, params.thrust_max * 4
print(f"Thrust Low:  {thrust_min}")
print(f"Thrust High: {thrust_max}")
print("\n" + "="*50)
