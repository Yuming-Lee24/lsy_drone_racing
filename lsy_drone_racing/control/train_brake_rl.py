"""PPO training (pure JAX) of an emergency-brake policy that stops a fast drone and holds hover.

The policy takes over right after the drone crosses the last gate or leaves the safety box.
Episodes start from randomized high-speed states (speed up to 7 m/s, tilted, rotating, takeover
height 0.4-1.5 m) and reward stopping fast with the shortest horizontal braking distance, then
holding a level hover anywhere above the MIN_HEIGHT floor. The distance penalty is the increase of
the maximum horizontal excursion from the takeover point, so it sums to the braking distance and
never pulls the drone back. A barrier penalizes flying below FLOOR_MARGIN. Time-limit truncations
are bootstrapped with the value of the final observation.

Observations are expressed in the heading (yaw) frame so the policy is invariant to world position
and yaw; see `brake_obs` in `brake_rl.py`, which the deployed controller shares. Actions are
normalized [roll, pitch, collective thrust]; yaw is held at its value when braking starts.

Domain randomization: mass, thrust gain (command-to-force mismatch), 0-1 step action delay,
observation noise and the racing disturbance forces of `config/level3.toml`.

The environment step is a pure function (`BrakeEnv.step_fn`), so a whole rollout plus the PPO
update compiles into one XLA program. Checkpoints are `.npz` files whose actor weights (W0..b2)
load directly into `BrakeRL`; copy the final one to `lsy_drone_racing/control/brake_rl.npz` to
deploy it.

Usage:
    pixi run -e gpu python lsy_drone_racing/control/train_brake_rl.py --total_timesteps 2e8
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import fire
import jax
import jax.numpy as jp
import numpy as np
import optax
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim import functional as F
from crazyflow.sim.pipeline import insert_fn_before
from crazyflow.utils import leaf_replace
from jax import Array
from jax.scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.brake_rl import (
    MAX_TILT_CMD,
    MIN_HEIGHT,
    N_OBS,
    brake_obs_jax,
    is_stopped,
)
from lsy_drone_racing.envs.race_core import build_dynamics_disturbance_fn, rng_spec2fn

if TYPE_CHECKING:
    from crazyflow.sim.data import SimData

MAX_HEIGHT = 2.0
FLOOR_MARGIN = 0.35
# Hover rotor speed (RPM) of the cf21B_500 first-principles model, used to spin up rotors on reset
HOVER_RPM = 15900.0


@dataclass
class Args:
    """Training configuration."""

    seed: int = 1
    dynamics: str = "first_principles"
    drone: str = "cf21B_500"
    env_freq: int = 50
    max_episode_time: float = 3.0
    # Initial state distribution
    speed_min: float = 0.5
    speed_max: float = 7.0
    max_elevation: float = 0.3  # rad, vertical angle of the initial velocity
    z_min: float = 0.4
    z_max: float = 1.5
    max_tilt0: float = 0.6  # rad, initial roll/pitch
    max_ang_vel0: float = 3.0  # rad/s
    # Domain randomization
    mass_delta: float = 0.005  # kg
    thrust_gain_delta: float = 0.15
    max_delay: int = 1  # env steps
    pos_noise: float = 0.005
    vel_noise: float = 0.03
    disturbance_force: float = 0.1  # N, as in level3.toml
    # Reward
    speed_coef: float = 0.5
    dist_coef: float = 10.0
    height_coef: float = 2.0  # per m below FLOOR_MARGIN
    tilt_coef: float = 0.1
    ang_vel_coef: float = 0.02
    smooth_coef: float = 0.05
    fail_penalty: float = 100.0
    # PPO
    total_timesteps: int = 1_000_000_000
    num_envs: int = 4096
    num_steps: int = 32
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8
    update_epochs: int = 5
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    hidden: int = 128
    log_every: int = 10
    ckpt_every: int = 200


class EnvState(NamedTuple):
    """All dynamic data of the vectorized braking environment."""

    sim: Any  # SimData
    p0: Array  # Takeover position
    yaw0: Array  # Yaw held during braking
    d_max: Array  # Maximum horizontal excursion from p0 so far
    gain: Array  # Thrust gain mismatch
    delay: Array  # Action delay in env steps
    act_buf: Array  # Actions [t, t-1, t-2]
    key: Array


class BrakeEnv(DroneEnv):
    """Vectorized braking environment. Actions are normalized [roll, pitch, thrust] in [-1, 1].

    `reset_fn`, `obs_fn` and `step_fn` are pure functions of an `EnvState` for compiled training.
    `reset` and `step` wrap them with the usual stateful interface for evaluation and rendering.
    """

    def __init__(self, args: Args, num_envs: int, device: str = "gpu"):
        """Build the simulation, randomization and disturbance pipeline."""
        self.args = args
        dynamics = Dynamics(args.dynamics)
        params = load_params(dynamics, args.drone)
        super().__init__(
            num_envs=num_envs,
            max_episode_time=args.max_episode_time,
            dynamics=dynamics,
            drone=args.drone,
            freq=args.env_freq,
            device=device,
            reset_randomization=self._build_reset_fn(args, dynamics, params),
        )
        self.thrust_min = float(params["thrust_min"]) * 4
        self.thrust_max = float(params["thrust_max"]) * 4
        if args.disturbance_force > 0:
            f = args.disturbance_force
            spec = {"fn": "uniform", "kwargs": {"minval": [-f] * 3, "maxval": [f] * 3}}
            insert_fn_before(
                self.sim.step_pipeline,
                "integration",
                build_dynamics_disturbance_fn(rng_spec2fn(spec)),
            )
            self.sim.build_step_fn()
        self._sim_step, self._sim_reset = self.sim._step, self.sim._reset
        n = num_envs
        self.hover_act = (
            2
            * (float(params["mass"]) * 9.81 - self.thrust_min)
            / (self.thrust_max - self.thrust_min)
            - 1
        )
        zeros = partial(jp.zeros, device=self.device)
        self.s = EnvState(
            sim=self.sim.data,
            p0=zeros((n, 3)),
            yaw0=zeros((n,)),
            d_max=zeros((n,)),
            gain=jp.ones((n,), device=self.device),
            delay=zeros((n,), dtype=jp.int32),
            act_buf=zeros((n, 3, 3)),
            key=jax.device_put(jax.random.key(args.seed), self.device),
        )
        self.obs_dim = N_OBS
        self.record = False  # Unused, kept for the evaluation scripts (info always has the state)
        self.noisy = {}  # Noisy pos/vel of the last observation, for the evaluation baselines
        self._jit_reset = jax.jit(self.reset_fn)
        self._jit_obs = jax.jit(self.obs_fn)
        self._jit_step = jax.jit(self.step_fn)

    @property
    def key(self) -> Array:
        """Rng key of the env (thrust gain, delay, sensor noise)."""
        return self.s.key

    @key.setter
    def key(self, key: Array) -> None:
        self.s = self.s._replace(key=key)

    @staticmethod
    def _build_reset_fn(args: Args, dynamics: Dynamics, params: dict):  # noqa: ANN205
        # Spin up rotors to hover. so_rpy_rotor* store the collective thrust state in rotor_vel[0]
        if dynamics == Dynamics.first_principles:
            rotor0 = HOVER_RPM
        elif "cmd_f_coef" in params:
            rotor0 = float(params["mass"]) * 9.81 / float(params["cmd_f_coef"])
        else:
            rotor0 = 0.0

        def reset_fn(data: SimData, default_data: SimData, mask: Array) -> SimData:
            shape = (data.core.n_worlds, data.core.n_drones)
            key, *ks = jax.random.split(data.core.rng_key, 10)
            pos = jp.stack(
                [
                    jax.random.uniform(ks[0], shape, minval=-1.0, maxval=1.0),
                    jax.random.uniform(ks[1], shape, minval=-1.0, maxval=1.0),
                    jax.random.uniform(ks[2], shape, minval=args.z_min, maxval=args.z_max),
                ],
                axis=-1,
            )
            speed = jax.random.uniform(ks[3], shape, minval=args.speed_min, maxval=args.speed_max)
            az = jax.random.uniform(ks[4], shape, minval=-jp.pi, maxval=jp.pi)
            el = jax.random.uniform(
                ks[5], shape, minval=-args.max_elevation, maxval=args.max_elevation
            )
            vel = speed[..., None] * jp.stack(
                [jp.cos(el) * jp.cos(az), jp.cos(el) * jp.sin(az), jp.sin(el)], axis=-1
            )
            rp = jax.random.uniform(
                ks[6], shape + (2,), minval=-args.max_tilt0, maxval=args.max_tilt0
            )
            yaw = jax.random.uniform(ks[7], shape + (1,), minval=-jp.pi, maxval=jp.pi)
            quat = R.from_euler("xyz", jp.concatenate([rp, yaw], axis=-1)).as_quat()
            ang_vel = jax.random.uniform(
                ks[8], shape + (3,), minval=-args.max_ang_vel0, maxval=args.max_ang_vel0
            )
            key, k_rot, k_mass = jax.random.split(key, 3)
            rotor = rotor0 * jax.random.uniform(k_rot, shape + (4,), minval=0.8, maxval=1.2)
            mass = default_data.params.mass + jax.random.uniform(
                k_mass, shape + (1,), minval=-args.mass_delta, maxval=args.mass_delta
            )
            states = leaf_replace(
                data.states, mask, pos=pos, vel=vel, quat=quat, ang_vel=ang_vel, rotor_vel=rotor
            )
            params = leaf_replace(data.params, mask, mass=mass)
            return data.replace(core=data.core.replace(rng_key=key), states=states, params=params)

        return reset_fn

    # region Pure functions

    def reset_fn(self, s: EnvState, mask: Array) -> EnvState:
        """Reset the masked worlds: new takeover state, randomization and action history."""
        data = self._sim_reset(s.sim, self.sim.default_data, mask)
        pos, quat = data.states.pos[:, 0], data.states.quat[:, 0]
        key, k1, k2 = jax.random.split(s.key, 3)
        n, m = mask.shape[0], mask[:, None]
        gain_delta = self.args.thrust_gain_delta
        new_gain = jax.random.uniform(k1, (n,), minval=1 - gain_delta, maxval=1 + gain_delta)
        new_delay = jax.random.randint(k2, (n,), 0, self.args.max_delay + 1)
        # Before takeover the drone is still flying the racing command; assume hover thrust, level
        new_act = jp.zeros_like(s.act_buf).at[..., 2].set(self.hover_act)
        return EnvState(
            sim=data,
            p0=jp.where(m, pos, s.p0),
            yaw0=jp.where(mask, R.from_quat(quat).as_euler("xyz")[:, 2], s.yaw0),
            d_max=jp.where(mask, 0.0, s.d_max),
            gain=jp.where(mask, new_gain, s.gain),
            delay=jp.where(mask, new_delay, s.delay),
            act_buf=jp.where(mask[:, None, None], new_act, s.act_buf),
            key=key,
        )

    def obs_fn(self, s: EnvState) -> tuple[EnvState, Array, dict[str, Array]]:
        """Heading-frame observation with sensor noise, and the noisy position and velocity."""
        key, k1, k2 = jax.random.split(s.key, 3)
        st = s.sim.states
        pos = st.pos[:, 0] + self.args.pos_noise * jax.random.normal(k1, (self.num_envs, 3))
        vel = st.vel[:, 0] + self.args.vel_noise * jax.random.normal(k2, (self.num_envs, 3))
        obs = brake_obs_jax(
            pos,
            st.quat[:, 0],
            vel,
            st.ang_vel[:, 0],
            s.p0,
            s.d_max,
            s.act_buf[:, 0],
            s.act_buf[:, 1],
        )
        return s._replace(key=key), obs, {"pos": pos, "vel": vel}

    def step_fn(
        self, s: EnvState, action: Array
    ) -> tuple[EnvState, Array, Array, Array, Array, dict]:
        """Apply a normalized [roll, pitch, thrust] action with delay and thrust-gain mismatch.

        Same-step autoreset: reward and termination use the post-step state, the returned obs is
        already the first observation of the next episode for finished envs. `info` holds the
        final observation and the pre-reset state.
        """
        action = jp.clip(action, -1.0, 1.0)
        prev_act = s.act_buf[:, 0]
        act_buf = jp.concatenate([action[:, None], s.act_buf[:, :-1]], axis=1)
        applied = act_buf[jp.arange(self.num_envs), s.delay]
        thrust = (applied[:, 2] + 1) / 2 * (self.thrust_max - self.thrust_min) + self.thrust_min
        thrust = jp.clip(thrust * s.gain, self.thrust_min, self.thrust_max)
        cmd = jp.stack(
            [applied[:, 0] * MAX_TILT_CMD, applied[:, 1] * MAX_TILT_CMD, s.yaw0, thrust], axis=-1
        )
        data = F.attitude_control(s.sim, cmd[:, None, :])
        data = self._sim_step(data, n_steps=self.n_substeps)
        st = data.states
        pos, quat, vel, ang_vel = st.pos[:, 0], st.quat[:, 0], st.vel[:, 0], st.ang_vel[:, 0]
        d_max, reward, terminated = self._reward(
            pos, quat, vel, ang_vel, s.p0, s.d_max, action, prev_act, self._coefs()
        )
        truncated = (data.core.steps / data.core.freq)[:, 0] >= self.max_episode_time
        s = s._replace(sim=data, d_max=d_max, act_buf=act_buf)
        s, final_obs, _ = self.obs_fn(s)
        info = {
            "final_obs": final_obs,
            "stopped": is_stopped(quat, vel, ang_vel, xp=jp),
            "state": {
                "pos": pos,
                "quat": quat,
                "vel": vel,
                "ang_vel": ang_vel,
                "p0": s.p0,
                "yaw0": s.yaw0,
            },
        }
        s = self.reset_fn(s, terminated | truncated)
        s, obs, noisy = self.obs_fn(s)
        info["noisy"] = noisy
        return s, obs, reward, terminated, truncated, info

    def _coefs(self) -> tuple:
        a = self.args
        return (
            a.speed_coef,
            a.dist_coef,
            a.height_coef,
            a.tilt_coef,
            a.ang_vel_coef,
            a.smooth_coef,
            a.fail_penalty,
        )

    @staticmethod
    def _reward(
        pos: Array,
        quat: Array,
        vel: Array,
        ang_vel: Array,
        p0: Array,
        d_max: Array,
        act: Array,
        prev_act: Array,
        coefs: tuple,
    ) -> tuple[Array, Array, Array]:
        c_v, c_d, c_h, c_tilt, c_w, c_s, fail = coefs
        d = jp.linalg.norm(pos[:, :2] - p0[:, :2], axis=-1)
        new_d_max = jp.maximum(d_max, d)
        cos_tilt = R.from_quat(quat).as_matrix()[:, 2, 2]
        r = -c_v * jp.tanh(jp.linalg.norm(vel, axis=-1) / 0.3)
        r -= c_d * (new_d_max - d_max)
        r -= c_h * jp.maximum(FLOOR_MARGIN - pos[:, 2], 0.0)
        r -= c_tilt * (1 - cos_tilt)
        r -= c_w * jp.linalg.norm(ang_vel, axis=-1)
        r -= c_s * jp.sum((act - prev_act) ** 2, axis=-1)
        terminated = (pos[:, 2] < MIN_HEIGHT) | (pos[:, 2] > MAX_HEIGHT) | (cos_tilt < 0)
        r = jp.where(terminated, -fail, r)
        return new_d_max, r, terminated

    # region Stateful interface

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Array, dict]:
        """Reset all worlds. `seed` reseeds the simulation rng (initial states, disturbances)."""
        if seed is not None:
            self.sim.seed(seed)
        s = self._jit_reset(self.s._replace(sim=self.sim.data), jp.ones(self.num_envs, bool))
        s, obs, self.noisy = self._jit_obs(s)
        self.s, self.sim.data = s, s.sim
        return obs, {}

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Step all worlds (see `step_fn`)."""
        action = jp.asarray(action, device=self.device)
        s, obs, reward, terminated, truncated, info = self._jit_step(self.s, action)
        self.s, self.sim.data, self.noisy = s, s.sim, info.pop("noisy")
        return obs, reward, terminated, truncated, info

    def stopped(self) -> Array:
        """Whether each drone satisfies the hand-off condition for the high-level commander."""
        st = self.sim.data.states
        return is_stopped(st.quat[:, 0], st.vel[:, 0], st.ang_vel[:, 0], xp=jp)

    def state(self) -> dict[str, Array]:
        """Raw state used by the evaluation and the analytic baselines."""
        st = self.sim.data.states
        return {
            "pos": st.pos[:, 0],
            "quat": st.quat[:, 0],
            "vel": st.vel[:, 0],
            "ang_vel": st.ang_vel[:, 0],
            "p0": self.s.p0,
            "yaw0": self.s.yaw0,
        }


# region Agent


def init_params(key: Array, hidden: int, obs_dim: int = N_OBS, act_dim: int = 3) -> dict:
    """Orthogonally initialized actor and critic MLPs. Weights are (out, in) as in `BrakeRL`."""

    def mlp(key: Array, out: int, std_out: float) -> list[tuple[Array, Array]]:
        sizes, stds = [obs_dim, hidden, hidden, out], [np.sqrt(2), np.sqrt(2), std_out]
        layers = []
        for k, n_in, n_out, std in zip(jax.random.split(key, 3), sizes[:-1], sizes[1:], stds):
            w = jax.nn.initializers.orthogonal(std)(k, (n_out, n_in), jp.float32)
            layers.append((w, jp.zeros(n_out)))
        return layers

    k_actor, k_critic = jax.random.split(key)
    return {
        "actor": mlp(k_actor, act_dim, 0.01),
        "critic": mlp(k_critic, 1, 1.0),
        "log_std": jp.full(act_dim, -0.7),
    }


def _mlp(layers: list, x: Array) -> Array:
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = jp.tanh(x)
    return x


def actor_mean(params: dict, obs: Array) -> Array:
    """Deterministic action. The output is tanh-squashed into [-1, 1]."""
    return jp.tanh(_mlp(params["actor"], obs))


def value(params: dict, obs: Array) -> Array:
    """State value."""
    return _mlp(params["critic"], obs)[..., 0]


def log_prob(params: dict, obs: Array, action: Array) -> tuple[Array, Array]:
    """Gaussian log-probability of the action and the mean."""
    mean, log_std = actor_mean(params, obs), params["log_std"]
    z = (action - mean) / jp.exp(log_std)
    return jp.sum(-0.5 * z**2 - log_std - 0.5 * jp.log(2 * jp.pi), axis=-1), mean


def save_params(params: dict, path: Path) -> None:
    """Save actor (W0..b2, loadable by `BrakeRL`), critic (C0..c2) and log_std to an `.npz`."""
    arrays = {"log_std": np.asarray(params["log_std"])}
    for i, (w, b) in enumerate(params["actor"]):
        arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(w, np.float64), np.asarray(b, np.float64)
    for i, (w, b) in enumerate(params["critic"]):
        arrays[f"C{i}"], arrays[f"c{i}"] = np.asarray(w), np.asarray(b)
    np.savez(path, **arrays)


# region PPO


class Rollout(NamedTuple):
    """Time-major on-policy samples."""

    obs: Array
    action: Array
    logp: Array
    value: Array
    reward: Array
    done: Array


class Runner(NamedTuple):
    """Carry of the compiled training loop."""

    env: EnvState
    obs: Array
    ep_ret: Array
    ep_len: Array
    key: Array


def make_train_iteration(env: BrakeEnv, args: Args, optimizer: optax.GradientTransformation):  # noqa: ANN201
    """Build the jitted iteration: rollout of `num_steps`, GAE and the multi-epoch PPO update.

    Returns `iteration(runner, params, opt_state) -> (runner, params, opt_state, metrics)`.
    """
    batch_size = args.num_envs * args.num_steps
    minibatch = batch_size // args.num_minibatches

    def env_step(carry: tuple, _: None) -> tuple[tuple, Rollout]:
        runner, params, log = carry
        key, k = jax.random.split(runner.key)
        mean = actor_mean(params, runner.obs)
        action = mean + jp.exp(params["log_std"]) * jax.random.normal(k, mean.shape)
        logp, _ = log_prob(params, runner.obs, action)
        v = value(params, runner.obs)
        s, obs, r, term, trunc, info = env.step_fn(runner.env, action)
        done = term | trunc
        ep_ret, ep_len = runner.ep_ret + r, runner.ep_len + 1
        # Bootstrap time-limit truncations with the value of the final observation
        r = r + args.gamma * value(params, info["final_obs"]) * (trunc & ~term)
        log = (
            log[0] + jp.sum(jp.where(done, ep_ret, 0.0)),
            log[1] + jp.sum(jp.where(done, ep_len, 0.0)),
            log[2] + jp.sum(term),
            log[3] + jp.sum(done),
        )
        sample = Rollout(runner.obs, action, logp, v, r, done)
        runner = Runner(s, obs, jp.where(done, 0.0, ep_ret), jp.where(done, 0.0, ep_len), key)
        return (runner, params, log), sample

    def gae(rollout: Rollout, last_value: Array) -> tuple[Array, Array]:
        def backward(carry: tuple, t: tuple) -> tuple[tuple, Array]:
            adv_next, v_next = carry
            r, v, done = t
            nonterm = 1.0 - done
            delta = r + args.gamma * v_next * nonterm - v
            adv = delta + args.gamma * args.gae_lambda * nonterm * adv_next
            return (adv, v), adv

        init = (jp.zeros_like(last_value), last_value)
        _, adv = jax.lax.scan(
            backward,
            init,
            (rollout.reward, rollout.value, rollout.done.astype(jp.float32)),
            reverse=True,
        )
        return adv, adv + rollout.value

    def loss_fn(params: dict, mb: tuple) -> tuple[Array, tuple]:
        obs, action, old_logp, adv, ret = mb
        logp, _ = log_prob(params, obs, action)
        log_ratio = logp - old_logp
        ratio = jp.exp(log_ratio)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        clipped = jp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jp.maximum(-adv * ratio, -adv * clipped).mean()
        v_loss = 0.5 * jp.mean((value(params, obs) - ret) ** 2)
        entropy = jp.sum(0.5 + 0.5 * jp.log(2 * jp.pi) + params["log_std"])
        loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss
        return loss, (v_loss, jp.mean((ratio - 1) - log_ratio))

    def update_minibatch(carry: tuple, mb: tuple) -> tuple[tuple, tuple]:
        params, opt_state = carry
        grads, aux = jax.grad(loss_fn, has_aux=True)(params, mb)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), aux

    def update_epoch(carry: tuple, key: Array) -> tuple[tuple, tuple]:
        params, opt_state, batch = carry
        perm = jax.random.permutation(key, batch_size)
        mbs = jax.tree.map(
            lambda x: x[perm].reshape((args.num_minibatches, minibatch) + x.shape[1:]), batch
        )
        (params, opt_state), aux = jax.lax.scan(update_minibatch, (params, opt_state), mbs)
        return (params, opt_state, batch), aux

    @jax.jit
    def iteration(runner: Runner, params: dict, opt_state: Any) -> tuple:
        log0 = (jp.zeros(()), jp.zeros(()), jp.zeros(()), jp.zeros(()))
        (runner, _, log), rollout = jax.lax.scan(
            env_step, (runner, params, log0), length=args.num_steps
        )
        adv, ret = gae(rollout, value(params, runner.obs))
        flat = lambda x: x.reshape((batch_size,) + x.shape[2:])  # noqa: E731
        batch = (flat(rollout.obs), flat(rollout.action), flat(rollout.logp), flat(adv), flat(ret))
        key, k = jax.random.split(runner.key)
        (params, opt_state, _), (v_loss, kl) = jax.lax.scan(
            update_epoch, (params, opt_state, batch), jax.random.split(k, args.update_epochs)
        )
        metrics = {
            "ret_sum": log[0],
            "len_sum": log[1],
            "fail_sum": log[2],
            "n_done": log[3],
            "v_loss": v_loss.mean(),
            "approx_kl": kl.mean(),
        }
        return runner._replace(key=key), params, opt_state, metrics

    return iteration


def train(args: Args, out_dir: Path) -> Path:
    """Train with PPO (structure and hyperparameters of cleanrl) and return the final checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "gpu" if jax.default_backend() == "gpu" else "cpu"
    env = BrakeEnv(args, args.num_envs, device=device)
    batch_size = args.num_envs * args.num_steps
    n_iter = args.total_timesteps // batch_size
    updates_per_iter = args.update_epochs * args.num_minibatches

    def lr(count: Array) -> Array:
        if not args.anneal_lr:
            return jp.asarray(args.learning_rate)
        return args.learning_rate * (1.0 - (count // updates_per_iter) / n_iter)

    optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(lr, eps=1e-5))
    key = jax.random.key(args.seed)
    key, k_params = jax.random.split(key)
    params = init_params(k_params, args.hidden)
    opt_state = optimizer.init(params)
    obs, _ = env.reset(seed=args.seed)
    n = args.num_envs
    runner = Runner(env.s, obs, jp.zeros(n), jp.zeros(n), key)
    iteration = make_train_iteration(env, args, optimizer)

    log_path = out_dir / "train_log.csv"
    log_path.write_text("iter,steps,sps,ep_return,ep_len,fail_rate,std,lr,v_loss,approx_kl\n")
    (out_dir / "args.json").write_text(json.dumps(asdict(args), indent=2))
    acc_keys = ("ret_sum", "len_sum", "fail_sum", "n_done")
    acc = {k: jp.zeros(()) for k in acc_keys}  # Episode statistics summed on the device
    t0 = time.time()
    for it in range(1, n_iter + 1):
        runner, params, opt_state, metrics = iteration(runner, params, opt_state)
        acc = {k: acc[k] + metrics[k] for k in acc_keys}
        if it == 1:
            jax.block_until_ready(params)
            t_compile = time.time() - t0
            print(f"Compiled and ran the first iteration in {t_compile:.1f}s", flush=True)
            t0, it0 = time.time(), it
        if it % args.log_every == 0 or it == n_iter:
            m, a = jax.device_get((metrics, acc))
            done = max(float(a["n_done"]), 1.0)
            sps = int((it - it0) * batch_size / max(time.time() - t0, 1e-9))
            row = [
                it,
                it * batch_size,
                sps,
                float(a["ret_sum"]) / done,
                float(a["len_sum"]) / done,
                float(a["fail_sum"]) / done,
                float(np.exp(np.asarray(params["log_std"])).mean()),
                float(args.learning_rate * (1.0 - (it - 1) / n_iter)),
                float(m["v_loss"]),
                float(m["approx_kl"]),
            ]
            acc = {k: jp.zeros(()) for k in acc_keys}
            with log_path.open("a") as f:
                f.write(
                    ",".join(f"{v:.5g}" if isinstance(v, float) else str(v) for v in row) + "\n"
                )
            print(
                f"it {it}/{n_iter} steps {row[1]:.3g} sps {sps} ret {row[3]:.2f} "
                f"len {row[4]:.1f} fail {row[5]:.3f} std {row[6]:.3f}",
                flush=True,
            )
        if it % args.ckpt_every == 0:
            save_params(params, out_dir / f"brake_{it}.npz")

    path = out_dir / "brake_final.npz"
    save_params(params, path)
    print(f"Saved {path} after {time.time() - t0:.0f}s (+{t_compile:.0f}s compile)", flush=True)
    return path


def main(out_dir: str = "runs/brake", **kwargs: float) -> None:
    """Train the brake policy. Any `Args` field can be overridden, e.g. `--total_timesteps 1e6`."""
    fields = Args.__dataclass_fields__
    kwargs = {k: type(fields[k].default)(v) for k, v in kwargs.items()}
    train(Args(**kwargs), Path(out_dir))


if __name__ == "__main__":
    fire.Fire(main)
