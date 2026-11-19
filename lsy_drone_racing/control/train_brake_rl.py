"""PPO training of an emergency-brake policy that stops a fast drone and holds hover.

The policy takes over right after the drone crosses the last gate. Episodes start from randomized
high-speed racing states (speed up to 7 m/s, tilted, rotating) and reward stopping fast with the
shortest horizontal braking distance, then holding a level hover anywhere above the 0.5 m floor.
The distance penalty is the increase of the maximum horizontal excursion from the takeover point,
so it sums to the braking distance and never pulls the drone back. A barrier keeps it above 0.75 m.
Time-limit truncations are bootstrapped with the value of the final observation.

Observations are expressed in the heading (yaw) frame so the policy is invariant to world position
and yaw; see `brake_obs` in `brake_rl.py`, which the deployed controller shares. Actions are
normalized [roll, pitch, collective thrust]; yaw is held at its value when braking starts.

Domain randomization: mass, thrust gain (command-to-force mismatch), 0-1 step action delay,
observation noise and the racing disturbance forces of `config/level3.toml`.

Usage:
    pixi run -e gpu-tests python lsy_drone_racing/control/train_brake_rl.py --total_timesteps 2e8
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import jax
import jax.numpy as jp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.sim.pipeline import insert_fn_before
from crazyflow.utils import leaf_replace
from jax import Array
from jax.scipy.spatial.transform import Rotation as R
from torch.distributions.normal import Normal

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
FLOOR_MARGIN = 0.75
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
    z_min: float = 0.95  # last gate of level3 is a tall gate (center 1.2 m)
    z_max: float = 1.45
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
    dist_coef: float = 5.0
    height_coef: float = 2.0  # per m below FLOOR_MARGIN
    tilt_coef: float = 0.1
    ang_vel_coef: float = 0.02
    smooth_coef: float = 0.05
    fail_penalty: float = 100.0
    # PPO
    total_timesteps: int = 200_000_000
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


class BrakeEnv(DroneEnv):
    """Vectorized braking environment. Actions are normalized [roll, pitch, thrust] in [-1, 1]."""

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
        n = num_envs
        self.key = jax.device_put(jax.random.key(args.seed), self.device)
        self.p0 = jp.zeros((n, 3), device=self.device)
        self.yaw0 = jp.zeros((n,), device=self.device)
        self.d_max = jp.zeros((n,), device=self.device)
        self.thrust_gain = jp.ones((n,), device=self.device)
        self.delay = jp.zeros((n,), dtype=jp.int32, device=self.device)
        self.act_buf = jp.zeros((n, 3, 3), device=self.device)  # [t, t-1, t-2]
        self.hover_act = (
            2
            * (float(params["mass"]) * 9.81 - self.thrust_min)
            / (self.thrust_max - self.thrust_min)
            - 1
        )
        self.obs_dim = N_OBS
        self.record = False  # Return the pre-reset state in `info` (evaluation only)
        self.noisy = {}  # Noisy pos/vel of the last observation, for the evaluation baselines

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

    def _reset(self, mask: Array | None = None, options: dict | None = None) -> None:
        self.sim.reset(mask=mask)
        if mask is None:
            mask = jp.ones((self.num_envs,), dtype=bool, device=self.device)
        self.key, k = jax.random.split(self.key)
        (self.p0, self.yaw0, self.d_max, self.thrust_gain, self.delay, self.act_buf) = (
            self._reset_buffers(
                k,
                mask,
                self.sim.data.states.pos[:, 0],
                self.sim.data.states.quat[:, 0],
                (self.p0, self.yaw0, self.d_max, self.thrust_gain, self.delay, self.act_buf),
                self.hover_act,
                self.args.thrust_gain_delta,
                self.args.max_delay,
            )
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["max_delay"])
    def _reset_buffers(
        key: Array,
        mask: Array,
        pos: Array,
        quat: Array,
        bufs: tuple,
        hover_act: float,
        gain_delta: float,
        max_delay: int,
    ) -> tuple:
        p0, yaw0, d_max, gain, delay, act_buf = bufs
        n = mask.shape[0]
        k1, k2 = jax.random.split(key)
        m = mask[:, None]
        new_gain = jax.random.uniform(k1, (n,), minval=1 - gain_delta, maxval=1 + gain_delta)
        new_delay = jax.random.randint(k2, (n,), 0, max_delay + 1)
        # Before takeover the drone is still flying the racing command; assume hover thrust, level
        new_act = jp.zeros_like(act_buf).at[..., 2].set(hover_act)
        return (
            jp.where(m, pos, p0),
            jp.where(mask, R.from_quat(quat).as_euler("xyz")[:, 2], yaw0),
            jp.where(mask, 0.0, d_max),
            jp.where(mask, new_gain, gain),
            jp.where(mask, new_delay, delay),
            jp.where(mask[:, None, None], new_act, act_buf),
        )

    def step(self, action: Array) -> tuple[Array, Array, Array, Array, dict]:
        """Apply a normalized [roll, pitch, thrust] action with delay and thrust-gain mismatch."""
        action = jp.clip(jp.asarray(action, device=self.device), -1.0, 1.0)
        prev_act = self.act_buf[:, 0]
        self.act_buf = jp.concatenate([action[:, None], self.act_buf[:, :-1]], axis=1)
        applied = self.act_buf[jp.arange(self.num_envs), self.delay]
        cmd = self._to_cmd(applied, self.yaw0, self.thrust_gain, self.thrust_min, self.thrust_max)
        self.sim.attitude_control(cmd[:, None, :])
        self.sim.step(self.n_substeps)
        states = self.sim.data.states
        self.d_max, reward, terminated = self._reward(
            states.pos[:, 0],
            states.quat[:, 0],
            states.vel[:, 0],
            states.ang_vel[:, 0],
            self.p0,
            self.d_max,
            action,
            prev_act,
            self._coefs(),
        )
        truncated = self.truncated()
        # Same-step autoreset: reward/termination use the post-step state, the returned obs is
        # already the first observation of the next episode for finished envs.
        info = {"final_obs": self.obs()}
        if self.record:
            info |= {"stopped": self.stopped(), "state": self.state()}
        self._reset(mask=terminated | truncated)
        return self.obs(), reward, terminated, truncated, info

    @staticmethod
    @jax.jit
    def _to_cmd(act: Array, yaw0: Array, gain: Array, t_min: float, t_max: float) -> Array:
        thrust = (act[:, 2] + 1) / 2 * (t_max - t_min) + t_min
        thrust = jp.clip(thrust * gain, t_min, t_max)
        return jp.stack([act[:, 0] * MAX_TILT_CMD, act[:, 1] * MAX_TILT_CMD, yaw0, thrust], axis=-1)

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
    @jax.jit
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

    def obs(self) -> Array:
        """Heading-frame observation with sensor noise."""
        s = self.sim.data.states
        self.key, k = jax.random.split(self.key)
        obs, pos, vel = self._obs(
            k,
            s.pos[:, 0],
            s.quat[:, 0],
            s.vel[:, 0],
            s.ang_vel[:, 0],
            self.p0,
            self.d_max,
            self.act_buf,
            self.args.pos_noise,
            self.args.vel_noise,
        )
        self.noisy = {"pos": pos, "vel": vel}
        return obs

    @staticmethod
    @jax.jit
    def _obs(
        key: Array,
        pos: Array,
        quat: Array,
        vel: Array,
        ang_vel: Array,
        p0: Array,
        d_max: Array,
        act_buf: Array,
        pos_noise: float,
        vel_noise: float,
    ) -> tuple[Array, Array, Array]:
        k1, k2 = jax.random.split(key)
        pos = pos + pos_noise * jax.random.normal(k1, pos.shape)
        vel = vel + vel_noise * jax.random.normal(k2, vel.shape)
        obs = brake_obs_jax(pos, quat, vel, ang_vel, p0, d_max, act_buf[:, 0], act_buf[:, 1])
        return obs, pos, vel

    def stopped(self) -> Array:
        """Whether each drone satisfies the hand-off condition for the high-level commander."""
        s = self.sim.data.states
        return is_stopped(s.quat[:, 0], s.vel[:, 0], s.ang_vel[:, 0], xp=jp)

    def state(self) -> dict[str, Array]:
        """Raw state used by the evaluation and the analytic baselines."""
        s = self.sim.data.states
        return {
            "pos": s.pos[:, 0],
            "quat": s.quat[:, 0],
            "vel": s.vel[:, 0],
            "ang_vel": s.ang_vel[:, 0],
            "p0": self.p0,
            "yaw0": self.yaw0,
        }


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    """Orthogonal init."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class BrakeAgent(nn.Module):
    """Gaussian MLP actor-critic. The actor mean is tanh-squashed into [-1, 1]."""

    def __init__(self, obs_dim: int = N_OBS, act_dim: int = 3, hidden: int = 128):
        """Init."""
        super().__init__()

        def mlp(out: int, std: float) -> nn.Sequential:
            return nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden, hidden)),
                nn.Tanh(),
                layer_init(nn.Linear(hidden, out), std=std),
            )

        self.critic = mlp(1, 1.0)
        self.actor_mean = nn.Sequential(mlp(act_dim, 0.01), nn.Tanh())
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.7))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Value."""
        return self.critic(x)

    def act(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic action."""
        return self.actor_mean(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action, log-prob, entropy and value."""
        mean = self.actor_mean(x)
        probs = Normal(mean, self.actor_logstd.expand_as(mean).exp())
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def to_torch(x: Array, device: torch.device) -> torch.Tensor:
    """Zero-copy JAX to torch via DLPack."""
    return torch.from_dlpack(x).to(device)


def to_jax(x: torch.Tensor) -> Array:
    """Zero-copy torch to JAX via DLPack."""
    return jax.dlpack.from_dlpack(x.detach().contiguous())


def train(args: Args, out_dir: Path) -> Path:
    """Train with PPO (adapted from cleanrl) and return the final checkpoint path."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = BrakeEnv(args, args.num_envs, device="gpu" if device.type == "cuda" else "cpu")
    agent = BrakeAgent(env.obs_dim, 3, args.hidden).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    batch_size = args.num_envs * args.num_steps
    minibatch = batch_size // args.num_minibatches
    n_iter = args.total_timesteps // batch_size

    obs_buf = torch.zeros((args.num_steps, args.num_envs, env.obs_dim), device=device)
    act_buf = torch.zeros((args.num_steps, args.num_envs, 3), device=device)
    logp_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    val_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs, _ = env.reset(seed=args.seed)
    next_obs = to_torch(next_obs, device)
    next_done = torch.zeros(args.num_envs, device=device)
    ep_ret = torch.zeros(args.num_envs, device=device)
    ep_len = torch.zeros(args.num_envs, device=device)
    fin_ret, fin_len, fin_fail = [], [], []
    t0 = time.time()
    log_path = out_dir / "train_log.csv"
    log_path.write_text("iter,steps,sps,ep_return,ep_len,fail_rate,std,lr,v_loss,approx_kl\n")

    for it in range(1, n_iter + 1):
        if args.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (it - 1.0) / n_iter) * args.learning_rate
        for step in range(args.num_steps):
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            val_buf[step] = value.flatten()
            act_buf[step] = action
            logp_buf[step] = logp
            o, r, term, trunc, info = env.step(to_jax(action))
            next_obs, r = to_torch(o, device), to_torch(r, device)
            term, trunc = to_torch(term, device), to_torch(trunc, device)
            ep_ret += r
            ep_len += 1
            timeout = trunc & ~term
            if timeout.any():  # Bootstrap time-limit truncations with the final state's value
                with torch.no_grad():
                    v_final = agent.get_value(to_torch(info["final_obs"], device)).flatten()
                r = r + args.gamma * v_final * timeout.float()
            rew_buf[step] = r
            next_done = (term | trunc).float()
            if next_done.any():
                d = next_done.bool()
                fin_ret += ep_ret[d].tolist()
                fin_len += ep_len[d].tolist()
                fin_fail += term[d].float().tolist()
                ep_ret[d] = 0
                ep_len[d] = 0

        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            adv = torch.zeros_like(rew_buf)
            lastgae = 0
            for t in reversed(range(args.num_steps)):
                nonterm = 1.0 - (next_done if t == args.num_steps - 1 else done_buf[t + 1])
                nv = next_value if t == args.num_steps - 1 else val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * nonterm - val_buf[t]
                adv[t] = lastgae = delta + args.gamma * args.gae_lambda * nonterm * lastgae
            ret = adv + val_buf

        b_obs = obs_buf.reshape(-1, env.obs_dim)
        b_act, b_logp = act_buf.reshape(-1, 3), logp_buf.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
        v_losses, kls = [], []
        for _ in range(args.update_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch):
                mb = perm[start : start + minibatch]
                _, newlogp, ent, newv = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()
                v_loss = 0.5 * ((newv.view(-1) - b_ret[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * ent.mean() + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                v_losses.append(v_loss.item())
                kls.append(approx_kl.item())

        if it % args.log_every == 0 or it == n_iter:
            sps = int(it * batch_size / (time.time() - t0))
            row = [
                it,
                it * batch_size,
                sps,
                np.mean(fin_ret) if fin_ret else np.nan,
                np.mean(fin_len) if fin_len else np.nan,
                np.mean(fin_fail) if fin_fail else np.nan,
                agent.actor_logstd.exp().mean().item(),
                optimizer.param_groups[0]["lr"],
                np.mean(v_losses),
                np.mean(kls),
            ]
            fin_ret, fin_len, fin_fail = [], [], []
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
            torch.save(agent.state_dict(), out_dir / f"brake_{it}.ckpt")

    path = out_dir / "brake_final.ckpt"
    torch.save(agent.state_dict(), path)
    (out_dir / "args.json").write_text(__import__("json").dumps(asdict(args), indent=2))
    print(f"Saved {path} after {time.time() - t0:.0f}s", flush=True)
    return path


def main(out_dir: str = "runs/brake", **kwargs: float) -> None:
    """Train the brake policy. Any `Args` field can be overridden, e.g. `--total_timesteps 1e6`."""
    fields = Args.__dataclass_fields__
    kwargs = {k: type(fields[k].default)(v) for k, v in kwargs.items()}
    train(Args(**kwargs), Path(out_dir))


if __name__ == "__main__":
    fire.Fire(main)
