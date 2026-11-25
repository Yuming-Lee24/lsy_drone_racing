"""Local PyTorch PPO reference used only for the JAX performance A/B gate.

This module intentionally does not participate in training, evaluation, or deployment. It keeps
the eager policy/JAX-environment boundary from the 25ws implementation while using the same
Crazyflow 0.4 transition, observation, reward, and PPO parameters as the fused JAX experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import Tensor, nn

from lsy_drone_racing.rl_training.config import get_profile
from lsy_drone_racing.rl_training.environment import (
    hover_action_bias,
    initialize_runner,
    make_functional_step,
)
from lsy_drone_racing.rl_training.runtime import PROJECT_ROOT, create_environment

LOG_2PI = math.log(2.0 * math.pi)


def _layer(layer: nn.Linear, scale: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, scale)
    nn.init.zeros_(layer.bias)
    return layer


class TorchActorCritic(nn.Module):
    """Eager Torch equivalent of the production Flax actor and critic."""

    def __init__(
        self,
        observation_dim: int = 88,
        hidden_dim: int = 256,
        action_bias: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    ):
        """Initialize independent two-layer actor and critic MLPs."""
        super().__init__()
        self.actor = nn.Sequential(
            _layer(nn.Linear(observation_dim, hidden_dim)),
            nn.Tanh(),
            _layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer(nn.Linear(hidden_dim, 4), 0.01),
        )
        self.critic = nn.Sequential(
            _layer(nn.Linear(observation_dim, hidden_dim)),
            nn.Tanh(),
            _layer(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer(nn.Linear(hidden_dim, 1), 1.0),
        )
        self.log_std = nn.Parameter(torch.tensor([-1.0, -1.0, -1.0, -0.5]))
        with torch.no_grad():
            self.actor[-1].bias.copy_(torch.asarray(action_bias))

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return unsquashed action mean, expanded log standard deviation, and value."""
        mean = self.actor(observation)
        return mean, self.log_std.expand_as(mean), self.critic(observation).squeeze(-1)


def _log_probability(raw_action: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    inverse_variance = torch.exp(-2.0 * log_std)
    base = -0.5 * ((raw_action - mean).square() * inverse_variance + 2.0 * log_std + LOG_2PI)
    action = torch.tanh(raw_action)
    correction = torch.log(torch.clamp(1.0 - action.square(), min=1e-6))
    return (base - correction).sum(dim=-1)


def _torch(array: jax.Array) -> Tensor:
    return torch.utils.dlpack.from_dlpack(array)


def _jax(tensor: Tensor) -> jax.Array:
    return jax.dlpack.from_dlpack(tensor.detach())


def reference_update(
    runner: object,
    environment_step: object,
    model: TorchActorCritic,
    optimizer: torch.optim.Optimizer,
    config: object,
    device: torch.device,
) -> tuple[object, Tensor]:
    """Collect one eager rollout and apply the same PPO epochs/minibatches as JAX."""
    observations = torch.empty((config.num_steps, config.num_envs, 88), device=device)
    raw_actions = torch.empty((config.num_steps, config.num_envs, 4), device=device)
    log_probabilities = torch.empty((config.num_steps, config.num_envs), device=device)
    values = torch.empty((config.num_steps, config.num_envs), device=device)
    rewards = torch.empty((config.num_steps, config.num_envs), device=device)
    bootstrap_values = torch.empty((config.num_steps, config.num_envs), device=device)
    terminated = torch.empty((config.num_steps, config.num_envs), dtype=torch.bool, device=device)
    truncated = torch.empty((config.num_steps, config.num_envs), dtype=torch.bool, device=device)
    episode_count = torch.zeros((), device=device)
    success_count = torch.zeros((), device=device)
    collision_count = torch.zeros((), device=device)
    completed_reward = torch.zeros((), device=device)
    completed_length = torch.zeros((), device=device)

    for step in range(config.num_steps):
        observation = _torch(runner.observation)
        with torch.no_grad():
            mean, log_std, value = model(observation)
            raw_action = mean + torch.exp(log_std) * torch.randn_like(mean)
            action = torch.tanh(raw_action)
            log_probability = _log_probability(raw_action, mean, log_std)
        transition = environment_step(runner, _jax(action))
        with torch.no_grad():
            _, _, bootstrap_value = model(_torch(transition.terminal_observation))
        observations[step] = observation
        raw_actions[step] = raw_action
        log_probabilities[step] = log_probability
        values[step] = value
        rewards[step] = _torch(transition.reward)
        bootstrap_values[step] = bootstrap_value
        terminated[step] = _torch(transition.terminated)
        truncated[step] = _torch(transition.truncated)
        episode_count += _torch(transition.metrics.done).sum()
        success_count += _torch(transition.metrics.success).sum()
        collision_count += _torch(transition.metrics.collision).sum()
        completed_reward += _torch(transition.metrics.episode_components).sum()
        completed_length += _torch(transition.metrics.episode_length).sum()
        runner = transition.next_runner

    advantages = torch.zeros_like(rewards)
    carry = torch.zeros(config.num_envs, device=device)
    done = terminated | truncated
    for step in reversed(range(config.num_steps)):
        delta = (
            rewards[step]
            + config.gamma * (~terminated[step]).float() * bootstrap_values[step]
            - values[step]
        )
        carry = delta + config.gamma * config.gae_lambda * (~done[step]).float() * carry
        advantages[step] = carry
    returns = advantages + values

    batch_observation = observations.flatten(0, 1)
    batch_action = raw_actions.flatten(0, 1)
    batch_log_probability = log_probabilities.flatten()
    batch_value = values.flatten()
    batch_advantage = advantages.flatten()
    batch_return = returns.flatten()
    for _ in range(config.update_epochs):
        permutation = torch.randperm(config.batch_size, device=device)
        for start in range(0, config.batch_size, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            mean, log_std, value = model(batch_observation[indices])
            new_log_probability = _log_probability(batch_action[indices], mean, log_std)
            ratio = torch.exp(new_log_probability - batch_log_probability[indices])
            advantage = batch_advantage[indices]
            if config.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            policy_loss = torch.maximum(
                -advantage * ratio,
                -advantage * torch.clamp(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef),
            ).mean()
            value_clipped = batch_value[indices] + torch.clamp(
                value - batch_value[indices], -config.clip_coef, config.clip_coef
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (value - batch_return[indices]).square(),
                    (value_clipped - batch_return[indices]).square(),
                ).mean()
            )
            entropy = (log_std + 0.5 * (1.0 + LOG_2PI)).sum(dim=-1).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    aggregate_metrics = torch.stack(
        (episode_count, success_count, collision_count, completed_reward, completed_length)
    )
    return runner, aggregate_metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("fast", "safe_fast"), default="fast")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=96)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=12)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "saves" / "jax_ppo" / "torch_reference.json"
    )
    return parser


def main() -> None:
    """Benchmark the adapted 25ws eager Torch/JAX boundary."""
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The PyTorch reference requires CUDA")
    config = get_profile(args.profile).with_overrides(
        device="gpu",
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        total_timesteps=max(args.timesteps, args.num_envs * args.num_steps),
    )
    torch.manual_seed(config.seed)
    device = torch.device("cuda")
    env = create_environment(config)
    runner = initialize_runner(env, config)
    environment_step = make_functional_step(env, config)
    low = jnp.asarray(env.single_action_space.low, dtype=jnp.float32)
    high = jnp.asarray(env.single_action_space.high, dtype=jnp.float32)
    nominal_mass = float(np.asarray(env.drone_mass).reshape(-1)[0])
    model = TorchActorCritic(
        hidden_dim=config.hidden_dim, action_bias=hover_action_bias(low, high, nominal_mass * 9.81)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)

    compile_start = time.perf_counter()
    runner, aggregate_metrics = reference_update(
        runner, environment_step, model, optimizer, config, device
    )
    torch.cuda.synchronize()
    compile_and_first_seconds = time.perf_counter() - compile_start
    torch.cuda.reset_peak_memory_stats()

    repeats = max(2, math.ceil(args.timesteps / config.batch_size))
    start = time.perf_counter()
    for _ in range(repeats):
        runner, aggregate_metrics = reference_update(
            runner, environment_step, model, optimizer, config, device
        )
    aggregate_metrics.cpu()
    elapsed = time.perf_counter() - start
    measured_steps = repeats * config.batch_size
    report = {
        "implementation": "adapted-25ws-main-pytorch-reference",
        "config": asdict(config),
        "compile_and_first_update_seconds": compile_and_first_seconds,
        "measured_environment_steps": measured_steps,
        "steady_seconds": elapsed,
        "end_to_end_sps": measured_steps / elapsed,
        "gpu_memory_peak_mib": torch.cuda.max_memory_allocated() / (1024**2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
