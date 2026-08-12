"""Eager PyTorch reference for the same Level 3 structured PPO benchmark."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import Tensor, nn

from lsy_drone_racing.rl_training.environment import hover_action_bias
from lsy_drone_racing.rl_training.level3_config import get_level3_profile
from lsy_drone_racing.rl_training.level3_environment import (
    initialize_level3_runner,
    make_level3_functional_step,
)
from lsy_drone_racing.rl_training.level3_runtime import create_level3_environment
from lsy_drone_racing.rl_training.runtime import PROJECT_ROOT

if TYPE_CHECKING:
    from typing import Any

LOG_2PI = math.log(2.0 * math.pi)


def _layer(layer: nn.Linear, scale: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, scale)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class TorchEntityEncoder(nn.Module):
    """Shared two-layer entity encoder."""

    def __init__(self, input_dim: int, width: int):
        """Create the encoder."""
        super().__init__()
        self.network = nn.Sequential(
            _layer(nn.Linear(input_dim, width)),
            nn.Tanh(),
            _layer(nn.Linear(width, width)),
            nn.Tanh(),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        """Encode each entity independently with shared weights."""
        return self.network(tokens)


class TorchEntityAttention(nn.Module):
    """Parameter-free ego-query pooling matching the Flax module."""

    def __init__(self, width: int, heads: int):
        """Record the static head layout."""
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads

    def forward(self, query: Tensor, entities: Tensor, valid: Tensor) -> Tensor:
        """Pool valid entities around one ego query."""
        batch, count = entities.shape[:2]
        q = query.reshape(batch, self.heads, self.head_width)
        entity_heads = entities.reshape(batch, count, self.heads, self.head_width)
        logits = torch.einsum("bhd,bkhd->bhk", q, entity_heads) / math.sqrt(self.head_width)
        logits = torch.where(valid[:, None], logits, torch.full_like(logits, -1e9))
        weights = torch.softmax(logits, dim=-1)
        context = torch.einsum("bhk,bkhd->bhd", weights, entity_heads).reshape(batch, self.width)
        return torch.tanh(context + query)


class TorchBeliefEncoder(nn.Module):
    """Torch equivalent of the Actor or critic-belief encoder."""

    def __init__(self, entity_dim: int, heads: int):
        """Create semantic encoders and attention pools."""
        super().__init__()
        self.ego = nn.Sequential(
            _layer(nn.Linear(28, entity_dim)),
            nn.Tanh(),
            _layer(nn.Linear(entity_dim, entity_dim)),
            nn.Tanh(),
        )
        self.history = nn.Sequential(
            _layer(nn.Linear(64, entity_dim)),
            nn.Tanh(),
            _layer(nn.Linear(entity_dim, entity_dim)),
            nn.Tanh(),
        )
        self.gates = TorchEntityEncoder(23, entity_dim)
        self.obstacles = TorchEntityEncoder(15, entity_dim)
        self.gate_attention = TorchEntityAttention(entity_dim, heads)
        self.obstacle_attention = TorchEntityAttention(entity_dim, heads)

    def forward(self, observation: Any) -> Tensor:
        """Return fused belief features with the direct geometry skip."""
        ego = self.ego(observation.ego)
        history = self.history(observation.history.flatten(start_dim=1))
        gates = self.gates(observation.gate_tokens)
        obstacles = self.obstacles(observation.obstacle_tokens)
        gate_context = self.gate_attention(
            ego,
            gates,
            torch.ones(observation.gate_tokens.shape[:2], dtype=torch.bool, device=ego.device),
        )
        obstacle_context = self.obstacle_attention(
            ego, obstacles, observation.obstacle_tokens[..., -1] > 0.5
        )
        return torch.cat(
            (ego, history, gate_context, obstacle_context, observation.direct_geometry), dim=-1
        )


class TorchLevel3ActorCritic(nn.Module):
    """Eager asymmetric network matching the production structured architecture."""

    def __init__(
        self, entity_dim: int, heads: int, fusion_dim: int, action_bias: tuple[float, ...]
    ):
        """Create independent actor and privileged critic parameter trees."""
        super().__init__()
        belief_width = entity_dim * 4 + 31
        privileged_width = entity_dim * 2 + 31 + 8
        self.entity_dim = entity_dim
        self.actor_encoder = TorchBeliefEncoder(entity_dim, heads)
        self.critic_belief_encoder = TorchBeliefEncoder(entity_dim, heads)
        self.true_gates = TorchEntityEncoder(23, entity_dim)
        self.true_obstacles = TorchEntityEncoder(15, entity_dim)
        self.true_gate_attention = TorchEntityAttention(entity_dim, heads)
        self.true_obstacle_attention = TorchEntityAttention(entity_dim, heads)
        self.actor_head = nn.Sequential(
            _layer(nn.Linear(belief_width, fusion_dim)),
            nn.Tanh(),
            _layer(nn.Linear(fusion_dim, fusion_dim)),
            nn.Tanh(),
            _layer(nn.Linear(fusion_dim, 4), 0.01),
        )
        self.critic_head = nn.Sequential(
            _layer(nn.Linear(belief_width + privileged_width, fusion_dim)),
            nn.Tanh(),
            _layer(nn.Linear(fusion_dim, fusion_dim)),
            nn.Tanh(),
            _layer(nn.Linear(fusion_dim, 1), 1.0),
        )
        self.log_std = nn.Parameter(torch.tensor([-1.0, -1.0, -1.0, -0.5]))
        with torch.no_grad():
            self.actor_head[-1].bias.copy_(torch.asarray(action_bias))

    def forward(self, actor_observation: Any, critic_observation: Any) -> tuple[Tensor, ...]:
        """Return mean, log standard deviation, and privileged value."""
        actor_belief = self.actor_encoder(actor_observation)
        mean = self.actor_head(actor_belief)
        critic_belief = self.critic_belief_encoder(actor_observation)
        query = critic_belief[:, : self.entity_dim]
        gates = self.true_gates(critic_observation.true_gate_tokens)
        obstacles = self.true_obstacles(critic_observation.true_obstacle_tokens)
        gate_context = self.true_gate_attention(
            query,
            gates,
            torch.ones(
                critic_observation.true_gate_tokens.shape[:2], dtype=torch.bool, device=query.device
            ),
        )
        obstacle_context = self.true_obstacle_attention(
            query, obstacles, critic_observation.true_obstacle_tokens[..., -1] > 0.5
        )
        privileged = torch.cat(
            (
                gate_context,
                obstacle_context,
                critic_observation.true_direct_geometry,
                critic_observation.dynamics,
            ),
            dim=-1,
        )
        value = self.critic_head(torch.cat((critic_belief, privileged), dim=-1)).squeeze(-1)
        return mean, self.log_std.expand_as(mean), value


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


def _torch_tree(tree: Any) -> Any:
    return jax.tree.map(_torch, tree)


def _allocate_rollout(tree: Any, steps: int, device: torch.device) -> Any:
    return jax.tree.map(
        lambda value: torch.empty((steps, *value.shape), dtype=value.dtype, device=device), tree
    )


def _copy_tree_step(destination: Any, source: Any, step: int) -> None:
    for target, value in zip(jax.tree.leaves(destination), jax.tree.leaves(source), strict=True):
        target[step].copy_(value)


def reference_update(
    runner: Any,
    environment_step: Any,
    model: TorchLevel3ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: Any,
    device: torch.device,
) -> tuple[Any, Tensor]:
    """Collect one eager DLPack rollout and apply equivalent PPO minibatches."""
    actor_observations = _allocate_rollout(
        _torch_tree(runner.actor_observation), config.num_steps, device
    )
    critic_observations = _allocate_rollout(
        _torch_tree(runner.critic_observation), config.num_steps, device
    )
    raw_actions = torch.empty((config.num_steps, config.num_envs, 4), device=device)
    log_probabilities = torch.empty((config.num_steps, config.num_envs), device=device)
    values = torch.empty((config.num_steps, config.num_envs), device=device)
    rewards = torch.empty((config.num_steps, config.num_envs), device=device)
    bootstrap_values = torch.empty((config.num_steps, config.num_envs), device=device)
    terminated = torch.empty((config.num_steps, config.num_envs), dtype=torch.bool, device=device)
    truncated = torch.empty((config.num_steps, config.num_envs), dtype=torch.bool, device=device)
    completed_episodes = torch.zeros((), device=device)

    for step in range(config.num_steps):
        actor = _torch_tree(runner.actor_observation)
        critic = _torch_tree(runner.critic_observation)
        with torch.no_grad():
            mean, log_std, value = model(actor, critic)
            raw_action = mean + torch.exp(log_std) * torch.randn_like(mean)
            action = torch.tanh(raw_action)
            log_probability = _log_probability(raw_action, mean, log_std)
        transition = environment_step(runner, _jax(action))
        terminal_actor = _torch_tree(transition.terminal_actor_observation)
        terminal_critic = _torch_tree(transition.terminal_critic_observation)
        with torch.no_grad():
            _, _, bootstrap_value = model(terminal_actor, terminal_critic)
        _copy_tree_step(actor_observations, actor, step)
        _copy_tree_step(critic_observations, critic, step)
        raw_actions[step] = raw_action
        log_probabilities[step] = log_probability
        values[step] = value
        rewards[step] = _torch(transition.reward)
        bootstrap_values[step] = bootstrap_value
        terminated[step] = _torch(transition.terminated)
        truncated[step] = _torch(transition.truncated)
        completed_episodes += _torch(transition.metrics.done).sum()
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

    batch_actor = jax.tree.map(lambda value: value.flatten(0, 1), actor_observations)
    batch_critic = jax.tree.map(lambda value: value.flatten(0, 1), critic_observations)
    batch_action = raw_actions.flatten(0, 1)
    batch_log_probability = log_probabilities.flatten()
    batch_value = values.flatten()
    batch_advantage = advantages.flatten()
    batch_return = returns.flatten()
    for _ in range(config.update_epochs):
        permutation = torch.randperm(config.batch_size, device=device)
        epoch_kl = torch.zeros((), device=device)
        for start in range(0, config.batch_size, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
            actor = jax.tree.map(lambda value: value[indices], batch_actor)
            critic = jax.tree.map(lambda value: value[indices], batch_critic)
            mean, log_std, value = model(actor, critic)
            new_log_probability = _log_probability(batch_action[indices], mean, log_std)
            log_ratio = new_log_probability - batch_log_probability[indices]
            ratio = torch.exp(log_ratio)
            advantage = batch_advantage[indices]
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
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
            epoch_kl += ((ratio - 1.0) - log_ratio).mean().detach()
        if bool(epoch_kl / config.num_minibatches > config.target_kl):
            break
    return runner, completed_episodes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT / "saves" / "jax_ppo" / "performance" / "level3_v2_final_torch_10m.json"
        ),
    )
    return parser


def main() -> None:
    """Benchmark eager Torch against the identical Level 3 transition and architecture."""
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The Level 3 PyTorch reference requires CUDA")
    config = get_level3_profile().with_overrides(
        device="gpu",
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        total_timesteps=max(args.timesteps, args.num_envs * args.num_steps),
    )
    torch.manual_seed(config.seed)
    device = torch.device("cuda")
    env = create_level3_environment(config)
    runner = initialize_level3_runner(env, config)
    environment_step = make_level3_functional_step(env, config)
    low = jnp.asarray(env.single_action_space.low, dtype=jnp.float32)
    high = jnp.asarray(env.single_action_space.high, dtype=jnp.float32)
    nominal_mass = float(np.asarray(env.drone_mass).reshape(-1)[0])
    model = TorchLevel3ActorCritic(
        config.entity_dim,
        config.attention_heads,
        config.fusion_dim,
        hover_action_bias(low, high, nominal_mass * 9.81),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate_start, eps=1e-5)

    compile_start = time.perf_counter()
    runner, completed_episodes = reference_update(
        runner, environment_step, model, optimizer, config, device
    )
    torch.cuda.synchronize()
    compile_and_first_seconds = time.perf_counter() - compile_start
    torch.cuda.reset_peak_memory_stats()

    repeats = max(2, math.ceil(args.timesteps / config.batch_size))
    start = time.perf_counter()
    for _ in range(repeats):
        runner, completed_episodes = reference_update(
            runner, environment_step, model, optimizer, config, device
        )
    completed_episodes.cpu()
    elapsed = time.perf_counter() - start
    measured_steps = repeats * config.batch_size
    report = {
        "implementation": "same-level3-v2-eager-pytorch-reference",
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
