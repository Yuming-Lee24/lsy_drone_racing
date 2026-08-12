"""JIT-compiled GAE and PPO optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import optax
from flax import struct

from lsy_drone_racing.rl_training.networks import gaussian_entropy, squashed_log_probability

if TYPE_CHECKING:
    from collections.abc import Callable

    from flax.training.train_state import TrainState
    from jax import Array

    from lsy_drone_racing.rl_training.config import PPOConfig


@struct.dataclass
class RolloutBatch:
    """Time-major on-policy samples."""

    observation: Array
    raw_action: Array
    log_probability: Array
    value: Array
    reward: Array
    bootstrap_value: Array
    terminated: Array
    truncated: Array


@struct.dataclass
class TrainingBatch:
    """Flattened samples with precomputed targets."""

    observation: Array
    raw_action: Array
    old_log_probability: Array
    old_value: Array
    advantage: Array
    returns: Array


def generalized_advantage_estimate(
    reward: Array,
    value: Array,
    bootstrap_value: Array,
    terminated: Array,
    truncated: Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """Compute GAE without bootstrapping true terminal states or crossing episode boundaries."""
    done = terminated | truncated

    def backward(carry: Array, transition: tuple[Array, ...]) -> tuple[Array, Array]:
        step_reward, step_value, step_bootstrap, step_terminated, step_done = transition
        delta = (
            step_reward
            + gamma * (~step_terminated).astype(jnp.float32) * step_bootstrap
            - step_value
        )
        advantage = delta + gamma * gae_lambda * (~step_done).astype(jnp.float32) * carry
        return advantage, advantage

    _, advantage = jax.lax.scan(
        backward,
        jnp.zeros_like(reward[0]),
        (reward, value, bootstrap_value, terminated, done),
        reverse=True,
    )
    return advantage, advantage + value


def _flatten_rollout(rollout: RolloutBatch, config: PPOConfig) -> TrainingBatch:
    advantage, returns = generalized_advantage_estimate(
        rollout.reward,
        rollout.value,
        rollout.bootstrap_value,
        rollout.terminated,
        rollout.truncated,
        config.gamma,
        config.gae_lambda,
    )
    return TrainingBatch(
        observation=rollout.observation.reshape((config.batch_size, -1)),
        raw_action=rollout.raw_action.reshape((config.batch_size, -1)),
        old_log_probability=rollout.log_probability.reshape(config.batch_size),
        old_value=rollout.value.reshape(config.batch_size),
        advantage=advantage.reshape(config.batch_size),
        returns=returns.reshape(config.batch_size),
    )


def build_optimizer(config: PPOConfig) -> optax.GradientTransformation:
    """Build the clipped Adam optimizer used by all profiles."""
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(config.learning_rate, eps=1e-5)
    )


def make_ppo_update(config: PPOConfig) -> Callable:
    """Create one fully compiled multi-epoch PPO update."""
    minibatch_size = config.minibatch_size

    def loss_function(
        params: Any, apply_fn: Callable, batch: TrainingBatch
    ) -> tuple[Array, dict[str, Array]]:
        mean, log_std, value = apply_fn({"params": params}, batch.observation)
        log_probability = squashed_log_probability(batch.raw_action, mean, log_std)
        log_ratio = log_probability - batch.old_log_probability
        ratio = jnp.exp(log_ratio)
        advantage = batch.advantage
        if config.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        policy_loss = jnp.maximum(
            -advantage * ratio,
            -advantage * jnp.clip(ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef),
        ).mean()
        if config.clip_value_loss:
            value_clipped = batch.old_value + jnp.clip(
                value - batch.old_value, -config.clip_coef, config.clip_coef
            )
            value_loss = (
                0.5
                * jnp.maximum(
                    (value - batch.returns) ** 2, (value_clipped - batch.returns) ** 2
                ).mean()
            )
        else:
            value_loss = 0.5 * jnp.mean((value - batch.returns) ** 2)
        entropy = gaussian_entropy(log_std).mean()
        total_loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        metrics = {
            "loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((ratio - 1.0) - log_ratio),
            "clip_fraction": jnp.mean(jnp.abs(ratio - 1.0) > config.clip_coef),
        }
        return total_loss, metrics

    def update(
        train_state: TrainState, rollout: RolloutBatch, key: Array
    ) -> tuple[TrainState, dict[str, Array]]:
        batch = _flatten_rollout(rollout, config)

        def epoch_step(carry: TrainState, epoch_key: Array) -> tuple[TrainState, dict[str, Array]]:
            state = carry
            permutation = jax.random.permutation(epoch_key, config.batch_size)
            minibatches = permutation.reshape((config.num_minibatches, minibatch_size))

            def minibatch_step(
                inner_state: TrainState, indices: Array
            ) -> tuple[TrainState, dict[str, Array]]:
                minibatch = jax.tree.map(lambda value: value[indices], batch)
                (_, metrics), gradients = jax.value_and_grad(loss_function, has_aux=True)(
                    inner_state.params, inner_state.apply_fn, minibatch
                )
                metrics["gradient_norm"] = optax.tree.norm(gradients)
                return inner_state.apply_gradients(grads=gradients), metrics

            return jax.lax.scan(minibatch_step, state, minibatches)

        epoch_keys = jax.random.split(key, config.update_epochs)
        train_state, metrics = jax.lax.scan(epoch_step, train_state, epoch_keys)
        metrics = jax.tree.map(jnp.mean, metrics)
        target_variance = jnp.var(batch.returns)
        metrics["explained_variance"] = jnp.where(
            target_variance > 0,
            1.0 - jnp.var(batch.returns - batch.old_value) / target_variance,
            jnp.nan,
        )
        return train_state, metrics

    return jax.jit(update)
