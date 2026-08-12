"""Compiled GAE and target-KL PPO update for Level 3 V2."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from lsy_drone_racing.rl_training.networks import gaussian_entropy, squashed_log_probability

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from lsy_drone_racing.rl_training.level3_config import Level3PPOConfig
    from lsy_drone_racing.rl_training.level3_observations import ActorObservation, CriticObservation


class Level3TrainState(TrainState):
    """Flax optimizer state plus an evaluation-only parameter EMA."""

    ema_params: Any

    @classmethod
    def create(
        cls, *, apply_fn: Callable, params: Any, tx: optax.GradientTransformation
    ) -> Level3TrainState:
        """Create optimizer and EMA state from the same initial parameters."""
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            ema_params=jax.tree.map(lambda value: value.copy(), params),
        )


@struct.dataclass
class Level3RolloutBatch:
    """Time-major on-policy samples for the asymmetric model."""

    actor_observation: ActorObservation
    critic_observation: CriticObservation
    raw_action: Array
    log_probability: Array
    value: Array
    reward: Array
    bootstrap_value: Array
    terminated: Array
    truncated: Array


@struct.dataclass
class Level3TrainingBatch:
    """Flattened V2 samples with precomputed advantage targets."""

    actor_observation: ActorObservation
    critic_observation: CriticObservation
    raw_action: Array
    old_log_probability: Array
    old_value: Array
    advantage: Array
    returns: Array


def level3_generalized_advantage_estimate(
    reward: Array,
    value: Array,
    bootstrap_value: Array,
    terminated: Array,
    truncated: Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """Compute GAE while respecting true terminal and time-limit boundaries."""
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


def _flatten_rollout(rollout: Level3RolloutBatch, config: Level3PPOConfig) -> Level3TrainingBatch:
    advantage, returns = level3_generalized_advantage_estimate(
        rollout.reward,
        rollout.value,
        rollout.bootstrap_value,
        rollout.terminated,
        rollout.truncated,
        config.gamma,
        config.gae_lambda,
    )

    def flatten(value: Array) -> Array:
        return value.reshape((config.batch_size, *value.shape[2:]))

    return Level3TrainingBatch(
        actor_observation=jax.tree.map(flatten, rollout.actor_observation),
        critic_observation=jax.tree.map(flatten, rollout.critic_observation),
        raw_action=flatten(rollout.raw_action),
        old_log_probability=rollout.log_probability.reshape(config.batch_size),
        old_value=rollout.value.reshape(config.batch_size),
        advantage=advantage.reshape(config.batch_size),
        returns=returns.reshape(config.batch_size),
    )


def build_level3_optimizer(config: Level3PPOConfig) -> optax.GradientTransformation:
    """Build clipped Adam with a fixed-budget linear learning-rate decay."""
    learning_rate = optax.linear_schedule(
        init_value=config.learning_rate_start,
        end_value=config.learning_rate_end,
        transition_steps=max(config.maximum_optimizer_steps, 1),
    )
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm), optax.adam(learning_rate, eps=1e-5)
    )


def make_level3_ppo_update(config: Level3PPOConfig) -> Callable:
    """Return a compiled PPO update with epoch-boundary target-KL stopping."""
    minibatch_size = config.minibatch_size
    zero_metrics = {
        "loss": jnp.asarray(0.0),
        "policy_loss": jnp.asarray(0.0),
        "value_loss": jnp.asarray(0.0),
        "entropy": jnp.asarray(0.0),
        "approx_kl": jnp.asarray(0.0),
        "clip_fraction": jnp.asarray(0.0),
        "gradient_norm": jnp.asarray(0.0),
    }

    def loss_function(
        params: Any, apply_fn: Callable, batch: Level3TrainingBatch
    ) -> tuple[Array, dict[str, Array]]:
        mean, log_std, value = apply_fn(
            {"params": params}, batch.actor_observation, batch.critic_observation
        )
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
        train_state: Level3TrainState, rollout: Level3RolloutBatch, key: Array
    ) -> tuple[Level3TrainState, dict[str, Array]]:
        batch = _flatten_rollout(rollout, config)

        def run_epoch(
            state_and_key: tuple[Level3TrainState, Array],
        ) -> tuple[Level3TrainState, dict[str, Array], Array]:
            state, epoch_key = state_and_key
            permutation = jax.random.permutation(epoch_key, config.batch_size)
            minibatches = permutation.reshape((config.num_minibatches, minibatch_size))

            def minibatch_step(
                inner_state: Level3TrainState, indices: Array
            ) -> tuple[Level3TrainState, dict[str, Array]]:
                minibatch = jax.tree.map(lambda value: value[indices], batch)
                (_, metrics), gradients = jax.value_and_grad(loss_function, has_aux=True)(
                    inner_state.params, inner_state.apply_fn, minibatch
                )
                metrics["gradient_norm"] = optax.tree.norm(gradients)
                return inner_state.apply_gradients(grads=gradients), metrics

            next_state, minibatch_metrics = jax.lax.scan(minibatch_step, state, minibatches)
            metrics = jax.tree.map(jnp.mean, minibatch_metrics)
            return next_state, metrics, metrics["approx_kl"]

        def epoch_step(
            carry: tuple[Level3TrainState, Array], epoch_key: Array
        ) -> tuple[tuple[Level3TrainState, Array], tuple[dict[str, Array], Array]]:
            state, active = carry

            def active_epoch(_: None) -> tuple[Level3TrainState, dict[str, Array], Array]:
                return run_epoch((state, epoch_key))

            def skipped_epoch(_: None) -> tuple[Level3TrainState, dict[str, Array], Array]:
                return state, zero_metrics, jnp.asarray(0.0)

            next_state, metrics, approximate_kl = jax.lax.cond(
                active, active_epoch, skipped_epoch, operand=None
            )
            executed = active.astype(jnp.float32)
            next_active = active & (approximate_kl <= config.target_kl)
            return (next_state, next_active), (metrics, executed)

        epoch_keys = jax.random.split(key, config.update_epochs)
        (train_state, _), (epoch_metrics, executed) = jax.lax.scan(
            epoch_step, (train_state, jnp.asarray(True)), epoch_keys
        )
        denominator = jnp.maximum(jnp.sum(executed), 1.0)
        metrics = jax.tree.map(lambda values: jnp.sum(values, axis=0) / denominator, epoch_metrics)
        metrics["epochs_executed"] = jnp.sum(executed)
        target_variance = jnp.var(batch.returns)
        metrics["explained_variance"] = jnp.where(
            target_variance > 0,
            1.0 - jnp.var(batch.returns - batch.old_value) / target_variance,
            jnp.nan,
        )
        ema_params = optax.incremental_update(
            train_state.params, train_state.ema_params, 1.0 - config.ema_decay
        )
        return train_state.replace(ema_params=ema_params), metrics

    return jax.jit(update, donate_argnums=(0,))
