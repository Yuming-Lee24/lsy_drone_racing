"""Flax actor-critic network and bounded Gaussian policy utilities."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import linen as nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

LOG_2PI = math.log(2.0 * math.pi)


def _orthogonal(scale: float) -> Callable:
    return nn.initializers.orthogonal(scale)


class ActorCritic(nn.Module):
    """Independent actor and critic MLPs for the racing policy."""

    action_dim: int = 4
    hidden_dim: int = 256
    action_bias: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)

    @nn.compact
    def __call__(self, observation: Array) -> tuple[Array, Array, Array]:
        """Return pre-tanh action mean, log standard deviation, and value."""
        actor = nn.Dense(self.hidden_dim, kernel_init=_orthogonal(math.sqrt(2.0)))(observation)
        actor = nn.tanh(actor)
        actor = nn.Dense(self.hidden_dim, kernel_init=_orthogonal(math.sqrt(2.0)))(actor)
        actor = nn.tanh(actor)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=_orthogonal(0.01),
            bias_init=lambda *_: jnp.asarray(self.action_bias, dtype=jnp.float32),
        )(actor)
        initial_log_std = jnp.asarray((-1.0, -1.0, -1.0, -0.5), dtype=jnp.float32)
        log_std = self.param("actor_log_std", lambda *_: initial_log_std)
        log_std = jnp.broadcast_to(log_std, mean.shape)

        critic = nn.Dense(self.hidden_dim, kernel_init=_orthogonal(math.sqrt(2.0)))(observation)
        critic = nn.tanh(critic)
        critic = nn.Dense(self.hidden_dim, kernel_init=_orthogonal(math.sqrt(2.0)))(critic)
        critic = nn.tanh(critic)
        value = nn.Dense(1, kernel_init=_orthogonal(1.0))(critic).squeeze(-1)
        return mean, log_std, value


def gaussian_log_probability(raw_action: Array, mean: Array, log_std: Array) -> Array:
    """Log probability of an unsquashed diagonal Gaussian sample."""
    inverse_variance = jnp.exp(-2.0 * log_std)
    return jnp.sum(
        -0.5 * ((raw_action - mean) ** 2 * inverse_variance + 2.0 * log_std + LOG_2PI), axis=-1
    )


def squashed_log_probability(raw_action: Array, mean: Array, log_std: Array) -> Array:
    """Log probability after a tanh bijector, including its Jacobian."""
    action = jnp.tanh(raw_action)
    correction = jnp.sum(jnp.log(jnp.maximum(1.0 - action * action, 1e-6)), axis=-1)
    return gaussian_log_probability(raw_action, mean, log_std) - correction


def gaussian_entropy(log_std: Array) -> Array:
    """Entropy of the base diagonal Gaussian used for PPO regularization."""
    return jnp.sum(log_std + 0.5 * (1.0 + LOG_2PI), axis=-1)


def sample_action(mean: Array, log_std: Array, key: Array) -> tuple[Array, Array, Array]:
    """Sample a bounded action and return action, raw sample, and corrected log probability."""
    raw_action = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    action = jnp.tanh(raw_action)
    return action, raw_action, squashed_log_probability(raw_action, mean, log_std)


def deterministic_action(mean: Array) -> Array:
    """Return the mode used for evaluation and deployment."""
    return jnp.tanh(mean)
