"""Structured asymmetric actor-critic for the Level 3 V2 policy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import linen as nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax import Array

    from lsy_drone_racing.rl_training.level3_observations import ActorObservation, CriticObservation


def _orthogonal(scale: float) -> Callable:
    return nn.initializers.orthogonal(scale)


class EntityEncoder(nn.Module):
    """Encode every entity with shared weights."""

    width: int

    @nn.compact
    def __call__(self, tokens: Array) -> Array:
        """Return one learned embedding per input entity."""
        encoded = nn.Dense(self.width, kernel_init=_orthogonal(math.sqrt(2.0)))(tokens)
        encoded = nn.tanh(encoded)
        encoded = nn.Dense(self.width, kernel_init=_orthogonal(math.sqrt(2.0)))(encoded)
        return nn.tanh(encoded)


class EntityAttention(nn.Module):
    """Parameter-free multi-head ego-query pooling over encoded entities.

    The ego and entity encoders learn the shared key space. Avoiding redundant query/key/value
    projections keeps PPO backward compact while preserving permutation invariance and four-head
    selection. Critical clearance and risk values still bypass this pool through the direct path.
    """

    width: int
    heads: int

    def __call__(self, query: Array, entities: Array, valid: Array) -> Array:
        """Pool valid entity embeddings around an ego query."""
        head_width = self.width // self.heads
        query_heads = query.reshape((query.shape[0], self.heads, head_width))
        entity_heads = entities.reshape(
            (entities.shape[0], entities.shape[1], self.heads, head_width)
        )
        logits = jnp.einsum("bhd,bkhd->bhk", query_heads, entity_heads) / math.sqrt(head_width)
        logits = jnp.where(valid[:, None], logits, -1e9)
        weights = nn.softmax(logits, axis=-1)
        context = jnp.einsum("bhk,bkhd->bhd", weights, entity_heads).reshape(
            (query.shape[0], self.width)
        )
        return nn.tanh(context + query)


class BeliefEncoder(nn.Module):
    """Encode the deployment-safe ego, history, gate, and obstacle structures."""

    entity_dim: int
    attention_heads: int

    @nn.compact
    def __call__(self, observation: ActorObservation) -> Array:
        """Return fused belief features with a raw direct-geometry skip."""
        ego = nn.Dense(self.entity_dim, kernel_init=_orthogonal(math.sqrt(2.0)), name="ego_input")(
            observation.ego
        )
        ego = nn.tanh(ego)
        ego = nn.Dense(self.entity_dim, kernel_init=_orthogonal(math.sqrt(2.0)), name="ego_output")(
            ego
        )
        ego = nn.tanh(ego)

        history = observation.history.reshape((observation.history.shape[0], -1))
        history = nn.Dense(
            self.entity_dim, kernel_init=_orthogonal(math.sqrt(2.0)), name="history_input"
        )(history)
        history = nn.tanh(history)
        history = nn.Dense(
            self.entity_dim, kernel_init=_orthogonal(math.sqrt(2.0)), name="history_output"
        )(history)
        history = nn.tanh(history)

        gates = EntityEncoder(self.entity_dim, name="gate_encoder")(observation.gate_tokens)
        obstacles = EntityEncoder(self.entity_dim, name="obstacle_encoder")(
            observation.obstacle_tokens
        )
        gate_context = EntityAttention(
            self.entity_dim, self.attention_heads, name="gate_attention"
        )(ego, gates, jnp.ones(observation.gate_tokens.shape[:2], dtype=bool))
        obstacle_context = EntityAttention(
            self.entity_dim, self.attention_heads, name="obstacle_attention"
        )(ego, obstacles, observation.obstacle_tokens[..., -1] > 0.5)
        # The normalized direct geometry remains uncompressed at this boundary. This is the
        # deliberate skip path that prevents attention from hiding gate clearance or risk.
        return jnp.concatenate(
            (ego, history, gate_context, obstacle_context, observation.direct_geometry), axis=-1
        )


class Level3ActorCritic(nn.Module):
    """Independent structured actor and asymmetric privileged critic."""

    action_dim: int = 4
    entity_dim: int = 128
    attention_heads: int = 4
    fusion_dim: int = 256
    action_bias: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)

    def setup(self) -> None:
        """Create distinct actor, critic-belief, and privileged modules."""
        self.actor_encoder = BeliefEncoder(
            self.entity_dim, self.attention_heads, name="actor_encoder"
        )
        self.critic_belief_encoder = BeliefEncoder(
            self.entity_dim, self.attention_heads, name="critic_belief_encoder"
        )
        self.critic_true_gate_encoder = EntityEncoder(
            self.entity_dim, name="critic_true_gate_encoder"
        )
        self.critic_true_obstacle_encoder = EntityEncoder(
            self.entity_dim, name="critic_true_obstacle_encoder"
        )
        self.critic_true_gate_attention = EntityAttention(
            self.entity_dim, self.attention_heads, name="critic_true_gate_attention"
        )
        self.critic_true_obstacle_attention = EntityAttention(
            self.entity_dim, self.attention_heads, name="critic_true_obstacle_attention"
        )
        self.actor_hidden_1 = nn.Dense(
            self.fusion_dim, kernel_init=_orthogonal(math.sqrt(2.0)), name="actor_hidden_1"
        )
        self.actor_hidden_2 = nn.Dense(
            self.fusion_dim, kernel_init=_orthogonal(math.sqrt(2.0)), name="actor_hidden_2"
        )
        self.actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=_orthogonal(0.01),
            bias_init=lambda *_: jnp.asarray(self.action_bias, dtype=jnp.float32),
            name="actor_mean",
        )
        self.actor_log_std = self.param(
            "actor_log_std", lambda *_: jnp.asarray((-1.0, -1.0, -1.0, -0.5), dtype=jnp.float32)
        )
        self.critic_hidden_1 = nn.Dense(
            self.fusion_dim, kernel_init=_orthogonal(math.sqrt(2.0)), name="critic_hidden_1"
        )
        self.critic_hidden_2 = nn.Dense(
            self.fusion_dim, kernel_init=_orthogonal(math.sqrt(2.0)), name="critic_hidden_2"
        )
        self.critic_value = nn.Dense(1, kernel_init=_orthogonal(1.0), name="critic_value")

    def actor(self, observation: ActorObservation) -> tuple[Array, Array]:
        """Return the actor's pre-tanh mean and diagonal log standard deviation."""
        hidden = nn.tanh(self.actor_hidden_1(self.actor_encoder(observation)))
        hidden = nn.tanh(self.actor_hidden_2(hidden))
        mean = self.actor_mean(hidden)
        return mean, jnp.broadcast_to(self.actor_log_std, mean.shape)

    def critic(
        self, actor_observation: ActorObservation, critic_observation: CriticObservation
    ) -> Array:
        """Return a value using belief features plus current privileged state."""
        belief = self.critic_belief_encoder(actor_observation)
        query = belief[:, : self.entity_dim]
        true_gates = self.critic_true_gate_encoder(critic_observation.true_gate_tokens)
        true_obstacles = self.critic_true_obstacle_encoder(critic_observation.true_obstacle_tokens)
        true_gate_context = self.critic_true_gate_attention(
            query, true_gates, jnp.ones(critic_observation.true_gate_tokens.shape[:2], dtype=bool)
        )
        true_obstacle_context = self.critic_true_obstacle_attention(
            query, true_obstacles, critic_observation.true_obstacle_tokens[..., -1] > 0.5
        )
        privileged = jnp.concatenate(
            (
                true_gate_context,
                true_obstacle_context,
                critic_observation.true_direct_geometry,
                critic_observation.dynamics,
            ),
            axis=-1,
        )
        hidden = jnp.concatenate((belief, privileged), axis=-1)
        hidden = nn.tanh(self.critic_hidden_1(hidden))
        hidden = nn.tanh(self.critic_hidden_2(hidden))
        return self.critic_value(hidden).squeeze(-1)

    def __call__(
        self, actor_observation: ActorObservation, critic_observation: CriticObservation
    ) -> tuple[Array, Array, Array]:
        """Return policy distribution parameters and privileged value."""
        mean, log_std = self.actor(actor_observation)
        return mean, log_std, self.critic(actor_observation, critic_observation)
