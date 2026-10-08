"""Entity-attention actor and critic (Flax). Token sets make the parameters independent of field size."""
from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

from kilithattu_core.spaces import HEAD_SIZES


@dataclass(frozen=True)
class NetConfig:
    d_model: int = 64
    layers: int = 2
    heads: int = 4
    hidden: int = 128
    gru: bool = True
    n_calls: int = 4

    def head_sizes(self) -> dict[str, int]:
        return {**HEAD_SIZES, "call": self.n_calls + 1}


class Block(nn.Module):
    d: int
    heads: int

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=self.d, out_features=self.d)(h)
        h = nn.LayerNorm()(x)
        h = nn.Dense(2 * self.d)(h)
        return x + nn.Dense(self.d)(nn.gelu(h))


class Encoder(nn.Module):
    cfg: NetConfig

    @nn.compact
    def __call__(self, players, lines, glob, ego):
        d = self.cfg.d_model
        P = players.shape[0]
        etype = nn.Embed(3, d)
        x = jnp.concatenate([
            nn.Dense(d)(players) + etype(jnp.zeros((P,), jnp.int32)),
            nn.Dense(d)(lines) + etype(jnp.ones((lines.shape[0],), jnp.int32)),
            (nn.Dense(d)(glob) + etype(jnp.array(2)))[None],
        ], 0)
        for _ in range(self.cfg.layers):
            x = Block(d, self.cfg.heads)(x)
        ego_tok = jnp.where(ego >= 0, x[jnp.maximum(ego, 0)], 0.0)
        return jnp.concatenate([ego_tok, x[:P].mean(0), x[-1]])


class Actor(nn.Module):
    """One agent step: tokens -> encoder -> MLP + role embedding -> GRU -> masked-categorical heads."""

    cfg: NetConfig

    @nn.compact
    def __call__(self, carry, players, lines, glob, ego, role, reset):
        c = self.cfg
        z = Encoder(c)(players, lines, glob, ego)
        z = nn.relu(nn.Dense(c.hidden)(z))
        z = z + nn.Embed(3, c.hidden)(role)
        if c.gru:
            carry = jnp.where(reset, 0.0, carry)
            carry, z = nn.GRUCell(features=c.hidden)(carry, z)
        logits = {k: nn.Dense(n, kernel_init=nn.initializers.orthogonal(0.01))(z) for k, n in c.head_sizes().items()}
        return carry, logits


class Critic(nn.Module):
    """Centralised team critic on un-fogged tokens (training only)."""

    cfg: NetConfig

    @nn.compact
    def __call__(self, players, lines, glob):
        z = Encoder(self.cfg)(players, lines, glob, jnp.array(-1))
        z = nn.relu(nn.Dense(self.cfg.hidden)(z))
        z = nn.relu(nn.Dense(self.cfg.hidden)(z))
        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(z)[0]
