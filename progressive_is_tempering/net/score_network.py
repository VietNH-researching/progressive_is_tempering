"""
β-conditioned excess-score network for MLP-based problems.

Architecture (IMPLEMENTATION_MLP.md §Score Network: Input Wiring):

    t_embed    = time_embedding(t,    freq_min=1.0, freq_max=10.0, dim=time_embedding_dim)
    beta_embed = time_embedding(beta, freq_min=0.5, freq_max=5.0,  dim=beta_embedding_dim)
    h = concat([x, t_embed, beta_embed])
    out = ResMLP(dim_output=x_dim, hidden_dim=..., n_blocks=...)(h)

The network approximates the *excess score*:

    s_θ(x_t, t, β) ≈ ∇_{x_t} log (p_t^β / p_t^ρ)

Single-sample interface (unbatched): use jax.vmap for batched evaluation.
At β=0 the excess score is zero; small output initialization (output_init_scale)
ensures near-zero predictions at initialization.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from progressive_is_tempering.net.embedding import time_embedding
from progressive_is_tempering.net.mlp import ResMLP

__all__ = ["ScoreNetwork"]


class ScoreNetwork(nn.Module):
    """
    Excess score network s_θ(x, t, β).

    Attributes
    ----------
    x_dim : int
        Ambient dimension (input and output size).
    hidden_dim : int
        Hidden dimension for all ResMLP layers.
    n_blocks : int
        Number of residual blocks in the ResMLP.
    time_embedding_dim : int
        Sinusoidal embedding dimension for t.
    beta_embedding_dim : int
        Sinusoidal embedding dimension for β.
    output_init_scale : float
        Std-dev for small-init output layer of ResMLP.
    time_freq_min, time_freq_max : float
        Frequency range for the t embedding.
    beta_freq_min, beta_freq_max : float
        Frequency range for the β embedding.
    """

    x_dim: int
    hidden_dim: int = 256
    n_blocks: int = 3
    time_embedding_dim: int = 16
    beta_embedding_dim: int = 8
    output_init_scale: float = 1e-4
    time_freq_min: float = 1.0
    time_freq_max: float = 10.0
    beta_freq_min: float = 0.5
    beta_freq_max: float = 5.0

    @nn.compact
    def __call__(self, x: Array, t: Array, beta: Array) -> Array:
        """
        Single-sample forward pass (no batch dimension).

        Parameters
        ----------
        x : Array, shape (x_dim,)
            Noised sample x_t.
        t : Array or float
            Noise time (scalar).
        beta : Array or float
            Inverse temperature (scalar).

        Returns
        -------
        Array, shape (x_dim,)
            Predicted excess score s_θ(x, t, β).
        """
        # Reshape scalars to (1,) for time_embedding (expects (batch,))
        t_1d = jnp.atleast_1d(jnp.asarray(t, dtype=jnp.float32)).reshape(1)
        beta_1d = jnp.atleast_1d(jnp.asarray(beta, dtype=jnp.float32)).reshape(1)

        t_emb = time_embedding(
            t_1d,
            freq_min=self.time_freq_min,
            freq_max=self.time_freq_max,
            embedding_dim=self.time_embedding_dim,
        )[0]  # (time_embedding_dim,)

        beta_emb = time_embedding(
            beta_1d,
            freq_min=self.beta_freq_min,
            freq_max=self.beta_freq_max,
            embedding_dim=self.beta_embedding_dim,
        )[0]  # (beta_embedding_dim,)

        h = jnp.concatenate(
            [x.astype(jnp.float32), t_emb, beta_emb]
        )  # (x_dim + time_embedding_dim + beta_embedding_dim,)

        return ResMLP(
            dim_output=self.x_dim,
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            output_init_scale=self.output_init_scale,
        )(h)