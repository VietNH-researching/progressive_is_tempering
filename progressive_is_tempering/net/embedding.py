# Time embedding using sinusoidal positional encoding
# Reference: Vaswani et al., "Attention Is All You Need" (2017)

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["time_embedding"]


def _time_embedding_single(
    t: float,
    freq_min: float,
    freq_max: float,
    embedding_dim: int,
) -> Array:
    """
    Compute sinusoidal time embedding for a single time value.

    Creates a feature vector using sine and cosine functions at
    log-spaced frequencies, similar to positional encodings in
    Transformers.

    Parameters
    ----------
    t : float
        Time value to embed.
    freq_min : float
        Minimum frequency.
    freq_max : float
        Maximum frequency.
    embedding_dim : int
        Total embedding dimension (must be even).

    Returns
    -------
    embedding : Array, shape (embedding_dim,)
        Concatenation of [sin(f_1*t), ..., sin(f_k*t), cos(f_1*t), ..., cos(f_k*t)]
        where k = embedding_dim // 2.
    """
    # Log-spaced frequencies from freq_min to freq_max
    freqs = jnp.exp(
        jnp.linspace(jnp.log(freq_min), jnp.log(freq_max), embedding_dim // 2)
    )

    # Compute sin and cos features
    angles = t * freqs
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])


# Vectorized version for batched inputs
_time_embedding_batched = jax.vmap(
    _time_embedding_single,
    in_axes=(0, None, None, None),
)


def time_embedding(
    t: Array,
    freq_min: float,
    freq_max: float,
    embedding_dim: int,
) -> Array:
    """
    Compute sinusoidal time embeddings for a batch of time values.

    Uses log-spaced frequencies to create rich temporal features,
    similar to positional encodings in Transformers but for continuous
    time values.

    Parameters
    ----------
    t : Array, shape (batch, 1) or (batch,)
        Time values to embed.
    freq_min : float
        Minimum frequency (e.g., 1.0).
    freq_max : float
        Maximum frequency (e.g., 10.0).
    embedding_dim : int
        Output dimension (must be even). The first half contains
        sine features, the second half cosine features.

    Returns
    -------
    embeddings : Array, shape (batch, embedding_dim)
        Sinusoidal time embeddings.

    Example
    -------
    >>> t = jnp.array([[0.1], [0.5], [1.0]])
    >>> emb = time_embedding(t, freq_min=1.0, freq_max=10.0, embedding_dim=16)
    >>> emb.shape
    (3, 16)
    """
    # Handle both (batch, 1) and (batch,) inputs
    t_flat = t.squeeze(-1) if t.ndim == 2 else t

    return _time_embedding_batched(t_flat, freq_min, freq_max, embedding_dim)
