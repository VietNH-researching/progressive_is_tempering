"""
Per-temperature replay buffers.

Each temperature level k has an independent replay buffer storing (x, log_w)
pairs with fixed capacity and FIFO eviction.  When sampling a mini-batch, the
IS log-weights are used for systematic resampling so the returned samples are
approximately i.i.d. from π_{β_k}.

Design choices:
- Python-level state (ptr, size are ints), JAX arrays for data & log_weights.
- FIFO eviction via a circular write pointer.
- Systematic resampling (see resampling.py) for weighted sampling.
- Optionally (`store_grads=True`) each row also holds ∇E(x), written once when
  the row is added, so the TSM loss reads it instead of recomputing it at every
  inner step. The two are stored side by side as one `[x | ∇E(x)]` array, so a
  mini-batch is still a single gather (the inner loop issues one per level per
  step, and on small targets those eager dispatches dominate its run time).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from progressive_is_tempering.resampling import systematic_resample

__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    """
    Fixed-capacity FIFO replay buffer with IS-weighted sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of (sample, log_weight) pairs stored.
    dim : int
        Sample dimension.
    store_grads : bool
        Also store ∇E(x) per row: `add` then requires `grads`, and
        `sample_with_grads` returns the rows as `[x | ∇E(x)]`.
    """

    def __init__(self, capacity: int, dim: int, store_grads: bool = False) -> None:
        self.capacity = capacity
        self.dim = dim
        self.store_grads = store_grads
        # Row storage, preallocated and zero-initialized: x, or [x | ∇E(x)]
        self.rows = jnp.zeros((capacity, 2 * dim if store_grads else dim),
                              dtype=jnp.float32)
        # -inf marks slots that have never been written (skipped in sampling)
        self.log_weights = jnp.full((capacity,), -jnp.inf, dtype=jnp.float32)
        self.size: int = 0  # number of valid entries (≤ capacity)
        self.ptr: int = 0   # next write position (circular)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    @property
    def data(self) -> Array:
        """Stored samples, shape (capacity, dim)."""
        return self.rows[:, : self.dim] if self.store_grads else self.rows

    @property
    def grads(self) -> Array:
        """Stored ∇E(x), shape (capacity, dim). Requires `store_grads`."""
        if not self.store_grads:
            raise AttributeError("this buffer was created with store_grads=False")
        return self.rows[:, self.dim :]

    def add(self, samples: Array, log_weights: Array, grads: Array | None = None) -> None:
        """
        Append samples to the buffer with FIFO eviction.

        Parameters
        ----------
        samples : Array, shape (n, dim)
        log_weights : Array, shape (n,)
        grads : Array, shape (n, dim), optional
            ∇E at each sample. Required if and only if `store_grads`.
        """
        if self.store_grads != (grads is not None):
            raise ValueError(
                "grads must be given if and only if the buffer has store_grads=True"
            )
        n = samples.shape[0]
        indices = (jnp.arange(n) + self.ptr) % self.capacity
        block = jnp.concatenate([samples, grads], axis=1) if self.store_grads else samples
        self.rows = self.rows.at[indices].set(block)
        self.log_weights = self.log_weights.at[indices].set(log_weights)
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def freeze(self) -> None:
        """Mark buffer as read-only (no-op; just a semantic marker)."""
        pass

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, key: Array, batch_size: int) -> tuple[Array, Array]:
        """
        Draw batch_size samples via systematic resampling on IS log-weights.

        Only uses the first self.size valid entries.

        Parameters
        ----------
        key : Array
            PRNG key.
        batch_size : int
            Number of samples to draw.

        Returns
        -------
        samples : Array, shape (batch_size, dim)
        log_weights_sampled : Array, shape (batch_size,)
            The IS log-weights of the selected particles (before resampling
            normalization; returned for diagnostic purposes).
        """
        valid_lw = self.log_weights[: self.size]
        indices = systematic_resample(key, valid_lw, batch_size)
        if self.store_grads:
            return self.rows[indices][:, : self.dim], self.log_weights[indices]
        return self.rows[indices], self.log_weights[indices]

    def sample_with_grads(self, key: Array, batch_size: int) -> tuple[Array, Array]:
        """
        As `sample` (same key, same draws), but returns each row as
        `[x | ∇E(x)]` in one array: one gather for both. Requires `store_grads`.

        Returns
        -------
        rows : Array, shape (batch_size, 2 * dim)
            `rows[:, :dim]` are the samples, `rows[:, dim:]` their ∇E.
        log_weights_sampled : Array, shape (batch_size,)
        """
        if not self.store_grads:
            raise ValueError("sample_with_grads needs a buffer with store_grads=True")
        valid_lw = self.log_weights[: self.size]
        indices = systematic_resample(key, valid_lw, batch_size)
        return self.rows[indices], self.log_weights[indices]

    def is_ready(self, min_size: int) -> bool:
        """Return True if the buffer holds at least min_size valid entries."""
        return self.size >= min_size

    # ------------------------------------------------------------------
    # IS bridging: compute weights to the next temperature level
    # ------------------------------------------------------------------

    @staticmethod
    def is_bridge_log_weights(
        samples: Array,
        energy_fn,
        beta_k: Array,
        beta_next: Array,
    ) -> Array:
        """
        Compute IS log-weights for bridging samples from π_{β_k} to π_{β_{k+1}}.

        From docs/TECHNIQUES_SUMMARY.md §1:
            log w_{k→k+1}(x) = (β_{k+1} - β_k) · (-E(x) + ½‖x‖²)

        Parameters
        ----------
        samples : Array, shape (K, d)
        energy_fn : callable, x → scalar
        beta_k : Array, scalar
        beta_next : Array, scalar

        Returns
        -------
        Array, shape (K,)
        """
        delta_beta = beta_next - beta_k
        energies = jax.vmap(energy_fn)(samples)  # (K,)
        norms_sq = jnp.sum(samples ** 2, axis=-1)  # (K,)
        return delta_beta * (-energies + 0.5 * norms_sq)