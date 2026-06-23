"""
Target distributions for score-based sampling experiments.

Each target exposes:
  - energy(x): E(x) = -log π(x) + const
  - log_prob(x): log π(x) up to a constant
  - grad_energy(x): ∇E(x), computed via jax.grad

For MPPT integration:
  - logpdf_ref(x): log N(x; 0, I)
  - logpdf_final_target(x): log π(x) (unnormalized OK)
  - rvs_ref(key): draw from N(0, I)
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["GMMTarget", "make_gmm_target"]


@dataclass
class GMMTarget:
    """
    Factorized Gaussian Mixture Model over R^d.

    Each coordinate is iid from the same 1-D mixture:
        π_1d(x_i) = Σ_k w_k N(x_i; μ_k, σ²)

    so the joint log-prob factorizes:
        log π(x) = Σ_i log π_1d(x_i)

    Attributes
    ----------
    dim : int
        Ambient dimension d.
    means : Array, shape (n_components,)
        Component means μ_k.
    log_weights : Array, shape (n_components,)
        Log of mixing weights (normalized).
    std : float
        Shared standard deviation σ for all components.
    """

    dim: int
    means: Array
    log_weights: Array
    std: float

    def log_prob(self, x: Array) -> Array:
        """log π(x) = Σ_i log[Σ_k w_k N(x_i; μ_k, σ²)]."""
        # x: (dim,)
        # per-dimension, per-component log-densities: (dim, n_components)
        log_comp = (
            self.log_weights[None, :]  # (1, K)
            - 0.5 * ((x[:, None] - self.means[None, :]) / self.std) ** 2
            - jnp.log(self.std)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        # log-sum-exp over components, then sum over dimensions
        return jnp.sum(jax.nn.logsumexp(log_comp, axis=-1))

    def energy(self, x: Array) -> Array:
        """E(x) = -log π(x)."""
        return -self.log_prob(x)

    def grad_energy(self, x: Array) -> Array:
        """∇E(x) via automatic differentiation."""
        return jax.grad(self.energy)(x)

    # MPPT interface ----------------------------------------------------------

    def logpdf_ref(self, x: Array) -> Array:
        """log N(x; 0, I) (normalized)."""
        d = x.shape[0]
        return -0.5 * jnp.sum(x ** 2) - 0.5 * d * jnp.log(2.0 * jnp.pi)

    def logpdf_final_target(self, x: Array) -> Array:
        """log π(x) (unnormalized OK for MPPT)."""
        return self.log_prob(x)

    def rvs_ref(self, key: Array) -> Array:
        """Draw one sample from N(0, I)."""
        return jax.random.normal(key, (self.dim,))


def make_gmm_target(
    dim: int = 16,
    means: tuple[float, ...] = (-1.5, 2.5),
    weights: tuple[float, ...] = (0.2, 0.8),
    std: float = 0.5,
) -> GMMTarget:
    """
    Construct a factorized GMM target with default GMM configuration.

    Default values follow IMPLEMENTATION_MLP.md:
      μ₁=-1.5, μ₂=2.5, σ=0.5, w₁=0.2, w₂=0.8, d=16.

    Parameters
    ----------
    dim : int
        Ambient dimension.
    means : tuple of float
        Component means.
    weights : tuple of float
        Mixing weights (need not be normalized; will be normalized).
    std : float
        Shared standard deviation.

    Returns
    -------
    GMMTarget
    """
    means_arr = jnp.array(means, dtype=jnp.float32)
    weights_arr = jnp.array(weights, dtype=jnp.float32)
    log_weights = jnp.log(weights_arr) - jax.nn.logsumexp(jnp.log(weights_arr))
    return GMMTarget(dim=dim, means=means_arr, log_weights=log_weights, std=std)