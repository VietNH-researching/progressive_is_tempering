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
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "GMMTarget",
    "make_gmm_target",
    "Phi4Target",
    "make_phi4_target",
    "NealFunnelTarget",
    "make_neal_funnel_target",
]


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


@dataclass
class Phi4Target:
    """
    1D phi^4 lattice field theory (stochastic Allen-Cahn model), wrapping
    `jax_pdf.PhiFour`.

        U(phi) = c * sum_i (phi_{i+1}-phi_i)^2/2
                 + (1/c) * sum_i [(1-phi_i^2)^2/4 + b*phi_i],   c = a*dim

    with Dirichlet or periodic boundary conditions. The double-well on-site
    potential pins each site near phi=+1 or phi=-1; unlike `GMMTarget`, the
    normalizing constant is intractable, so there is no closed-form log_prob.

    Attributes
    ----------
    dim : int
        Number of lattice sites.
    phi4_dist : jax_pdf.PhiFour
        Callable unnormalized log-density -beta * U(phi).
    """

    dim: int
    phi4_dist: Any

    def log_prob(self, x: Array) -> Array:
        """log π(x) = phi4_dist(x) (unnormalized: log Z is intractable)."""
        return self.phi4_dist(x)

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


def make_phi4_target(
    dim: int = 100,
    a: float = 0.1,
    b: float = 0.0,
    inv_temp: float = 20.0,
    periodic: bool = False,
) -> Phi4Target:
    """
    Construct a 1D phi^4 lattice field theory target, imported from jax_pdf.

    Default values match `jax_pdf.PhiFour`'s own defaults for a, b, periodic;
    inv_temp=20 makes the target sharply bimodal (its own energy-scale
    parameter -- unrelated to the beta-tempering ladder built around it).

    Parameters
    ----------
    dim : int
        Number of lattice sites.
    a : float
        Coupling constant (smaller -> stronger neighbor correlation, harder).
    b : float
        External field bias (0 -> exact Z2 symmetry, phi -> -phi).
    inv_temp : float
        phi^4 model's own inverse temperature (sharpens the double wells).
    periodic : bool
        False = Dirichlet boundary conditions, True = periodic.

    Returns
    -------
    Phi4Target
    """
    from jax_pdf import PhiFour

    phi4_dist = PhiFour(a=a, b=b, dim_grid=dim, beta=inv_temp, periodic=periodic)
    return Phi4Target(dim=dim, phi4_dist=phi4_dist)


@dataclass
class NealFunnelTarget:
    """
    Neal's funnel distribution over R^d, wrapping `jax_pdf.NealFunnel`.

        p(x) = N(x_0; 0, sigma^2) * prod_{i=1}^{d-1} N(x_i; 0, exp(x_0))

    The first coordinate x_0 sets the scale of every other coordinate: large
    x_0 spreads the remaining coordinates out (the "mouth"), very negative x_0
    collapses them toward zero (the "neck"). Already exactly normalized
    (log Z = 0 by construction), and `nf` also ships an exact sampler.

    Attributes
    ----------
    dim : int
        Ambient dimension d (>= 2).
    nf : Any
        Underlying jax_pdf.NealFunnel instance.
    """

    dim: int
    nf: Any

    def log_prob(self, x: Array) -> Array:
        """log π(x) (exactly normalized: log Z = 0)."""
        return self.nf(x)

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
        """log π(x) (already normalized)."""
        return self.log_prob(x)

    def rvs_ref(self, key: Array) -> Array:
        """Draw one sample from N(0, I)."""
        return jax.random.normal(key, (self.dim,))

    def rvs_exact(self, key: Array, n: int) -> Array:
        """Draw n exact i.i.d. samples from the funnel, shape (n, dim)."""
        return self.nf.sample(key, n)


def make_neal_funnel_target(dim: int = 10, sigma: float = 3.0) -> NealFunnelTarget:
    """
    Construct Neal's funnel target, imported from jax_pdf.

    Default values (d=10, sigma=3.0) match `jax_pdf.NealFunnel`'s own
    defaults, the standard benchmark setting.

    Parameters
    ----------
    dim : int
        Ambient dimension (>= 2).
    sigma : float
        Std dev of the neck coordinate x_0; larger widens the funnel.

    Returns
    -------
    NealFunnelTarget
    """
    from jax_pdf import NealFunnel

    nf = NealFunnel(dim=dim, sigma=sigma)
    return NealFunnelTarget(dim=dim, nf=nf)