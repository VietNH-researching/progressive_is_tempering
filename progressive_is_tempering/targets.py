"""
Target distributions for score-based sampling experiments.

Each target exposes:
  - energy(x): E(x) = -log π(x) + const
  - log_prob(x): log π(x) up to a constant
  - grad_energy(x): ∇E(x), computed via jax.grad

For the SMC ladder (`smc.smc_sampler` via `mala_kernel.PTProblem`):
  - logpdf_ref(x): log N(x; 0, I)
  - logpdf_final_target(x): log π(x) (unnormalized OK)
  - rvs_ref(key): draw from N(0, I)

The particle-system energies `DW4` and `LennardJones` are lower-level: each is a
callable returning the unnormalized log-density of particle positions shaped
(..., n_particles, spatial_dim). Their notebooks wrap them into `energy(x)` on
flat inputs and add the zero-CoM projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

__all__ = [
    "GMMTarget",
    "make_gmm_target",
    "DW4",
    "LennardJones",
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

    # PTProblem interface -----------------------------------------------------

    def logpdf_ref(self, x: Array) -> Array:
        """log N(x; 0, I) (normalized)."""
        d = x.shape[0]
        return -0.5 * jnp.sum(x ** 2) - 0.5 * d * jnp.log(2.0 * jnp.pi)

    def logpdf_final_target(self, x: Array) -> Array:
        """log π(x) (unnormalized OK for SMC)."""
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

    Defaults: μ₁=-1.5, μ₂=2.5, σ=0.5, w₁=0.2, w₂=0.8, d=16.

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


# ---------------------------------------------------------------------------
# Particle systems
# ---------------------------------------------------------------------------

_TRAP_MODES = ("individual", "com")


def _pair_distances(pos: Array, n_particles: int) -> Array:
    """Distances r_ij over the unique pairs i < j, shape (..., n(n-1)/2).

    A 1e-8 floor inside the square root keeps the gradient finite at r = 0.
    """
    diff = pos[..., :, None, :] - pos[..., None, :, :]
    r = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
    i, j = jnp.triu_indices(n_particles, k=1)
    return r[..., i, j]


def _harmonic_trap(pos: Array, trap_scale: float, trap_mode: str) -> Array:
    """Harmonic trap energy, shape (...).

    'individual': (trap_scale/2) Σ_i ||x_i||²  (pulls every particle to 0)
    'com':        (trap_scale/2) N ||CoM||²    (pulls only the centre of mass)
    """
    if trap_mode == "individual":
        return 0.5 * trap_scale * jnp.sum(pos**2, axis=(-2, -1))
    n = pos.shape[-2]
    com = pos.mean(axis=-2)
    return 0.5 * trap_scale * n * jnp.sum(com**2, axis=-1)


@dataclass(frozen=True)
class DW4:
    """
    Double-well potential over 4 particles in 2D.

        U(x) = Σ_{i<j} [a (r_ij - r0)⁴ + b (r_ij - r0)² + c] + U_trap(x)
        log π(x) = -beta · U(x) + const

    With a=0.9, b=-4, r0=4 each pair potential has minima at r ≈ 2.51 and
    r ≈ 5.49. `trap_scale=0` removes the trap and leaves the energy
    translation-invariant (the setting used by the DW4 notebook).

    Attributes
    ----------
    a, b, c : float
        Quartic, quadratic and constant coefficients of the pair potential.
    r0 : float
        Centre of the quartic.
    trap_scale : float
        Harmonic trap strength; 0 disables it.
    trap_mode : {'individual', 'com'}
        See `_harmonic_trap`.
    beta : float
        Inverse temperature.
    """

    a: float = 0.9
    b: float = -4.0
    c: float = 0.0
    r0: float = 4.0
    trap_scale: float = 1.0
    trap_mode: str = "individual"
    beta: float = 1.0

    n_particles = 4
    spatial_dim = 2

    def __post_init__(self):
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.trap_mode not in _TRAP_MODES:
            raise ValueError(f"trap_mode must be one of {_TRAP_MODES}, got {self.trap_mode!r}")

    @property
    def dim(self) -> int:
        return self.n_particles * self.spatial_dim

    def __call__(self, x: Array) -> Array:
        """Unnormalized log-density; x: (..., 4, 2) → (...)."""
        if x.shape[-2:] != (4, 2):
            raise ValueError(f"DW4 expects x.shape[-2:] == (4, 2), got {tuple(x.shape)}")
        d = _pair_distances(x, 4) - self.r0
        u_dw = jnp.sum(self.a * d**4 + self.b * d**2 + self.c, axis=-1)
        u_trap = _harmonic_trap(x, self.trap_scale, self.trap_mode)
        return -self.beta * (u_dw + u_trap)


@dataclass(frozen=True)
class LennardJones:
    """
    Lennard-Jones cluster with a harmonic trap.

        U(x) = Σ_{i<j} ε [(rm/r_ij)¹² - 2 (rm/r_ij)⁶] + U_trap(x)
        log π(x) = -beta · U(x) + const

    Each pair term has its minimum, -ε, at r = rm. LJ13 is
    n_particles=13, spatial_dim=3 (d = 39).

    Attributes
    ----------
    n_particles, spatial_dim : int
        Number of particles and dimension of each.
    epsilon : float
        Well depth ε.
    rm : float
        Equilibrium pair distance.
    trap_scale : float
        Harmonic trap strength; 0 disables it.
    trap_mode : {'individual', 'com'}
        See `_harmonic_trap`.
    beta : float
        Inverse temperature.
    """

    n_particles: int = 13
    spatial_dim: int = 3
    epsilon: float = 1.0
    rm: float = 1.0
    trap_scale: float = 1.0
    trap_mode: str = "individual"
    beta: float = 1.0

    def __post_init__(self):
        if self.n_particles < 2:
            raise ValueError(f"n_particles must be >= 2, got {self.n_particles}")
        if self.spatial_dim < 1:
            raise ValueError(f"spatial_dim must be >= 1, got {self.spatial_dim}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.rm <= 0:
            raise ValueError(f"rm must be positive, got {self.rm}")
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.trap_mode not in _TRAP_MODES:
            raise ValueError(f"trap_mode must be one of {_TRAP_MODES}, got {self.trap_mode!r}")

    @property
    def dim(self) -> int:
        return self.n_particles * self.spatial_dim

    def __call__(self, x: Array) -> Array:
        """Unnormalized log-density; x: (..., n_particles, spatial_dim) → (...)."""
        n, d = self.n_particles, self.spatial_dim
        if x.shape[-2:] != (n, d):
            raise ValueError(
                f"LennardJones expects x.shape[-2:] == ({n}, {d}), got {tuple(x.shape)}"
            )
        inv_r = self.rm / _pair_distances(x, n)
        u_lj = jnp.sum(self.epsilon * (inv_r**12 - 2.0 * inv_r**6), axis=-1)
        u_trap = _harmonic_trap(x, self.trap_scale, self.trap_mode)
        return -self.beta * (u_lj + u_trap)
