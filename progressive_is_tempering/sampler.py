"""
Backward SDE sampler with path-space importance weights.

The backward SDE (time-reversal of the OU forward process) driven by the
learned excess-score control u_θ(x, t, β) = 2 s_θ(x, t, β):

    dY_t = (-Y_t + u_θ(Y_t, t, β)) dt + √2 dW_t,   Y_T ~ N(0, I)

Exponential-integrator discretization (running backward from T to 0):

    x_{n-1} = λ x_n + (1-λ) u_θ(x_n, t_n, β) + σ ε_n,
              λ = e^{-Δt},   σ² = 1 - e^{-2Δt}

Obtained by freezing u_θ constant over [t_{n-1}, t_n] and solving the
resulting linear SDE in closed form. When u_θ ≡ 0 the step reduces exactly
to the OU transition kernel, preserving detailed balance at the discrete level.

Path-space log-weight (box formula from TECHNIQUES_SUMMARY.md §7):

    log w = Σ_n [log p_ρ(x_n|x_{n-1}) - log q_θ(x_{n-1}|x_n)]
            - (1-β) ½‖x_0‖² - β E(x_0) + ½‖x_T‖²

where
  p_ρ(x_n | x_{n-1}) = N(x_n; λ x_{n-1}, σ² I)          [exact OU step]
  q_θ(x_{n-1} | x_n) = N(x_{n-1}; λ x_n + (1-λ)u, σ² I) [backward proposal]

Both kernels share the same variance σ², so the -d/2 log(2π σ²) normalization
cancels exactly from log p_ρ - log q_θ at every step.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from progressive_is_tempering.sde import ou_log_transition_density

__all__ = ["backward_sde_step", "backward_sde_single", "backward_sde_batch"]


def backward_sde_step(
    key: Array,
    x_n: Array,
    t_n: Array,
    dt: float,
    control: Array,
) -> tuple[Array, Array, Array]:
    """
    One exponential-integrator step of the backward SDE and its per-step log-weight.

    Parameters
    ----------
    key : Array
        PRNG key.
    x_n : Array, shape (d,)
        Current state at time t_n (running backward, so t_n > t_{n-1}).
    t_n : Array or float
        Current time.
    dt : float
        Step size (positive; we step from t_n to t_n - dt).
    control : Array, shape (d,)
        u_θ(x_n, t_n, β) = 2 s_θ(x_n, t_n, β).

    Returns
    -------
    x_prev : Array, shape (d,)
        State at time t_n - dt.
    log_step_weight : Array
        Per-step contribution log p_ρ(x_n|x_{n-1}) - log q_θ(x_{n-1}|x_n).
    eps : Array, shape (d,)
        Noise sample used (for reproducibility / debugging).
    """
    # OU coefficients for a step of length dt
    lam = jnp.exp(-dt)           # λ = e^{-Δt}
    sigma_sq = 1.0 - jnp.exp(-2.0 * dt)   # σ² = 1 - e^{-2Δt}

    eps = jax.random.normal(key, x_n.shape)
    # Freeze u_θ over [t_{n-1}, t_n] and solve the linear SDE exactly:
    #   x_{n-1} = λ x_n + (1-λ) u_θ + σ ε
    x_prev = lam * x_n + (1.0 - lam) * control + jnp.sqrt(sigma_sq) * eps

    # log p_ρ(x_n | x_{n-1}) = N(x_n; λ x_{n-1}, σ² I)  [exact OU forward step]
    log_p = ou_log_transition_density(x_n, x_prev, dt)

    # log q_θ(x_{n-1} | x_n) = N(x_{n-1}; λ x_n + (1-λ) u_θ, σ² I)
    # Same σ² as p_ρ, so the -d/2 log(2π σ²) constants cancel in log p - log q.
    mean_q = lam * x_n + (1.0 - lam) * control
    log_q = -0.5 * jnp.sum((x_prev - mean_q) ** 2) / sigma_sq

    return x_prev, log_p - log_q, eps


def backward_sde_single(
    key: Array,
    control_fn: Callable[[Array, Array, Array], Array],
    beta: Array,
    energy_fn: Callable[[Array], Array],
    dim: int,
    T: float = 4.0,
    n_steps: int = 100,
) -> tuple[Array, Array]:
    """
    Single-particle backward SDE with accumulated path-space log-weight.

    Uses jax.lax.scan for JIT-friendly time integration.

    Parameters
    ----------
    key : Array
        PRNG key.
    control_fn : callable
        u_θ(x, t, β) → (d,).  Called with single (unbatched) arrays.
    beta : Array
        Inverse temperature scalar.
    energy_fn : callable
        E(x) → scalar.
    dim : int
        Ambient dimension.
    T : float
        Terminal time (SDE runs from T to 0).
    n_steps : int
        Number of discretization steps.

    Returns
    -------
    x0 : Array, shape (d,)
        Terminal sample at t=0.
    log_w : Array
        Path-space log-weight for this trajectory.
    """
    dt = T / n_steps
    # Backward time grid: t_N, t_{N-1}, ..., t_1  (step from t_n → t_n - dt)
    times = jnp.linspace(T, dt, n_steps)

    # Draw x_T ~ N(0, I) and accumulate prior correction +½‖x_T‖²
    key, subkey = jax.random.split(key)
    x_T = jax.random.normal(subkey, (dim,))
    init_log_w = 0.5 * jnp.sum(x_T ** 2)

    def step_fn(carry: tuple, t_n: Array) -> tuple:
        x_n, log_w, rng = carry
        rng, subkey = jax.random.split(rng)
        control = control_fn(x_n, t_n, beta)
        x_prev, step_lw, _ = backward_sde_step(subkey, x_n, t_n, dt, control)
        return (x_prev, log_w + step_lw, rng), None

    (x0, log_w, _), _ = jax.lax.scan(step_fn, (x_T, init_log_w, key), times)

    # Terminal correction: log π_β(x_0) = -(1-β)½‖x_0‖² - β E(x_0) (up to const)
    log_w = log_w - (1.0 - beta) * 0.5 * jnp.sum(x0 ** 2) - beta * energy_fn(x0)

    return x0, log_w


def backward_sde_batch(
    key: Array,
    control_fn: Callable[[Array, Array, Array], Array],
    beta: Array,
    energy_fn: Callable[[Array], Array],
    dim: int,
    n_particles: int,
    T: float = 4.0,
    n_steps: int = 100,
) -> tuple[Array, Array]:
    """
    Batch backward SDE: run n_particles independent trajectories in parallel.

    Parameters
    ----------
    key : Array
        PRNG key (split internally).
    control_fn : callable
        Single-sample u_θ(x, t, β) → (d,).
    beta : Array
        Inverse temperature scalar.
    energy_fn : callable
        E(x) → scalar.
    dim : int
        Ambient dimension.
    n_particles : int
        Number of independent particles.
    T : float
        Terminal time.
    n_steps : int
        Number of discretization steps.

    Returns
    -------
    x0_batch : Array, shape (n_particles, d)
        Terminal samples.
    log_w_batch : Array, shape (n_particles,)
        Path-space log-weights.
    """
    keys = jax.random.split(key, n_particles)

    def single(k):
        return backward_sde_single(k, control_fn, beta, energy_fn, dim, T, n_steps)

    x0_batch, log_w_batch = jax.vmap(single)(keys)
    return x0_batch, log_w_batch