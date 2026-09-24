"""
Adaptive-temperature SMC sampler: builds the temperature ladder.

Anneals a particle population from the reference (t=0) to the target (t=1)
along the geometric path. Each step:

1. bisects for the next temperature at which reweighting drops the
   normalized ESS to `local_ess_ratio`;
2. reweights and systematically resamples;
3. adapts the MALA kernel (`mala_kernel`) at the new rung and runs
   `n_mcmc_steps` moves;
4. estimates the local log normalizing-constant ratio with three estimators.

`smc_sampler` returns the particle block at every rung, the ladder, the
cumulative log Z estimates, and the MALA parameters adapted at every rung.
"""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from numbers import Integral, Real
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, block_until_ready, jit, random, vmap
from jax.scipy.optimize import minimize
from jax.scipy.special import logsumexp, softmax

from progressive_is_tempering.mala_kernel import (
    MALAParticle,
    PTProblem,
    batch_init_extend,
    batch_reset_temperature,
    bisection,
    geometric_anneal,
    mala_kernel,
    repeated_batch_move,
)

__all__ = ["smc_sampler", "one_step_smc"]

_LOGZ_ESTIMATORS = ("SMC_like", "robust", "reverse_logistic")


# ---------------------------------------------------------------------------
# log Z estimators
#
# Each takes (T, N) arrays of reference / final-target log-densities of the
# particles at each of T temperatures, plus the temperatures (0 ... 1), and
# returns an estimate of log(Z_1 / Z_0). SMC calls them with T = 2.
# ---------------------------------------------------------------------------

def _logZ_shape_check(logpdf_ref, logpdf_final_target, temperatures):
    T, N = logpdf_ref.shape
    assert logpdf_final_target.shape == (T, N)
    assert temperatures.shape == (T, )
    return T, N


def _logZ_SMC_like(logpdf_ref, logpdf_final_target, temperatures, full=False):
    """Standard SMC (importance-sampling) estimator."""
    T, N = _logZ_shape_check(logpdf_ref, logpdf_final_target, temperatures)
    logpdf_ref, logpdf_final_target = logpdf_ref[:-1], logpdf_final_target[:-1]
    anneal = vmap(geometric_anneal)
    logpdf_local_base = anneal(temperatures[:-1], logpdf_ref, logpdf_final_target)
    logpdf_local_target = anneal(temperatures[1:], logpdf_ref, logpdf_final_target)
    log_radon_nikodyms = logpdf_local_target - logpdf_local_base
    log_radon_nikodyms = log_radon_nikodyms - jnp.log(N)
    log_Z_locals = logsumexp(log_radon_nikodyms, axis=1)
    assert log_Z_locals.shape == (T-1, )
    if not full:
        return jnp.sum(log_Z_locals)
    else:
        return jnp.r_[0, jnp.cumsum(log_Z_locals)]


def _logZ_robust(logpdf_ref, logpdf_final_target, temperatures):
    """
    Bidirectional stepping-stone estimator (finite variance without tail
    conditions). For each adjacent pair (t_k, t_{k+1}):

      fw_k = log mean_i exp(0.5 * [pi_{t_{k+1}}(x^k_i) - pi_{t_k}(x^k_i)])
      bw_k = log mean_i exp(0.5 * [pi_{t_k}(x^{k+1}_i) - pi_{t_{k+1}}(x^{k+1}_i)])

    and the estimate is sum_k (fw_k - bw_k).
    """
    T, N = _logZ_shape_check(logpdf_ref, logpdf_final_target, temperatures)
    anneal = vmap(geometric_anneal)

    # logpdf_local[k, i] = pi_{t_k}(x^k_i)
    logpdf_local = anneal(temperatures, logpdf_ref, logpdf_final_target)
    assert logpdf_local.shape == (T, N)

    # logpdf_forward[k, i] = pi_{t_{k+1}}(x^k_i); row T-1 is unused +inf padding
    logpdf_forward = jnp.zeros_like(logpdf_local) + jnp.inf
    logpdf_forward = logpdf_forward.at[0:T-1].set(
        anneal(temperatures[1:], logpdf_ref[:-1], logpdf_final_target[:-1])
    )

    # logpdf_backward[k, i] = pi_{t_{k-1}}(x^k_i); row 0 is unused +inf padding
    logpdf_backward = jnp.zeros_like(logpdf_local) + jnp.inf
    logpdf_backward = logpdf_backward.at[1:T].set(
        anneal(temperatures[0:-1], logpdf_ref[1:], logpdf_final_target[1:])
    )

    local_half_estimate_fw = 0.5 * (logpdf_forward[0:-1] - logpdf_local[0:-1])
    local_half_estimate_bw = 0.5 * (logpdf_backward[1:] - logpdf_local[1:])
    local_half_estimate_fw = logsumexp(local_half_estimate_fw, axis=1) - jnp.log(N)
    local_half_estimate_bw = logsumexp(local_half_estimate_bw, axis=1) - jnp.log(N)
    assert local_half_estimate_bw.shape == local_half_estimate_fw.shape == (T-1, )
    return jnp.sum(local_half_estimate_fw - local_half_estimate_bw)


def _logZ_reverse_logistic(logpdf_ref, logpdf_final_target, temperatures):
    """
    Reverse logistic regression estimator (Geyer 1994): minimize a convex
    objective in the log partition functions by BFGS, initialized from the
    SMC estimate.
    """
    T, N = _logZ_shape_check(logpdf_ref, logpdf_final_target, temperatures)
    # L[i, j, k] = pi_{t_i}(x^j_k)
    anneal = vmap(geometric_anneal, in_axes=(0, None, None))
    L = anneal(temperatures, logpdf_ref, logpdf_final_target)
    assert L.shape == (T, T, N)

    def objective(log_Zs):
        assert log_Zs.shape == (T, )
        P = L - log_Zs[:, jnp.newaxis, jnp.newaxis]
        Q = jnp.sum(logsumexp(P, axis=0)) / N
        return Q + jnp.sum(log_Zs)

    init = _logZ_SMC_like(logpdf_ref, logpdf_final_target, temperatures, True)
    res = minimize(objective, init, method='BFGS').x
    return res[-1] - res[0]


_LOGZ_FUNCS = {
    "SMC_like": _logZ_SMC_like,
    "robust": _logZ_robust,
    "reverse_logistic": _logZ_reverse_logistic,
}


def _get_logZ(key, x: MALAParticle, x_next: MALAParticle, t, t_next):
    """
    Local log(Z_{t_next} / Z_t) from the population before (`x`, at `t`) and
    after (`x_next`, at `t_next`) the step, by every estimator. The key is
    split once per estimator for stream compatibility; none of them is random.
    """
    x_old = x.logpdf_target
    x_new = geometric_anneal(t_next, x.logpdf_ref, x.logpdf_final_target)
    x_next_old = geometric_anneal(t, x_next.logpdf_ref, x_next.logpdf_final_target)
    x_next_new = x_next.logpdf_target

    N = x.position.shape[0]
    assert x_old.shape == x_new.shape == x_next_old.shape == \
           x_next_new.shape == (N, )

    arr_old = jnp.array([x_old, x_next_old])
    arr_new = jnp.array([x_new, x_next_new])

    res = {}
    for est in _LOGZ_ESTIMATORS:
        key, subkey = random.split(key)
        res[est] = _LOGZ_FUNCS[est](arr_old, arr_new, jnp.array([0., 1.]))
    return res


# ---------------------------------------------------------------------------
# One SMC step
# ---------------------------------------------------------------------------

def _systematic_resampling_core(key, w):
    """Systematic resampling of normalized weights `w` (no pre-permutation)."""
    N = w.shape[0]
    key_u, key_perm = jax.random.split(key)
    sorted_us = (jnp.arange(N) + jax.random.uniform(key_u)) / N
    # Clip to N-1: floating-point error in cumsum(w) can make the last CDF
    # value slightly < 1.0, causing searchsorted to return N (out of bounds).
    idx = jnp.searchsorted(jnp.cumsum(w), sorted_us)
    return jnp.minimum(idx, N - 1)


def _systematic_resampling(key, w):
    """
    Systematic resampling after a random permutation of the particles, which
    breaks ordering artifacts. Returns indices into the original array.

    Distinct from `progressive_is_tempering.resampling.systematic_resample`
    (log-weights in, no pre-permutation), which the training loop uses.
    """
    N = w.shape[0]
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, N)
    key, subkey = jax.random.split(key)
    return perm[_systematic_resampling_core(subkey, w[perm])]


def _ess_ratio(lw):
    """Normalized ESS `1 / (N * sum W_i^2)` of log-weights; 0 if degenerate."""
    log_total = logsumexp(lw)
    W = jax.lax.cond(
        jnp.isfinite(log_total),
        lambda: jnp.exp(lw - log_total),
        lambda: jnp.zeros_like(lw),
    )
    denom = lw.shape[0] * jnp.sum(W**2)
    return jax.lax.cond(
        denom > 0,
        lambda: 1 / denom,
        lambda: jnp.array(0., dtype=denom.dtype),
    )


def _get_next_temperature(x: MALAParticle, new_ess_ratio, tol):
    """
    Next temperature `t_next > t` at which reweighting the population from
    `t` drops the normalized ESS to `new_ess_ratio` (bisection on [t, 1]).
    Returns 1.0 if the ESS stays above the threshold all the way to t=1, or
    if the bisection lands within 1e-6 of 1.
    """
    t_old = x.temperature[0]
    assert jnp.ndim(t_old) == 0

    def obj(t):
        lw = geometric_anneal(t, x.logpdf_ref, x.logpdf_final_target) - \
            geometric_anneal(t_old, x.logpdf_ref, x.logpdf_final_target)
        return _ess_ratio(lw) - new_ess_ratio

    def _bisect():
        t = bisection(obj, t_old, 1., tol, 30)
        return jax.lax.cond(1.0 - t < 1e-6, lambda: 1.0, lambda: t)

    return jax.lax.cond(obj(1.0) >= 0, lambda: 1.0, _bisect)


def _pytree_index(tree, idx):
    """`leaf[idx]` on every leaf; out-of-range indices fill with NaN / int max."""

    def get_idx(obj):
        fill_value = jnp.nan
        if jnp.issubdtype(obj.dtype, jnp.integer):
            fill_value = jnp.iinfo(jnp.int32).max
        return obj.at[idx].get(mode='fill', fill_value=fill_value)

    return jax.tree.map(get_idx, tree)


class OneStepSMCResult(NamedTuple):
    """
    xnext : MALAParticle
        Resampled-and-moved population at the new temperature.
    logZ_local : dict[str, Array]
        Local log(Z_{t_next} / Z_t), one entry per estimator.
    niter_mcmc : int
        Number of MALA moves applied.
    mcmc_params : MALAParameters
        Kernel parameters adapted at, and used for, the new rung.
    """

    xnext: MALAParticle
    logZ_local: dict
    niter_mcmc: Any
    mcmc_params: Any


def one_step_smc(key, x: MALAParticle, problem: PTProblem, niter_mcmc: int,
                 new_ess_ratio: float = 0.5) -> OneStepSMCResult:
    """
    One SMC annealing step: next temperature, reweight + resample, adapt the
    MALA kernel at the new rung, run `niter_mcmc` moves, estimate local log Z.
    """
    # Choose next temperature
    t_next = _get_next_temperature(x, new_ess_ratio, 1e-2)

    # Resample
    key, subkey = random.split(key)
    lw = geometric_anneal(t_next, x.logpdf_ref, x.logpdf_final_target) - \
        geometric_anneal(x.temperature[0], x.logpdf_ref, x.logpdf_final_target)
    idx = _systematic_resampling(subkey, softmax(lw))
    x_next = _pytree_index(x, idx)

    # Change temperature
    kernel = mala_kernel(problem, t_next, x.temperature_idx[0] + 1)
    x_next = batch_reset_temperature(x_next, kernel)

    # Move with adapted parameters
    key, subkey = random.split(key)
    mcmc_params = kernel.adapt_params(subkey, x_next)
    key, subkey = random.split(key)
    x_next = repeated_batch_move(subkey, x_next, mcmc_params, kernel, niter_mcmc)

    # Get logZ
    key, subkey = random.split(key)
    logZ = _get_logZ(subkey, x, x_next, x.temperature[0], t_next)

    return OneStepSMCResult(x_next, logZ, niter_mcmc, mcmc_params)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _validate_int_count(name: str, value, minimum: int):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum} (got {value!r})")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} (got {value})")
    return value


def _validate_open_unit_interval(name: str, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a float in (0, 1) (got {value!r})")
    value = float(value)
    if not 0. < value < 1.:
        raise ValueError(f"{name} must be a float in (0, 1) (got {value})")
    return value


def smc_sampler(key, problem: PTProblem, n_particles: int, n_mcmc_steps: int,
                local_ess_ratio: float, verbose: bool = True):
    """
    Anneal from t=0 to t=1 with an adaptive temperature ladder.

    Parameters
    ----------
    key : PRNG key
    problem : PTProblem
    n_particles : int
        Population size (constant across rungs).
    n_mcmc_steps : int
        MALA moves per rung after resampling (>= 0).
    local_ess_ratio : float in (0, 1)
        Each new temperature is chosen so that reweighting to it drops the
        normalized ESS to this value.
    verbose : bool
        Print one line per rung (temperature, acceptance, local and cumulative
        log Z by each estimator).

    Returns
    -------
    particles : Array, shape (M+1, n_particles, d)
        Particle positions at every rung, rung 0 being the reference draw.
    logZ : dict[str, Array]
        Cumulative log(Z_1 / Z_0), one entry per estimator.
    temperatures : list
        The ladder `[0., t_1, ..., 1.]` (rungs >= 1 as scalar Arrays).
    mcmc_params : MALAParameters
        Stacked over a leading axis of length M: entry k-1 is the kernel
        adapted and used for the move into rung k (rung 0 has no move).
    """
    n_particles = _validate_int_count('n_particles', n_particles, 1)
    n_mcmc_steps = _validate_int_count('n_mcmc_steps', n_mcmc_steps, 0)
    local_ess_ratio = _validate_open_unit_interval('local_ess_ratio', local_ess_ratio)

    # Initialize and compile
    if verbose:
        print('Initializing and compiling...', flush=True)
    keys = random.split(key, n_particles+1)
    key, subkeys = keys[0], keys[1:]
    x = vmap(problem.rvs_ref)(subkeys)
    key, subkey = random.split(key)
    x = jit(batch_init_extend, static_argnums=(2, ))(subkey, x,
                                                     mala_kernel(problem, 0., 0))
    x = block_until_ready(x)

    mover = jit(partial(one_step_smc, problem=problem, niter_mcmc=n_mcmc_steps,
                        new_ess_ratio=local_ess_ratio))

    # Run
    logZ_ret = defaultdict(float)   # cumulative log(Z_t / Z_0) per estimator
    all_temperatures = [0.]
    all_states = [x.position]
    all_mcmc_params = []
    header_printed = False
    while not jnp.allclose(x.temperature, 1):
        key, subkey = random.split(key)
        x, logZ_local, niter_mcmc, mcmc_params = mover(subkey, x)
        for _k, _v in logZ_local.items():
            logZ_ret[_k] += _v
        if verbose:
            if not header_printed:
                est_order = ', '.join(logZ_local.keys())
                print(f'--- SMC anneal | estimators: ({est_order}) ---')
                print('    dlogZ = log(Z_t / Z_prev)   |   logZ(t) = log(Z_t / Z_0)')
                header_printed = True
            dlogZ_str = '(' + ', '.join(f'{float(v):>9.3f}' for v in logZ_local.values()) + ')'
            cumZ_str  = '(' + ', '.join(f'{float(logZ_ret[k]):>9.3f}' for k in logZ_local.keys()) + ')'
            t_val = float(x.temperature[0])
            t_str = f'{t_val:.2e}' if t_val < 0.001 else f'{t_val:.6f}'
            print(f'iter {int(x.temperature_idx[0]):>3d} | t={t_str} '
                  f'| acc={float(jnp.mean(x.accepted)):.2f} | niter={niter_mcmc:>3d} '
                  f'| dlogZ={dlogZ_str} | logZ(t)={cumZ_str}')
        all_temperatures.append(x.temperature[0])
        all_states.append(x.position)
        all_mcmc_params.append(mcmc_params)

    if verbose:
        est_order = ', '.join(logZ_ret.keys())
        total_str = '(' + ', '.join(f'{float(v):+.4f}' for v in logZ_ret.values()) + ')'
        print(f'--- SMC done | logZ = log(Z_1 / Z_0) = {total_str}   [{est_order}] ---')

    particles = jnp.array(all_states)
    mcmc_params_stacked = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves), *all_mcmc_params
    )
    return particles, logZ_ret, all_temperatures, mcmc_params_stacked
