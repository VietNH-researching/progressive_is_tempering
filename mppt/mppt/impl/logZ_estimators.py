r"""This module implements different estimators of logZ.
"""
from mppt.utils import follow_protocol, dummy_use
from mppt.abstract.pt_concept import LogZ_Estimator, geometric_anneal
from jax.typing import ArrayLike
from jax import Array, numpy as jnp, vmap
from jax.scipy.special import logsumexp
from jax.scipy.optimize import minimize

Float = ArrayLike

@follow_protocol(LogZ_Estimator)
def SMC_like(key, logpdf_ref: Array, logpdf_final_target: Array,
             temperatures: Array, full=False) -> Float:
    r"""Standard SMC estimator for logZ.
    """
    dummy_use(key)
    T, N = LogZ_Estimator.shape_check(logpdf_ref, logpdf_final_target,
                                      temperatures)
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

@follow_protocol(LogZ_Estimator)
def robust(key, logpdf_ref: Array, logpdf_final_target: Array,
           temperatures: Array) -> Float:
    r"""Bidirectional stepping-stone estimator for logZ with guaranteed finite
    variance (no tail conditions required on the distributions).

    For each adjacent pair (t_k, t_{k+1}), computes two half-estimates:
      fw_k = log mean_i exp(0.5 * [pi_{t_{k+1}}(x^k_i) - pi_{t_k}(x^k_i)])
      bw_k = log mean_i exp(0.5 * [pi_{t_k}(x^{k+1}_i) - pi_{t_{k+1}}(x^{k+1}_i)])
    where x^k_i are particles sampled at temperature t_k.

    The total estimate is sum_k (fw_k - bw_k), which telescopes to log(Z_1/Z_0).
    The 0.5 factor and bidirectional structure ensure finite variance even when
    the distributions have heavy tails.
    """
    dummy_use(key)
    T, N = LogZ_Estimator.shape_check(logpdf_ref, logpdf_final_target,
                                      temperatures)
    anneal = vmap(geometric_anneal)

    # logpdf_local[k, i] = pi_{t_k}(x^k_i): each particle evaluated at its own temperature
    logpdf_local = anneal(temperatures, logpdf_ref, logpdf_final_target)
    assert logpdf_local.shape == (T, N)

    # logpdf_forward[k, i] = pi_{t_{k+1}}(x^k_i): particles from level k evaluated at next temperature
    # Only rows 0..T-2 are meaningful; row T-1 is +inf padding (never accessed in slicing below)
    logpdf_forward = jnp.zeros_like(logpdf_local) + jnp.inf
    logpdf_forward = logpdf_forward.at[0:T-1].set(
        anneal(temperatures[1:], logpdf_ref[:-1], logpdf_final_target[:-1])
    )

    # logpdf_backward[k, i] = pi_{t_{k-1}}(x^k_i): particles from level k evaluated at previous temperature
    # Only rows 1..T-1 are meaningful; row 0 is +inf padding (never accessed in slicing below)
    logpdf_backward = jnp.zeros_like(logpdf_local) + jnp.inf
    logpdf_backward = logpdf_backward.at[1:T].set(
        anneal(temperatures[0:-1], logpdf_ref[1:], logpdf_final_target[1:])
    )

    # fw_k uses rows 0..T-2 of forward and local; bw_k uses rows 1..T-1 of backward and local
    local_half_estimate_fw = 0.5 * (logpdf_forward[0:-1] - logpdf_local[0:-1])
    local_half_estimate_bw = 0.5 * (logpdf_backward[1:] - logpdf_local[1:])
    local_half_estimate_fw = logsumexp(local_half_estimate_fw, axis=1) - jnp.log(N)
    local_half_estimate_bw = logsumexp(local_half_estimate_bw, axis=1) - jnp.log(N)
    assert local_half_estimate_bw.shape == local_half_estimate_fw.shape == (T-1, )
    return jnp.sum(local_half_estimate_fw - local_half_estimate_bw)

@follow_protocol(LogZ_Estimator)
def reverse_logistic(key, logpdf_ref: Array, logpdf_final_target: Array,
                     temperatures: Array) -> Float:
    r"""Reverse logistic regression estimator (Geyer 1994). Fits log-partition
    functions by minimizing a convex objective via BFGS. Most stable for many
    temperatures but O(T^2 * N) in memory due to the cross-evaluation matrix.
    Initialized from the SMC_like estimate.
    """
    dummy_use(key)
    T, N = LogZ_Estimator.shape_check(logpdf_ref, logpdf_final_target,
                                      temperatures)
    # L[i, j, k] = pi_{t_i}(x^j_k): particle k from level j evaluated at temperature t_i
    anneal = vmap(geometric_anneal, in_axes=(0, None, None))
    L = anneal(temperatures, logpdf_ref, logpdf_final_target)
    assert L.shape == (T, T, N)

    def objective(log_Zs):
        assert log_Zs.shape == (T, )
        P = L - log_Zs[:, jnp.newaxis, jnp.newaxis]
        Q = jnp.sum(logsumexp(P, axis=0)) / N
        return Q + jnp.sum(log_Zs)

    init = SMC_like(key, logpdf_ref, logpdf_final_target, temperatures, True)
    res = minimize(objective, init, method='BFGS').x
    return res[-1] - res[0]