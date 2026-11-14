"""
HMC with radian-circle coordinates, and the adaptive SMC that builds its ladder.

The ALDP notebook runs its MCMC in internal coordinates whose 18 torsions are
circles. This module is the kernel and SMC it uses:

* `hmc_kernel(problem, t, t_idx, refresh_rate, periodic_indices)` — one
  leapfrog step (half position, full velocity, half position), Metropolis
  accept/reject on position and velocity, then a partial velocity refresh
  `v <- -sqrt(1-r) v + sqrt(r) z`. Coordinates in `periodic_indices` are
  wrapped to [-pi, pi) after every position update. Identity preconditioner;
  `adapt_params` tunes only the step size, on a log scale, to an acceptance
  rate in [0.8, 0.95]. At rung `t_idx == 0` a move draws fresh from the
  reference instead (the reference is sampled exactly).
* `smc_sampler` — anneals a population from the reference (t=0) to the target
  (t=1): ESS-bisection for the next temperature, systematic resampling,
  step-size adaptation at the new rung, `n_mcmc_steps` moves, and three local
  log Z estimates. Returns the particle block at every rung and the ladder.

Order-sensitive: seemingly equivalent rewrites change the random stream or
the floating-point results, so keep expression order and key splits as they
are.

Differences from the package's own `smc.py` / `hmc_kernel.py`: circles,
identity preconditioning with scalar step-size search (`log_bisection`
instead of covariance + bisection), particles that carry log-density *values
and gradients* of both endpoints, a 64-step temperature bisection, and no
per-rung kernel parameters in `smc_sampler`'s return value (the notebook
re-adapts them on each rung's SMC block).

Leaf module: imports only JAX.
"""

from __future__ import annotations

from collections import defaultdict, namedtuple
from functools import partial
from numbers import Integral, Real
from typing import Any, Callable, NamedTuple

import jax
from jax import Array, block_until_ready, jit, lax, random, value_and_grad, vmap
from jax import numpy as jnp
from jax.scipy.optimize import minimize
from jax.scipy.special import logsumexp, softmax

__all__ = [
    "PTProblem",
    "HMCParticle",
    "HMCParameters",
    "FixedCovMVNormal",
    "hmc_kernel",
    "make_engine",
    "batch_init_extend",
    "batch_move",
    "repeated_batch_move",
    "batch_reset_temperature",
    "wrap_periodic",
    "geometric_anneal",
    "smc_sampler",
    "one_step_smc",
]


# ---------------------------------------------------------------------------
# Problem, annealing path, particle and parameter types
# ---------------------------------------------------------------------------

class PTProblem(NamedTuple):
    """Reference sampler and the two endpoint log densities of the geometric
    path log pi_t = (1-t) logpdf_ref + t logpdf_final_target, all on (d,).
    `exact_ref`: `rvs_ref` draws exact reference samples, so rung 0 is
    refreshed with fresh draws rather than moved."""
    rvs_ref: Callable
    logpdf_ref: Callable
    logpdf_final_target: Callable
    exact_ref: bool = True


def geometric_anneal(t, logpdf_0, logpdf_1):
    """`(1-t) logpdf_0 + t logpdf_1`; a -inf endpoint does not leak into the
    other endpoint's value (t = 0 or 1). NaN / +inf stay arithmetic errors."""
    ref = jnp.where((t == 1) & jnp.isneginf(logpdf_0), 0., logpdf_0)
    target = jnp.where((t == 0) & jnp.isneginf(logpdf_1), 0., logpdf_1)
    return (1-t) * ref + t * target


class HMCParticle(NamedTuple):
    """One particle (or, with a leading axis, a population). Carries the log
    density and gradient of both endpoints, so a temperature change needs no
    re-evaluation."""
    position: Array
    logpdf_ref: Any
    logpdf_final_target: Any
    gradlogpdf_ref: Array
    gradlogpdf_final_target: Array
    temperature: Any
    temperature_idx: Any
    logpdf_target: Any
    gradlogpdf_target: Array
    veloc: Array
    accepted: Any


_fields = list(HMCParticle._fields)
del _fields[_fields.index('veloc')]
_HMCParticleNoVeloc = namedtuple('_HMCParticleNoVeloc', _fields)


class FixedCovMVNormal(NamedTuple):
    """Preconditioner `chol` (Cholesky factor of `cov`) and step size `eps`."""
    eps: Any
    cov: Array
    chol: Array
    log_det_chol: Any
    chol_failed: Any


def _chol_diag(m):
    return jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(m), 1e-30)))


def fixed_cov_mv_normal(eps, cov: Array) -> FixedCovMVNormal:
    """Symmetrize `cov` and take its Cholesky factor (the diagonal's square
    root if `cov` is not positive definite)."""
    d, _d = cov.shape
    assert _d == d
    cov = (cov + cov.T) / 2
    eigvals = jnp.linalg.eigvalsh(cov)
    chol_failed = jnp.logical_or(
        jnp.logical_not(jnp.all(jnp.isfinite(eigvals))),
        jnp.min(eigvals) <= 0.,
    )
    chol = lax.cond(
        chol_failed,
        lambda: _chol_diag(cov),
        lambda: jnp.linalg.cholesky(cov),
    )
    log_det_chol = jnp.sum(jnp.log(jnp.diag(chol)))
    return FixedCovMVNormal(eps, cov, chol, log_det_chol, chol_failed)


class HMCParameters(NamedTuple):
    """`scaled_cov`: preconditioner and step size; `refresh_rate` r of the
    partial velocity refresh."""
    scaled_cov: FixedCovMVNormal
    refresh_rate: Any


def wrap_periodic(position, indices):
    """Put the radian-circle axes `indices` (static tuple) into [-pi, pi)."""
    if not isinstance(indices, tuple) or any(
            isinstance(i, bool) or not isinstance(i, Integral) for i in indices):
        raise TypeError("periodic_indices must be a tuple of integer indices.")
    if not indices:
        return position
    index = jnp.asarray(indices, dtype=jnp.int32)
    angles = position[..., index]
    return position.at[..., index].set((angles+jnp.pi) % (2*jnp.pi) - jnp.pi)


# ---------------------------------------------------------------------------
# Step-size search
# ---------------------------------------------------------------------------

_log_bisection_type = namedtuple('_log_bisection_type',
                                 ['log_lo', 'log_hi', 'x_best', 'f_best', 'done'])


def log_bisection(f, x_low, x_high, ftol, n_eval, x_first=None):
    """Root of a decreasing `f` on [x_low, x_high] by bisection on log x, at
    most `n_eval` evaluations, the first at `x_first`. Returns the evaluated
    point with the smallest |f| (ties: the later one), stopping at the first
    with |f| <= ftol. Never returns an endpoint such as 0."""
    assert n_eval >= 1

    def probe(log_lo, log_hi, log_x):
        x = jnp.exp(log_x)
        fx = f(x)
        above = fx > 0  # f decreasing: the root lies above x
        return _log_bisection_type(jnp.where(above, log_x, log_lo), jnp.where(above, log_hi, log_x),
                                   x, fx, jnp.abs(fx) <= ftol)

    def body(_, s: _log_bisection_type) -> _log_bisection_type:
        def step():
            new = probe(s.log_lo, s.log_hi, 0.5 * (s.log_lo + s.log_hi))
            better = jnp.abs(new.f_best) <= jnp.abs(s.f_best)
            return new._replace(x_best=jnp.where(better, new.x_best, s.x_best),
                                f_best=jnp.where(better, new.f_best, s.f_best))
        return lax.cond(s.done, lambda: s, step)

    log_lo, log_hi = jnp.log(x_low), jnp.log(x_high)
    log_first = 0.5 * (log_lo + log_hi) if x_first is None else jnp.log(x_first)
    init = probe(log_lo, log_hi, log_first)
    return lax.fori_loop(1, n_eval, body, init).x_best


def _stepsize_adapter(key, sample, stepsize_to_params, mcmc, target_acc_range=(0.8, 0.95),
                      niter_max: int = 10):
    """Tune eps so that one move of `sample` (same key for every trial) has a
    mean acceptance in `target_acc_range`; first guess 1.65 / d^(1/6), search
    range [2^-20, 2^6] times it."""
    N, d = sample.position.shape
    eps_upper = 1.65 / d**(1/6)
    acc_default = (target_acc_range[0] + target_acc_range[1])/2
    acc_delta = abs(target_acc_range[1] - target_acc_range[0])/2

    def f(eps):
        params = stepsize_to_params(eps)
        acc = batch_move(key, sample, params, mcmc).accepted  # reuse key for smoothness
        return jnp.mean(acc) - acc_default

    eps_star = log_bisection(f, eps_upper * 2.**-20, eps_upper * 2.**6, acc_delta, niter_max,
                             x_first=eps_upper)
    return stepsize_to_params(eps_star)


# ---------------------------------------------------------------------------
# HMC
# ---------------------------------------------------------------------------

class _HMCCore(NamedTuple):
    """The HMC move on extended particles; `feed_extend(x)` returns every
    field but the velocity."""
    feed_extend: Callable
    refresh_rate: float = 0.07
    periodic_indices: tuple = ()

    def init_extend(self, key, x):
        assert x.ndim == 1
        x = wrap_periodic(x, self.periodic_indices)
        return HMCParticle(veloc=random.normal(key, shape=x.shape),
                           **self.feed_extend(x)._asdict())

    def inner_extend(self, x, v):
        assert x.ndim == v.ndim == 1
        assert x.shape == v.shape
        x = wrap_periodic(x, self.periodic_indices)
        return HMCParticle(veloc=v, **self.feed_extend(x)._asdict())

    @staticmethod
    def _joint_logpdf(x):
        return x.logpdf_target - 0.5 * jnp.sum(x.veloc**2)

    def move(self, key, x, params: HMCParameters):
        """Leapfrog jump, Metropolis accept/reject, velocity refresh. Only
        position and velocity are selected; every other field is rebuilt from
        the chosen position."""
        x_new = self._hmc_jump(x, params)
        log_acc_rate = self._joint_logpdf(x_new) - self._joint_logpdf(x)
        key, subkey = random.split(key)
        log_u = jnp.log(random.uniform(subkey))
        accept = log_u < log_acc_rate
        position = jnp.where(accept, x_new.position, x.position)
        veloc = jnp.where(accept, x_new.veloc, x.veloc)
        x_ret = self.inner_extend(position, veloc)._replace(accepted=accept)
        key, subkey = random.split(key)
        return self._hmc_refresh_veloc(subkey, x_ret, params.refresh_rate)

    def _hmc_jump(self, x, params):
        x = self._hmc_jump_pos(x, params)
        x = self._hmc_jump_veloc(x, params)
        x = self._hmc_jump_veloc(x, params)
        x = self._hmc_jump_pos(x, params)
        return x._replace(veloc=-x.veloc)

    def _hmc_jump_pos(self, x, params):
        pos_new = x.position + \
                  0.5 * params.scaled_cov.eps * params.scaled_cov.chol @ x.veloc
        return self.inner_extend(pos_new, x.veloc)

    def _hmc_jump_veloc(self, x, params):
        return x._replace(veloc=x.veloc + (
            0.5*params.scaled_cov.eps*params.scaled_cov.chol.T@x.gradlogpdf_target
        ))

    @staticmethod
    def _hmc_refresh_veloc(key, x, refresh_rate):
        z = random.normal(key, x.position.shape)
        v_new = -jnp.sqrt(1-refresh_rate) * x.veloc + jnp.sqrt(refresh_rate) * z
        return x._replace(veloc=v_new)

    def adapt_params(self, key, x) -> HMCParameters:
        """Identity preconditioner, step size by log-bisection on acceptance."""
        emp_cov = jnp.eye(x.position.shape[-1])
        emp_cov = fixed_cov_mv_normal(eps=1, cov=emp_cov)

        def stepsize_to_params(eps):
            cov_obj = emp_cov._replace(eps=eps)
            return HMCParameters(cov_obj, self.refresh_rate)

        return _stepsize_adapter(key, x, stepsize_to_params, self, (0.8, 0.95), niter_max=10)


class HMCKernel(NamedTuple):
    """The kernel at one rung (hashable: usable as a static jit argument)."""
    init_extend: Callable
    move: Callable
    adapt_params: Callable
    reset_temperature: Callable


def hmc_kernel(problem: PTProblem, t, t_idx, refresh_rate: float = 0.07,
               periodic_indices: tuple = ()) -> HMCKernel:
    """The HMC kernel at rung `t_idx`, temperature `t` (either may be traced).

    `init_extend(key, x)`, `move(key, particle, params)`, `adapt_params(key,
    particles)`, `reset_temperature(particle)`. It is assumed that t = 0 iff
    t_idx = 0; there, with an exact reference, `move` draws from it.
    """
    if not isinstance(problem.exact_ref, bool):
        raise TypeError("PTProblem.exact_ref must be a Python bool")

    def feed_extend(x):
        assert jnp.ndim(x) == 1
        logpdf_ref, gradlogpdf_ref = value_and_grad(problem.logpdf_ref)(x)
        logpdf_final_target, gradlogpdf_final_target = \
            value_and_grad(problem.logpdf_final_target)(x)
        logpdf_target = geometric_anneal(t, logpdf_ref, logpdf_final_target)
        gradlogpdf_target = geometric_anneal(t, gradlogpdf_ref, gradlogpdf_final_target)
        return _HMCParticleNoVeloc(
            position=x, logpdf_ref=logpdf_ref, gradlogpdf_ref=gradlogpdf_ref,
            logpdf_final_target=logpdf_final_target,
            gradlogpdf_final_target=gradlogpdf_final_target,
            temperature=t, temperature_idx=t_idx,
            logpdf_target=logpdf_target, gradlogpdf_target=gradlogpdf_target,
            accepted=True
        )

    core = _HMCCore(feed_extend=feed_extend, refresh_rate=refresh_rate,
                    periodic_indices=periodic_indices)

    def new_from_ref(key):
        key_ref, key_extend = random.split(key)
        x = problem.rvs_ref(key_ref)
        return core.init_extend(key_extend, x)

    def move(key, x, params: HMCParameters):
        assert jnp.ndim(x.temperature) == jnp.ndim(x.temperature_idx) == 0
        if not problem.exact_ref:  # Python-level: no level-0 branch is traced
            return core.move(key, x, params)
        return lax.cond(t_idx == 0,
                        lambda: new_from_ref(key),
                        lambda: core.move(key, x, params))

    def reset_temperature(x):
        return x._replace(
            temperature=t, temperature_idx=t_idx,
            logpdf_target=geometric_anneal(t, x.logpdf_ref, x.logpdf_final_target),
            gradlogpdf_target=geometric_anneal(t, x.gradlogpdf_ref, x.gradlogpdf_final_target)
        )

    return HMCKernel(init_extend=core.init_extend, move=move, adapt_params=core.adapt_params,
                     reset_temperature=reset_temperature)


def make_engine(refresh_rate: float = 0.07, periodic_indices: tuple = ()):
    """`engine(problem, t, t_idx) -> HMCKernel` with fixed refresh rate and circles."""
    return partial(hmc_kernel, refresh_rate=refresh_rate, periodic_indices=periodic_indices)


# ---------------------------------------------------------------------------
# Population moves
# ---------------------------------------------------------------------------

def _pytree_length(tree):
    leaves = jax.tree.leaves(tree)
    lengths = {leaf.shape[0] for leaf in leaves}
    assert len(lengths) == 1, f'pytree leaves have different lengths: {lengths}'
    return len(leaves[0])


def batch_move(key, x, params, mcmc):
    """One move of every particle (vmap), one key each."""
    N = _pytree_length(x)
    subkeys = random.split(key, N)

    @vmap
    def mover(_k, _x):
        return mcmc.move(_k, _x, params)

    return mover(subkeys, x)


def _rbm_body_func(i, val, params, mcmc):
    key, x = val
    key, subkey = random.split(key)
    x = batch_move(subkey, x, params, mcmc)
    return key, x


def repeated_batch_move(key, x, params, mcmc, nrepeat: int):
    """`nrepeat` population moves in a `lax.fori_loop`."""
    body_func = partial(_rbm_body_func, params=params, mcmc=mcmc)
    val = (key, x)
    return lax.fori_loop(0, nrepeat, body_func, val)[1]


def batch_init_extend(key, x, mcmc):
    """Extend positions (N, d) to particles; one key per particle (velocity)."""
    subkeys = random.split(key, _pytree_length(x))
    return vmap(mcmc.init_extend)(subkeys, x)


def batch_reset_temperature(x, mcmc):
    return vmap(mcmc.reset_temperature)(x)


# ---------------------------------------------------------------------------
# SMC: resampling, temperature search, log Z
# ---------------------------------------------------------------------------

def _systematic_resampling_core(key, w):
    N = w.shape[0]
    key_u, key_perm = jax.random.split(key)
    sorted_us = (jnp.arange(N) + jax.random.uniform(key_u)) / N
    # Clip to N-1: floating-point error in cumsum(w) can leave its last value < 1.
    idx = jnp.searchsorted(jnp.cumsum(w), sorted_us)
    return jnp.minimum(idx, N - 1)


def _systematic_resampling(key, w):
    """Systematic resampling of normalized weights after a random
    permutation; indices into the original array."""
    N = w.shape[0]
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, N)
    key, subkey = jax.random.split(key)
    return perm[_systematic_resampling_core(subkey, w[perm])]


def _ess_ratio(lw):
    """Normalized ESS of log-weights; 0 if degenerate."""
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


_bisection_type = namedtuple('_bisection_type', ['xleft', 'xright', 'fleft', 'fright', 'niter'])


def _bisection(f, xleft_, xright_, ftol, max_iter):
    """Bisection for a root of a monotone `f`; returns the left end."""
    def create(xleft, xright, fleft, fright, niter):
        return _bisection_type(xleft, xright, fleft, fright, niter)

    def loop(s):
        xleft, xright, fleft, fright, niter = s
        return lax.cond(
            jnp.sign(fleft) * jnp.sign(fright) >= 0,
            lambda: same_sign(s), lambda: different_sign(s)
        )

    def same_sign(s):
        xleft, xright, fleft, fright, niter = s
        return lax.cond(
            jnp.abs(fleft) >= jnp.abs(fright),
            lambda: create(xright, 2*xright-xleft, fright, f(2*xright-xleft), niter+1),
            lambda: create(2*xleft-xright, xleft, f(2*xleft-xright), fleft, niter+1)
        )

    def different_sign(s):
        xleft, xright, fleft, fright, niter = s
        xmid = 0.5 * xleft + 0.5 * xright
        fmid = f(xmid)
        return lax.cond(
            jnp.sign(fleft) * jnp.sign(fmid) <= 0,
            lambda: create(xleft, xmid, fleft, fmid, niter+1),
            lambda: create(xmid, xright, fmid, fright, niter+1)
        )

    def cond(s):
        cond_1 = jnp.abs(s.fleft) > ftol
        cond_2 = s.niter < max_iter
        return jnp.logical_and(cond_1, cond_2)

    init_val = create(xleft_, xright_, f(xleft_), f(xright_), 0)
    return lax.while_loop(cond, loop, init_val).xleft


def _get_next_temperature(x, new_ess_ratio, tol):
    """Next t > t_old at which reweighting drops the ESS ratio to
    `new_ess_ratio`; 1 if it stays above all the way, or lands within 1e-6."""
    t_old = x.temperature[0]
    assert jnp.ndim(t_old) == 0

    def obj(t):
        lw = geometric_anneal(t, x.logpdf_ref, x.logpdf_final_target) -\
            geometric_anneal(t_old, x.logpdf_ref, x.logpdf_final_target)
        return _ess_ratio(lw) - new_ess_ratio

    def _bisect():
        t = _bisection(obj, t_old, 1., tol, 64)
        return jax.lax.cond(1.0 - t < 1e-6, lambda: 1.0, lambda: t)
    return jax.lax.cond(obj(1.0) >= 0, lambda: 1.0, _bisect)


def _pytree_index(tree, idx):
    def get_idx(obj):
        fill_value = jnp.nan
        if jnp.issubdtype(obj.dtype, jnp.integer):
            fill_value = jnp.iinfo(jnp.int32).max
        return obj.at[idx].get(mode='fill', fill_value=fill_value)
    return jax.tree.map(get_idx, tree)


def _logZ_shape_check(logpdf_ref, logpdf_final_target, temperatures):
    T, N = logpdf_ref.shape
    assert logpdf_final_target.shape == (T, N)
    assert temperatures.shape == (T, )
    return T, N


def _logZ_SMC_like(key, logpdf_ref, logpdf_final_target, temperatures, full=False):
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


def _logZ_robust(key, logpdf_ref, logpdf_final_target, temperatures):
    """Bidirectional stepping-stone estimator (finite variance)."""
    T, N = _logZ_shape_check(logpdf_ref, logpdf_final_target, temperatures)
    anneal = vmap(geometric_anneal)
    logpdf_local = anneal(temperatures, logpdf_ref, logpdf_final_target)
    assert logpdf_local.shape == (T, N)
    logpdf_forward = jnp.zeros_like(logpdf_local) + jnp.inf
    logpdf_forward = logpdf_forward.at[0:T-1].set(
        anneal(temperatures[1:], logpdf_ref[:-1], logpdf_final_target[:-1])
    )
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


def _logZ_reverse_logistic(key, logpdf_ref, logpdf_final_target, temperatures):
    """Reverse logistic regression (Geyer 1994), BFGS from the SMC estimate."""
    T, N = _logZ_shape_check(logpdf_ref, logpdf_final_target, temperatures)
    anneal = vmap(geometric_anneal, in_axes=(0, None, None))
    L = anneal(temperatures, logpdf_ref, logpdf_final_target)
    assert L.shape == (T, T, N)

    def objective(log_Zs):
        assert log_Zs.shape == (T, )
        P = L - log_Zs[:, jnp.newaxis, jnp.newaxis]
        Q = jnp.sum(logsumexp(P, axis=0)) / N
        return Q + jnp.sum(log_Zs)

    init = _logZ_SMC_like(key, logpdf_ref, logpdf_final_target, temperatures, True)
    res = minimize(objective, init, method='BFGS').x
    return res[-1] - res[0]


_LOGZ_ESTIMATORS = {"SMC_like": _logZ_SMC_like, "robust": _logZ_robust,
                    "reverse_logistic": _logZ_reverse_logistic}


def _get_logZ(key, x, x_next, t, t_next):
    """Local log(Z_{t_next} / Z_t) from the populations before and after the step."""
    x_old = x.logpdf_target
    x_new = geometric_anneal(t_next, x.logpdf_ref, x.logpdf_final_target)
    x_next_old = geometric_anneal(t, x_next.logpdf_ref, x_next.logpdf_final_target)
    x_next_new = x_next.logpdf_target

    N = _pytree_length(x)
    assert x_old.shape == x_new.shape == x_next_old.shape ==\
           x_next_new.shape == (N, )

    arr_old = jnp.array([x_old, x_next_old])
    arr_new = jnp.array([x_new, x_next_new])

    res = {}
    for est in ('SMC_like', 'robust', 'reverse_logistic'):
        key, subkey = random.split(key)
        res[est] = _LOGZ_ESTIMATORS[est](subkey, arr_old, arr_new, jnp.array([0., 1.]))
    return res


class OneStepSMCResult(NamedTuple):
    xnext: HMCParticle
    logZ_local: dict
    niter_mcmc: int


def one_step_smc(key, x, problem: PTProblem, mcmc_engine, niter_mcmc: int,
                 new_ess_ratio: float = 0.5) -> OneStepSMCResult:
    """Next temperature, reweight + resample, adapt the step size at the new
    rung, `niter_mcmc` moves, local log Z."""
    t_next = _get_next_temperature(x, new_ess_ratio, 1e-2)

    key, subkey = random.split(key)
    lw = geometric_anneal(t_next, x.logpdf_ref, x.logpdf_final_target) -\
        geometric_anneal(x.temperature[0], x.logpdf_ref, x.logpdf_final_target)
    idx = _systematic_resampling(subkey, softmax(lw))
    x_next = _pytree_index(x, idx)

    mcmc = mcmc_engine(problem, t_next, x.temperature_idx[0] + 1)
    x_next = batch_reset_temperature(x_next, mcmc)

    key, subkey = random.split(key)
    mcmc_params = mcmc.adapt_params(subkey, x_next)
    key, subkey = random.split(key)
    x_next = repeated_batch_move(subkey, x_next, mcmc_params, mcmc, niter_mcmc)

    key, subkey = random.split(key)
    logZ = _get_logZ(subkey, x, x_next, x.temperature[0], t_next)
    return OneStepSMCResult(x_next, logZ, niter_mcmc)


def smc_sampler(key, problem: PTProblem, n_particles: int, engine, n_mcmc_steps: int,
                local_ess_ratio: float, verbose: bool = True, keep_all_states: bool = False,
                *, max_stages: int = 1000):
    """Anneal `n_particles` from the reference (t=0) to the target (t=1).

    `engine(problem, t, t_idx) -> HMCKernel` (`make_engine`). Each stage picks
    the temperature at which reweighting drops the normalized ESS to
    `local_ess_ratio`, resamples, adapts the step size and makes
    `n_mcmc_steps` moves.

    Returns `(x, logZ, temperatures)`: `x` the final particles, or with
    `keep_all_states` the positions at every rung, shape (M+1, N, d); `logZ`
    the cumulative log(Z_1 / Z_0) by estimator; the ladder `[0., t_1, ..., 1.]`.
    """
    if isinstance(n_particles, bool) or not isinstance(n_particles, Integral) or n_particles < 1:
        raise ValueError(f"n_particles must be an integer >= 1 (got {n_particles!r})")
    if isinstance(n_mcmc_steps, bool) or not isinstance(n_mcmc_steps, Integral) or n_mcmc_steps < 0:
        raise ValueError(f"n_mcmc_steps must be an integer >= 0 (got {n_mcmc_steps!r})")
    if isinstance(local_ess_ratio, bool) or not isinstance(local_ess_ratio, Real) \
            or not 0. < float(local_ess_ratio) < 1.:
        raise ValueError(f"local_ess_ratio must be a float in (0, 1) (got {local_ess_ratio!r})")
    n_particles, n_mcmc_steps, local_ess_ratio = int(n_particles), int(n_mcmc_steps), float(local_ess_ratio)

    if verbose:
        print('Initializing and compiling...', flush=True)
    keys = jax.random.split(key, n_particles+1)
    key, subkeys = keys[0], keys[1:]
    x = vmap(problem.rvs_ref)(subkeys)
    key, subkey = random.split(key)
    x = jit(batch_init_extend, static_argnums=(2, ))(subkey, x, engine(problem, 0., 0))
    x = block_until_ready(x)

    mover = jit(partial(one_step_smc, problem=problem, mcmc_engine=engine, niter_mcmc=n_mcmc_steps,
                        new_ess_ratio=local_ess_ratio))

    logZ_ret = defaultdict(float)   # cumulative log(Z_t / Z_0) per estimator
    all_temperatures = [0.]
    all_states = [x.position]
    header_printed = False
    while float(x.temperature[0]) != 1.:
        if len(all_temperatures)-1 >= max_stages:
            raise RuntimeError(f"SMC exceeded max_stages={max_stages} at t={float(x.temperature[0])}")
        previous_t = float(x.temperature[0])
        key, subkey = random.split(key)
        x, logZ_local, niter_mcmc = mover(subkey, x)
        next_t = float(x.temperature[0])
        if not previous_t < next_t <= 1.:
            raise RuntimeError(f"SMC temperature failed to advance in (t, 1]: {previous_t} -> {next_t}")
        if not bool(jnp.all(jnp.isfinite(x.position))) or not bool(jnp.all(jnp.isfinite(x.logpdf_target))):
            raise RuntimeError("SMC produced invalid positions or current log densities")
        for _k, _v in logZ_local.items():
            logZ_ret[_k] += _v
        if verbose:
            if not header_printed:
                est_order = ', '.join(logZ_local.keys())
                print(f'--- SMC anneal | estimators: ({est_order}) ---', flush=True)
                print('    dlogZ = log(Z_t / Z_prev)   |   logZ(t) = log(Z_t / Z_0)', flush=True)
                header_printed = True
            dlogZ_str = '(' + ', '.join(f'{float(v):>9.3f}' for v in logZ_local.values()) + ')'
            cumZ_str = '(' + ', '.join(f'{float(logZ_ret[k]):>9.3f}' for k in logZ_local.keys()) + ')'
            t_val = float(x.temperature[0])
            t_str = f'{t_val:.2e}' if t_val < 0.001 else f'{t_val:.6f}'
            print(f'iter {int(x.temperature_idx[0]):>3d} | t={t_str} '
                  f'| acc={float(jnp.mean(x.accepted)):.2f} | niter={niter_mcmc:>3d} '
                  f'| dlogZ={dlogZ_str} | logZ(t)={cumZ_str}', flush=True)
        all_temperatures.append(x.temperature[0])
        if keep_all_states:
            all_states.append(x.position)

    if verbose:
        est_order = ', '.join(logZ_ret.keys())
        total_str = '(' + ', '.join(f'{float(v):+.4f}' for v in logZ_ret.values()) + ')'
        print(f'--- SMC done | logZ = log(Z_1 / Z_0) = {total_str}   [{est_order}] ---', flush=True)

    if keep_all_states:
        x = jnp.array(all_states)
    return x, logZ_ret, all_temperatures
