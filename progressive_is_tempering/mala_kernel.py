"""
Covariance-preconditioned MALA kernel on the geometric tempered path.

The kernel targets

    log pi_t(x) = (1 - t) * logpdf_ref(x) + t * logpdf_final_target(x)

and is written as HMC with a single Verlet step and a full velocity refresh
(refresh rate 1), which is exactly MALA. Its preconditioner is the empirical
covariance of a particle population, and its step size is bisected towards an
acceptance rate in [0.4, 0.65].

This is the kernel `progressive_is_tempering.smc.smc_sampler` adapts at every
rung while building the ladder; the adapted `MALAParameters` are then reused
by `progressive_is_tempering.rejuvenate.rejuvenate` in the outer loop.

Particles carry both the reference and the final-target log-density (and
gradient), so moving a particle to a new temperature (`reset_temperature`)
needs no re-evaluation of either density.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax, random, value_and_grad, vmap

__all__ = [
    "PTProblem",
    "geometric_anneal",
    "ScaledCov",
    "MALAParameters",
    "MALAParticle",
    "MALAKernel",
    "mala_kernel",
    "batch_init_extend",
    "batch_move",
    "repeated_batch_move",
    "batch_reset_temperature",
    "bisection",
]


class PTProblem(NamedTuple):
    """
    A tempering problem: the path from a reference (t=0) to a target (t=1).

    rvs_ref : key -> x of shape (d,)
        One draw from the reference distribution.
    logpdf_ref : x -> scalar
        Log-density of the reference.
    logpdf_final_target : x -> scalar
        Log-density of the target (may be unnormalized).

    All three must trace under jit/grad/vmap. Instances are passed as static
    arguments, so the same object (by identity) should be reused across calls.
    """

    rvs_ref: Callable[[Array], Array]
    logpdf_ref: Callable[[Array], Array]
    logpdf_final_target: Callable[[Array], Array]


def geometric_anneal(t, logpdf_0, logpdf_1):
    """Geometric annealing: `(1 - t) * logpdf_0 + t * logpdf_1`."""
    return (1 - t) * logpdf_0 + t * logpdf_1


class ScaledCov(NamedTuple):
    """
    Proposal preconditioner: covariance `eps**2 * cov` with Cholesky factor
    `chol` of `cov`. If `cov` is not positive definite, `chol` falls back to
    the square root of its diagonal and `chol_failed` is True.
    """

    eps: Any
    cov: Array
    chol: Array
    log_det_chol: Any
    chol_failed: Any


class MALAParameters(NamedTuple):
    """
    MALA kernel parameters.

    scaled_cov : ScaledCov
        Step size `eps` and preconditioner.
    refresh_rate : float
        Velocity refresh rate; always 1.0 (full refresh = MALA).
    """

    scaled_cov: ScaledCov
    refresh_rate: Any


class MALAParticle(NamedTuple):
    """
    One particle (or, with a leading axis on every field, a population).

    `logpdf_target` / `gradlogpdf_target` refer to the particle's *current*
    temperature; `logpdf_final_target` is the t=1 density.
    """

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


# Same fields as MALAParticle, without the velocity.
_particle_fields = list(MALAParticle._fields)
del _particle_fields[_particle_fields.index("veloc")]
_MALAParticleNoVeloc = namedtuple("_MALAParticleNoVeloc", _particle_fields)


class MALAKernel(NamedTuple):
    """
    The MALA kernel at one rung `(t, t_idx)`. Built by `mala_kernel`.

    init_extend : (key, x) -> MALAParticle
        Wrap a position into a particle with a fresh N(0, I) velocity.
    move : (key, particle, params) -> MALAParticle
        One MALA step. At `t_idx == 0` it instead draws a fresh particle from
        the reference, ignoring the input.
    adapt_params : (key, particles) -> MALAParameters
        Empirical-covariance preconditioner + bisected step size.
    reset_temperature : particle -> MALAParticle
        Re-anneal a particle's cached densities to this rung.
    """

    init_extend: Callable
    move: Callable
    adapt_params: Callable
    reset_temperature: Callable


def _pytree_length(tree) -> int:
    """Length of the leading axis shared by every leaf of `tree`."""
    leaves = jax.tree.leaves(tree)
    n = leaves[0].shape[0]
    assert all(leaf.shape[:1] == (n,) for leaf in leaves), (
        "all leaves must share the same leading dimension"
    )
    return n


_bisection_type = namedtuple(
    "_bisection_type", ["xleft", "xright", "fleft", "fright", "niter"]
)


def bisection(f, xleft_, xright_, ftol, max_iter):
    """
    Root of a monotonic `f` by bisection, returning the left bracket.

    `f(xleft_)` and `f(xright_)` need not have opposite signs: while they
    share a sign, the bracket is shifted by its own width towards the smaller
    `|f|`. Stops once `|f(xleft)| <= ftol` or after `max_iter` updates.
    """

    def create(xleft, xright, fleft, fright, niter):
        return _bisection_type(xleft, xright, fleft, fright, niter)

    def loop(s: _bisection_type) -> _bisection_type:
        xleft, xright, fleft, fright, niter = s
        return lax.cond(
            jnp.sign(fleft) * jnp.sign(fright) >= 0,
            lambda: same_sign(s), lambda: different_sign(s)
        )

    def same_sign(s: _bisection_type) -> _bisection_type:
        xleft, xright, fleft, fright, niter = s
        return lax.cond(
            jnp.abs(fleft) >= jnp.abs(fright),
            lambda: create(xright, 2*xright-xleft, fright, f(2*xright-xleft), niter+1),
            lambda: create(2*xleft-xright, xleft, f(2*xleft-xright), fleft, niter+1)
        )

    def different_sign(s: _bisection_type) -> _bisection_type:
        xleft, xright, fleft, fright, niter = s
        xmid = 0.5 * xleft + 0.5 * xright
        fmid = f(xmid)
        return lax.cond(
            jnp.sign(fleft) * jnp.sign(fmid) <= 0,
            lambda: create(xleft, xmid, fleft, fmid, niter+1),
            lambda: create(xmid, xright, fmid, fright, niter+1)
        )

    def cond(s: _bisection_type):
        cond_1 = jnp.abs(s.fleft) > ftol
        cond_2 = s.niter < max_iter
        return jnp.logical_and(cond_1, cond_2)

    init_val = create(xleft_, xright_, f(xleft_), f(xright_), 0)
    return lax.while_loop(cond, loop, init_val).xleft


def _chol_diag(m):
    # Floor diagonal at a small epsilon to avoid log(0) = -inf in log_det_chol
    return jnp.diag(jnp.sqrt(jnp.maximum(jnp.diag(m), 1e-30)))


def _scaled_cov(eps, cov: Array) -> ScaledCov:
    """Build a `ScaledCov` from a (symmetrized) covariance matrix."""
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

    return ScaledCov(eps, cov, chol, log_det_chol, chol_failed)


def _vmapped_move(move_fn, key, x, params):
    """Apply a single-particle `move_fn` to every particle, one key each."""
    N = _pytree_length(x)
    subkeys = random.split(key, N)

    @vmap
    def mover(_k, _x):
        return move_fn(_k, _x, params)

    return mover(subkeys, x)


# Step-size bisection settings for MALA: target acceptance in [0.4, 0.65],
# initial upper bracket 1.25 * 1.65 / d**(1/6), at most 10 bisection updates.
_TARGET_ACC_RANGE = (0.4, 0.65)
_STEPSIZE_NITER_MAX = 10
_REFRESH_RATE = 1.0


def mala_kernel(problem: PTProblem, t, t_idx) -> MALAKernel:
    """
    Build the MALA kernel for rung `(t, t_idx)` of `problem`.

    Parameters
    ----------
    problem : PTProblem
    t : float or scalar Array
        Temperature in [0, 1]. May be a tracer.
    t_idx : int or scalar Array
        Rung index. May be a tracer. `t_idx == 0` is taken to mean `t == 0`:
        `move` then draws fresh reference samples instead of stepping.

    Returns
    -------
    MALAKernel
    """

    def feed_extend(x: Array) -> _MALAParticleNoVeloc:
        assert jnp.ndim(x) == 1
        logpdf_ref, gradlogpdf_ref = value_and_grad(problem.logpdf_ref)(x)
        logpdf_final_target, gradlogpdf_final_target = \
            value_and_grad(problem.logpdf_final_target)(x)
        logpdf_target = geometric_anneal(t, logpdf_ref, logpdf_final_target)
        gradlogpdf_target = geometric_anneal(t, gradlogpdf_ref, gradlogpdf_final_target)
        return _MALAParticleNoVeloc(
            position=x, logpdf_ref=logpdf_ref, gradlogpdf_ref=gradlogpdf_ref,
            logpdf_final_target=logpdf_final_target,
            gradlogpdf_final_target=gradlogpdf_final_target,
            temperature=t, temperature_idx=t_idx,
            logpdf_target=logpdf_target, gradlogpdf_target=gradlogpdf_target,
            accepted=True
        )

    def init_extend(key, x: Array) -> MALAParticle:
        assert x.ndim == 1
        return MALAParticle(veloc=random.normal(key, shape=x.shape),
                            **feed_extend(x)._asdict())

    def inner_extend(x: Array, v: Array) -> MALAParticle:
        assert x.ndim == v.ndim == 1
        assert x.shape == v.shape
        return MALAParticle(veloc=v, **feed_extend(x)._asdict())

    def check_shapes(x: MALAParticle):
        d, = x.position.shape
        assert x.veloc.shape == (d, )
        assert x.logpdf_target.shape == ()
        assert x.gradlogpdf_target.shape == (d, )
        assert x.accepted.shape == ()

    def check_shapes_multiple(x: MALAParticle):
        N, d = x.position.shape
        assert x.veloc.shape == (N, d)
        assert x.logpdf_target.shape == (N, )
        assert x.gradlogpdf_target.shape == (N, d)
        assert x.accepted.shape == (N, )

    def joint_logpdf(x: MALAParticle):
        return x.logpdf_target - 0.5 * jnp.sum(x.veloc**2)

    def jump_pos(x: MALAParticle, params: MALAParameters) -> MALAParticle:
        pos_new = x.position + \
                  0.5 * params.scaled_cov.eps * params.scaled_cov.chol @ x.veloc
        return inner_extend(pos_new, x.veloc)

    def jump_veloc(x: MALAParticle, params: MALAParameters) -> MALAParticle:
        return x._replace(veloc=x.veloc + (
            0.5*params.scaled_cov.eps*params.scaled_cov.chol.T@x.gradlogpdf_target
        ))

    def jump(x: MALAParticle, params: MALAParameters) -> MALAParticle:
        # Half position step, full velocity step (as two halves), half
        # position step, then flip the velocity.
        x = jump_pos(x, params)
        x = jump_veloc(x, params)
        x = jump_veloc(x, params)
        x = jump_pos(x, params)
        return x._replace(veloc=-x.veloc)

    def refresh_veloc(key, x: MALAParticle, refresh_rate) -> MALAParticle:
        z = random.normal(key, x.position.shape)
        v_new = -jnp.sqrt(1-refresh_rate) * x.veloc + jnp.sqrt(refresh_rate) * z
        return x._replace(veloc=v_new)

    def core_move(key, x: MALAParticle, params: MALAParameters) -> MALAParticle:
        """One step: Verlet jump, Metropolis accept/reject, velocity refresh."""
        check_shapes(x)
        x_new = jump(x, params)
        log_acc_rate = joint_logpdf(x_new) - joint_logpdf(x)
        key, subkey = random.split(key)
        log_u = jnp.log(random.uniform(subkey))
        x_ret = lax.cond(log_u < log_acc_rate,
                         lambda: x_new._replace(accepted=True),
                         lambda: x._replace(accepted=False))
        key, subkey = random.split(key)
        return refresh_veloc(subkey, x_ret, params.refresh_rate)

    def new_from_ref(key):
        key_ref, key_extend = random.split(key)
        x = problem.rvs_ref(key_ref)
        return init_extend(key_extend, x)

    def move(key, x: MALAParticle, params: MALAParameters) -> MALAParticle:
        assert jnp.ndim(x.temperature) == jnp.ndim(x.temperature_idx) == 0
        return lax.cond(t_idx == 0,
                        lambda: new_from_ref(key),
                        lambda: core_move(key, x, params))

    def adapt_params(key, x: MALAParticle) -> MALAParameters:
        """
        Preconditioner = empirical covariance of `x.position`; step size
        bisected (reusing `key` at every probe, for a smooth objective) until
        the mean acceptance of one move is within the target range.
        """
        check_shapes_multiple(x)
        emp_cov = jnp.cov(x.position, rowvar=False)
        emp_cov = _scaled_cov(eps=1, cov=emp_cov)

        def stepsize_to_params(eps):
            cov_obj = emp_cov._replace(eps=eps)
            return MALAParameters(cov_obj, _REFRESH_RATE)

        N, d = x.position.shape
        eps_upper = 1.25 * 1.65 / d**(1/6)
        acc_default = (_TARGET_ACC_RANGE[0] + _TARGET_ACC_RANGE[1])/2
        acc_delta = abs(_TARGET_ACC_RANGE[1] - _TARGET_ACC_RANGE[0])/2

        def f(eps):
            params = stepsize_to_params(eps)
            acc = _vmapped_move(core_move, key, x, params).accepted
            return jnp.mean(acc) - acc_default

        eps_star = bisection(f, 0., eps_upper, acc_delta, _STEPSIZE_NITER_MAX)
        return stepsize_to_params(eps_star)

    def reset_temperature(x: MALAParticle) -> MALAParticle:
        return x._replace(
            temperature=t, temperature_idx=t_idx,
            logpdf_target=geometric_anneal(t, x.logpdf_ref, x.logpdf_final_target),
            gradlogpdf_target=geometric_anneal(t, x.gradlogpdf_ref, x.gradlogpdf_final_target)
        )

    return MALAKernel(
        init_extend=init_extend, move=move, adapt_params=adapt_params,
        reset_temperature=reset_temperature,
    )


def batch_init_extend(key, x: Array, kernel: MALAKernel) -> MALAParticle:
    """Wrap positions `x: (N, d)` into particles, one key per particle."""
    subkeys = random.split(key, _pytree_length(x))
    return vmap(kernel.init_extend)(subkeys, x)


def batch_move(key, x: MALAParticle, params: MALAParameters, kernel: MALAKernel) -> MALAParticle:
    """One `kernel.move` on every particle in parallel."""
    return _vmapped_move(kernel.move, key, x, params)


def _rbm_body_func(i, val, params, kernel):
    key, x = val
    key, subkey = random.split(key)
    x = batch_move(subkey, x, params, kernel)
    return key, x


def repeated_batch_move(key, x: MALAParticle, params: MALAParameters,
                        kernel: MALAKernel, nrepeat: int) -> MALAParticle:
    """`nrepeat` successive `batch_move`s under `lax.fori_loop`."""
    def body_func(i, val):
        return _rbm_body_func(i, val, params, kernel)
    return lax.fori_loop(0, nrepeat, body_func, (key, x))[1]


def batch_reset_temperature(x: MALAParticle, kernel: MALAKernel) -> MALAParticle:
    """`kernel.reset_temperature` on every particle."""
    return vmap(kernel.reset_temperature)(x)
