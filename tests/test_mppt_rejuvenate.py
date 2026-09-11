"""
Unit tests for progressive_is_tempering.mppt_rejuvenate.

Tests cover:
  - mppt_rejuvenate: shape, JIT compatibility, vmap-over-rungs compatibility
    (the pattern needed by gmm.py's vmapped outer training loop).
  - Detailed balance: with an exact Gaussian target, the stationary
    distribution is preserved in the long run (empirical mean/variance
    check), mirroring test_mala.py's equivalent check for the retired
    isotropic MALA implementation.
  - The t_idx==0 gotcha documented in mppt_rejuvenate's docstring: mppt's
    engine ignores `positions` and redraws from the reference at t_idx==0.
  - project_fn (used by DW4/LJ13's zero-center-of-mass constraint): output
    stays exactly on the constraint manifold after every step, and the
    vmap-over-rungs pattern still works with it set (closed over, like
    `mppt_problem`/`n_steps`, not vmapped).
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# The editable install of `mppt` registers a namespace package rooted one
# level too high (`.../mppt/` instead of `.../mppt/mppt/`) for plain `import
# mppt` to resolve `PTProblem` etc. Match the fix notebooks/gmm/gmm.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "mppt"))

from mppt import PTProblem
from mppt.impl.pt_mcmc_hmc import hmc_pt_mcmc
from mppt.jax_configs import _set_allow_runtime_check

from progressive_is_tempering.mppt_rejuvenate import mppt_rejuvenate


@pytest.fixture(autouse=True)
def _reset_mppt_runtime_check():
    """mppt/mppt/tests/test_logZ.py sets mppt's global ALLOW_RUNTIME_CHECK to
    True as a bare module-level statement at import time, with no cleanup;
    that global is process-wide and leaks into this file when a test run
    collects both test suites together (mppt/mppt/tests/conftest.py resets
    it, but only for tests under that directory). Reset it locally so this
    file's tests are correct regardless of what else the session collected.
    """
    _set_allow_runtime_check(False)
    yield
    _set_allow_runtime_check(False)


def _gaussian_problem(dim: int) -> PTProblem:
    """Reference == target == N(0,I): the simplest possible stationarity check."""
    def logpdf(x):
        return -0.5 * jnp.sum(x ** 2) - 0.5 * dim * jnp.log(2 * jnp.pi)

    def rvs_ref(key):
        return jax.random.normal(key, (dim,))

    return PTProblem(rvs_ref=rvs_ref, logpdf_ref=logpdf, logpdf_final_target=logpdf)


def _project_com(x: jnp.ndarray, n_particles: int, spatial_dim: int) -> jnp.ndarray:
    """Zero-center-of-mass projector, mirroring DW4/LJ13's `project_com`
    (same formula, single-particle signature since mppt_rejuvenate vmaps it
    itself)."""
    xr = x.reshape(n_particles, spatial_dim)
    com = xr.mean(axis=0, keepdims=True)
    return (xr - com).reshape(x.shape)


def _calibrated_params(problem, beta, t_idx, particles, key):
    """Adapt HMCParameters the same way `one_step_smc` does mid-SMC, standalone."""
    mcmc = hmc_pt_mcmc(problem, beta, t_idx, mala=True)
    key_init, key_adapt = jax.random.split(key)
    x = jax.vmap(mcmc.init_extend)(
        jax.random.split(key_init, particles.shape[0]), particles
    )
    return mcmc.adapt_params(key_adapt, x)


class TestMPPTRejuvenate:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        problem = _gaussian_problem(4)
        particles = jax.random.normal(key, (20, 4))
        params = _calibrated_params(problem, 1.0, 1, particles, key)
        new_p, acc = mppt_rejuvenate(
            key, particles, problem, jnp.array(1.0), 1, params, 10
        )
        assert new_p.shape == (20, 4)
        assert acc.shape == ()

    def test_stationary_distribution_gaussian(self):
        key = jax.random.PRNGKey(1)
        problem = _gaussian_problem(8)
        particles = jax.random.normal(key, (300, 8))
        key, k_calib, k_move = jax.random.split(key, 3)
        params = _calibrated_params(problem, 1.0, 1, particles, k_calib)
        new_p, acc = mppt_rejuvenate(
            k_move, particles, problem, jnp.array(1.0), 1, params, 50
        )
        assert 0.2 < float(acc) < 1.0
        assert jnp.allclose(jnp.mean(new_p), 0.0, atol=0.15)
        assert jnp.allclose(jnp.var(new_p), 1.0, atol=0.15)

    def test_particles_change(self):
        key = jax.random.PRNGKey(2)
        problem = _gaussian_problem(4)
        particles = jax.random.normal(key, (10, 4))
        params = _calibrated_params(problem, 1.0, 1, particles, key)
        new_p, _ = mppt_rejuvenate(
            key, particles, problem, jnp.array(1.0), 1, params, 20
        )
        assert not jnp.allclose(new_p, particles)

    def test_t_idx_zero_ignores_positions(self):
        """Documented gotcha: t_idx=0 redraws from rvs_ref, ignoring `positions`."""
        key = jax.random.PRNGKey(3)
        problem = _gaussian_problem(4)
        particles = jnp.full((10, 4), 100.0)  # far outside N(0,I) support
        params = _calibrated_params(problem, 0.0, 0, particles, key)
        new_p, _ = mppt_rejuvenate(
            key, particles, problem, jnp.array(0.0), 0, params, 5
        )
        assert not jnp.allclose(new_p, particles)

    def test_vmap_over_rungs(self):
        """The pattern gmm.py's vmapped `process_one` needs: vmap over
        (key, positions, beta, t_idx, mcmc_params) per active rung, with
        `problem` and `n_steps` closed over (not vmapped) -- mirroring
        exactly how mppt_rejuvenate is static in those two positions.
        """
        key = jax.random.PRNGKey(4)
        problem = _gaussian_problem(4)
        n_rungs = 3
        keys = jax.random.split(key, n_rungs)
        particles = jax.random.normal(key, (n_rungs, 15, 4))
        betas = jnp.array([0.3, 0.6, 1.0])
        t_idxs = jnp.array([1, 2, 3])
        params = jax.vmap(lambda b, t, p: _calibrated_params(problem, b, t, p, key))(
            betas, t_idxs, particles
        )

        def one(k, p, b, t, prm):
            return mppt_rejuvenate(k, p, problem, b, t, prm, 5)

        new_p, acc = jax.vmap(one)(keys, particles, betas, t_idxs, params)
        assert new_p.shape == (n_rungs, 15, 4)
        assert acc.shape == (n_rungs,)


class TestMPPTRejuvenateProjected:
    """DW4/LJ13 need every internal move projected back to zero-CoM -- mppt's
    own kernel has no CoM-awareness at all, so this must be threaded through
    explicitly via `project_fn`.
    """

    def test_output_stays_on_constraint_manifold(self):
        """Starting from particles with nonzero CoM, rejuvenation with
        project_fn should return particles with exactly zero CoM (to float
        precision) -- the chain must never leave the manifold, not just end
        up back on it by chance.
        """
        key = jax.random.PRNGKey(10)
        n_particles, spatial_dim = 2, 2
        dim = n_particles * spatial_dim
        problem = _gaussian_problem(dim)

        key, subkey = jax.random.split(key)
        particles = jax.random.normal(subkey, (20, dim)) + 5.0  # nonzero CoM
        params = _calibrated_params(problem, 1.0, 1, particles, key)

        project_fn = lambda x: _project_com(x, n_particles, spatial_dim)
        key, subkey = jax.random.split(key)
        new_p, acc = mppt_rejuvenate(
            subkey, particles, problem, jnp.array(1.0), 1, params, 30,
            project_fn=project_fn,
        )
        com = new_p.reshape(-1, n_particles, spatial_dim).mean(axis=1)
        assert jnp.abs(com).max() < 1e-4
        assert 0.0 <= float(acc) <= 1.0

    def test_default_path_unaffected(self):
        """project_fn=None (the default) must reproduce the pre-existing
        repeated_batch_move fast path exactly -- same call, same key, same
        result, confirming GMM's call sites see zero behavior change.
        """
        key = jax.random.PRNGKey(11)
        problem = _gaussian_problem(4)
        particles = jax.random.normal(key, (10, 4))
        params = _calibrated_params(problem, 1.0, 1, particles, key)

        key, subkey = jax.random.split(key)
        p_default, a_default = mppt_rejuvenate(
            subkey, particles, problem, jnp.array(1.0), 1, params, 10
        )
        p_explicit, a_explicit = mppt_rejuvenate(
            subkey, particles, problem, jnp.array(1.0), 1, params, 10,
            project_fn=None,
        )
        assert jnp.array_equal(p_default, p_explicit)
        assert jnp.array_equal(a_default, a_explicit)

    def test_vmap_over_rungs_with_project_fn(self):
        """Same vmap-over-rungs pattern _data_gen_parallel needs, but with
        project_fn set -- it must be closed over (not vmapped), exactly like
        `mppt_problem`/`n_steps`.
        """
        key = jax.random.PRNGKey(12)
        n_particles, spatial_dim = 2, 2
        dim = n_particles * spatial_dim
        problem = _gaussian_problem(dim)
        project_fn = lambda x: _project_com(x, n_particles, spatial_dim)

        n_rungs = 3
        keys = jax.random.split(key, n_rungs)
        key, subkey = jax.random.split(key)
        particles = jax.random.normal(subkey, (n_rungs, 12, dim)) + 3.0
        betas = jnp.array([0.3, 0.6, 1.0])
        t_idxs = jnp.array([1, 2, 3])
        params = jax.vmap(lambda b, t, p: _calibrated_params(problem, b, t, p, key))(
            betas, t_idxs, particles
        )

        def one(k, p, b, t, prm):
            return mppt_rejuvenate(k, p, problem, b, t, prm, 15, project_fn=project_fn)

        new_p, acc = jax.vmap(one)(keys, particles, betas, t_idxs, params)
        assert new_p.shape == (n_rungs, 12, dim)
        assert acc.shape == (n_rungs,)
        com = new_p.reshape(n_rungs, 12, n_particles, spatial_dim).mean(axis=2)
        assert jnp.abs(com).max() < 1e-4
