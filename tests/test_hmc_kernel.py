"""
Unit tests for progressive_is_tempering.hmc_kernel (partial-refresh HMC) and
its selection by name in smc.smc_sampler / rejuvenate.rejuvenate.

Tests cover:
  - adapt_params: refresh rate 0.07, empirical-covariance preconditioner, step
    size landing the one-move acceptance in [0.8, 0.95].
  - move: with the same parameters it is exactly mala_kernel's move (the two
    kernels differ only in adapt_params); with partial refresh the velocity
    persists across a move, with full refresh it does not.
  - Stationarity: repeated moves reach a Gaussian target.
  - smc_sampler(kernel="hmc"): ladder 0 -> 1, log Z and final moments on a
    Gaussian path, refresh rate 0.07 at every rung; unknown kernel rejected;
    kernel="mala" is the default.
  - rejuvenate(kernel="hmc"): stationarity, with and without a zero-CoM
    projection (the projected chain stays on the manifold); unknown kernel
    rejected.
"""

import jax
import jax.numpy as jnp
import pytest

from progressive_is_tempering.hmc_kernel import (
    REFRESH_RATE,
    TARGET_ACC_RANGE,
    hmc_kernel,
)
from progressive_is_tempering.mala_kernel import (
    PTProblem,
    batch_init_extend,
    batch_move,
    mala_kernel,
    repeated_batch_move,
)
from progressive_is_tempering.rejuvenate import rejuvenate
from progressive_is_tempering.smc import smc_sampler


def _problem(dim: int, mu: float = 2.0, sigma: float = 1.0) -> PTProblem:
    """Normalized reference N(0, I); unnormalized target exp(-|x-mu|^2 / 2σ²)."""
    def logpdf_ref(x):
        return -0.5 * jnp.sum(x ** 2) - 0.5 * dim * jnp.log(2 * jnp.pi)

    def logpdf_final_target(x):
        return -0.5 * jnp.sum((x - mu) ** 2) / sigma ** 2

    def rvs_ref(key):
        return jax.random.normal(key, (dim,))

    return PTProblem(rvs_ref, logpdf_ref, logpdf_final_target)


def _adapted(problem, t, t_idx, pos, key, factory=hmc_kernel):
    kernel = factory(problem, t, t_idx)
    k_init, k_adapt = jax.random.split(key)
    x = batch_init_extend(k_init, pos, kernel)
    return kernel, x, kernel.adapt_params(k_adapt, x), k_adapt


class TestAdaptParams:
    def test_refresh_covariance_and_acceptance(self):
        problem = _problem(4)
        key = jax.random.PRNGKey(0)
        pos = 2.0 + jax.random.normal(key, (500, 4))
        kernel, x, params, k_adapt = _adapted(problem, 1.0, 1, pos, key)

        assert float(params.refresh_rate) == pytest.approx(REFRESH_RATE)
        assert jnp.allclose(params.scaled_cov.cov, jnp.cov(pos, rowvar=False), atol=1e-5)
        assert float(params.scaled_cov.eps) > 0
        # The bisection probes acceptance with this same key.
        acc = float(jnp.mean(batch_move(k_adapt, x, params, kernel).accepted))
        assert TARGET_ACC_RANGE[0] <= acc <= TARGET_ACC_RANGE[1]

    def test_smaller_step_than_full_refresh(self):
        # A higher acceptance target needs a smaller step size.
        problem = _problem(4)
        key = jax.random.PRNGKey(1)
        pos = 2.0 + jax.random.normal(key, (500, 4))
        _, _, p_hmc, _ = _adapted(problem, 1.0, 1, pos, key, hmc_kernel)
        _, _, p_mala, _ = _adapted(problem, 1.0, 1, pos, key, mala_kernel)
        assert float(p_hmc.scaled_cov.eps) < float(p_mala.scaled_cov.eps)


class TestMove:
    def test_move_is_mala_kernel_move(self):
        problem = _problem(3)
        key = jax.random.PRNGKey(2)
        pos = jax.random.normal(key, (50, 3))
        k_hmc, x, params, _ = _adapted(problem, 0.5, 1, pos, key)
        k_mala = mala_kernel(problem, 0.5, 1)
        k_move = jax.random.PRNGKey(3)
        a = batch_move(k_move, x, params, k_hmc)
        b = batch_move(k_move, x, params, k_mala)
        for la, lb in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            assert jnp.array_equal(la, lb, equal_nan=True)

    def test_partial_refresh_keeps_momentum(self):
        problem = _problem(3)
        key = jax.random.PRNGKey(4)
        pos = jax.random.normal(key, (2000, 3))
        kernel, x, params, _ = _adapted(problem, 0.5, 1, pos, key)
        y = batch_move(jax.random.PRNGKey(5), x, params, kernel)
        # Accepted moves flip the velocity and the refresh negates it again, so
        # v_new ~ sqrt(1 - 0.07) * (v_old + gradient kick) + small noise for
        # the accepted particles: strongly correlated with v_old.
        acc = y.accepted
        corr = jnp.corrcoef(x.veloc[acc].ravel(), y.veloc[acc].ravel())[0, 1]
        assert float(corr) > 0.5

        full = params._replace(refresh_rate=1.0)
        z = batch_move(jax.random.PRNGKey(5), x, full, kernel)
        corr_full = jnp.corrcoef(x.veloc.ravel(), z.veloc.ravel())[0, 1]
        assert abs(float(corr_full)) < 0.1

    def test_repeated_moves_reach_target(self):
        problem = _problem(2, mu=1.0)
        key = jax.random.PRNGKey(6)
        pos = jax.random.normal(key, (1000, 2))
        kernel, x, params, _ = _adapted(problem, 1.0, 1, pos, key)
        y = jax.jit(repeated_batch_move, static_argnums=(3, 4))(key, x, params, kernel, 300)
        assert jnp.allclose(jnp.mean(y.position, axis=0), 1.0, atol=0.15)
        assert jnp.allclose(jnp.var(y.position, axis=0), 1.0, atol=0.2)


SMC_DIM, SMC_MU, SMC_SIGMA = 2, 2.0, 0.5
SMC_PROBLEM = _problem(SMC_DIM, SMC_MU, SMC_SIGMA)
TRUE_LOGZ = 0.5 * SMC_DIM * jnp.log(2 * jnp.pi * SMC_SIGMA ** 2)


@pytest.fixture(scope="module")
def smc_hmc_run():
    return smc_sampler(jax.random.PRNGKey(0), SMC_PROBLEM, n_particles=2000,
                       n_mcmc_steps=10, local_ess_ratio=0.5, verbose=False,
                       kernel="hmc")


class TestSMCWithHMC:
    def test_ladder_and_refresh_rate(self, smc_hmc_run):
        particles, _, temps, params = smc_hmc_run
        temps = jnp.array(temps)
        assert float(temps[0]) == 0.0 and float(temps[-1]) == 1.0
        assert jnp.all(jnp.diff(temps) > 0)
        assert particles.shape == (len(temps), 2000, SMC_DIM)
        assert jnp.allclose(params.refresh_rate, REFRESH_RATE)

    def test_logZ_and_moments(self, smc_hmc_run):
        particles, logZ, _, _ = smc_hmc_run
        for name, est in logZ.items():
            assert abs(float(est) - float(TRUE_LOGZ)) < 0.15, name
        final = particles[-1]
        assert jnp.allclose(jnp.mean(final, axis=0), SMC_MU, atol=0.1)
        assert jnp.allclose(jnp.std(final, axis=0), SMC_SIGMA, atol=0.1)

    def test_default_is_mala(self):
        kw = dict(n_particles=500, n_mcmc_steps=2, local_ess_ratio=0.5, verbose=False)
        a = smc_sampler(jax.random.PRNGKey(1), SMC_PROBLEM, **kw)
        b = smc_sampler(jax.random.PRNGKey(1), SMC_PROBLEM, kernel="mala", **kw)
        assert jnp.array_equal(a[0], b[0])
        assert jnp.allclose(a[3].refresh_rate, 1.0)

    def test_unknown_kernel_rejected(self):
        with pytest.raises(ValueError, match="kernel"):
            smc_sampler(jax.random.PRNGKey(0), SMC_PROBLEM, n_particles=10,
                        n_mcmc_steps=1, local_ess_ratio=0.5, verbose=False,
                        kernel="rwm")


def _gaussian_problem(dim: int) -> PTProblem:
    """Reference == target == N(0, I)."""
    def logpdf(x):
        return -0.5 * jnp.sum(x ** 2)
    return PTProblem(lambda k: jax.random.normal(k, (dim,)), logpdf, logpdf)


def _project_com(x, n_particles=4, spatial_dim=2):
    xr = x.reshape(n_particles, spatial_dim)
    return (xr - xr.mean(axis=0, keepdims=True)).reshape(x.shape)


class TestRejuvenateWithHMC:
    def test_stationary_gaussian(self):
        problem = _gaussian_problem(8)
        key = jax.random.PRNGKey(7)
        particles = jax.random.normal(key, (500, 8))
        _, _, params, _ = _adapted(problem, 1.0, 1, particles, key)
        new_p, acc = rejuvenate(jax.random.PRNGKey(8), particles, problem,
                                jnp.array(1.0), 1, params, 50, kernel="hmc")
        assert 0.5 < float(acc) <= 1.0
        assert jnp.allclose(jnp.mean(new_p), 0.0, atol=0.1)
        assert jnp.allclose(jnp.var(new_p), 1.0, atol=0.15)

    @pytest.mark.parametrize("invariant", [False, True])
    def test_projected_stays_on_manifold_and_stationary(self, invariant):
        # N(0, I) restricted to the zero-CoM subspace: every coordinate has
        # variance 1 - 1/n_particles = 0.75. (The Gaussian is invariant under
        # the projection only in law, which `project_invariant` does not
        # need for this check.)
        problem = _gaussian_problem(8)
        key = jax.random.PRNGKey(9)
        particles = jax.vmap(_project_com)(jax.random.normal(key, (1000, 8)))
        _, _, params, _ = _adapted(problem, 1.0, 1, particles, key)
        new_p, _ = rejuvenate(jax.random.PRNGKey(10), particles, problem,
                              jnp.array(1.0), 1, params, 40, _project_com,
                              invariant, False, "hmc")
        com = new_p.reshape(-1, 4, 2).mean(axis=1)
        assert jnp.abs(com).max() < 1e-5
        assert jnp.allclose(jnp.var(new_p, axis=0), 0.75, atol=0.12)

    def test_unknown_kernel_rejected(self):
        problem = _gaussian_problem(2)
        key = jax.random.PRNGKey(11)
        particles = jax.random.normal(key, (10, 2))
        _, _, params, _ = _adapted(problem, 1.0, 1, particles, key)
        with pytest.raises(ValueError, match="kernel"):
            rejuvenate(key, particles, problem, jnp.array(1.0), 1, params, 2,
                       kernel="rwm")
