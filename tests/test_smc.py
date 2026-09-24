"""
Unit tests for progressive_is_tempering.smc.

Tests cover:
  - smc_sampler output contract: ladder runs 0 -> 1 strictly increasing,
    one particle block per rung, one stacked MALAParameters entry per move.
  - Correctness on a Gaussian path with known normalizing constants: all
    three log Z estimators, and the final population's moments.
  - Reference == target: the ladder is a single jump to t=1.
  - Determinism for a fixed key; argument validation; verbose=False is silent.
"""

import jax
import jax.numpy as jnp
import pytest

from progressive_is_tempering.mala_kernel import PTProblem
from progressive_is_tempering.smc import smc_sampler

DIM = 2
MU = 2.0
SIGMA = 0.5


def _problem() -> PTProblem:
    """Normalized reference N(0, I); unnormalized target exp(-|x-MU|^2 / 2σ²).

    log(Z_1 / Z_0) = (DIM / 2) * log(2π σ²).
    """
    def logpdf_ref(x):
        return -0.5 * jnp.sum(x ** 2) - 0.5 * DIM * jnp.log(2 * jnp.pi)

    def logpdf_final_target(x):
        return -0.5 * jnp.sum((x - MU) ** 2) / SIGMA ** 2

    def rvs_ref(key):
        return jax.random.normal(key, (DIM,))

    return PTProblem(rvs_ref, logpdf_ref, logpdf_final_target)


PROBLEM = _problem()
TRUE_LOGZ = 0.5 * DIM * jnp.log(2 * jnp.pi * SIGMA ** 2)


@pytest.fixture(scope="module")
def smc_run():
    return smc_sampler(jax.random.PRNGKey(0), PROBLEM, n_particles=2000,
                       n_mcmc_steps=10, local_ess_ratio=0.5, verbose=False)


class TestSMCSampler:
    def test_output_shapes(self, smc_run):
        particles, logZ, temps, params = smc_run
        M = len(temps) - 1
        assert M >= 2
        assert particles.shape == (M + 1, 2000, DIM)
        assert params.scaled_cov.eps.shape == (M,)
        assert params.scaled_cov.chol.shape == (M, DIM, DIM)
        assert set(logZ) == {"SMC_like", "robust", "reverse_logistic"}

    def test_ladder_monotone_from_zero_to_one(self, smc_run):
        temps = jnp.array(smc_run[2])
        assert float(temps[0]) == 0.0
        assert float(temps[-1]) == 1.0
        assert jnp.all(jnp.diff(temps) > 0)

    def test_logZ_estimates(self, smc_run):
        for name, est in smc_run[1].items():
            assert abs(float(est) - float(TRUE_LOGZ)) < 0.15, name

    def test_final_population_moments(self, smc_run):
        final = smc_run[0][-1]
        assert jnp.allclose(jnp.mean(final, axis=0), MU, atol=0.1)
        assert jnp.allclose(jnp.std(final, axis=0), SIGMA, atol=0.1)

    def test_deterministic(self, smc_run):
        again = smc_sampler(jax.random.PRNGKey(0), PROBLEM, n_particles=2000,
                            n_mcmc_steps=10, local_ess_ratio=0.5, verbose=False)
        assert jnp.array_equal(smc_run[0], again[0])
        assert all(jax.tree.leaves(jax.tree.map(jnp.array_equal, smc_run[3], again[3])))

    def test_reference_equals_target_single_step(self):
        def logpdf(x):
            return -0.5 * jnp.sum(x ** 2)
        problem = PTProblem(lambda k: jax.random.normal(k, (DIM,)), logpdf, logpdf)
        particles, _, temps, params = smc_sampler(
            jax.random.PRNGKey(1), problem, n_particles=100, n_mcmc_steps=1,
            local_ess_ratio=0.5, verbose=False)
        assert len(temps) == 2 and float(temps[-1]) == 1.0
        assert particles.shape == (2, 100, DIM)
        assert params.scaled_cov.eps.shape == (1,)

    def test_verbose_false_is_silent(self, capsys):
        smc_sampler(jax.random.PRNGKey(2), PROBLEM, n_particles=50,
                    n_mcmc_steps=1, local_ess_ratio=0.5, verbose=False)
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("kwargs", [
        {"n_particles": 0},
        {"n_mcmc_steps": -1},
        {"n_mcmc_steps": 1.5},
        {"local_ess_ratio": 1.0},
        {"local_ess_ratio": 0.0},
    ])
    def test_validation(self, kwargs):
        args = {"n_particles": 10, "n_mcmc_steps": 1, "local_ess_ratio": 0.5}
        args.update(kwargs)
        with pytest.raises(ValueError):
            smc_sampler(jax.random.PRNGKey(0), PROBLEM, verbose=False, **args)
