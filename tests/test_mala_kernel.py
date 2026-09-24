"""
Unit tests for progressive_is_tempering.mala_kernel.

Tests cover:
  - init_extend / reset_temperature: cached densities and gradients match the
    geometric anneal of the reference and final target.
  - adapt_params: preconditioner is the empirical covariance, step size lands
    the one-move acceptance rate in the MALA target range, degenerate
    covariance falls back to a diagonal Cholesky factor.
  - move: t_idx == 0 redraws from the reference; t_idx >= 1 is a genuine
    Metropolis step (a rejected proposal leaves the position unchanged).
  - bisection: finds the root of a monotone function, including when the
    initial bracket does not contain it.
"""

import jax
import jax.numpy as jnp

from progressive_is_tempering.mala_kernel import (
    PTProblem,
    batch_init_extend,
    batch_move,
    batch_reset_temperature,
    bisection,
    geometric_anneal,
    mala_kernel,
    repeated_batch_move,
)


def _problem(dim: int, mu: float = 2.0) -> PTProblem:
    """Reference N(0, I); target N(mu, I) (unnormalized)."""
    def logpdf_ref(x):
        return -0.5 * jnp.sum(x ** 2)

    def logpdf_final_target(x):
        return -0.5 * jnp.sum((x - mu) ** 2)

    def rvs_ref(key):
        return jax.random.normal(key, (dim,))

    return PTProblem(rvs_ref, logpdf_ref, logpdf_final_target)


class TestExtend:
    def test_init_extend_fields(self):
        problem = _problem(3)
        t = 0.3
        kernel = mala_kernel(problem, t, 1)
        x = jnp.array([0.5, -1.0, 2.0])
        p = kernel.init_extend(jax.random.PRNGKey(0), x)
        expected = geometric_anneal(t, problem.logpdf_ref(x), problem.logpdf_final_target(x))
        expected_grad = -(1 - t) * x - t * (x - 2.0)
        assert jnp.allclose(p.logpdf_target, expected)
        assert jnp.allclose(p.gradlogpdf_target, expected_grad, atol=1e-6)
        assert p.veloc.shape == (3,)
        assert bool(p.accepted)

    def test_reset_temperature_reanneals(self):
        problem = _problem(3)
        key = jax.random.PRNGKey(1)
        pos = jax.random.normal(key, (5, 3))
        x = batch_init_extend(key, pos, mala_kernel(problem, 0.2, 1))
        y = batch_reset_temperature(x, mala_kernel(problem, 0.7, 2))
        z = batch_init_extend(key, pos, mala_kernel(problem, 0.7, 2))
        assert jnp.allclose(y.logpdf_target, z.logpdf_target)
        assert jnp.allclose(y.gradlogpdf_target, z.gradlogpdf_target, atol=1e-6)
        assert jnp.all(y.temperature_idx == 2)


class TestAdaptParams:
    def test_covariance_and_acceptance(self):
        problem = _problem(4)
        key = jax.random.PRNGKey(2)
        k_pos, k_init, k_adapt = jax.random.split(key, 3)
        pos = 2.0 + jax.random.normal(k_pos, (500, 4))
        kernel = mala_kernel(problem, 1.0, 1)
        x = batch_init_extend(k_init, pos, kernel)
        params = kernel.adapt_params(k_adapt, x)

        assert jnp.allclose(params.scaled_cov.cov, jnp.cov(pos, rowvar=False), atol=1e-5)
        assert not bool(params.scaled_cov.chol_failed)
        assert float(params.refresh_rate) == 1.0
        assert float(params.scaled_cov.eps) > 0
        # The bisection probes acceptance with this same key.
        acc = jnp.mean(batch_move(k_adapt, x, params, kernel).accepted)
        assert 0.4 <= float(acc) <= 0.65

    def test_degenerate_covariance_falls_back_to_diagonal(self):
        problem = _problem(3)
        key = jax.random.PRNGKey(3)
        col = jax.random.normal(key, (50, 1))
        pos = jnp.concatenate([col, col, col], axis=1)  # rank-1 covariance
        kernel = mala_kernel(problem, 1.0, 1)
        x = batch_init_extend(key, pos, kernel)
        params = kernel.adapt_params(key, x)
        chol = params.scaled_cov.chol
        assert bool(params.scaled_cov.chol_failed)
        assert jnp.allclose(chol, jnp.diag(jnp.diag(chol)))
        assert jnp.isfinite(params.scaled_cov.log_det_chol)


class TestMove:
    def test_t_idx_zero_redraws_from_reference(self):
        problem = _problem(3)
        key = jax.random.PRNGKey(4)
        pos = jnp.full((10, 3), 100.0)
        kernel = mala_kernel(problem, 0.0, 0)
        x = batch_init_extend(key, pos, kernel)
        params = kernel.adapt_params(key, batch_init_extend(key, jax.random.normal(key, (10, 3)), kernel))
        y = batch_move(key, x, params, kernel)
        assert jnp.all(jnp.abs(y.position) < 10.0)

    def test_rejected_moves_keep_position(self):
        problem = _problem(3)
        key = jax.random.PRNGKey(5)
        pos = jax.random.normal(key, (200, 3))
        kernel = mala_kernel(problem, 0.5, 1)
        x = batch_init_extend(key, pos, kernel)
        params = kernel.adapt_params(key, x)
        y = batch_move(jax.random.PRNGKey(6), x, params, kernel)
        rejected = ~y.accepted
        assert jnp.any(rejected)
        assert jnp.array_equal(y.position[rejected], x.position[rejected])

    def test_repeated_moves_reach_target(self):
        problem = _problem(2, mu=1.0)
        key = jax.random.PRNGKey(7)
        pos = jax.random.normal(key, (1000, 2))
        kernel = mala_kernel(problem, 1.0, 1)
        x = batch_init_extend(key, pos, kernel)
        params = kernel.adapt_params(key, x)
        y = jax.jit(repeated_batch_move, static_argnums=(3, 4))(key, x, params, kernel, 200)
        assert jnp.allclose(jnp.mean(y.position, axis=0), 1.0, atol=0.15)
        assert jnp.allclose(jnp.var(y.position, axis=0), 1.0, atol=0.2)


class TestBisection:
    def test_bracketed_root(self):
        root = bisection(lambda x: x - 0.3, 0.0, 1.0, 1e-4, 50)
        assert abs(float(root) - 0.3) < 1e-3

    def test_root_outside_initial_bracket(self):
        root = bisection(lambda x: x - 2.5, 0.0, 1.0, 1e-4, 50)
        assert abs(float(root) - 2.5) < 1e-3
