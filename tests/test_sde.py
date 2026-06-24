"""
Unit tests for progressive_is_tempering.sde.

Tests cover:
  - ou_mean / ou_std: boundary conditions (t=0, t→∞)
  - ou_forward_sample: shape, mean / variance empirical checks
  - ou_log_transition_density: log-density of a Gaussian is correct
  - reference_ou_score: -x
  - JIT / vmap compatibility
"""

import jax
import jax.numpy as jnp
import pytest

from progressive_is_tempering.sde import (
    ou_forward_sample,
    ou_log_transition_density,
    ou_mean,
    ou_std,
    reference_ou_score,
)


class TestOUMeanStd:
    def test_mean_at_t0(self):
        x0 = jnp.array([1.0, 2.0, -3.0])
        assert jnp.allclose(ou_mean(x0, 0.0), x0)

    def test_mean_decays_to_zero(self):
        x0 = jnp.ones(4)
        mean_large = ou_mean(x0, 20.0)
        assert jnp.allclose(mean_large, jnp.zeros(4), atol=1e-6)

    def test_std_at_t0_is_zero(self):
        assert jnp.allclose(ou_std(0.0), 0.0, atol=1e-6)

    def test_std_at_large_t_approaches_one(self):
        assert jnp.allclose(ou_std(20.0), 1.0, atol=1e-6)

    def test_std_is_nonnegative(self):
        ts = jnp.linspace(0.01, 5.0, 20)
        stds = jax.vmap(ou_std)(ts)
        assert jnp.all(stds >= 0)


class TestOUForwardSample:
    def test_shape(self):
        key = jax.random.PRNGKey(0)
        x0 = jnp.zeros(8)
        xt = ou_forward_sample(key, x0, 1.0)
        assert xt.shape == (8,)

    def test_mean_empirical(self):
        """Empirical mean over N samples should be close to e^{-t} x0."""
        key = jax.random.PRNGKey(1)
        x0 = jnp.array([2.0, -1.0])
        t = 1.0
        N = 5000
        keys = jax.random.split(key, N)
        samples = jax.vmap(lambda k: ou_forward_sample(k, x0, t))(keys)
        empirical_mean = jnp.mean(samples, axis=0)
        expected_mean = jnp.exp(-t) * x0
        assert jnp.allclose(empirical_mean, expected_mean, atol=0.05)

    def test_variance_empirical(self):
        """Empirical variance should be close to σ_t² = 1 - e^{-2t}."""
        key = jax.random.PRNGKey(2)
        x0 = jnp.zeros(2)
        t = 1.0
        N = 5000
        keys = jax.random.split(key, N)
        samples = jax.vmap(lambda k: ou_forward_sample(k, x0, t))(keys)
        empirical_var = jnp.var(samples, axis=0)
        expected_var = 1.0 - jnp.exp(-2.0 * t)
        assert jnp.allclose(empirical_var, jnp.full(2, expected_var), atol=0.05)

    def test_jit(self):
        key = jax.random.PRNGKey(3)
        x0 = jnp.zeros(4)
        sample = jax.jit(ou_forward_sample)(key, x0, 1.0)
        assert sample.shape == (4,)


class TestOULogTransitionDensity:
    def test_gaussian_log_density(self):
        """Value should match a manually computed Gaussian log-density using exact OU coefficients."""
        x_n = jnp.array([0.5, -0.3])
        x_prev = jnp.array([0.0, 0.0])
        dt = 0.1
        lam = jnp.exp(-dt)
        sigma_sq = 1.0 - jnp.exp(-2.0 * dt)
        mean = lam * x_prev
        expected = -0.5 * jnp.sum((x_n - mean) ** 2) / sigma_sq
        result = ou_log_transition_density(x_n, x_prev, dt)
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_peak_at_mode(self):
        """Log-density is maximized when x_n equals the OU mean e^{-dt} x_prev."""
        x_prev = jnp.array([1.0, -1.0])
        dt = 0.05
        mean = jnp.exp(-dt) * x_prev
        perturbed = mean + 0.1 * jnp.ones(2)
        assert ou_log_transition_density(mean, x_prev, dt) > ou_log_transition_density(
            perturbed, x_prev, dt
        )


class TestReferenceOUScore:
    def test_minus_x(self):
        x = jnp.array([1.0, 2.0, -3.0])
        assert jnp.allclose(reference_ou_score(x), -x)