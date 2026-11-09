"""
Unit tests for progressive_is_tempering.sampler.

Tests cover:
  - backward_sde_step: output shapes, log-weight formula.
  - backward_sde_single: shape, finite log-weight, weight = 0 for exact score.
  - backward_sde_batch: batched shapes.
  - exponential_time_grid / t_end: grid shape and endpoints; with zero control
    at β=0 the exact path weight is 0 on the exponential grid too.
  - JIT / vmap compatibility.

The "exact score" test uses the reference target (β=0, π_β = N(0,I)) where
the true backward SDE is the identity (the reference OU is already stationary),
and with a zero-control the log-weight should be close to zero.
"""

import jax
import jax.numpy as jnp
import pytest

from progressive_is_tempering.sampler import (
    backward_sde_batch,
    backward_sde_single,
    backward_sde_step,
    exponential_time_grid,
    power_time_grid,
)


def zero_control(x, t, beta):
    """Zero control: u_θ = 0 (score = 0)."""
    return jnp.zeros_like(x)


def exact_reference_control(x, t, beta):
    """Exact control for N(0,I) target at any β (excess score = 0 → u = 0)."""
    return jnp.zeros_like(x)


def gaussian_energy(x):
    return 0.5 * jnp.sum(x ** 2)


class TestBackwardSDEStep:
    def test_output_shapes(self):
        key = jax.random.PRNGKey(0)
        x = jnp.ones(4)
        control = jnp.zeros(4)
        x_prev, lw, eps = backward_sde_step(key, x, 1.0, dt=0.1, control=control)
        assert x_prev.shape == (4,)
        assert lw.shape == ()
        assert eps.shape == (4,)

    def test_log_weight_formula(self):
        """
        For u=0, log p_ρ(x_n|x_{n-1}) and log q(x_{n-1}|x_n) are both Gaussian
        log-densities using exact OU coefficients λ=e^{-dt}, σ²=1-e^{-2dt}.
        Verify the formula matches a manual computation with those same coefficients.
        """
        key = jax.random.PRNGKey(1)
        x_n = jnp.array([0.2, -0.1, 0.5])
        control = jnp.zeros(3)
        dt = 0.05
        x_prev, lw, eps = backward_sde_step(key, x_n, 1.0, dt=dt, control=control)

        lam = jnp.exp(-dt)
        sigma_sq = 1.0 - jnp.exp(-2.0 * dt)
        # log p_ρ(x_n | x_prev) = N(x_n; λ x_prev, σ²I)
        log_p_manual = -0.5 * jnp.sum((x_n - lam * x_prev) ** 2) / sigma_sq
        # log q(x_prev | x_n) = N(x_prev; λ x_n + (1-λ) u, σ²I);  u=0 here
        mean_q = lam * x_n
        log_q_manual = -0.5 * jnp.sum((x_prev - mean_q) ** 2) / sigma_sq
        expected_lw = log_p_manual - log_q_manual

        assert jnp.allclose(lw, expected_lw, atol=1e-5)

    def test_jit_compatible(self):
        key = jax.random.PRNGKey(2)
        x = jnp.zeros(3)
        fn = jax.jit(backward_sde_step)
        x_prev, lw, _ = fn(key, x, jnp.array(1.0), jnp.array(0.1), jnp.zeros(3))
        assert x_prev.shape == (3,)


class TestBackwardSDESingle:
    def test_output_shapes(self):
        key = jax.random.PRNGKey(0)
        x0, lw = backward_sde_single(
            key, zero_control, jnp.array(0.0), gaussian_energy,
            dim=4, T=1.0, n_steps=10
        )
        assert x0.shape == (4,)
        assert lw.shape == ()

    def test_log_weight_is_finite(self):
        key = jax.random.PRNGKey(1)
        _, lw = backward_sde_single(
            key, zero_control, jnp.array(0.5), gaussian_energy,
            dim=4, T=2.0, n_steps=20
        )
        assert jnp.isfinite(lw)

    def test_reference_beta0_log_weight_near_zero(self):
        """
        At β=0 with zero control, the backward SDE targets N(0,I).
        The path-space weight formula becomes:
          log w = Σ [log p_ρ - log q] + ½‖x_T‖² - ½‖x_0‖²
        which for the reference OU (stationary) should be small on average.
        We verify that mean log-weight across many particles is finite.
        """
        key = jax.random.PRNGKey(2)
        N = 100
        _, log_ws = backward_sde_batch(
            key, zero_control, jnp.array(0.0), gaussian_energy,
            dim=4, n_particles=N, T=2.0, n_steps=50
        )
        assert jnp.all(jnp.isfinite(log_ws))

    def test_jit_compatible(self):
        key = jax.random.PRNGKey(3)
        # control_fn (arg 1), energy_fn (arg 3), dim (arg 4), T (arg 5), n_steps (arg 6)
        fn = jax.jit(backward_sde_single, static_argnums=(1, 3, 4, 5, 6))
        x0, lw = fn(key, zero_control, jnp.array(0.0), gaussian_energy, 4, 1.0, 10)
        assert x0.shape == (4,)


class TestBackwardSDEBatch:
    def test_output_shapes(self):
        key = jax.random.PRNGKey(0)
        x0_batch, lw_batch = backward_sde_batch(
            key, zero_control, jnp.array(0.5), gaussian_energy,
            dim=4, n_particles=8, T=1.0, n_steps=10
        )
        assert x0_batch.shape == (8, 4)
        assert lw_batch.shape == (8,)

    def test_particles_are_independent(self):
        """Different particles should (with overwhelming probability) be distinct."""
        key = jax.random.PRNGKey(1)
        x0_batch, _ = backward_sde_batch(
            key, zero_control, jnp.array(0.5), gaussian_energy,
            dim=4, n_particles=4, T=1.0, n_steps=10
        )
        # All particles distinct (extremely unlikely to be identical)
        for i in range(4):
            for j in range(i + 1, 4):
                assert not jnp.allclose(x0_batch[i], x0_batch[j])

    def test_log_weights_finite(self):
        key = jax.random.PRNGKey(2)
        _, lw = backward_sde_batch(
            key, zero_control, jnp.array(0.5), gaussian_energy,
            dim=4, n_particles=10, T=1.0, n_steps=10
        )
        assert jnp.all(jnp.isfinite(lw))

class TestExponentialGrid:
    def test_grid_shape_and_endpoints(self):
        times, dts = exponential_time_grid(4.0, 100, 1e-3)
        assert times.shape == dts.shape == (100,)
        assert float(times[0]) == pytest.approx(4.0)
        assert float(times[-1]) == pytest.approx(1e-3)
        assert float(dts[-1]) == pytest.approx(1e-3)        # last step: t_1 -> 0
        assert float(jnp.sum(dts)) == pytest.approx(4.0, rel=1e-5)
        assert bool(jnp.all(jnp.diff(times) < 0))           # decreasing
        # Steps shrink geometrically down to t_1; the final step t_1 -> 0 is t_end
        # itself, larger than the one before it (t_2 - t_1).
        assert bool(jnp.all(jnp.diff(dts[:-1]) < 0))

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            exponential_time_grid(4.0, 1, 1e-3)
        with pytest.raises(ValueError):
            exponential_time_grid(4.0, 10, 5.0)

    def test_zero_control_reference_weight_is_zero(self):
        """β=0, u=0: the proposal is the stationary reference path itself, so
        the log-weight is 0 for every particle (up to float32 rounding)."""
        _, lw = backward_sde_batch(
            jax.random.PRNGKey(4), zero_control, jnp.array(0.0), gaussian_energy,
            dim=8, n_particles=64, T=4.0, n_steps=100, t_end=1e-3,
        )
        assert float(jnp.max(jnp.abs(lw))) < 1e-3

    def test_uniform_default_unchanged(self):
        key = jax.random.PRNGKey(5)
        a = backward_sde_batch(key, zero_control, jnp.array(0.5), gaussian_energy,
                               dim=3, n_particles=8, T=2.0, n_steps=20)
        b = backward_sde_batch(key, zero_control, jnp.array(0.5), gaussian_energy,
                               dim=3, n_particles=8, T=2.0, n_steps=20, t_end=None)
        assert jnp.array_equal(a[0], b[0]) and jnp.array_equal(a[1], b[1])


class TestPowerGrid:
    def test_grid_shape_and_endpoints(self):
        times, dts = power_time_grid(1.5, 55, 2.0)
        assert times.shape == dts.shape == (55,)
        assert float(times[0]) == pytest.approx(1.5)
        assert float(times[-1]) == pytest.approx(1.5 / 55 ** 2)
        assert float(dts[-1]) == pytest.approx(1.5 / 55 ** 2)   # last step: t_1 -> 0
        assert float(jnp.sum(dts)) == pytest.approx(1.5, rel=1e-5)
        assert bool(jnp.all(jnp.diff(times) < 0))
        assert bool(jnp.all(jnp.diff(dts) < 0))                 # steps shrink towards 0

    def test_power_one_is_uniform(self):
        times, dts = power_time_grid(2.0, 20, 1.0)
        assert jnp.allclose(dts, 0.1, atol=1e-6)
        assert jnp.allclose(times, jnp.linspace(2.0, 0.1, 20), atol=1e-6)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            power_time_grid(1.5, 0, 2.0)
        with pytest.raises(ValueError):
            power_time_grid(0.0, 10, 2.0)
        with pytest.raises(ValueError):
            power_time_grid(1.5, 10, 0.5)
