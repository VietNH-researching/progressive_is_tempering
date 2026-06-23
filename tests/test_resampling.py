"""
Unit tests for progressive_is_tempering.resampling.

Tests cover:
  - systematic_resample: output shape / dtype, indices in range,
    distribution of indices matches weights, all-equal weights → uniform,
    single dominant weight → that index selected.
  - normalized_ess: uniform weights → 1.0, single weight → 1/N.
  - JIT compatibility.
"""

import jax
import jax.numpy as jnp
import pytest

from progressive_is_tempering.resampling import normalized_ess, systematic_resample


class TestSystematicResample:
    def test_output_shape(self):
        key = jax.random.PRNGKey(0)
        log_w = jnp.zeros(10)
        idx = systematic_resample(key, log_w, 5)
        assert idx.shape == (5,)

    def test_output_dtype(self):
        key = jax.random.PRNGKey(0)
        log_w = jnp.zeros(10)
        idx = systematic_resample(key, log_w, 5)
        assert idx.dtype == jnp.int32

    def test_indices_in_range(self):
        key = jax.random.PRNGKey(1)
        N = 20
        log_w = jax.random.normal(key, (N,))
        idx = systematic_resample(key, log_w, 10)
        assert jnp.all(idx >= 0)
        assert jnp.all(idx < N)

    def test_uniform_weights_sample_all_indices(self):
        """With equal weights, all N particles should appear roughly equally."""
        key = jax.random.PRNGKey(2)
        N = 8
        n = 8 * 100
        log_w = jnp.zeros(N)
        all_indices = []
        for i in range(100):
            key, subkey = jax.random.split(key)
            idx = systematic_resample(subkey, log_w, N)
            all_indices.append(idx)
        flat = jnp.concatenate(all_indices)
        counts = jnp.bincount(flat, length=N)
        # Each particle should appear roughly n/N = 100 times
        assert jnp.all(counts > 50), f"counts: {counts}"

    def test_single_dominant_weight(self):
        """Particle 0 with much higher weight should dominate all draws."""
        key = jax.random.PRNGKey(3)
        N = 10
        log_w = jnp.full(N, -100.0).at[0].set(0.0)
        idx = systematic_resample(key, log_w, N)
        assert jnp.all(idx == 0)

    def test_jit_compatible(self):
        key = jax.random.PRNGKey(4)
        log_w = jnp.zeros(5)
        jitted = jax.jit(systematic_resample, static_argnums=(2,))
        idx = jitted(key, log_w, 5)
        assert idx.shape == (5,)

    def test_empirical_distribution_matches_weights(self):
        """Frequency of each index should be proportional to its weight."""
        key = jax.random.PRNGKey(5)
        N = 4
        # Weights: [0.1, 0.2, 0.3, 0.4]
        log_w = jnp.log(jnp.array([0.1, 0.2, 0.3, 0.4]))
        n_draws = 10_000
        counts = jnp.zeros(N)
        for _ in range(n_draws // N):
            key, subkey = jax.random.split(key)
            idx = systematic_resample(subkey, log_w, N)
            counts = counts + jnp.bincount(idx, length=N)
        freqs = counts / jnp.sum(counts)
        expected = jnp.array([0.1, 0.2, 0.3, 0.4])
        assert jnp.allclose(freqs, expected, atol=0.02), f"freqs={freqs}"


class TestNormalizedESS:
    def test_uniform_weights_gives_one(self):
        N = 10
        log_w = jnp.zeros(N)
        ess = normalized_ess(log_w)
        assert jnp.allclose(ess, 1.0, atol=1e-5)

    def test_single_particle_gives_one_over_n(self):
        N = 10
        log_w = jnp.full(N, -1e6).at[0].set(0.0)
        ess = normalized_ess(log_w)
        assert jnp.allclose(ess, 1.0 / N, atol=1e-4)

    def test_ess_bounded(self):
        key = jax.random.PRNGKey(0)
        log_w = jax.random.normal(key, (20,))
        ess = normalized_ess(log_w)
        assert 0.0 < float(ess) <= 1.0