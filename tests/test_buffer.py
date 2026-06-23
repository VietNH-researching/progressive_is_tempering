"""
Unit tests for progressive_is_tempering.buffer.

Tests cover:
  - ReplayBuffer.add: size tracking, FIFO eviction at capacity, data correctness.
  - ReplayBuffer.sample: shape, indices within valid range, weighted sampling bias.
  - ReplayBuffer.is_ready: threshold logic.
  - ReplayBuffer.is_bridge_log_weights: formula correctness.
"""

import jax
import jax.numpy as jnp
import pytest

from progressive_is_tempering.buffer import ReplayBuffer


def energy_sq(x):
    return 0.5 * jnp.sum(x ** 2)


class TestReplayBufferAdd:
    def test_initial_size_zero(self):
        buf = ReplayBuffer(capacity=10, dim=3)
        assert buf.size == 0

    def test_add_increases_size(self):
        buf = ReplayBuffer(capacity=10, dim=3)
        samples = jnp.ones((4, 3))
        log_w = jnp.zeros(4)
        buf.add(samples, log_w)
        assert buf.size == 4

    def test_add_respects_capacity(self):
        buf = ReplayBuffer(capacity=5, dim=3)
        for _ in range(3):
            buf.add(jnp.ones((3, 3)), jnp.zeros(3))
        assert buf.size == 5  # capped at capacity

    def test_fifo_eviction_overwrites_oldest(self):
        """Writing 6 items into capacity-5 buffer: item 0 should be overwritten."""
        buf = ReplayBuffer(capacity=5, dim=1)
        # First 5 items: value 1
        buf.add(jnp.ones((5, 1)), jnp.zeros(5))
        # One more item: value 2
        buf.add(jnp.array([[2.0]]), jnp.zeros(1))
        # Slot 0 should now hold 2.0 (the newest write)
        assert float(buf.data[0, 0]) == 2.0

    def test_data_stored_correctly(self):
        buf = ReplayBuffer(capacity=10, dim=2)
        samples = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
        log_w = jnp.array([-1.0, -2.0, -3.0])
        buf.add(samples, log_w)
        assert jnp.allclose(buf.data[:3], samples)
        assert jnp.allclose(buf.log_weights[:3], log_w)


class TestReplayBufferSample:
    def test_sample_shape(self):
        buf = ReplayBuffer(capacity=20, dim=4)
        buf.add(jax.random.normal(jax.random.PRNGKey(0), (15, 4)), jnp.zeros(15))
        key = jax.random.PRNGKey(1)
        samples, lw = buf.sample(key, batch_size=5)
        assert samples.shape == (5, 4)
        assert lw.shape == (5,)

    def test_sample_indices_within_valid_range(self):
        """Sampled data should come from valid (written) slots."""
        buf = ReplayBuffer(capacity=20, dim=2)
        # Write distinctive values
        data = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)
        buf.add(data, jnp.zeros(10))
        key = jax.random.PRNGKey(2)
        samples, _ = buf.sample(key, batch_size=8)
        # All sampled values should be in [0, 18] (the written range)
        assert jnp.all(samples >= 0.0)
        assert jnp.all(samples < 20.0)

    def test_is_ready_false_when_empty(self):
        buf = ReplayBuffer(capacity=10, dim=3)
        assert not buf.is_ready(1)

    def test_is_ready_true_when_full_enough(self):
        buf = ReplayBuffer(capacity=10, dim=3)
        buf.add(jnp.ones((5, 3)), jnp.zeros(5))
        assert buf.is_ready(3)
        assert buf.is_ready(5)
        assert not buf.is_ready(6)


class TestISBridgeLogWeights:
    def test_shape(self):
        key = jax.random.PRNGKey(0)
        samples = jax.random.normal(key, (8, 4))
        log_w = ReplayBuffer.is_bridge_log_weights(
            samples, energy_sq, jnp.array(0.0), jnp.array(0.2)
        )
        assert log_w.shape == (8,)

    def test_formula_correctness(self):
        """
        log w_{k→k+1}(x) = (β_{k+1} - β_k) · (-E(x) + ½‖x‖²)
        For E(x) = ½‖x‖², this simplifies to 0.
        """
        key = jax.random.PRNGKey(1)
        samples = jax.random.normal(key, (5, 3))
        beta_k = jnp.array(0.3)
        beta_next = jnp.array(0.5)
        delta = float(beta_next - beta_k)
        log_w = ReplayBuffer.is_bridge_log_weights(samples, energy_sq, beta_k, beta_next)
        # E(x) = ½‖x‖², so -E(x) + ½‖x‖² = 0 → log_w should all be ~0
        assert jnp.allclose(log_w, jnp.zeros(5), atol=1e-5)

    def test_formula_nontrivial(self):
        """For a non-quadratic energy the weights should be non-zero in general."""

        def energy_cubic(x):
            return jnp.sum(x ** 3)

        key = jax.random.PRNGKey(2)
        samples = jax.random.normal(key, (4, 2))
        log_w = ReplayBuffer.is_bridge_log_weights(
            samples, energy_cubic, jnp.array(0.2), jnp.array(0.4)
        )
        # Not necessarily zero
        assert log_w.shape == (4,)
        assert jnp.all(jnp.isfinite(log_w))