"""
Unit tests for progressive_is_tempering.net.score_network.

Tests cover:
  - ScoreNetwork: output shape for a single sample.
  - β=0 initialization: output near zero (small output_init_scale).
  - vmap for batched evaluation.
  - JIT compatibility.
  - Gradient w.r.t. params flows cleanly.
  - Input wiring: different t/β embeddings affect output.
"""

import jax
import jax.numpy as jnp
import pytest

from progressive_is_tempering.net.score_network import ScoreNetwork

DIM = 4


@pytest.fixture
def net():
    return ScoreNetwork(x_dim=DIM, hidden_dim=32, n_blocks=1, output_init_scale=1e-4)


@pytest.fixture
def params(net):
    key = jax.random.PRNGKey(0)
    x = jnp.zeros(DIM)
    return net.init(key, x, 1.0, 0.5)


class TestScoreNetworkShape:
    def test_single_sample_output_shape(self, net, params):
        x = jnp.zeros(DIM)
        out = net.apply(params, x, 1.0, 0.5)
        assert out.shape == (DIM,)

    def test_batched_via_vmap(self, net, params):
        B = 6
        xs = jnp.zeros((B, DIM))
        ts = jnp.ones(B)
        betas = jnp.full(B, 0.5)
        batched_apply = jax.vmap(lambda x, t, b: net.apply(params, x, t, b))
        out = batched_apply(xs, ts, betas)
        assert out.shape == (B, DIM)


class TestScoreNetworkInitialization:
    def test_small_output_at_init(self, net, params):
        """Initial outputs should be near zero due to small output_init_scale."""
        key = jax.random.PRNGKey(1)
        xs = jax.random.normal(key, (50, DIM))
        ts = jax.random.uniform(key, (50,), minval=0.01, maxval=4.0)
        betas = jax.random.uniform(key, (50,), minval=0.0, maxval=1.0)

        batched = jax.vmap(lambda x, t, b: net.apply(params, x, t, b))
        outs = batched(xs, ts, betas)
        max_abs = float(jnp.max(jnp.abs(outs)))
        # With output_init_scale=1e-4 and reasonable inputs, outputs should be small
        assert max_abs < 0.1, f"Initial output too large: {max_abs}"


class TestScoreNetworkGradients:
    def test_grad_wrt_params_is_finite(self, net, params):
        x = jax.random.normal(jax.random.PRNGKey(0), (DIM,))
        t = 1.0
        beta = 0.5

        def loss(p):
            return jnp.sum(net.apply(p, x, t, beta) ** 2)

        grads = jax.grad(loss)(params)
        leaves = jax.tree_util.tree_leaves(grads)
        for leaf in leaves:
            assert jnp.all(jnp.isfinite(leaf)), "Non-finite gradient in params"

    def test_grad_wrt_x(self, net, params):
        """Score network output should be differentiable w.r.t. x."""
        x = jax.random.normal(jax.random.PRNGKey(0), (DIM,))

        grad_x = jax.grad(lambda xi: jnp.sum(net.apply(params, xi, 1.0, 0.5)))(x)
        assert jnp.all(jnp.isfinite(grad_x))


class TestScoreNetworkJIT:
    def test_jit_single(self, net, params):
        x = jnp.zeros(DIM)
        fn = jax.jit(net.apply)
        out = fn(params, x, 1.0, 0.5)
        assert out.shape == (DIM,)

    def test_jit_vmap(self, net, params):
        B = 4
        xs = jnp.zeros((B, DIM))
        ts = jnp.ones(B)
        betas = jnp.full(B, 0.5)
        fn = jax.jit(jax.vmap(lambda x, t, b: net.apply(params, x, t, b)))
        out = fn(xs, ts, betas)
        assert out.shape == (B, DIM)


class TestScoreNetworkInputWiring:
    def test_different_betas_give_different_outputs(self, net, params):
        """Changing β should change the output (embedding is non-trivial)."""
        x = jax.random.normal(jax.random.PRNGKey(10), (DIM,))
        out_low = net.apply(params, x, 1.0, 0.0)
        out_high = net.apply(params, x, 1.0, 1.0)
        # They may both be small at init, but should differ
        assert not jnp.allclose(out_low, out_high)

    def test_different_times_give_different_outputs(self, net, params):
        x = jax.random.normal(jax.random.PRNGKey(11), (DIM,))
        out_t1 = net.apply(params, x, 0.1, 0.5)
        out_t2 = net.apply(params, x, 3.0, 0.5)
        assert not jnp.allclose(out_t1, out_t2)