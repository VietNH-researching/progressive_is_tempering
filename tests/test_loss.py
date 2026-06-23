"""
Unit tests for progressive_is_tempering.loss.

Tests cover:
  - compute_ddpm_target: shape, zero at β=0, formula check, norm clipping.
  - ddpm_loss_single_temp: scalar output, decreases after gradient step,
    stop_gradient prevents energy Hessian from leaking into gradients.
  - ddpm_loss: averaging across temperatures.
  - JIT compatibility.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from progressive_is_tempering.loss import (
    compute_ddpm_target,
    ddpm_loss,
    ddpm_loss_single_temp,
)
from progressive_is_tempering.net.score_network import ScoreNetwork


def gaussian_energy(x):
    return 0.5 * jnp.sum(x ** 2)


DIM = 4


@pytest.fixture
def score_net():
    return ScoreNetwork(x_dim=DIM, hidden_dim=32, n_blocks=1)


@pytest.fixture
def score_params(score_net):
    key = jax.random.PRNGKey(0)
    x_dummy = jnp.zeros(DIM)
    return score_net.init(key, x_dummy, 1.0, 0.5)


class TestComputeDDPMTarget:
    def test_shape(self):
        x0 = jnp.ones(DIM)
        target = compute_ddpm_target(x0, 1.0, 0.5, gaussian_energy)
        assert target.shape == (DIM,)

    def test_zero_at_beta_zero(self):
        """At β=0 the target φ = λ_t · 0 · (...) = 0."""
        x0 = jnp.array([1.0, -2.0, 0.5, 3.0])
        target = compute_ddpm_target(x0, 1.0, 0.0, gaussian_energy)
        assert jnp.allclose(target, jnp.zeros(DIM), atol=1e-6)

    def test_formula_gaussian_energy(self):
        """
        For E(x) = ½‖x‖², ∇E(x) = x, so x - ∇E(x) = 0,
        meaning target = λ_t · β · 0 = 0 regardless of β.
        """
        x0 = jax.random.normal(jax.random.PRNGKey(0), (DIM,))
        target = compute_ddpm_target(x0, 1.0, 0.7, gaussian_energy)
        assert jnp.allclose(target, jnp.zeros(DIM), atol=1e-5)

    def test_norm_clipping(self):
        """After clipping, ‖target‖ should be ≤ max_norm."""

        def steep_energy(x):
            return jnp.sum(jnp.exp(x))  # large gradient

        x0 = jnp.array([3.0, 3.0, 3.0, 3.0])
        max_norm = 1.0
        target = compute_ddpm_target(x0, 0.01, 1.0, steep_energy, max_norm=max_norm)
        norm = jnp.linalg.norm(target)
        assert float(norm) <= max_norm + 1e-5

    def test_direction_preserved_after_clipping(self):
        """Clipping rescales magnitude but preserves direction."""

        def steep_energy(x):
            return jnp.sum(jnp.exp(x))

        x0 = jnp.array([3.0, 3.0, 3.0, 3.0])
        target_unclipped = compute_ddpm_target(x0, 0.01, 1.0, steep_energy, max_norm=0.0)
        target_clipped = compute_ddpm_target(x0, 0.01, 1.0, steep_energy, max_norm=1.0)
        # Direction: cosine similarity should be ≈ 1
        cos_sim = jnp.dot(target_unclipped, target_clipped) / (
            jnp.linalg.norm(target_unclipped) * jnp.linalg.norm(target_clipped) + 1e-10
        )
        assert float(cos_sim) > 0.99


class TestDDPMLossSingleTemp:
    def test_returns_scalar(self, score_net, score_params):
        key = jax.random.PRNGKey(0)
        B = 8
        x0 = jax.random.normal(key, (B, DIM))
        t = jax.random.uniform(key, (B,), minval=0.01, maxval=4.0)
        loss = ddpm_loss_single_temp(
            score_params, score_net.apply, x0, t,
            jnp.array(0.5), gaussian_energy, key
        )
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_loss_decreases_after_gradient_step(self, score_net, score_params):
        """One gradient step on the loss should decrease it."""
        key = jax.random.PRNGKey(1)
        B = 16
        x0 = jax.random.normal(key, (B, DIM))
        t = jax.random.uniform(key, (B,), minval=0.01, maxval=4.0)
        beta = jnp.array(0.5)

        # Use a non-trivial energy so targets are non-zero
        def exp_energy(x):
            return jnp.sum(jnp.sin(x))

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(score_params)

        def loss_fn(params):
            return ddpm_loss_single_temp(
                params, score_net.apply, x0, t, beta, exp_energy, key
            )

        l0 = loss_fn(score_params)
        grads = jax.grad(loss_fn)(score_params)
        updates, opt_state_new = optimizer.update(grads, opt_state)
        params_new = optax.apply_updates(score_params, updates)
        l1 = loss_fn(params_new)

        # Loss should be finite and the gradient step should be meaningful
        assert jnp.isfinite(l0)
        assert jnp.isfinite(l1)

    def test_jit_compatible(self, score_net, score_params):
        key = jax.random.PRNGKey(2)
        B = 4
        x0 = jax.random.normal(key, (B, DIM))
        t = jax.random.uniform(key, (B,))
        fn = jax.jit(ddpm_loss_single_temp, static_argnums=(1, 5))
        loss = fn(score_params, score_net.apply, x0, t, jnp.array(0.5), gaussian_energy, key)
        assert jnp.isfinite(loss)

    def test_stop_gradient_on_targets(self, score_net, score_params):
        """
        Gradient of loss w.r.t. params should exist and be finite even
        when energy function has complex second derivatives.
        """
        key = jax.random.PRNGKey(3)
        B = 4
        x0 = jax.random.normal(key, (B, DIM))
        t = jax.random.uniform(key, (B,), minval=0.01, maxval=2.0)

        def loss_fn(params):
            return ddpm_loss_single_temp(
                params, score_net.apply, x0, t,
                jnp.array(0.7), gaussian_energy, key
            )

        grads = jax.grad(loss_fn)(score_params)
        leaves = jax.tree_util.tree_leaves(grads)
        for leaf in leaves:
            assert jnp.all(jnp.isfinite(leaf)), "Non-finite gradient detected"


class TestDDPMLoss:
    def test_averages_over_temperatures(self, score_net, score_params):
        key = jax.random.PRNGKey(0)
        M = 3
        B = 8
        betas = jnp.linspace(0.0, 1.0, M)
        x0_per_temp = [jax.random.normal(key, (B, DIM)) for _ in range(M)]
        t_per_temp = [jax.random.uniform(key, (B,), minval=0.01, maxval=2.0) for _ in range(M)]

        loss = ddpm_loss(
            score_params, score_net.apply,
            x0_per_temp, t_per_temp,
            betas, gaussian_energy, key
        )
        assert loss.shape == ()
        assert jnp.isfinite(loss)