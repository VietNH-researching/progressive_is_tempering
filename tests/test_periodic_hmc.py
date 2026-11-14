"""
Tests for progressive_is_tempering.periodic_hmc (HMC with circles, SMC ladder).

Tests cover:
  - wrap_periodic: circle axes land in [-pi, pi), other axes untouched.
  - geometric_anneal: endpoints, and -inf not leaking at t = 0 / 1.
  - HMC kernel: moves keep circle coordinates canonical; rung 0 draws fresh
    from the reference; adapt_params gives a step size with acceptance near
    the target band; the kernel leaves a Gaussian x von Mises target invariant
    (statistical check).
  - smc_sampler: ladder from 0 to 1, increasing; final particles match the
    target's moments (statistical check); keep_all_states shapes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progressive_is_tempering import periodic_hmc as ph

KAPPA, MU, SIG = 2.0, 1.0, 0.5   # target: N(MU, SIG^2) on axis 0, von Mises(0, KAPPA) on axis 1


def _rvs_ref(key):
    k1, k2 = jax.random.split(key)
    return jnp.stack([jax.random.normal(k1), jax.random.uniform(k2, minval=-jnp.pi, maxval=jnp.pi)])


def _logpdf_ref(x):
    return -0.5 * x[0] ** 2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(2 * jnp.pi)


def _logpdf_target(x):
    return -0.5 * ((x[0] - MU) / SIG) ** 2 + KAPPA * jnp.cos(x[1])


PROBLEM = ph.PTProblem(_rvs_ref, _logpdf_ref, _logpdf_target, exact_ref=True)
ENGINE = ph.make_engine(0.07, (1,))


def _exact_target(key, n):
    k1, k2 = jax.random.split(key)
    x0 = MU + SIG * jax.random.normal(k1, (n,))
    # von Mises by rejection from the uniform (envelope e^{KAPPA})
    c = jax.random.uniform(k2, (8 * n,), minval=-jnp.pi, maxval=jnp.pi)
    u = jax.random.uniform(jax.random.fold_in(k2, 1), (8 * n,))
    acc = np.asarray(c)[np.asarray(u) < np.exp(KAPPA * (np.cos(np.asarray(c)) - 1))]
    return jnp.stack([x0, jnp.asarray(acc[:n])], axis=1)


def test_wrap_periodic():
    x = jnp.array([[10.0, 3.5, -7.0], [0.1, -np.pi, np.pi]])
    w = ph.wrap_periodic(x, (1, 2))
    np.testing.assert_allclose(np.asarray(w[:, 0]), np.asarray(x[:, 0]))
    assert np.all(np.asarray(w[:, 1:]) >= -np.pi) and np.all(np.asarray(w[:, 1:]) < np.pi)
    np.testing.assert_allclose(np.cos(np.asarray(w[:, 1:])), np.cos(np.asarray(x[:, 1:])), atol=1e-6)
    assert ph.wrap_periodic(x, ()) is x


def test_geometric_anneal_endpoints():
    np.testing.assert_allclose(float(ph.geometric_anneal(0.3, 1.0, 2.0)), 1.3)
    assert float(ph.geometric_anneal(0.0, 1.0, -jnp.inf)) == 1.0
    assert float(ph.geometric_anneal(1.0, -jnp.inf, 2.0)) == 2.0


def test_rung_zero_draws_from_reference():
    kern = ENGINE(PROBLEM, 0.0, 0)
    x = ph.batch_init_extend(jax.random.key(0), jnp.zeros((64, 2)), kern)
    params = kern.adapt_params(jax.random.key(1), x)
    y = ph.batch_move(jax.random.key(2), x, params, kern)
    assert np.std(np.asarray(y.position[:, 0])) > 0.5      # fresh N(0,1) draws, not the zeros


def test_adapted_kernel_preserves_target():
    n = 4096
    x0 = _exact_target(jax.random.key(3), n)
    kern = ENGINE(PROBLEM, 1.0, 1)
    x = ph.batch_init_extend(jax.random.key(4), x0, kern)
    params = kern.adapt_params(jax.random.key(5), x)
    acc = float(jnp.mean(ph.batch_move(jax.random.key(6), x, params, kern).accepted))
    assert 0.7 < acc <= 1.0
    y = ph.repeated_batch_move(jax.random.key(7), x, params, kern, 30)
    pos = np.asarray(y.position)
    assert np.all((pos[:, 1] >= -np.pi) & (pos[:, 1] < np.pi))
    np.testing.assert_allclose(pos[:, 0].mean(), MU, atol=0.05)
    np.testing.assert_allclose(pos[:, 0].std(), SIG, atol=0.05)
    np.testing.assert_allclose(np.cos(pos[:, 1]).mean(), np.cos(np.asarray(x0[:, 1])).mean(), atol=0.05)


def test_smc_sampler_ladder_and_target_moments():
    states, logz, ladder = ph.smc_sampler(jax.random.key(8), PROBLEM, n_particles=2048, engine=ENGINE,
                                          n_mcmc_steps=10, local_ess_ratio=0.5, verbose=False,
                                          keep_all_states=True)
    t = np.array([float(v) for v in ladder])
    assert t[0] == 0.0 and t[-1] == 1.0 and np.all(np.diff(t) > 0)
    assert states.shape == (len(ladder), 2048, 2)
    assert set(logz) == {"SMC_like", "robust", "reverse_logistic"}
    top = np.asarray(states[-1])
    np.testing.assert_allclose(top[:, 0].mean(), MU, atol=0.08)
    # E[cos] under von Mises(KAPPA) = I1(KAPPA) / I0(KAPPA)
    from scipy.special import i0, i1
    np.testing.assert_allclose(np.cos(top[:, 1]).mean(), i1(KAPPA) / i0(KAPPA), atol=0.08)
    # log Z_target / Z_ref: Z_target = sqrt(2 pi) SIG * 2 pi I0(KAPPA), Z_ref = 1
    log_z = np.log(np.sqrt(2 * np.pi) * SIG * 2 * np.pi * i0(KAPPA))
    np.testing.assert_allclose(float(logz["SMC_like"]), log_z, atol=0.1)


def test_smc_sampler_validates_arguments():
    with pytest.raises(ValueError):
        ph.smc_sampler(jax.random.key(0), PROBLEM, n_particles=0, engine=ENGINE, n_mcmc_steps=1,
                       local_ess_ratio=0.5, verbose=False)
    with pytest.raises(ValueError):
        ph.smc_sampler(jax.random.key(0), PROBLEM, n_particles=8, engine=ENGINE, n_mcmc_steps=1,
                       local_ess_ratio=1.0, verbose=False)
