"""Tests for progressive_is_tempering.utils.metrics (Distance/Energy/Geometric-W2)."""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from progressive_is_tempering.utils.metrics import (
    distance_w2,
    energy_w2,
    geometric_d2_matrix,
    geometric_w2,
    wasserstein2_1d,
)


def _random_isometry(rng, x):
    """Permute particles, rotate+reflect, and translate one (k, s) configuration."""
    k, s = x.shape
    Q, _ = np.linalg.qr(rng.normal(size=(s, s)))
    Q[:, 0] *= -1.0                                  # include a reflection
    return x[rng.permutation(k)] @ Q + rng.normal(size=(1, s))


def _random_relabel(rng, x):
    """Permute the particles of one (k, s) configuration."""
    return x[rng.permutation(x.shape[0])]


def _d2_reference(x0, x1):
    """Per-pair transcription of the specification, for comparison."""
    C = ((x0[:, None, :] - x1[None, :, :]) ** 2).sum(-1)
    _, col = linear_sum_assignment(C)
    x1p = x1[col]
    A = x0 - x0.mean(axis=0)
    B = x1p - x1p.mean(axis=0)
    U, _, Vt = np.linalg.svd(B.T @ A)
    return np.sum((A - B @ (U @ Vt)) ** 2)


# ── 1-D W2 ────────────────────────────────────────────────────────────────────

def test_w2_1d_shift_is_exact():
    a = np.random.default_rng(0).normal(size=1000)
    assert wasserstein2_1d(a, a + 0.3) == pytest.approx(0.3, abs=1e-12)


def test_w2_1d_drops_non_finite():
    a = np.array([0.0, 1.0, 2.0, np.inf, np.nan])
    b = np.array([0.0, 1.0, 2.0])
    assert wasserstein2_1d(a, b) == pytest.approx(0.0, abs=1e-12)


def test_w2_1d_empty_is_nan():
    assert np.isnan(wasserstein2_1d(np.array([np.inf]), np.array([1.0])))


def test_w2_1d_unequal_counts_uses_quantiles():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=20_000), rng.normal(size=30_000) + 1.0
    assert wasserstein2_1d(a, b) == pytest.approx(1.0, abs=0.05)


def test_distance_w2_pools_features():
    rng = np.random.default_rng(2)
    t, r = rng.normal(size=(100, 6)), rng.normal(size=(100, 6))
    assert distance_w2(t, r) == wasserstein2_1d(t.ravel(), r.ravel())


def test_energy_w2_invariant_to_common_constant():
    rng = np.random.default_rng(3)
    e_t, e_r = rng.normal(size=500), rng.normal(size=500) + 0.5
    assert energy_w2(e_t + 7.0, e_r + 7.0) == pytest.approx(energy_w2(e_t, e_r), abs=1e-12)


# ── Geometric-W2 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k,s", [(4, 2), (13, 3)])
def test_d2_zero_for_relabelled_copy(k, s):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(k, s))
    d2 = geometric_d2_matrix(x[None], _random_relabel(rng, x)[None], k, s)
    assert d2[0, 0] == pytest.approx(0.0, abs=1e-10)


def test_d2_zero_for_small_rotation_and_translation():
    # Assignment runs on unaligned coordinates (one-shot scheme), so only a
    # small motion is guaranteed to keep the identity assignment; centring and
    # Procrustes must then remove the rotation and translation exactly.
    x = 3.0 * np.random.default_rng(9).normal(size=(13, 3))
    th = 0.02
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    y = x @ R + np.array([[0.01, -0.02, 0.005]])
    d2 = geometric_d2_matrix(x[None], y[None], 13, 3)
    assert d2[0, 0] == pytest.approx(0.0, abs=1e-10)


def test_d2_zero_for_reflection():
    # Nearly planar configuration: flipping z barely moves any particle, so the
    # assignment stays the identity, and only an O(3) (not SO(3)) alignment
    # can bring the distance to zero.
    rng = np.random.default_rng(11)
    x = np.concatenate([3.0 * rng.normal(size=(13, 2)),
                        0.01 * rng.normal(size=(13, 1))], axis=1)
    y = x * np.array([[1.0, 1.0, -1.0]])
    d2 = geometric_d2_matrix(x[None], y[None], 13, 3)
    assert d2[0, 0] == pytest.approx(0.0, abs=1e-10)


def test_d2_is_upper_bound_never_negative():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(20, 13, 3))
    y = np.stack([_random_isometry(rng, c) for c in x])
    d2 = geometric_d2_matrix(x, y, 13, 3)
    assert (d2 >= -1e-12).all()


@pytest.mark.parametrize("k,s", [(4, 2), (13, 3)])
def test_d2_matrix_matches_per_pair_spec(k, s):
    rng = np.random.default_rng(5)
    xt, xr = rng.normal(size=(5, k, s)), rng.normal(size=(7, k, s))
    d2 = geometric_d2_matrix(xt.reshape(5, -1), xr.reshape(7, -1), k, s)
    expected = np.array([[_d2_reference(a, b) for b in xr] for a in xt])
    np.testing.assert_allclose(d2, expected, rtol=1e-10, atol=1e-12)


def test_geometric_w2_zero_for_relabelled_copy():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(40, 4, 2))
    y = np.stack([_random_relabel(rng, c) for c in x])
    assert geometric_w2(x, y, 4, 2, n_workers=1) == pytest.approx(0.0, abs=1e-6)


def test_geometric_w2_drops_non_finite_and_subsamples():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 8))
    x_bad = np.concatenate([x, np.full((1, 8), np.inf)])
    assert geometric_w2(x_bad, x, 4, 2, n_workers=1) == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(geometric_w2(x, x, 4, 2, max_n=10, n_workers=1))


def test_geometric_w2_parallel_matches_serial():
    rng = np.random.default_rng(8)
    xt, xr = rng.normal(size=(50, 8)), rng.normal(size=(40, 8))
    serial = geometric_w2(xt, xr, 4, 2, n_workers=1)
    parallel = geometric_w2(xt, xr, 4, 2, n_workers=2)
    assert parallel == pytest.approx(serial, rel=1e-12)
