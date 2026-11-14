"""
Tests for progressive_is_tempering.alanine (ALDP force field, coordinates, target).

Tests cover:
  - Force-field file: loads, has the ALDP shapes, rejects unsupported terms.
  - Energy: finite at the reference structure, invariant under rigid motions,
    gradient finite; the target is -inf on the D enantiomer.
  - Internal coordinates: Cartesian -> internal -> Cartesian preserves the
    geometry, internal -> Cartesian -> internal is the identity; the internal
    density is the Cartesian one plus the conformational log volume.
  - Sampling coordinates: round trip, log-Jacobian against autodiff.
  - (phi, psi): `angles_from_internal` equals the backbone dihedrals.

float64 throughout (the module requires it), scoped to each test.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progressive_is_tempering import alanine

FF_PATH = Path(__file__).resolve().parents[1] / "data" / "aldp_ff14sb_obc1.npz"


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


@pytest.fixture
def params():
    return alanine.load_force_field(FF_PATH)


@pytest.fixture
def target(params):
    return alanine.AlanineDipeptide(params, 300.0)


@pytest.fixture
def x_ref():
    return jnp.asarray(alanine.REFERENCE_POSITIONS)


def _random_rigid(key, x):
    """A uniform-ish rotation (QR of a Gaussian, det +1) and a translation."""
    k1, k2 = jax.random.split(key)
    Q, R = jnp.linalg.qr(jax.random.normal(k1, (3, 3)))
    Q = Q * jnp.sign(jnp.diag(R))[None, :]
    Q = Q * jnp.linalg.det(Q)
    return x @ Q.T + jax.random.normal(k2, (1, 3))


def _perturbed(key, x, n, scale=0.003):
    return x[None] + scale * jax.random.normal(key, (n,) + x.shape)


def _dihedral_biochem(p0, p1, p2, p3):
    """IUPAC dihedral (the sign mdtraj and biochemistry use), radians."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1n, -1, keepdims=True) * b1n
    w = b2 - np.sum(b2 * b1n, -1, keepdims=True) * b1n
    return np.arctan2(np.sum(np.cross(b1n, v) * w, -1), np.sum(v * w, -1))


class TestForceField:
    def test_loads_aldp_shapes(self, params):
        assert params.n_atoms == 22
        assert params.bonds.atom_i.shape == (21,)
        assert params.masses.shape == (22,)
        assert params.gbsa.charges.shape == (22,)

    def test_rejects_unsupported_terms(self, tmp_path):
        d = dict(np.load(FF_PATH))
        meta = json.loads(str(d["_metadata"]))
        meta["has_cmap"] = True
        d["_metadata"] = np.array(json.dumps(meta))
        np.savez(tmp_path / "bad.npz", **d)
        with pytest.raises(ValueError):
            alanine.load_force_field(tmp_path / "bad.npz")

    def test_requires_x64(self):
        with jax.enable_x64(False):
            with pytest.raises(RuntimeError):
                alanine.load_force_field(FF_PATH)


class TestTarget:
    def test_reference_finite_and_in_l_sector(self, target, x_ref):
        assert bool(alanine.aldp_in_l_sector(x_ref))
        assert np.isfinite(float(target(x_ref)))
        assert np.all(np.isfinite(np.asarray(jax.grad(target)(x_ref))))

    def test_value_is_minus_energy_over_kt(self, params, target, x_ref):
        u = alanine.total_energy(x_ref, params)
        np.testing.assert_allclose(float(target(x_ref)), -float(u) / (alanine.KB * 300.0), rtol=1e-12)

    def test_rigid_motion_invariance(self, target, x_ref):
        xs = _perturbed(jax.random.key(0), x_ref, 8)
        moved = jax.vmap(_random_rigid)(jax.random.split(jax.random.key(1), 8), xs)
        np.testing.assert_allclose(np.asarray(jax.vmap(target)(moved)),
                                   np.asarray(jax.vmap(target)(xs)), rtol=1e-9)

    def test_d_enantiomer_rejected(self, target, x_ref):
        mirror = x_ref * jnp.array([1.0, 1.0, -1.0])
        assert not bool(alanine.aldp_in_l_sector(mirror))
        assert float(target(mirror)) == -np.inf
        # zero (not NaN) gradient where the target is -inf
        assert np.all(np.asarray(jax.grad(target)(mirror)) == 0.0)

    def test_batched_call(self, target, x_ref):
        xs = _perturbed(jax.random.key(2), x_ref, 6).reshape(2, 3, 22, 3)
        out = target(xs)
        assert out.shape == (2, 3)
        np.testing.assert_allclose(np.asarray(out).ravel(),
                                   np.asarray(jax.vmap(target)(xs.reshape(6, 22, 3))), rtol=1e-12)


class TestInternalCoordinates:
    def test_dimensions(self):
        ic = alanine.InternalCoordinates(alanine.aldp_zmatrix())
        assert ic.dim == 60
        assert ic.periodic_indices == tuple(range(41, 60))

    def test_round_trips(self, x_ref):
        ic = alanine.InternalCoordinates(alanine.aldp_zmatrix())
        xs = _perturbed(jax.random.key(3), x_ref, 16)
        q = jax.vmap(ic.to_internal)(xs)
        x_back = jax.vmap(ic.to_cartesian)(q)
        iu, ju = np.triu_indices(22, 1)
        d = lambda x: np.linalg.norm(np.asarray(x)[:, iu] - np.asarray(x)[:, ju], axis=-1)
        np.testing.assert_allclose(d(x_back), d(xs), atol=1e-12)
        np.testing.assert_allclose(np.asarray(jax.vmap(ic.to_internal)(x_back)), np.asarray(q), atol=1e-12)
        assert np.all(np.asarray(ic.in_support(q)))

    def test_internal_density_is_pullback(self, target, x_ref):
        internal = alanine.AlanineDipeptideInternal(target)
        ic = internal.transform
        q = jax.vmap(ic.to_internal)(_perturbed(jax.random.key(4), x_ref, 8))
        expected = jax.vmap(lambda v: target(ic.to_cartesian(v)) + ic.log_volume(v))(q)
        np.testing.assert_allclose(np.asarray(jax.vmap(internal)(q)), np.asarray(expected), rtol=1e-12)

    def test_angles_from_internal_are_phi_psi(self, x_ref):
        ic = alanine.InternalCoordinates(alanine.aldp_zmatrix())
        xs = np.asarray(_perturbed(jax.random.key(5), x_ref, 16, scale=0.02))
        a = alanine.angles_from_internal(np.asarray(jax.vmap(ic.to_internal)(jnp.asarray(xs))))
        phi = _dihedral_biochem(xs[:, 4], xs[:, 6], xs[:, 8], xs[:, 14])      # C(ACE)-N-CA-C
        psi = _dihedral_biochem(xs[:, 6], xs[:, 8], xs[:, 14], xs[:, 16])     # N-CA-C-N(NME)
        wrap = lambda v: (v + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(wrap(a[:, 0] - phi), 0.0, atol=1e-10)
        np.testing.assert_allclose(wrap(a[:, 1] - psi), 0.0, atol=1e-10)


class TestSamplingCoordinates:
    @pytest.fixture
    def coords(self, x_ref):
        ic = alanine.InternalCoordinates(alanine.aldp_zmatrix())
        q0 = ic.to_internal(x_ref)
        lower, upper = ic.bounds
        lower, upper = lower.at[41].set(0.), upper.at[41].set(jnp.pi)
        scales = jnp.concatenate((jnp.full(21, alanine.BOND_SCALE),
                                  jnp.full(21, alanine.ANGLE_SCALE), jnp.ones(18)))
        return alanine.SamplingCoordinates(q0, scales, lower, upper, tuple(range(42, 60)))

    def test_reference_maps_to_zero(self, coords):
        np.testing.assert_allclose(np.asarray(coords.to_sampling(coords.reference))[:42], 0.0, atol=1e-10)

    def test_round_trip(self, coords):
        z = 0.5 * jax.random.normal(jax.random.key(6), (10, 60))
        z = z.at[:, 42:].set(jax.random.uniform(jax.random.key(7), (10, 18), minval=-np.pi, maxval=np.pi))
        np.testing.assert_allclose(np.asarray(coords.to_sampling(coords.to_physical(z))),
                                   np.asarray(z), atol=1e-9)

    def test_log_jacobian_matches_autodiff(self, coords):
        z = 0.5 * jax.random.normal(jax.random.key(8), (60,))
        # the map is coordinate-wise: the Jacobian is diagonal
        diag = jax.vmap(lambda e: jax.jvp(coords.to_physical, (z,), (e,))[1])(jnp.eye(60)).diagonal()
        np.testing.assert_allclose(float(coords.log_abs_det_jacobian(z)),
                                   float(jnp.sum(jnp.log(jnp.abs(diag)))), rtol=1e-10)
