"""
Alanine dipeptide (ALDP): force field, internal coordinates and target density.

Everything the ALDP notebook (`notebooks/aldp/aldp.py`) needs to evaluate the
target and to move between Cartesian and internal coordinates, in pure JAX,
float64 only (`jax.config.update("jax_enable_x64", True)` first):

* **Force field** (`ForceFieldParams`, `load_force_field`, `total_energy`):
  harmonic bonds and angles, periodic torsions, Lennard-Jones + Coulomb with
  1-2/1-3 exclusions and 1-4 exceptions (no cutoff), and GBSA-OBC implicit
  solvent (GB polar term + ACE surface-area term), in nm and kJ/mol. The
  parameter file is `data/aldp_ff14sb_obc1.npz`: PITA's force field (AMBER
  ff14SB + OBC1).
* **Target** (`AlanineDipeptide`): the Cartesian log Boltzmann factor
  `-U(x) / (k_B T)` restricted to the L-alanine sector (`aldp_in_l_sector`);
  `-inf` outside it.
* **Internal coordinates** (`aldp_zmatrix`, `InternalCoordinates`): the 60
  packed bonds/angles/torsions of a backbone-rooted Z-matrix, the physical
  conformational volume `log_volume`, and the `pullback` of a Cartesian density.
* **Sampling coordinates** (`SamplingCoordinates`): log/logit maps of bonds and
  angles to the real line, torsions left as radian circles.
* `angles_from_internal`: the biochemical (phi, psi) of an internal event.

The arithmetic is order-sensitive: keep expressions and their order as they
are, or results for a given seed change in the last bits.

Atom order: the 22-atom openmmtools / PITA topology
  0 H1  1 CH3  2 H2  3 H3  4 C  5 O            (ACE)
  6 N   7 H    8 CA  9 HA  10 CB 11 HB1 12 HB2 13 HB3 14 C 15 O   (ALA)
  16 N  17 H   18 C  19 H1 20 H2 21 H3         (NME)

Leaf module: imports only JAX and NumPy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "KB",
    "BOND_SCALE",
    "ANGLE_SCALE",
    "N_ATOMS",
    "REFERENCE_POSITIONS",
    "ForceFieldParams",
    "load_force_field",
    "total_energy",
    "log_boltzmann",
    "map_events",
    "ZMatrix",
    "aldp_zmatrix",
    "aldp_in_l_sector",
    "zmatrix_to_cartesian",
    "cartesian_to_zmatrix",
    "InternalCoordinates",
    "SamplingCoordinates",
    "AlanineDipeptide",
    "AlanineDipeptideInternal",
    "angles_from_internal",
]

#: Boltzmann constant in kJ/(mol K).
KB = 8.314462618e-3
#: Coulomb constant in kJ nm / (mol e^2).
ONE_4PI_EPS0 = 138.93545764438198
#: OBC dielectric offset (nm).
DIELECTRIC_OFFSET = 0.009
#: Declared bond / angle scales of the sampling coordinates (nm, rad); FAB's
#: `default_std` values. Torsions keep unit scale.
BOND_SCALE = 0.005
ANGLE_SCALE = 0.15

N_ATOMS = 22

#: A valid L-alanine structure (nm), used as the fallback event of invalid
#: inputs and as the centre of the sampling coordinates. OpenMM Reference
#: platform minimum of the openmmtools topology.
REFERENCE_POSITIONS = np.array([
    [0.20000010000000001, 0.1, -1.3e-07],
    [0.20000010000000001, 0.209, 1e-08],
    [0.14862640000000002, 0.24538490000000002, 0.0889824],
    [0.1486259, 0.2453852, -0.088982],
    [0.34274200000000005, 0.26407949999999997, -3.0000000000000004e-07],
    [0.439058, 0.1877406, -6.6e-07],
    [0.35553754000000004, 0.39696488, -3.1e-07],
    [0.273312, 0.45561600999999996, -1.3e-07],
    [0.48532621000000004, 0.46139253, -4.3000000000000007e-07],
    [0.5407596, 0.43155388, 0.08898152000000001],
    [0.56613044, 0.42208425, -0.12321480000000001],
    [0.51232615, 0.45213630000000005, -0.21312016],
    [0.6630484000000001, 0.47189354000000006, -0.12057907000000001],
    [0.58085401, 0.31408724000000005, -0.1241385],
    [0.47126759, 0.61294185, 1.4e-07],
    [0.36006445000000004, 0.66527027, 6.2e-07],
    [0.58460533, 0.6834883300000001, 2.5000000000000004e-07],
    [0.67370014, 0.6359162, -4e-08],
    [0.58460551, 0.8283883700000001, 6.2e-07],
    [0.48185761, 0.86477349, 1.04e-06],
    [0.6359798400000001, 0.8647731300000001, 0.08898283000000001],
    [0.6359790000000001, 0.8647735400000001, -0.08898187],
], dtype=np.float64)


def _require_x64():
    if not jax.config.jax_enable_x64:
        raise RuntimeError("progressive_is_tempering.alanine requires float64: call "
                           "jax.config.update('jax_enable_x64', True) first.")


# ---------------------------------------------------------------------------
# Force-field parameters
# ---------------------------------------------------------------------------

class BondParams(NamedTuple):
    atom_i: jax.Array
    atom_j: jax.Array
    r0: jax.Array        # nm
    k: jax.Array         # kJ/mol/nm^2


class AngleParams(NamedTuple):
    atom_i: jax.Array
    atom_j: jax.Array    # central atom
    atom_k: jax.Array
    theta0: jax.Array    # rad
    k: jax.Array         # kJ/mol/rad^2


class TorsionParams(NamedTuple):
    atom_i: jax.Array
    atom_j: jax.Array
    atom_k: jax.Array
    atom_l: jax.Array
    periodicity: jax.Array
    phase: jax.Array     # rad
    k: jax.Array         # kJ/mol


class NonbondedParams(NamedTuple):
    charges: jax.Array
    sigmas: jax.Array
    epsilons: jax.Array
    n_atoms: int
    exclusion_pairs: jax.Array      # (n, 2): no interaction (1-2, 1-3)
    exception_pairs: jax.Array      # (n, 2): explicit parameters (1-4)
    exception_chargeprod: jax.Array
    exception_sigma: jax.Array
    exception_epsilon: jax.Array


class GBSAParams(NamedTuple):
    charges: jax.Array
    radii: jax.Array
    scale_factors: jax.Array
    solute_dielectric: float
    solvent_dielectric: float
    probe_radius: float
    sa_energy: float
    alpha: float
    beta: float
    gamma: float


class ForceFieldParams(NamedTuple):
    bonds: BondParams
    angles: AngleParams
    torsions: TorsionParams
    nonbonded: NonbondedParams
    gbsa: GBSAParams
    masses: jax.Array
    n_atoms: int


def load_force_field(path) -> ForceFieldParams:
    """Read a force-field parameter file (`.npz`; NoCutoff, GBSA-OBC).

    Only the terms ALDP in implicit solvent has are supported; a file with
    CMAP, RB torsions, restraints, a periodic box or a nonbonded cutoff is
    rejected rather than silently evaluated without them.
    """
    _require_x64()
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["_metadata"]))
    unsupported = [k for k in ("has_rb_torsions", "has_cmap", "has_restraints", "has_box")
                   if metadata.get(k)]
    if unsupported or not metadata.get("has_gbsa") \
            or metadata.get("nb_cutoff") is not None or metadata.get("nb_switch_distance") is not None:
        raise ValueError(f"{path}: expected bonds/angles/torsions/nonbonded (no cutoff) + GBSA-OBC only; "
                         f"metadata {metadata}")
    bonds = BondParams(
        atom_i=jnp.array(data["bonds_atom_i"]),
        atom_j=jnp.array(data["bonds_atom_j"]),
        r0=jnp.array(data["bonds_r0"]),
        k=jnp.array(data["bonds_k"]),
    )
    angles = AngleParams(
        atom_i=jnp.array(data["angles_atom_i"]),
        atom_j=jnp.array(data["angles_atom_j"]),
        atom_k=jnp.array(data["angles_atom_k"]),
        theta0=jnp.array(data["angles_theta0"]),
        k=jnp.array(data["angles_k"]),
    )
    torsions = TorsionParams(
        atom_i=jnp.array(data["torsions_atom_i"]),
        atom_j=jnp.array(data["torsions_atom_j"]),
        atom_k=jnp.array(data["torsions_atom_k"]),
        atom_l=jnp.array(data["torsions_atom_l"]),
        periodicity=jnp.array(data["torsions_periodicity"]),
        phase=jnp.array(data["torsions_phase"]),
        k=jnp.array(data["torsions_k"]),
    )
    nonbonded = NonbondedParams(
        charges=jnp.array(data["nb_charges"]),
        sigmas=jnp.array(data["nb_sigmas"]),
        epsilons=jnp.array(data["nb_epsilons"]),
        n_atoms=metadata["n_atoms"],
        exclusion_pairs=jnp.array(data["nb_exclusion_pairs"]),
        exception_pairs=jnp.array(data["nb_exception_pairs"]),
        exception_chargeprod=jnp.array(data["nb_exception_chargeprod"]),
        exception_sigma=jnp.array(data["nb_exception_sigma"]),
        exception_epsilon=jnp.array(data["nb_exception_epsilon"]),
    )
    gbsa = GBSAParams(
        charges=jnp.array(data["gbsa_charges"]),
        radii=jnp.array(data["gbsa_radii"]),
        scale_factors=jnp.array(data["gbsa_scale_factors"]),
        solute_dielectric=metadata["gbsa_solute_dielectric"],
        solvent_dielectric=metadata["gbsa_solvent_dielectric"],
        probe_radius=metadata["gbsa_probe_radius"],
        sa_energy=metadata["gbsa_sa_energy"],
        alpha=metadata["gbsa_alpha"],
        beta=metadata["gbsa_beta"],
        gamma=metadata["gbsa_gamma"],
    )
    return ForceFieldParams(bonds=bonds, angles=angles, torsions=torsions, nonbonded=nonbonded,
                            gbsa=gbsa, masses=jnp.array(data["masses"]), n_atoms=metadata["n_atoms"])


# ---------------------------------------------------------------------------
# Energy terms (positions (n_atoms, 3) in nm -> kJ/mol)
# ---------------------------------------------------------------------------

def bond_energy(positions, params: BondParams):
    """sum 0.5 k (r - r0)^2."""
    ri = positions[params.atom_i]
    rj = positions[params.atom_j]
    dr = ri - rj
    # Safe norm: avoids NaN gradients at zero distance
    r = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-30)
    return jnp.sum(0.5 * params.k * (r - params.r0) ** 2)


def angle_energy(positions, params: AngleParams):
    """sum 0.5 k (theta - theta0)^2, theta = atan2(|v1 x v2|, v1 . v2)."""
    ri = positions[params.atom_i]
    rj = positions[params.atom_j]  # central atom
    rk = positions[params.atom_k]
    v1 = ri - rj
    v2 = rk - rj
    cross = jnp.cross(v1, v2)
    cross_norm = jnp.sqrt(jnp.sum(cross**2, axis=-1) + 1e-30)
    dot = jnp.sum(v1 * v2, axis=-1)
    theta = jnp.arctan2(cross_norm, dot)
    return jnp.sum(0.5 * params.k * (theta - params.theta0) ** 2)


def torsion_energy(positions, params: TorsionParams):
    """sum k (1 + cos(n phi - phase)), phi with OpenMM's sign convention."""
    ri = positions[params.atom_i]
    rj = positions[params.atom_j]
    rk = positions[params.atom_k]
    rl = positions[params.atom_l]
    b1 = rj - ri
    b2 = rk - rj
    b3 = rl - rk
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    b2_norm = b2 / jnp.sqrt(jnp.sum(b2**2, axis=-1, keepdims=True) + 1e-30)
    m1 = jnp.cross(n1, b2_norm)
    x = jnp.sum(n1 * n2, axis=-1)
    y = jnp.sum(m1 * n2, axis=-1)
    phi = jnp.arctan2(y, x)
    return jnp.sum(params.k * (1.0 + jnp.cos(params.periodicity * phi - params.phase)))


def _pair_energy(dist, sigma, epsilon, chargeprod):
    """LJ + Coulomb for pairs at distance `dist`."""
    sig_over_r = sigma / dist
    sig_over_r6 = sig_over_r**6
    lj = 4.0 * epsilon * (sig_over_r6**2 - sig_over_r6)
    coulomb = ONE_4PI_EPS0 * chargeprod / dist
    return lj + coulomb


def _nonbonded_energy_with_dist(params: NonbondedParams, dist):
    """LJ + Coulomb over all pairs (Lorentz-Berthelot), exclusions skipped,
    1-4 exceptions with their own parameters. No cutoff."""
    n_atoms = params.n_atoms
    upper = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)

    sigma_ij = (params.sigmas[:, None] + params.sigmas[None, :]) / 2.0
    epsilon_ij = jnp.sqrt(params.epsilons[:, None] * params.epsilons[None, :])
    chargeprod_ij = params.charges[:, None] * params.charges[None, :]

    # Excluded and exception pairs never enter the ordinary pair sum (adding
    # huge short-range terms and subtracting them later loses digits).
    exc_i, exc_j = params.exclusion_pairs[:, 0], params.exclusion_pairs[:, 1]
    exn_i, exn_j = params.exception_pairs[:, 0], params.exception_pairs[:, 1]
    mask = upper.at[exc_i, exc_j].set(False).at[exn_i, exn_j].set(False)
    normal_dist = jnp.where(mask, dist, 1.0)
    pair_e = _pair_energy(normal_dist, sigma_ij, epsilon_ij, chargeprod_ij)
    total = jnp.sum(jnp.where(mask, pair_e, 0.0))

    total = total + jnp.sum(_pair_energy(
        dist[exn_i, exn_j], params.exception_sigma,
        params.exception_epsilon, params.exception_chargeprod,
    ))
    return total


def _distance_matrix(positions):
    """(dist, dist_sq), both (n, n); dist has a 1e-30 floor under the root."""
    dr = positions[:, None, :] - positions[None, :, :]
    dist_sq = jnp.sum(dr**2, axis=-1)
    dist = jnp.sqrt(dist_sq + 1e-30)
    return dist, dist_sq


def _born_radii(positions, params: GBSAParams, dist):
    """OBC Born radii: HCT pair integral with OBC tanh rescaling (OpenMM's
    CustomGBForce expressions; Onufriev, Bashford, Case 2004)."""
    n_atoms = positions.shape[0]
    offset_radii = params.radii - DIELECTRIC_OFFSET
    scaled_radii = offset_radii * params.scale_factors

    or_i = offset_radii[:, None]
    sr_j = scaled_radii[None, :]

    r_scaled = dist + sr_j

    abs_r_minus_sr = jnp.abs(dist - sr_j)
    L_raw = jnp.maximum(or_i, abs_r_minus_sr)
    L = 1.0 / L_raw
    U = 1.0 / r_scaled

    L2 = L**2
    U2 = U**2
    r_safe = jnp.where(dist > 1e-15, dist, 1.0)
    inv_r = jnp.where(dist > 1e-15, 1.0 / r_safe, 0.0)

    term = (
        L - U
        + 0.25 * (dist - sr_j**2 * inv_r) * (U2 - L2)
        + 0.5 * inv_r * jnp.log(L_raw / r_scaled)
    )
    term = 0.5 * term

    active = (or_i < r_scaled) & ~jnp.eye(n_atoms, dtype=bool)
    term = jnp.where(active, term, 0.0)

    psi = offset_radii * jnp.sum(term, axis=1)

    tanh_sum = jnp.tanh(
        params.alpha * psi + params.beta * psi**2 + params.gamma * psi**3
    )
    born_radii = 1.0 / (1.0 / offset_radii - tanh_sum / params.radii)
    return born_radii


def _gb_energy(born_radii, dist_sq, params: GBSAParams):
    """Generalized Born polar solvation energy (Still's f_GB)."""
    n_atoms = born_radii.shape[0]
    pre_factor = (
        -ONE_4PI_EPS0
        * (1.0 / params.solute_dielectric - 1.0 / params.solvent_dielectric)
    )
    alpha2 = born_radii[:, None] * born_radii[None, :]
    D = dist_sq / (4.0 * alpha2)
    f_GB = jnp.sqrt(dist_sq + alpha2 * jnp.exp(-D))

    charge_prod = params.charges[:, None] * params.charges[None, :]
    Gpol = pre_factor * charge_prod / f_GB

    upper = jnp.triu(jnp.ones((n_atoms, n_atoms), dtype=bool), k=1)
    diag = jnp.eye(n_atoms, dtype=bool)
    energy = jnp.sum(jnp.where(upper, Gpol, 0.0)) + 0.5 * jnp.sum(jnp.where(diag, Gpol, 0.0))
    return energy


def _sa_energy(born_radii, params: GBSAParams):
    """ACE non-polar surface-area term."""
    surface_area_factor = 4.0 * jnp.pi * params.sa_energy
    r_plus_probe = params.radii + params.probe_radius
    ratio6 = (params.radii / born_radii) ** 6
    return jnp.sum(surface_area_factor * r_plus_probe**2 * ratio6)


def _gbsa_energy_with_dist(positions, params: GBSAParams, dist, dist_sq):
    br = _born_radii(positions, params, dist)
    return _gb_energy(br, dist_sq, params) + _sa_energy(br, params)


def total_energy(positions, params: ForceFieldParams):
    """Potential energy U(x) in kJ/mol, positions (n_atoms, 3) in nm."""
    dist, dist_sq = _distance_matrix(positions)
    components = {
        "bonds": bond_energy(positions, params.bonds),
        "angles": angle_energy(positions, params.angles),
        "torsions": torsion_energy(positions, params.torsions),
        "nonbonded": _nonbonded_energy_with_dist(params.nonbonded, dist),
        "gbsa": _gbsa_energy_with_dist(positions, params.gbsa, dist, dist_sq),
    }
    # Python's sum, in this order (0 + bonds + angles + ...): the summation
    # order of the original code.
    return sum(components.values())


def log_boltzmann(positions, params: ForceFieldParams, temperature: float):
    """-U(x) / (k_B T), positions (n_atoms, 3) in nm, temperature in K."""
    energy = total_energy(positions, params)
    return -energy / (KB * temperature)


def map_events(function, values, event_ndim):
    """Apply a single-event function over arbitrary leading batch axes (vmap
    over the flattened batch, then restore its shape)."""
    batch = values.shape[:-event_ndim]
    events = values.reshape((math.prod(batch),) + values.shape[-event_ndim:])
    result = jax.vmap(function)(events)
    return jax.tree_util.tree_map(lambda v: v.reshape(batch + v.shape[1:]), result)


# ---------------------------------------------------------------------------
# Z-matrix geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZMatrix:
    """Reference atoms per construction step, plus the construction order.

    `bond_ref[c]`, `angle_ref[c]`, `torsion_ref[c]` are construction steps
    (each below `c`); `atom_order[c]` is the atom placed at step c. Steps 0, 1,
    2 are the frame: atom at step 0 at the origin, step 1 on +x, step 2 in the
    xy-plane.
    """

    bond_ref: jax.Array
    angle_ref: jax.Array
    torsion_ref: jax.Array
    atom_order: Optional[jax.Array] = None

    @property
    def construction_order(self):
        if self.atom_order is None:
            return jnp.arange(self.n_atoms, dtype=jnp.int32)
        return jnp.asarray(self.atom_order, dtype=jnp.int32)

    @property
    def n_atoms(self) -> int:
        return int(self.bond_ref.shape[0])


def aldp_zmatrix() -> ZMatrix:
    """The backbone-rooted ALDP Z-matrix (openmmtools atom order).

    Construction steps 0, 1, 2 place C (14), CA (8) and N (6) of the alanine
    residue. CA's substituents are measured against a frame triple, so torsion
    0 is a stiff improper whose sign is the handedness: restricting it to
    (0, pi) selects L-alanine exactly. phi and psi are torsions 12 and 5.
    Within each methyl one hydrogen is referenced to the backbone and the other
    two to that hydrogen, so they sit near +/-120 degrees.
    """
    atom_order = jnp.array(
        [14, 8, 6, 10, 9, 11, 12, 13, 16, 15, 18, 17, 19, 20, 21, 4, 7, 5, 1, 0, 2, 3],
        dtype=jnp.int32,
    )
    bond_ref = jnp.array(
        [-1, 0, 1, 1, 1, 3, 3, 3, 0, 0, 8, 8, 10, 10, 10, 2, 2, 15, 15, 18, 18, 18],
        dtype=jnp.int32,
    )
    angle_ref = jnp.array(
        [-1, -1, 0, 2, 2, 1, 1, 1, 1, 1, 0, 0, 8, 8, 8, 1, 1, 2, 2, 15, 15, 15],
        dtype=jnp.int32,
    )
    torsion_ref = jnp.array(
        [-1, -1, -1, 0, 0, 2, 5, 5, 2, 8, 1, 10, 0, 12, 12, 0, 15, 1, 17, 2, 19, 19],
        dtype=jnp.int32,
    )
    return ZMatrix(bond_ref=bond_ref, angle_ref=angle_ref, torsion_ref=torsion_ref,
                   atom_order=atom_order)


def aldp_in_l_sector(positions):
    """L-alanine handedness, shape (...): the signed tetrahedron at CA (8)
    ordered N (6), C (14), CB (10) is positive. Positions (..., 22, 3)."""
    ca = positions[..., 8, :]
    n = positions[..., 6, :] - ca
    c = positions[..., 14, :] - ca
    cb = positions[..., 10, :] - ca
    return jnp.sum(jnp.cross(n, c)*cb, axis=-1) > 0


def _safe_normalize(x, eps: float = 1e-30):
    return x / jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def _dihedral_openmm(a, b, c, d):
    """The OpenMM-sign dihedral of four points (opposite to the biochemical sign)."""
    b1 = b - a
    b2 = c - b
    b3 = d - c
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    b2_hat = _safe_normalize(b2)
    m1 = jnp.cross(n1, b2_hat)
    x = jnp.sum(n1 * n2)
    y = jnp.sum(m1 * n2)
    return jnp.arctan2(y, x)


def zmatrix_to_cartesian(z_matrix: ZMatrix, bonds, angles, torsions):
    """Internal coordinates (in construction order) -> positions (n_atoms, 3)
    in the canonical frame and in atom order."""
    dtype = jnp.result_type(bonds, angles, torsions)
    bonds = jnp.asarray(bonds, dtype=dtype)
    angles = jnp.asarray(angles, dtype=dtype)
    torsions = jnp.asarray(torsions, dtype=dtype)
    n_atoms = z_matrix.n_atoms

    positions = jnp.zeros((n_atoms, 3), dtype=dtype)
    positions = positions.at[1].set(jnp.array([bonds[0], 0.0, 0.0], dtype=dtype))
    atom2 = positions[1] + bonds[1] * jnp.array(
        [-jnp.cos(angles[0]), jnp.sin(angles[0]), 0.0],
        dtype=dtype,
    )
    positions = positions.at[2].set(atom2)

    def place_atom(atom_index, pos):
        j = z_matrix.bond_ref[atom_index]
        k = z_matrix.angle_ref[atom_index]
        l = z_matrix.torsion_ref[atom_index]
        r = bonds[atom_index - 1]
        theta = angles[atom_index - 2]
        phi = torsions[atom_index - 3]

        a = pos[l]
        b = pos[k]
        c = pos[j]
        bc = _safe_normalize(c - b)
        normal = _safe_normalize(jnp.cross(b - a, bc))
        m = jnp.cross(normal, bc)
        direction = (
            -jnp.cos(theta) * bc
            + jnp.sin(theta) * (jnp.cos(phi) * m - jnp.sin(phi) * normal)
        )
        return pos.at[atom_index].set(c + r * direction)

    built = jax.lax.fori_loop(3, n_atoms, place_atom, positions)
    if z_matrix.atom_order is None:
        return built
    return jnp.zeros_like(built).at[z_matrix.construction_order].set(built)


def cartesian_to_zmatrix(z_matrix: ZMatrix, positions):
    """Positions (n_atoms, 3) -> (bonds, angles, torsions) in construction order."""
    positions = jnp.asarray(positions)
    if z_matrix.atom_order is not None:
        positions = positions[z_matrix.construction_order]
    n_atoms = z_matrix.n_atoms

    def bond_for_atom(atom_index):
        j = z_matrix.bond_ref[atom_index]
        dr = positions[atom_index] - positions[j]
        return jnp.sqrt(jnp.sum(dr**2) + 1e-30)

    def angle_for_atom(atom_index):
        j = z_matrix.bond_ref[atom_index]
        k = z_matrix.angle_ref[atom_index]
        u = positions[atom_index] - positions[j]
        v = positions[k] - positions[j]
        cross = jnp.cross(u, v)
        cross_norm = jnp.sqrt(jnp.sum(cross**2) + 1e-30)
        return jnp.arctan2(cross_norm, jnp.sum(u * v))

    def torsion_for_atom(atom_index):
        j = z_matrix.bond_ref[atom_index]
        k = z_matrix.angle_ref[atom_index]
        l = z_matrix.torsion_ref[atom_index]
        return _dihedral_openmm(positions[l], positions[k], positions[j], positions[atom_index])

    atom_indices = jnp.arange(n_atoms)
    bonds = jax.vmap(bond_for_atom)(atom_indices[1:])
    angles = jax.vmap(angle_for_atom)(atom_indices[2:])
    torsions = jax.vmap(torsion_for_atom)(atom_indices[3:])
    return bonds, angles, torsions


# ---------------------------------------------------------------------------
# Packed internal coordinates and the Cartesian conformational measure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InternalCoordinates:
    """`q = [bonds (N-1), angles (N-2), torsions (N-3)]` of a Z-matrix, in
    nm / rad, torsions in [-pi, pi). Rigid translation and orientation are
    integrated out by `log_volume` (orientation constant 8 pi^2 omitted).
    """

    z_matrix: ZMatrix

    @property
    def n_atoms(self):
        return self.z_matrix.n_atoms

    @property
    def dim(self):
        return 3 * self.n_atoms - 6

    @property
    def bond_slice(self):
        return slice(0, self.n_atoms - 1)

    @property
    def angle_slice(self):
        return slice(self.n_atoms - 1, 2 * self.n_atoms - 3)

    @property
    def torsion_slice(self):
        return slice(2 * self.n_atoms - 3, self.dim)

    @property
    def periodic_indices(self):
        return tuple(range(self.torsion_slice.start, self.dim))

    @property
    def bounds(self):
        """(lower, upper): bonds (0, inf), angles (0, pi), torsions [-pi, pi)."""
        n = self.n_atoms
        return (jnp.concatenate((jnp.zeros(2*n-3), jnp.full(n-3, -jnp.pi))),
                jnp.concatenate((jnp.full(n-1, jnp.inf),
                                 jnp.full(n-2, jnp.pi), jnp.full(n-3, jnp.pi))))

    def pack(self, bonds, angles, torsions):
        blocks = [jnp.asarray(v) for v in (bonds, angles, torsions)]
        return jnp.concatenate(blocks, axis=-1)

    def unpack(self, q):
        q = jnp.asarray(q)
        return q[..., self.bond_slice], q[..., self.angle_slice], q[..., self.torsion_slice]

    def _in_bounds(self, q):
        b, a, t = self.unpack(q)
        return (jnp.all(jnp.isfinite(q), axis=-1)
                & jnp.all(b > 0, axis=-1)
                & jnp.all((a > 0) & (a < jnp.pi), axis=-1)
                & jnp.all((t >= -jnp.pi) & (t < jnp.pi), axis=-1))

    def in_support(self, q):
        """Boolean (...): in bounds and a regular (non-collinear) construction."""
        q = jnp.asarray(q)
        bounds_ok = self._in_bounds(q)
        safe = jnp.where(bounds_ok[..., None], q, self._fallback())
        return bounds_ok & self._regular_geometry(self._reconstruct(safe))

    def _fallback(self):
        n = self.n_atoms
        return self.pack(jnp.ones(n-1), jnp.ones(n-2), jnp.linspace(0.2, 2., n-3))

    def _reconstruct(self, q):
        return map_events(lambda v: zmatrix_to_cartesian(
            self.z_matrix, *self.unpack(v)), q, 1)

    def _regular_geometry(self, x):
        """Exclude coincident bonds and collinear construction reference triples."""
        z = self.z_matrix
        ordered = x[..., z.construction_order, :]
        bonds = ordered[..., 1:, :] - ordered[..., z.bond_ref[1:], :]
        u = ordered[..., 0, :] - ordered[..., 1, :]
        v = ordered[..., 2, :] - ordered[..., 1, :]
        root_ok = jnp.sum(jnp.cross(u, v)**2, axis=-1) > 0
        u = ordered[..., z.torsion_ref[3:], :] - ordered[..., z.angle_ref[3:], :]
        v = ordered[..., z.bond_ref[3:], :] - ordered[..., z.angle_ref[3:], :]
        refs_ok = jnp.all(jnp.sum(jnp.cross(u, v)**2, axis=-1) > 0, axis=-1)
        return (root_ok & refs_ok & jnp.all(jnp.sum(bonds**2, axis=-1) > 0, axis=-1)
                & jnp.all(jnp.isfinite(x), axis=(-2, -1)))

    def to_cartesian(self, q):
        """(..., dim) -> (..., N, 3), canonical frame, atom order."""
        q = jnp.asarray(q)
        return self._reconstruct(q)

    def to_internal(self, x):
        """(..., N, 3) -> (..., dim); the rigid pose is discarded."""
        x = jnp.asarray(x)

        def convert(v):
            b, a, t = cartesian_to_zmatrix(self.z_matrix, v)
            return self.pack(b, a, (t + jnp.pi) % (2*jnp.pi) - jnp.pi)
        return map_events(convert, x, 2)

    def log_volume(self, q):
        """Physical conformational log volume, (...); -inf off support."""
        q = jnp.asarray(q)
        valid = self.in_support(q)
        safe = jnp.where(valid[..., None], q, self._fallback())
        return jnp.where(valid, self._log_volume(safe), -jnp.inf)

    def _log_volume(self, q):
        b, a, _ = self.unpack(q)
        return (2*jnp.sum(jnp.log(b), axis=-1)
                + jnp.sum(jnp.log(jnp.sin(a)), axis=-1))

    def pullback(self, log_density, q, *, reference=None):
        """A rotation/translation-invariant Cartesian log density expressed in
        internal coordinates (physical measure): `log_density(x(q)) +
        log_volume(q)`, -inf (zero gradient) off the chart or where
        `log_density` is -inf. `reference`: a valid fallback event (dim,)."""
        q = jnp.asarray(q)
        if reference is None:
            reference = self._fallback()
        else:
            reference = jnp.asarray(reference)
        valid = self.in_support(q)
        safe = jnp.where(valid[..., None], q, reference)
        values = jnp.asarray(log_density(self._reconstruct(safe)))
        valid = valid & ~jnp.isneginf(values)
        result = values + self._log_volume(safe)
        return jnp.where(valid, result, -jnp.inf)


# ---------------------------------------------------------------------------
# Sampling coordinates: log / logit maps, radian circles unchanged
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingCoordinates:
    """Map real sampling coordinates z to physical intervals.

    Half-lines: `q = lower + (reference - lower) exp(rate z)`; finite intervals:
    a logit map; both centred at `reference` with derivative `scale` at z = 0.
    Axes in `periodic_indices` are radian circles, unchanged (unit derivative).
    """

    reference: jax.Array
    scale: jax.Array
    lower: jax.Array
    upper: jax.Array
    periodic_indices: tuple = ()

    def __post_init__(self):
        for name in ("reference", "scale", "lower", "upper"):
            object.__setattr__(self, name, jnp.asarray(getattr(self, name)))

    @property
    def dim(self):
        return self.reference.shape[0]

    @property
    def _periodic(self):
        return jnp.zeros(self.dim, dtype=bool).at[jnp.asarray(self.periodic_indices, dtype=jnp.int32)].set(True)

    def _in_domain(self, q):
        left = jnp.where(self._periodic, q >= self.lower, q > self.lower)
        return jnp.all(jnp.isfinite(q) & left & (q < self.upper), axis=-1)

    def _coefficients(self):
        bounded = jnp.isfinite(self.upper) & ~self._periodic
        positive = ~jnp.isfinite(self.upper)
        # Every inactive branch has regular operands (no NaN adjoints).
        width = jnp.where(bounded, self.upper, self.lower + 1) - self.lower
        distance = jnp.where(positive, self.reference - self.lower, 1.)
        fraction = jnp.where(bounded, (self.reference-self.lower)/width, .5)
        offset = jnp.log(fraction) - jnp.log1p(-fraction)
        rate = self.scale / jnp.where(bounded, width*fraction*(1-fraction), distance)
        return bounded, positive, width, distance, offset, rate

    def _forward(self, z):
        bounded, positive, width, distance, offset, rate = self._coefficients()
        exponent = rate*jnp.where(positive, z, 0.)
        eta = offset + rate*jnp.where(bounded, z, 0.)
        q = jnp.where(self._periodic, z, jnp.where(
            bounded, self.lower + width*jax.nn.sigmoid(eta),
            self.lower + distance*jnp.exp(exponent)))
        log_det = jnp.where(self._periodic, 0., jnp.where(
            bounded, jnp.log(width) + jnp.log(rate) - jax.nn.softplus(-eta) - jax.nn.softplus(eta),
            jnp.log(self.scale) + exponent))
        return q, jnp.sum(log_det, axis=-1)

    def to_physical(self, z):
        """(..., dim) sampling -> physical coordinates."""
        z = jnp.asarray(z)
        q, _ = self._forward(z)
        return q

    def to_sampling(self, q):
        """(..., dim) physical -> sampling coordinates (circles unchanged)."""
        q = jnp.asarray(q)
        bounded, positive, width, distance, offset, rate = self._coefficients()
        fraction = jnp.where(bounded, (q-self.lower)/width, .5)
        logit = jnp.log(fraction) - jnp.log1p(-fraction)
        exponent = jnp.log(jnp.where(positive, q-self.lower, distance)/distance)
        return jnp.where(self._periodic, q, jnp.where(bounded, (logit-offset)/rate, exponent/rate))

    def log_abs_det_jacobian(self, z):
        """log |dq/dz|, shape (...)."""
        z = jnp.asarray(z)
        q, log_det = self._forward(z)
        return log_det

    def pullback(self, log_density, z):
        """`log_density(q(z)) + log|dq/dz|`; -inf (zero gradient) for events the
        map cannot represent (overflow, logit saturation) or where
        `log_density` is -inf."""
        z = jnp.asarray(z)
        fallback = self.to_sampling(self.reference)
        finite = jnp.all(jnp.isfinite(z), axis=-1)
        trial, trial_det = self._forward(jnp.where(
            finite[..., None], jax.lax.stop_gradient(z), fallback))
        valid = finite & self._in_domain(trial) & jnp.isfinite(trial_det)
        # Probe representability without differentiating overflow, then
        # evaluate only safe inputs.
        q, log_det = self._forward(jnp.where(valid[..., None], z, fallback))
        values = jnp.asarray(log_density(q))
        return jnp.where(valid & ~jnp.isneginf(values), values+log_det, -jnp.inf)


# ---------------------------------------------------------------------------
# The ALDP target
# ---------------------------------------------------------------------------

class AlanineDipeptide:
    """Flexible ACE-L-ALA-NME Boltzmann factor `-U(x) / (k_B T)`.

    Positions (..., 22, 3) in nm, float64 -> shape (...); -inf outside the
    L-alanine sector (and for non-finite input). Translation and rotation
    invariant: not a proper density on R^66 (sampling fixes the centre of mass).
    """

    def __init__(self, params: ForceFieldParams, temperature: float = 300.0):
        _require_x64()
        self.params = params
        self.temperature = temperature
        self.reference_positions = jnp.asarray(REFERENCE_POSITIONS)

    def __call__(self, x):
        valid = jnp.all(jnp.isfinite(x), axis=(-2, -1))
        valid = valid & aldp_in_l_sector(x)
        safe = jnp.where(valid[..., None, None], x, self.reference_positions)
        values = map_events(lambda y: log_boltzmann(y, self.params, self.temperature), safe, 2)
        return jnp.where(valid, values, -jnp.inf)


class AlanineDipeptideInternal:
    """The same target as a density of the 60 internal coordinates
    (`aldp_zmatrix`), physical measure: `cartesian(x(q)) + log_volume(q)`."""

    def __init__(self, cartesian: AlanineDipeptide):
        self.cartesian = cartesian

    @property
    def transform(self) -> InternalCoordinates:
        return InternalCoordinates(aldp_zmatrix())

    @property
    def dim(self) -> int:
        return self.transform.dim

    def __call__(self, q):
        transform = self.transform
        return transform.pullback(
            self.cartesian, q,
            reference=transform.to_internal(self.cartesian.reference_positions))


def angles_from_internal(q):
    """Biochemical (phi, psi) in radians, [-pi, pi), from internal events
    (..., 60) of `aldp_zmatrix` (torsions 12 and 5, sign flipped). NumPy."""
    q = np.asarray(q)
    return (-q[..., [53, 46]]+np.pi) % (2*np.pi)-np.pi
