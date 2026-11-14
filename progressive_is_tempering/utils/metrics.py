"""
Final-evaluation metrics: Distance-W2, Energy-W2 and Geometric-W2.

These are the three metrics reported for the method:

  - `distance_w2`  — 2-Wasserstein between the *pooled* summary statistics
    (pairwise distances for particle targets, the coordinates for GMM).
    Recorded as `w2_pooled`.
  - `energy_w2`    — 2-Wasserstein between the energy distributions. Recorded
    as `w2_energy`.
  - `geometric_w2` — 2-Wasserstein between configurations under the ground
    distance "min over particle permutations and rotations/reflections",
    approximated one-shot (assignment first, then Procrustes). Particle targets
    only: it is undefined for GMM, which has no particle symmetry.

Alanine dipeptide is scored with the metrics of PITA's Table 2 instead (no
particle permutation symmetry, so Geometric-W2 does not apply):

  - `ramachandran_kl` — KL(reference || sample) between 2-D (phi, psi)
    histograms (Rama-KL).
  - `wasserstein_1d_exact` — exact 1-D W1 / W2 between energy distributions
    (Energy-W1, Energy-W2, in kT).
  - `torus_w2` — exact 2-Wasserstein on the (phi, psi) torus (T-W2).

NumPy / SciPy / POT rather than JAX: these run once, at the end of a run, on
finished samples pulled off the device. Never imports from the core package
(one-directional layering).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "wasserstein2_1d",
    "distance_w2",
    "energy_w2",
    "geometric_d2_matrix",
    "geometric_w2",
    "ramachandran_kl",
    "wasserstein_1d_exact",
    "torus_w2",
]


# ---------------------------------------------------------------------------
# 1-D Wasserstein (Distance-W2, Energy-W2)
# ---------------------------------------------------------------------------

def wasserstein2_1d(a, b, n_quantiles: int = 1000) -> float:
    """2-Wasserstein distance between two 1-D empirical distributions.

    In one dimension optimal transport is sorting, so W2 is the L2 distance
    between the quantile functions. With equal sample counts that is the RMS
    gap between the sorted samples (exact); with unequal counts both sides are
    evaluated on `n_quantiles` midpoint quantiles `(i + ½) / n_quantiles`.

    Non-finite entries are dropped from each side independently — an LJ13
    collision has infinite energy and must not poison the whole statistic.
    Returns nan if either side is empty after dropping.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")

    if a.size == b.size:
        return float(np.sqrt(np.mean((np.sort(a) - np.sort(b)) ** 2)))

    q = (np.arange(n_quantiles) + 0.5) / n_quantiles
    return float(np.sqrt(np.mean((np.quantile(a, q) - np.quantile(b, q)) ** 2)))


def distance_w2(test_stats, ref_stats) -> float:
    """Distance-W2: W2 between the pooled summary statistics.

    `test_stats` (n, f) and `ref_stats` (m, f) are flattened into one 1-D
    distribution each. Pooling is legitimate because the features are
    exchangeable — i.i.d. coordinates under a factorised GMM, particle
    permutation symmetry for the pairwise distances of DW4 / LJ13.
    """
    return wasserstein2_1d(np.asarray(test_stats).reshape(-1),
                           np.asarray(ref_stats).reshape(-1))


def energy_w2(test_energies, ref_energies) -> float:
    """Energy-W2: W2 between the energy distributions, E(x) per sample.

    Invariant to a common additive constant in E, so an unnormalised energy is
    fine as long as test and reference go through the same function.
    """
    return wasserstein2_1d(test_energies, ref_energies)


# ---------------------------------------------------------------------------
# Geometric-W2
# ---------------------------------------------------------------------------

def _geometric_d2_rows(x_rows: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """Squared aligned distances from each of `x_rows` (r, k, s) to every
    configuration of `x_ref` (m, k, s). Returns (r, m) float64.

    Per pair (x0 = test, x1 = reference), one-shot, as specified:
      1. permutation of x1 by linear assignment on the *squared* Euclidean
         cost between unaligned particles;
      2. both configurations centred;
      3. orthogonal Procrustes over O(s) — reflections allowed;
      4. squared aligned distance, raw (no division by the particle count).
    """
    from scipy.optimize import linear_sum_assignment

    m, k, _ = x_ref.shape
    out = np.empty((x_rows.shape[0], m), dtype=np.float64)
    cols = np.empty((m, k), dtype=np.intp)
    for r, x0 in enumerate(x_rows):
        # 1. C[j, a, b] = ||x0[a] - x_ref[j, b]||²
        C = np.sum((x0[None, :, None, :] - x_ref[:, None, :, :]) ** 2, axis=-1)
        for j in range(m):
            cols[j] = linear_sum_assignment(C[j])[1]
        x1p = np.take_along_axis(x_ref, cols[:, :, None], axis=1)   # (m, k, s)

        # 2. centre
        A = x0 - x0.mean(axis=0)                                     # (k, s)
        B = x1p - x1p.mean(axis=1, keepdims=True)                    # (m, k, s)

        # 3. Omega = U Vt with U S Vt = svd(Bᵀ A)
        U, _, Vt = np.linalg.svd(np.swapaxes(B, 1, 2) @ A)           # (m, s, s)
        Omega = U @ Vt

        # 4.
        out[r] = np.sum((A[None] - B @ Omega) ** 2, axis=(1, 2))
    return out


_WORKER_REF: Optional[np.ndarray] = None


def _worker_init(ref_path: str) -> None:
    global _WORKER_REF
    _WORKER_REF = np.load(ref_path)


def _worker_rows(x_rows: np.ndarray) -> np.ndarray:
    return _geometric_d2_rows(x_rows, _WORKER_REF)


def geometric_d2_matrix(
    x_test: np.ndarray,
    x_ref: np.ndarray,
    n_particles: int,
    spatial_dim: int,
    n_workers: int = 1,
    chunk_rows: int = 16,
) -> np.ndarray:
    """(n, m) matrix of squared aligned distances, test × reference.

    Runs in-process when `n_workers == 1`; otherwise over a spawn-context
    process pool. Spawn (not fork) because the caller usually has JAX loaded,
    and forking a multithreaded JAX process can deadlock. Only call the
    multi-worker path from a process whose `__main__` is safe to re-import —
    `geometric_w2` guarantees that by running it in a fresh subprocess.
    """
    x_test = np.asarray(x_test, dtype=np.float64).reshape(-1, n_particles, spatial_dim)
    x_ref = np.asarray(x_ref, dtype=np.float64).reshape(-1, n_particles, spatial_dim)
    if n_workers <= 1:
        return _geometric_d2_rows(x_test, x_ref)

    chunks = [x_test[i:i + chunk_rows] for i in range(0, x_test.shape[0], chunk_rows)]
    with tempfile.TemporaryDirectory() as tmp:
        ref_path = os.path.join(tmp, "ref.npy")
        np.save(ref_path, x_ref)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=get_context("spawn"),
            initializer=_worker_init,
            initargs=(ref_path,),
        ) as pool:
            rows = list(pool.map(_worker_rows, chunks))
    return np.concatenate(rows, axis=0)


def _emd_w2(d2: np.ndarray) -> float:
    """sqrt of the exact optimal-transport cost under uniform marginals."""
    import ot

    n, m = d2.shape
    cost = ot.emd2(ot.unif(n), ot.unif(m), np.ascontiguousarray(d2), numItermax=100_000_000)
    return float(np.sqrt(cost))


def _geometric_w2_local(x_test, x_ref, n_particles, spatial_dim, n_workers) -> float:
    d2 = geometric_d2_matrix(x_test, x_ref, n_particles, spatial_dim, n_workers)
    return _emd_w2(d2)


def geometric_w2(
    x_test,
    x_ref,
    n_particles: int,
    spatial_dim: int,
    max_n: Optional[int] = None,
    seed: int = 0,
    n_workers: Optional[int] = None,
) -> float:
    """Geometric-W2 between two sets of particle configurations.

    Exact optimal transport (POT `emd2`, uniform marginals, outer square root)
    on the ground cost `d²` of `_geometric_d2_rows`. Both inputs are flat
    (n, n_particles * spatial_dim) or already (n, n_particles, spatial_dim).
    Non-finite configurations are dropped first.

    The cost is n × m assignments: 10k × 10k takes minutes on a many-core CPU,
    which is why it runs in parallel. `max_n` subsamples each side to at most
    that many configurations, without replacement, with a fixed `seed`. A value
    computed at one subsample size is not comparable with one at another.

    The one-shot alignment is an upper bound on the true ground distance and is
    asymmetric (the reference is the side being permuted), so the result is not
    strictly a Wasserstein distance; it matches the definition the reported
    numbers use.
    """
    rng = np.random.default_rng(seed)
    sets = []
    for x in (x_test, x_ref):
        x = np.asarray(x, dtype=np.float64).reshape(-1, n_particles * spatial_dim)
        x = x[np.isfinite(x).all(axis=1)]
        if max_n is not None and x.shape[0] > max_n:
            x = x[rng.choice(x.shape[0], max_n, replace=False)]
        sets.append(x)
    x_test, x_ref = sets
    if x_test.shape[0] == 0 or x_ref.shape[0] == 0:
        return float("nan")

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 1) - 1)
    if n_workers <= 1:
        return _geometric_w2_local(x_test, x_ref, n_particles, spatial_dim, 1)

    # The process pool must be created from a clean interpreter: with the spawn
    # start method every worker re-imports the parent's __main__, which in a
    # notebook-as-script run would re-execute the whole notebook.
    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.npz", Path(tmp) / "out.npy"
        np.savez(in_path, x_test=x_test, x_ref=x_ref)
        code = (
            "import sys, numpy as np\n"
            "from progressive_is_tempering.utils.metrics import _geometric_w2_local\n"
            "d = np.load(sys.argv[1])\n"
            "w = _geometric_w2_local(d['x_test'], d['x_ref'],"
            " int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))\n"
            "np.save(sys.argv[2], np.array(w))\n"
        )
        env = {**os.environ, "JAX_PLATFORMS": "cpu"}   # workers never touch the GPU
        subprocess.run(
            [sys.executable, "-c", code, str(in_path), str(out_path),
             str(n_particles), str(spatial_dim), str(n_workers)],
            check=True, env=env,
        )
        return float(np.load(out_path))


# ---------------------------------------------------------------------------
# Alanine dipeptide: Rama-KL, Energy-W1 / W2 (exact), T-W2
# ---------------------------------------------------------------------------

def ramachandran_kl(angles_ref, angles, bins: int = 100, eps: float = 1e-10) -> float:
    """Rama-KL: KL(P_ref || P_sample) of 2-D (phi, psi) histograms.

    `angles_ref` (m, 2) and `angles` (n, 2) in radians. `bins` x `bins` cells
    on [-pi, pi)^2; `eps` is added to every cell count before normalising.
    PITA reports this metric without releasing its definition; this is ours,
    so its absolute value depends on `bins`, `eps` and the sample sizes:
    compare it against the same statistic between two reference sets (the MD
    floor), not against published numbers.
    """
    rng = [[-np.pi, np.pi], [-np.pi, np.pi]]
    angles_ref, angles = np.asarray(angles_ref), np.asarray(angles)
    p = np.histogram2d(angles_ref[:, 0], angles_ref[:, 1], bins=bins, range=rng)[0] + eps
    q = np.histogram2d(angles[:, 0], angles[:, 1], bins=bins, range=rng)[0] + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def wasserstein_1d_exact(ref, test, p: int) -> float:
    """Exact 1-D W_p (p = 1 or 2) between two empirical distributions, by POT's
    `emd2_1d` (sorting), uniform weights, any sample sizes. PITA's Energy-W1
    and Energy-W2 (W2 = square root of the squared-cost transport)."""
    import ot

    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    test = np.asarray(test, dtype=np.float64).reshape(-1)
    if p == 1:
        return float(ot.emd2_1d(ref, test, metric="euclidean"))
    if p == 2:
        return float(np.sqrt(ot.emd2_1d(ref, test)))
    raise ValueError(f"p must be 1 or 2 (got {p!r})")


def torus_w2(angles, angles_ref) -> float:
    """T-W2: exact 2-Wasserstein between (phi, psi) samples on the torus.

    Ground cost: sum over the two angles of the squared wrapped difference
    `min(|d|, 2 pi - |d|)^2`; exact optimal transport (POT `emd2`, uniform
    marginals), outer square root. Memory is n x m float64: 0.8 GB at 10k x 10k.
    """
    import ot

    a, b = np.asarray(angles, dtype=np.float64), np.asarray(angles_ref, dtype=np.float64)
    d = np.abs(a[:, None] - b[None])
    return float(np.sqrt(ot.emd2(ot.unif(len(a)), ot.unif(len(b)),
                                 (np.minimum(d, 2 * np.pi - d) ** 2).sum(-1),
                                 numItermax=int(1e8))))
