# Example Notebooks

End-to-end examples showing how to sample from challenging distributions using MPPT.
Each notebook follows the same pattern: define the target, run SMC to discover a
temperature schedule, then run parallel tempering for further mixing.

| Notebook | Target | Dimension | Description |
|----------|--------|-----------|-------------|
| [DW4.ipynb](DW4.ipynb) | Double-well (4 particles, 2D) | 6 | Quartic double-well pair potential with COM=0 constraint. Bimodal pairwise distances create combinatorial multimodality. |
| [LJ13.ipynb](LJ13.ipynb) | Lennard-Jones 13 (3D) | 39 | LJ 12-6 pair potential with harmonic trap. Hundreds of local minima separated by energy barriers. |

## Prerequisites

```bash
pip install -e git+https://github.com/alexxthiery/jax-pdf.git#egg=jax-pdf
```

Both notebooks assume MPPT is accessible via `sys.path` (they add the parent directory automatically).

## API reference

See [docs/api.md](../docs/api.md) for the full API: engines, explorers, collectors, and all parameters.
