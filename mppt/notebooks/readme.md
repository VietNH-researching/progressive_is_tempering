# Example Notebooks

End-to-end examples showing how to sample from challenging distributions using MPPT.
Each notebook follows the same pattern: define the target, run SMC to discover a
temperature schedule, then run parallel tempering for further mixing.

| Notebook | Target | Dimension | Description |
|----------|--------|-----------|-------------|
| [Gaussian_sanity_check.ipynb](Gaussian_sanity_check.ipynb) | Two-component Gaussian mixture | 50 | $0.7\,\mathcal{N}(5\cdot\mathbf{1}, \mathrm{diag}(1..d)) + 0.3\,\mathcal{N}(-5\cdot\mathbf{1}, P\,\mathrm{diag}(1..d)\,P^\top)$, $P$ random orthogonal. Well-separated modes; closed-form truth ($\log Z = 0$, moments, mixture weights). |
| [DW4.ipynb](DW4.ipynb) | Double-well (4 particles, 2D) | 6 | Quartic double-well pair potential with COM=0 constraint. Bimodal pairwise distances create combinatorial multimodality. |
| [LJ13.ipynb](LJ13.ipynb) | Lennard-Jones 13 (3D) | 36 | LJ 12-6 pair potential with COM=0 constraint. Hundreds of local minima separated by energy barriers. |
| [LJ55.ipynb](LJ55.ipynb) | Lennard-Jones 55 (3D) | 162 | LJ 12-6 pair potential with COM=0 constraint. Vast number of local minima; significantly harder than LJ13. |

## Prerequisites

```bash
pip install -e git+https://github.com/alexxthiery/jax-pdf.git#egg=jax-pdf
```

Both notebooks assume MPPT is accessible via `sys.path` (they add the parent directory automatically).

## Jupytext workflow

Each notebook exists as a pair: a `.py` file (jupytext percent format) and a `.ipynb` file.
The `.py` file is the source of truth for code; the `.ipynb` holds outputs and is used for
interactive exploration.

After editing a `.py` file, sync the `.ipynb`:

```bash
jupytext --sync <notebook>.ipynb
```

After running cells in the `.ipynb`, sync back:

```bash
jupytext --sync <notebook>.ipynb
```

`--sync` updates whichever file is older from the newer one.

**Agents**: always edit the `.py` file, never the `.ipynb` directly.

## API reference

See [docs/api.md](../docs/api.md) for the full API: engines, explorers, collectors, and all parameters.
