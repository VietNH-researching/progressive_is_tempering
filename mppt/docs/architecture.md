# Architecture

This document explains the design decisions behind MPPT.
For the structural layout, see `docs/codemap.md`.
For the extend pattern specifically, see `docs/extend.md`.

## Problem Setting

Sample from an unnormalized target $\pi_1$ given access to a tractable reference $\pi_0$.
Parallel tempering bridges between them via geometric annealing:

$$
\pi_t(x) \propto \pi_0(x)^{1-t} \, \pi_1(x)^t, \quad t \in [0,1]
$$

The algorithm maintains $N$ particles at each of $K$ temperature levels $0 = t_0 < t_1 < \cdots < t_{K-1} = 1$.
At each iteration: explore (MCMC at each temperature), then swap particles between adjacent temperatures.

## Three-Layer Architecture

```
abstract/   Protocols + algorithm logic    (what the algorithm does)
impl/       Concrete kernels               (how each piece works)
ui/         User API + config parsing      (how users interact)
```

**Why this split?**
The PT algorithm is the same regardless of MCMC kernel.
Separating `abstract/` from `impl/` means adding a new kernel (e.g., NUTS, slice sampling) requires only new files in `impl/` and a registration in `ui/engine_ui.py`.
No changes to `abstract/pt_exec.py`.

The `ui/` layer decouples configuration parsing (dicts, defaults) from algorithm logic.
This keeps `abstract/` clean of string-keyed dispatch.

## Index-Based Swaps (PTState)

```python
class PTState(NamedTuple):
    X: PyTree   # flat array of K*N particles
    Z: Array    # (K, N) index matrix
```

Swap operations modify only `Z`, never `X`.
To get particles at temperature $k$: `pytree_index(state.X, state.Z[k])`.

**Why?**
Swapping by copying particle data is $O(N \cdot d)$ per swap where $d$ is the particle dimension.
Index swaps are $O(N)$ and avoid moving large arrays in device memory.
With 500K particles in 10+ dimensions, this matters.

**Trade-off**: after swaps, `X` and `Z` are out of sync.
The exploration step needs particles grouped by temperature, so `make_contiguous()` must be called before exploration.
This reindexes `X` to match `Z`, then resets `Z` to the identity.
The cost is paid once per iteration rather than once per swap.

## Two-Half Exploration

The exploration step splits particles at each temperature into two halves:

1. Adapt MCMC parameters (covariance, step size) using half A
2. Run MCMC on half B with those parameters
3. Adapt using the now-updated half B
4. Run MCMC on half A with those parameters

**Why not adapt and move the same particles?**
Adapting covariance from the same particles you then move creates a feedback loop.
The MCMC transition is no longer $\pi_t$-invariant because the proposal depends on the current sample.
Splitting avoids this bias while still using the full population for both adaptation and exploration.

**Why not a held-out set?**
Splitting 50/50 and swapping roles means every particle gets moved and every particle contributes to adaptation.
A fixed held-out set would waste half the particles.

## The Extend Pattern

Core idea: bundle a position $x$ with its computed attributes (logpdf, gradient, temperature) so MCMC kernels never call the target density directly.

For random walk, there is one extend function:

```python
def extend(x) -> Particle:
    return Particle(position=x, logpdf_target=target(x), ...)
```

For HMC, the pattern splits into three:

```
feed_extend(x)       -> position + logpdf + gradient (no velocity)
init_extend(key, x)  -> feed_extend(x) + random velocity
inner_extend(x, v)   -> feed_extend(x) + given velocity
```

**Why?**
The PT wrapper adds temperature, `logpdf_ref`, `logpdf_final_target` to the particle.
By wrapping `feed_extend`, the HMC kernel carries these fields automatically without knowing about them.
Temperature reset becomes cheap: just recompute `logpdf_target = (1-t)*logpdf_ref + t*logpdf_final_target` using cached ref/target values.
No need to re-evaluate the actual density functions.

See `docs/extend.md` for the full explanation with examples.

## WhileLoop as Data

Adapting the number of MCMC steps is temperature-dependent.
Cold temperatures mix fast; hot temperatures need more steps.
Naively running each temperature's adaptation loop sequentially creates a bottleneck at the slowest temperature.

Solution: represent each adaptation as a `WhileLoop` object (cond, body, init, extractor).
Then `while_loops_executor` runs all $K$ loops in parallel (via vmap) until a configurable proportion (default 50%) have converged.
The maximum iteration count among finished loops is used for all temperatures.

**Why 50%?**
One stuck temperature shouldn't hold up the whole system.
The 50% threshold is a heuristic balancing adaptation quality against compute waste.
Configurable via the `completion_threshold` parameter.

## Swap Mechanism

Swaps alternate between even pairs $(0,1), (2,3), \ldots$ and odd pairs $(1,2), (3,4), \ldots$.
This is repeated `n_outer_swaps` times (default: $K$).

Within each pair, `n_inner_swaps` (default: 3) pairwise Metropolis swaps are applied.
The `mix=True` option randomly permutes the "right" set of particles before pairing, so different particles get matched across swap attempts.

**Acceptance criterion** for swapping particles $x_i$ at temperature $t_k$ and $x_j$ at temperature $t_{k+1}$:

$$
\alpha = \frac{\pi_{t_k}(x_j) \, \pi_{t_{k+1}}(x_i)}{\pi_{t_k}(x_i) \, \pi_{t_{k+1}}(x_j)}
$$

In log space with geometric annealing, this reduces to:

$$
\log \alpha = (t_{k+1} - t_k) \bigl[\ell_1(x_i) - \ell_0(x_i) - \ell_1(x_j) + \ell_0(x_j)\bigr]
$$

where $\ell_0 = \log \pi_0$ and $\ell_1 = \log \pi_1$.

## LogZ Estimation

Three estimators are provided, all operating on the $(K, N)$ array of particles:

- **SMC_like**: importance sampling with the stepping-stone identity. Standard but can have high variance.
- **robust**: symmetric forward/backward estimator with guaranteed finite variance. No tail conditions needed.
- **reverse_logistic**: Geyer's method, minimizes a convex objective via BFGS. Most stable for many temperatures.

All three are computed by default to allow cross-checking.

## SMC Sampler (BETA)

An alternative to fixed-temperature PT: start at $t=0$ and adaptively increase temperature using ESS-based bisection.
At each step: find the next temperature that drops ESS to a target ratio (default 0.5), resample, then run MCMC.

Currently less tested than the PT path.
The SMC output can be fed into `pt_manual_runners` as initialization for subsequent PT iterations.

## JAX Patterns

- **vmap** everywhere: over particles (batch_move), over temperatures (exploration, swap), over independent runs
- **lax.scan**: main PT loop (via `verbose_scan` for progress bars)
- **lax.fori_loop**: repeated swaps, repeated MCMC steps
- **lax.cond**: temperature-dependent branching (e.g., identity kernel at $t=0$)
- **lax.while_loop**: adaptive nsteps, bisection
- **donate_argnums**: memory reuse for the state across iterations
- **No pmap**: single-device only; multi-device parallelism is not implemented

## Configuration Flow

User-facing functions accept plain dicts:

```python
engine={'name': 'hmc', 'refresh_rate': 0.07}
explorer={'mode': 'adaptive'}
swapper={'mix': True}
collectors=[{'name': 'logZ', 'estimator': 'robust'}, ...]
```

These are parsed by `ui/engine_ui.py`, `ui/explorers.py`, `ui/swappers_api.py`, `ui/collectors.py` into callables or protocol-satisfying objects.
The `FrozenDict`/`FrozenList` wrappers prevent accidental mutation of default arguments.
