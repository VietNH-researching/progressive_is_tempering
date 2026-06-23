# MPPT User Guide and API Reference

## How the algorithm works

MPPT implements **multi-particle parallel tempering**: a variant of parallel
tempering that maintains $N$ particles at each temperature level instead of one.

### Background: standard parallel tempering

Standard parallel tempering runs one MCMC chain at each of $T$ temperatures
$0 = t_0 < t_1 < \cdots < t_{T-1} = 1$.
At temperature $t$, the chain targets an interpolated distribution:

$$
\pi_t(x) \propto \pi_{\text{ref}}(x)^{1-t} \, \pi_{\text{target}}(x)^t
$$

At $t=0$ the chain samples from an easy reference distribution.
At $t=1$ it samples from the target.
Intermediate temperatures create a bridge so that particles can gradually
move from reference-like regions to target-like regions.

Periodically, the algorithm proposes to **swap** particles between adjacent
temperatures. A particle at temperature $t_k$ with high affinity for the
target tends to migrate toward $t=1$, while one with low affinity migrates
toward $t=0$. This accelerates mixing across energy barriers that a single
chain would struggle to cross.

### What MPPT adds: multiple particles per temperature

Instead of one chain per temperature, MPPT maintains $N$ particles at each temperature. With $N$ particles, we can split them into two halves. One half provides statistics for tuning the MCMC kernel (step size, proposal covariance); the other half gets moved using those tuned parameters. Then the roles swap. This avoids the feedback loop where adapting based on the same samples you are moving would bias the chain.

### One iteration of the algorithm

Each PT iteration has three phases:

**1. Explore** -- at each temperature independently, run MCMC to move particles.

The $N$ particles at each temperature are split into two halves $F$ and $S$.
First, MCMC parameters (step size, covariance) are estimated from $F$,
and those parameters are used to run MCMC on $S$.
Then the roles reverse: parameters are estimated from the updated $S$,
and MCMC runs on $F$.
Every particle gets moved once, and the adaptation never depends on the
particles being moved.

**2. Swap** -- exchange particles between adjacent temperatures.

Swaps happen in alternating even/odd sweeps (like a bubble-sort pass):
even sweeps propose swaps at pairs $(t_0, t_1), (t_2, t_3), \ldots$;
odd sweeps at $(t_1, t_2), (t_3, t_4), \ldots$.
This pattern is repeated `n_outer_swaps` times (default: $T$).

Within each pair of adjacent temperatures $(t_k, t_{k+1})$, the $N$
particles from one side are randomly matched with the $N$ particles from
the other side (when `mix=True`), and each matched pair undergoes a
Metropolis accept/reject swap decision. This random matching and swap
attempt is repeated `n_inner_swaps` times (default: 3), each time with a
fresh random permutation, so that different particles get paired together
across rounds.

**3. Reset** -- after swaps, particles may have moved to new temperatures.
Their cached values (log-densities, gradients) are recomputed for the
new temperature assignment.


## Defining a problem

A problem is defined by three functions bundled into a `PTProblem`:

```python
from mppt import PTProblem

problem = PTProblem(
    rvs_ref=rvs_ref,                   # key -> sample from reference
    logpdf_ref=logpdf_ref,              # x -> log-density of reference
    logpdf_final_target=logpdf_target,  # x -> log-density of target (unnormalized ok)
)
```

| Field | Type | Description |
|-------|------|-------------|
| `rvs_ref` | `Key -> Array` | Draw one sample from the reference distribution |
| `logpdf_ref` | `Array -> float` | Log-density of the reference (must be normalized) |
| `logpdf_final_target` | `Array -> float` | Log-density of the target (can be unnormalized) |

All three functions must operate on vectors of the same dimension $d$
and be compatible with JAX transformations (`jit`, `grad`, `vmap`).


## Entry points

### `parallel_tempering`

Runs the full algorithm with a fixed temperature schedule.

```python
from mppt import parallel_tempering

final_state, intermediate_stats, final_stats = parallel_tempering(
    key, problem, temperatures, n_iter,
    n_particles=1000,
    engine={'name': 'hmc', 'refresh_rate': 0.07},
    explorer={'mode': 'adaptive', 'niter': 50},
    swapper={'mix': True},
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | JAX key | required | Random seed |
| `problem` | `PTProblem` | required | Problem definition |
| `temperatures` | array or list | required | Schedule from 0 to 1, e.g. `[0, 0.1, 0.3, 0.6, 1.0]` |
| `n_iter` | int | required | Number of PT iterations (each = explore + swap) |
| `init_state` | `PTState` or array | `None` | Starting particles. If array, shape `(T, N, d)`. If `None`, draws from reference. |
| `n_particles` | int | `None` | Particles per temperature. Ignored if `init_state` given. |
| `engine` | dict | `{'name': 'random_walk', 'maxniter': 100}` | MCMC kernel (see [Engines](#engines)) |
| `explorer` | dict | `{'mode': 'adaptive'}` | Exploration mode (see [Explorers](#explorers)) |
| `collectors` | list | `default_collectors` | Statistics to collect (see [Collectors](#collectors)) |
| `n_inner_swaps` | int | 3 | How many times to re-pair and attempt swaps within each adjacent temperature pair (see [algorithm description](#one-iteration-of-the-algorithm)) |
| `n_outer_swaps` | int | `len(temperatures)` | How many even/odd sweep passes to perform (see [algorithm description](#one-iteration-of-the-algorithm)) |
| `verbose` | bool | `True` | Print progress bar and timing |
| `runtime_check` | bool | `False` | Enable chex debug assertions (slow) |
| `swapper` | dict | `{'mix': True}` | Swapper config (see [Swappers](#swappers)) |

**Returns** `(final_state, intermediate_stats, final_stats)`:

- `final_state`: algorithm state after the last iteration
- `intermediate_stats`: dict of arrays, one entry per collector, stacked over iterations 0 to n_iter-2
- `final_stats`: dict from the last iteration, includes all collectors


### `pt_manual_runners`

Pre-compiles a one-step function for manual iteration.
Useful when you want to save checkpoints between iterations or implement
custom stopping criteria.

```python
from mppt import pt_manual_runners

pt_runner = pt_manual_runners(
    problem, temperatures,
    n_particles=1000,
    engine={'name': 'hmc', 'refresh_rate': 0.07},
    explorer={'mode': 'adaptive', 'niter': 50},
    swapper={'mix': True},
)

# Iterate manually
for i in range(n_iters):
    key, subkey = random.split(key)
    pt_state, diag = pt_runner(subkey, pt_state, full_output=False)
```

Parameters are the same as `parallel_tempering` except there is no `n_iter`,
`verbose`, or `runtime_check` (you control the loop yourself).

The returned `pt_runner(key, prev, full_output)` callable accepts:

- `prev`: either a `PTState` from a previous iteration, or a `(T, N, d)` array
  (e.g., from SMC output). On the first call with an array, it automatically
  initializes the internal state.
- `full_output`: `False` collects only per-iteration statistics (logZ, acceptance
  rates, etc.); `True` also collects heavier statistics like adapted MCMC parameters.

All internal functions are JIT-compiled when `pt_manual_runners` is called.
This takes 10--60s depending on problem size, but subsequent calls run with
zero compilation overhead.

**Important**: `n_particles` must match the actual particle count you pass at
runtime. A mismatch raises a clear `ValueError`.


### `smc_sampler`

SMC (Sequential Monte Carlo) sampler that anneals from reference to target
with an **adaptive** temperature schedule. At each step it:

1. Finds the next temperature by bisection so that the effective sample
   size (ESS) drops to a target ratio
2. Resamples particles according to importance weights
3. Runs MCMC to rejuvenate the resampled particles

This is useful as a **first pass** to discover a good temperature schedule
and initialize particles before running parallel tempering.

```python
from mppt import smc_sampler

particles, logZ, temperatures = smc_sampler(
    key, problem,
    n_particles=10_000,
    engine={'name': 'hmc', 'refresh_rate': 0.07},
    n_mcmc_steps=10,
    local_ess_ratio=0.5,
    verbose=True,
    runtime_check=False,
    keep_all_states=True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | JAX key | required | Random seed |
| `problem` | `PTProblem` | required | Problem definition |
| `n_particles` | int | required | Number of particles |
| `engine` | dict | required | MCMC kernel config |
| `n_mcmc_steps` | int | required | MCMC steps per SMC iteration |
| `local_ess_ratio` | float | required | Target ESS ratio for choosing the next temperature. Lower values (e.g., 0.3) take bigger temperature jumps but lose more particles to resampling. Higher values (e.g., 0.9) take smaller jumps but preserve more diversity. 0.5 is a reasonable default. |
| `verbose` | bool | required | Print iteration info |
| `runtime_check` | bool | required | Enable chex checks |
| `keep_all_states` | bool | `False` | If `True`, return stacked particle positions `(n_steps+1, N, d)` across all temperature steps. Useful for initializing PT. |

**Returns** `(particles, logZ_dict, temperatures_list)`:

- `particles`: final particle state, or stacked array if `keep_all_states=True`
- `logZ_dict`: dict with keys `'SMC_like'`, `'reverse_logistic'`, `'robust'`.
  Values are the total logZ estimate $\log(Z_{\text{target}} / Z_{\text{ref}})$.
- `temperatures_list`: list of discovered temperatures from 0 to 1

When `verbose=True`, each iteration prints logZ as a tuple
`(SMC_like, reverse_logistic, robust)`.

**Typical workflow**: run SMC to discover temperatures, then feed the output
into parallel tempering:

```python
smc_output, logZ, temperatures = smc_sampler(key, problem, ..., keep_all_states=True)
pt_runner = pt_manual_runners(problem, temperatures, n_particles=n_particles, ...)
pt_state, diag = pt_runner(key, smc_output, full_output=False)
```


## Engines

The `engine` parameter selects the MCMC kernel used to move particles at each
temperature. Pass a dict with `'name'` and any additional parameters.

### Step size tuning

All engines need a step size (or proposal scale) that balances exploration
speed against acceptance rate. Too large and most moves get rejected; too
small and the chain barely moves.

For HMC and MALA, the step size is always tuned automatically via **bisection**:
at each PT iteration, the algorithm performs a binary search over step sizes,
running a trial batch of MCMC moves at each candidate to measure the acceptance
rate, and converges on a step size that falls within the target acceptance range.

For random walk, you can choose between bisection tuning and a simple
analytical rule ($2.38/\sqrt{d}$, which is optimal for Gaussian targets).

Each engine targets a different acceptance range:

| Engine | Target acceptance rate |
|--------|----------------------|
| HMC | 0.80 -- 0.95 |
| MALA | 0.40 -- 0.65 |
| Random Walk (bisection) | 0.20 -- 0.35 |

### HMC (recommended)

Hamiltonian Monte Carlo. Uses gradients of the log-density to make large,
efficient moves. The recommended choice for most problems.

```python
engine = {'name': 'hmc', 'refresh_rate': 0.07}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `refresh_rate` | 0.07 | Controls how often the momentum (velocity) variable is resampled. At each MCMC step, the velocity is partially refreshed: a fraction `refresh_rate` is replaced with fresh noise. The velocity is roughly fully randomized after `1/refresh_rate` steps (~14 at the default). Lower values produce longer, more correlated trajectories that can traverse the space more efficiently. |

### MALA

Metropolis-Adjusted Langevin Algorithm. A special case of HMC where the
velocity is fully refreshed at every step (`refresh_rate=1`).
Takes smaller, more cautious steps than HMC.

```python
engine = {'name': 'mala'}
```

No configurable parameters.

### Random Walk

Random walk Metropolis-Hastings. Proposes new positions by adding Gaussian
noise to the current position. Does not use gradients, so it works for
non-differentiable targets but scales poorly to high dimensions.

```python
engine = {'name': 'random_walk'}
engine = {'name': 'random_walk', 'use_bisection': True}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `maxniter` | 100 | Upper bound on MCMC steps per exploration when using adaptive step count |
| `use_bisection` | `False` | How to set the proposal scale. `True`: tune by bisection (see above). `False`: use the analytical $2.38/\sqrt{d}$ rule, which is optimal for Gaussian targets and a reasonable starting point otherwise. |
| `identity_kernel` | `False` | Skip MCMC moves entirely (particles stay fixed). A debugging tool for testing swap mechanics in isolation. |

### Choosing an engine

| Situation | Recommendation |
|-----------|---------------|
| Target is differentiable | HMC (default) |
| HMC has numerical issues (NaN gradients, very stiff dynamics) | MALA |
| Target is not differentiable | Random Walk |
| Low dimension ($d < 10$) | Any engine works; Random Walk is simplest |
| High dimension ($d > 20$) | HMC strongly recommended |


## Explorers

The `explorer` parameter controls how MCMC parameters are tuned and how many
MCMC steps are taken at each PT iteration.

### Adaptive (default)

Automatically tunes the MCMC step size at each iteration.

```python
explorer = {'mode': 'adaptive', 'niter': 50}  # adapt step size, fixed 50 MCMC steps
explorer = {'mode': 'adaptive'}                # adapt both step size and step count
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `niter` | `None` | Number of MCMC steps per exploration. If given, use this fixed count. If omitted, the algorithm also adapts the step count: it runs MCMC until a statistical test determines that particles have moved far enough from their starting positions. |

The step size adaptation uses the two-half splitting described in the
[algorithm overview](#one-iteration-of-the-algorithm). At each temperature,
bisection search finds a step size that achieves the target acceptance rate
for the chosen engine.

### Non-adaptive

Uses fixed, pre-specified MCMC parameters. No tuning overhead, but you need
to supply parameters (typically obtained from a previous adaptive run).

```python
explorer = {'mode': 'nonadaptive', 'niter': 100, 'mcmc_params': params}
```

Both `niter` and `mcmc_params` are required.


## Swappers

The `swapper` parameter controls how particles are paired when proposing
swaps between adjacent temperatures.

```python
swapper = {'mix': True}   # recommended
swapper = {'mix': False}  # deterministic pairing (for debugging)
```

| Parameter | Description |
|-----------|-------------|
| `mix` | `True` (recommended): before each round of swap proposals, randomly shuffle the particles at each temperature so that different particles get paired each time. This is important when $N$ is large, as it gives each particle a chance to find a good swap partner across multiple rounds. `False`: always pair particles in index order (particle 0 with particle 0, etc.). Faster, but can leave some particles stuck. |

See the [algorithm overview](#one-iteration-of-the-algorithm) for how
`n_inner_swaps` and `n_outer_swaps` control the swap phase.


## Collectors

Collectors compute statistics at each PT iteration. They are specified as a
list of dicts passed to the `collectors` parameter.

### Default collectors

The default set (used when you don't specify `collectors`) collects:

| Collector | Key in diagnostics | Shape | When |
|-----------|--------------------|-------|------|
| logZ (SMC_like) | `logZ_SMC_like` | scalar | every iteration |
| logZ (robust) | `logZ_robust` | scalar | every iteration |
| logZ (reverse_logistic) | `logZ_reverse_logistic` | scalar | every iteration |
| MCMC acceptance rates | `mcmc_acc_rates` | `(T,)` | every iteration |
| MCMC step count | `mcmc_niter` | scalar | every iteration |
| MCMC parameters | `mcmc_parameters` | varies | last iteration only |
| swap rejection rates | `swap_rejection_rates` | `(T-1,)` | every iteration |

### Available collectors

**`logZ`** -- estimates $\log(Z_{\text{target}} / Z_{\text{ref}})$, the log
normalizing constant ratio.

```python
{'name': 'logZ', 'estimator': 'SMC_like'}
{'name': 'logZ', 'estimator': 'robust'}
{'name': 'logZ', 'estimator': 'reverse_logistic'}
```

Three estimators:
- **SMC_like**: importance sampling estimator. Fast and simple.
- **robust**: bidirectional stepping-stone estimator. Uses particles from both
  sides of each temperature gap, guaranteeing finite variance.
- **reverse_logistic**: Geyer's (1994) reverse logistic regression. Solves an
  optimization problem (BFGS) to jointly fit all log-partition functions.
  Most stable when there are many temperatures, but uses $O(T^2 N)$ memory.

**`mcmc_acc_rates`** -- mean MCMC acceptance rate at each temperature.
Shape `(T,)`. Temperature 0 is always 1.0 (particles are drawn iid from the reference).

```python
{'name': 'mcmc_acc_rates'}
```

**`niter`** -- number of MCMC steps taken during exploration (scalar; the
maximum across temperatures when using adaptive step count).

```python
{'name': 'niter'}
```

**`mcmc_parameters`** -- the adapted MCMC parameters (proposal covariance,
step size, etc.). Can be large; set `at_every_iter=False` to collect only at
the last iteration.

```python
{'name': 'mcmc_parameters', 'at_every_iter': False}
```

**`swap_rejection_rates`** -- fraction of rejected swaps between each pair of
adjacent temperatures. Shape `(T-1,)`.

```python
{'name': 'swap_rejection_rates'}
```

**`index_matrix`** -- the full particle-to-temperature assignment matrix after
swaps. Shape `(T, N)`. Disabled by default because it uses a lot of memory.

```python
{'name': 'index_matrix'}
```


## Extracting samples

After running PT, extract particles at the target temperature ($t=1$):

```python
samples = pt_state.X.position[pt_state.Z[-1]]  # shape (N, d)
```

The state stores all $T \times N$ particles in a flat array (`pt_state.X`).
The index matrix `pt_state.Z` has shape `(T, N)`: row $k$ contains the indices
of particles currently assigned to temperature $k$.
So `Z[-1]` picks out the $N$ particles at the highest temperature (the target).


## Interpreting diagnostics

**logZ estimates**: the three estimators should roughly agree if the chain has
converged. Large discrepancies suggest insufficient mixing or too few particles.

**Swap rejection rates**: one value per adjacent temperature pair. High rates
(say > 0.7) mean the temperature gap is too large for particles to cross
efficiently. Consider adding more intermediate temperatures.

**MCMC acceptance rates**: one value per temperature. The expected range
depends on the engine: 0.8--0.95 for HMC, 0.4--0.65 for MALA, 0.2--0.35 for
random walk (with bisection). Temperature 0 always shows 1.0 because particles
there are drawn iid from the reference rather than moved by MCMC.
