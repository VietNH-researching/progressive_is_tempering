# Extend functions
## Review: NamedTuple
Named tuples consistute the central data type in this package.
Here is an example of a simple NamedTuple object:

```python
from typing import NamedTuple
class Point2D(NamedTuple):
    x: float
    y: float

my_point = Point2D(5, 6)
my_point_new = my_point._replace(y=7)  # create a new point because named tuples are immutable
print(my_point_new)  # (5,7)
print(my_point._asdict())  # {'x': 5, 'y': 6}
```

Note that `Protocol` is not the same as `NamedTuple`.
The former is an abstract object to define classes having a certain number of attributes, whereas the latter is a concrete implementation.
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Have_Y_Attribute(Protocol):
    y: float

class Point2D_Bis(NamedTuple):
    x: float
    z: float

my_point = Point2D(x=3, y=4)
my_point_bis = Point2D_Bis(x=3, z=4)
assert isinstance(my_point, Have_Y_Attribute)
assert not isinstance(my_point_bis, Have_Y_Attribute)
```

## Concept
The `extend` functions are ubiquitous in MCMC. 
Everyone who has implemented the random walk Metropolis-Hastings algorithm knows that it is advantageous to bundle the current position with the logpdf of the target:

```python
def extend(x):
    return State(position=x, target_logpdf=target_logpdf(x))
```

More importantly the move step can be implemented using `extend` only, **without ever** explicitly invoking the `target_logpdf` function.

```python
def RWMH_move(key, x: State) -> State:
    key_proposal, key_accept = random.split(key)
    x_star_position = x.position + eps * random.normal(key_proposal)
    x_star = extend(x_star_position)  # target_logpdf calculated implicitly
    if jnp.log(random.uniform(key_accept)) < x_star.target_logpdf - x.target_logpdf:
        return x_star
    else:
        return x
```

This means that we can have `extend` functions that add more attributes to the particles, including attributes that are not directly related to the random walk algorithm (e.g. temperature).
But interestingly we will still have these attributes carried along automatically in the random walk execution.
Consider the following `extend` function for a RWMH kernel at temperature $1/2$:

```python
def extend(x):
    return State(
        position=x,
        base_logpdf=base_logpdf(x), final_logpdf=final_logpdf(x),
        temperature=1/2,
        target_logpdf=0.5 * base_logpdf(x) + 0.5 * final_logpdf(x)
    )
```

This allows to change the temperature of `x` later without recalculating the logpdfs of the base and the final target. 

## Advanced case: Hamiltonian Monte Carlo
However in more complicated situations there are two `extend` functions, one for initialization and another for dealing with the internal state of the algorithm.
For example we have the following in Hamiltonian Monte Carlo:

```python
def init_extend(key, x):
    return State(position=x, velocity=jax.random.normal(key),
                 target_logpdf=target_logpdf(x),
                 grad_logpdf=grad_logpdf(x))

def internal_extend(x, v):
    return State(position=x, velocity=v,
                 target_logpdf=target_logpdf(x),
                 grad_logpdf=grad_logpdf(x))
```

A cleaner way to define these two functions is to use a third feeder function `feed_extend` and infer `init_extend` and `internal_extend` based on it. This gives

```python
def extend_functions(feed_extend):
    def init_extend(key, x):
        return HMCState(velocity=jax.random.normal(key),
                        **feed_extend(x)._asdict())
    def internal_extend(x, v):
        return HMCState(velocity=v, 
                        **feed_extend(x)._asdict())
    return init_extend, internal_extend

def feed_extend(x):
    return State(position=x, target_logpdf=target_logpdf(x),
                 grad_logpdf=grad_logpdf(x),
                 another_attribute=another_attribute(x))

init_extend, internal_extend = extend_functions(feed_extend)
```

Again, the advantage of this approach is that `feed_extend` can add to the particles the attributes that are unknown to the programmer but are still useful in some way.
For instace the `another_attribute` above can be the temperature of the particle.

# This package
In this package the implementation of HMC follows the above philosophy.

For random walk Metropolis, the three functions coincide. (The function `init_extend` does not use its `key` argument and is effectively equivalent to `extend` and `feed_extend`.)
As such we allow some abusive calls, e.g. `feed_extend` or `extend` is directly invoked instead of properly calling `init_extend` with a dummy key.