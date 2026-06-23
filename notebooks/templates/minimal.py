# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Minimal controlled_sampler notebook

# %%
import jax
import jax.numpy as jnp

from controlled_sampler.targets import make_double_well_target


# %%
seed = 0
key = jax.random.PRNGKey(seed)

target = make_double_well_target()
x = jnp.zeros(target.event_shape)

log_prob = target.log_prob(x)
grad_log_prob = target.grad_log_prob(x)

log_prob, grad_log_prob