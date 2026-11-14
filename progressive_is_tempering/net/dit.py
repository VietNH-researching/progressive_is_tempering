"""PITA's DIT3D score network (Flax port): the ALDP notebook's network.

Source: PITA (Tan et al., *Progressive Inference-Time Annealing*), class DIT3D in
src/models/components/dit.py, with its ALDP configuration (configs/model/net/dit.yaml):
hidden_size 192, cond_dim 64, n_blocks 6, n_heads 6, dropout 0.1, mlp_ratio 4,
condition_on_temperature True, giving 3,179,267 parameters.

Reproduced as in PITA:
* one token per atom: Linear(3 → hidden) on the atom's coordinates;
* conditioning c = SiLU(TimestepEmbedder(t) + TimestepEmbedder(β)), each a
  256-dim [cos | sin] sinusoid (max_period 10 000) → Linear → SiLU → Linear;
* 6 DDiT blocks: bias-free LayerNorm, adaLN shift/scale/gate (zero-init),
  bias-free qkv / out projections, rotary position embedding over the atom
  index on q and k (torchtune's interleaved RoPE, base 10 000), softmax
  attention (no attention dropout: PITA's dropout_p is 0), GELU(tanh) MLP ×4,
  dropout 0.1 on the attention and MLP branches;
* final layer: LayerNorm, adaLN shift/scale, Linear(hidden → 3), all zero-init,
  so the network outputs exactly 0 at initialization;
* PyTorch default initialization elsewhere (U(±1/√fan_in) weights and biases).

Differences:
* float32 throughout (PITA runs the blocks under bfloat16 autocast);
* dropout is active only when `deterministic=False` (training); the backward
  SDE and evaluation run it deterministically, as PITA's eval mode does.

Interface: **batched** — `DIT3D()(c_noise (B,), x (B, 3·n_particles), β (B,))`
→ `(B, 3·n_particles)`. The output-scaling wrapper that turns it into a score
lives in the notebook (`notebooks/aldp/aldp.py`, Section 5).

Module and parameter names must stay as they are: they are the keys of the
trained parameter trees in `results/aldp/seedN/checkpoint.pkl`.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import linen as nn


def _torch_bias_init(fan_in):
    bound = 1.0 / math.sqrt(fan_in)
    return lambda key, shape, dtype=jnp.float32: jax.random.uniform(key, shape, dtype, -bound, bound)


# U(-1/sqrt(fan_in), 1/sqrt(fan_in)) = variance_scaling(1/3, fan_in, uniform)
_torch_kernel_init = nn.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform")


class TorchLinear(nn.Module):
    features: int
    use_bias: bool = True
    zero_init: bool = False

    @nn.compact
    def __call__(self, x):
        fan_in = x.shape[-1]
        if self.zero_init:
            k_init, b_init = nn.initializers.zeros, nn.initializers.zeros
        else:
            k_init, b_init = _torch_kernel_init, _torch_bias_init(fan_in)
        return nn.Dense(self.features, use_bias=self.use_bias, kernel_init=k_init, bias_init=b_init)(x)


def layer_norm(x, weight):
    """F.layer_norm without affine (eps 1e-5), then × weight (PITA's LayerNorm)."""
    mu = jnp.mean(x, -1, keepdims=True)
    var = jnp.mean((x - mu) ** 2, -1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * weight


class LayerNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        return layer_norm(x, self.param("weight", nn.initializers.ones, (x.shape[-1],)))


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class Dropout(nn.Module):
    """Inverted dropout like nn.Dropout, but drawing the mask in x's dtype.

    nn.Dropout draws its Bernoulli mask through the default float type, which
    is float64 when jax_enable_x64 is on (as the ALDP energy requires) — 12 float64 masks
    of size (B, 22, 192) per training step on GPUs with slow float64.
    """
    rate: float

    @nn.compact
    def __call__(self, x, deterministic: bool):
        if deterministic or self.rate == 0.0:
            return x
        keep = jax.random.uniform(self.make_rng("dropout"), x.shape, x.dtype) >= self.rate
        return jnp.where(keep, x / (1.0 - self.rate), jnp.zeros_like(x))


class TimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256
    max_period: float = 10000.0

    @nn.compact
    def __call__(self, t):  # t: (B,)
        half = self.frequency_embedding_size // 2
        freqs = jnp.exp(-math.log(self.max_period) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[:, None].astype(jnp.float32) * freqs[None]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], -1)
        h = nn.silu(TorchLinear(self.hidden_size)(emb))
        return TorchLinear(self.hidden_size)(h)


def rope(x, base=10000.0):
    """torchtune RotaryPositionalEmbeddings on x: (B, S, H, D); positions 0..S-1."""
    S, D = x.shape[1], x.shape[-1]
    theta = 1.0 / (base ** (jnp.arange(0, D, 2, dtype=jnp.float32)[: D // 2] / D))
    ang = jnp.arange(S, dtype=jnp.float32)[:, None] * theta[None]  # (S, D/2)
    cos, sin = jnp.cos(ang)[None, :, None, :], jnp.sin(ang)[None, :, None, :]
    xs = x.reshape(*x.shape[:-1], D // 2, 2)
    x0, x1 = xs[..., 0], xs[..., 1]
    out = jnp.stack([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1)
    return out.reshape(x.shape)


class DDiTBlock(nn.Module):
    dim: int
    n_heads: int
    mlp_ratio: int = 4
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, c, deterministic: bool):
        B, S, _ = x.shape
        H, Dh = self.n_heads, self.dim // self.n_heads
        mod = TorchLinear(6 * self.dim, zero_init=True)(c)[:, None]  # (B, 1, 6·dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(mod, 6, axis=-1)

        x_skip = x
        h = modulate(LayerNorm()(x), shift_msa, scale_msa)
        qkv = TorchLinear(3 * self.dim, use_bias=False)(h).reshape(B, S, 3, H, Dh)  # "b s (three h d)"
        q, k, v = rope(qkv[:, :, 0]), rope(qkv[:, :, 1]), qkv[:, :, 2]
        att = jnp.einsum("bshd,bthd->bhst", q, k) / math.sqrt(Dh)
        h = jnp.einsum("bhst,bthd->bshd", jax.nn.softmax(att, -1), v).reshape(B, S, self.dim)
        h = TorchLinear(self.dim, use_bias=False)(h)
        x = x_skip + gate_msa * Dropout(self.dropout)(h, deterministic)

        h = modulate(LayerNorm()(x), shift_mlp, scale_mlp)
        h = TorchLinear(self.mlp_ratio * self.dim)(h)
        h = TorchLinear(self.dim)(jax.nn.gelu(h, approximate=True))
        return x + gate_mlp * Dropout(self.dropout)(h, deterministic)


class DIT3D(nn.Module):
    """F(c_noise, x, β): x (B, n_particles·3), c_noise (B,), β (B,) → (B, n_particles·3)."""

    n_particles: int = 22
    vocab_size: int = 3          # spatial dim
    hidden_size: int = 192
    cond_dim: int = 64
    n_blocks: int = 6
    n_heads: int = 6
    dropout: float = 0.1

    @nn.compact
    def __call__(self, t, x, inverse_temp, deterministic: bool = True):
        B = x.shape[0]
        h = TorchLinear(self.hidden_size)(x.reshape(B, self.n_particles, self.vocab_size))
        c = nn.silu(TimestepEmbedder(self.cond_dim)(t) + TimestepEmbedder(self.cond_dim)(inverse_temp))
        for _ in range(self.n_blocks):
            h = DDiTBlock(self.hidden_size, self.n_heads, dropout=self.dropout)(h, c, deterministic)
        shift, scale = jnp.split(TorchLinear(2 * self.hidden_size, zero_init=True)(c)[:, None], 2, axis=-1)
        h = modulate(LayerNorm()(h), shift, scale)
        out = TorchLinear(self.vocab_size, zero_init=True)(h)
        return out.reshape(B, -1)
