"""
Neural network architectures for iDEM.

This module provides MLP architectures designed for score function approximation
in diffusion-based sampling methods.

Classes
-------
RMSNorm
    Root Mean Square normalization layer.
ResBlock
    Pre-norm residual block with RMSNorm.
ResMLP
    Residual MLP with small output initialization (recommended for score networks).
MLP
    Simple feedforward MLP with ELU activations.

Notes
-----
For score networks in iDEM, use ResMLP which has built-in small output
initialization for training stability.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn
from jax import Array

__all__ = ["MLP", "ResMLP"]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes inputs by their RMS (root mean square) value, then applies a
    learnable scale. Simpler and often faster than LayerNorm since it skips
    mean centering and bias terms.

    Formula
    -------
    RMSNorm(x) = scale * x / RMS(x)

    where RMS(x) = sqrt(mean(x^2) + epsilon)

    Attributes
    ----------
    epsilon : float
        Small constant for numerical stability. Default: 1e-6.

    References
    ----------
    Zhang & Sennrich (2019). "Root Mean Square Layer Normalization"
    https://arxiv.org/abs/1910.07467
    """
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Apply RMS normalization.

        Parameters
        ----------
        x : Array, shape (..., features)
            Input tensor. Normalization is applied over the last axis.

        Returns
        -------
        Array, shape (..., features)
            Normalized tensor with learnable per-feature scale.
        """
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.epsilon)
        return scale * x / rms


class ResBlock(nn.Module):
    """
    Pre-norm residual block with RMSNorm.

    Architecture
    ------------
    output = x + MLP(RMSNorm(x))

    where MLP is a two-layer network: Dense → ELU → Dense

    Uses pre-normalization (norm before the transformation) which tends to be
    more stable than post-normalization, especially in deep networks.

    Attributes
    ----------
    hidden_dim : int
        Hidden dimension for both Dense layers. Should match input dimension
        for the residual connection to work without projection.

    Notes
    -----
    The input and output dimensions must match since there is no projection
    on the skip connection. This block preserves dimensionality.
    """
    hidden_dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Forward pass through the residual block.

        Parameters
        ----------
        x : Array, shape (..., hidden_dim)
            Input tensor. Last dimension must equal hidden_dim.

        Returns
        -------
        Array, shape (..., hidden_dim)
            Output tensor with residual connection applied.
        """
        residual = x

        # Pre-norm
        h = RMSNorm()(x)

        # Two-layer MLP
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.elu(h)
        h = nn.Dense(self.hidden_dim)(h)

        return residual + h


class ResMLP(nn.Module):
    """
    Residual MLP with RMSNorm and small output initialization.

    Architecture
    ------------
    Input → Dense(hidden) → [ResBlock × n_blocks] → Dense(output)

    The output layer uses small weight initialization (default stddev=1e-4) to
    ensure near-zero initial outputs. This improves training stability by:
    - Preventing large initial predictions that cause gradient explosions
    - Allowing the network to gradually "turn on" during training
    - Breaking symmetry while maintaining conservative initial behavior

    This technique is similar to zero-initialization used in ControlNet and
    diffusion model fine-tuning, but with small random values instead of
    exact zeros to enable immediate gradient flow.

    Attributes
    ----------
    dim_output : int
        Output dimension.
    hidden_dim : int
        Hidden dimension for all layers. Default: 256.
    n_blocks : int
        Number of residual blocks. Default: 2.
    output_init_scale : float
        Standard deviation for output layer weight initialization.
        Smaller values → outputs closer to zero at initialization.
        Default: 1e-4.

    Examples
    --------
    >>> mlp = ResMLP(dim_output=16, hidden_dim=128, n_blocks=3)
    >>> params = mlp.init(key, x)
    >>> out = mlp.apply(params, x)  # Initial outputs ~1e-4 magnitude

    >>> # For even more conservative initialization:
    >>> mlp = ResMLP(dim_output=16, output_init_scale=1e-6)
    """
    dim_output: int
    hidden_dim: int = 256
    n_blocks: int = 2
    output_init_scale: float = 1e-4

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Forward pass through the residual MLP.

        Parameters
        ----------
        x : Array, shape (..., input_dim)
            Input tensor. Can be batched.

        Returns
        -------
        Array, shape (..., dim_output)
            Output tensor with small initial magnitude (~output_init_scale).
        """
        # Input projection (standard Lecun init)
        x = nn.Dense(self.hidden_dim)(x)

        # Residual blocks (standard init, preserves signal magnitude)
        for _ in range(self.n_blocks):
            x = ResBlock(self.hidden_dim)(x)

        # Output projection (small init for stability)
        x = nn.Dense(
            self.dim_output,
            kernel_init=nn.initializers.normal(stddev=self.output_init_scale),
            bias_init=nn.initializers.zeros,
        )(x)
        return x


class MLP(nn.Module):
    """
    Multi-layer perceptron with ELU activation.

    A simple feedforward network with ELU nonlinearities between hidden layers
    and a linear output layer. Uses standard Lecun initialization.

    Architecture
    ------------
    Input → [Dense → ELU] × (n_layers - 1) → Dense → Output

    Attributes
    ----------
    dim_list : list[int]
        Dimensions of each layer (excluding input dimension).
        The last element is the output dimension.

    Examples
    --------
    >>> # Two hidden layers (256 each) and output dimension 16
    >>> mlp = MLP(dim_list=[256, 256, 16])
    >>> params = mlp.init(key, x)
    >>> out = mlp.apply(params, x)  # shape: (..., 16)

    >>> # Single hidden layer
    >>> mlp = MLP(dim_list=[128, 10])

    Notes
    -----
    For networks requiring near-zero initial outputs (e.g., score networks),
    consider using ResMLP which has built-in small output initialization.
    """
    dim_list: list[int]

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : Array, shape (..., input_dim)
            Input tensor. Can be batched.

        Returns
        -------
        Array, shape (..., dim_list[-1])
            Output tensor.
        """
        # Hidden layers with ELU activation
        for dim in self.dim_list[:-1]:
            x = nn.Dense(dim)(x)
            x = nn.elu(x)

        # Output layer (no activation)
        x = nn.Dense(self.dim_list[-1])(x)
        return x
