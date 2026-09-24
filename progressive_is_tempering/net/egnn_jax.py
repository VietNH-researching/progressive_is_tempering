"""E(n)-equivariant graph neural network for particle systems.

Score-network backbone for targets made of N interacting particles in D
spatial dimensions (flat state x in R^{N*D}). Uses dense all-pairs
computation (no explicit scatter/gather) since the graph is fully connected
and fixed.

Reference:
    Satorras, Hoogeboom, Welling (2021). E(n) Equivariant Graph Neural Networks.
    https://arxiv.org/abs/2102.09844
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from progressive_is_tempering.net.embedding import time_embedding


def remove_mean(x: Array, n_particles: int, spatial_dim: int) -> Array:
    """Project x to the zero center-of-mass subspace.

    Subtracts the per-sample mean particle position so that
    sum_i x_i = 0 for every sample in the batch.

    Args:
        x:           Flat particle positions, shape (..., n_particles * spatial_dim).
        n_particles: Number of particles N.
        spatial_dim: Spatial dimension D per particle.

    Returns:
        x with zero center of mass, same shape as input.
    """
    shape = x.shape
    coord = x.reshape(-1, n_particles, spatial_dim)
    coord = coord - coord.mean(axis=1, keepdims=True)
    return coord.reshape(shape)


class EGNNLayer(nn.Module):
    """Single E(n)-equivariant graph convolutional layer.

    Implements the E_GCL update from Satorras et al. (2021) for a
    fully-connected graph, using dense all-pairs operations instead of
    explicit scatter/gather (valid since the graph topology is fixed).

    Update equations
    ----------------
    Edge messages:
        m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2)

    Coordinate update (equivariant):
        x_i <- x_i + sum_{j != i} (x_i - x_j) / (||x_i - x_j|| + 1) * phi_x(m_ij)

    Node update:
        h_i <- phi_h(h_i, sum_j m_ij)   [+ h_i if recurrent]

    Attributes:
        hidden_nf:    Hidden feature dimension.
        act_fn:       Element-wise activation. Default: nn.silu.
        recurrent:    Residual skip on node features. Default: True.
        attention:    Gate edge messages with learned attention. Default: True.
        tanh:         Apply tanh to coordinate update weights. Default: True.
        coords_range: Scale factor for tanh'd coordinate updates. Default: 1.0.
    """

    hidden_nf: int
    act_fn: Callable = nn.silu
    recurrent: bool = True
    attention: bool = True
    tanh: bool = True
    coords_range: float = 1.0

    @nn.compact
    def __call__(self, h: Array, coord: Array, edge_attr: Array) -> tuple[Array, Array]:
        """
        Args:
            h:         Node features,                   shape (B, N, hidden_nf).
            coord:     Particle positions,              shape (B, N, D).
            edge_attr: Initial squared pairwise dists,  shape (B, N, N, 1).
                       Computed once from the input coordinates in EGNNDynamics
                       and passed unchanged to every layer.

        Returns:
            h_new:     Updated node features,  shape (B, N, hidden_nf).
            coord_new: Updated coordinates,    shape (B, N, D).
        """
        B, N, _ = coord.shape

        # ---- Pairwise geometry ------------------------------------------------
        # coord_diff[b, i, j] = coord[b, i] - coord[b, j]
        coord_diff = coord[:, :, None, :] - coord[:, None, :, :]   # (B, N, N, D)
        radial = jnp.sum(coord_diff ** 2, axis=-1, keepdims=True)  # (B, N, N, 1)
        # Normalised direction (safe: avoid div-by-zero with +1 in denominator)
        coord_diff_norm = coord_diff / (jnp.sqrt(radial + 1e-8) + 1.0)  # (B, N, N, D)

        # ---- Edge messages phi_e ----------------------------------------------
        h_i = jnp.broadcast_to(h[:, :, None, :], (B, N, N, self.hidden_nf))
        h_j = jnp.broadcast_to(h[:, None, :, :], (B, N, N, self.hidden_nf))
        # Concatenate current radial AND fixed initial edge_attr:
        # cat([source, target, radial, edge_attr])
        edge_input = jnp.concatenate([h_i, h_j, radial, edge_attr], axis=-1)  # (B, N, N, 2*hid+2)

        m = nn.Dense(self.hidden_nf)(edge_input)
        m = self.act_fn(m)
        m = nn.Dense(self.hidden_nf)(m)
        m = self.act_fn(m)  # (B, N, N, hidden_nf)

        if self.attention:
            att = nn.sigmoid(nn.Dense(1)(m))  # (B, N, N, 1)
            m = m * att

        # ---- Coordinate update phi_x -----------------------------------------
        w = nn.Dense(self.hidden_nf)(m)
        w = self.act_fn(w)
        w = nn.Dense(1, use_bias=False)(w)  # (B, N, N, 1)
        if self.tanh:
            w = jnp.tanh(w) * self.coords_range

        # Mask diagonal (no self-interaction)
        diag_mask = (1.0 - jnp.eye(N))[None, :, :, None]  # (1, N, N, 1)
        w = w * diag_mask
        m = m * diag_mask  # also mask messages for node aggregation

        # Sum equivariant displacements over neighbours
        coord_new = coord + jnp.sum(coord_diff_norm * w, axis=2)  # (B, N, D)

        # ---- Node update phi_h -----------------------------------------------
        m_agg = jnp.sum(m, axis=2)  # aggregate messages: (B, N, hidden_nf)
        node_input = jnp.concatenate([h, m_agg], axis=-1)  # (B, N, 2*hidden_nf)

        h_new = nn.Dense(self.hidden_nf)(node_input)
        h_new = self.act_fn(h_new)
        h_new = nn.Dense(self.hidden_nf)(h_new)
        if self.recurrent:
            h_new = h + h_new

        return h_new, coord_new


class EGNNDynamics(nn.Module):
    """E(n)-equivariant GNN dynamics network for particle systems.

    The network takes flat particle coordinates x, time t and (optionally)
    the inverse temperature beta, and returns a learned displacement vector
    of the same shape, projected to zero centre of mass. This output is used
    directly as the beta-conditioned excess score

        s_theta(x_t, t, beta) ~ grad_x log(p_t^beta / p_t^rho),

    with no further wrapping.

    Equivariance properties
    -----------------------
    - Rotation / reflection equivariant: output transforms as a vector field.
    - Permutation equivariant: invariant to relabelling of particles.
    - Translation equivariant: vel = coord_out - coord_in is translation-
      invariant (coord_diff is translation-invariant). With trap_scale > 0
      the target breaks translational symmetry, which is consistent.

    Time conditioning
    -----------------
    Time t is broadcast as a constant scalar feature across all N nodes
    before the input embedding (when condition_time=True).

    Attributes:
        n_particles:    Number of particles N. Default: 4.
        spatial_dim:    Spatial dimension D per particle. Default: 2.
        hidden_nf:      Hidden feature dimension. Default: 128.
        n_layers:       Number of EGNN layers. Default: 5.
        recurrent:      Residual skip on node features. Default: True.
        attention:      Attention gating on edge messages. Default: True.
        tanh:           Tanh on coordinate update weights. Default: True.
        condition_time: Use t as initial node feature. Default: True.
        coords_range:   Total coord-update scale, split evenly across layers.
                        Default: 15.0.
    """

    n_particles: int = 4
    spatial_dim: int = 2
    hidden_nf: int = 128
    n_layers: int = 5
    recurrent: bool = True
    attention: bool = True
    tanh: bool = True
    condition_time: bool = True
    condition_beta: bool = False
    beta_embedding_dim: int = 8
    coords_range: float = 15.0

    @nn.compact
    def __call__(self, x: Array, t: Array, beta: Array | None = None) -> Array:
        """
        Args:
            x:    Flat particle positions,        shape (B, N*D).
            t:    Time,                           shape (B, 1).
            beta: Tempering inverse temperature,  shape (B, 1).
                  Required when condition_beta=True.

        Returns:
            vel: Learned displacements, shape (B, N*D).
        """
        B = x.shape[0]
        N, D = self.n_particles, self.spatial_dim
        coords_range_per_layer = self.coords_range / self.n_layers

        # Reshape flat state to particle coordinates
        coord = x.reshape(B, N, D)   # (B, N, D)
        coord_in = coord             # keep for velocity computation

        # Compute edge_attr = sum((x[i] - x[j])^2) once from the INITIAL
        # coordinates, before the layer loop, and pass it unchanged to every
        # layer. From layer 1 onwards, coord changes but edge_attr stays fixed, so
        # it carries different information from the per-layer radial.
        coord_diff_init = coord_in[:, :, None, :] - coord_in[:, None, :, :]  # (B, N, N, D)
        edge_attr = jnp.sum(coord_diff_init ** 2, axis=-1, keepdims=True)    # (B, N, N, 1)

        # Build initial node features by concatenating conditioning signals,
        # broadcast to all N particles.
        features = []
        if self.condition_time:
            features.append(jnp.ones((B, N, 1)) * t[:, None, :])  # (B, N, 1)
        if self.condition_beta:
            # Sinusoidal embedding (freq range 0.5–5.0, distinct from time's 1.0–10.0)
            # so the network can distinguish nearby β values in [0.05, 1.0].
            beta_embed = time_embedding(beta, 0.5, 5.0, self.beta_embedding_dim)  # (B, E)
            features.append(jnp.broadcast_to(
                beta_embed[:, None, :], (B, N, self.beta_embedding_dim)
            ))  # (B, N, E)
        h = jnp.concatenate(features, axis=-1) if features else jnp.ones((B, N, 1))

        # Input embedding -> hidden_nf
        h = nn.Dense(self.hidden_nf)(h)  # (B, N, hidden_nf)

        # Stack of EGNN layers — each layer has its own independent parameters.
        # edge_attr is passed unchanged to every layer (fixed initial distances).
        for _ in range(self.n_layers):
            h, coord = EGNNLayer(
                hidden_nf=self.hidden_nf,
                act_fn=nn.silu,
                recurrent=self.recurrent,
                attention=self.attention,
                tanh=self.tanh,
                coords_range=coords_range_per_layer,
            )(h, coord, edge_attr)

        # Output: velocity = displacement from input (equivariant vector field)
        vel = coord - coord_in                              # (B, N, D)
        # Project to zero-CoM subspace: ensures the learned correction
        # has no net translation component, consistent with the constraint.
        vel = vel - vel.mean(axis=1, keepdims=True)         # (B, N, D)
        return vel.reshape(B, N * D)                        # (B, N*D)
