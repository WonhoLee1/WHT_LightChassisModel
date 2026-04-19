"""
objectives.py
=============
WHT FEM Framework — JAX-based Objective Functions

All functions are JAX-differentiable and jit-compatible.

Functions
---------
mac                         - Modal Assurance Criterion
mass_weighted_mac           - Mass-weighted MAC
mac_matrix                  - Full (n×m) MAC matrix
freq_loss_with_mac_assign   - Frequency MSE with MAC soft-assignment (mode-switch safe)
laplacian_smoothness        - Topography smoothness regularization ||L·z||²
multi_objective_loss        - Combined loss (freq + MAC + static RMSE + smooth)
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import jax.numpy as jnp
import jax

if TYPE_CHECKING:
    from .wht_result import WHTSolverResult
    from .wht_mapper import WHTMapper


# ---------------------------------------------------------------------------
# MAC
# ---------------------------------------------------------------------------

def mac(phi_a: jnp.ndarray, phi_b: jnp.ndarray) -> jnp.ndarray:
    """
    Standard Modal Assurance Criterion.

    MAC(a,b) = (φ_a · φ_b)² / ((φ_a · φ_a)(φ_b · φ_b))

    Returns scalar ∈ [0, 1].  1.0 = perfect correlation.
    """
    num   = jnp.square(jnp.dot(phi_a, phi_b))
    denom = jnp.dot(phi_a, phi_a) * jnp.dot(phi_b, phi_b) + 1e-12
    return num / denom


def mass_weighted_mac(
    phi_a:  jnp.ndarray,
    phi_b:  jnp.ndarray,
    M_diag: jnp.ndarray,
) -> jnp.ndarray:
    """
    Mass-weighted MAC.

    MAC_M(a,b) = (φ_aᵀ M φ_b)² / ((φ_aᵀ M φ_a)(φ_bᵀ M φ_b))

    M_diag : (ndof,) lumped mass diagonal
    """
    Mphi_b = M_diag * phi_b
    Mphi_a = M_diag * phi_a
    num   = jnp.square(jnp.dot(phi_a, Mphi_b))
    denom = jnp.dot(phi_a, Mphi_a) * jnp.dot(phi_b, Mphi_b) + 1e-12
    return num / denom


def mac_matrix(
    phis_opt:    jnp.ndarray,
    phis_target: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute full MAC matrix between two sets of mode shapes.

    Parameters
    ----------
    phis_opt    : (n_opt,   N) flattened mode shapes
    phis_target : (n_target, N) flattened mode shapes

    Returns
    -------
    (n_opt, n_target) MAC matrix
    """
    # Normalise rows
    norm_opt = jnp.linalg.norm(phis_opt,    axis=1, keepdims=True) + 1e-12
    norm_tgt = jnp.linalg.norm(phis_target, axis=1, keepdims=True) + 1e-12
    phi_o = phis_opt    / norm_opt
    phi_t = phis_target / norm_tgt

    cross = phi_o @ phi_t.T              # (n_opt, n_target)
    return jnp.square(cross)             # MAC = (cos θ)²


# ---------------------------------------------------------------------------
# Mode-switching-safe frequency loss
# ---------------------------------------------------------------------------

def freq_loss_with_mac_assign(
    freqs_opt:    jnp.ndarray,
    freqs_target: jnp.ndarray,
    phis_opt:     jnp.ndarray,
    phis_target:  jnp.ndarray,
) -> jnp.ndarray:
    """
    Frequency MSE with MAC soft-assignment.

    Avoids the mode-switching problem by weighting each frequency pair
    with MAC(i,j) instead of using argmax (non-differentiable).

    loss = Σ_i Σ_j  MAC[i,j] · (freqs_opt[i] - freqs_target[j])²
           / Σ_i Σ_j  MAC[i,j]
    """
    mac_mat = mac_matrix(phis_opt, phis_target)   # (n_opt, n_target)

    freq_o = freqs_opt[:, None]    # (n_opt, 1)
    freq_t = freqs_target[None, :] # (1, n_target)
    diff_sq = jnp.square(freq_o - freq_t)   # (n_opt, n_target)

    weighted  = jnp.sum(mac_mat * diff_sq)
    normalise = jnp.sum(mac_mat) + 1e-12
    return weighted / normalise


# ---------------------------------------------------------------------------
# Topography smoothness regularization
# ---------------------------------------------------------------------------

def laplacian_smoothness(
    z_offsets: jnp.ndarray,
    adjacency: jnp.ndarray,
    lambda_smooth: float = 0.01,
) -> jnp.ndarray:
    """
    Laplacian graph regularization: ||L · z||²

    Parameters
    ----------
    z_offsets     : (N,) nodal Z-offsets
    adjacency     : (N, max_neighbors) 0-based neighbor indices.
                    Pad with -1 for nodes with fewer neighbors.
    lambda_smooth : regularization weight

    Returns
    -------
    Scalar penalty term.
    """
    # For each node, compute z[i] - mean(z[neighbors])
    # Mask out padding (-1 entries)
    valid = adjacency >= 0                                 # (N, K) bool
    safe_adj = jnp.where(valid, adjacency, 0)

    neighbor_z = z_offsets[safe_adj]                      # (N, K)
    neighbor_z = jnp.where(valid, neighbor_z, 0.0)
    n_neighbors = jnp.sum(valid, axis=1, keepdims=True).clip(1)  # (N, 1)
    mean_z = jnp.sum(neighbor_z, axis=1) / n_neighbors[:, 0]    # (N,)

    diff = z_offsets - mean_z
    return lambda_smooth * jnp.sum(jnp.square(diff))


# ---------------------------------------------------------------------------
# Combined multi-objective loss
# ---------------------------------------------------------------------------

def multi_objective_loss(
    freqs_opt:    jnp.ndarray,
    freqs_target: jnp.ndarray,
    phis_opt:     jnp.ndarray,
    phis_target:  jnp.ndarray,
    disp_opt:     Optional[jnp.ndarray] = None,
    disp_target:  Optional[jnp.ndarray] = None,
    z_offsets:    Optional[jnp.ndarray] = None,
    adjacency:    Optional[jnp.ndarray] = None,
    weights: dict = None,
) -> jnp.ndarray:
    """
    Combined loss for topology / thickness / material optimization.

    f_total = w_freq   · freq_loss   (MAC soft-assignment)
            + w_mac    · mac_loss    (1 - mean_MAC)
            + w_static · rmse_loss   (displacement RMSE, optional)
            + w_smooth · smooth_loss (Laplacian, optional)

    Parameters
    ----------
    freqs_opt    : (n_modes,)
    freqs_target : (n_modes,)
    phis_opt     : (n_modes, N) flattened
    phis_target  : (n_modes, N) flattened (same space, RBF-mapped)
    disp_opt     : (N_load_cases, N, 3) optional static displacements
    disp_target  : (N_load_cases, N, 3) optional reference displacements
    z_offsets    : (N,) for smoothness regularization
    adjacency    : (N, K) for smoothness regularization
    weights      : {"freq": float, "mac": float, "static": float, "smooth": float}
    """
    if weights is None:
        weights = {"freq": 1.0, "mac": 1.0, "static": 1.0, "smooth": 0.01}

    loss = jnp.array(0.0)

    # --- Frequency loss ---
    if weights.get("freq", 0.0) > 0.0:
        floss = freq_loss_with_mac_assign(
            freqs_opt, freqs_target, phis_opt, phis_target
        )
        loss = loss + weights["freq"] * floss

    # --- MAC loss ---
    if weights.get("mac", 0.0) > 0.0:
        mac_mat  = mac_matrix(phis_opt, phis_target)
        mac_diag = jnp.diag(mac_mat)
        mac_loss = jnp.mean(1.0 - mac_diag)
        loss = loss + weights["mac"] * mac_loss

    # --- Static displacement RMSE ---
    if (weights.get("static", 0.0) > 0.0
            and disp_opt is not None
            and disp_target is not None):
        diff      = disp_opt - disp_target
        rmse_loss = jnp.sqrt(jnp.mean(jnp.square(diff)))
        loss = loss + weights["static"] * rmse_loss

    # --- Topography smoothness ---
    if (weights.get("smooth", 0.0) > 0.0
            and z_offsets is not None
            and adjacency is not None):
        smooth_loss = laplacian_smoothness(
            z_offsets, adjacency, lambda_smooth=weights["smooth"]
        )
        loss = loss + smooth_loss

    return loss
