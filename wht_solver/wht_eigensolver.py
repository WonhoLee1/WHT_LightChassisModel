"""
wht_eigensolver.py
==================
WHT FEM Framework — JAX-differentiable Modal Frequency Function

Wraps WHTSolver.solve_modal() in a jax.custom_vjp so that:
  - Forward pass : scipy-based solver (ARPACK / dense eigh)
  - Backward pass: analytical eigensensitivity from WHTSensitivity

This makes modal frequencies fully differentiable inside JAX's gradient
engine without requiring the element code to be JAX-native.

Usage
-----
    from wht_solver.wht_eigensolver import make_modal_freq_fn

    freq_fn = make_modal_freq_fn(base_model, num_modes=10)

    # freq_fn(t_field, z_offsets, E, rho) → jnp.ndarray (n_modes,)
    # jax.grad(lambda *args: jnp.sum(freq_fn(*args)))  — works ✓
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from wht_modeler.wht_mesh_model import WHTMeshModel


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_modal_freq_fn(
    base_model:     "WHTMeshModel",
    num_modes:      int = 10,
    solver_method:  str = 'auto',
    topo_grad:      bool = False,
) -> Callable:
    """
    Build a JAX-differentiable function:

        freq_fn(t_field, z_offsets, E, rho) → frequencies (n_modes,)

    where
        t_field   : (n_elem,) jnp.ndarray — per-element thickness
        z_offsets : (n_nodes,) jnp.ndarray — per-node Z offset
        E         : scalar jnp.ndarray     — global Young's modulus
        rho       : scalar jnp.ndarray     — global density

    Design variables are applied to a COPY of base_model each call,
    so the original model is never mutated.

    Parameters
    ----------
    base_model    : reference model (not mutated)
    num_modes     : number of modal frequencies to return
    solver_method : 'auto' | 'dense' | 'sparse'
    """
    import copy
    from .wht_solver import WHTSolver
    from .wht_sensitivity import WHTSensitivity

    # Precompute stable element / node ordering once
    sorted_eids = sorted(base_model.elements.keys())
    sorted_nids = sorted(base_model.nodes.keys())
    n_nodes     = len(sorted_nids)

    def _apply_design(t_np, dz_np, E_val, rho_val) -> "WHTMeshModel":
        """Return a shallow-copy model with updated design variables."""
        model = copy.deepcopy(base_model)
        # Thickness
        for col, eid in enumerate(sorted_eids):
            prop = model.properties.get(model.elements[eid].pid)
            if prop is not None:
                prop.t = float(t_np[col])
        # Z offsets
        for col, nid in enumerate(sorted_nids):
            model.nodes[nid].z += float(dz_np[col])
        # Global material
        for mat in model.materials.values():
            mat.E   = float(E_val)
            mat.rho = float(rho_val)
        return model

    # ---------------------------------------------------------------
    # custom_vjp definition
    # ---------------------------------------------------------------

    @jax.custom_vjp
    def modal_frequencies(t_field, z_offsets, E, rho):
        """Forward: returns frequencies as JAX array (no grad tracking here)."""
        t_np   = np.asarray(t_field)
        dz_np  = np.asarray(z_offsets)
        E_val  = float(E)
        rho_val = float(rho)

        model  = _apply_design(t_np, dz_np, E_val, rho_val)
        solver = WHTSolver(model, stiffness_scale=1e3)
        result = solver.solve_modal(num_modes=num_modes, method=solver_method)
        return jnp.array(result.frequencies[:num_modes])

    def _fwd(t_field, z_offsets, E, rho):
        freqs = modal_frequencies(t_field, z_offsets, E, rho)
        # Save inputs as residuals for backward
        return freqs, (t_field, z_offsets, E, rho, freqs)

    def _bwd(res, g_freqs):
        t_field, z_offsets, E, rho, freqs = res

        t_np    = np.asarray(t_field)
        dz_np   = np.asarray(z_offsets)
        E_val   = float(E)
        rho_val = float(rho)

        # Re-solve to get eigenvectors (needed for sensitivity)
        model  = _apply_design(t_np, dz_np, E_val, rho_val)
        solver = WHTSolver(model, stiffness_scale=1e3)
        result = solver.solve_modal(num_modes=num_modes, method=solver_method)
        sens   = WHTSensitivity(solver, result)

        # Chain rule from frequency f to eigenvalue λ = (2πf)²:
        #   ∂L/∂λᵢ = ∂L/∂fᵢ · ∂fᵢ/∂λᵢ = g_freqs[i] / (4π²fᵢ)
        f_np     = np.asarray(freqs) + 1e-12
        g_lambda = np.asarray(g_freqs) / (4.0 * np.pi ** 2 * f_np)  # (n_modes,)

        # Thickness gradient: g_t[e] = Σᵢ g_lambda[i] · ∂λᵢ/∂tₑ
        dλ_dt  = sens.thickness()             # (n_modes, n_elem)
        g_t    = jnp.array(g_lambda @ dλ_dt) # (n_elem,)

        # Z-offset gradient: only computed when topo_grad=True (expensive FD)
        if topo_grad:
            dλ_dz_arr, _ = sens.topography(node_ids=sorted_nids)  # (n_modes, n_nodes)
            g_z = jnp.array(g_lambda @ dλ_dz_arr)                  # (n_nodes,)
        else:
            g_z = jnp.zeros(n_nodes)

        # Global material gradients
        dλ_dE   = sens.E_global()                     # (n_modes,)
        dλ_dρ   = sens.rho_global()                   # (n_modes,)
        g_E     = jnp.array(float(g_lambda @ dλ_dE))
        g_rho   = jnp.array(float(g_lambda @ dλ_dρ))

        return g_t, g_z, g_E, g_rho

    modal_frequencies.defvjp(_fwd, _bwd)
    return modal_frequencies
