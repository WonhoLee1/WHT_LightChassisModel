import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Callable
from JaxSSO import assemblemodel, solver
from .wht_mesh_model import WHTMeshModel


class WHTOptimizationEngine:
    """
    Core engine for structural optimization linking WHTMeshModel to JaxSSO.
    """
    
    def __init__(self, target_freqs=None, target_vecs=None, target_disp=None):
        self.target_freqs = jnp.array(target_freqs) if target_freqs is not None else None
        self.target_vecs = jnp.array(target_vecs) if target_vecs is not None else None
        self.target_disp = jnp.array(target_disp) if target_disp is not None else None
        
    @staticmethod
    def calculate_mac_jax(phi1, phi2, M=None):
        """
        Calculates Modal Assurance Criterion (MAC) in JAX.
        If M is provided, calculates Mass-weighted MAC.
        phi1, phi2: (ndof,)
        M: (ndof, ndof) or (ndof,) for lumped mass
        """
        if M is None:
            numerator = jnp.square(jnp.abs(jnp.dot(phi1.conj(), phi2)))
            denominator = jnp.dot(phi1.conj(), phi1) * jnp.dot(phi2.conj(), phi2)
        else:
            if M.ndim == 1: # Lumped mass
                numerator = jnp.square(jnp.abs(jnp.sum(phi1.conj() * M * phi2)))
                denominator = (jnp.sum(phi1.conj() * M * phi1)) * (jnp.sum(phi2.conj() * M * phi2))
            else: # Full mass matrix
                numerator = jnp.square(jnp.abs(phi1.conj().T @ M @ phi2))
                denominator = (phi1.conj().T @ M @ phi1) * (phi2.conj().T @ M @ phi2)
        
        return numerator / (denominator + 1e-12)

    def assemble_lumped_mass_jax(self, crds, cnct_quads, t_list, rho, mesh_size):
        """
        Assembles a simplified lumped mass matrix in JAX for MAC weighting.
        crds: (N, 3)
        cnct_quads: (M, 4)
        t_list: (M,) thicknesses
        """
        # Node coordinates for each element
        c = crds[cnct_quads] # (M, 4, 3)
        v1 = c[:, 1, :] - c[:, 0, :]
        v2 = c[:, 3, :] - c[:, 0, :]
        
        # Simplified area calculation for quads
        areas = jnp.linalg.norm(jnp.cross(v1, v2), axis=1) # (M,)
        elem_masses = areas * t_list * rho # (M,)
        mass_per_node = elem_masses / 4.0
        
        ndof = crds.shape[0] * 6
        M_diag = jnp.zeros(ndof)
        
        # Vectorized assembly (simplified for JIT)
        # In a real scenario, we use scatter_add or similar
        # For lumped mass on nodes:
        def add_to_diag(carry, i):
            nid_group = cnct_quads[:, i]
            # Add translation mass
            carry = carry.at[nid_group * 6].add(mass_per_node)
            carry = carry.at[nid_group * 6 + 1].add(mass_per_node)
            carry = carry.at[nid_group * 6 + 2].add(mass_per_node)
            # Add rotation inertia (simplified)
            rot_inertia = mass_per_node * (mesh_size**2) / 12.0
            carry = carry.at[nid_group * 6 + 3].add(rot_inertia)
            carry = carry.at[nid_group * 6 + 4].add(rot_inertia)
            carry = carry.at[nid_group * 6 + 5].add(rot_inertia)
            return carry, None

        M_diag, _ = jax.lax.scan(add_to_diag, M_diag, jnp.arange(4))
        return M_diag

    def create_loss_fn(self, model_fixed_params: dict):
        """
        Returns a JAX-differentiable loss function.
        model_fixed_params: connectivity, boundary conditions, etc.
        """
        
        def loss_fn(design_params: dict):
            # design_params: {"z_offsets": (N,), "thicknesses": (M,)}
            crds = model_fixed_params['base_crds'].at[:, 2].add(design_params['z_offsets'])
            quad_props = model_fixed_params['base_quad_props'].at[:, 0].set(design_params['thicknesses'])
            
            # 1. Assemble Stiffness Matrix
            # (Assuming JaxSSO model_K is available)
            K = assemblemodel.K_func(
                crds, model_fixed_params['ndof'],
                0, None, None, # No beams for now in optimization loop
                model_fixed_params['cnct_quads'], quad_props
            )
            
            # 2. Solve Modal (simplified for Loss)
            # Since JAX eigenvalue solvers are complex, we might use a surrogate 
            # or power iteration for primary mode if only 1st mode is target.
            # For now, let's focus on Static Compliance as a proxy if Modal is too slow.
            
            loss = 0.0
            # Static Case: Compliance minimization
            if self.target_disp is not None:
                # solve K u = f
                # ...
                pass
            
            return loss

        return loss_fn

    def run_optimization(self, initial_params: dict, loss_fn: Callable, n_iter: int = 100, lr: float = 0.01):
        """Executes the optimization loop using a native JAX Adam implementation."""
        
        # Adam Hyperparameters
        b1, b2 = 0.9, 0.999
        eps = 1e-8
        
        # Initialize states for Adam
        # m: first moment, v: second moment
        m = jax.tree_map(jnp.zeros_like, initial_params)
        v = jax.tree_map(jnp.zeros_like, initial_params)
        params = initial_params
        
        @jax.jit
        def step(params, m, v, t):
            grads = jax.grad(loss_fn)(params)
            
            # Update biased first moment estimate
            m = jax.tree_map(lambda mi, gi: b1 * mi + (1 - b1) * gi, m, grads)
            # Update biased second raw moment estimate
            v = jax.tree_map(lambda vi, gi: b2 * vi + (1 - b2) * jnp.square(gi), v, grads)
            
            # Compute bias-corrected first moment estimate
            m_hat = jax.tree_map(lambda mi: mi / (1 - b1**t), m)
            # Compute bias-corrected second raw moment estimate
            v_hat = jax.tree_map(lambda vi: vi / (1 - b2**t), v)
            
            # Update parameters
            new_params = jax.tree_map(
                lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                params, m_hat, v_hat
            )
            return new_params, m, v, grads

        print(f" -> Optimization Start (Steps: {n_iter}, LR: {lr})")
        for t in range(1, n_iter + 1):
            params, m, v, grads = step(params, m, v, t)
            
            if t % 10 == 0 or t == 1:
                grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_leaves(grads)))
                print(f"  [Step {t:03d}] Grad Norm: {grad_norm:.6e}")
        
        return params
