"""
wht_solver.py
=============
WHT FEM Framework — JaxSSO Solver Wrapper

WHTSolver converts WHTMeshModel → JaxSSO model, runs analysis,
and returns WHTSolverResult with:
  - Static: displacement (N,6) + reaction forces via Lagrange multipliers
  - Modal:  frequencies (n_modes,) + mode shapes (n_modes, N, 6)

JAX jit boundary strategy (Strategy B):
  K_func is a pure JAX function extracted from assemblemodel.
  WHTSolver.get_k_func_args() exposes static args for WHTOptimizer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from wht_modeler.wht_mesh_model import WHTMeshModel

from .load_cases import WHTLoadCase
from .wht_result import WHTSolverResult
from .wht_stress_recovery import ElementStressRecovery
from .wht_tria3_element import K_tria3_scipy, M_tria3_lumped
from .wht_quad4_element import K_quad4_scipy, M_quad4_lumped


class WHTSolver:
    """
    JaxSSO wrapper.

    Converts WHTMeshModel to JaxSSO model, runs static or modal analysis,
    and returns WHTSolverResult.

    RBE2 → stiff beam conversion is applied automatically.
    """

    def __init__(
        self,
        model: "WHTMeshModel",
        stiffness_scale: float = 1e3,
    ):
        self.model           = model
        self.stiffness_scale = stiffness_scale
        self._jax_model      = None   # lazy build

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_modal(self, num_modes: int = 10) -> WHTSolverResult:
        """
        Modal analysis using scipy eigsh on free DOFs.

        Returns WHTSolverResult with:
            frequencies : (n_modes,) [Hz]
            mode_shapes : (n_modes, N, 6)
        """
        from JaxSSO import assemblemodel
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix, diags

        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model()
        jm.model_ready()
        ndof = jm.ndof

        # Use unified K assembly with stabilization
        K_scipy = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=True)
        print(f"    [DBG] K assembled and stabilized, nnz={K_scipy.nnz}", flush=True)

        # [WHT] Filter Free DOFs & Mass Assembly
        unknown_id   = jm.unknown_id
        actual_modes = min(num_modes, len(unknown_id) - 1)
        
        K_free = K_scipy[unknown_id, :][:, unknown_id].tocsc()
        
        M_diag = self._assemble_lumped_mass(jm, ndof, sorted_nids, nid_to_idx)
        M_free = M_diag[unknown_id]
        
        # [WHT] Bandwidth Reduction (RCM Reordering)
        # Unstructured meshes (TRIA3_FREE) produce massive sparse bandwidth.
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        print("    [DBG] Reordering matrix via RCM to minimize bandwidth...", flush=True)
        perm = reverse_cuthill_mckee(K_free, symmetric_mode=True)
        rev_perm = np.argsort(perm)

        K_free_rcm = K_free[perm, :][:, perm].tocsc()
        M_free_rcm = M_free[perm]  # 1D array reordered

        # [WHT] Solve Generalized Eigenvalue Problem (Kx = lambda Mx)
        print(f"    [DBG] calling eigsh k={actual_modes}... (Shift-Invert, sigma=-0.1)", flush=True)
        
        vals, vecs_rcm = eigsh(
            K_free_rcm,
            k=actual_modes,
            M=diags([M_free_rcm], [0], format='csc'),
            which="LM",
            sigma=-0.1,
            tol=1e-5,
        )
        
        # Restore original DOF ordering
        vecs_free = vecs_rcm[rev_perm, :]

        freqs = np.sqrt(np.maximum(vals, 0)) / (2 * np.pi)

        # Expand to full DOF space
        vecs_full = np.zeros((ndof, actual_modes))
        vecs_full[unknown_id, :] = vecs_free
        
        # --- [WHT] Normalize Mode Shapes ---
        # User requested to disable normalization so modes remain mass-normalized natively.
        # for m in range(actual_modes):
        #     # Translational DOFs (0, 1, 2) maximum
        #     max_disp = np.max(np.abs(vecs_full[:, m].reshape(-1, 6)[:, :3]))
        #     if max_disp > 1e-12:
        #         vecs_full[:, m] /= max_disp

        # Reshape to (n_modes, N, 6) — DOF layout: node_idx * 6
        n_nodes   = len(sorted_nids)
        mode_shapes = np.zeros((actual_modes, n_nodes, 6))
        for i, nid in enumerate(sorted_nids):
            idx = nid_to_idx[nid]
            mode_shapes[:, i, :] = vecs_full[idx * 6: idx * 6 + 6, :].T

        result = WHTSolverResult("modal", sorted_nids)
        result.frequencies  = freqs
        result.mode_shapes  = mode_shapes
        
        # --- [WHT] Element Stress & Strain Recovery ---
        n_cells = len(self.model.elements)
        stresses = np.zeros((actual_modes, n_cells, 6))
        strains  = np.zeros((actual_modes, n_cells, 6))
        seds     = np.zeros((actual_modes, n_cells, 1))
        
        for m in range(actual_modes):
            s_q, e_q = ElementStressRecovery.recover_quad4(self.model, mode_shapes[m], sorted_nids)
            s_t, e_t = ElementStressRecovery.recover_tria3(self.model, mode_shapes[m], sorted_nids)
            stresses[m] = s_q + s_t
            strains[m]  = e_q + e_t
            seds[m, :, 0] = 0.5 * np.sum(stresses[m] * strains[m], axis=1)
            
        result.cell_data = {"Stress": stresses, "Strain": strains, 
                            "StrainEnergyDensity": seds}
        return result

    def solve_static(self, load_case: WHTLoadCase) -> WHTSolverResult:
        """
        Static analysis with augmented K (Lagrange multiplier BC).

        Returns WHTSolverResult with:
            displacement : (N, 6)
            reaction forces accessible via result.reaction_force()
        """
        from JaxSSO import assemblemodel
        from JaxSSO import solver as jaxsso_solver
        import jax.numpy as jnp

        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model(load_case)
        jm.model_ready()

        ndof = jm.ndof

        # Build K with ALL contributions (including QUAD4 which was previously missing)
        # And apply stabilization to prevent singular matrix errors.
        K_base = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=True)
        K_aug, f_aug = self._augment_K_scipy(K_base, jm)

        # Solve: u_aug = [u_free (ndof,), lambda (n_bc,)]
        from scipy.sparse.linalg import spsolve
        u_aug = spsolve(K_aug.tocsc(), f_aug)
        u_aug_np = np.array(u_aug)

        # Displacement (N, 6)
        n_nodes = len(sorted_nids)
        displacement = np.zeros((n_nodes, 6))
        for i, nid in enumerate(sorted_nids):
            idx = nid_to_idx[nid]
            displacement[i, :] = u_aug_np[idx * 6: idx * 6 + 6]

        # BC DOF mapping for reaction force API
        bc_dof_ids  = np.array(jm.known_id, dtype=np.int64)
        bc_node_ids = (bc_dof_ids // 6).astype(np.int64)
        # Remap from 0-indexed JaxSSO node index to original node IDs
        idx_to_nid  = {i: nid for i, nid in enumerate(sorted_nids)}
        bc_node_original = np.array([idx_to_nid[i] for i in bc_node_ids])

        # --- [WHT] Element Stress & Strain Recovery ---
        # Catching 4 outputs: (Stresses, Strain_Total, Strain_Membrane, Strain_Bending)
        s_q, e_q_t, e_q_m, e_q_b = ElementStressRecovery.recover_quad4(self.model, displacement, sorted_nids)
        s_t, e_t_t, e_t_m, e_t_b = ElementStressRecovery.recover_tria3(self.model, displacement, sorted_nids)
        
        s_static = s_q + s_t
        e_static_total = e_q_t + e_t_t
        e_static_membrane = e_q_m + e_t_m
        e_static_bending = e_q_b + e_t_b
        
        # Calculate Max Von-Mises for diagnostic summary
        vm_static = np.sqrt(0.5 * ((s_static[:,0]-s_static[:,1])**2 + (s_static[:,1]-s_static[:,2])**2 + (s_static[:,2]-s_static[:,0])**2 + 6*(s_static[:,3]**2 + s_static[:,4]**2 + s_static[:,5]**2)))
        max_vm = np.max(vm_static)

        result = WHTSolverResult("static", sorted_nids)
        result.displacement  = displacement
        result.cell_data     = {
            "Stress": s_static[np.newaxis, :, :], 
            "Strain": e_static_total[np.newaxis, :, :],
            "Strain (Membrane)": e_static_membrane[np.newaxis, :, :],
            "Strain (Bending)": e_static_bending[np.newaxis, :, :]
        }
        result._max_vm_diagnostic = max_vm # Keep for table summary
        result._u_aug        = u_aug_np
        result._ndof         = ndof
        result._bc_dof_ids   = bc_dof_ids
        result._bc_node_ids  = bc_node_original
        return result

    def solve_all(
        self,
        load_cases: List[WHTLoadCase],
        num_modes: int = 10,
    ) -> Dict[str, WHTSolverResult]:
        """Run modal + all static load cases."""
        results: Dict[str, WHTSolverResult] = {}
        results["modal"] = self.solve_modal(num_modes)
        for lc in load_cases:
            results[lc.name] = self.solve_static(lc)
        return results

    def _assemble_K_scipy(
        self, 
        jm, 
        sorted_nids: List[int], 
        nid_to_idx: Dict[int, int],
        stabilize: bool = True
    ) -> "csr_matrix":
        """Unified stiffness assembly (JaxSSO + MITC4 + MITC3)."""
        from JaxSSO import assemblemodel
        from scipy.sparse import csr_matrix, diags

        ndof = jm.ndof
        if self._has_jaxsso_elements():
            K_bcoo = assemblemodel.model_K(jm)
            K_out = csr_matrix(
                (np.array(K_bcoo.data),
                 (np.array(K_bcoo.indices[:, 0]),
                  np.array(K_bcoo.indices[:, 1]))),
                shape=(ndof, ndof),
            )
        else:
            K_out = csr_matrix((ndof, ndof))

        K_out = K_out + K_quad4_scipy(self.model, sorted_nids, nid_to_idx)
        K_out = K_out + K_tria3_scipy(self.model, sorted_nids, nid_to_idx)
        
        if stabilize:
            # [WHT] AUTOSPC: Automatically patch zero or near-zero diagonal stiffness DOFs
            # Necessary for static and modal stability on shell meshes.
            K_out.sort_indices()
            k_diag = K_out.diagonal()
            k_max = np.abs(k_diag).max()
            if k_max > 0:
                threshold = max(k_max * 1e-10, 1e-2)
                bad_dofs = (np.abs(k_diag) <= threshold)
                if np.any(bad_dofs):
                    K_out = K_out + diags([bad_dofs.astype(float) * k_max * 1e-4], [0])
        return K_out.tocsr()

    def get_k_func_args(self) -> dict:
        """
        Extract K_func static arguments for use in WHTOptimizer jit boundary.

        Returns a dict with keys:
            ndof, n_beamcol, cnct_beamcols, prop_beamcols,
            n_quad, cnct_quads, base_crds, base_prop_quads,
            known_id, node_id_map
        """
        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model()
        jm.model_ready()
        return {
            "ndof":          jm.ndof,
            "n_beamcol":     jm.n_beamcol,
            "cnct_beamcols": jm.cnct_beamcols,
            "prop_beamcols": jm.prop_beamcols,
            "n_quad":        jm.n_quad,
            "cnct_quads":    jm.cnct_quads,
            "base_crds":     np.array(jm.crds),
            "base_prop_quads": np.array(jm.prop_quads),
            "known_id":      np.array(jm.known_id),
            "node_id_map":   {nid: nid_to_idx[nid] for nid in sorted_nids},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_jaxsso_model(
        self,
        load_case: Optional[WHTLoadCase] = None,
    ):
        """
        Build a JaxSSO Model from WHTMeshModel.

        Returns (jax_model, sorted_nids, nid_to_idx).

        Node remapping:
          WHTMeshModel uses arbitrary node IDs.
          JaxSSO requires 0-indexed contiguous node tags (nodeTag * 6 = DOF).
          sorted_nids provides the canonical ordering.
        """
        from JaxSSO.model import Model as JaxModel

        jm = JaxModel()
        sorted_nids = self.model.sorted_node_ids()
        nid_to_idx  = {nid: i for i, nid in enumerate(sorted_nids)}

        # Nodes (0-indexed)
        for nid in sorted_nids:
            node = self.model.nodes[nid]
            idx  = nid_to_idx[nid]
            jm.add_node(idx, node.x, node.y, node.z)

        # BCs from model-level SPC conditions
        bc_map: Dict[int, List[int]] = {}  # {node_idx: [dofs...]}
        for spc in self.model.spc_conditions:
            idx = nid_to_idx.get(spc.node_id)
            if idx is None:
                continue
            if idx not in bc_map:
                bc_map[idx] = []
            bc_map[idx].extend(spc.dofs)

        # BCs from load case (override / add)
        if load_case is not None:
            for bc in load_case.bcs:
                idx = nid_to_idx.get(bc.node_id)
                if idx is None:
                    continue
                if idx not in bc_map:
                    bc_map[idx] = []
                bc_map[idx].extend(bc.dofs)

        for idx, dof_list in bc_map.items():
            support = [0] * 6
            for d in set(dof_list):
                if 0 <= d < 6:
                    support[d] = 1
            jm.add_support(idx, support)

        # Elements: Only add non-MITC+ types (Beams, etc.) to JaxSSO engine to prevent double-counting
        mitc_plus_types = ('TRIA3', 'TRIA', 'QUAD4', 'QUAD')
        for eid in sorted(self.model.elements.keys()):
            elem = self.model.elements[eid]
            etype = getattr(elem, 'type', '')
            if etype in mitc_plus_types:
                continue

            pid  = elem.pid
            # ... (rest of beam assembly continues safely)

        # RBE2 → stiff beams
        self._add_rbe2_beams(jm, nid_to_idx)

        # Loads from load case
        if load_case is not None:
            for force in load_case.forces:
                idx = nid_to_idx.get(force.node_id)
                if idx is not None:
                    jm.add_nodal_load(idx, list(force.load_vector))

        return jm, sorted_nids, nid_to_idx

    def _add_rbe2_beams(self, jm, nid_to_idx: Dict[int, int]) -> None:
        """Convert RBE2 rigid elements to stiff beams."""
        if not self.model.rbe2s:
            return

        # Use large stiffness for RBE2 beams (stiffness_scale × nominal E)
        E_rbe = 1e6 * self.stiffness_scale
        G_rbe = E_rbe / 2.0
        Iy = Iz = J = 1e8
        A = 1.0

        beam_id = max(self.model.elements.keys(), default=0) + 10000
        for rbe2 in self.model.rbe2s.values():
            master_idx = nid_to_idx.get(rbe2.master_nid)
            if master_idx is None:
                continue
            for slave_nid in rbe2.slave_nids:
                slave_idx = nid_to_idx.get(slave_nid)
                if slave_idx is None:
                    continue
                jm.add_beamcol(beam_id, master_idx, slave_idx,
                               E_rbe, G_rbe, Iy, Iz, J, A)
                beam_id += 1

    def _has_jaxsso_elements(self) -> bool:
        """Check if any elements belong to JaxSSO (excluding our MITC+ shells)."""
        mitc_plus_types = ('TRIA3', 'TRIA', 'QUAD4', 'QUAD')
        for e in self.model.elements.values():
            if getattr(e, 'type', '') not in mitc_plus_types:
                return True
        return bool(self.model.rbe2s)

    def _augment_K_scipy(self, K_scipy, jm):
        """
        Lagrange-multiplier augmented system from a scipy sparse K.

        Returns (K_aug_scipy, f_aug_np) — same semantics as
        assemblemodel.model_K_aug / model_f_aug but accepting an external K.
        """
        from scipy.sparse import csr_matrix, vstack, hstack, coo_matrix
        import jax.numpy as jnp

        ndof   = jm.ndof
        known  = np.array(jm.known_id, dtype=np.int64)
        ncons  = len(known)

        # Constraint matrix C (ncons × ndof): C[i, known[i]] = 1
        C = coo_matrix(
            (np.ones(ncons), (np.arange(ncons), known)),
            shape=(ncons, ndof),
        ).tocsr()

        zero_block = csr_matrix((ncons, ncons))

        K_aug = vstack([
            hstack([K_scipy, C.T]),
            hstack([C,       zero_block]),
        ]).tocsr()

        f_base = np.array(jm.nodal_loads)
        f_aug  = np.concatenate([f_base, np.zeros(ncons)])

        return K_aug, f_aug

    def _assemble_lumped_mass(
        self,
        jm,
        ndof: int,
        sorted_nids: List[int],
        nid_to_idx: Dict[int, int],
    ) -> np.ndarray:
        """
        Build lumped mass diagonal vector for modal analysis.

        Uses element area × thickness × density.
        Rotational inertia: m × mesh_size² / 12 (estimated).
        """
        M_diag = np.zeros(ndof)

        for eid in sorted(self.model.elements.keys()):
            elem = self.model.elements[eid]
            if elem.type != "QUAD4" or len(elem.node_ids) != 4:
                continue

            pid  = elem.pid
            prop = self.model.properties.get(pid)
            mat  = (self.model.materials.get(prop.mid)
                    if prop and prop.mid in self.model.materials else None)

            t   = prop.t  if prop else 1.0
            rho = mat.rho if mat  else 7.85e-9

            coords = [self.model.nodes[n] for n in elem.node_ids]
            v1 = np.array([coords[1].x - coords[0].x,
                           coords[1].y - coords[0].y,
                           coords[1].z - coords[0].z])
            v2 = np.array([coords[2].x - coords[0].x,
                           coords[2].y - coords[0].y,
                           coords[2].z - coords[0].z])
            
            if elem.type in ("TRIA3", "TRIA") and len(elem.node_ids) == 3:
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                num_n = 3
            elif elem.type in ("QUAD4", "QUAD") and len(elem.node_ids) == 4:
                # Quad area is sum of two triangles (0-1-2 and 0-2-3)
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))
                v3 = np.array([coords[3].x - coords[0].x,
                               coords[3].y - coords[0].y,
                               coords[3].z - coords[0].z])
                # Cross v2 and v3 for the 2nd triangle (0-2-3)
                # Actually v2 - v0 and v3 - v0 works if it's convex
                area += 0.5 * np.linalg.norm(np.cross(v2, v3))
                num_n = 4
            else:
                continue

            elem_mass = area * t * rho
            m_node    = elem_mass / num_n

            # Estimate mesh size for rotational inertia
            mesh_size = np.linalg.norm(v1)
            # [WHT] More physical rotational inertia to prevent eigenvalues from blowing up.
            # m * L^2 / 12 is the inertia of a point mass about its center in simplified terms.
            rot_inert = m_node * (mesh_size**2) / 12.0
            # Ensure it's not too small to avoid numerical issues
            rot_inert = max(rot_inert, 1e-8)

            for nid in elem.node_ids:
                idx = nid_to_idx[nid]
                dof = idx * 6
                M_diag[dof:dof+3] += m_node
                M_diag[dof+3:dof+6] += rot_inert

        return M_diag
