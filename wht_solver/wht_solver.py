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
        print(f"    [DBG] model built: {len(sorted_nids)} nodes", flush=True)
        jm.model_ready()
        print(f"    [DBG] model_ready done, ndof={jm.ndof}", flush=True)

        ndof = jm.ndof

        # JaxSSO K: skip assembly when no QUAD4/beam elements (pure TRIA3 mesh)
        if self._has_jaxsso_elements():
            K_bcoo = assemblemodel.model_K(jm)
            K_scipy = csr_matrix(
                (np.array(K_bcoo.data),
                 (np.array(K_bcoo.indices[:, 0]),
                  np.array(K_bcoo.indices[:, 1]))),
                shape=(ndof, ndof),
            )
        else:
            K_scipy = csr_matrix((ndof, ndof))
        print(f"    [DBG] K_jaxsso done (has_jax={self._has_jaxsso_elements()})", flush=True)

        # Add TRIA3 stiffness contribution (skipped if no TRIA3 elements)
        K_scipy = K_scipy + K_tria3_scipy(self.model, sorted_nids, nid_to_idx)
        print(f"    [DBG] K_tria3 added, nnz={K_scipy.nnz}", flush=True)

        # Lumped mass matrix (diagonal): QUAD4 + TRIA3
        M_diag = self._assemble_lumped_mass(jm, ndof, sorted_nids, nid_to_idx)
        M_diag += M_tria3_lumped(self.model, ndof, sorted_nids, nid_to_idx)
        print(f"    [DBG] M_diag assembled, min={M_diag.min():.3e} max={M_diag.max():.3e}", flush=True)

        # Free DOF indices
        known_id = jm.known_id
        all_ids  = np.arange(ndof)
        unknown_id = np.setdiff1d(all_ids, known_id)
        print(f"    [DBG] unknown_id: {len(unknown_id)} free DOFs", flush=True)

        K_free = K_scipy[unknown_id, :][:, unknown_id]
        M_free = M_diag[unknown_id]
        print(f"    [DBG] K_free shape={K_free.shape}, M_free min={M_free.min():.3e}", flush=True)

        actual_modes = min(num_modes, len(unknown_id) - 1)

        # Force exact symmetry to prevent ARPACK/SuperLU C-level crashes (floating point noise)
        K_free = (K_free + K_free.transpose()) / 2.0
        
        # [CRITICAL] Clean up sparse structure. 
        # Unsorted indices after transpose addition cause SuperLU to SegFault silently.
        K_free.sum_duplicates()
        K_free.eliminate_zeros()
        K_free.sort_indices()

        # [WHT] AUTOSPC: Automatically patch zero or near-zero diagonal stiffness DOFs
        # Prevents SuperLU zero-pivot crashes on unstructured TRIA3_FREE meshes or hanging nodes.
        k_diag = K_free.diagonal()
        k_max = np.abs(k_diag).max()
        k_min = np.abs(k_diag).min()
        
        # Dynamic relative threshold to reliably catch weak drilling DOFs
        threshold = max(k_max * 1e-10, 1e-2)
        bad_dofs = (np.abs(k_diag) <= threshold)
        
        print(f"    [DBG] K_free diag min={k_min:.3e}, max={k_max:.3e}, threshold={threshold:.3e}", flush=True)
        if np.any(bad_dofs):
            num_bad = int(np.sum(bad_dofs))
            print(f"    [DBG] AUTOSPC: Stabilizing {num_bad} near-zero stiffness DOFs.", flush=True)
            K_free = K_free + diags([bad_dofs.astype(float) * k_max * 1e-4], [0])
            M_free = M_free + bad_dofs.astype(float) * 1e-4
            
        # [WHT] Explicitly cast to CSC format to prevent C-level SuperLU / ARPACK crashes.
        K_free_csc = K_free.tocsc()
        K_free_csc.sort_indices()
        
        # [WHT] Bandwidth Reduction (RCM Reordering)
        # Unstructured meshes (TRIA3_FREE) produce massive sparse bandwidth.
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        print("    [DBG] Reordering matrix via RCM to minimize bandwidth...", flush=True)
        perm = reverse_cuthill_mckee(K_free_csc, symmetric_mode=True)
        rev_perm = np.argsort(perm)

        K_free_rcm = K_free_csc[perm, :][:, perm].tocsc()
        M_free_rcm = M_free[perm]  # 1D array reordered

        # [WHT] Solve Generalized Eigenvalue Problem (Kx = lambda Mx)
        # Using Generalized form directly is MUCH more stable than Transforming to Standard form
        # when M is diagonal but has very small values (shell rotational inertia).
        print(f"    [DBG] calling eigsh k={actual_modes}... (Shift-Invert, sigma=-0.1)", flush=True)
        
        # Scipy eigsh handles diagonal M efficiently when passed as a 1D array or diags.
        # sigma=-0.1 ensures we find the lowest positive frequencies including rigid body modes.
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

        # Build K with TRIA3 contribution, then augment with BC (Lagrange multiplier)
        K_tria = K_tria3_scipy(self.model, sorted_nids, nid_to_idx)
        has_jax = self._has_jaxsso_elements()
        if K_tria.nnz > 0 or not has_jax:
            from scipy.sparse import csr_matrix as _csr
            if has_jax:
                K_bcoo = assemblemodel.model_K(jm)
                K_base = _csr(
                    (np.array(K_bcoo.data),
                     (np.array(K_bcoo.indices[:, 0]),
                      np.array(K_bcoo.indices[:, 1]))),
                    shape=(ndof, ndof),
                ) + K_tria
            else:
                K_base = _csr((ndof, ndof)) + K_tria
            K_aug, f_aug = self._augment_K_scipy(K_base, jm)
        else:
            K_aug = assemblemodel.model_K_aug(jm)
            f_aug = assemblemodel.model_f_aug(jm)

        # Solve: u_aug = [u_free (ndof,), lambda (n_bc,)]
        u_aug = jaxsso_solver.sci_sparse_solve(K_aug, f_aug)
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
        s_q, e_q = ElementStressRecovery.recover_quad4(self.model, displacement, sorted_nids)
        s_t, e_t = ElementStressRecovery.recover_tria3(self.model, displacement, sorted_nids)
        s_static = s_q + s_t
        e_static = e_q + e_t
        sed_static = 0.5 * np.sum(s_static * e_static, axis=1, keepdims=True)

        result = WHTSolverResult("static", sorted_nids)
        result.displacement  = displacement
        result.cell_data     = {"Stress": s_static[np.newaxis, :, :], 
                                "Strain": e_static[np.newaxis, :, :],
                                "StrainEnergyDensity": sed_static[np.newaxis, :, :]}
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

        # Elements
        for eid in sorted(self.model.elements.keys()):
            elem = self.model.elements[eid]
            pid  = elem.pid
            prop = self.model.properties.get(pid)
            mat  = (self.model.materials.get(prop.mid)
                    if prop and prop.mid in self.model.materials else None)

            t  = prop.t  if prop else 1.0
            E  = mat.E   if mat  else 1000.0
            nu = mat.nu  if mat  else 0.3

            remapped = [nid_to_idx[n] for n in elem.node_ids]

            if elem.type == "QUAD4" and len(remapped) == 4:
                jm.add_quad(eid, *remapped, t, E, nu)
            elif elem.type == "BEAM2" and len(remapped) == 2:
                # Stiff beam for RBE2 or regular beam
                G  = E / (2 * (1 + nu))
                A  = 1.0
                Iy = Iz = 1e6
                J  = 1e6
                jm.add_beamcol(eid, remapped[0], remapped[1], E, G, Iy, Iz, J, A)

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
        """Return True if the model contains QUAD4 or BEAM2 elements handled by JaxSSO."""
        return any(
            getattr(e, 'type', '') in ('QUAD4', 'BEAM2')
            for e in self.model.elements.values()
        )

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
            v2 = np.array([coords[3].x - coords[0].x,
                           coords[3].y - coords[0].y,
                           coords[3].z - coords[0].z])
            area      = np.linalg.norm(np.cross(v1, v2))
            elem_mass = area * t * rho
            m_node    = elem_mass / 4.0

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
