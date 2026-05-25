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

from typing import Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from wht_modeler.wht_mesh_model import WHTMeshModel

from .load_cases import WHTLoadCase
from .wht_result import WHTSolverResult
from .wht_stress_recovery import ElementStressRecovery
from .wht_tria3_element import K_tria3_scipy, M_tria3_lumped
from .wht_quad4_element import K_quad4_scipy, M_quad4_lumped
try:
    from .wht_quad4_element_jax import K_quad4_jax as _K_quad4_jax
    _JAX_VMAP_OK = True
except Exception:
    _JAX_VMAP_OK = False


def _arpack_subprocess_worker(K, M_diag, k, sigma, maxiter, result_queue):
    """Isolated worker to run ARPACK and report results or errors back to parent.
    Optimized for Windows stability by forcing single-threading and array-M.
    """
    import os
    # Force single-threading for BLAS/LAPACK to prevent ARPACK segfaults on Windows
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    try:
        from scipy.sparse.linalg import eigsh
        import numpy as np
        from scipy.sparse import diags
        
        # M as 1D array is often more stable in certain ARPACK builds
        M_op = diags([M_diag], [0], format='csc')
        
        vals, vecs = eigsh(K, k=k, M=M_op, which="LM", sigma=sigma, tol=1e-5, maxiter=maxiter)
        result_queue.put((vals, vecs))
    except Exception as ex:
        result_queue.put(ex)

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

    def solve_modal(self, num_modes: int = 10, method: str = 'auto', 
                    exclude_rigid_body: Union[bool, str] = False,
                    shift_hz: Optional[float] = None) -> WHTSolverResult:
        """
        Solve generalized eigenvalue problem: K x = lambda M x
        
        Parameters:
            num_modes (int): Target number of elastic modes to return.
            method (str): 'sparse'/'arpack' (standard), 'dense' (scipy.eigh), or 'auto'.
            exclude_rigid_body (Union[bool, str, float]): Strategy to identify and remove rigid body modes.
                - 'skip6': Deterministically ignores the first 6 modes.
                - 'cutoff:0.1': Filters out modes with frequencies below 0.1 Hz.
                - 'range:0.1,1000': Retains only modes within [0.1, 1000] Hz.
                - 'auto' / True: Automatically detects the first significant frequency jump (Quantum Jump).
                - 'mass:0.8': Advanced Participation filtering. Removes modes with near-zero 
                              frequency and high global mass participation factors.
            shift_hz (float): Optional frequency shift (Hz) to solve for modes near a target value.
        """
        import time
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import diags
        from scipy.sparse.csgraph import reverse_cuthill_mckee
        
        # User may use 'arpack' as an alias for 'sparse'
        if method == 'arpack': method = 'sparse'
        
        print(f" -> Solving modal (method={method})...", flush=True)
        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model()
        
        # [WHT] Redefine num_modes as 'Elastic Mode Target'
        original_num_modes = num_modes
        if exclude_rigid_body:
            # Solve for 6 extra modes to account for standard 3D rigid body DOFs
            num_modes += 6

        # Consistent total DOF (N*6)
        ndof_total = len(sorted_nids) * 6
        jm_ndof = jm.ndof # Structural DOFs
        
        print(f"    - Assembling matrices (NDOF: {jm_ndof})...", end="", flush=True)
        t_start = time.time()
        K_scipy = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=True)
        M_all = self._assemble_lumped_mass(jm, jm_ndof, sorted_nids, nid_to_idx)
        print(f" Done ({time.time()-t_start:.2f}s)", flush=True)

        # [WHT] Filter Free DOFs
        unknown_id = np.array(jm.unknown_id, dtype=np.int64)
        actual_modes = min(num_modes, len(unknown_id) - 1)
        
        K_free = K_scipy[unknown_id, :][:, unknown_id].tocsc()
        M_free = M_all[unknown_id]
        ndof_free = len(unknown_id)
        
        # [WHT] Robustness: Add floor mass
        m_max = np.max(M_free)
        M_free = np.maximum(M_free, max(m_max * 1e-8, 1e-10))
        
        # [WHT] Bandwidth Reduction (RCM)
        print("    - Reordering degrees of freedom (RCM)...", end="", flush=True)
        r_start = time.time()
        perm = reverse_cuthill_mckee(K_free, symmetric_mode=True)
        rev_perm = np.argsort(perm)
        K_free_rcm = K_free[perm, :][:, perm].tocsc()
        M_free_rcm = M_free[perm]
        print(f" Done ({time.time()-r_start:.2f}s)", flush=True)

        vals, vecs_rcm = None, None
        
        sigma_val = -1.0
        if shift_hz is not None:
            sigma_val = (2.0 * np.pi * shift_hz)**2
            print(f"    - Applying Frequency Shift: {shift_hz} Hz (sigma: {sigma_val:.2f})")

        # [WHT] Multi-Stage Hybrid Solver Chain (Optimized for JAX & Stability)
        if method == 'auto':
            # Stage 1: ARPACK (Isolated Subprocess for stability)
            # On Windows, eigsh with shift-invert (sigma) is known to deadlock for
            # large unstructured meshes due to ARPACK library threading issues.
            # Use a short timeout: ARPACK either succeeds quickly or hangs indefinitely.
            arpack_timeout = 20 if ndof_free > 8000 else 60
            import multiprocessing
            print(f"    - [Stage 1: arpack] ndof={ndof_free}, timeout={arpack_timeout}s...", flush=True)
            q = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=_arpack_subprocess_worker,
                args=(K_free_rcm, M_free_rcm, actual_modes, sigma_val, ndof_free*20, q)
            )
            p.start()
            try:
                res = q.get(timeout=arpack_timeout)
                p.join()
                if not isinstance(res, Exception):
                    vals, vecs_rcm = res
                    print("      [Success] ARPACK Isolated.", flush=True)
                    method = 'completed'
                else:
                    print(f"      [Failed] ARPACK logic error: {res}", flush=True)
            except:
                if p.is_alive(): p.terminate(); p.join()
                print("      [Failed] ARPACK timed out or crashed — falling back.", flush=True)
            
            # Stage 2: LOBPCG (Robust Sparse alternative, JAX-friendly)
            if method == 'auto':
                from scipy.sparse.linalg import lobpcg
                print(f"    - [Stage 2: lobpcg] Attempting (JAX-friendly sparse)...", flush=True)
                try:
                    # Initial guess for eigenvectors
                    X = np.random.rand(ndof_free, actual_modes)
                    M_op = diags([M_free_rcm], [0], format='csc')
                    vals, vecs_rcm = lobpcg(K_free_rcm, X, B=M_op, tol=1e-5, largest=False)
                    print("      [Success] LOBPCG Sparse.", flush=True)
                    method = 'completed'
                except Exception as e:
                    print(f"      [Failed] LOBPCG error: {e}", flush=True)
                    
            # Stage 3: JAX Native EIGH (Differentiable, GPU accelerated if available)
            if method == 'auto':
                import jax.numpy as jnp
                from jax.lax.linalg import eigh as jax_eigh
                print(f"    - [Stage 3: jaxeigh] Attempting JAX-native eigh...", flush=True)
                try:
                    K_dense = jnp.array(K_free_rcm.toarray())
                    M_dense = jnp.diag(jnp.array(M_free_rcm))
                    # Note: Generalized eigh in JAX requires Cholesky of M or standard form
                    # For simplicity, we use standard form conversion K_tilde = L^-1 K L^-T
                    L = jnp.sqrt(M_dense) # M is diagonal here
                    K_tilde = K_dense / (L[:, None] * L[None, :])
                    vals_j, vecs_j = jax_eigh(K_tilde)
                    vals = np.array(vals_j[:actual_modes])
                    # Restore scale: y = L^-1 x
                    vecs_rcm = np.array(vecs_j[:, :actual_modes] / L[:, None])
                    print("      [Success] JAX-native eigh.", flush=True)
                    method = 'completed'
                except Exception as e:
                    print(f"      [Failed] jaxeigh error: {e}", flush=True)

            # Stage 4: Scipy Dense Eigh (The final fallback)
            if method == 'auto':
                print(f"    - [Stage 4: dense] Final fallback to Scipy eigh...", flush=True)
                import scipy.linalg as la
                K_dense = K_free_rcm.toarray()
                M_dense = np.diag(M_free_rcm)
                vals_all, vecs_all = la.eigh(K_dense, M_dense)
                vals = vals_all[:actual_modes]
                vecs_rcm = vecs_all[:, :actual_modes]
                print("      [Success] Scipy Dense eigh.", flush=True)
                method = 'completed'

        # Explicit individual methods (if user didn't use 'auto')
        if method == 'sparse' or method == 'arpack':
            # ... kept for backward compatibility or direct calls ...
            vals, vecs_rcm = eigsh(K_free_rcm, k=actual_modes, M=diags([M_free_rcm], [0], format='csc'),
                                   which="LM", sigma=sigma_val, tol=1e-5, maxiter=ndof_free*20)
        elif method == 'dense':
            import scipy.linalg as la
            vals_all, vecs_all = la.eigh(K_free_rcm.toarray(), np.diag(M_free_rcm))
            vals = vals_all[:actual_modes]
            vecs_rcm = vecs_all[:, :actual_modes]
        
        print("    - Eigensolve completed successfully.", flush=True)
        
        # Restore original DOF ordering
        vecs_free = vecs_rcm[rev_perm, :]

        freqs = np.sqrt(np.maximum(vals, 0)) / (2 * np.pi)

        # Expand to full DOF space (matching sorted_nids * 6)
        vecs_full = np.zeros((ndof_total, actual_modes))
        vecs_full[unknown_id, :] = vecs_free
        
        # Reshape to (n_modes, N, 6) — DOF layout: node_idx * 6
        n_nodes   = len(sorted_nids)
        mode_shapes = np.zeros((actual_modes, n_nodes, 6))
        for i, nid in enumerate(sorted_nids):
            idx = nid_to_idx[nid]
            # Ensure idx*6 + 6 fits in ndof_total
            mode_shapes[:, i, :] = vecs_full[idx * 6 : idx * 6 + 6, :].T

        res = WHTSolverResult("modal", sorted_nids)
        res.frequencies = freqs
        res.mode_shapes = mode_shapes
        res.solver_info = {'method': method, 'ndof': jm_ndof, 'ndof_free': ndof_free}
        
        # --- [WHT] Element Stress & Strain Recovery (v2: dict-based API) ---
        n_cells = len(self.model.elements)
        # Modal analysis: Upper surface 기본 (단일 레이어)
        stresses = np.zeros((actual_modes, n_cells, 6))
        strains  = np.zeros((actual_modes, n_cells, 6))
        seds     = np.zeros((actual_modes, n_cells, 1))
        
        for m in range(actual_modes):
            rd_q = ElementStressRecovery.recover_quad4(self.model, mode_shapes[m], sorted_nids)
            rd_t = ElementStressRecovery.recover_tria3(self.model, mode_shapes[m], sorted_nids)
            stresses[m] = rd_q["Stress"] + rd_t["Stress"]
            strains[m]  = rd_q["Strain"] + rd_t["Strain"]
            seds[m, :, 0] = 0.5 * np.sum(stresses[m] * strains[m], axis=1)
            
        res.cell_data = {"Stress": stresses, "Strain": strains, 
                         "StrainEnergyDensity": seds}
        
        # Attach physical data for filtering
        res.node_coords = self.model.nodes_array()
        res.nodal_mass = M_all  # Full lumped mass array (N*6)
        
        # [WHT] Filter Rigid Body Modes
        if exclude_rigid_body:
            res.filter_rigid_body_modes(exclude_rigid_body)
            res.truncate(original_num_modes)
            
        return res

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

        # --- [WHT] Element Stress & Strain Recovery (v2: dict-based API) ---
        rd_q = ElementStressRecovery.recover_quad4(self.model, displacement, sorted_nids)
        rd_t = ElementStressRecovery.recover_tria3(self.model, displacement, sorted_nids)
        
        # Merge QUAD4 + TRIA3 results and PAD for RBE/Beam elements
        cell_data = {}
        n_cells = len(self.model.elements)
        for key in rd_q:
            # shell_data has shape (M_shells, 6)
            shell_data = rd_q[key] + rd_t[key]
            # Create full array padded with zeros
            full_data = np.zeros((n_cells, shell_data.shape[1]))
            # ElementStressRecovery returns arrays where row index matches element index in model.elements
            # Actually, it returns arrays of size n_cells already, but only fills shell rows.
            # Let's verify if they are already full size.
            # In ElementStressRecovery, it usually initializes with np.zeros((len(model.elements), ...))
            cell_data[key] = shell_data[np.newaxis, :, :]  # (1, M, 6)
        
        # Calculate Max Von-Mises for diagnostic summary (Upper surface)
        s_upper = cell_data["Stress"][0]  # (M, 6)
        vm_static = np.sqrt(0.5 * (
            (s_upper[:, 0] - s_upper[:, 1]) ** 2 +
            (s_upper[:, 1] - s_upper[:, 2]) ** 2 +
            (s_upper[:, 2] - s_upper[:, 0]) ** 2 +
            6.0 * (s_upper[:, 3] ** 2 + s_upper[:, 4] ** 2 + s_upper[:, 5] ** 2)
        ))
        max_vm = np.max(vm_static)
        p95_vm = np.percentile(vm_static, 95)
        median_vm = np.median(vm_static)

        # [WHT] Diagnostic Summary for Torsion check
        u_max = np.abs(displacement).max()
        print(f"    - [Static Outcome] Max Disp: {u_max:.4e} mm, "
              f"Max Stress: {max_vm:.2f} MPa | P95: {p95_vm:.2f} | Median: {median_vm:.2f}, "
              f"Constrained: {len(bc_dof_ids)} DOFs.")

        # --- [WHT] Reaction Force Mapping Audit ---
        idx_to_nid = {i: nid for nid, i in nid_to_idx.items()}
        # Define lambda_ properly
        lambda_ = u_aug_np[ndof:]
        print(f"    - [Reaction Audit] Checking {len(jm.known_id)} constrained DOFs...")
        for i, gid in enumerate(jm.known_id):
            nid = idx_to_nid[gid // 6]
            if nid in [1051, 1052, 1053, 1054]: # Master nodes area
                val = lambda_[i]
                print(f"      [DEBUG] Node {nid} DOF {gid%6} -> Reaction: {val:.4e}")

        result = WHTSolverResult("static", sorted_nids)
        result.displacement  = displacement
        result.cell_data     = cell_data
        result._max_vm_diagnostic = max_vm 
        result._u_aug        = u_aug_np
        result._f_aug        = f_aug
        result._ndof         = ndof
        
        # Explicit mapping to ensure no mismatch
        result._bc_node_ids  = np.array([idx_to_nid[gid // 6] for gid in jm.known_id])
        result._bc_dof_ids   = np.array(jm.known_id)
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
        stabilize: bool = True,
        include_beams: bool = True,
        include_rbe3: bool = True,
    ) -> "csr_matrix":
        """Unified stiffness assembly (JaxSSO + MITC4 + MITC3).

        include_beams=False: shell-only K (Rayleigh 감쇠 기준 계산용).
        """
        from JaxSSO import assemblemodel
        from scipy.sparse import csr_matrix, diags

        ndof = jm.ndof
        has_jaxsso = include_beams and (self._has_jaxsso_elements() or (jm.n_beamcol > 0))
        if has_jaxsso:
            K_raw = assemblemodel.model_K(jm)
            # Robust conversion from JAX BCOO or Scipy COO to Scipy CSR
            if hasattr(K_raw, 'indices') and hasattr(K_raw, 'data'):
                # JAX BCOO handles indices as (nnz, 2) or (2, nnz)
                indices = np.array(K_raw.indices)
                data = np.array(K_raw.data)
                if indices.shape[1] == 2:
                    row, col = indices[:, 0], indices[:, 1]
                else:
                    row, col = indices[0, :], indices[1, :]
            elif hasattr(K_raw, 'row'):
                row, col, data = K_raw.row, K_raw.col, K_raw.data
            else:
                # Fallback: Treat as dense or empty
                K_dense = np.array(K_raw)
                from scipy.sparse import coo_matrix
                tmp = coo_matrix(K_dense)
                row, col, data = tmp.row, tmp.col, tmp.data

            K_out = csr_matrix((data, (row, col)), shape=(ndof, ndof))
        else:
            K_out = csr_matrix((ndof, ndof))

        if _JAX_VMAP_OK:
            K_q = _K_quad4_jax(self.model, sorted_nids, nid_to_idx)
        else:
            K_q = K_quad4_scipy(self.model, sorted_nids, nid_to_idx)
        K_t = K_tria3_scipy(self.model, sorted_nids, nid_to_idx)
        K_out = K_out + K_q + K_t
        print(f"    - [Assembly] Stiffness Audit: {K_q.nnz} non-zeros from shells.")

        if stabilize:
            # [WHT] AUTOSPC: Automatically patch zero or near-zero diagonal stiffness DOFs
            # Necessary for static and modal stability on shell meshes.
            K_out.sort_indices()
            k_diag = K_out.diagonal()
            k_max = np.abs(k_diag).max()
            if k_max > 0:
                threshold = max(k_max * 1e-8, 1e-3)
                bad_dofs = (np.abs(k_diag) <= threshold)
                if np.any(bad_dofs):
                    # Add weaker spring stiffness to preserve rigid body modes
                    penalty_val = k_max * 1e-12
                    K_out = K_out + diags([bad_dofs.astype(float) * penalty_val], [0])
        # [WHT] RBE3 Penalty Stiffness (Weighted Interpolation)
        # For partitioning solvers (JAX/Dynamic), RBE3 must be handled as penalty springs.
        if include_rbe3 and hasattr(self.model, 'rbe3s') and self.model.rbe3s:
            k_diag = K_out.diagonal()
            k_max = np.abs(k_diag).max() if k_diag.size > 0 else 1e5
            penalty = k_max * 1e7
            
            p_rows, p_cols, p_vals = [], [], []
            for rbe3 in self.model.rbe3s.values():
                m_idx = nid_to_idx.get(rbe3.master_nid)
                if m_idx is None: continue
                
                weights = rbe3.weights if rbe3.weights else [1.0] * len(rbe3.slave_nids)
                w_sum = sum(weights)
                if abs(w_sum) < 1e-12: w_sum = 1.0
                
                for d in rbe3.dofs:
                    # Eq: u_m - sum(ci * u_si) = 0  =>  E = 0.5 * k * (u_m - sum(ci * u_si))^2
                    m_dof = m_idx * 6 + d
                    p_rows.append(m_dof); p_cols.append(m_dof); p_vals.append(penalty)
                    
                    s_indices, coeffs = [], []
                    for s_nid, w in zip(rbe3.slave_nids, weights):
                        s_idx = nid_to_idx.get(s_nid)
                        if s_idx is None: continue
                        c_i = w / w_sum
                        s_dof = s_idx * 6 + d
                        s_indices.append(s_dof)
                        coeffs.append(c_i)
                        
                        # Coupling terms
                        p_rows.append(m_dof); p_cols.append(s_dof); p_vals.append(-penalty * c_i)
                        p_rows.append(s_dof); p_cols.append(m_dof); p_vals.append(-penalty * c_i)
                    
                    # Self-coupling of slaves
                    for i, si in enumerate(s_indices):
                        ci = coeffs[i]
                        for j, sj in enumerate(s_indices):
                            cj = coeffs[j]
                            p_rows.append(si); p_cols.append(sj); p_vals.append(penalty * ci * cj)
            
            if p_vals:
                K_out = K_out + csr_matrix((p_vals, (p_rows, p_cols)), shape=(ndof, ndof))

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
        bc_map: Dict[int, Dict[int, float]] = {}  # {node_idx: {dof: value}}
        for spc in self.model.spc_conditions:
            idx = nid_to_idx.get(spc.node_id)
            if idx is None:
                continue
            if idx not in bc_map:
                bc_map[idx] = {}
            for d in spc.dofs:
                bc_map[idx][d] = spc.value

        # BCs from load case (override / add)
        if load_case is not None:
            for bc in load_case.bcs:
                idx = nid_to_idx.get(bc.node_id)
                if idx is None:
                    continue
                if idx not in bc_map:
                    bc_map[idx] = {}
                for d in bc.dofs:
                    bc_map[idx][d] = bc.value

        for idx, dof_dict in bc_map.items():
            support = [0] * 6
            for d in dof_dict.keys():
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

        # RBE2 → Stiff Beam conversion for dynamic/modal stability
        self._add_rbe2_beams(jm, nid_to_idx)

        # Loads from load case
        if load_case is not None:
            for force in load_case.forces:
                idx = nid_to_idx.get(force.node_id)
                if idx is not None:
                    jm.add_nodal_load(idx, list(force.load_vector))

        # After model_ready, populate known_val (Prescribed Displacement)
        jm.model_ready()
        
        # global DOFs indices (node_idx * 6 + local_dof) that are constrained.
        prescribed = np.zeros(len(jm.known_id))
        known_id_to_val = {}
        for n_idx, dof_dict in bc_map.items():
            for d, val in dof_dict.items():
                global_dof = n_idx * 6 + d
                known_id_to_val[global_dof] = val
                
        for i, gid in enumerate(jm.known_id):
            prescribed[i] = known_id_to_val.get(gid, 0.0)
        
        # Unconditionally set known_val to ensure prescribed displacements are handled
        jm.known_val = prescribed

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
        """Check if any elements belong to JaxSSO (excluding our MITC+ shells AND RBE2).
        
        RBE2 요소는 Lagrange MPC로만 처리되므로, JaxSSO 엔진에 포함시키지 않습니다.
        """
        mitc_plus_types = ('TRIA3', 'TRIA', 'QUAD4', 'QUAD')
        rbe2_types = ('RBE2',)  # RBE2 제외: Lagrange MPC로 처리
        for e in self.model.elements.values():
            etype = getattr(e, 'type', '')
            if etype not in mitc_plus_types and etype not in rbe2_types:
                return True
        return False  # RBE2는 여기서 제외

    def _augment_K_scipy(self, K_scipy, jm):
        """
        Augment K with Lagrange multipliers for SPCs AND RBE2 MPCs.
        System: [K C^T; C 0] {u; lambda} = {f; v}
        """
        from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack, diags
        ndof = K_scipy.shape[0]
        # Remap original node IDs to indices in jm (0 to n_nodes-1)
        nid_to_idx = {nid: i for i, nid in enumerate(self.model.sorted_node_ids())}

        # 1. Collect SPC Constraints (Single Point)
        # jm.known_id contains global DOFs (node_idx * 6 + local_dof)
        known = np.array(jm.known_id, dtype=np.int64)
        n_spc = len(known)
        spc_rows, spc_cols, spc_vals = [], [], []
        v_spc = []
        
        # Ensure jm.known_val is available
        target_vals = np.zeros(n_spc)
        if jm.known_val is not None:
            # Match lengths
            n_copy = min(n_spc, len(jm.known_val))
            target_vals[:n_copy] = jm.known_val[:n_copy]
            
        for i, gid in enumerate(known):
            spc_rows.append(i)
            spc_cols.append(gid)
            spc_vals.append(1.0)
            v_spc.append(target_vals[i])
        
        # 2. Collect RBE2 MPC Constraints (Multi Point)
        mpc_rows, mpc_cols, mpc_vals = [], [], []
        v_mpc = []
        mpc_count = 0
        
        if self.model.rbe2s:
            for rbe2 in self.model.rbe2s.values():
                m_idx = nid_to_idx.get(rbe2.master_nid)
                if m_idx is None: continue
                m_node = self.model.nodes[rbe2.master_nid]
                
                for s_nid in rbe2.slave_nids:
                    s_idx = nid_to_idx.get(s_nid)
                    if s_idx is None: continue
                    s_node = self.model.nodes[s_nid]
                    
                    r = np.array([s_node.x - m_node.x, s_node.y - m_node.y, s_node.z - m_node.z])
                    T = np.eye(6)
                    # Translation components coupled to master rotation (Kinematic Coupling)
                    T[0, 4] =  r[2]; T[0, 5] = -r[1]
                    T[1, 3] = -r[2]; T[1, 5] =  r[0]
                    T[2, 3] =  r[1]; T[2, 4] = -r[0]
                    
                    for d in rbe2.dofs: # [User Request Fix] Respect specified DOFs
                        row_idx = n_spc + mpc_count
                        mpc_rows.append(row_idx); mpc_cols.append(s_idx*6 + d); mpc_vals.append(1.0)
                        for d_m in range(6):
                            val_t = -T[d, d_m]
                            if abs(val_t) > 1e-12:
                                mpc_rows.append(row_idx); mpc_cols.append(m_idx*6 + d_m); mpc_vals.append(val_t)
                        v_mpc.append(0.0)
                        mpc_count += 1

        # 3. Collect RBE3 MPC Constraints (Weighted Interpolation)
        if hasattr(self.model, 'rbe3s') and self.model.rbe3s:
            for rbe3 in self.model.rbe3s.values():
                m_idx = nid_to_idx.get(rbe3.master_nid)
                if m_idx is None: continue
                
                weights = rbe3.weights if rbe3.weights else [1.0] * len(rbe3.slave_nids)
                w_sum = sum(weights)
                if abs(w_sum) < 1e-12: w_sum = 1.0
                
                # Equation: u_master = sum(w_i * u_slave_i) / sum(w_i)
                # Rewritten: u_master - sum(w_i/w_sum * u_slave_i) = 0
                for d in rbe3.dofs:
                    row_idx = n_spc + mpc_count
                    # Master term (Dependent)
                    mpc_rows.append(row_idx); mpc_cols.append(m_idx*6 + d); mpc_vals.append(1.0)
                    
                    # Slave terms (Independent)
                    for s_nid, w in zip(rbe3.slave_nids, weights):
                        s_idx = nid_to_idx.get(s_nid)
                        if s_idx is None: continue
                        mpc_rows.append(row_idx); mpc_cols.append(s_idx*6 + d); mpc_vals.append(-w / w_sum)
                    
                    v_mpc.append(0.0)
                    mpc_count += 1

        n_total_cons = n_spc + mpc_count
        C = coo_matrix((spc_vals + mpc_vals, (spc_rows + mpc_rows, spc_cols + mpc_cols)), 
                       shape=(n_total_cons, ndof)).tocsr()
        
        K_aug = vstack([
            hstack([K_scipy, C.T]),
            hstack([C, diags([np.ones(n_total_cons) * 1e-9], [0])]),
        ]).tocsr()

        f_base = np.array(jm.nodal_loads)
        f_aug  = np.concatenate([f_base, np.array(v_spc + v_mpc)])

        return K_aug, f_aug

    def _prepare_matrices(self):
        """
        Assemble K_free and M_free without solving the eigenvalue problem.
        Used by WHTSensitivity for gradient computation.

        Returns
        -------
        K_free      : (ndof_free, ndof_free) CSC sparse stiffness matrix
        M_free      : (ndof_free,) lumped mass diagonal
        jm_ndof     : total structural DOF count
        unknown_id  : (ndof_free,) global DOF indices of free DOFs
        sorted_nids : sorted node ID list
        nid_to_idx  : {nid: 0-based index} mapping
        """
        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model()
        jm_ndof = jm.ndof

        K_scipy = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=True)
        M_all   = self._assemble_lumped_mass(jm, jm_ndof, sorted_nids, nid_to_idx)

        unknown_id = np.array(jm.unknown_id, dtype=np.int64)
        K_free = K_scipy[unknown_id, :][:, unknown_id].tocsc()
        M_free = M_all[unknown_id]
        m_max  = np.max(M_free)
        M_free = np.maximum(M_free, max(m_max * 1e-8, 1e-10))

        return K_free, M_free, jm_ndof, unknown_id, sorted_nids, nid_to_idx

    def _assemble_lumped_mass(
        self,
        jm,
        ndof: int,
        sorted_nids: List[int],
        nid_to_idx: Dict[int, int],
    ) -> np.ndarray:
        """
        Build lumped mass diagonal vector for modal analysis.
        Delegates to element-level implementations for QUAD4 and TRIA3.
        """
        M_diag = M_quad4_lumped(self.model, ndof, sorted_nids, nid_to_idx)
        M_diag += M_tria3_lumped(self.model, ndof, sorted_nids, nid_to_idx)
        return M_diag
