"""
wht_result.py
=============
WHT FEM Framework — Solver Result Container

WHTSolverResult stores displacements, modal results, and reaction forces.
Reaction forces are extracted from JaxSSO Lagrange multipliers (u_aug[ndof:]).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from wht_modeler.wht_mesh_model import WHTMeshModel
    from wht_converter.wht_models import WHTMetadata


class WHTSolverResult:
    """
    FEM solver result container.

    Supports static (displacement + reaction force) and modal
    (frequencies + mode shapes) analysis types.

    Reaction forces use JaxSSO Lagrange multipliers directly:
        u_aug = [u_free (ndof,), lambda (n_bc,)]
        lambda = reaction forces at BC-constrained DOFs
    """

    def __init__(
        self,
        analysis_type: str,
        node_ids: List[int],                    # sorted node IDs
    ):
        self.analysis_type = analysis_type
        self.node_ids      = list(node_ids)
        self.n_nodes       = len(node_ids)

        # Static result
        self.displacement: Optional[np.ndarray] = None   # (N, 6)

        # Modal result
        self.frequencies:  Optional[np.ndarray] = None   # (n_modes,)
        self.mode_shapes:  Optional[np.ndarray] = None   # (n_modes, N, 6)

        # Lagrange multiplier storage (reaction force source)
        self._u_aug:       Optional[np.ndarray] = None   # (ndof + n_bc,)
        self._ndof:        int                  = 0
        self._bc_dof_ids:  Optional[np.ndarray] = None   # global DOF indices for BC
        self._bc_node_ids: Optional[np.ndarray] = None   # node IDs for each BC DOF

    # ------------------------------------------------------------------
    # Reaction force API
    # ------------------------------------------------------------------

    def reaction_force(
        self,
        node_ids: Union[int, List[int], None] = None,
    ) -> np.ndarray:
        """
        Return reaction forces from Lagrange multipliers (u_aug[ndof:]).

        Parameters
        ----------
        node_ids : None   → all BC nodes, returns (n_bc_nodes, 6) [Fx, Fy, Fz, Mx, My, Mz]
                   int    → single node, returns (6,)
                   list   → selected nodes, returns (len, 6)

        Returns
        -------
        np.ndarray — nodal reaction forces and moments [Fx, Fy, Fz, Mx, My, Mz].
        """
        if self._u_aug is None:
            raise RuntimeError("No augmented solution stored — solve_static() first.")

        lambda_ = self._u_aug[self._ndof:]   # (n_bc,) Lagrange multipliers

        # Rebuild per-node reaction vector from BC DOF mapping
        node_reactions: Dict[int, np.ndarray] = {}
        for i, (bc_dof, bc_nid) in enumerate(
            zip(self._bc_dof_ids, self._bc_node_ids)
        ):
            if bc_nid not in node_reactions:
                node_reactions[bc_nid] = np.zeros(6)
            local_dof = bc_dof % 6
            if i < len(lambda_):
                node_reactions[bc_nid][local_dof] = float(lambda_[i])

        def _get(nid: int) -> np.ndarray:
            return node_reactions.get(nid, np.zeros(6))

        if node_ids is None:
            # All BC nodes in sorted order
            bc_node_set = sorted(set(self._bc_node_ids.tolist()))
            return np.array([_get(n) for n in bc_node_set])
        elif isinstance(node_ids, int):
            return _get(node_ids)
        else:
            return np.array([_get(n) for n in node_ids])

    # ------------------------------------------------------------------
    # Modal Post-processing
    # ------------------------------------------------------------------

    def filter_rigid_body_modes(self, threshold: Union[float, str, bool] = 'auto'):
        """
        Filters out rigid body modes based on various physical and statistical criteria.
        
        Supported Methods (pass via threshold):
            - 'skip6'        : Explicitly ignores the first 6 modes (3D free-free standard).
            - 'cutoff:0.1'   : Removes all modes with frequency < 0.1 Hz.
            - 'range:0.1,500': Retains only modes within the specified frequency band.
            - 'auto' / True  : Statistical 'jump' detection; finds the first elastic mode gap.
            - 'mass:0.8'     : Effective Mass Participation method. Identifies modes 
                               with high rigid-body participation and near-zero frequency.
            - False / 'none' : No filtering.
        """
        if self.analysis_type != "modal" or self.frequencies is None:
            return
        
        n_orig = len(self.frequencies)
        if n_orig < 1: return

        method = str(threshold).lower()
        if method == 'none' or threshold is False:
            return
            
        keep_indices = np.arange(n_orig)
        
        if method == 'skip6':
            keep_indices = np.arange(6, n_orig) if n_orig > 6 else np.array([], dtype=int)
            
        elif method == 'auto' or threshold is True:
            # Look for a jump in frequencies
            if n_orig > 1:
                diffs = np.zeros(min(n_orig - 1, 10))
                for i in range(len(diffs)):
                    diffs[i] = self.frequencies[i+1] - self.frequencies[i]
                
                found_jump = -1
                for i in range(len(diffs)):
                    if (diffs[i] > 0.5 and self.frequencies[i] < 0.5) or (i == 5 and diffs[i] > 0.1):
                        found_jump = i
                
                if found_jump != -1:
                    keep_indices = np.arange(found_jump + 1, n_orig)
                else:
                    keep_indices = np.where(self.frequencies >= 0.05)[0]
                    
        elif method.startswith('cutoff:'):
            val = float(method.split(':')[1])
            keep_indices = np.where(self.frequencies >= val)[0]
            
        elif method.startswith('range:'):
            vals = [float(v) for v in method.split(':')[1].split(',')]
            f_min, f_max = vals[0], vals[1]
            keep_indices = np.where((self.frequencies >= f_min) & (self.frequencies <= f_max))[0]
            
        elif method.startswith('mass'):
            target_ratio = 0.8
            if ':' in method:
                target_ratio = float(method.split(':')[1])
            
            # Use Mass Participation to identify rigid modes
            # Rigid modes usually have high participation (>0.1) and freq < 0.1 Hz
            eff_mass, total_mass = self.calculate_effective_mass()
            if eff_mass is not None:
                # Mode is likely rigid if it has significant participation in global DOFs and freq is low
                is_rigid = []
                for i in range(n_orig):
                    # Check if mode i is "rigid" (high mass participation AND low frequency)
                    # Sum of ratios across 6 DOFs
                    participation_sum = np.sum(eff_mass[i]) / total_mass.sum()
                    if participation_sum > 0.01 and self.frequencies[i] < 0.2:
                        is_rigid.append(True)
                    else:
                        is_rigid.append(False)
                
                keep_indices = np.where(~np.array(is_rigid))[0]
            else:
                # Fallback to auto if mass data is missing
                print("    - [Warning] Mass data missing for 'mass' filter. Falling back to 'auto'.")
                return self.filter_rigid_body_modes('auto')
        else:
            # Default to float cutoff if possible
            try:
                val = float(threshold)
                keep_indices = np.where(self.frequencies >= val)[0]
            except:
                pass

        n_now = len(keep_indices)
        if n_now == n_orig:
            return

        print(f"    - [Post] Filtered {n_orig - n_now} rigid body modes (method: {method}).")

        self.frequencies = self.frequencies[keep_indices]
        if self.mode_shapes is not None:
            self.mode_shapes = self.mode_shapes[keep_indices]
            
        if hasattr(self, 'cell_data') and self.cell_data:
            for key in self.cell_data:
                data = self.cell_data[key]
                if isinstance(data, np.ndarray) and data.ndim >= 1 and data.shape[0] == n_orig:
                    self.cell_data[key] = data[keep_indices]

    def calculate_effective_mass(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calculates Modal Participation Factors (L) and Effective Mass (meff) for each mode.
        
        Requires:
            - res.node_coords: (N, 3) NumPy float array
            - res.nodal_mass:  (N*6,) Lumped mass array
        
        Mathematical definition:
            1. Find CG = (sum m_i * r_i) / (sum m_i)
            2. Build R matrix mapping CG displacement to nodal DOFs (including rotation).
            3. Participation Factor L = phi^T * M * R
            4. Effective Mass meff = L^2 (Assuming mass-normalized modes)
        
        Returns:
            eff_mass (np.ndarray): (n_modes, 6) Effective mass per direction [X, Y, Z, Rx, Ry, Rz]
            total_mass (np.ndarray): (6,) Total physical mass/inertia components at CG
        """
        if not hasattr(self, 'node_coords') or not hasattr(self, 'nodal_mass'):
            return None, None
        if self.mode_shapes is None:
            return None, None
            
        n_modes, n_nodes, _ = self.mode_shapes.shape
        M = self.nodal_mass.reshape(n_nodes, 6) # (N, 6)
        coords = self.node_coords # (N, 3)
        
        # 1. Find Center of Gravity (CG)
        total_m = np.sum(M[:, :3], axis=0) # Total mass in X, Y, Z
        cg = np.sum(M[:, :3] * coords, axis=0) / total_m
        
        # 2. Build Rigid Body Matrix R (N*6, 6)
        # R relates CG 6-DOF to Nodal 6-DOFs
        R = np.zeros((n_nodes, 6, 6))
        
        # Identity parts for Translation (0:3, 0:3) and Rotation (3:6, 3:6)
        I3 = np.eye(3)
        R[:, 0:3, 0:3] = I3
        R[:, 3:6, 3:6] = I3
        
        # Skew-symmetric part for r (relates CG rotation to nodal translation)
        # nodal_u_trans = cg_u_trans + cg_u_rot x r
        #               = cg_u_trans - r x cg_u_rot
        # Matrix form: u_node = [I, -skew(r)] * [u_cg, rot_cg]^T
        r = coords - cg
        R[:, 0, 4] = r[:, 2]
        R[:, 0, 5] = -r[:, 1]
        R[:, 1, 3] = -r[:, 2]
        R[:, 1, 5] = r[:, 0]
        R[:, 2, 3] = r[:, 1]
        R[:, 2, 4] = -r[:, 0]
            
        # 3. Calculate Participation Factors L = phi^T * M * R
        # phi is (n_modes, n_nodes, 6)
        L = np.zeros((n_modes, 6))
        for m in range(n_modes):
            # (n_nodes, 6) * (n_nodes, 6) -> sum
            phi_m = self.mode_shapes[m]
            # Element-wise multiply then sum
            L[m] = np.sum(phi_m[:, :, np.newaxis] * M[:, :, np.newaxis] * R, axis=(0, 1))
            
        # 4. Effective Mass E = L^2 / (phi^T M phi)
        # Assuming mass-normalized modes (phi^T M phi = 1)
        eff_mass = L**2
        
        # 5. Total physical mass matrix at CG (diagonal 6x6)
        total_mass_cg = np.zeros(6)
        total_mass_cg[:3] = total_m
        # Moment of Inertia (approx)
        for i in range(n_nodes):
            r = coords[i] - cg
            total_mass_cg[3] += M[i, 0] * (r[1]**2 + r[2]**2)
            total_mass_cg[4] += M[i, 1] * (r[0]**2 + r[2]**2)
            total_mass_cg[5] += M[i, 2] * (r[0]**2 + r[1]**2)
            
        return eff_mass, total_mass_cg

    def truncate(self, n: int):
        """Truncate the result set to the first n modes."""
        if self.frequencies is not None and len(self.frequencies) > n:
            n_orig = len(self.frequencies)
            self.frequencies = self.frequencies[:n]
            if self.mode_shapes is not None:
                self.mode_shapes = self.mode_shapes[:n]
            if hasattr(self, 'cell_data') and self.cell_data:
                for key in self.cell_data:
                    data = self.cell_data[key]
                    if isinstance(data, np.ndarray) and data.ndim >= 1 and data.shape[0] == n_orig:
                        self.cell_data[key] = data[:n]
            print(f"    - [Post] Truncated results to {n} elastic modes.")

    # ------------------------------------------------------------------
    # Conversion to wht_converter IR
    # ------------------------------------------------------------------

    def to_wht_result_data(
        self,
        metadata: "WHTMetadata",
        mesh_model: Optional["WHTMeshModel"] = None,
    ):
        """
        Convert to WHTResultData for ParaView export.

        mesh_model : if provided, geometry is taken from the model
                     (handles post-optimization updated coordinates).
                     If None, geometry arrays must be set externally via
                     _nodes / _connectivity etc. (advanced use).
        """
        from wht_converter.wht_models import WHTResultData

        if mesh_model is None:
            raise ValueError("mesh_model is required for geometry source.")

        # Get base geometry IR
        base_rd = mesh_model.to_wht_result_data(metadata)

        # Attach result arrays
        point_data: Dict[str, np.ndarray] = {}
        field_data: Dict[str, np.ndarray] = {}

        if self.analysis_type == "static" and self.displacement is not None:
            # shape: (T=1, N, 6)
            disp_6dof = self.displacement[np.newaxis, :, :]   # (1, N, 6)
            point_data["Displacement"] = disp_6dof[:, :, :3]
            if disp_6dof.shape[2] == 6:
                point_data["Rotation"] = disp_6dof[:, :, 3:]

        if self.analysis_type == "modal" and self.mode_shapes is not None:
            # mode_shapes: (n_modes, N, 6) → time axis = mode index
            point_data["ModeShape"]      = self.mode_shapes[:, :, :3]
            if self.mode_shapes.shape[2] == 6:
                point_data["ModeRotation"] = self.mode_shapes[:, :, 3:]
            field_data["Frequency_Hz"]   = self.frequencies           # (T,)

        return WHTResultData(
            nodes       = base_rd.nodes,
            connectivity= base_rd.connectivity,
            offsets     = base_rd.offsets,
            cell_types  = base_rd.cell_types,
            node_sets   = base_rd.node_sets,
            element_sets= base_rd.element_sets,
            point_data  = point_data,
            cell_data   = self.cell_data if self.cell_data is not None else {},
            field_data  = field_data,
            time_values = (self.frequencies if self.analysis_type == "modal"
                           else np.array([0.0])),
            metadata    = metadata,
        )

    def __repr__(self) -> str:
        if self.analysis_type == "modal":
            nm = len(self.frequencies) if self.frequencies is not None else 0
            return f"WHTSolverResult(modal, {nm} modes, {self.n_nodes} nodes)"
        return f"WHTSolverResult(static, {self.n_nodes} nodes)"
