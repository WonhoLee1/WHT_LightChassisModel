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
        node_ids : None   → all BC nodes, returns (n_bc_nodes, 3) [Rx, Ry, Rz]
                   int    → single node, returns (3,)
                   list   → selected nodes, returns (len, 3)

        Returns
        -------
        np.ndarray — translational reaction forces [Rx, Ry, Rz] only.

        Note
        ----
        Sign convention: lambda_ = u_aug[ndof:] follows JaxSSO's Lagrange
        multiplier convention where lambda has the same sign as the applied
        load direction (i.e. sum(R[:, 2]) = -total_applied_Fz).
        For equilibrium check use: abs(sum(R[:, 2])) == abs(F_applied_total).
        Only BC-constrained nodes have non-zero reactions.
        Non-BC nodes return (0, 0, 0).
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
            return node_reactions.get(nid, np.zeros(6))[:3]

        if node_ids is None:
            # All BC nodes in sorted order
            bc_node_set = sorted(set(self._bc_node_ids.tolist()))
            return np.array([_get(n) for n in bc_node_set])
        elif isinstance(node_ids, int):
            return _get(node_ids)
        else:
            return np.array([_get(n) for n in node_ids])

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
            cell_data   = {},
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
