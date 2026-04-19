"""
wht_adapters.py
===============
WHT Universal FEM Result Converter — Adapter Layer

Converts solver-native output objects into the ``WHTResultData`` IR.

Classes
-------
BaseAdapter     : Abstract base. Subclass for every new solver.
JaxSSOAdapter   : Handles JaxSSO models (Static / Modal / Buckling).
JaxFEMAdapter   : Handles jax-fem problems (Static / Transient / Modal).
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Union, Any

import numpy as np

from .wht_models import (
    WHTMetadata,
    WHTResultData,
    WHTValidationError,
    WHTExportWarning,
)
from .wht_utils import (
    VTKCellType,
    merge_csr,
    node_dict_to_array,
    remap_connectivity,
    to_vtk_csr,
)


# ===========================================================================
# BaseAdapter
# ===========================================================================

class BaseAdapter(ABC):
    """
    Abstract base class for all solver adapters.

    Subclasses must implement ``convert()``.
    The ``validate()`` method performs post-conversion checks beyond the
    ``WHTResultData.__post_init__`` shape checks (monotonicity, sign, etc.).
    """

    @abstractmethod
    def convert(self, *args, **kwargs) -> WHTResultData:
        """Convert solver-native data to WHTResultData (IR)."""
        ...

    def validate(self, data: WHTResultData) -> None:
        """
        Post-conversion semantic validation.

        Checks
        ------
        - transient : time_values is strictly monotonically increasing.
        - modal     : all frequencies are positive.
        - buckling  : warns on non-positive load factors (non-physical modes).
        """
        atype = data.metadata.analysis_type

        if atype == "transient":
            if not np.all(np.diff(data.time_values) > 0):
                raise WHTValidationError(
                    "time_values must be strictly monotonically increasing "
                    "for transient analysis."
                )

        if atype == "modal":
            if np.any(data.time_values <= 0):
                raise WHTValidationError(
                    "Modal natural frequencies (time_values) must all be positive. "
                    "Negative or zero values indicate non-physical modes. "
                    "Check eigenvalue solver output."
                )

        if atype == "buckling":
            if np.any(data.time_values <= 0):
                warnings.warn(
                    "One or more buckling load factors (time_values) are <= 0. "
                    "These may indicate non-physical (tension-driven) buckling modes.",
                    WHTExportWarning,
                    stacklevel=3,
                )
                
    def _require_keys(self, d: dict, keys: list, analysis_type: str) -> None:
        missing = [k for k in keys if k not in d]
        if missing:
            raise WHTValidationError(
                f"{self.__class__.__name__} ({analysis_type}): "
                f"results dict is missing required keys: {missing}. "
                f"Got keys: {list(d.keys())}."
            )


# ===========================================================================
# JaxSSOAdapter
# ===========================================================================

class JaxSSOAdapter(BaseAdapter):
    """
    Adapter for JaxSSO models.

    Supported analysis types
    ------------------------
    "static"   : Linear static. Results: displacement.
    "modal"    : Natural frequency analysis. Results: mode shapes + frequencies.
    "buckling" : Linear buckling. Results: buckling mode shapes + load factors.

    Expected JaxSSO model attributes (confirmed via source survey)
    --------------------------------------------------------------
    model.nodes     : dict {node_id (int): [x, y, z]}
    model.quads     : dict {elem_id (int): [n0, n1, n2, n3]}   — shell elements
    model.beamcols  : dict {elem_id (int): [n0, n1]}            — beam/column elements
    model.truss     : dict {elem_id (int): [n0, n1]}            — truss elements

    Notes
    -----
    Node / element IDs are not assumed to be contiguous; they are sorted and
    remapped to 0-based indices before building the CSR mesh.

    Quad node ordering is assumed to follow VTK CCW convention.
    Verify with the JaxSSO source if results look inverted.
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def convert(
        self,
        model: Any,
        results: Optional[dict] = None,
        analysis_type: str = "static",
        metadata: Optional[WHTMetadata] = None,
    ) -> WHTResultData:
        """
        Supports both result extraction (Model -> WHTResultData) 
        and model construction (WHTMeshModel -> JaxSSO Model).
        """
        if model.__class__.__name__ == "WHTMeshModel":
            return self.to_native(model)
        
        # ... existing result conversion logic ...
        """
        Parameters
        ----------
        model : JaxSSO model object
        results : dict
            "static"   → {"u": np.ndarray (N, D)}
            "modal"    → {"vecs": np.ndarray (n_modes, N, D),
                          "freqs": np.ndarray (n_modes,)}   [Hz assumed]
            "buckling" → {"modes": np.ndarray (n_modes, N, D),
                          "load_factors": np.ndarray (n_modes,)}
        analysis_type : str
            "static" | "modal" | "buckling"
        metadata : WHTMetadata

        Returns
        -------
        WHTResultData
        """
        if analysis_type not in {"static", "modal", "buckling"}:
            raise WHTValidationError(
                f"JaxSSOAdapter does not support analysis_type='{analysis_type}'. "
                f"Valid options: 'static', 'modal', 'buckling'."
            )

        # --- Geometry (common to all analysis types) ---
        nodes, node_id_map = node_dict_to_array(model.nodes)
        connectivity, offsets, cell_types = self._build_mesh(model, node_id_map)

        # --- Results (type-specific) ---
        if analysis_type == "static":
            point_data, cell_data, field_data, time_values = \
                self._convert_static(results, node_id_map)
        elif analysis_type == "modal":
            point_data, cell_data, field_data, time_values = \
                self._convert_modal(results, node_id_map)
        else:  # buckling
            point_data, cell_data, field_data, time_values = \
                self._convert_buckling(results, node_id_map)

        data = WHTResultData(
            nodes=nodes,
            connectivity=connectivity,
            offsets=offsets,
            cell_types=cell_types,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
            time_values=time_values,
            metadata=metadata,
        )
        self.validate(data)
        return data

    def _extract_optional_cell_data(self, results: dict) -> dict:
        """
        results 딕셔너리에서 stress, strain 등 요소 데이터를 추출하여 
        cell_data 포맷에 맞게 변환합니다.
        """
        cell_data = {}
        for key in ["stress", "strain"]:
            if key in results:
                val = np.asarray(results[key], dtype=np.float64)
                # 단일 스텝(정적 해석 등) 데이터일 경우 시간 축(T) 추가
                if val.ndim == 2:
                    val = val[np.newaxis, :, :]
                # WHTResultData의 명명 규칙(Capitalize)에 맞춤
                cell_data[key.capitalize()] = val
        return cell_data

    # ------------------------------------------------------------------
    # Geometry extraction
    # ------------------------------------------------------------------

    def _build_mesh(
        self,
        model,
        node_id_map: dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Iterate through model.quads, model.beamcols, model.truss and
        build a single mixed CSR mesh.
        """
        from .wht_utils import to_vtk_csr, merge_csr, VTKCellType
        groups = []

        # Shell elements (Quad4 → VTK_QUAD = 9)
        if hasattr(model, "quads") and model.quads:
            q_conn_list = []
            for q in model.quads.values():
                # JaxSSO Quads use i_nodeTag, j_nodeTag, m_nodeTag, n_nodeTag
                q_conn_list.append([
                    node_id_map[q.i_nodeTag], node_id_map[q.j_nodeTag], 
                    node_id_map[q.m_nodeTag], node_id_map[q.n_nodeTag]
                ])
            groups.append(to_vtk_csr(np.array(q_conn_list), VTKCellType.QUAD))

        # Beam / column elements (Line2 → VTK_LINE = 3)
        if hasattr(model, "beamcols") and model.beamcols:
            b_conn_list = []
            for b in model.beamcols.values():
                b_conn_list.append([node_id_map[b.i_nodeTag], node_id_map[b.j_nodeTag]])
            groups.append(to_vtk_csr(np.array(b_conn_list), VTKCellType.LINE))

        # Truss elements (Line2 → VTK_LINE = 3)
        if hasattr(model, "truss") and model.truss:
            t_conn_list = []
            for t in model.truss.values():
                t_conn_list.append([node_id_map[t.i_nodeTag], node_id_map[t.j_nodeTag]])
            groups.append(to_vtk_csr(np.array(t_conn_list), VTKCellType.LINE))

        if not groups:
            raise WHTValidationError(
                "JaxSSO model has no recognized element groups "
                "(quads / beamcols / truss are all empty or missing)."
            )

        return merge_csr(groups)

    # ------------------------------------------------------------------
    # Result extraction (per analysis type)
    # ------------------------------------------------------------------

    def _convert_static(self, results: dict, node_id_map: dict):
        self._require_keys(results, ["u"], "static")
        u = np.asarray(results["u"], dtype=np.float64)   # (N, D)
        if u.ndim == 1:
            u = u[:, np.newaxis]
        # Add time axis → (1, N, D)
        point_data  = {"Displacement": u[np.newaxis, :, :]}
        cell_data   = self._extract_optional_cell_data(results)
        field_data  = {}
        time_values = np.array([0.0])
        return point_data, cell_data, field_data, time_values

    def _convert_modal(self, results: dict, node_id_map: dict):
        self._require_keys(results, ["vecs", "freqs"], "modal")
        vecs  = np.asarray(results["vecs"],  dtype=np.float64)  # (n_modes, N, D)
        freqs = np.asarray(results["freqs"], dtype=np.float64)  # (n_modes,) Hz

        if vecs.ndim == 2:
            # Single mode: (N, D) → (1, N, D)
            vecs = vecs[np.newaxis, :, :]
        if freqs.ndim == 0:
            freqs = freqs[np.newaxis]

        point_data  = {"Displacement": vecs}
        cell_data   = self._extract_optional_cell_data(results)
        field_data  = {}
        time_values = freqs
        return point_data, cell_data, field_data, time_values

    def _convert_buckling(self, results: dict, node_id_map: dict):
        self._require_keys(results, ["modes", "load_factors"], "buckling")
        modes        = np.asarray(results["modes"],        dtype=np.float64)
        load_factors = np.asarray(results["load_factors"], dtype=np.float64)

        if modes.ndim == 2:
            modes = modes[np.newaxis, :, :]
        if load_factors.ndim == 0:
            load_factors = load_factors[np.newaxis]

        point_data  = {"BucklingMode": modes}
        cell_data   = self._extract_optional_cell_data(results)
        field_data  = {"LoadFactor": load_factors}
        time_values = load_factors
        return point_data, cell_data, field_data, time_values

    # ------------------------------------------------------------------
    # Model Construction (WHTMeshModel -> JaxSSO Model)
    # ------------------------------------------------------------------

    def to_native(self, wht_model: WHTMeshModel, rbe2_stiffness_scale: float = 1000.0):
        """
        Converts WHTMeshModel to a native JaxSSO Model.
        Implements RBE2 via the Stiff-Spoke (Beam) method.
        """
        from JaxSSO.model import Model
        native_model = Model()

        # 1. Nodes
        for nid, node in wht_model.nodes.items():
            native_model.add_node(nid, node.x, node.y, node.z)

        # 2. Elements (Shells)
        for eid, elem in wht_model.elements.items():
            # Get property/material if available
            thickness = 1.0
            E_val = 2.1e5
            nu_val = 0.3
            if elem.pid in wht_model.properties:
                prop = wht_model.properties[elem.pid]
                thickness = prop.t
                if prop.mid in wht_model.materials:
                    mat = wht_model.materials[prop.mid]
                    E_val = mat.E
                    nu_val = mat.nu
            
            if "QUAD" in elem.type:
                native_model.add_quad(eid, *elem.node_ids, t=thickness, E=E_val, nu=nu_val)
            elif "TRIA" in elem.type:
                # JaxSSO might have add_tria, assuming for now
                if hasattr(native_model, "add_tria"):
                    native_model.add_tria(eid, *elem.node_ids, t=thickness, E=E_val, nu=nu_val)

        # 3. RBE2 (Stiff Spoke)
        # We use add_beamcol to connect Master to Slaves
        base_E = 2.1e5
        E_rigid = base_E * rbe2_stiffness_scale
        
        for rbe_id, rbe in wht_model.rbe2s.items():
            for j, slave_nid in enumerate(rbe.slave_nids):
                link_eid = 1000000 + rbe_id * 1000 + j
                # Very stiff cross-section properties
                native_model.add_beamcol(
                    link_eid, rbe.master_nid, slave_nid,
                    E=E_rigid, G=E_rigid/2.6, 
                    Iy=1e6, Iz=1e6, J=2e6, A=1e4
                )

        # 4. Supports (SPC)
        for spc in wht_model.spc_conditions:
            native_model.add_support(spc.node_id, list(spc.dofs))

        # 5. Loads (Note: JaxSSO nodal loads are 6-comp vectors)
        for load in wht_model.loads:
            native_model.add_nodal_load(load.node_id, list(load.load_vector))

        return native_model

    @staticmethod
    def solve_with_reactions(model):
        """
        Custom solver wrapper that captures Reaction Forces 
        from Lagrange multipliers.
        """
        from JaxSSO import assemblemodel, solver
        
        model.model_ready()
        K_aug = assemblemodel.model_K_aug(model)
        f_aug = assemblemodel.model_f_aug(model)
        ndof = model.get_dofs()
        
        # solve returns [u, lambda]
        u_aug = solver.sci_sparse_solve(K_aug, f_aug)
        
        u = u_aug[:ndof]
        lambdas = u_aug[ndof:] # Lagrange multipliers = Reaction Forces
        
        model.u = u
        
        # Map lambdas back to nodal reaction forces
        reactions = np.zeros(ndof)
        known_ids = model.known_id
        for i, kid in enumerate(known_ids):
            reactions[kid] = lambdas[i]
            
        return u, reactions.reshape(-1, 6)


# ===========================================================================
# JaxFEMAdapter
# ===========================================================================

class JaxFEMAdapter(BaseAdapter):
    """
    Adapter for jax-fem problems.

    Supported analysis types
    ------------------------
    "static"    : Linear static. Results: displacement field.
    "transient" : Time-domain dynamic. Results: displacement history.
    "modal"     : Eigenvalue analysis. Results: eigenvectors + natural frequencies.

    Expected jax-fem problem attributes (partially confirmed)
    ---------------------------------------------------------
    problem.mesh.node_coords : JAX DeviceArray (N, D) — confirmed
    problem.mesh.*           : Element connectivity layout — TBD (see TODO below)

    TODO (complete before implementing _extract_mesh)
    -------------------------------------------------
    [ ] Confirm field name for element connectivity in problem.mesh
        (candidates: cells, elements, connect, cell_connectivity)
    [ ] Confirm element connectivity layout: flat (K,) + offsets, or 2D (M, V)?
    [ ] Confirm VTK cell type for each jax-fem element type
        (e.g. HEX8 → VTKCellType.HEXAHEDRON = 12)
    [ ] Confirm eigvals units: rad²/s² assumed → convert to Hz
    [ ] Confirm time step storage: all-in-memory (T, N, D) or generator?
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def convert(
        self,
        problem,
        results: dict,
        analysis_type: str,
        metadata: WHTMetadata,
    ) -> WHTResultData:
        """
        Parameters
        ----------
        problem : jax-fem Problem object (must have problem.mesh.node_coords)
        results : dict
            "static"    → {"u": array (N, D)}
            "transient" → {"u": array (T, N, D), "t": array (T,)}
            "modal"     → {"eigvecs": array (n_modes, N, D),
                           "eigvals": array (n_modes,)}   [rad²/s² assumed]
        analysis_type : str
            "static" | "transient" | "modal"
        metadata : WHTMetadata
        """
        if analysis_type not in {"static", "transient", "modal"}:
            raise WHTValidationError(
                f"JaxFEMAdapter does not support analysis_type='{analysis_type}'. "
                f"Valid options: 'static', 'transient', 'modal'."
            )

        # jax_fem Problem convert mesh to a list internally, fetch the first one
        mesh_obj = problem.mesh[0] if type(problem.mesh) == list else problem.mesh

        # Convert JAX DeviceArray → NumPy
        nodes = np.asarray(mesh_obj.points, dtype=np.float64)
        if nodes.ndim == 1:
            raise WHTValidationError(
                "problem.mesh.points appears to be 1-D; expected (N, D)."
            )
        # Ensure 3-D coordinates (pad Z=0 for 2-D problems)
        if nodes.shape[1] == 2:
            nodes = np.hstack([nodes, np.zeros((nodes.shape[0], 1))])
        elif nodes.shape[1] != 3:
            raise WHTValidationError(
                f"points must have 2 or 3 spatial components, "
                f"got {nodes.shape[1]}."
            )

        connectivity, offsets, cell_types = self._extract_mesh(mesh_obj)

        if analysis_type == "static":
            point_data, cell_data, field_data, time_values = \
                self._convert_static(results)
        elif analysis_type == "transient":
            point_data, cell_data, field_data, time_values = \
                self._convert_transient(results)
        else:  # modal
            point_data, cell_data, field_data, time_values = \
                self._convert_modal(results)

        data = WHTResultData(
            nodes=nodes,
            connectivity=connectivity,
            offsets=offsets,
            cell_types=cell_types,
            point_data=point_data,
            cell_data=cell_data,
            field_data=field_data,
            time_values=time_values,
            metadata=metadata,
        )
        self.validate(data)
        return data

    # ------------------------------------------------------------------
    # Geometry extraction
    # ------------------------------------------------------------------

    def _extract_mesh(self, mesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract CSR mesh from a jax-fem mesh object.

        ⚠️  Implementation is provisional.
        The connectivity field name and layout need to be confirmed
        from the jax-fem source before this is used in production.

        Current strategy
        ----------------
        1. Try common field names for element connectivity.
        2. Assume 2-D layout (M, V) with uniform element type.
        3. Infer VTK cell type from V (nodes-per-element).
        """
        # Try known candidate field names
        conn_array = None
        for attr in ("cells", "elements", "connect", "cell_connectivity",
                     "connectivity", "elem_conn"):
            if hasattr(mesh, attr):
                conn_array = np.asarray(getattr(mesh, attr), dtype=np.int64)
                break

        if conn_array is None:
            raise NotImplementedError(
                "Could not find element connectivity in jax-fem mesh object. "
                "Inspected attributes: cells, elements, connect, cell_connectivity, "
                "connectivity, elem_conn. "
                "Please update _extract_mesh() after completing the data survey "
                "(see docs/jaxfem_data_survey.md)."
            )

        # Handle flat (K,) or 2-D (M, V) layouts
        if conn_array.ndim == 1:
            # Flat layout — cannot infer topology without offsets
            raise NotImplementedError(
                "jax-fem mesh connectivity appears to be a flat 1-D array. "
                "An offsets array is also needed. "
                "Update _extract_mesh() after the data survey."
            )

        # 2-D uniform (M, V) — infer VTK type from nodes-per-element
        M, V = conn_array.shape
        vtk_type = self._infer_vtk_type(V)
        return to_vtk_csr(conn_array, vtk_type)

    @staticmethod
    def _infer_vtk_type(nodes_per_elem: int) -> int:
        """Heuristic VTK type inference from nodes-per-element count."""
        mapping = {
            2:  VTKCellType.LINE,
            3:  VTKCellType.TRIANGLE,
            4:  VTKCellType.QUAD,        # could also be TETRA; prefer QUAD
            6:  VTKCellType.WEDGE,
            8:  VTKCellType.HEXAHEDRON,
            27: VTKCellType.BIQUADRATIC_HEXAHEDRON, # [WHT] Support HEX27
        }
        if nodes_per_elem not in mapping:
            raise WHTValidationError(
                f"Cannot infer VTK cell type for {nodes_per_elem} nodes/element. "
                f"Explicitly set vtk_type in _extract_mesh()."
            )
        return mapping[nodes_per_elem]

    # ------------------------------------------------------------------
    # Result extraction (per analysis type)
    # ------------------------------------------------------------------

    def _convert_static(self, results: dict):
        self._require_keys(results, ["u"], "static")
        u = np.asarray(results["u"], dtype=np.float64)   # (N, D)
        if u.ndim == 1:
            u = u[:, np.newaxis]
        point_data  = {"Displacement": u[np.newaxis, :, :]}  # (1, N, D)
        cell_data   = {}
        field_data  = {}
        time_values = np.array([0.0])
        return point_data, cell_data, field_data, time_values

    def _convert_transient(self, results: dict):
        self._require_keys(results, ["u", "t"], "transient")
        u = np.asarray(results["u"], dtype=np.float64)   # (T, N, D)
        t = np.asarray(results["t"], dtype=np.float64)   # (T,)
        if u.ndim == 2:
            u = u[:, :, np.newaxis]                      # (T, N, 1)
        point_data  = {"Displacement": u}
        cell_data   = {}
        field_data  = {}
        time_values = t
        return point_data, cell_data, field_data, time_values

    def _convert_modal(self, results: dict):
        self._require_keys(results, ["eigvecs", "eigvals"], "modal")
        # Ensure numerical safety with np.nan_to_num
        eigvecs = np.nan_to_num(np.asarray(results["eigvecs"], dtype=np.float64), nan=0.0)  # (n_modes, N, D)
        eigvals = np.nan_to_num(np.asarray(results["eigvals"], dtype=np.float64), nan=0.0)  # (n_modes,) rad²/s²

        if eigvecs.ndim == 2:
            eigvecs = eigvecs[np.newaxis, :, :]

        # rad²/s² → Hz (Absolute to handle potential near-zero negative modes)
        freqs = np.sqrt(np.abs(eigvals)) / (2.0 * np.pi)

        point_data  = {"Displacement": eigvecs}
        cell_data   = {}
        field_data  = {}
        time_values = freqs
        return point_data, cell_data, field_data, time_values
