"""
wht_mesh_model.py
=================
WHT FEM Framework — Pre-processing Data Container

WHTMeshModel stores all FEM entities (nodes, elements, sets, BCs, loads)
and provides conversion to wht_converter IR (WHTResultData).

Dependency chain: wht_modeler → wht_converter (geometry IR only)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .wht_entities import (
    WHTNode, WHTElement, WHTNodeSet, WHTElemSet,
    WHTRBE2, WHTRBE3, WHTProperty, WHTMaterial,
    WHTSPCEntry, WHTLoadEntry,
)

# VTK cell type constants
_VTK_TYPE = {"QUAD4": 9, "TRIA3": 5, "BEAM2": 3, "TETRA4": 10, "HEXA8": 12}


class WHTMeshModel:
    """
    FEM pre-processing data container.

    Stores geometry, sets, properties, materials, BCs, and loads.
    IO readers populate this; users set BCs/loads on top.
    """

    def __init__(self, name: str = "WHT"):
        self.name = name

        # Geometry
        self.nodes:    Dict[int, WHTNode]    = {}
        self.elements: Dict[int, WHTElement] = {}

        # Sets
        self.node_sets: Dict[int, WHTNodeSet] = {}
        self.elem_sets: Dict[int, WHTElemSet] = {}

        # Rigid elements
        self.rbe2s: Dict[int, WHTRBE2] = {}
        self.rbe3s: Dict[int, WHTRBE3] = {}

        # Properties / Materials
        self.properties: Dict[int, WHTProperty] = {}
        self.materials:  Dict[int, WHTMaterial]  = {}

        # BCs and Loads
        self.spc_conditions: List[WHTSPCEntry] = []
        self.loads:          List[WHTLoadEntry] = []
        
        # Load Cases (Multiple Steps from external solvers)
        self.load_cases: List['WHTLoadCase'] = []

        # Internal geometry bookkeeping for SET_GENERAL
        self._boxes: Dict[int, np.ndarray] = {}   # {bid: [xmin,xmax,ymin,ymax,zmin,zmax]}
        self._parts: Dict[int, List[int]]  = {}   # {pid: [eid, ...]}

    # ------------------------------------------------------------------
    # Node / Element CRUD
    # ------------------------------------------------------------------

    def add_node(self, nid: int, x: float, y: float, z: float) -> None:
        self.nodes[nid] = WHTNode(nid, x, y, z)

    def add_element(
        self,
        eid: int,
        node_ids: List[int],
        elem_type: str = "QUAD4",
        pid: int = 0,
    ) -> None:
        self.elements[eid] = WHTElement(eid, elem_type, list(node_ids), pid)
        if pid not in self._parts:
            self._parts[pid] = []
        self._parts[pid].append(eid)

    def add_property(self, pid: int, ptype: str, t: float, mid: int) -> None:
        self.properties[pid] = WHTProperty(pid, ptype, t, mid)

    def add_material(
        self, mid: int, E: float, nu: float, rho: float
    ) -> None:
        self.materials[mid] = WHTMaterial(mid, E, nu, rho)

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------

    def add_node_set(
        self, sid: int, node_ids: List[int], name: str = ""
    ) -> None:
        self.node_sets[sid] = WHTNodeSet(sid, list(node_ids), name)

    def add_node_set_by_name(self, name: str, node_ids: List[int]) -> int:
        """Adds or updates a node set by name. Returns the sid."""
        # Find existing set by name
        for sid, ns in self.node_sets.items():
            if ns.name == name:
                ns.node_ids = list(node_ids)
                return sid
        # Create new if not found
        sid = max(self.node_sets.keys(), default=0) + 1
        self.add_node_set(sid, node_ids, name=name)
        return sid

    def add_elem_set(
        self, sid: int, elem_ids: List[int], name: str = ""
    ) -> None:
        self.elem_sets[sid] = WHTElemSet(sid, list(elem_ids), name)

    def add_elem_set_by_name(self, name: str, elem_ids: List[int]) -> int:
        """Adds or updates an element set by name. Returns the sid."""
        for sid, es in self.elem_sets.items():
            if es.name == name:
                es.elem_ids = list(elem_ids)
                return sid
        sid = max(self.elem_sets.keys(), default=0) + 1
        self.add_elem_set(sid, elem_ids, name=name)
        return sid

    def get_nodes_by_set(self, sid: int) -> List[int]:
        if sid not in self.node_sets:
            raise KeyError(f"Node set {sid} not found.")
        return self.node_sets[sid].node_ids

    def get_nodes_by_set_name(self, name: str) -> List[int]:
        """Find nodes in first set matching the name."""
        for ns in self.node_sets.values():
            if ns.name == name:
                return ns.node_ids
        raise KeyError(f"Node set with name '{name}' not found.")

    def get_elems_by_set(self, sid: int) -> List[int]:
        if sid not in self.elem_sets:
            raise KeyError(f"Element set {sid} not found.")
        return self.elem_sets[sid].elem_ids

    def get_elems_by_set_name(self, name: str) -> List[int]:
        """Find elements in first set matching the name."""
        for es in self.elem_sets.values():
            if es.name == name:
                return es.elem_ids
        raise KeyError(f"Element set with name '{name}' not found.")

    def get_nodes_from_elem_set(self, sid: int) -> List[int]:
        """Returns unique node IDs belonging to all elements in the set."""
        if sid not in self.elem_sets:
            raise KeyError(f"Element set {sid} not found.")
        nids = set()
        for eid in self.elem_sets[sid].elem_ids:
            if eid in self.elements:
                nids.update(self.elements[eid].node_ids)
        return sorted(list(nids))

    def get_nodes_from_elem_set_name(self, name: str) -> List[int]:
        """Returns unique node IDs belonging to all elements in the set by name."""
        for sid, es in self.elem_sets.items():
            if es.name == name:
                return self.get_nodes_from_elem_set(sid)
        raise KeyError(f"Element set name '{name}' not found.")

    # ------------------------------------------------------------------
    # RBE2
    # ------------------------------------------------------------------

    def add_rbe2(
        self,
        rbe2_id: int,
        master_nid: int,
        slave_nids: List[int],
        dofs: Tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    ) -> None:
        self.rbe2s[rbe2_id] = WHTRBE2(rbe2_id, master_nid, list(slave_nids), dofs)

    def get_rbe2_slaves(self, master_nid: int) -> List[int]:
        for rbe2 in self.rbe2s.values():
            if rbe2.master_nid == master_nid:
                return rbe2.slave_nids
        return []

    def get_rbe2_masters(self) -> List[int]:
        return [r.master_nid for r in self.rbe2s.values()]

    # ------------------------------------------------------------------
    # RBE3
    # ------------------------------------------------------------------

    def add_rbe3(
        self,
        rbe3_id: int,
        master_nid: int,
        slave_nids: List[int],
        dofs: Tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        weights: Optional[List[float]] = None,
    ) -> None:
        self.rbe3s[rbe3_id] = WHTRBE3(rbe3_id, master_nid, list(slave_nids), dofs, weights)

    # ------------------------------------------------------------------
    # Boundary Conditions
    # ------------------------------------------------------------------

    def apply_spc(
        self,
        node_ids: Union[int, List[int]],
        dofs: Tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        value: float = 0.0,
    ) -> None:
        """Apply single-point constraint to node(s)."""
        if isinstance(node_ids, int):
            node_ids = [node_ids]
        for nid in node_ids:
            self.spc_conditions.append(WHTSPCEntry(nid, tuple(dofs), value))

    def apply_spc_set(
        self,
        set_id: int,
        dofs: Tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        value: float = 0.0,
    ) -> None:
        """Apply SPC to all nodes in a named set."""
        self.apply_spc(self.get_nodes_by_set(set_id), dofs, value)

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------

    def apply_force(
        self,
        node_ids: Union[int, List[int]],
        dofs: Tuple[int, ...],
        values: Tuple[float, ...],
    ) -> None:
        """Apply nodal force to node(s). dofs and values must have the same length."""
        if isinstance(node_ids, int):
            node_ids = [node_ids]
        load_vec = [0.0] * 6
        for d, v in zip(dofs, values):
            load_vec[d] = v
        for nid in node_ids:
            self.loads.append(WHTLoadEntry(nid, tuple(load_vec)))

    def apply_force_set(
        self,
        set_id: int,
        dofs: Tuple[int, ...],
        total_values: Tuple[float, ...],
    ) -> None:
        """Distribute total_values evenly across all nodes in a set."""
        nids = self.get_nodes_by_set(set_id)
        n = len(nids)
        distributed = tuple(v / n for v in total_values)
        for nid in nids:
            load_vec = [0.0] * 6
            for d, v in zip(dofs, distributed):
                load_vec[d] = v
            self.loads.append(WHTLoadEntry(nid, tuple(load_vec)))

    # ------------------------------------------------------------------
    # Geometry utilities
    # ------------------------------------------------------------------

    def sorted_node_ids(self) -> List[int]:
        """Node IDs sorted in ascending order."""
        return sorted(self.nodes.keys())

    def sorted_element_ids(self) -> List[int]:
        """Element IDs sorted in ascending order."""
        return sorted(self.elements.keys())

    def nodes_array(self) -> np.ndarray:
        """(N, 3) node coordinates, sorted by node ID."""
        nids = self.sorted_node_ids()
        return np.array([[self.nodes[n].x, self.nodes[n].y, self.nodes[n].z]
                         for n in nids], dtype=np.float64)

    def node_id_to_index(self) -> Dict[int, int]:
        """Returns {nid: 0-based index} mapping for sorted node IDs."""
        return {nid: i for i, nid in enumerate(self.sorted_node_ids())}

    def nodes_array_dict(self) -> Dict[int, List[float]]:
        """Returns {nid: [x, y, z]} for all nodes."""
        return {nid: [n.x, n.y, n.z] for nid, n in self.nodes.items()}

    def get_adjacency(self) -> Dict[int, List[int]]:
        """Returns {nid: [neighbor_nids]} mapping based on element connectivity."""
        adj = {nid: set() for nid in self.nodes.keys()}
        for elem in self.elements.values():
            nids = elem.node_ids
            for i, nid in enumerate(nids):
                # Add previous and next nodes in the element loop
                prev_n = nids[i - 1]
                next_n = nids[(i + 1) % len(nids)]
                adj[nid].add(prev_n)
                adj[nid].add(next_n)
        return {nid: sorted(list(neighbors)) for nid, neighbors in adj.items()}

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    @property
    def n_elements(self) -> int:
        return len(self.elements)

    # ------------------------------------------------------------------
    # Conversion to wht_converter IR
    # ------------------------------------------------------------------

    def to_wht_result_data(
        self,
        metadata=None,
    ):
        """
        Convert geometry to WHTResultData (no result arrays).

        Used as a geometry-only IR for ParaView export or
        as a base for WHTSolverResult.to_wht_result_data().

        Parameters
        ----------
        metadata : WHTMetadata | None
            If None, a default static metadata is created.
        """
        from wht_converter.wht_models import WHTResultData, WHTMetadata

        if metadata is None:
            metadata = WHTMetadata(
                solver_name="WHT",
                solver_version="0.1.0",
                analysis_type="static",
                coordinate_system="cartesian",
                unit_length="mm",
                unit_force="N",
                unit_mass="tonne",
                unit_time="s",
            )

        nid_to_idx = self.node_id_to_index()
        nodes_arr  = self.nodes_array()

        # Build VTK CSR connectivity
        connectivity_list = []
        offsets_list      = [0]
        cell_types_list   = []

        for eid in sorted(self.elements.keys()):
            elem = self.elements[eid]
            # Remap to 0-based indices
            remapped = [nid_to_idx[n] for n in elem.node_ids]
            connectivity_list.extend(remapped)
            offsets_list.append(len(connectivity_list))
            cell_types_list.append(_VTK_TYPE.get(elem.type, 9))

        # [WHT] Rigid Elements (RBE2/RBE3) → Lines in IR Connectivity
        # We add them as VTK_LINE (3) to a separate Element Set for visualization
        rigid_start_idx = len(cell_types_list)
        for rbe in list(self.rbe2s.values()) + list(self.rbe3s.values()):
            m_idx = nid_to_idx.get(rbe.master_nid)
            if m_idx is None: continue
            for s_nid in rbe.slave_nids:
                s_idx = nid_to_idx.get(s_nid)
                if s_idx is None: continue
                connectivity_list.extend([m_idx, s_idx])
                offsets_list.append(len(connectivity_list))
                cell_types_list.append(3) # VTK_LINE
        rigid_end_idx = len(cell_types_list)

        connectivity = np.array(connectivity_list, dtype=np.int64)
        offsets      = np.array(offsets_list,      dtype=np.int64)
        cell_types   = np.array(cell_types_list,   dtype=np.int64)

        # Named sets → IR format
        node_sets_ir: Dict[str, np.ndarray] = {}
        for sid, ns in self.node_sets.items():
            key = ns.name if ns.name else f"nset_{sid}"
            node_sets_ir[key] = np.array(
                [nid_to_idx[n] for n in ns.node_ids if n in nid_to_idx],
                dtype=np.int64,
            )

        # [WHT] Boundary Conditions (SPC) → Node Set for Visualization
        spc_nids = sorted(list(set(spc.node_id for spc in self.spc_conditions)))
        if spc_nids:
            node_sets_ir["SPC"] = np.array(
                [nid_to_idx[n] for n in spc_nids if n in nid_to_idx],
                dtype=np.int64,
            )

        elem_sets_ir: Dict[str, np.ndarray] = {}
        sorted_eids = sorted(self.elements.keys())
        eid_to_cidx = {eid: i for i, eid in enumerate(sorted_eids)}
        for sid, es in self.elem_sets.items():
            key = es.name if es.name else f"eset_{sid}"
            elem_sets_ir[key] = np.array(
                [eid_to_cidx[e] for e in es.elem_ids if e in eid_to_cidx],
                dtype=np.int64,
            )
        
        # [WHT] Rigids set
        if rigid_end_idx > rigid_start_idx:
            elem_sets_ir["RIGIDS"] = np.arange(rigid_start_idx, rigid_end_idx, dtype=np.int64)

        return WHTResultData(
            nodes=nodes_arr,
            connectivity=connectivity,
            offsets=offsets,
            cell_types=cell_types,
            node_sets=node_sets_ir,
            element_sets=elem_sets_ir,
            point_data={},
            cell_data={},
            field_data={},
            time_values=np.array([0.0]),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_node_elem_db(
        cls,
        node_db: Dict[int, "array-like"],
        elem_db: Dict[int, List[int]],
        name: str = "WHT",
        is_solid: bool = False,
    ) -> "WHTMeshModel":
        """
        Build WHTMeshModel from the dict format used by exam1_nf / mesh_utils.

        node_db : {nid: [x, y, z]}
        elem_db : {eid: [nodes...]}
        is_solid: If True, treats 4-node elements as TETRA4 and 8-node as HEXA8.
        """
        model = cls(name=name)
        for nid, coords in node_db.items():
            model.add_node(nid, float(coords[0]), float(coords[1]), float(coords[2]))
        for eid, nids in elem_db.items():
            if is_solid:
                etype = "TETRA4" if len(nids) == 4 else "HEXA8"
            else:
                etype = "QUAD4" if len(nids) == 4 else "TRIA3" if len(nids) == 3 else "BEAM2"
            model.add_element(eid, list(nids), etype, pid=0)
        return model

    def export_to_solver(self, solver_type: str, path: str, reorder: bool = False) -> None:
        """
        Exports the current mesh and sets to industrial solver formats.
        :param solver_type: 'lsdyna', 'radioss', or 'optistruct'
        """
        from wht_converter.wht_exporters_industrial import LSDYNAExporter, RadiossExporter, OptistructExporter
        if solver_type.lower() == 'lsdyna':
            LSDYNAExporter().export(self, path, reorder)
        elif solver_type.lower() == 'radioss':
            RadiossExporter().export(self, path, reorder)
        elif solver_type.lower() == 'optistruct':
            OptistructExporter().export(self, path, reorder)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    @classmethod
    def import_from_solver(cls, solver_type: str, path: str) -> "WHTMeshModel":
        """
        Imports mesh and sets from industrial solver formats.
        :param solver_type: 'lsdyna', 'radioss', or 'optistruct'
        """
        from wht_converter.wht_importers_industrial import LSDYNAImporter, RadiossImporter, OptistructImporter
        if solver_type.lower() == 'lsdyna':
            return LSDYNAImporter().read(path)
        elif solver_type.lower() == 'radioss':
            return RadiossImporter().read(path)
        elif solver_type.lower() == 'optistruct':
            return OptistructImporter().read(path)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    def __repr__(self) -> str:
        return (
            f"WHTMeshModel(name='{self.name}', "
            f"nodes={self.n_nodes}, elements={self.n_elements}, "
            f"sets={len(self.node_sets)}, rbe2s={len(self.rbe2s)}, rbe3s={len(self.rbe3s)})"
        )
