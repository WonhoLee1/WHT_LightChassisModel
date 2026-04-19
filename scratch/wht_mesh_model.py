"""
wht_mesh_model.py
================
WHT Universal FEM Framework — Pre-processing Data Model

Defines the WHTMeshModel class, which manages high-level FEM entities:
  - Nodes and Elements (Shell/Solid/Rigid)
  - Named Sets (Node sets, Element sets)
  - Boundary Conditions (Supports) and Nodal Loads
  - RBE2 (Rigid Body Elements)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
import numpy as np


@dataclass
class WHTNodeSet:
    set_id: int
    name: str
    node_ids: List[int] = field(default_factory=list)


@dataclass
class WHTElementSet:
    set_id: int
    name: str
    element_ids: List[int] = field(default_factory=list)


@dataclass
class WHTRBE2:
    master_id: int
    slave_ids: List[int]
    dofs: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])


@dataclass
class WHTLoad:
    node_id: int
    vector: np.ndarray  # (6,) for Fx, Fy, Fz, Mx, My, Mz


@dataclass
class WHTSupport:
    node_id: int
    dofs: List[int]  # [1, 1, 1, 0, 0, 0] 1=fixed, 0=free


class WHTMeshModel:
    """
    High-level FEM model for pre-processing and optimization setup.
    """
    def __init__(self, name: str = "WHTModel"):
        self.name = name
        self.nodes: Dict[int, np.ndarray] = {}  # {id: [x, y, z]}
        self.elements: Dict[int, List[int]] = {}  # {id: [n1, n2, ...]}
        self.element_types: Dict[int, str] = {}  # {id: "QUAD" | "TRIA" | "BEAM"}
        
        # Sets
        self.node_sets: Dict[int, WHTNodeSet] = {}
        self.element_sets: Dict[int, WHTElementSet] = {}
        
        # Geometry / Entities for SET_GENERAL
        self.boxes: Dict[int, np.ndarray] = {}  # {bid: [xmin, xmax, ymin, ymax, zmin, zmax]}
        self.parts: Dict[int, List[int]] = {}   # {pid: [elem_id1, ...]}
        
        # Constraints & Loads
        self.rigid_elements: List[WHTRBE2] = []
        self.loads: List[WHTLoad] = []
        self.supports: List[WHTSupport] = []

    # --- Node Management ---
    def add_node(self, node_id: int, x: float, y: float, z: float):
        self.nodes[node_id] = np.array([x, y, z], dtype=np.float64)

    def get_node_coords(self, node_id: int) -> np.ndarray:
        return self.nodes[node_id]

    # --- Element Management ---
    def add_element(self, elem_id: int, node_ids: List[int], elem_type: str = "QUAD"):
        self.elements[elem_id] = node_ids
        self.element_types[elem_id] = elem_type

    # --- SET Management ---
    def create_node_set(self, set_id: int, name: str, node_ids: List[int]):
        self.node_sets[set_id] = WHTNodeSet(set_id, name, node_ids)

    def get_nodes_by_set(self, set_id: int) -> List[int]:
        if set_id not in self.node_sets:
            raise KeyError(f"Node Set {set_id} not found.")
        return self.node_sets[set_id].node_ids

    # --- Rigid Body Elements (RBE2) ---
    def add_rbe2(self, master_id: int, slave_ids: List[int], dofs: List[int] = None):
        if dofs is None:
            dofs = [1, 2, 3, 4, 5, 6]
        self.rigid_elements.append(WHTRBE2(master_id, slave_ids, dofs))

    # --- Boundary Conditions & Loads ---
    def apply_support(self, node_id: int, dofs: List[int]):
        self.supports.append(WHTSupport(node_id, dofs))

    def apply_nodal_load(self, node_id: int, vector: List[float]):
        self.loads.append(WHTLoad(node_id, np.array(vector, dtype=np.float64)))

    def apply_load_to_set(self, set_id: int, total_vector: List[float]):
        """Distributes total_vector across all nodes in the set."""
        node_ids = self.get_nodes_by_set(set_id)
        if not node_ids:
            return
        vec = np.array(total_vector, dtype=np.float64) / len(node_ids)
        for nid in node_ids:
            self.apply_nodal_load(nid, vec.tolist())

    # --- Utility ---
    def __repr__(self):
        return (f"WHTMeshModel(name='{self.name}', "
                f"nodes={len(self.nodes)}, elements={len(self.elements)}, "
                f"sets={len(self.node_sets)}, rbe2={len(self.rigid_elements)})")
