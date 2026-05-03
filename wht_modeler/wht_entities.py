"""
wht_entities.py
===============
WHT FEM Framework — Core Entity Dataclasses

All FEM entities stored inside WHTMeshModel.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class WHTNode:
    nid: int
    x: float
    y: float
    z: float

    def coords(self):
        return (self.x, self.y, self.z)


@dataclass
class WHTElement:
    eid: int
    type: str           # "QUAD4" | "TRIA3" | "BEAM2"
    node_ids: List[int]
    pid: int = 0        # property ID (0 = unassigned)


@dataclass
class WHTNodeSet:
    sid: int
    node_ids: List[int] = field(default_factory=list)
    name: str = ""


@dataclass
class WHTElemSet:
    sid: int
    elem_ids: List[int] = field(default_factory=list)
    name: str = ""


@dataclass
class WHTRBE2:
    rbe2_id: int
    master_nid: int
    slave_nids: List[int]
    dofs: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)


@dataclass
class WHTRBE3:
    """
    RBE3 Interpolation Element.
    The master node (dependent) displacement is a weighted average of slave nodes (independent).
    Used to distribute loads or sense displacement without adding artificial stiffness.
    """
    rbe3_id: int
    master_nid: int
    slave_nids: List[int]
    dofs: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    weights: Optional[List[float]] = None # Default is uniform weights (1.0)


@dataclass
class WHTProperty:
    pid: int
    type: str       # "PSHELL" | "PBEAM"
    t: float = 1.0  # shell thickness
    mid: int = 0    # material ID ref


@dataclass
class WHTMaterial:
    mid: int
    E: float = 210000.0   # Young's modulus [MPa, or N/mm²]
    nu: float = 0.3       # Poisson's ratio
    rho: float = 7.85e-9  # density [tonne/mm³]


@dataclass
class WHTSPCEntry:
    """Single-point constraint (fixed displacement)."""
    node_id: int
    dofs: Tuple[int, ...]   # e.g. (0,1,2,3,4,5) for full fix
    value: float = 0.0


@dataclass
class WHTLoadEntry:
    """Nodal load entry."""
    node_id: int
    load_vector: Tuple[float, ...]  # 6 components: [Fx,Fy,Fz,Mx,My,Mz]
