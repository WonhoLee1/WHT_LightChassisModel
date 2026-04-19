"""
load_cases.py
=============
WHT FEM Framework — Load Case Definitions and Library

WHTLoadCase stores BCs + loads for a single analysis.
LoadCaseLibrary provides 5 preset structural test cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from wht_modeler.wht_mesh_model import WHTMeshModel


@dataclass
class WHTBCEntry:
    """Boundary condition for a load case (may differ from model-level BCs)."""
    node_id: int
    dofs: Tuple[int, ...]
    value: float = 0.0


@dataclass
class WHTForceEntry:
    """Nodal force entry for a load case."""
    node_id: int
    load_vector: Tuple[float, ...]  # 6 components [Fx,Fy,Fz,Mx,My,Mz]


@dataclass
class WHTLoadCase:
    """
    A single static load case: boundary conditions + nodal forces.

    wht_solver.WHTSolver.solve_static() consumes this directly.
    """
    name: str = "LoadCase"
    bcs:    List[WHTBCEntry]    = field(default_factory=list)
    forces: List[WHTForceEntry] = field(default_factory=list)

    def add_bc(
        self,
        node_ids: Union[int, List[int]],
        dofs: Tuple[int, ...] = (0, 1, 2, 3, 4, 5),
        value: float = 0.0,
    ) -> "WHTLoadCase":
        if isinstance(node_ids, int):
            node_ids = [node_ids]
        for nid in node_ids:
            self.bcs.append(WHTBCEntry(nid, tuple(dofs), value))
        return self

    def add_force(
        self,
        node_ids: Union[int, List[int]],
        dofs: Tuple[int, ...],
        values: Tuple[float, ...],
        distribute: bool = False,
    ) -> "WHTLoadCase":
        """
        Add nodal forces.

        distribute=True: total values are split evenly across node_ids.
        """
        if isinstance(node_ids, int):
            node_ids = [node_ids]
        n = len(node_ids) if distribute else 1
        for nid in node_ids:
            vec = [0.0] * 6
            for d, v in zip(dofs, values):
                vec[d] = v / n
            self.forces.append(WHTForceEntry(nid, tuple(vec)))
        return self

    def __repr__(self) -> str:
        return (f"WHTLoadCase(name='{self.name}', "
                f"bcs={len(self.bcs)}, forces={len(self.forces)})")


# ---------------------------------------------------------------------------
# Preset library
# ---------------------------------------------------------------------------

def _resolve_nodes(
    model: "WHTMeshModel",
    target: Union[int, List[int]],
) -> List[int]:
    """Accept raw node IDs or a single set ID (int resolved via model)."""
    if isinstance(target, int):
        # Try as set ID first; fall back to single node ID
        if target in model.node_sets:
            return model.get_nodes_by_set(target)
        return [target]
    return list(target)


class LoadCaseLibrary:
    """
    Standard structural load cases for chassis/tray testing.

    All methods accept node IDs (int | list[int]) or node set IDs (int).
    """

    @staticmethod
    def three_point_bending(
        model: "WHTMeshModel",
        support_sets: List[Union[int, List[int]]],
        load_target:  Union[int, List[int]],
        load_z: float = -1000.0,
        constrain_dofs: Tuple[int, ...] = (0, 1, 2),
    ) -> WHTLoadCase:
        """
        3-point bending: 2 support points, 1 central load point.

        support_sets : list of 2 targets (each is set_id or node list)
        load_target  : central loading target
        load_z       : total Z-force [N] (negative = downward)
        """
        lc = WHTLoadCase(name="3pt_bending")

        for sup in support_sets:
            nids = _resolve_nodes(model, sup)
            for nid in nids:
                lc.add_bc(nid, constrain_dofs)

        load_nids = _resolve_nodes(model, load_target)
        n = len(load_nids)
        for nid in load_nids:
            vec = (0.0, 0.0, load_z / n, 0.0, 0.0, 0.0)
            lc.forces.append(WHTForceEntry(nid, vec))

        return lc

    @staticmethod
    def four_point_bending(
        model: "WHTMeshModel",
        support_sets:  List[Union[int, List[int]]],
        load_targets:  List[Union[int, List[int]]],
        load_z: float = -1000.0,
        constrain_dofs: Tuple[int, ...] = (0, 1, 2),
    ) -> WHTLoadCase:
        """4-point bending: 2 supports, 2 load points."""
        lc = WHTLoadCase(name="4pt_bending")

        for sup in support_sets:
            nids = _resolve_nodes(model, sup)
            for nid in nids:
                lc.add_bc(nid, constrain_dofs)

        total_load_nodes = sum(
            len(_resolve_nodes(model, t)) for t in load_targets
        )
        for tgt in load_targets:
            for nid in _resolve_nodes(model, tgt):
                vec = (0.0, 0.0, load_z / total_load_nodes, 0.0, 0.0, 0.0)
                lc.forces.append(WHTForceEntry(nid, vec))

        return lc

    @staticmethod
    def twisting(
        model: "WHTMeshModel",
        fixed_corner: Union[int, List[int]],
        twist_corner: Union[int, List[int]],
        load_z: float = -1000.0,
    ) -> WHTLoadCase:
        """
        Torsional twist: fix one corner, apply Z-load on opposite corner.
        """
        lc = WHTLoadCase(name="twisting")

        fixed_nids = _resolve_nodes(model, fixed_corner)
        for nid in fixed_nids:
            lc.add_bc(nid, (0, 1, 2, 3, 4, 5))

        twist_nids = _resolve_nodes(model, twist_corner)
        n = len(twist_nids)
        for nid in twist_nids:
            vec = (0.0, 0.0, load_z / n, 0.0, 0.0, 0.0)
            lc.forces.append(WHTForceEntry(nid, vec))

        return lc

    @staticmethod
    def corner_lift(
        model: "WHTMeshModel",
        support_sets: List[Union[int, List[int]]],
        lift_target:  Union[int, List[int]],
        load_z: float = 1000.0,
    ) -> WHTLoadCase:
        """
        Corner lift: fix 3 corners, push up on 1 corner.
        """
        lc = WHTLoadCase(name="corner_lift")

        for sup in support_sets:
            nids = _resolve_nodes(model, sup)
            for nid in nids:
                lc.add_bc(nid, (0, 1, 2))

        lift_nids = _resolve_nodes(model, lift_target)
        n = len(lift_nids)
        for nid in lift_nids:
            vec = (0.0, 0.0, load_z / n, 0.0, 0.0, 0.0)
            lc.forces.append(WHTForceEntry(nid, vec))

        return lc

    @staticmethod
    def end_bending(
        model: "WHTMeshModel",
        fixed_end: Union[int, List[int]],
        load_end:  Union[int, List[int]],
        load_z: float = -1000.0,
    ) -> WHTLoadCase:
        """
        Cantilever-style end bending: one end fixed, other end loaded.
        """
        lc = WHTLoadCase(name="end_bending")

        for nid in _resolve_nodes(model, fixed_end):
            lc.add_bc(nid, (0, 1, 2, 3, 4, 5))

        load_nids = _resolve_nodes(model, load_end)
        n = len(load_nids)
        for nid in load_nids:
            vec = (0.0, 0.0, load_z / n, 0.0, 0.0, 0.0)
            lc.forces.append(WHTForceEntry(nid, vec))

        return lc
