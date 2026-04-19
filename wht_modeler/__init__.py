"""
wht_modeler
===========
WHT FEM Framework — Pre-processing / Mesh Model Package

Provides FEM entity management, FEM file I/O, and conversion to
wht_converter IR (WHTResultData).

Quick start
-----------
    from wht_modeler import WHTMeshModel
    from wht_modeler.io import LSDYNAReader, LSDYNAWriter

    model = LSDYNAReader().read("chassis.k")
    model.apply_spc([0, 1, 2], dofs=(0, 1, 2, 3, 4, 5))
    rd = model.to_wht_result_data()
"""

from .wht_entities import (
    WHTNode,
    WHTElement,
    WHTNodeSet,
    WHTElemSet,
    WHTRBE2,
    WHTProperty,
    WHTMaterial,
    WHTSPCEntry,
    WHTLoadEntry,
)
from .wht_mesh_model import WHTMeshModel

__version__ = "0.1.0"
__all__ = [
    "WHTNode", "WHTElement", "WHTNodeSet", "WHTElemSet",
    "WHTRBE2", "WHTProperty", "WHTMaterial",
    "WHTSPCEntry", "WHTLoadEntry",
    "WHTMeshModel",
]
