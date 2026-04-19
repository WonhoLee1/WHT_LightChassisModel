"""
wht_selectors.py
================
WHT FEM Framework — Model-Agnostic Entity Selectors

Provides geometry-based selection utilities for WHTMeshModel.
Works with absolute coordinates to ensure compatibility across
generated and loaded (LS-DYNA, Radioss, etc.) models.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .wht_mesh_model import WHTMeshModel

def select_nodes_by_cylinder(
    model: WHTMeshModel, 
    locations: List[Tuple[float, float, float]], 
    axis: str = 'z',
    z_range: Optional[Tuple[float, float]] = None
) -> List[int]:
    """
    Selects nodes within one or more cylindrical regions.
    
    Parameters
    ----------
    model : WHTMeshModel
        The mesh model to search.
    locations : List[Tuple[float, float, float]]
        List of (center_x, center_y, radius) if axis='z'.
        Adjust accordingly for other axes.
    axis : str
        The axial direction of the cylinders ('x', 'y', or 'z').
    z_range : Tuple[float, float] | None
        Optional height range filter for the cylinder.
        
    Returns
    -------
    List[int]
        Sorted list of selected node IDs.
    """
    selected_nids = set()
    nodes = model.nodes
    
    for cx, cy, r in locations:
        r_sq = r * r
        for nid, node in nodes.items():
            # Check axis-specific distance
            if axis == 'z':
                dist_sq = (node.x - cx)**2 + (node.y - cy)**2
                if dist_sq <= r_sq:
                    if z_range is None or (z_range[0] <= node.z <= z_range[1]):
                        selected_nids.add(nid)
            elif axis == 'x':
                dist_sq = (node.y - cx)**2 + (node.z - cy)**2
                if dist_sq <= r_sq:
                    selected_nids.add(nid)
            elif axis == 'y':
                dist_sq = (node.x - cx)**2 + (node.z - cy)**2
                if dist_sq <= r_sq:
                    selected_nids.add(nid)
                    
    return sorted(list(selected_nids))

def select_nodes_by_box(
    model: WHTMeshModel,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None
) -> List[int]:
    """
    Selects nodes within a bounding box.
    """
    selected_nids = []
    for nid, node in model.nodes.items():
        in_x = (x_range[0] <= node.x <= x_range[1]) if x_range else True
        in_y = (y_range[0] <= node.y <= y_range[1]) if y_range else True
        in_z = (z_range[0] <= node.z <= z_range[1]) if z_range else True
        if in_x and in_y and in_z:
            selected_nids.append(nid)
    return sorted(selected_nids)

def apply_named_sets_by_recipe(model: WHTMeshModel, recipe: dict):
    """
    Applies multiple sets to a model based on a selection recipe.
    
    Example recipe:
    {
        "set_node-spk_A": {"type": "cylinder", "data": [(-100, -400, 20)], "axis": "z"},
        "set_node-flange": {"type": "box", "z_range": (28.0, 32.0)}
    }
    """
    for name, config in recipe.items():
        stype = config.get("type", "box")
        if stype == "cylinder":
            nids = select_nodes_by_cylinder(
                model, 
                config["data"], 
                axis=config.get("axis", "z"),
                z_range=config.get("z_range")
            )
        elif stype == "box":
            nids = select_nodes_by_box(
                model,
                x_range=config.get("x_range"),
                y_range=config.get("y_range"),
                z_range=config.get("z_range")
            )
        else:
            continue
            
        if nids:
            model.add_node_set_by_name(name, nids)
