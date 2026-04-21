# -*- coding: utf-8 -*-
"""
mesh_utils.py
=============
WHT FEM Framework — Geometry & Mesh Utilities

This module provides high-level geometric functions to generate 3D shells and solids
using the Gmsh (OpenCASCADE) engine. Optimized for structural optimization.
"""

import gmsh
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union


def generate_shell_tray(
    width: float = 100.0,
    length: float = 100.0,
    height: float = 10.0,
    mesh_size_xy: float = 50.0,
    mesh_size_z: float = 5.0,
    draft_angle: float = 0.0,
    flange_width: float = 0.0,
    flange_segments: Optional[List[Tuple[float, float]]] = None,
    origin: str = 'corner',
    mesh_type: str = 'quad4',
    flanges: Tuple[bool, bool, bool, bool] = (True, True, True, True)
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[int]]]:
    """
    Generates a Shell Tray mesh using Gmsh (OpenCASCADE engine).
    Uses a stable 4-point rim architecture with uniform height.

    Parameters:
        width: Internal width of the tray [mm].
        length: Internal length of the tray [mm].
        height: Wall height from base to rim [mm].
        mesh_size_xy: Target element size on the plane [mm].
        mesh_size_z: Target vertical discretization size [mm].
        draft_angle: Taper angle for the walls [deg].
        flange_segments: List of (width, delta_z) for each flange tier.
        origin: 'corner' (0,0) or 'center' (-w/2, -l/2).
        mesh_type: 'quad4' (structured), 'tria3' (free), or 'mixed'.
        flanges: (Y-min, X-max, Y-max, X-min) selective activation.

    Returns:
        nodes: {node_id: [x, y, z]} mapping.
        elements: {elem_id: [node_id, ...]} mapping.
    """
    gmsh.initialize()
    gmsh.model.add("ShellTray")
    
    # --- 1. Initial Setup & Constants ---
    if flange_segments is None:
        flange_segments = [(flange_width, 0.0)] if flange_width > 0 else []
    
    x0, y0 = (0.0, 0.0) if origin == 'corner' else (-width / 2.0, -length / 2.0)
    
    rad_draft = math.radians(draft_angle)
    wall_offset = height * math.tan(rad_draft)
    
    # --- 2. Create Geometric Points ---
    # Base Points (Z = 0)
    p_base = [
        gmsh.model.occ.addPoint(x0, y0, 0, mesh_size_xy),
        gmsh.model.occ.addPoint(x0 + width, y0, 0, mesh_size_xy),
        gmsh.model.occ.addPoint(x0 + width, y0 + length, 0, mesh_size_xy),
        gmsh.model.occ.addPoint(x0, y0 + length, 0, mesh_size_xy)
    ]
    
    # Rim Points (Z = Height)
    p_rim = [
        gmsh.model.occ.addPoint(x0 - wall_offset, y0 - wall_offset, height, mesh_size_xy),
        gmsh.model.occ.addPoint(x0 + width + wall_offset, y0 - wall_offset, height, mesh_size_xy),
        gmsh.model.occ.addPoint(x0 + width + wall_offset, y0 + length + wall_offset, height, mesh_size_xy),
        gmsh.model.occ.addPoint(x0 - wall_offset, y0 + length + wall_offset, height, mesh_size_xy)
    ]

    # --- 3. Create Boundary Lines ---
    l_base = [gmsh.model.occ.addLine(p_base[i], p_base[(i + 1) % 4]) for i in range(4)]
    l_rim  = [gmsh.model.occ.addLine(p_rim[i], p_rim[(i + 1) % 4]) for i in range(4)]
    l_wall = [gmsh.model.occ.addLine(p_base[i], p_rim[i]) for i in range(4)]
    
    gmsh.model.occ.synchronize()
    
    # --- 4. Define Surface Topology (Base & Walls) ---
    cl_base = gmsh.model.occ.addCurveLoop(l_base)
    s_base = gmsh.model.occ.addPlaneSurface([cl_base])
    
    s_walls = []
    for i in range(4):
        # Wall loop order: base_line[i], wall_vert[i+1], -rim_line[i], -wall_vert[i]
        loop = [l_base[i], l_wall[(i + 1) % 4], -l_rim[i], -l_wall[i]]
        cl_wall = gmsh.model.occ.addCurveLoop(loop)
        s_walls.append(gmsh.model.occ.addPlaneSurface([cl_wall]))

    # --- 5. Tiered Flanges Generation ---
    tiers_p = [p_rim]
    tiers_l_rim = [l_rim]
    flange_surfs_data = [] # Side-wise data for transfinite mapping
    
    current_offset = wall_offset
    current_z = height
    
    for t_idx, (seg_w, seg_dz) in enumerate(flange_segments):
        prev_points = tiers_p[-1]
        current_offset += seg_w
        current_z += seg_dz
        
        # New points for this flange tier
        new_points = [
            gmsh.model.occ.addPoint(x0 - current_offset, y0 - current_offset, current_z, mesh_size_xy),
            gmsh.model.occ.addPoint(x0 + width + current_offset, y0 - current_offset, current_z, mesh_size_xy),
            gmsh.model.occ.addPoint(x0 + width + current_offset, y0 + length + current_offset, current_z, mesh_size_xy),
            gmsh.model.occ.addPoint(x0 - current_offset, y0 + length + current_offset, current_z, mesh_size_xy)
        ]
        new_l_rim = [gmsh.model.occ.addLine(new_points[i], new_points[(i + 1) % 4]) for i in range(4)]
        new_l_rad = [gmsh.model.occ.addLine(prev_points[i], new_points[i]) for i in range(4)]
        
        for i in range(4):
            if flanges[i]:
                # Flange surface loop
                loop = [tiers_l_rim[-1][i], new_l_rad[(i + 1) % 4], -new_l_rim[i], -new_l_rad[i]]
                cl_flange = gmsh.model.occ.addCurveLoop(loop)
                s_tag = gmsh.model.occ.addPlaneSurface([cl_flange])
                flange_surfs_data.append({
                    'side': i, 'tier': t_idx, 'surface': s_tag, 
                    'radial': (new_l_rad[i], new_l_rad[(i + 1) % 4]), 'outer': new_l_rim[i]
                })
        
        tiers_p.append(new_points)
        tiers_l_rim.append(new_l_rim)

    gmsh.model.occ.synchronize()

    # --- 6. Transfinite Meshing Configuration ---
    if mesh_type in ('quad4', 'tria3', 'mixed'):
        # Density calculations
        nx = max(2, int(round(width / mesh_size_xy)) + 1)
        ny = max(2, int(round(length / mesh_size_xy)) + 1)
        nz = max(1, int(round(height / mesh_size_z)))
        
        # 6.1 Base Mesh
        gmsh.model.mesh.setTransfiniteCurve(l_base[0], nx); gmsh.model.mesh.setTransfiniteCurve(l_base[2], nx)
        gmsh.model.mesh.setTransfiniteCurve(l_base[1], ny); gmsh.model.mesh.setTransfiniteCurve(l_base[3], ny)
        gmsh.model.mesh.setTransfiniteSurface(s_base)
        if mesh_type in ('quad4', 'mixed'): gmsh.model.mesh.setRecombine(2, s_base)
        
        # 6.2 Wall Mesh
        for i in range(4):
            gmsh.model.mesh.setTransfiniteCurve(l_rim[i], nx if i in (0, 2) else ny)
            gmsh.model.mesh.setTransfiniteCurve(l_wall[i], nz + 1)
        for s in s_walls:
            gmsh.model.mesh.setTransfiniteSurface(s)
            if mesh_type == 'quad4': gmsh.model.mesh.setRecombine(2, s)
            
        # 6.3 Flange Mesh
        for f_data in flange_surfs_data:
            t_idx = f_data['tier']
            seg_w, seg_dz = flange_segments[t_idx]
            nw = max(1, int(round(math.sqrt(seg_w**2 + seg_dz**2) / mesh_size_xy)))
            
            gmsh.model.mesh.setTransfiniteCurve(f_data['radial'][0], nw + 1)
            gmsh.model.mesh.setTransfiniteCurve(f_data['radial'][1], nw + 1)
            gmsh.model.mesh.setTransfiniteCurve(f_data['outer'], nx if f_data['side'] in (0, 2) else ny)
            gmsh.model.mesh.setTransfiniteSurface(f_data['surface'])
            if mesh_type == 'quad4': gmsh.model.mesh.setRecombine(2, f_data['surface'])

    # --- 7. Final Mesh Generation ---
    if mesh_type == 'quad4':
        gmsh.option.setNumber("Mesh.RecombineAll", 1); gmsh.option.setNumber("Mesh.Algorithm", 6)
    elif mesh_type == 'tria3':
        gmsh.option.setNumber("Mesh.RecombineAll", 0); gmsh.option.setNumber("Mesh.Algorithm", 5)
    else:
        gmsh.option.setNumber("Mesh.RecombineAll", 0); gmsh.option.setNumber("Mesh.Algorithm", 6)
    
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_xy)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_xy)
    gmsh.model.mesh.generate(2)
    
    # --- 8. Export Node & Element Data ---
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes_xyz = node_coords.reshape(-1, 3)
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
    
    node_data = {i: nodes_xyz[i] for i in range(len(node_tags))}
    
    elem_data = {}
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        num_v = 3 if etype == 2 else 4 # TRIA3=2, QUAD4=3
        nodes_flat = enodes.reshape(-1, num_v)
        for i, tag in enumerate(etags):
            elem_data[int(tag)] = [tag_to_idx[int(n)] for n in nodes_flat[i]]
            
    gmsh.finalize()
    return node_data, elem_data


def get_nodes_in_box(
    nodes: Dict[int, np.ndarray],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None
) -> List[int]:
    """Helper to select node IDs within a specific 3D bounding box."""
    selected_ids = []
    for nid, r in nodes.items():
        if (x_range[0] <= r[0] <= x_range[1] if x_range else True) and \
           (y_range[0] <= r[1] <= y_range[1] if y_range else True) and \
           (z_range[0] <= r[2] <= z_range[1] if z_range else True):
            selected_ids.append(nid)
    return selected_ids
