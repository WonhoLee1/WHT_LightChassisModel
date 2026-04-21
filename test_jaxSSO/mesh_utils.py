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
        loop = [l_base[i], l_wall[(i + 1) % 4], -l_rim[i], -l_wall[i]]
        cl_wall = gmsh.model.occ.addCurveLoop(loop)
        s_walls.append(gmsh.model.occ.addPlaneSurface([cl_wall]))

    # --- 5. Tiered Flanges Generation ---
    tiers_p = [p_rim]
    tiers_l_rim = [l_rim]
    flange_surfs_data = [] 
    
    current_offset = wall_offset
    current_z = height
    
    for t_idx, (seg_w, seg_dz) in enumerate(flange_segments):
        prev_points = tiers_p[-1]
        current_offset += seg_w
        current_z += seg_dz
        
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
        nx = max(2, int(round(width / mesh_size_xy)) + 1)
        ny = max(2, int(round(length / mesh_size_xy)) + 1)
        nz = max(1, int(round(height / mesh_size_z)))
        
        gmsh.model.mesh.setTransfiniteCurve(l_base[0], nx); gmsh.model.mesh.setTransfiniteCurve(l_base[2], nx)
        gmsh.model.mesh.setTransfiniteCurve(l_base[1], ny); gmsh.model.mesh.setTransfiniteCurve(l_base[3], ny)
        gmsh.model.mesh.setTransfiniteSurface(s_base)
        if mesh_type in ('quad4', 'mixed'): gmsh.model.mesh.setRecombine(2, s_base)
        
        for i in range(4):
            gmsh.model.mesh.setTransfiniteCurve(l_rim[i], nx if i in (0, 2) else ny)
            gmsh.model.mesh.setTransfiniteCurve(l_wall[i], nz + 1)
        for s in s_walls:
            gmsh.model.mesh.setTransfiniteSurface(s)
            if mesh_type == 'quad4': gmsh.model.mesh.setRecombine(2, s)
            
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
    
    node_data = {tag_to_idx[int(tag)]: nodes_xyz[i] for i, tag in enumerate(node_tags)}
    
    elem_data = {}
    dim_elem = 2
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim_elem)
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        num_v = 3 if etype == 2 else 4
        nodes_flat = enodes.reshape(-1, num_v)
        for i, tag in enumerate(etags):
            elem_data[int(tag)] = [tag_to_idx[int(n)] for n in nodes_flat[i]]
            
    gmsh.finalize()
    return node_data, elem_data


def generate_solid_hexa_tray(
    width: float = 100.0,
    length: float = 100.0,
    height: float = 10.0,
    mesh_size_xy: float = 50.0,
    mesh_size_z: float = 5.0,
    draft_angle: float = 0.0,
    wall_layers: int = 2,
    flange_segments: Optional[List[Tuple[float, float]]] = None,
    origin: str = 'corner',
    flanges: Tuple[bool, bool, bool, bool] = (True, True, True, True)
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[int]]]:
    """
    Generates a Solid Hexa (HEXA8) Tray mesh using Gmsh Extrude.
    Converts 2D surface elements to 3D volume elements.

    Parameters:
        wall_layers: Number of element layers through the wall thickness.
    """
    node_shell, elem_shell = generate_shell_tray(
        width=width, length=length, height=height,
        mesh_size_xy=mesh_size_xy, mesh_size_z=mesh_size_z,
        draft_angle=draft_angle, flange_segments=flange_segments,
        origin=origin, mesh_type='quad4', flanges=flanges
    )
    
    gmsh.initialize()
    gmsh.model.add("SolidTray")
    
    # Re-create the shell nodes in Gmsh to extrude them
    tag_to_new = {}
    for nid, r in node_shell.items():
        tag_to_new[nid] = gmsh.model.occ.addPoint(r[0], r[1], r[2])
    
    # We use a trick: Re-import shell connectivity and extrude volumes.
    # But for a cleaner JAX-friendly result, we can manually "stack" nodes.
    # Here we'll do a programmatic stack/extrude as it's more stable for HEXA8.
    
    # 0.6mm thickness example (Scale it or use a default)
    t = 0.6 
    
    node_data_3d = {}
    elem_data_3d = {}
    
    # Stack nodes: Layer 0 (Bottom/Shell) ... Layer N (Top)
    num_nodes_shell = len(node_shell)
    for layer in range(wall_layers + 1):
        z_offset = (layer / wall_layers) * t
        for nid, r in node_shell.items():
            new_id = nid + layer * num_nodes_shell
            node_data_3d[new_id] = np.array([r[0], r[1], r[2] + z_offset])
            
    # Create HEXA8 elements
    e_idx = 1
    for eid, nids in elem_shell.items():
        if len(nids) != 4: continue # Only Quad-based HEXA8
        for layer in range(wall_layers):
            # Bottom face nodes
            b1, b2, b3, b4 = [n + layer * num_nodes_shell for n in nids]
            # Top face nodes
            t1, t2, t3, t4 = [n + (layer + 1) * num_nodes_shell for n in nids]
            # Standard HEXA8 ordering
            elem_data_3d[e_idx] = [b1, b2, b3, b4, t1, t2, t3, t4]
            e_idx += 1
            
    gmsh.finalize()
    return node_data_3d, elem_data_3d


def apply_auto_beads(
    node_db: Dict[int, np.ndarray],
    width: float,
    length: float,
    margin: float = 30.0,
    target_ratio: float = 0.5,
    max_depth: float = 10.0,
    origin: str = 'center',
    mode: str = 'grid',
    **kwargs
) -> Dict[int, np.ndarray]:
    """
    Applies topography bead patterns to the tray floor.
    
    Modes:
        'grid': Symmetric sinusoidal patterns (design intent).
        'random': Symmetric random rectangular patches (topography optimization feel).
    """
    new_node_db = {nid: np.copy(coords) for nid, coords in node_db.items()}
    
    if origin == 'corner':
        cx, cy = width / 2.0, length / 2.0
    else:
        cx, cy = 0.0, 0.0
        
    inner_w = width - 2 * margin
    inner_l = length - 2 * margin
    
    if mode == 'grid':
        freq_x, freq_y = 2.0, 2.0 
        for nid, r in new_node_db.items():
            if abs(r[2]) < 0.1 and abs(r[0] - cx) < (inner_w/2) and abs(r[1] - cy) < (inner_l/2):
                nx = (r[0] - cx) / (inner_w/2)
                ny = (r[1] - cy) / (inner_l/2)
                val_x = math.cos(math.pi * nx * freq_x)
                val_y = math.cos(math.pi * ny * freq_y)
                surface = val_x * val_y
                threshold = 1.0 - target_ratio
                if abs(surface) > threshold:
                    smooth = (abs(surface) - threshold) / (1.0 - threshold)
                    new_node_db[nid][2] += max_depth * np.sign(surface) * (smooth ** 1.5)
                    
    elif mode == 'random':
        # 1. Setup Random Seed for Reproducibility
        np.random.seed(42)
        
        # 2. Identify candidate nodes in the FIRST quadrant (Reference)
        # We only calculate for the (x >= cx, y >= cy) quadrant and mirror later.
        ref_nids = []
        for nid, r in new_node_db.items():
            dx, dy = r[0] - cx, r[1] - cy
            if abs(r[2]) < 0.1 and 0 <= dx < (inner_w/2) and 0 <= dy < (inner_l/2):
                ref_nids.append(nid)
                
        if not ref_nids: return new_node_db
        
        target_count = int(len(ref_nids) * target_ratio)
        morphed_set = set()
        z_offsets = {nid: 0.0 for nid in node_db.keys()}
        
        # 3. Iteratively generate rectangles until target area ratio is met
        max_iters = 500 # Safety guard
        iter_count = 0
        
        while len(morphed_set) < target_count and iter_count < max_iters:
            iter_count += 1
            
            # Random Rectangle in the reference quadrant
            # Center coordinates relative to the quadrant [0, 1]
            rx = np.random.uniform(0, inner_w/2)
            ry = np.random.uniform(0, inner_l/2)
            
            # Random Size (Controlled by user-defined ratios)
            min_sz = kwargs.get('min_size_ratio', 0.05)
            max_sz = kwargs.get('max_size_ratio', 0.20)
            
            rw = np.random.uniform(min_sz * inner_w, max_sz * inner_w)
            rh = np.random.uniform(min_sz * inner_l, max_sz * inner_l)
            
            # Random Height/Depth
            min_d = kwargs.get('min_depth', max_depth * 0.3)
            rz = np.random.uniform(min_d, max_depth)
            polarity = np.random.choice([-1.0, 1.0])
            
            # Update nodes in this rectangle (within the quadrant)
            for nid in ref_nids:
                r = new_node_db[nid]
                dx, dy = r[0] - cx, r[1] - cy
                
                # Check if point is inside the generated rectangle
                if abs(dx - rx) < rw/2 and abs(dy - ry) < rh/2:
                    z_offsets[nid] += rz * polarity
                    morphed_set.add(nid)
        
        # 4. Mirror displacements to all 4 quadrants for perfect symmetry
        for nid, r in new_node_db.items():
            dx, dy = r[0] - cx, r[1] - cy
            if abs(r[2]) < 0.1 and abs(dx) < (inner_w/2) and abs(dy) < (inner_l/2):
                # Map this node to its reference peer in the first quadrant
                # We find the closest reference node by mapping abs(dx), abs(dy)
                # But since the mesh is transfinite/symmetric, we can find it by coordinate mapping.
                pass
                
        # Better Mirroring: Re-traverse the whole floor and map to reference quadrant result
        # To make it efficient, we build a coordinate map for the reference quadrant
        ref_coord_to_val = {}
        for nid in ref_nids:
            r = new_node_db[nid]
            # Key by rounded local coords to overcome float precision
            key = (round(r[0] - cx, 2), round(r[1] - cy, 2))
            ref_coord_to_val[key] = z_offsets[nid]
            
        for nid, r in new_node_db.items():
            dx, dy = r[0] - cx, r[1] - cy
            if abs(r[2]) < 0.1 and abs(dx) < (inner_w/2) and abs(dy) < (inner_l/2):
                key = (round(abs(dx), 2), round(abs(dy), 2))
                if key in ref_coord_to_val:
                    # Apply mirrored value with capping
                    val = ref_coord_to_val[key]
                    new_node_db[nid][2] += np.clip(val, -max_depth, max_depth)
                    
    return new_node_db


def get_nodes_in_box(
    nodes: Dict[int, np.ndarray],
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    z_range: Optional[Tuple[float, float]] = None
) -> List[int]:
    """Selection helper based on bounding box."""
    selected_ids = []
    for nid, r in nodes.items():
        if (x_range[0] <= r[0] <= x_range[1] if x_range else True) and \
           (y_range[0] <= r[1] <= y_range[1] if y_range else True) and \
           (z_range[0] <= r[2] <= z_range[1] if z_range else True):
            selected_ids.append(nid)
    return selected_ids
