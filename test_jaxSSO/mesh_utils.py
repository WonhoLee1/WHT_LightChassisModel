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
    node_data = {int(tag): nodes_xyz[i] for i, tag in enumerate(node_tags)}
    
    elem_data = {}
    dim_elem = 2
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim_elem)
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        num_v = 3 if etype == 2 else 4
        nodes_flat = enodes.reshape(-1, num_v)
        for i, tag in enumerate(etags):
            elem_data[int(tag)] = [int(n) for n in nodes_flat[i]]
            
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
    flanges: Tuple[bool, bool, bool, bool] = (True, True, True, True),
    thickness: float = 0.6
) -> Tuple[Dict[int, np.ndarray], Dict[int, List[int]]]:
    """
    Gmsh 압출 방식을 개선하여 쉘 메시를 솔리드 육면체(HEXA8) 메시로 변환합니다.
    단순 Z축 압출이 아닌, 법선 방향 압출을 모사하여 구배 각도에서도 균일한 두께를 유지합니다.

    Args:
        wall_layers (int): 두께 방향 요소 레이어 수.
        thickness (float): 트레이의 벽면 두께 [mm].
    """
    node_shell, elem_shell = generate_shell_tray(
        width=width, length=length, height=height,
        mesh_size_xy=mesh_size_xy, mesh_size_z=mesh_size_z,
        draft_angle=draft_angle, flange_segments=flange_segments,
        origin=origin, mesh_type='quad4', flanges=flanges
    )
    
    # 1. 노드별 법선 벡터 계산 (단순화를 위해 바닥면은 (0,0,1), 측벽은 경사각 고려)
    # 실제로는 주변 요소의 면 법선을 평균내어 사용함
    node_normals = {}
    rad_draft = math.radians(draft_angle)

    # [수정 1/5 — 법선 벡터 좌표계 불일치 수정]
    # 이전 코드의 문제:
    #   origin='center'일 때 x_rel = r[0] + width/2.0 으로 계산하면,
    #   예컨대 width=1800이고 -x 벽의 실제 x좌표가 약 -900mm일 때
    #   x_rel = -900 + 900 = 0mm → 임계치(0.1)에 걸리지 않아 벽면 법선이
    #   항상 (0,0,1)로 남음 → 두께 압출이 수직(Z)으로만 이루어져 비물리적 형상 발생.
    #
    # 수정 방법:
    #   x_rel을 트레이 내부 좌표(0 ~ width)로 정규화하지 않고,
    #   generate_shell_tray가 실제로 사용하는 절대 좌표 기준점(x0, y0)으로
    #   벽면 위치를 직접 판별한다.
    #   - x0 = 0 (corner) or -width/2 (center)
    #   - 벽면 x 좌표: x0(좌벽), x0+width(우벽) + wall_offset(구배 보정)
    #   허용 오차(tol)는 mesh_size_xy의 절반으로 설정하여 메시 크기 변화에 강건하게 대응.
    x0 = 0.0 if origin == 'corner' else -width / 2.0
    y0 = 0.0 if origin == 'corner' else -length / 2.0
    wall_offset = height * math.tan(rad_draft)
    # 벽면 절대 좌표 (구배로 인해 림(rim)이 base보다 바깥으로 나온 위치)
    x_left  = x0 - wall_offset   # 좌(-X) 림 x좌표
    x_right = x0 + width + wall_offset  # 우(+X) 림 x좌표
    y_front = y0 - wall_offset   # 전(-Y) 림 y좌표
    y_back  = y0 + length + wall_offset  # 후(+Y) 림 y좌표
    # 허용 오차: mesh_size_xy 절반 (고정값 0.1 대신 메시 크기에 비례)
    tol_wall = mesh_size_xy * 0.5

    for nid, r in node_shell.items():
        nx, ny, nz = 0.0, 0.0, 1.0  # 기본: 바닥면 수직 방향

        if abs(r[2]) > 0.1:  # 벽면 또는 플랜지 노드 (바닥 z=0 제외)
            # 각 벽면의 절대 위치에 대해 허용 오차 내에 있는지 판별
            # nx, ny 성분이 존재하면 nz도 구배각에 따라 재설정
            on_left  = r[0] < x_left  + tol_wall
            on_right = r[0] > x_right - tol_wall
            on_front = r[1] < y_front + tol_wall
            on_back  = r[1] > y_back  - tol_wall

            if on_left:  nx = -math.cos(rad_draft); nz = math.sin(rad_draft)
            if on_right: nx =  math.cos(rad_draft); nz = math.sin(rad_draft)
            if on_front: ny = -math.cos(rad_draft); nz = math.sin(rad_draft)
            if on_back:  ny =  math.cos(rad_draft); nz = math.sin(rad_draft)

        norm = math.sqrt(nx**2 + ny**2 + nz**2)
        node_normals[nid] = np.array([nx/norm, ny/norm, nz/norm])

    # 2. 노드 스택 생성 (법선 방향으로 압출)
    node_data_3d = {}
    num_nodes_shell = len(node_shell)
    for layer in range(wall_layers + 1):
        dist = (layer / wall_layers) * thickness
        for nid, r in node_shell.items():
            new_id = nid + layer * num_nodes_shell
            node_data_3d[new_id] = r + node_normals[nid] * dist
            
    # 3. HEXA8 요소 생성
    elem_data_3d = {}
    e_idx = 1
    for eid, nids in elem_shell.items():
        if len(nids) != 4: continue 
        for layer in range(wall_layers):
            b1, b2, b3, b4 = [n + layer * num_nodes_shell for n in nids]
            t1, t2, t3, t4 = [n + (layer + 1) * num_nodes_shell for n in nids]
            elem_data_3d[e_idx] = [b1, b2, b3, b4, t1, t2, t3, t4]
            e_idx += 1
            
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
    bead_direction: float = 0.0, # 1.0: Up, -1.0: Down, 0.0: Both
    **kwargs
) -> Dict[int, np.ndarray]:
    """
    Applies topography bead patterns to the tray floor.
    
    Modes:
        'grid': Symmetric local rectangular patches (structured).
        'rib': Continuous intersecting stiffening ribs (X & Y directions).
        'network': Randomly branched interconnected rib network (organic feel).
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
        # --- Structured Rectangular Patches (Topography Style) ---
        freq_x, freq_y = 2.0, 3.0  # Number of beads per quadrant
        bead_w = (inner_w / 2.0) / (freq_x + 0.5)
        bead_h = (inner_l / 2.0) / (freq_y + 0.5)
        
        for nid, r in new_node_db.items():
            if abs(r[2]) < 0.1 and abs(r[0] - cx) < (inner_w/2) and abs(r[1] - cy) < (inner_l/2):
                nx = (r[0] - cx) / (inner_w/2)
                ny = (r[1] - cy) / (inner_l/2)
                
                # Create a grid of points
                # Use absolute coordinates for local patches
                lx, ly = abs(r[0] - cx), abs(r[1] - cy)
                
                # Determine which "cell" we are in
                ix = int(lx / (bead_w * 1.5))
                iy = int(ly / (bead_h * 1.5))
                
                # Center of the nearest bead
                target_x = (ix + 0.5) * bead_w * 1.5
                target_y = (iy + 0.5) * bead_h * 1.5
                
                # Distance to bead center
                dx = abs(lx - target_x)
                dy = abs(ly - target_y)
                
                # Trapezoidal profile (Flat top with ramps)
                ramp = 10.0 # 10mm ramp
                wx = max(0, min(1, (bead_w/2 - dx) / ramp))
                wy = max(0, min(1, (bead_h/2 - dy) / ramp))
                
                shape = wx * wy
                if shape > 0:
                    # Pattern polarity (checkerboard feel or fixed)
                    if bead_direction != 0.0:
                        polarity = np.sign(bead_direction)
                    else:
                        polarity = 1.0 if (ix + iy) % 2 == 0 else -1.0
                    new_node_db[nid][2] += max_depth * polarity * (shape ** 1.2)
                    
    elif mode == 'rib':
        # --- Continuous Intersecting Ribs ---
        # freq_x/y defines how many ribs in each direction
        freq_x, freq_y = 4, 3 
        pitch_x = (inner_w / 2.0) / (freq_x + 0.5)
        pitch_y = (inner_l / 2.0) / (freq_y + 0.5)
        rib_width = 30.0 # 30mm width
        ramp = 10.0      # 10mm ramp
        
        for nid, r in new_node_db.items():
            if abs(r[2]) < 0.1 and abs(r[0] - cx) < (inner_w/2) and abs(r[1] - cy) < (inner_l/2):
                lx, ly = abs(r[0] - cx), abs(r[1] - cy)
                
                # Closest X-rib and Y-rib indices
                ix = round(lx / pitch_x)
                iy = round(ly / pitch_y)
                
                # Distance to rib center lines
                dx = abs(lx - ix * pitch_x)
                dy = abs(ly - iy * pitch_y)
                
                # Rib profiles
                wx = max(0, min(1, (rib_width/2 - dx) / ramp))
                wy = max(0, min(1, (rib_width/2 - dy) / ramp))
                
                # Connection logic: UNION of X and Y ribs
                shape = max(wx, wy)
                
                if shape > 0:
                    if bead_direction != 0.0:
                        polarity = np.sign(bead_direction)
                    else:
                        # Fixed polarity for ribs usually looks better than alternating
                        polarity = 1.0
                    new_node_db[nid][2] += max_depth * polarity * (shape ** 1.1)
                    
    elif mode == 'network':
        # --- Randomly Branched Interconnected Network ---
        np.random.seed(42)
        num_seeds = 12
        k_neighbors = 2
        rib_width = 25.0
        ramp = 10.0
        
        # 1. Generate random seeds in the FIRST quadrant
        seeds = []
        for _ in range(num_seeds):
            sx = np.random.uniform(0, inner_w/2)
            sy = np.random.uniform(0, inner_l/2)
            seeds.append(np.array([sx, sy]))
        
        # 2. Build edges (connect each seed to k nearest neighbors)
        edges = []
        for i, p1 in enumerate(seeds):
            dists = [np.linalg.norm(p1 - p2) for p2 in seeds]
            nearest_indices = np.argsort(dists)[1:k_neighbors+1]
            for idx in nearest_indices:
                if i < idx: # Avoid duplicates
                    edges.append((p1, seeds[idx]))
        
        # 3. Add some boundary connections to ensure global connectivity if needed
        # (Optional: connect seeds to center/edges)
        
        # 4. Calculate distance to nearest edge for each node
        def dist_to_segment(p, a, b):
            ap = p - a
            ab = b - a
            t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
            nearest = a + t * ab
            return np.linalg.norm(p - nearest)

        for nid, r in new_node_db.items():
            if abs(r[2]) < 0.1 and abs(r[0] - cx) < (inner_w/2) and abs(r[1] - cy) < (inner_l/2):
                lx, ly = abs(r[0] - cx), abs(r[1] - cy)
                p = np.array([lx, ly])
                
                min_d = 1e9
                for a, b in edges:
                    min_d = min(min_d, dist_to_segment(p, a, b))
                
                # Trapezoidal profile
                shape = max(0, min(1, (rib_width/2 - min_d) / ramp))
                
                if shape > 0:
                    if bead_direction != 0.0:
                        polarity = np.sign(bead_direction)
                    else:
                        polarity = 1.0
                    new_node_db[nid][2] += max_depth * polarity * (shape ** 1.1)

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
        # [수정 4/5 — while 루프 silent fail 방지]
        # 이전 코드의 문제:
        #   max_iters(500) 초과로 루프가 조기 종료되어도 경고 없이 반환되었다.
        #   비드 크기가 너무 작거나, margin이 너무 커서 유효 영역이 협소하면
        #   target_count에 절대 도달하지 못해 항상 500회 낭비 후 조용히 실패.
        #   사용자는 비드가 의도대로 생성됐다고 오해할 수 있다.
        #
        # 수정 방법:
        #   루프 종료 후 목표 달성 여부를 확인하여 경고를 출력한다.
        #   루프 종료 조건(목표 달성 vs. 반복 초과)을 명확히 분리하기 위해
        #   iter_count를 루프 밖에서 검사한다.
        max_iters = 500
        iter_count = 0

        while len(morphed_set) < target_count and iter_count < max_iters:
            iter_count += 1
            
            # Random Rectangle in the reference quadrant
            # Center coordinates relative to the quadrant [0, 1]
            rx = np.random.uniform(0, inner_w/2)
            ry = np.random.uniform(0, inner_l/2)
            
            # Random Size (Absolute mm values from kwargs)
            min_sz = kwargs.get('min_size', 50.0)  # Default 50mm
            max_sz = kwargs.get('max_size', 200.0) # Default 200mm
            
            rw = np.random.uniform(min_sz, max_sz)
            rh = np.random.uniform(min_sz, max_sz)
            
            # Random Height/Depth
            min_d = kwargs.get('min_depth', max_depth * 0.3)
            rz = np.random.uniform(min_d, max_depth)
            
            if bead_direction != 0.0:
                polarity = np.sign(bead_direction)
            else:
                polarity = np.random.choice([-1.0, 1.0])
            
            # [수정 3/5 — z_offsets 누적 시 clip 적용]
            # 이전 코드의 문제:
            #   여러 사각형 패치가 동일 노드에 중첩될 경우 z_offsets[nid]가
            #   누적 합산되어 max_depth를 초과할 수 있다.
            #   최종 적용(new_node_db 반영) 단계에서만 clip을 수행하면,
            #   중간 누적값이 이미 포화 상태여도 그 사실을 알 수 없고
            #   패턴 형태(극성, 크기)가 왜곡된다.
            #
            # 수정 방법:
            #   값을 누적하는 시점마다 즉시 [-max_depth, +max_depth] 범위로 clip한다.
            #   이렇게 하면 어떤 순서로 패치가 중첩되더라도 최종값이 의도한
            #   깊이 범위를 벗어나지 않으며, 포화 시 추가 누적이 무의미함을 방지한다.
            for nid in ref_nids:
                r = new_node_db[nid]
                dx, dy = abs(r[0] - cx - rx), abs(r[1] - cy - ry)

                # Trapezoidal profile (Flat top with ramps)
                ramp = 10.0 # 10mm ramp
                wx = max(0, min(1, (rw/2 - dx) / ramp))
                wy = max(0, min(1, (rh/2 - dy) / ramp))
                shape = wx * wy

                if shape > 0:
                    raw = z_offsets[nid] + rz * polarity * (shape ** 1.2)
                    z_offsets[nid] = float(np.clip(raw, -max_depth, max_depth))
                    morphed_set.add(nid)

        # 루프 종료 후 목표 달성 여부 검사 (수정 4/5 연속)
        if iter_count >= max_iters and len(morphed_set) < target_count:
            achieved = len(morphed_set) / max(len(ref_nids), 1)
            print(f"[WARNING] apply_auto_beads: max_iters({max_iters}) 초과로 루프 조기 종료. "
                  f"달성 비율={achieved:.1%} (목표={target_ratio:.1%}). "
                  f"bead_min_size 축소 또는 bead_margin 축소를 권장.")

        # 4. Mirror displacements to all 4 quadrants for perfect symmetry
        # KDTree를 사용하여 대칭 위치의 가장 가까운 노드를 찾아 매핑함 (수치 오차 극복)
        from scipy.spatial import KDTree
        
        # 전체 바닥면 노드 식별
        floor_nids = []
        floor_coords = []
        for nid, r in new_node_db.items():
            if abs(r[2]) < 0.1 and abs(r[0] - cx) < (inner_w/2) and abs(r[1] - cy) < (inner_l/2):
                floor_nids.append(nid)
                floor_coords.append([r[0], r[1]])
                
        if not floor_nids: return new_node_db
        
        # 참조 Quadrant (1사분면)의 노드들에 대해 KDTree 구축
        ref_coords = []
        ref_vals = []
        for nid in ref_nids:
            r = new_node_db[nid]
            ref_coords.append([abs(r[0] - cx), abs(r[1] - cy)])
            ref_vals.append(z_offsets[nid])
            
        tree = KDTree(ref_coords)
        
        # 모든 바닥 노드에 대해 대칭 매핑 적용
        for nid in floor_nids:
            r = new_node_db[nid]
            # 해당 노드의 1사분면 대응 위치 계산
            local_x, local_y = abs(r[0] - cx), abs(r[1] - cy)
            
            # KDTree로 가장 가까운 참조 노드 인덱스 검색
            dist, idx = tree.query([local_x, local_y])
            
            # 매핑 성공 시 비드 높이 적용 (오차 방지를 위해 거리 제한 1.0mm)
            if dist < 1.0:
                val = ref_vals[idx]
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
