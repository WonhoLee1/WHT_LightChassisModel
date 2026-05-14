# -*- coding: utf-8 -*-
"""
wht_dynamic_utils.py
====================
CSV 실측 데이터 처리 및 동적 하중 추출을 위한 유틸리티 함수 모음.

CSV 형식
--------
  # chassis, width, length, height
  # start_time, 0.0
  # C1, x_mm, y_mm, z_mm
  ...
  # C8, x_mm, y_mm, z_mm
  Frame,Time,C1_X,C1_Y,C1_Z,...,C8_X,C8_Y,C8_Z
  0,0.0,...
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_csv_header(csv_path: str) -> Dict:
    """
    CSV 파일의 # 주석 헤더에서 코너 기준 좌표(mm)와 start_time(s)을 파싱합니다.

    Returns
    -------
    dict
        'corner_positions': {name: (x_mm, y_mm, z_mm), ...}  — C1~C8
        'start_time'      : float | None
    """
    corner_positions: Dict[str, Tuple[float, float, float]] = {}
    start_time: Optional[float] = None
    with open(Path(csv_path), encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith('#'):
                break
            parts = [p.strip() for p in stripped[1:].split(',')]
            if not parts:
                continue
            key = parts[0].lower().replace(' ', '_')
            if key == 'start_time' and len(parts) >= 2:
                start_time = float(parts[1])
            elif re.fullmatch(r'C[1-8]', parts[0]) and len(parts) >= 4:
                corner_positions[parts[0]] = (
                    float(parts[1]), float(parts[2]), float(parts[3])
                )
    return {'corner_positions': corner_positions, 'start_time': start_time}


def find_nodes_for_corners(
    node_db: Dict[int, List[float]],
    corner_positions: Dict[str, Tuple[float, float, float]],
    n_nodes: int = 3,
) -> Dict[str, List[int]]:
    """
    각 코너 기준 좌표에서 가장 가까운 FEM 노드를 n_nodes개씩 탐색합니다.

    전역적으로 중복 없이 그리디 할당 (거리 오름차순 처리).

    Parameters
    ----------
    node_db          : {nid: [x, y, z]}
    corner_positions : {corner_name: (x_mm, y_mm, z_mm)}
    n_nodes          : 코너당 할당 노드 수 (기본: 3)

    Returns
    -------
    {corner_name: [nid, ...]}
    """
    nid_arr = np.array(list(node_db.keys()), dtype=int)
    xyz_arr = np.array(list(node_db.values()))
    used: set = set()
    result: Dict[str, List[int]] = {}
    for name, (cx, cy, cz) in corner_positions.items():
        dists2 = ((xyz_arr[:, 0] - cx) ** 2 +
                  (xyz_arr[:, 1] - cy) ** 2 +
                  (xyz_arr[:, 2] - cz) ** 2)
        chosen: List[int] = []
        for idx in np.argsort(dists2):
            nid = int(nid_arr[idx])
            if nid not in used:
                chosen.append(nid)
                used.add(nid)
                if len(chosen) == n_nodes:
                    break
        result[name] = chosen
    return result


class InterpLoadGroup:
    """시계열 데이터를 선형 보간하여 하중/변위를 반환하는 하중 그룹."""
    def __init__(self, node_ids: List[int], dof: int, time_arr: np.ndarray, val_arr: np.ndarray, load_type: str = "SPCD"):
        self.node_ids  = node_ids
        self.dof       = dof
        self.load_type = load_type.upper()
        self._t = time_arr
        self._u = val_arr

    def evaluate(self, t: float) -> float:
        return float(np.interp(t, self._t, self._u))

    def u_value(self, t: float) -> float:
        return self.evaluate(t)

    def ud_value(self, t: float) -> float:
        """속도 (중앙 차분)"""
        eps = 1e-6
        return (self.evaluate(t + eps) - self.evaluate(t - eps)) / (2 * eps)

    def udd_value(self, t: float) -> float:
        """가속도 (중앙 차분)"""
        eps = 1e-6
        return (self.evaluate(t + eps) - 2 * self.evaluate(t) +
                self.evaluate(t - eps)) / (eps ** 2)


def find_corner_nodes(node_db: Dict[int, List[float]], 
                      width: float, length: float,
                      radius: float, z_min: float, z_max: float) -> List[Tuple[Tuple[float, float], List[int]]]:
    """
    4 코너 기준으로 반경 내 + z 범위 노드 그룹 반환.
    순서: C5(+X,+Y), C6(+X,-Y), C7(-X,-Y), C8(-X,+Y)
    
    Returns
    -------
    List[Tuple[Tuple[float, float], List[int]]]
        [((cx, cy), [nid, ...]), ...]
    """
    all_xyz = np.array([v for v in node_db.values()])
    mask_z  = (all_xyz[:, 2] >= z_min) & (all_xyz[:, 2] <= z_max)
    xyz_z   = all_xyz[mask_z]

    if len(xyz_z) == 0:
        raise RuntimeError(f"z=[{z_min},{z_max}]mm 범위 내 노드가 없습니다.")

    x_min, x_max = xyz_z[:, 0].min(), xyz_z[:, 0].max()
    y_min, y_max = xyz_z[:, 1].min(), xyz_z[:, 1].max()

    targets = [
        (x_max, y_max),  # C5: +X, +Y
        (x_max, y_min),  # C6: +X, -Y
        (x_min, y_min),  # C7: -X, -Y
        (x_min, y_max),  # C8: -X, +Y
    ]

    nid_arr = np.array(list(node_db.keys()))
    xyz_arr = np.array(list(node_db.values()))

    groups = []
    for cx, cy in targets:
        in_z   = (xyz_arr[:, 2] >= z_min) & (xyz_arr[:, 2] <= z_max)
        in_r   = (xyz_arr[:, 0] - cx) ** 2 + (xyz_arr[:, 1] - cy) ** 2 < radius ** 2
        mask   = in_z & in_r
        nids   = nid_arr[mask].tolist()
        if not nids:
            # 완화된 조건으로 재시도 (가장 가까운 노드 1개라도 찾기)
            dists = (xyz_arr[:, 0] - cx) ** 2 + (xyz_arr[:, 1] - cy) ** 2
            nearest_idx = np.argmin(dists)
            nids = [int(nid_arr[nearest_idx])]
            
        groups.append(((cx, cy), nids))
    return groups


def calculate_local_z_history(csv_df, time_arr: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    4개 코너(C5, C6, C7, C8)의 3D 궤적으로부터 강체 회전을 제거한 후,
    샤시 로컬 좌표계 기준의 순수 수직(Z) 변위만을 추출합니다.
    
    Returns
    -------
    Dict[str, np.ndarray]
        { 'C5': z_arr, 'C6': z_arr, ... } (T,)
    np.ndarray
        (T, 4, 3) Trajectory in mm
    """
    n_steps = len(time_arr)
    corner_labels = ['C5', 'C6', 'C7', 'C8']
    
    # 1. (T, 4, 3) 궤적 데이터 구축 (mm 단위)
    traj = np.zeros((n_steps, 4, 3))
    for i, lbl in enumerate(corner_labels):
        for j, ax in enumerate(['X', 'Y', 'Z']):
            col = f"{lbl}_{ax}" if f"{lbl}_{ax}" in csv_df.columns else f"{lbl}_pos_{ax}"
            if col in csv_df.columns:
                traj[:, i, j] = csv_df[col].to_numpy(dtype=float) * 1000.0

    local_z_results = {lbl: np.zeros(n_steps) for lbl in corner_labels}
    p_loc_t0 = None

    for t in range(n_steps):
        pts = traj[t] # (4, 3)
        origin = np.mean(pts, axis=0)
        p_c = pts - origin
        
        # X축 방향: C7->C6 및 C8->C5의 평균
        v_x = ( (p_c[1] - p_c[2]) + (p_c[0] - p_c[3]) ) / 2.0
        # Y축 방향: C7->C8 및 C6->C5의 평균
        v_y = ( (p_c[3] - p_c[2]) + (p_c[0] - p_c[1]) ) / 2.0
        
        z_loc = np.cross(v_x, v_y)
        z_norm = np.linalg.norm(z_loc)
        if z_norm < 1e-12:
            z_loc = np.array([0, 0, 1.0])
        else:
            z_loc /= z_norm
            
        if z_loc @ np.array([0, 0, 1.0]) < 0:
            z_loc = -z_loc
            
        x_loc = v_x / (np.linalg.norm(v_x) + 1e-12)
        y_loc = np.cross(z_loc, x_loc)
        x_loc = np.cross(y_loc, z_loc)
        
        R = np.stack([x_loc, y_loc, z_loc], axis=1) # (3, 3)
        p_loc = p_c @ R # (4, 3)
        
        if t == 0:
            p_loc_t0 = p_loc.copy()
            
        delta_p_loc = p_loc - p_loc_t0
        for i, lbl in enumerate(corner_labels):
            local_z_results[lbl][t] = delta_p_loc[i, 2]

    return local_z_results, traj


def calculate_corner_accelerations(traj: np.ndarray, dt: float) -> np.ndarray:
    """
    (T, 4, 3) 궤적 데이터로부터 4개 코너 각각의 Z 가속도를 산출합니다.
    """
    n_steps = traj.shape[0]
    accels = np.zeros((n_steps, 4))
    
    def smooth(y, box_pts=5):
        box = np.ones(box_pts)/box_pts
        y_s = np.convolve(y, box, mode='same')
        y_s[:box_pts] = y[:box_pts]
        y_s[-box_pts:] = y[-box_pts:]
        return y_s

    for i in range(4):
        z = traj[:, i, 2]
        z_s = smooth(z)
        v_z = np.gradient(z_s, dt)
        v_s = smooth(v_z)
        a_z = np.gradient(v_s, dt)
        accels[:, i] = smooth(a_z)
        
    return accels
