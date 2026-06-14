# -*- coding: utf-8 -*-
"""
exam5_dynamic_with_oc.py — 섀시 + 오픈셀(글라스 패널) Soft 결합 동적 해석 및 시각화 검증
====================================================================================

[해석 파이프라인]
  [1] 섀시 기본 메시 생성 (generate_shell_tray)
  [2] 플랜지 최종단 노드 자동 검색 (Z_max 기준)
  [3] 글라스 패널(오픈셀) 2D Grid 메시 생성 (Z_glass = Z_max + 10.0 mm 시각적 갭)
  [4] 플랜지-글라스 간 최단거리 1:1 매핑 및 BEAM2 소프트 빔 연결 요소 생성 (E=10 MPa)
  [5] 섀시 + 글라스 패널 + 소프트 빔 어셈블리 모델 병합 및 물성/속성 등록
  [6] CSV 실측 데이터(structural_dynamics_c235.csv) 기반 SPCD 하중 및 분포 관성 하중 인가
  [7] 고유치 해석(solve_modal) 및 직접 과도 응답 해석(solve_direct_dynamic) 수행
  [8] ParaView HDF 내보내기 및 WHTVisualizer 3D 시각화 구동
"""

import sys
import re
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# 루트 경로를 sys.path에 추가하여 패키지 임포트 가능하도록 설정
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_dynamic_solver import WHTDynamicSolver
from wht_solver.wht_dynamic_common import DampingSpec, DynamicResult
from wht_solver.load_cases import WHTLoadCase
from wht_modeler.wht_dynamic_utils import (
    find_corner_nodes,
    find_nodes_for_corners,
    parse_csv_header,
    InterpLoadGroup,
    KabschPreprocessor,
    compute_vk_inertia_scale,
    print_vk_scale_report,
    compute_inertia_scale_via_fem,
)
from wht_converter.wht_models import WHTMetadata, WHTResultData
from wht_converter.wht_exporters import VTKHDFExporter
from wht_visualizer.wht_visualizer import WHTVisualizer
from test_jaxSSO.mesh_utils import generate_shell_tray

# 한글 깨짐 방지 폰트 설정
try:
    import koreanize_matplotlib
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# 해석용 모듈 상수
# ─────────────────────────────────────────────────────────────────────────────
MAT_CHASSIS = dict(E=210000.0, nu=0.3, rho=7.85e-9, t=1.2)  # 스틸 섀시 물성

# 섀시 기하 정보
WIDTH, LENGTH, HEIGHT = 1600.0, 1200.0, 30.0
MESH_XY, MESH_Z      = 30.0, 10.0
DRAFT, FLANGE        = 10.0, 15.0

# ─────────────────────────────────────────────────────────────────────────────
# 섀시 + 오픈셀 글라스 어셈블리 빌더 함수
# ─────────────────────────────────────────────────────────────────────────────

def _build_chassis_with_glass_assembly(
    width: float = WIDTH,
    length: float = LENGTH,
    height: float = HEIGHT,
    mesh_xy: float = MESH_XY,
    mesh_z: float = MESH_Z,
    draft_angle: float = DRAFT,
    flange_width: float = FLANGE,
    glass_t: float = 1.0,      # 사용자 입력 글라스 두께
    glass_E: float = 40000.0,   # 글라스 탄성계수 (40,000 MPa)
    sealant_E: float = 1.0,    # 소프트 결합 빔의 탄성계수 (1.0 MPa)
    chassis_mass: float = 20.0, # 섀시 목표 질량 (kg)
) -> Tuple[WHTMeshModel, dict]:
    """
    섀시 트레이 메시를 생성하고, 플랜지 영역에 맞게 글라스 패널을 메싱한 뒤 
    소프트 빔 요소(BEAM2)로 연결하여 결합된 단일 WHTMeshModel을 빌드합니다.

    Parameters
    ----------
    width, length, height : 섀시 치수 (mm)
    mesh_xy, mesh_z : 메시 크기 (mm)
    draft_angle, flange_width : 구배각(deg), 플랜지 폭(mm)
    glass_t : 글라스 패널 두께 (mm)
    glass_E : 글라스 탄성계수 (MPa)
    sealant_E : 소프트 결합 빔의 탄성계수 (MPa)
    chassis_mass : 섀시 목표 질량 (kg)

    Returns
    -------
    model : WHTMeshModel
        결합된 어셈블리 모델
    node_db_combined : dict
        전체 노드 {nid: coords} 딕셔너리
    """
    print(" -> [Mesh Setup] 섀시 기본 쉘 메시 생성...")
    # 1. 섀시 기본 쉘 생성
    # flanges = (Y-min, X-max, Y-max, X-min) 중 전면(Y-min) 제외 3개면에만 플랜지 적용
    chassis_nodes, chassis_elems = generate_shell_tray(
        width=width, length=length, height=height,
        mesh_size_xy=mesh_xy, mesh_size_z=mesh_z,
        draft_angle=draft_angle, flange_width=flange_width,
        origin='center',
        flanges=(False, True, True, True),
        mesh_type='quad4'
    )
    
    # 섀시 총 면적 계산
    total_area = 0.0
    for nodes_in_elem in chassis_elems.values():
        if len(nodes_in_elem) == 4:
            p0 = chassis_nodes[nodes_in_elem[0]]
            p1 = chassis_nodes[nodes_in_elem[1]]
            p2 = chassis_nodes[nodes_in_elem[2]]
            p3 = chassis_nodes[nodes_in_elem[3]]
            area1 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            area2 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0))
            total_area += (area1 + area2)
        elif len(nodes_in_elem) == 3:
            p0 = chassis_nodes[nodes_in_elem[0]]
            p1 = chassis_nodes[nodes_in_elem[1]]
            p2 = chassis_nodes[nodes_in_elem[2]]
            total_area += 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            
    # 섀시 질량이 chassis_mass가 되도록 밀도 역산 (kg -> tonne = *1e-3)
    t_val = MAT_CHASSIS['t']
    rho_chassis = (chassis_mass * 1e-3) / (total_area * t_val)
    print(f"    - [Mass Calibration] 섀시 총 면적: {total_area:.2f} mm^2, 두께: {t_val:.1f} mm")
    print(f"    - 목표 질량: {chassis_mass:.1f} kg => 조정된 섀시 밀도 (rho): {rho_chassis:.5e} tonne/mm^3")
    
    # 섀시 WHTMeshModel 인스턴스 초기화 및 속성 부여
    model = WHTMeshModel.from_node_elem_db(chassis_nodes, chassis_elems, name="TrayWithGlass", is_solid=False)
    model.add_material(1, E=MAT_CHASSIS['E'], nu=MAT_CHASSIS['nu'], rho=rho_chassis)
    model.add_property(1, "PSHELL", t=MAT_CHASSIS['t'], mid=1)
    for eid in model.elements:
        model.elements[eid].pid = 1

    # 2. 플랜지 최종단 노드 자동 검색 (Z_max 기준)
    # 섀시 쉘 노드 중 최대 Z 좌표를 플랜지 최종단의 Z 좌표로 판단합니다.
    coords_arr = np.array(list(chassis_nodes.values()))
    z_max = float(coords_arr[:, 2].max())
    print(f"    - 플랜지 최종단 Z 좌표 감지: {z_max:.2f} mm")
    
    flange_nids = []
    for nid, coords in chassis_nodes.items():
        if abs(coords[2] - z_max) < 0.5:
            flange_nids.append(nid)
    print(f"    - 플랜지 최종단 노드 수: {len(flange_nids)}개")

    # 3. 글라스 패널 생성 범위를 위한 플랜지 노드의 X, Y 바운딩 박스 계산
    flange_coords = np.array([chassis_nodes[nid] for nid in flange_nids])
    x_min, x_max = flange_coords[:, 0].min(), flange_coords[:, 0].max()
    y_min, y_max = flange_coords[:, 1].min(), flange_coords[:, 1].max()
    
    glass_w = x_max - x_min
    glass_l = y_max - y_min
    z_glass = z_max + 10.0  # 10mm 시각적 갭 부여 (실란트 연결부 가시화 극대화)
    print(f"    - 글라스 크기 계산: {glass_w:.1f} x {glass_l:.1f} mm, 위치 Z = {z_glass:.1f} mm")

    # 4. 글라스 패널(평판) 메싱
    nx = int(round(glass_w / mesh_xy))
    ny = int(round(glass_l / mesh_xy))
    dx = glass_w / nx
    dy = glass_l / ny
    print(f"    - 글라스 격자 분할 수: Nx={nx}, Ny={ny} (요소 크기 dx={dx:.1f}, dy={dy:.1f} mm)")

    # 글라스 노드 생성 (노드 ID 200,000번 대 오프셋)
    glass_node_offset = 200000
    glass_node_db = {}
    for i in range(nx + 1):
        x = x_min + i * dx
        for j in range(ny + 1):
            y = y_min + j * dy
            nid = glass_node_offset + i * (ny + 1) + j
            model.add_node(nid, x, y, z_glass)
            glass_node_db[nid] = np.array([x, y, z_glass])

    # 글라스 요소 생성 (QUAD4, 요소 ID 200,000번 대 오프셋)
    glass_elem_offset = 200000
    model.add_material(3, E=glass_E, nu=0.2, rho=2.5e-9)  # 글라스 밀도 ~2.5 g/cm³
    model.add_property(3, "PSHELL", t=glass_t, mid=3)
    
    for i in range(nx):
        for j in range(ny):
            n1 = glass_node_offset + i * (ny + 1) + j
            n2 = glass_node_offset + (i + 1) * (ny + 1) + j
            n3 = glass_node_offset + (i + 1) * (ny + 1) + (j + 1)
            n4 = glass_node_offset + i * (ny + 1) + (j + 1)
            
            eid = glass_elem_offset + i * ny + j
            model.add_element(eid, [n1, n2, n3, n4], elem_type="QUAD4", pid=3)

    # 5. 소프트 빔 결합 생성 (BEAM2, 요소 ID 300,000번 대 오프셋)
    # 각 플랜지 노드에 대해 XY 투영상에서 가장 가까운 글라스 경계 노드를 탐색
    beam_elem_offset = 300000
    model.add_material(2, E=sealant_E, nu=0.49, rho=1e-15)  # 질량 영향이 없는 소프트 접착제 물성
    model.add_property(2, "PBEAM", t=0.0, mid=2)  # BEAM2용 pid=2 등록 (PBEAM 속성)
    
    # 글라스 평판의 경계(가장자리) 노드들 추출
    glass_boundary_nids = []
    for nid, coords in glass_node_db.items():
        # X나 Y가 외곽 경계선에 있는 노드 식별 (Y-min은 체결에서 제외)
        on_boundary = (abs(coords[0] - x_min) < 0.1 or abs(coords[0] - x_max) < 0.1 or
                       abs(coords[1] - y_max) < 0.1)
        if on_boundary:
            glass_boundary_nids.append(nid)

    beam_count = 0
    max_link_dist = 1.5 * mesh_xy  # 플랜지가 없는 전면부에 연결이 생성되지 않도록 거리 임계치 설정
    
    for fnid in flange_nids:
        fc = chassis_nodes[fnid]
        # XY면 기준 최단 거리 탐색
        best_gnid = None
        min_d_sq = 1e9
        for gnid in glass_boundary_nids:
            gc = glass_node_db[gnid]
            d_sq = (fc[0] - gc[0])**2 + (fc[1] - gc[1])**2
            if d_sq < min_d_sq:
                min_d_sq = d_sq
                best_gnid = gnid
                
        if best_gnid is not None and np.sqrt(min_d_sq) < max_link_dist:
            eid = beam_elem_offset + beam_count
            model.add_element(eid, [fnid, best_gnid], elem_type="BEAM2", pid=2)
            beam_count += 1

    print(f"    - 소프트 연결 빔(BEAM2) 생성: {beam_count}개 연결 완료")
    
    # 6. 전체 노드 딕셔너리 병합
    node_db_combined = {**chassis_nodes, **glass_node_db}
    
    # 시각화 검증을 위한 파트 분할용 Element Set 정의
    chassis_eids = list(chassis_elems.keys())
    glass_eids = [glass_elem_offset + i * ny + j for i in range(nx) for j in range(ny)]
    beam_eids = [beam_elem_offset + b for b in range(beam_count)]
    
    model.add_elem_set_by_name("ChassisShells", chassis_eids)
    model.add_elem_set_by_name("GlassPanel", glass_eids)
    model.add_elem_set_by_name("SoftBeams", beam_eids)
    
    return model, node_db_combined

# ─────────────────────────────────────────────────────────────────────────────
# 동적 해석 파이프라인 구현
# ─────────────────────────────────────────────────────────────────────────────

class GlassAssemblyPipeline:
    """오픈셀 글라스가 소프트 결합된 섀시의 과도 응답 해석 파이프라인."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        stamp = datetime.now().strftime("D%Y%m%d_%H%M%S")
        self.out_dir = Path(__file__).resolve().parent.parent / "results" / f"oc_test_{stamp}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.node_db = None
        self.bot_groups = None
        self.csv_df = None
        self.csv_header = {}
        self.time_arr = None
        self.load_groups = []
        self.kabsch = None
        self.solver = None
        self.dyn = None
        self.wht_data = None

    def run(self) -> None:
        """전체 과정을 순차 구동합니다."""
        self._load_csv()
        self._build_assembly()
        self._find_corners()
        self._build_load_groups()
        self._run_modal_analysis()
        self._run_dynamic()
        self._recover_stress()
        self._export_results()
        self._visualize()

    def _load_csv(self) -> None:
        """CSV 위치 데이터를 읽고 전처리용 시간을 파싱합니다."""
        csv_path = Path(self.cfg.pos_data)
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        print(f"\n [1] CSV 위치 데이터 로드: {csv_path}")
        
        self.csv_header = parse_csv_header(str(csv_path))
        t_start = self.csv_header.get('start_time') or 0.0
        
        df = pd.read_csv(csv_path, comment='#', encoding='utf-8')
        df.columns = [c.replace('Chassis_', '') for c in df.columns]
        
        if t_start > 0:
            df = df[df['Time'] >= t_start].copy()
            
        time_raw = df['Time'].to_numpy(dtype=float)
        self.csv_df = df
        self.time_arr = time_raw - time_raw[0]
        print(f"    - 시간 범위: 0.0s ~ {self.time_arr[-1]:.4f}s ({len(self.time_arr)} frames)")

    def _build_assembly(self) -> None:
        """섀시 + 글라스 평판 메시를 생성 및 soft 빔으로 결합합니다."""
        print("\n [2] 섀시 + 오픈셀 글라스 어셈블리 생성 및 결합...")
        self.model, self.node_db = _build_chassis_with_glass_assembly(
            width=WIDTH, length=LENGTH, height=HEIGHT,
            mesh_xy=MESH_XY, mesh_z=MESH_Z,
            draft_angle=DRAFT, flange_width=FLANGE,
            glass_t=self.cfg.glass_t,
            glass_E=self.cfg.glass_E,
            sealant_E=self.cfg.sealant_E,
            chassis_mass=self.cfg.chassis_mass
        )
        print(f"    - 어셈블리 총 노드 수: {self.model.n_nodes}개, 총 요소 수: {self.model.n_elements}개")

    def _find_corners(self) -> None:
        """4코너 노드 영역(C5~C8)을 탐색합니다."""
        print("\n [3] 섀시 코너 노드 탐색...")
        header_corners = self.csv_header.get('corner_positions', {})
        c5c8 = {k: v for k, v in header_corners.items() if k in ('C5', 'C6', 'C7', 'C8')}
        if c5c8:
            corner_nodes = find_nodes_for_corners(self.node_db, c5c8, n_nodes=3)
            self.bot_groups = [
                (name, corner_nodes[name])
                for name in ['C5', 'C6', 'C7', 'C8'] if name in corner_nodes
            ]
            for name, nids in self.bot_groups:
                print(f"    - {name} 코너: {len(nids)}개 노드 바인딩")
        else:
            # 섀시 영역만 기준으로 코너 탐색
            chassis_nodes = {nid: coord for nid, coord in self.node_db.items() if nid < 200000}
            raw = find_corner_nodes(chassis_nodes, WIDTH, LENGTH, CORNER_RADIUS, z_min=0.0, z_max=2.0)
            self.bot_groups = [(CORNER_MAP[i], g[1]) for i, g in enumerate(raw)]
            print("    - [WARN] CSV 헤더 코너 좌표가 없어 기하 기반 코너 노드 강제 탐색")

    def _build_load_groups(self) -> None:
        """Kabsch 변위 제거를 통한 3-DOF SPCD 및 관성 가속도 하중을 생성합니다."""
        print("\n [4] Kabsch 변위 전처리 및 SPCD 하중 생성...")
        header_corners = self.csv_header.get('corner_positions', {})
        corner_positions = {k: v for k, v in header_corners.items() if k in ('C5', 'C6', 'C7', 'C8')} or None
        
        diag_base = str(self.out_dir / "kabsch_oc_diag")
        self.kabsch = KabschPreprocessor(
            contact_accel_threshold=self.cfg.contact_threshold
        ).fit(
            self.csv_df, self.time_arr,
            corner_positions=corner_positions,
            corner_labels=['C5', 'C6', 'C7', 'C8'],
            diag_save_paths=[diag_base]
        )

        # 마스터 노드(#900000~) + RBE3 + SPCD 인가
        for idx, (cname, corner_nids) in enumerate(self.bot_groups):
            pts = np.array([self.node_db[nid] for nid in corner_nids])
            center = np.mean(pts, axis=0)
            mnid = 900000 + idx
            self.model.add_node(mnid, center[0], center[1], center[2])
            self.model.add_rbe3(mnid, mnid, corner_nids, dofs=(0, 1, 2))
            
            d = self.kabsch.deformation.get(cname)
            if d is None:
                continue
            for dof_idx, dof in enumerate([0, 1, 2]):
                self.load_groups.append(InterpLoadGroup(
                    node_ids=[mnid], dof=dof,
                    time_arr=self.time_arr, val_arr=d[:, dof_idx],
                    load_type="SPCD"
                ))
                
        # 관성 하중
        if self.cfg.add_inertia:
            print("\n [4-1] 관성 하중(F = -m*a) 생성 및 인가...")
            a_world = self.kabsch.accel_arr.mean(axis=1)
            a_body = np.einsum('tji,tj->ti', self.kabsch.R_arr, a_world)
            a_body_z = a_body[:, 2]
            
            self.solver = WHTDynamicSolver(self.model)
            jm, s_nids, n2i = self.solver._build_jaxsso_model()
            m_diag = self.solver._assemble_lumped_mass(jm, jm.ndof, s_nids, n2i)
            
            # 섀시 영역 노드(nid < 200,000) 및 글라스 노드(200,000 <= nid < 300,000) 총 질량 합산
            chassis_mass_lumped = sum(
                m_diag[n2i[nid] * 6 + 2]
                for nid in self.model.nodes
                if nid < 200000 and n2i.get(nid) is not None
                and m_diag[n2i[nid] * 6 + 2] > 0
            )
            glass_mass_lumped = sum(
                m_diag[n2i[nid] * 6 + 2]
                for nid in self.model.nodes
                if 200000 <= nid < 300000 and n2i.get(nid) is not None
                and m_diag[n2i[nid] * 6 + 2] > 0
            )
            total_mass = chassis_mass_lumped + glass_mass_lumped
            print(f"    - 섀시 질량 (lumped): {chassis_mass_lumped*1e3:.3f} kg (목표: {self.cfg.chassis_mass:.1f} kg)")
            print(f"    - 글라스 질량 (lumped): {glass_mass_lumped*1e3:.3f} kg")
            print(f"    - 어셈블리 총 질량 (lumped): {total_mass*1e3:.3f} kg")
            
            # Von Karman 척도 및 FEM 척도 계산 (섀시 단독 기준 역산)
            _, _, w_NL, vk_info = compute_vk_inertia_scale(
                E=MAT_CHASSIS['E'], nu=MAT_CHASSIS['nu'], t=MAT_CHASSIS['t'],
                plate_a=WIDTH, plate_b=LENGTH,
                total_mass=total_mass, a_body_z=a_body_z
            )
            corner_nids = [nid for _, nids in self.bot_groups for nid in nids]
            fem_scale, _ = compute_inertia_scale_via_fem(
                model=self.model,
                corner_nids=corner_nids,
                n2i=n2i, m_diag=m_diag,
                a_body_z=a_body_z,
                w_NL_target=w_NL
            )
            a_body_z_scaled = a_body_z * fem_scale
            
            n_inertia = 0
            for nid in self.model.nodes:
                if nid >= 900000:
                    continue
                idx = n2i.get(nid)
                if idx is None:
                    continue
                node_mass = m_diag[idx * 6 + 2]
                if node_mass > 0:
                    self.load_groups.append(InterpLoadGroup(
                        [nid], 2, self.time_arr, -node_mass * a_body_z_scaled, load_type="FORCE"
                    ))
                    n_inertia += 1
            print(f"    - 관성 하중 인가 노드 수: {n_inertia}개")

    def _run_modal_analysis(self) -> None:
        """고유치 해석을 돌려서 어셈블리 구조의 주요 모드를 파악합니다."""
        print("\n [5] 고유치 해석 수행 (NF)...")
        from wht_solver.wht_solver import WHTSolver
        modal_solver = WHTSolver(self.model)
        
        # 1차~6차는 강체 모드이므로 10개 추출 시 탄성 모드 확인 가능
        res = modal_solver.solve_modal(num_modes=10, exclude_rigid_body=True)
        print("    - 추출된 고유 진동수 (탄성 모드):")
        for i, f in enumerate(res.frequencies):
            print(f"      모드 {i+1:2d}: {f:6.2f} Hz")

    def _run_dynamic(self) -> None:
        """모달 중첩법 또는 직접적분법 과도 동역학 해석을 수행합니다."""
        dt_val = self.cfg.dt if self.cfg.dt else 0.005
        t_total = float(self.time_arr[-1])
        
        if self.cfg.n_modes > 0:
            print(f"\n [6] 과도 동해석 (모달 중첩법, modes={self.cfg.n_modes}, dt={dt_val:.1e}s)...")
            self.solver = WHTDynamicSolver(self.model)
            self.dyn = self.solver.solve_modal_dynamic(
                load_groups=self.load_groups,
                dt=dt_val,
                T=t_total,
                n_modes=self.cfg.n_modes,
                n_save=100,
                damping=DampingSpec(mode="zeta", zeta=0.02)
            )
        else:
            print(f"\n [6] 과도 동해석 (직접 Newmark-beta 적분, dt={dt_val:.1e}s)...")
            self.solver = WHTDynamicSolver(self.model)
            self.dyn = self.solver.solve_direct_dynamic(
                load_groups=self.load_groups,
                dt=dt_val,
                T=t_total,
                n_save=100,
                damping=DampingSpec(mode="zeta", zeta=0.02)
            )
        print(f"    - 동역학 거동 해석 완료: {self.dyn.summary()}")

    def _recover_stress(self) -> None:
        """쉘 요소들에 대해 동역학 이력의 응력을 복원합니다."""
        print("\n [7] 응력·변형률 필드 복원 (Stress Recovery)...")
        self.solver.recover_stress_history(self.dyn)

    def _export_results(self) -> None:
        """ParaView HDF 포맷으로 결과를 출력합니다."""
        print("\n [8] ParaView HDF로 파일 내보내기...")
        meta = WHTMetadata(
            solver_name="WHTDynamicSolverWithOC", solver_version="1.0.0",
            analysis_type="transient", coordinate_system="cartesian",
            unit_length="mm", unit_force="N"
        )
        self.wht_data = self.dyn.to_wht_result_data(meta, self.model)
        
        paraview_dir = self.out_dir / "paraview"
        paraview_dir.mkdir(parents=True, exist_ok=True)
        self.hdf_path = str(paraview_dir / "chassis_oc_dynamic.hdf")
        VTKHDFExporter().export(self.wht_data, self.hdf_path)
        print(f"    - ParaView HDF 파일 저장: {self.hdf_path}")

    def _visualize(self) -> None:
        """결과 3D 인터랙티브 뷰어 렌더링."""
        if self.cfg.no_viz:
            return
        print("\n [9] WHTVisualizer 3D 뷰어 구동 (인터랙티브 분석)...")
        viz = WHTVisualizer(title="Chassis + Open-Cell Glass Dynamic Assembly Test")
        viz.show_result(self.wht_data, group_name="TrayWithGlass", kabsch=self.kabsch)
        
        # 기본 설정: XY 투영 및 Isometric 뷰 설정
        viz.plotter.view_isometric()
        viz.plotter.reset_camera()
        
        if hasattr(viz.plotter, 'app'):
            viz.plotter.app.exec_()

# ─────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chassis + Open-Cell Glass Transient Validation Example")
    
    parser.add_argument("--pos-data", type=str, default="wht_topo/structural_dynamics_c235.csv",
                        help="실측 코너 위치 CSV 경로")
    parser.add_argument("--dt", type=float, default=0.005,
                        help="과도 응답 해석 시간 간격 (s)")
    parser.add_argument("--chassis-mass", type=float, default=20.0,
                        help="스틸 섀시 목표 질량 kg (기본: 20.0)")
    parser.add_argument("--glass-t", type=float, default=1.0,
                        help="글라스 패널 두께 mm (기본: 1.0)")
    parser.add_argument("--glass-E", type=float, default=40000.0,
                        help="글라스 패널 탄성 계수 MPa (기본: 40000.0)")
    parser.add_argument("--sealant-E", type=float, default=1.0,
                        help="소프트 결합 빔 탄성 계수 MPa (기본: 1.0)")
    parser.add_argument("--n-modes", type=int, default=20,
                        help="모달 중첩법 사용 모드 수 (0인 경우 직접 적분법 사용, 기본: 20)")
    parser.add_argument("--add-inertia", action="store_true", default=True,
                        help="분포 관성 하중 F=-ma 활성화")
    parser.add_argument("--no-inertia", action="store_false", dest="add_inertia",
                        help="분포 관성 하중 비활성화")
    parser.add_argument("--contact-threshold", type=float, default=24516.6,
                        help="Kabsch 충격 임계 가속도 mm/s^2 (기본: 24517 = ~2.5g)")
    parser.add_argument("--no-viz", action="store_true",
                        help="시각화 창 생략")
    
    args = parser.parse_args()
    
    # 윈도우 환경 스폰 에러 방지
    import multiprocessing
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    GlassAssemblyPipeline(args).run()
