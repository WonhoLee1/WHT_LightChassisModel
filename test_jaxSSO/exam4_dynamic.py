# -*- coding: utf-8 -*-
"""
[WHT_LightChassisModel] Exam 4: Implicit Dynamic Analysis (Direct Newmark-beta)
================================================================================
4-코너 staggered half-sine SPCD -> 직접 동해석 -> WHTVisualizer + ParaView.

설계:
  - 하단 4코너: 코너 중심에 RBE2 마스터 노드 생성, 주변 슬레이브 묶음
    -> 마스터 Tz에 SPCD (half-sine, 10mm)
  - 상단 플랜지 4코너: SPC 고정 (4개 클러스터만)
  - _assemble_K_scipy 수정으로 RBE2 stiff beam이 K에 포함됨

단위계: MPa, ton, mm -> N, s
"""

import sys
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_dynamic_solver import WHTDynamicSolver
from wht_solver.wht_dynamic_common import DynamicLoadGroup, DampingSpec
from wht_converter.wht_models import WHTMetadata
from wht_converter.wht_exporters import VTKHDFExporter
from wht_visualizer.wht_visualizer import WHTVisualizer
from test_jaxSSO.mesh_utils import generate_shell_tray

# --- 재료 -------------------------------------------------------------------
MAT = dict(E=210000.0, nu=0.3, rho=7.85e-9, t=1.2)

# --- 형상 -------------------------------------------------------------------
WIDTH, LENGTH, HEIGHT = 1800.0, 1200.0, 30.0
MESH_XY, MESH_Z      = 60.0, 10.0
DRAFT, FLANGE        = 10.0, 15.0

# --- 해석 기본값 -------------------------------------------------------------
DT      = 1e-4   # s
T_TOTAL = 0.06   # s
N_SAVE  = 200

# --- SPCD 기본값 (Half-sine) -------------------------------------------------
U_AMP         = 10.0    # mm  처방 변위 진폭
T_PULSE       = 0.010   # s   half-sine 지속 시간
T_OFFSET      = 0.005   # s   코너 간 시차
CORNER_RADIUS = 100.0   # mm  코너 탐색 반경


class _InterpLoadGroup:
    """시계열 데이터를 선형 보간하여 SPCD 변위를 반환하는 그룹."""
    def __init__(self, node_ids, dof, time_arr, disp_arr):
        self.node_ids  = node_ids
        self.dof       = dof
        self.load_type = "SPCD"
        self._t = time_arr
        self._u = disp_arr

    def evaluate(self, t: float) -> float:
        return float(np.interp(t, self._t, self._u))

    def u_value(self, t: float) -> float:
        return self.evaluate(t)

    def ud_value(self, t: float) -> float:
        eps = 1e-6
        return (self.evaluate(t + eps) - self.evaluate(t - eps)) / (2 * eps)

    def udd_value(self, t: float) -> float:
        eps = 1e-6
        return (self.evaluate(t + eps) - 2 * self.evaluate(t) +
                self.evaluate(t - eps)) / (eps ** 2)


def find_corner_nodes(node_db: dict, width: float, length: float,
                      radius: float, z_min: float, z_max: float) -> list:
    """
    4 코너 (+-W/2, +-L/2) 기준으로 반경 내 + z 범위 노드 그룹 반환.
    순서: (-X,-Y), (+X,-Y), (+X,+Y), (-X,+Y)
    반환: [(center_xy, [nid, ...]), ...]
    """
    hw, hl = width / 2.0, length / 2.0
    targets = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
    groups = []
    for cx, cy in targets:
        nids = [
            nid for nid, xyz in node_db.items()
            if z_min <= xyz[2] <= z_max
            and (xyz[0] - cx) ** 2 + (xyz[1] - cy) ** 2 < radius ** 2
        ]
        if not nids:
            raise RuntimeError(
                f"코너 ({cx:.0f},{cy:.0f}) 반경 {radius:.0f}mm / "
                f"z=[{z_min},{z_max}]mm 내 노드 없음."
            )
        groups.append(((cx, cy), nids))
    return groups


def run(args):
    import pandas as pd
    
    # --- 전역 상수/인자 연동 ---
    dt_val = args.dt if args.dt else DT
    t_total_val = T_TOTAL
    n_save_val = N_SAVE
    
    csv_df = None
    if args.pos_data:
        csv_path = Path(args.pos_data)
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        print(f" [0] CSV 로드: {csv_path}")
        csv_df = pd.read_csv(csv_path, encoding='utf-8')
        
        time_arr = csv_df['Time'].to_numpy(dtype=float)
        t_total_val = float(time_arr[-1])
        n_save_val = 100  # CSV 사용 시 100프레임 고정
        print(f"     프레임: {len(time_arr)}, 총 시간: {t_total_val:.4f}s")

    print("\n" + "=" * 65)
    print("  Exam 4: Direct Newmark-b - SPCD 4-Corner Staggered")
    if args.pos_data:
        print("  [Mode] CSV Position Data Analysis")
    else:
        print("  [Mode] Synthetic Half-Sine Excitation")
    print("=" * 65 + "\n")

    # 1. 메시 -----------------------------------------------------------------
    print(" [1] 메시 생성...")
    node_db, elem_db = generate_shell_tray(
        width=WIDTH, length=LENGTH, height=HEIGHT,
        mesh_size_xy=MESH_XY, mesh_size_z=MESH_Z,
        draft_angle=DRAFT, flange_width=FLANGE,
        origin='center',
    )
    print(f"     nodes={len(node_db)}, elements={len(elem_db)}")

    # 2. 모델 -----------------------------------------------------------------
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db,
                                           name="DynamicTray", is_solid=False)
    model.add_material(1, E=MAT['E'], nu=MAT['nu'], rho=MAT['rho'])
    model.add_property(1, "PSHELL", t=MAT['t'], mid=1)
    for eid in model.elements:
        model.elements[eid].pid = 1

    # 3. 코너 노드 탐색 -------------------------------------------------------
    print(f" [2] 코너 노드 탐색 (반경={CORNER_RADIUS:.0f}mm)...")

    bot_groups = find_corner_nodes(
        node_db, WIDTH, LENGTH, CORNER_RADIUS, z_min=0.0, z_max=2.0
    )
    # find_corner_nodes 순서: 0:(-X,-Y), 1:(+X,-Y), 2:(+X,+Y), 3:(-X,+Y)
    # CSV 매핑: C7, C6, C5, C8

    # 4. 하단 코너 SPCD 하중 그룹 구성 -----------------------------------------
    load_groups = []

    if csv_df is not None:
        # CSV 모드
        CORNER_MAP = {
            2: 'C5', # (+X,+Y)
            1: 'C6', # (+X,-Y)
            0: 'C7', # (-X,-Y)
            3: 'C8', # (-X,+Y)
        }
        time_arr = csv_df['Time'].to_numpy(dtype=float)
        
        for idx, (_, corner_nids) in enumerate(bot_groups):
            cn_label = CORNER_MAP[idx]
            for axis_idx, ax in enumerate(['X', 'Y', 'Z']):
                col = f'{cn_label}_pos_{ax}'
                vals_mm = csv_df[col].to_numpy(dtype=float) * 1000.0
                vals_rel = vals_mm - vals_mm[0]
                
                lg = _InterpLoadGroup(
                    node_ids = corner_nids,
                    dof      = axis_idx, # 0=Tx, 1=Ty, 2=Tz
                    time_arr = time_arr,
                    disp_arr = vals_rel
                )
                load_groups.append(lg)
        print(f" [3] CSV 기반 XYZ SPCD 하중 구성 완료 (4개 코너)")
    else:
        # 기존 Half-sine 모드
        for i, ((cx, cy), slave_nids) in enumerate(bot_groups):
            corner_nodes = slave_nids
            load_groups.append(DynamicLoadGroup(
                node_ids  = corner_nodes,
                dof       = 2,           # Tz
                magnitude = U_AMP,
                time_func = "half_sine",
                load_type = "SPCD",
                t_pulse   = T_PULSE,
                t_start   = i * T_OFFSET,
                distribute= False,
            ))
        print(f" [3] Half-Sine SPCD 하중 구성 완료 (4개 코너)")

    # 5. 동해석 ---------------------------------------------------------------
    print(f"\n [4] Direct Newmark-b 동해석 (dt={dt_val:.1e}s, T={t_total_val:.3f}s)...")
    solver  = WHTDynamicSolver(model)
    damping = DampingSpec(mode="zeta", zeta=0.02)

    dyn = solver.solve_direct_dynamic(
        load_groups = load_groups,
        dt          = dt_val,
        T           = t_total_val,
        damping     = damping,
        n_save      = n_save_val,
    )
    print(f"\n     {dyn.summary()}")

    # 6. 응력/변형률 이력 복원 -------------------------------------------------
    print(f"\n [5] 응력/변형률 복원...")
    solver.recover_stress_history(dyn)   # dyn.stress_data 채움

    # 8. 결과 변환 ------------------------------------------------------------
    meta = WHTMetadata(
        solver_name       = "WHTDynamicSolver",
        solver_version    = "0.1.0",
        analysis_type     = "transient",
        coordinate_system = "cartesian",
        unit_length       = "mm",
        unit_force        = "N",
    )
    wht_data = dyn.to_wht_result_data(meta, model)


    # 8. ParaView HDF 저장 ----------------------------------------------------
    stamp   = datetime.now().strftime("D%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent.parent / "results" / stamp
    pv_dir  = out_dir / "paraview"
    pv_dir.mkdir(parents=True, exist_ok=True)

    hdf_path = str(pv_dir / "dynamic_result.hdf")
    VTKHDFExporter().export(wht_data, hdf_path)
    print(f"\n [6] ParaView HDF 저장: {hdf_path}")

    # 8. 시각화 --------------------------------------------------------------
    if not args.no_viz:
        print(" [7] WHTVisualizer 실행...")
        title = "Exam 4: CSV Pos Dynamic" if args.pos_data else "Exam 4: Half-Sine Dynamic"
        viz = WHTVisualizer(title=title)
        viz.show_result(wht_data, group_name="DynamicTray")
        viz.plotter.view_isometric()
        viz.plotter.reset_camera()
        if hasattr(viz.plotter, 'app'):
            viz.plotter.app.exec_()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exam 4: Dynamic Analysis with SPCD")
    parser.add_argument("--pos-data", type=str, help="CSV position data file path")
    parser.add_argument("--dt", type=float, help="Time step (s)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    run(args)
