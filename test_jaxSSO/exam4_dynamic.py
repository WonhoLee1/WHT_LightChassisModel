# -*- coding: utf-8 -*-
"""
[WHT_LightChassisModel] Exam 4: Implicit Dynamic Analysis (Modal Superposition)
================================================================================
4-코너 staggered half-sine 하중 → 모달 동해석 → WHTVisualizer + ParaView 저장.

하중 설정:
  - 샤시 4 코너 주변 노드에 Fz = 5000 N half-sine (t_pulse=0.01 s)
  - 코너별 0.005 s 간격 시차 적용 (staggered)
  - 감쇠: Rayleigh zeta=0.02
  - 총 해석 시간: 0.05 s, dt=1e-4 s, 저장 200 스텝

단위계: MPa, ton, mm  →  힘 N, 시간 s
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_dynamic_solver import WHTDynamicSolver
from wht_solver.wht_dynamic_common import DynamicLoadGroup, DampingSpec
from wht_converter.wht_models import WHTMetadata
from wht_converter.wht_exporters import VTKHDFExporter
from wht_visualizer.wht_visualizer import WHTVisualizer
from test_jaxSSO.mesh_utils import generate_shell_tray

# ─── 재료 (Steel, MPa/ton/mm) ───────────────────────────────────────────────
MAT = dict(E=210000.0, nu=0.3, rho=7.85e-9, t=1.2)

# ─── 해석 파라미터 ───────────────────────────────────────────────────────────
WIDTH, LENGTH, HEIGHT = 1800.0, 1200.0, 30.0
MESH_XY, MESH_Z      = 60.0, 10.0
DRAFT, FLANGE        = 10.0, 15.0

N_MODES   = 30
DT        = 1e-4    # s
T_TOTAL   = 0.05    # s
N_SAVE    = 200

F_AMP     = 5000.0  # N per corner
T_PULSE   = 0.010   # s  half-sine duration
T_OFFSET  = 0.005   # s  stagger per corner

CORNER_RADIUS = 120.0  # 코너 탐색 반경 [mm]


# ─────────────────────────────────────────────────────────────────────────────

def find_corner_nodes(node_db: dict, width: float, length: float,
                      radius: float) -> list[list[int]]:
    """
    4 코너 (±W/2, ±L/2) 주변 노드를 수평 거리 radius 내에서 탐색.

    Returns
    -------
    [corner0_nids, corner1_nids, corner2_nids, corner3_nids]
    순서: (-X,-Y), (+X,-Y), (+X,+Y), (-X,+Y)
    """
    hw, hl = width / 2.0, length / 2.0
    targets = [
        (-hw, -hl),
        ( hw, -hl),
        ( hw,  hl),
        (-hw,  hl),
    ]
    groups = []
    for cx, cy in targets:
        nids = [
            nid for nid, xyz in node_db.items()
            if (xyz[0] - cx) ** 2 + (xyz[1] - cy) ** 2 < radius ** 2
        ]
        if not nids:
            raise RuntimeError(
                f"코너 ({cx:.0f}, {cy:.0f}) 주변 {radius:.0f}mm 내 노드 없음."
            )
        groups.append(nids)
    return groups


def build_staggered_loads(corner_groups: list[list[int]]) -> list[DynamicLoadGroup]:
    """각 코너에 0.005 s 간격 staggered half-sine Fz 하중 생성."""
    loads = []
    for i, nids in enumerate(corner_groups):
        t_start = i * T_OFFSET

        def _fn(t, ts=t_start):
            t_rel = t - ts
            if T_PULSE > 0 and 0.0 <= t_rel <= T_PULSE:
                return np.sin(np.pi * t_rel / T_PULSE)
            return 0.0

        loads.append(DynamicLoadGroup(
            node_ids        = nids,
            dof             = 2,           # Fz
            force_magnitude = F_AMP,
            time_func       = _fn,
            distribute      = True,        # 코너 노드 수로 균등 분배
        ))
    return loads


def run():
    print("\n" + "=" * 65)
    print("  Exam 4: Implicit Dynamic Analysis — 4-Corner Staggered Loads")
    print("=" * 65 + "\n")

    # 1. 메시 생성 ─────────────────────────────────────────────────────────
    print(" [1] 메시 생성 중...")
    node_db, elem_db = generate_shell_tray(
        width=WIDTH, length=LENGTH, height=HEIGHT,
        mesh_size_xy=MESH_XY, mesh_size_z=MESH_Z,
        draft_angle=DRAFT, flange_width=FLANGE,
        origin='center',
    )
    print(f"     nodes={len(node_db)}, elements={len(elem_db)}")

    # 2. WHTMeshModel 구성 ─────────────────────────────────────────────────
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db,
                                           name="DynamicTray", is_solid=False)
    model.add_material(1, E=MAT['E'], nu=MAT['nu'], rho=MAT['rho'])
    model.add_property(1, "PSHELL", t=MAT['t'], mid=1)
    for eid in model.elements:
        model.elements[eid].pid = 1

    # 3. 경계조건: 상단 플랜지 고정 ──────────────────────────────────────
    fixed = [nid for nid, xyz in node_db.items() if abs(xyz[2] - HEIGHT) < 0.5]
    for nid in fixed:
        model.apply_spc(nid, dofs=(0, 1, 2, 3, 4, 5))
    print(f" [2] 고정 노드: {len(fixed)}개 (Z={HEIGHT:.0f}mm 플랜지)")

    # 4. 코너 노드 탐색 ────────────────────────────────────────────────────
    print(f" [3] 코너 노드 탐색 (반경={CORNER_RADIUS:.0f}mm)...")
    corner_groups = find_corner_nodes(node_db, WIDTH, LENGTH, CORNER_RADIUS)
    for i, g in enumerate(corner_groups):
        print(f"     코너 {i}: {len(g)}개 노드")

    # 5. 하중 생성 ─────────────────────────────────────────────────────────
    load_groups = build_staggered_loads(corner_groups)
    print(f" [4] DynamicLoadGroup {len(load_groups)}개 생성 완료")
    print(f"     하중: Fz={F_AMP:.0f} N, half-sine {T_PULSE*1000:.0f}ms, "
          f"stagger {T_OFFSET*1000:.0f}ms")

    # 6. 동해석 ─────────────────────────────────────────────────────────────
    print(f"\n [5] Modal Superposition 동해석 시작...")
    print(f"     n_modes={N_MODES}, dt={DT:.1e}s, T={T_TOTAL:.3f}s, n_save={N_SAVE}")

    solver = WHTDynamicSolver(model)
    damping = DampingSpec(mode="zeta", zeta=0.02)

    dyn = solver.solve_modal_dynamic(
        load_groups = load_groups,
        dt          = DT,
        T           = T_TOTAL,
        n_modes     = N_MODES,
        damping     = damping,
        n_save      = N_SAVE,
    )
    print(f"\n {dyn.summary()}")

    # 7. 결과 변환 ──────────────────────────────────────────────────────────
    meta = WHTMetadata(
        solver_name    = "WHTDynamicSolver",
        solver_version = "0.1.0",
        analysis_type  = "dynamic",
        coordinate_system = "cartesian",
        unit_length    = "mm",
        unit_force     = "N",
    )
    wht_data = dyn.to_wht_result_data(meta, model)

    # 8. 저장 ───────────────────────────────────────────────────────────────
    stamp   = datetime.now().strftime("D%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent.parent / "results" / stamp
    pv_dir  = out_dir / "paraview"
    pv_dir.mkdir(parents=True, exist_ok=True)

    hdf_path = str(pv_dir / "dynamic_result.hdf")
    VTKHDFExporter().export(wht_data, hdf_path)
    print(f"\n [6] ParaView HDF 저장: {hdf_path}")

    # 9. WHTVisualizer 실행 ─────────────────────────────────────────────────
    print(" [7] WHTVisualizer 시작...")
    viz = WHTVisualizer(title="Exam 4: Dynamic Analysis — Staggered Corner Loads")
    viz.show_result(wht_data, group_name="DynamicTray")

    # 자동으로 Displacement 선택됨 (analysis_type != "modal" 분기)
    viz.plotter.view_isometric()
    viz.plotter.reset_camera()

    if hasattr(viz.plotter, 'app'):
        viz.plotter.app.exec_()


if __name__ == "__main__":
    run()
