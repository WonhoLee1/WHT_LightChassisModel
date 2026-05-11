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

# --- 해석 -------------------------------------------------------------------
DT      = 1e-4   # s
T_TOTAL = 0.06   # s
N_SAVE  = 200

# --- SPCD -------------------------------------------------------------------
U_AMP         = 10.0    # mm  처방 변위 진폭
T_PULSE       = 0.010   # s   half-sine 지속 시간
T_OFFSET      = 0.005   # s   코너 간 시차
CORNER_RADIUS = 100.0   # mm  코너 탐색 반경


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


def run():
    print("\n" + "=" * 65)
    print("  Exam 4: Direct Newmark-b - SPCD 4-Corner RBE2 Staggered")
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

    for i, ((_cx, _cy), bg) in enumerate(bot_groups):
        print(f"     코너 {i}: 하단 {len(bg)}개 (RBE2 slave+SPCD)")

    # 4. 하단 코너 RBE2 + SPCD ------------------------------------------------
    # 각 코너 중심 좌표(z=0)에 마스터 노드 생성 → 슬레이브 노드들을 RBE2로 묶음
    # 마스터 노드 Tz에 SPCD 처방 변위 적용
    max_nid  = max(model.nodes.keys())
    max_rbe2 = 1

    load_groups = []
    master_nids = []

    for i, ((cx, cy), slave_nids) in enumerate(bot_groups):
        # RBE2 대신 코너 노드(slave_nids)에 직접 SPCD 및 SPC 적용
        corner_nodes = slave_nids
        
        # Rigid body motion 방지를 위한 최소 구속 (3-2-1 원리)
        if i == 0:
            # 첫 번째 코너의 첫 번째 노드: X, Y 고정
            model.apply_spc(corner_nodes[0], dofs=(0, 1))
        elif i == 1:
            # 두 번째 코너의 첫 번째 노드: Y 고정
            model.apply_spc(corner_nodes[0], dofs=(1,))

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

    print(f" [3] 하단 코너 노드 직접 처방: 4개 그룹")
    print(f"     SPCD: 마스터 Tz {U_AMP:.0f}mm half-sine "
          f"{T_PULSE*1000:.0f}ms, stagger {T_OFFSET*1000:.0f}ms")

    # 6. 동해석 ---------------------------------------------------------------
    print(f"\n [5] Direct Newmark-b 동해석 (dt={DT:.1e}s, T={T_TOTAL:.3f}s)...")
    solver  = WHTDynamicSolver(model)
    damping = DampingSpec(mode="zeta", zeta=0.02)

    dyn = solver.solve_direct_dynamic(
        load_groups = load_groups,
        dt          = DT,
        T           = T_TOTAL,
        damping     = damping,
        n_save      = N_SAVE,
    )
    print(f"\n     {dyn.summary()}")

    # 7. 응력/변형률 이력 복원 -------------------------------------------------
    print(f"\n [6] 응력/변형률 복원...")
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

    # 9. WHTVisualizer --------------------------------------------------------
    print(" [7] WHTVisualizer 실행...")
    viz = WHTVisualizer(title="Exam 4: SPCD Staggered Corner Excitation (RBE2)")
    viz.show_result(wht_data, group_name="DynamicTray")
    viz.plotter.view_isometric()
    viz.plotter.reset_camera()

    if hasattr(viz.plotter, 'app'):
        viz.plotter.app.exec_()


if __name__ == "__main__":
    run()
