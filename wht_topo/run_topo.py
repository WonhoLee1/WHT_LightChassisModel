# -*- coding: utf-8 -*-
"""
run_topo.py
===========
Standalone Topography Optimization Tool with Industrial Options.
Supports Discrete Beads, Minimum Width, and Draw Angle Control.

Also supports:
  --pos-data PATH   CSV 파일로부터 4코너 실측 위치 데이터를 읽어
                    동적 구조 응답 해석을 수행합니다.
"""

import argparse
import numpy as np
import sys
import io
import multiprocessing
from pathlib import Path

# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_topo.loads import StochasticLoadManager
from wht_topo.constraints import DynamicConstraint, StressConstraint
from wht_topo.solver import WHTopographySolver
from wht_visualizer.wht_visualizer import WHTVisualizer
from wht_converter.wht_models import WHTMetadata
from test_jaxSSO.mesh_utils import generate_shell_tray


# ─────────────────────────────────────────────────────────────────────────────
# CSV 위치 데이터 기반 동적 응답 해석
# ─────────────────────────────────────────────────────────────────────────────

def _find_corner_nodes(node_db: dict, width: float, length: float,
                       radius: float, z_min: float, z_max: float) -> list:
    """
    4 코너 기준으로 반경 내 노드 그룹을 반환합니다.

    코너 순서 (CSV 코너명 대응):
      index 0 → (+X, +Y) 코너 ↔ C5
      index 1 → (+X, -Y) 코너 ↔ C6
      index 2 → (-X, -Y) 코너 ↔ C7
      index 3 → (-X, +Y) 코너 ↔ C8

    메시의 실제 X, Y 범위 min/max를 자동 감지하여 코너 중심을 결정하므로,
    원점 중심(center 오리진) 또는 한쪽 끝(0-base 오리진) 어느 배치에도 대응합니다.

    Parameters
    ----------
    node_db : dict
        노드 ID → (x, y, z) 좌표 딕셔너리.
    width : float
        섀시 폭 (X방향, mm) — 현재 미사용, 자동 감지로 대체.
    length : float
        섀시 길이 (Y방향, mm) — 현재 미사용, 자동 감지로 대체.
    radius : float
        코너 탐색 반경 (mm).
    z_min, z_max : float
        탐색 Z 범위 (mm).

    Returns
    -------
    list of ((cx, cy), [nid, ...])
        코너 중심 좌표와 소속 노드 ID 리스트.
    """
    import numpy as np

    # z 범위 내 노드만 필터링하여 XY 범위 자동 감지
    all_xyz = np.array([v for v in node_db.values()])
    mask_z  = (all_xyz[:, 2] >= z_min) & (all_xyz[:, 2] <= z_max)
    xyz_z   = all_xyz[mask_z]

    if len(xyz_z) == 0:
        raise RuntimeError(f"z=[{z_min},{z_max}]mm 범위 내 노드가 없습니다.")

    x_min, x_max = xyz_z[:, 0].min(), xyz_z[:, 0].max()
    y_min, y_max = xyz_z[:, 1].min(), xyz_z[:, 1].max()

    # 4코너: +X+Y, +X-Y, -X-Y, -X+Y (CSV 순서 C5, C6, C7, C8에 대응)
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
            raise RuntimeError(
                f"코너 ({cx:.0f},{cy:.0f}) 반경 {radius:.0f}mm / "
                f"z=[{z_min},{z_max}]mm 내 노드 없음. "
                f"XY 범위: x=[{x_min:.0f},{x_max:.0f}], y=[{y_min:.0f},{y_max:.0f}]"
            )
        groups.append(((cx, cy), nids))
    return groups


def run_pos_dynamic(args):
    """
    CSV 위치 데이터를 읽어 4코너에 강제 변위(SPCD)를 적용한 동적 응답 해석을 수행합니다.

    CSV 포맷 (sample_pos.csv 기준):
      - 열: Frame, Time, C5_pos_X, C6_pos_X, C7_pos_X, C8_pos_X,
                         C5_pos_Y, C6_pos_Y, C7_pos_Y, C8_pos_Y,
                         C5_pos_Z, C6_pos_Z, C7_pos_Z, C8_pos_Z
      - 단위: m (내부적으로 mm 변환)
      - 코너 대응: C5=(+x,+y), C6=(+x,-y), C7=(-x,-y), C8=(-x,+y)

    강제 변위 처방 방식:
      - t0 (첫 행)의 Z값을 기준(0)으로 상대 변위 계산
      - Z방향(Tz)만 SPCD로 적용 (X, Y는 제외)
      - 각 코너의 하단 노드군 전체에 동일한 처방 변위 인가
      - 총 시간: CSV 최대 Time 값 사용
      - 저장 프레임: 100개 (시간 균등 분할)

    Parameters
    ----------
    args : argparse.Namespace
        pos_data   : CSV 파일 경로 (str)
        corner_r   : 코너 탐색 반경 mm (float, 기본 120.0)
        zeta       : 감쇠비 (float, 기본 0.02)
        dt         : 적분 시간 스텝 s (float, 기본 1e-4)
        no_viz     : 시각화 생략 여부 (bool)
    """
    import pandas as pd
    from datetime import datetime
    from wht_solver.wht_dynamic_solver import WHTDynamicSolver
    from wht_solver.wht_dynamic_common import DynamicLoadGroup, DampingSpec
    from wht_converter.wht_exporters import VTKHDFExporter

    print(f"\n{'='*65}")
    print(f"  run_topo: CSV Position Data → Dynamic Response Analysis")
    print(f"{'='*65}\n")

    # ── [1] CSV 읽기 ─────────────────────────────────────────────────────────
    csv_path = Path(args.pos_data)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path   # CWD 기준 상대 경로
    print(f" [1] CSV 로드: {csv_path}")
    df = pd.read_csv(csv_path, encoding='utf-8')

    # 시간 축 (s)
    time_arr = df['Time'].to_numpy(dtype=float)          # shape (N_frames,)
    T_total  = float(time_arr[-1])
    print(f"     프레임 수: {len(time_arr)}, 총 시간: {T_total:.4f}s")

    # 코너별 X, Y, Z 변위 (단위: m → mm 변환, t0 기준 상대 변위)
    CORNER_NAMES = ['C5', 'C6', 'C7', 'C8']
    corner_disp = {cn: {} for cn in CORNER_NAMES}   # {name: {axis: ndarray}}
    for cn in CORNER_NAMES:
        for ax in ['X', 'Y', 'Z']:
            col = f'{cn}_pos_{ax}'
            vals_mm = df[col].to_numpy(dtype=float) * 1000.0
            corner_disp[cn][ax] = vals_mm - vals_mm[0]

    # 최대 절댓값 확인 (Z축 기준)
    for cn in CORNER_NAMES:
        peak_z = np.max(np.abs(corner_disp[cn]['Z']))
        print(f"     {cn} Tz 최대 변위: {peak_z:.3f} mm")

    # ── [2] 메시 생성 (run_topo 와 동일 파라미터) ────────────────────────────
    print(f"\n [2] 메시 생성...")
    TRAY_W, TRAY_L, TRAY_H = 1800.0, 1200.0, 35.0
    CORNER_RADIUS = getattr(args, 'corner_r', 120.0)

    hook_sequence = [(0.0, 12.0), (-5.0, 0.0), (0.0, -10.0), (-10.0, 0.0)]
    node_db, elem_db = generate_shell_tray(
        width=TRAY_W, length=TRAY_L, height=TRAY_H,
        mesh_size_xy=40.0, mesh_size_z=10.0,
        draft_angle=25.0, flange_segments=hook_sequence,
        flanges=(False, True, True, True), mesh_type='quad4',
    )
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db,
                                           name="DynamicTray", is_solid=False)
    model.add_material(1, E=210000.0, nu=0.3, rho=7.85e-9)
    model.add_property(1, "PSHELL", t=1.2, mid=1)
    for eid in model.elements:
        model.elements[eid].pid = 1
    print(f"     노드={len(node_db)}, 요소={len(elem_db)}")

    # ── [3] 코너 노드 탐색 ───────────────────────────────────────────────────
    print(f"\n [3] 하단 코너 노드 탐색 (반경={CORNER_RADIUS:.0f}mm)...")
    bot_groups = _find_corner_nodes(
        node_db, TRAY_W, TRAY_L, CORNER_RADIUS, z_min=0.0, z_max=2.0
    )
    for i, ((cx, cy), bg) in enumerate(bot_groups):
        print(f"     코너 {CORNER_NAMES[i]} ({cx:+.0f},{cy:+.0f}): {len(bg)}개 노드")

    # ── [4] SPCD 하중 그룹 구성 (시계열 보간) ────────────────────────────────
    print(f"\n [4] SPCD 하중 그룹 구성 (시계열 보간 방식)...")

    # DynamicLoadGroup 의 time_func='table' 을 지원하지 않으므로,
    # 각 코너에 대해 numpy 보간 함수를 감싸는 커스텀 방식 사용.
    # → 매 스텝마다 evaluate() 호출 시 보간값을 반환하도록
    #   time_func='half_sine' 대신 사전 보간 객체를 활용합니다.

    class _InterpLoadGroup:
        """
        시계열 데이터를 선형 보간하여 임의 시각 t에서 SPCD 변위를 반환합니다.

        Parameters
        ----------
        node_ids : list of int
            처방 대상 노드 ID 리스트.
        dof : int
            자유도 (2 = Tz).
        time_arr : np.ndarray
            시간 배열 (s).
        disp_arr : np.ndarray
            변위 배열 (mm).
        """
        def __init__(self, node_ids, dof, time_arr, disp_arr):
            self.node_ids  = node_ids
            self.dof       = dof
            self.load_type = "SPCD"
            self._t  = time_arr
            self._u  = disp_arr

        def evaluate(self, t: float) -> float:
            """시각 t에서 선형 보간된 변위값 반환 (mm)."""
            return float(np.interp(t, self._t, self._u))

        def u_value(self, t: float) -> float:
            return self.evaluate(t)

        def ud_value(self, t: float) -> float:
            """속도: 중앙 차분으로 근사."""
            eps = 1e-6
            return (self.evaluate(t + eps) - self.evaluate(t - eps)) / (2 * eps)

        def udd_value(self, t: float) -> float:
            """가속도: 중앙 차분으로 근사."""
            eps = 1e-6
            return (self.evaluate(t + eps) - 2 * self.evaluate(t) +
                    self.evaluate(t - eps)) / (eps ** 2)

    load_groups = []

    for i, (cn, (_, corner_nids)) in enumerate(zip(CORNER_NAMES, bot_groups)):
        # X, Y, Z (dof 0, 1, 2) 모두에 대해 SPCD 적용
        for dof_idx, ax in enumerate(['X', 'Y', 'Z']):
            lg = _InterpLoadGroup(
                node_ids = corner_nids,
                dof      = dof_idx,           # 0=Tx, 1=Ty, 2=Tz
                time_arr = time_arr,
                disp_arr = corner_disp[cn][ax],
            )
            load_groups.append(lg)

    # ── [5] 동해석 실행 ───────────────────────────────────────────────────────
    DT     = getattr(args, 'dt', 1e-4)         # 적분 스텝 (s)
    N_SAVE = 100                                # 저장 프레임 수 (고정)
    ZETA   = getattr(args, 'zeta', 0.02)

    print(f"\n [5] Direct Newmark-β 동해석")
    print(f"     dt={DT:.1e}s, T_total={T_total:.4f}s, n_save={N_SAVE}")

    solver  = WHTDynamicSolver(model)
    damping = DampingSpec(mode="zeta", zeta=ZETA)

    dyn = solver.solve_direct_dynamic(
        load_groups = load_groups,
        dt          = DT,
        T           = T_total,
        damping     = damping,
        n_save      = N_SAVE,
    )
    print(f"\n     {dyn.summary()}")

    # ── [6] 응력/변형률 복원 ─────────────────────────────────────────────────
    print(f"\n [6] 응력/변형률 복원...")
    solver.recover_stress_history(dyn)

    # ── [7] 결과 변환 및 저장 ────────────────────────────────────────────────
    meta = WHTMetadata(
        solver_name       = "WHTDynamicSolver",
        solver_version    = "0.1.0",
        analysis_type     = "transient",
        coordinate_system = "cartesian",
        unit_length       = "mm",
        unit_force        = "N",
    )
    wht_data = dyn.to_wht_result_data(meta, model)

    stamp   = datetime.now().strftime("D%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent.parent / "results" / stamp
    pv_dir  = out_dir / "paraview"
    pv_dir.mkdir(parents=True, exist_ok=True)

    hdf_path = str(pv_dir / "pos_dynamic_result.hdf")
    VTKHDFExporter().export(wht_data, hdf_path)
    print(f"\n [7] ParaView HDF 저장: {hdf_path}")

    # ── [8] 시각화 ────────────────────────────────────────────────────────────
    if not args.no_viz:
        print(" [8] WHTVisualizer 실행...")
        viz = WHTVisualizer(title="Pos-Data Dynamic Response")
        viz.show_result(wht_data, group_name="DynamicTray")
        viz.plotter.view_isometric()
        viz.plotter.reset_camera()
        if hasattr(viz.plotter, 'app'):
            viz.plotter.app.exec_()


def run_industrial_topo(args):
    print(f"\n" + "="*80)
    print(f" [wht_topo] Industrial Topography Optimization Pipeline")
    print("="*80)


    # 1. 메시 생성
    hook_sequence = [(0.0, 12.0), (-5.0, 0.0), (0.0, -10.0), (-10.0, 0.0)]
    node_db, elem_db = generate_shell_tray(
        width=1800.0, length=1200.0, height=35.0,
        mesh_size_xy=40.0, mesh_size_z=10.0,
        draft_angle=25.0, flange_segments=hook_sequence,
        flanges=(False, True, True, True), mesh_type='quad4'
    )
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db)
    
    # 2. 하중 관리자 생성
    load_manager = StochasticLoadManager(model)
    # [설계 원칙] 하중 케이스별 개별 BC는 solver 내부에서 자동으로 설정됩니다.
    # (Bending=플랜지 전체 고정, Twisting=대각 코너 고정, Lifting=3코너 고정)
    # 모델 수준의 전역 SPC는 설정하지 않습니다.

    # 제약 조건 리스트 구성 (향후 확장용)
    constraints = []
    if args.freq_min > 0.1 or args.freq_max > 0.1:
        print(f" -> [제약] 목표 1차 진동수 범위 설정: {args.freq_min}Hz ~ {args.freq_max}Hz")
        constraints.append(DynamicConstraint(min_freq=args.freq_min, max_freq=args.freq_max))

    # 3. 최적화 실행 — WHTopographySolver (설계 변수: 노드별 비드 높이)
    print(f" -> [최적화] 최소 비드 폭: {args.min_width}mm | 최대 비드 높이: {args.bead_height}mm")
    print(f" -> [제약] 비드 생성 영역 제한 (Bead Area Constraint): {args.bead_area * 100:.1f}%")
    weights = {
        "bending":       args.w_bending,
        "bending_xspan": args.w_bending_xspan,
        "bending_yspan": args.w_bending_yspan,
        "twisting":      args.w_twisting,
        "twisting_alt":  args.w_twisting_alt,
        "lifting":       args.w_lifting,
    }

    solver = WHTopographySolver(
        model, load_manager, constraints,
        bead_height_max=args.bead_height,
        bead_height_ratio=args.bead_area,    # 비드 면적 비율 (0~1)
        min_width=args.min_width,
        draw_dir=args.draw_dir,
        weights=weights,
        mesh_size_z=args.mesh_size,
        sym_x=args.sym_x,
        bead_connect=args.bead_connect,
        connect_gap=args.connect_gap,
        bead_steps=args.height_steps,
    )
    
    # 모니터링 GUI 실행
    ui_process = None
    callback = None
    stop_event = None
    if not args.no_gui:
        from wht_topo.monitor_ui import start_monitor_ui
        queue = multiprocessing.Queue()
        stop_event = multiprocessing.Event()
        ui_process = multiprocessing.Process(target=start_monitor_ui, args=(queue, stop_event))
        ui_process.daemon = False  # 사용자가 직접 닫을 때까지 유지
        ui_process.start()
        callback = queue.put

    final_heights = solver.solve(max_iter=args.iters, callback=callback, stop_event=stop_event)

    if ui_process and ui_process.is_alive():
        print(" -> [GUI] 최적화 완료. 모니터 창은 사용자가 직접 닫을 때까지 유지됩니다.")
        queue.put("STOP")  # UI에 완료 상태 표시 (창은 닫히지 않음)

    # 4. 비드 높이 이산화 (--height-steps N 지정 시)
    discrete_height = args.height_steps >= 2
    if discrete_height:
        n = args.height_steps
        # N단계 이산 레벨 (0, ..., h_max)
        levels = np.linspace(0.0, solver.h_max, n)
        # 각 높이값을 가장 가까운 레벨로 매핑
        indices = np.abs(solver.heights[:, None] - levels).argmin(axis=1)
        solver.heights = levels[indices]
        
        final_levels = np.unique(np.round(solver.heights, 4))
        print(f" -> [이산화] {n}개 레벨 양자화 완료: {final_levels} mm")

    # 5. 최종 형상을 모델 노드 좌표에 영구 적용
    # 이산화 후에는 필터 재적용 시 양자화값이 블러링되므로 skip_filter=True
    solver.apply_final_shape(skip_filter=discrete_height)

    # 6. 익스포트
    if args.export:
        model.export_to_solver('lsdyna', args.export, reorder=True)
        print(f" -> [성공] 결과 저장 완료: {args.export}")

    # 7. 시각화
    if not args.no_viz:
        print(" -> [시각화] 결과 렌더링 중...")
        # WHTMeshModel을 WHTResultData로 변환 (시각화 모듈 호환성)
        result_data = model.to_wht_result_data()
        
        # 비드 높이 데이터 추가 (0~1 비율로 정규화하여 레전드와 일치시킴)
        heights_full = solver.get_full_heights(skip_filter=discrete_height)
        # heights_full은 0~h_max 범위이므로 h_max로 나누어 0~1 범위를 만듦
        bead_ratio = (heights_full / (solver.h_max + 1e-12)).reshape(1, -1, 1)
        result_data.point_data["Bead_Height"] = bead_ratio
        
        # 메타데이터 업데이트 (필수 필드 포함)
        result_data.metadata = WHTMetadata(
            solver_name="WHTopographySolver",
            solver_version="2.0.0",
            analysis_type="static",
            coordinate_system="cartesian",
            unit_length="mm",
            unit_force="N",
            unit_mass="tonne"
        )
        
        viz = WHTVisualizer()
        viz.load_results(result_data)
        viz.show()

def main():
    """
    [WHT] Industrial Topography Optimization CLI
    ============================================================

    ■ 하중 케이스 구성 (9개, Weighted Sum Method)
    ------------------------------------------------------------------------------------------------
    Case 1  bending       : 플랜지 전체 고정 + 중앙 하향 하중 (접시 눌림)        w=1.0
    Case 2  twisting      : corner0+corner3 고정, corner1↑/corner2↓ (대각 비틀림) w=1.5
    Case 3  bending_xspan : Y-min/Y-max 엣지만 고정 → X방향 1800mm 스팬 굽힘    w=0.8
    Case 4  bending_yspan : X-min/X-max 엣지만 고정 → Y방향 1200mm 스팬 굽힘    w=0.8
    Case 5  twisting_alt  : corner1+corner2 고정, corner0↑/corner3↓ (반전 비틀림) w=1.5
    Case 6~9 lifting_c0~3 : 각 코너 리프팅 (3코너 고정 + 1코너 상향, 가중치 F/4)  w=0.3×4
    ------------------------------------------------------------------------------------------------

    ■ 실행 시나리오 예시 (Usage Examples)
    ------------------------------------------------------------------------------------------------
    1. 기본 실행 (GUI 모니터링 + 결과 저장):
       python wht_topo/run_topo.py --iters 30 --export chassis_result.k

    2. 강건 설계 (좌우 대칭 + 리프팅 가중치 강화):
       python wht_topo/run_topo.py --iters 40 --sym-x --w-lifting 2.0

    3. 제조 제약 강화 (비드 영역 25% 제한 + 최소 폭 100mm):
       python wht_topo/run_topo.py --bead-area 0.25 --min-width 100.0

    4. 비드 연결 활성화 (단절된 비드 자동 연결, 기본 갭 80mm):
       python wht_topo/run_topo.py --iters 30 --bead-connect

    5. 비드 연결 + 갭 크기 조정 (120mm 이하 갭까지 채움):
       python wht_topo/run_topo.py --iters 30 --bead-connect --connect-gap 120.0

    6. 대칭 + 연결 조합 (최고 품질 설계):
       python wht_topo/run_topo.py --iters 40 --sym-x --bead-connect --connect-gap 100.0
       python wht_topo/run_topo.py --iters 40 --sym-x --bead-connect --connect-gap 100.0 --bead-area 0.25
       python wht_topo/run_topo.py --iters 15 --sym-x --bead-connect --connect-gap 120.0 --min-width 30.0 --bead-area 0.35 --height-steps 2

    7. 스팬 굽힘 강조 (X/Y 스팬 비드 강제):
       python wht_topo/run_topo.py --iters 40 --sym-x --w-bending-xspan 1.5 --w-bending-yspan 1.5

    8. 배치 프로세스 (GUI 없이 최적화만 수행 후 종료):
       python wht_topo/run_topo.py --iters 50 --no-gui --no-viz --export final_bead_pattern.k

    9. 실측 위치 데이터(CSV) 기반 동적 응답 해석 (최적화 생략):
       python wht_topo/run_topo.py --pos-data wht_topo/sample_pos.csv --dt 2e-4
    ------------------------------------------------------------------------------------------------

    ■ 전체 옵션 레퍼런스
    ─────────────────────────────────────────────────────────────────────────────
    [기본 최적화]
      --iters N           최대 반복 횟수 (기본: 15)
      --bead-height F     최대 비드 높이 mm (기본: 10.0)
      --min-width F       최소 비드 폭 / 공간 필터 반경 mm (기본: 30.0)
      --bead-area F       비드 점유 면적 비율 0.0~1.0 (기본: 0.35 = 35%)

    [비드 형상 제어]
      --no-sym-x          좌우 대칭 제약 비활성화 (기본: 활성화)
      --no-bead-connect   비드 자동 연결 비활성화 (기본: 활성화)
      --connect-gap F     비드 연결 시 채울 최대 갭 크기 mm (기본: 120.0)
      --draw-dir X Y Z    비드 돌출 방향 벡터 (기본: 0 0 1 = +Z 방향)
      --height-steps N    비드 높이 이산화 단계 수 (기본: 2 → {0, h_max})
                          → 3: {0, h_max/2, h_max}

    [하중 케이스 가중치]
      --w-bending F       플랜지 전체 고정 굽힘 가중치 (기본: 1.0)
      --w-bending-xspan F X방향 스팬 굽힘 가중치 — Y엣지 2개 고정 (기본: 0.8)
      --w-bending-yspan F Y방향 스팬 굽힘 가중치 — X엣지 2개 고정 (기본: 0.8)
      --w-twisting F      대각 비틀림 가중치 — corner0+corner3 고정 (기본: 1.5)
      --w-twisting-alt F  반전 대각 비틀림 가중치 — corner1+corner2 고정 (기본: 1.5)
      --w-lifting F       4코너 리프팅 전체 가중치 / 4개 케이스 분할 (기본: 1.2)
                          → 케이스당 실효 가중치 = F/4
      --freq-min F        목표 1차 고유 진동수 하한 Hz (기본: 0.0 = 미사용)
      --freq-max F        목표 1차 고유 진동수 상한 Hz (기본: 0.0 = 미사용)

    [시각화 및 출력]
      --no-gui            실시간 모니터링 GUI 비활성화 (기본: GUI 실행)
      --no-viz            최적화 완료 후 최종 3D 시각화 생략
      --export PATH       최종 결과 파일 저장 경로 LS-DYNA .k (기본: industrial_bead.k)
      --mesh-size F       BC 탐색 기준 메시 크기 mm (기본: 10.0)
    
    [동적 응답 해석 (CSV 기반)]
      --pos-data PATH     CSV 위치 데이터 파일 경로 (지정 시 최적화 생략)
      --dt F              적분 시간 스텝 (s, 기본: 1e-4)
      --zeta F            감쇠비 (기본: 0.02)
      --corner-r F        코너 노드 탐색 반경 (mm, 기본: 150.0)
    ─────────────────────────────────────────────────────────────────────────────
    """
    parser = argparse.ArgumentParser(
        description="WHT 산업용 섀시 비드 최적화(Topography) 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 기본 최적화 설정 ─────────────────────────────────────────────────────
    parser.add_argument("--iters",       type=int,   default=15,   help="최대 반복 횟수 (기본: 15)")
    parser.add_argument("--bead-height", type=float, default=10.0, help="최대 비드 높이 (mm, 기본: 10.0)")
    parser.add_argument("--min-width",   type=float, default=30.0, help="최소 비드 폭 / 필터 반경 (mm, 기본: 30.0)")
    parser.add_argument("--bead-area",   type=float, default=0.35, help="비드 점유 면적 비율 (0.0~1.0, 기본: 0.35)")

    # ── 비드 형상 제어 ───────────────────────────────────────────────────────
    parser.add_argument("--sym-x",        action="store_true", default=True,
                        help="Y-Z 평면 기준 좌우 대칭 제약 조건 활성화 (기본: True)")
    parser.add_argument("--no-sym-x",     action="store_true",
                        help="좌우 대칭 제약 비활성화")
    parser.add_argument("--bead-connect", action="store_true", default=True,
                        help="Morphological Closing으로 단절된 비드 자동 연결 (기본: True)")
    parser.add_argument("--no-bead-connect", action="store_true",
                        help="비드 연결 비활성화")
    parser.add_argument("--connect-gap",  type=float, default=120.0,
                        help="비드 연결 시 채울 최대 간격 (mm, 기본: 120.0)")
    parser.add_argument("--draw-dir",     type=float, nargs=3, default=[0.0, 0.0, 1.0],
                        help="비드 돌출 방향 벡터 (X Y Z, 기본: 0 0 1)")
    parser.add_argument("--height-steps", type=int, default=2,
                        help="비드 높이 이산화 단계 수 (기본: 2 → {0, h_max})")

    # ── 하중 케이스 가중치 (Weighted Sum Method) ─────────────────────────────
    parser.add_argument("--w-bending",        type=float, default=1.0, help="중앙 굽힘 하중 가중치 (기본: 1.0)")
    parser.add_argument("--w-bending-xspan",  type=float, default=0.8, help="X방향 스팬 굽힘 하중 가중치 (기본: 0.8)")
    parser.add_argument("--w-bending-yspan",  type=float, default=0.8, help="Y방향 스팬 굽힘 하중 가중치 (기본: 0.8)")
    parser.add_argument("--w-twisting",       type=float, default=1.5, help="대각 비틀림 하중 가중치 (기본: 1.5)")
    parser.add_argument("--w-twisting-alt",   type=float, default=1.5, help="반전 대각 비틀림 하중 가중치 (기본: 1.5)")
    parser.add_argument("--w-lifting",        type=float, default=1.2, help="4코너 리프팅 하중 전체 가중치 / 4개 케이스 분할 (기본: 1.2)")
    parser.add_argument("--freq-min",    type=float, default=0.01, help="목표 1차 고유 진동수 하한 (Hz, 기본: 0.0 = 미사용)")
    parser.add_argument("--freq-max",    type=float, default=0.0, help="목표 1차 고유 진동수 상한 (Hz, 기본: 0.0 = 미사용)")

    # ── 시각화 및 출력 ───────────────────────────────────────────────────────
    parser.add_argument("--no-gui",   action="store_true",         help="실시간 모니터링 GUI 비활성화 (기본: GUI 실행)")
    parser.add_argument("--no-viz",   action="store_true",         help="최적화 완료 후 최종 3D 시각화 생략")
    parser.add_argument("--export",   type=str, default="industrial_bead.k",
                        help="최종 결과 파일 저장 경로 (LS-DYNA .k, 기본: industrial_bead.k)")
    parser.add_argument("--mesh-size", type=float, default=10.0,  help="BC 탐색을 위한 기준 메시 크기 (mm)")

    # ── CSV 위치 데이터 기반 동적 응답 해석 ─────────────────────────────────
    parser.add_argument(
        "--pos-data", type=str, default=None,
        help=(
            "CSV 파일 경로: 4코너(C5~C8) 실측 위치 데이터 기반 동적 해석 수행. "
            "이 인자가 지정되면 최적화 파이프라인을 건너뛰고 동적 응답 해석만 실행합니다. "
            "예: --pos-data wht_topo/sample_pos.csv"
        ),
    )
    parser.add_argument(
        "--dt", type=float, default=1e-4,
        help="동적 해석 적분 시간 스텝 (s, 기본: 1e-4)",
    )
    parser.add_argument(
        "--zeta", type=float, default=0.02,
        help="Rayleigh 감쇠비 ζ (기본: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--corner-r", type=float, default=150.0,
        help="코너 노드 탐색 반경 (mm, 기본: 150.0)",
    )

    args = parser.parse_args()

    # --no-* 플래그로 기본값 오버라이드
    if args.no_sym_x:
        args.sym_x = False
    if args.no_bead_connect:
        args.bead_connect = False

    # 실행 분기: --pos-data 지정 시 동적 응답 해석, 아닐 시 최적화 파이프라인
    if args.pos_data:
        run_pos_dynamic(args)
    else:
        run_industrial_topo(args)

if __name__ == "__main__":
    try:
        multiprocessing.freeze_support()
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
