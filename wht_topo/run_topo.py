# -*- coding: utf-8 -*-
"""
run_topo.py
===========
Standalone Topography Optimization Tool with Industrial Options.
Supports Discrete Beads, Minimum Width, and Draw Angle Control.
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
    if args.gui:
        from wht_topo.monitor_ui import start_monitor_ui
        queue = multiprocessing.Queue()
        stop_event = multiprocessing.Event()
        ui_process = multiprocessing.Process(target=start_monitor_ui, args=(queue, stop_event))
        ui_process.daemon = True # 메인 프로세스 종료 시 함께 종료
        ui_process.start()
        callback = queue.put

    final_heights = solver.solve(max_iter=args.iters, callback=callback, stop_event=stop_event)

    if ui_process and ui_process.is_alive():
        print(" -> [GUI] 최적화 완료. 모니터링 창을 유지한 상태로 시각화를 준비합니다.")
        # UI 종료를 기다리거나 계속 진행할 수 있음. 
        # 여기서는 비동기적으로 두기 위해 STOP 신호만 보냄
        queue.put("STOP")

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
       python wht_topo/run_topo.py --iters 30 --gui --export chassis_result.k

    2. 강건 설계 (좌우 대칭 + 리프팅 가중치 강화):
       python wht_topo/run_topo.py --iters 40 --sym-x --w-lifting 2.0 --gui

    3. 제조 제약 강화 (비드 영역 25% 제한 + 최소 폭 100mm):
       python wht_topo/run_topo.py --bead-area 0.25 --min-width 100.0 --gui

    4. 비드 연결 활성화 (단절된 비드 자동 연결, 기본 갭 80mm):
       python wht_topo/run_topo.py --iters 30 --bead-connect --gui

    5. 비드 연결 + 갭 크기 조정 (120mm 이하 갭까지 채움):
       python wht_topo/run_topo.py --iters 30 --bead-connect --connect-gap 120.0 --gui

    6. 대칭 + 연결 조합 (최고 품질 설계):
       python wht_topo/run_topo.py --iters 40 --sym-x --bead-connect --connect-gap 100.0 --gui
       python wht_topo/run_topo.py --iters 40 --sym-x --bead-connect --connect-gap 100.0 --bead-area 0.25 --gui
       python wht_topo/run_topo.py --iters 15 --sym-x --bead-connect --connect-gap 120.0 --min-width 30.0 --bead-area 0.35 --height-steps 2 --gui

    7. 스팬 굽힘 강조 (X/Y 스팬 비드 강제):
       python wht_topo/run_topo.py --iters 40 --sym-x --w-bending-xspan 1.5 --w-bending-yspan 1.5 --gui

    8. 배치 프로세스 (UI 없이 최적화만 수행 후 종료):
       python wht_topo/run_topo.py --iters 50 --no-viz --export final_bead_pattern.k
    ------------------------------------------------------------------------------------------------

    ■ 전체 옵션 레퍼런스
    ─────────────────────────────────────────────────────────────────────────────
    [기본 최적화]
      --iters N           최대 반복 횟수 (기본: 30)
      --bead-height F     최대 비드 높이 mm (기본: 10.0)
      --min-width F       최소 비드 폭 / 공간 필터 반경 mm (기본: 80.0)
      --bead-area F       비드 점유 면적 비율 0.0~1.0 (기본: 0.3 = 30%)

    [비드 형상 제어]
      --sym-x             Y-Z 평면 기준 좌우 대칭 제약 강제
                          → KDTree로 대칭 노드 쌍을 매핑, 설계변수/민감도 평균화
      --bead-connect      단절된 비드 영역을 Morphological Closing으로 자동 연결
                          → 2-phase: MMA 전 bridge 민감도 승격 + MMA 후 물리적 closing
      --connect-gap F     비드 연결 시 채울 최대 갭 크기 mm (기본: 80.0)
                          → 메시 간격(40mm)의 2~3배 권장. 너무 크면 면적 제약 느슨해짐
      --draw-dir X Y Z    비드 돌출 방향 벡터 (기본: 0 0 1 = +Z 방향)
      --height-steps N    비드 높이 이산화 단계 수 (기본: 0 = 연속)
                          → 2: {0, h_max}, 3: {0, h_max/2, h_max}

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
      --gui               실시간 모니터링 GUI (PySide6) 실행
      --no-viz            최적화 완료 후 최종 3D 시각화 생략
      --export PATH       최종 결과 파일 저장 경로 LS-DYNA .k (기본: industrial_bead.k)
      --mesh-size F       BC 탐색 기준 메시 크기 mm (기본: 10.0)
    ─────────────────────────────────────────────────────────────────────────────
    """
    parser = argparse.ArgumentParser(
        description="WHT 산업용 섀시 비드 최적화(Topography) 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 기본 최적화 설정 ─────────────────────────────────────────────────────
    parser.add_argument("--iters",       type=int,   default=30,   help="최대 반복 횟수 (기본: 30)")
    parser.add_argument("--bead-height", type=float, default=10.0, help="최대 비드 높이 (mm, 기본: 10.0)")
    parser.add_argument("--min-width",   type=float, default=80.0, help="최소 비드 폭 / 필터 반경 (mm, 기본: 80.0)")
    parser.add_argument("--bead-area",   type=float, default=0.3,  help="비드 점유 면적 비율 (0.0~1.0, 기본: 0.3)")

    # ── 비드 형상 제어 ───────────────────────────────────────────────────────
    parser.add_argument("--sym-x",        action="store_true",
                        help="Y-Z 평면 기준 좌우 대칭 제약 조건 활성화 (KDTree 대칭 노드 쌍 매핑)")
    parser.add_argument("--bead-connect", action="store_true",
                        help="Morphological Closing으로 단절된 비드 자동 연결")
    parser.add_argument("--connect-gap",  type=float, default=80.0,
                        help="비드 연결 시 채울 최대 간격 (mm, 기본: 80.0 ≈ 메시 간격×2)")
    parser.add_argument("--draw-dir",     type=float, nargs=3, default=[0.0, 0.0, 1.0],
                        help="비드 돌출 방향 벡터 (X Y Z, 기본: 0 0 1)")
    parser.add_argument("--height-steps", type=int, default=0,
                        help="비드 높이 이산화 단계 수 (예: 3 → 0/h*1/3/h*2/3/h_max, 기본: 0 = 연속)")

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
    parser.add_argument("--gui",      action="store_true",         help="실시간 모니터링 GUI(PySide6) 실행")
    parser.add_argument("--no-viz",   action="store_true",         help="최적화 완료 후 최종 3D 시각화 생략")
    parser.add_argument("--export",   type=str, default="industrial_bead.k",
                        help="최종 결과 파일 저장 경로 (LS-DYNA .k, 기본: industrial_bead.k)")
    parser.add_argument("--mesh-size", type=float, default=10.0,  help="BC 탐색을 위한 기준 메시 크기 (mm)")

    args = parser.parse_args()

    # 최적화 파이프라인 실행
    run_industrial_topo(args)

if __name__ == "__main__":
    try:
        multiprocessing.freeze_support()
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
