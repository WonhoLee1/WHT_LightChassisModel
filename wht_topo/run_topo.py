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
from wht_topo.solver import WHTopographySolver, JaxTopoSolver  # JaxTopoSolver = 별칭
from wht_visualizer.wht_visualizer import WHTVisualizer
from wht_converter.wht_models import WHTMetadata
from test_jaxSSO.mesh_utils import generate_shell_tray

def apply_industrial_morphing(model, densities, max_height=10.0, vol_frac=0.3, discrete=True, draw_dir=None):
    """
    상용 S/W 수준의 비드 형상 생성 로직.
    - draw_dir: 비드가 돌출될 방향 벡터
    """
    print(f" -> [Morphing] 제조 가능성 고려 비드 생성 중 (Height: {max_height}mm, Dir: {draw_dir})...")
    
    if draw_dir is None:
        draw_dir = [0.0, 0.0, 1.0]
    d_dir = np.array(draw_dir, dtype=np.float64)
    d_dir = d_dir / (np.linalg.norm(d_dir) + 1e-10)
    
    all_z = [node.z for node in model.nodes.values()]
    z_min = min(all_z)
    z_threshold = z_min + 5.0 # 바닥면 근처만 비드 허용
    
    node_sum_density = {nid: 0.0 for nid in model.nodes.keys()}
    node_count = {nid: 0 for nid in model.nodes.keys()}
    elem_ids = sorted(model.elements.keys())
    
    for idx, eid in enumerate(elem_ids):
        d = float(densities[idx])
        for nid in model.elements[eid].node_ids:
            node_sum_density[nid] += d
            node_count[nid] += 1
            
    displacements = {}
    for nid, count in node_count.items():
        if count == 0: continue
        avg_d = node_sum_density[nid] / count
        
        if discrete:
            disp_factor = 1.0 if avg_d > vol_frac * 1.2 else 0.0
        else:
            diff = avg_d - vol_frac
            denom = np.sqrt(max(vol_frac, 1.0 - vol_frac)) + 1e-6
            disp_factor = np.sign(diff) * np.sqrt(np.abs(diff)) / denom
            
        displacements[nid] = disp_factor * max_height

    # Laplacian Smoothing
    smoothed_disps = displacements.copy()
    node_to_node = {nid: set() for nid in model.nodes.keys()}
    for elem in model.elements.values():
        for i, n1 in enumerate(elem.node_ids):
            for n2 in elem.node_ids[i+1:]:
                node_to_node[n1].add(n2); node_to_node[n2].add(n1)
                
    smooth_iters = 3
    for _ in range(smooth_iters):
        temp_disps = smoothed_disps.copy()
        for nid, neighbors in node_to_node.items():
            if nid not in smoothed_disps or not neighbors: continue
            neighbor_vals = [temp_disps[nn] for nn in neighbors if nn in temp_disps]
            if neighbor_vals:
                smoothed_disps[nid] = 0.7 * temp_disps[nid] + 0.3 * np.mean(neighbor_vals)

    moved_count = 0
    for nid, node in model.nodes.items():
        # 바닥면 노드이면서 변위가 있는 경우만 이동
        if node.z <= z_threshold and nid in smoothed_disps and smoothed_disps[nid] > 0.1:
            move_vec = d_dir * float(smoothed_disps[nid])
            node.x += move_vec[0]
            node.y += move_vec[1]
            node.z += move_vec[2]
            moved_count += 1
            
    print(f"    - 비드 생성 완료 ({moved_count} 노드 조정됨).")

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
    if args.target_freq > 0.1:
        print(f" -> [제약] 목표 진동수 설정: {args.target_freq}Hz")
        constraints.append(DynamicConstraint(target_freq=args.target_freq))

    # 3. 최적화 실행 — WHTopographySolver (설계 변수: 노드별 비드 높이)
    print(f" -> [최적화] 최소 비드 폭: {args.min_width}mm | 최대 비드 높이: {args.bead_height}mm")
    print(f" -> [제약] 비드 생성 영역 제한 (Bead Area Constraint): {args.bead_area * 100:.1f}%")
    weights = {"bending": args.w_bending, "twisting": args.w_twisting, "lifting": args.w_lifting}

    solver = WHTopographySolver(
        model, load_manager, constraints,
        bead_height_max=args.bead_height,
        bead_height_ratio=args.bead_area,    # 비드 면적 비율 (0~1)
        min_width=args.min_width,
        draw_dir=args.draw_dir,
        weights=weights,
        mesh_size_z=10.0,
        sym_x=args.sym_x,
        bead_connect=args.bead_connect,
        connect_gap=args.connect_gap,
    )
    
    # 모니터링 GUI 실행
    ui_process = None
    callback = None
    if args.gui:
        from wht_topo.monitor_ui import start_monitor_ui
        queue = multiprocessing.Queue()
        ui_process = multiprocessing.Process(target=start_monitor_ui, args=(queue,))
        ui_process.daemon = True # 메인 프로세스 종료 시 함께 종료
        ui_process.start()
        callback = queue.put

    final_heights = solver.solve(max_iter=args.iters, callback=callback)

    if ui_process and ui_process.is_alive():
        print(" -> [GUI] 최적화 완료. 모니터링 창을 닫으면 시각화로 넘어갑니다.")
        # UI 종료를 기다리거나 계속 진행할 수 있음. 
        # 여기서는 비동기적으로 두기 위해 STOP 신호만 보냄
        queue.put("STOP")

    # 4. 최종 형상을 모델 노드 좌표에 영구 적용
    solver.apply_final_shape()
    
    # 5. 익스포트
    if args.export:
        model.export_to_solver('lsdyna', args.export, reorder=True)
        print(f" -> [성공] 결과 저장 완료: {args.export}")

    # 6. 시각화 (옵션)
    if not args.no_viz:
        print(" -> [시각화] 결과 렌더링 중...")
        # WHTMeshModel을 WHTResultData로 변환 (시각화 모듈 호환성)
        result_data = model.to_wht_result_data()
        
        # 비드 높이 데이터 추가 (0~1 비율로 정규화하여 레전드와 일치시킴)
        heights_full = solver.get_full_heights()
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

    7. 배치 프로세스 (UI 없이 최적화만 수행 후 종료):
       python wht_topo/run_topo.py --iters 50 --no-viz --export final_bead_pattern.k
    ------------------------------------------------------------------------------------------------

    ■ 전체 옵션 레퍼런스
    ─────────────────────────────────────────────────────────────────────────────
    [기본 최적화]
      --iters N          최대 반복 횟수 (기본: 30)
      --bead-height F    최대 비드 높이 mm (기본: 10.0)
      --min-width F      최소 비드 폭 / 공간 필터 반경 mm (기본: 80.0)
      --bead-area F      비드 점유 면적 비율 0.0~1.0 (기본: 0.3 = 30%)

    [비드 형상 제어]
      --sym-x            Y-Z 평면 기준 좌우 대칭 제약 강제
                         → KDTree로 대칭 노드 쌍을 매핑, 설계변수/민감도 평균화
      --bead-connect     단절된 비드 영역을 Morphological Closing으로 자동 연결
                         → Dilation(팽창) → Erosion(수축)으로 갭을 채움
      --connect-gap F    비드 연결 시 채울 최대 갭 크기 mm (기본: 80.0)
                         → 메시 간격(40mm)의 2~3배 권장. 너무 크면 면적 제약 느슨해짐
      --draw-dir X Y Z   비드 돌출 방향 벡터 (기본: 0 0 1 = +Z 방향)

    [하중 케이스 가중치]
      --w-bending F      중앙 굽힘 하중 가중치 (기본: 1.0)
      --w-twisting F     대각 비틀림 하중 가중치 (기본: 1.5)
      --w-lifting F      4코너 개별 리프팅 하중 전체 가중치 (기본: 1.2)
                         → 4개 케이스로 분할되므로 케이스당 실효 가중치 = F/4
      --target-freq F    목표 고유 진동수 제약 Hz (기본: 0.0 = 미사용)

    [시각화 및 출력]
      --gui              실시간 모니터링 GUI (PySide6) 실행
      --no-viz           최적화 완료 후 최종 3D 시각화 생략
      --export PATH      최종 결과 파일 저장 경로 LS-DYNA .k (기본: industrial_bead.k)
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

    # ── 하중 케이스 가중치 (Weighted Sum Method) ─────────────────────────────
    parser.add_argument("--w-bending",   type=float, default=1.0, help="중앙 굽힘 하중 가중치 (기본: 1.0)")
    parser.add_argument("--w-twisting",  type=float, default=1.5, help="대각 비틀림 하중 가중치 (기본: 1.5)")
    parser.add_argument("--w-lifting",   type=float, default=1.2, help="4코너 리프팅 하중 전체 가중치 / 4개 케이스 분할 (기본: 1.2)")
    parser.add_argument("--target-freq", type=float, default=0.0, help="목표 고유 진동수 제약 (Hz, 기본: 0.0 = 미사용)")

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
    main()
