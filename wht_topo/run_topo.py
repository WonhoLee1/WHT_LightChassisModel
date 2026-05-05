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
    weights = {"bending": args.w_bending, "twisting": args.w_twisting, "lifting": args.w_lifting}

    solver = WHTopographySolver(
        model, load_manager, constraints,
        bead_height_max=args.bead_height,
        bead_height_ratio=args.vol_frac,    # 비드 면적 비율 (0~1)
        min_width=args.min_width,
        draw_dir=args.draw_dir,
        weights=weights,
        mesh_size_z=10.0,
    )
    final_heights = solver.solve(max_iter=args.iters)

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
        
        # 비드 높이 데이터 추가 (스칼라 필드)
        heights_full = solver.get_full_heights()
        result_data.add_scalar_field("Bead_Height", heights_full)
        
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
    parser = argparse.ArgumentParser(description="Industrial Topography Optimization Tool")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--bead-height", type=float, default=10.0)
    parser.add_argument("--min-width", type=float, default=80.0, help="최소 비드 폭 (mm)")
    parser.add_argument("--vol-frac", type=float, default=0.3)
    parser.add_argument("--discrete", action="store_true", help="이산적(0 or Max) 비드 생성")
    parser.add_argument("--export", type=str, default="industrial_bead.k")
    parser.add_argument("--no-viz", action="store_true", help="시각화 창 띄우지 않음")
    
    # 하중 가중치 인자 추가
    parser.add_argument("--w-bending", type=float, default=1.0, help="벤딩 강성 가중치")
    parser.add_argument("--w-twisting", type=float, default=1.5, help="비틀림 강성 가중치")
    parser.add_argument("--w-lifting", type=float, default=1.2, help="리프팅 강성 가중치")
    parser.add_argument("--target-freq", type=float, default=0.0, help="목표 고유 진동수 (Hz)")
    parser.add_argument("--draw-dir", type=float, nargs=3, default=[0.0, 0.0, 1.0], help="비드 돌출 방향 (X Y Z)")
    
    args = parser.parse_args()
    run_industrial_topo(args)

if __name__ == "__main__":
    main()
