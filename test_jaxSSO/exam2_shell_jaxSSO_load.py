# -*- coding: utf-8 -*-
"""
exam2_shell_jaxSSO_load.py
==========================
WHT FEM Framework — 쉘(Shell) 트레이 정적 해석 예제 (개선판)

개요:
---------
본 스크립트는 중앙 면적 하중을 받는 구조용 쉘 트레이의 정밀 정적 해석을 수행합니다.
모드 해석 예제와 동일한 고급 훅-폴드(Hook-Fold) 플랜지 기하구조를 활용합니다.

주요 기능:
-------------
- 다중 노드에 분산된 중앙 면적 하중(Area Load) 적용.
- 고정밀 정적 응답 해석 (변위 및 본-미세스 응력).
- 다양한 메시 타입(QUAD4, TRIA3_FREE) 간의 성능 및 정확도 비교.
- RBE3 요소 및 WHTSelector를 활용한 고급 노드 선택 시나리오 포함.
"""

import numpy as np
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_selector import WHTSelector
from wht_solver.wht_solver import WHTSolver
from wht_solver.load_cases import WHTLoadCase
from wht_visualizer.wht_visualizer import WHTVisualizer
from wht_converter.wht_models import WHTMetadata
from mesh_utils import generate_shell_tray


# ==============================================================================
# 0. 설정 및 데이터 스키마 (CONFIGURATION & SCHEMA)
# ==============================================================================

@dataclass
class PipelineConfig:
    """
    쉘 트레이 정적 해석 파이프라인을 위한 통합 설정 클래스입니다.
    """
    width:  float = 1800.0
    length: float = 1200.0
    height: float = 35.0
    thickness: float = 0.5
    mesh_type: str = 'quad4'
    mesh_size_xy: float = 60.0
    mesh_size_z:  float = 10.0
    draft_angle:  float = 35.0
    flanges: tuple = (False, True, True, True) 
    flange_segments: List[Tuple[float, float]] = field(default_factory=list)
    load_area_size: float = 200.0
    total_force: float = 100.0
    E: float = 210000.0
    nu: float = 0.3
    rho: float = 7.85e-9


# ==============================================================================
# 1. 핵심 파이프라인 함수 (CORE PIPELINE FUNCTIONS)
# ==============================================================================

def build_structural_model(cfg: PipelineConfig) -> Tuple[WHTMeshModel, Dict]:
    """기하구조를 생성하고 WHT 모델을 조립합니다."""
    print(f" -> 기하구조 생성 중 [{cfg.mesh_type.upper()}]...")
    node_db, elem_db = generate_shell_tray(
        width=cfg.width, length=cfg.length, height=cfg.height,
        mesh_size_xy=cfg.mesh_size_xy, mesh_size_z=cfg.mesh_size_z,
        draft_angle=cfg.draft_angle, flange_segments=cfg.flange_segments,
        flanges=cfg.flanges, mesh_type=cfg.mesh_type
    )
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db)
    MID, PID = 1, 1
    model.add_material(MID, E=cfg.E, nu=cfg.nu, rho=cfg.rho)
    model.add_property(PID, "PSHELL", cfg.thickness, MID)
    for elem in model.elements.values(): elem.pid = PID
    
    # 상단 림 고정 (SPC)
    max_z = cfg.height + sum([seg[1] for seg in cfg.flange_segments])
    rim_nids = WHTSelector(model).by_box(z=(max_z - 0.1, max_z + 0.1)).get_ids()
    model.apply_spc(rim_nids, (0, 1, 2, 3, 4, 5))
    print(f"    [경계조건] 최상단 림 노드 {len(rim_nids)}개 고정.")
    return model, node_db


def evaluate_static_response(model: WHTMeshModel, cfg: PipelineConfig):
    """중앙 영역 하중 해석을 수행합니다."""
    # Selector를 사용하여 중앙 영역 노드 선택
    cx, cy = cfg.width / 2.0, cfg.length / 2.0
    half_s = cfg.load_area_size / 2.0
    target_nodes = (WHTSelector(model)
                    .by_box(x=(cx - half_s, cx + half_s), 
                            y=(cy - half_s, cy + half_s), 
                            z=(-0.1, 1.0))
                    .get_ids())
    
    if not target_nodes:
        target_nodes = [min(model.nodes.keys(), key=lambda n: (model.nodes[n].x - cx)**2 + (model.nodes[n].y - cy)**2)]
    
    lc = WHTLoadCase("AreaLoad")
    force_per_node = -cfg.total_force / len(target_nodes)
    for nid in target_nodes: lc.add_force(nid, (2,), (force_per_node,))
    
    solver = WHTSolver(model)
    return solver.solve_static(lc)


def evaluate_static_response_twist(model: WHTMeshModel, cfg: PipelineConfig):
    """
    RBE3 및 WHTSelector를 활용한 비틀림(Twist) 테스트 시나리오입니다.
    Selector를 체이닝하여 복잡한 선택 로직을 직관적으로 구현합니다.
    """
    print("\n" + "-"*80)
    print(" -> [비틀림 시나리오] WHTSelector를 활용한 정밀 노드 선택 중...")
    print("-"*80)
    
    model.spc_conditions = [] # 기존 SPC 제거
    
    nodes_arr = model.nodes_array()
    xmin, xmax = nodes_arr[:, 0].min(), nodes_arr[:, 0].max()
    ymin, ymax = nodes_arr[:, 1].min(), nodes_arr[:, 1].max()
    zmax = nodes_arr[:, 2].max()
    
    # 좌측 시드 (Xmin, Ymin, Zmax 부근)
    seed_l = min(model.nodes.keys(), key=lambda n: (model.nodes[n].x - xmin)**2 + (model.nodes[n].y - ymin)**2 + (model.nodes[n].z - zmax)**2)
    seed_r = min(model.nodes.keys(), key=lambda n: (model.nodes[n].x - xmax)**2 + (model.nodes[n].y - ymin)**2 + (model.nodes[n].z - zmax)**2)
    
    y_range = (ymin + (ymax-ymin)*0.2, ymax - (ymax-ymin)*0.2) # 중앙 60% 영역
    z_flange_min = cfg.height - 1.0 
    
    # --- [WHTSelector 체이닝 적용] ---
    # 1. 시드로부터 림 경로(Path) 추출 -> 2. Y 영역 필터링(Box) -> 3. 전체 플랜지 단면 확장(Face Curvature)
    left_slave_nids = (WHTSelector(model)
                       .by_path(seed_l, angle_limit_deg=20.0)
                       .by_box(y=y_range)
                       .expand_by_face(angle_limit_deg=30.0, z_min=z_flange_min)
                       .get_ids())
                       
    right_slave_nids = (WHTSelector(model)
                        .by_path(seed_r, angle_limit_deg=20.0)
                        .by_box(y=y_range)
                        .expand_by_face(angle_limit_deg=30.0, z_min=z_flange_min)
                        .get_ids())
    
    # [Fallback] 만약 고급 선택 기능으로 노드를 찾지 못한 경우, 단순 영역 선택으로 대체 (강건성 확보)
    if not left_slave_nids:
        print("    <!> 좌측 플랜지 선택 실패. 단순 영역 선택(X-min)으로 전환합니다.")
        left_slave_nids = WHTSelector(model).by_box(x=(xmin-0.1, xmin+1.0), y=y_range).get_ids()
    if not right_slave_nids:
        print("    <!> 우측 플랜지 선택 실패. 단순 영역 선택(X-max)으로 전환합니다.")
        right_slave_nids = WHTSelector(model).by_box(x=(xmax-1.0, xmax+0.1), y=y_range).get_ids()

    print(f"    [선택] 최종 결과: 좌측 {len(left_slave_nids)}개, 우측 {len(right_slave_nids)}개 선택됨.")
    
    # 마스터 노드 추가 및 RBE3 설정
    if not left_slave_nids or not right_slave_nids:
        raise ValueError("비틀림 테스트를 위한 노드 선택에 실패했습니다. 모델의 기하구조를 확인하십시오.")

    l_mid = np.mean([model.nodes[n].coords() for n in left_slave_nids], axis=0)
    r_mid = np.mean([model.nodes[n].coords() for n in right_slave_nids], axis=0)
    
    m_left_id, m_right_id = max(model.nodes.keys()) + 1, max(model.nodes.keys()) + 2
    model.add_node(m_left_id, l_mid[0], l_mid[1], l_mid[2])
    model.add_node(m_right_id, r_mid[0], r_mid[1], r_mid[2])
    
    model.add_rbe3(900, m_left_id, left_slave_nids)
    model.add_rbe3(901, m_right_id, right_slave_nids)
    
    lc = WHTLoadCase("Twist_10deg_X")
    lc.add_bc(m_left_id, (0, 1, 2, 3, 4, 5), 0.0)
    lc.add_bc(m_right_id, (0, 1, 2, 4, 5), 0.0)
    lc.add_bc(m_right_id, (3,), 10.0 * np.pi / 180.0)

    solver = WHTSolver(model)
    return solver.solve_static(lc)


def print_result_table(all_results: Dict[str, any]):
    """해석 결과를 요약 출력합니다."""
    print("\n" + "#"*95)
    print(" #  정적 해석 결과 비교 요약 (WHTSelector 적용)                                                #")
    print("#"*95)
    print(f"  {'메시 타입':<15} | {'노드 수':<10} | {'최대 Uz (mm)':<15} | {'최대 응력 (MPa)':<15}")
    print("-" * 95)
    for mtype, res in all_results.items():
        if res is None: continue
        max_uz = np.abs(res.displacement[:, 2]).max()
        max_vm = getattr(res, '_max_vm_diagnostic', 0.0)
        print(f"  {mtype.upper():<15} | {res.displacement.shape[0]:<10} | {max_uz:15.5f} | {max_vm:15.5f}")
    print("-" * 95)


def main():
    hook_sequence = [(0, 12), (-5, 0), (0, -10), (-10, 0)]
    test_suite = [
        PipelineConfig(mesh_type='quad4', flange_segments=hook_sequence),
        PipelineConfig(mesh_type='tria3_free', flange_segments=hook_sequence),
    ]
    
    all_summary = {}
    vis_queue = []
    
    for cfg in test_suite:
        try:
            model, _ = build_structural_model(cfg)
            result = evaluate_static_response(model, cfg)
            all_summary[cfg.mesh_type] = result
            vis_queue.append((cfg, result, model))
        except Exception:
            traceback.print_exc()

    print_result_table(all_summary)
    
    # 비틀림 시나리오 실행 (모든 변에 플랜지가 있는 설정으로 테스트)
    cfg_t = PipelineConfig(mesh_type='quad4', mesh_size_xy=60.0, flanges=(True, True, True, True))
    model_t, _ = build_structural_model(cfg_t)
    result_t = evaluate_static_response_twist(model_t, cfg_t)
    
    # 시각화
    viz = WHTVisualizer(title="WHTSelector 기반 비틀림 해석 결과", show=True)
    meta = WHTMetadata(
        solver_name="JaxSSO", 
        solver_version="2.1.0",
        analysis_type="static", 
        coordinate_system="cartesian",
        unit_length="mm", 
        unit_force="N"
    )
    viz.load_results(result_t.to_wht_result_data(meta, model_t))
    if hasattr(viz.plotter, 'app'): viz.plotter.app.exec_()

if __name__ == "__main__":
    main()
