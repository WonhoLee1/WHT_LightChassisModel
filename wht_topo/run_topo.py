# -*- coding: utf-8 -*-
"""
run_topo.py
===========
WHT 산업용 섀시 비드 최적화(Topography Optimization) 통합 도구.

[핵심 기능]
1. 정적 하중 최적화: Bending, Twisting, Lifting 등 표준 케이스 지원
2. 동적 하중 최적화: 실측 CSV 데이터를 기반으로 충격 시점의 Snapshot을 자동 추출하여 반영
3. 지능형 가중치 제어: 모든 하중 케이스의 변형 에너지(Compliance)를 기준으로 가중치를 자동 정규화
4. 제조 제약 조건: 좌우 대칭(Sym-X), 비드 자동 연결(Bead Connect), 최소 폭(Min Width) 제어

[동적 하중 추출 전략]
- 바닥면을 N x N 그리드로 분할하고, 각 영역에서 '변형 에너지 합'이 최대가 되는 시점을 Peak Time으로 간주
- 해당 시점의 변위 및 관성 하중 상태를 정적 하중 케이스로 변환하여 최적화 루프에 주입
"""

import argparse
import numpy as np
import sys
import io
import multiprocessing
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# 터미널 출력 인코딩 강제 설정 (한글 깨짐 방지)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_topo.loads import StochasticLoadManager
from wht_topo.solver import WHTopographySolver
from wht_visualizer.wht_visualizer import WHTVisualizer
from wht_solver.load_cases import WHTLoadCase
from test_jaxSSO.exam4_dynamic import (
    calculate_local_z_history, calculate_corner_accelerations, _InterpLoadGroup
)

# ── 하중 케이스 설정 함수군 (정적) ───────────────────────────────────────────

def setup_bending_bc(model: WHTMeshModel):
    """중앙 집중 굽힘 하중: 플랜지 전체 고정 및 중앙부 하중 인가"""
    manager = StochasticLoadManager(model)
    flange = manager.get_boundary_nodes()
    center = manager.get_load_nodes()
    model.clear_bcs()
    model.apply_spc(flange, (0,1,2,3,4,5))
    model.apply_force(center, (2,), (-5000.0,), distribute=True)

def setup_bending_xspan_bc(model: WHTMeshModel):
    """X방향 스팬 굽힘: Y방향 양쪽 엣지만 고정 (긴 스팬 굽힘 모사)"""
    manager = StochasticLoadManager(model)
    ymin = manager.get_edge_flange_nodes('Y', 'min')
    ymax = manager.get_edge_flange_nodes('Y', 'max')
    center = manager.get_load_nodes()
    model.clear_bcs()
    model.apply_spc(ymin + ymax, (0,1,2,3,4,5))
    model.apply_force(center, (2,), (-5000.0,), distribute=True)

def setup_bending_yspan_bc(model: WHTMeshModel):
    """Y방향 스팬 굽힘: X방향 양쪽 엣지만 고정 (짧은 스팬 굽힘 모사)"""
    manager = StochasticLoadManager(model)
    xmin = manager.get_edge_flange_nodes('X', 'min')
    xmax = manager.get_edge_flange_nodes('X', 'max')
    center = manager.get_load_nodes()
    model.clear_bcs()
    model.apply_spc(xmin + xmax, (0,1,2,3,4,5))
    model.apply_force(center, (2,), (-5000.0,), distribute=True)

def setup_twisting_bc(model: WHTMeshModel):
    """대각 비틀림: C0/C3 고정 및 C1/C2 반전 하중"""
    manager = StochasticLoadManager(model)
    c0, c1, c2, c3 = [manager.get_corner_nodes(i) for i in range(4)]
    model.clear_bcs()
    model.apply_spc(c0 + c3, (0,1,2,3,4,5))
    model.apply_force(c1, (2,), (-3000.0,), distribute=True)
    model.apply_force(c2, (2,), (3000.0,), distribute=True)

def setup_twisting_alt_bc(model: WHTMeshModel):
    """반전 대각 비틀림: C1/C2 고정 및 C0/C3 반전 하중"""
    manager = StochasticLoadManager(model)
    c0, c1, c2, c3 = [manager.get_corner_nodes(i) for i in range(4)]
    model.clear_bcs()
    model.apply_spc(c1 + c2, (0,1,2,3,4,5))
    model.apply_force(c0, (2,), (-3000.0,), distribute=True)
    model.apply_force(c3, (2,), (3000.0,), distribute=True)

def setup_lifting_bc(model: WHTMeshModel, corner_idx: int):
    """1코너 리프팅: 나머지 3개 코너 고정 후 1개 코너 상향 하중"""
    manager = StochasticLoadManager(model)
    corners = [manager.get_corner_nodes(i) for i in range(4)]
    fixed = [nid for j, c in enumerate(corners) if j != corner_idx for nid in c]
    model.clear_bcs()
    model.apply_spc(fixed, (0,1,2))
    model.apply_force(corners[corner_idx], (2,), (3000.0,), distribute=True)

# ── 메인 파이프라인 ─────────────────────────────────────────────────────────

def run(args):
    print(f"\n [0] WHT Industrial Topography Pipeline ({args.solver_method.upper()})")
    
    # 1. 해석 모델 생성
    from test_jaxSSO.mesh_utils import generate_shell_tray
    model = generate_shell_tray(mesh_size=args.mesh_size)
    
    # 2. 동적 하중 케이스 추출 (Snapshot)
    dynamic_snapshots = []
    if args.dynamic_opts:
        opts = [s.strip() for s in args.dynamic_opts.split(',')]
        csv_path = opts[0]
        t_start = float(opts[1]) if len(opts) > 1 else 1.6
        add_inertia = "--add-inertia" in opts or args.add_inertia
        
        dynamic_snapshots = extract_dynamic_snapshots(
            model, csv_path, t_start, 
            grid_n=args.dyn_grid, 
            add_inertia=add_inertia,
            solver_method=args.solver_method
        )

    # 3. 최적화 하중 구성 및 정규화
    static_configs = [
        ("Bending", args.w_bending, setup_bending_bc),
        ("X-Span",  args.w_bending_xspan, setup_bending_xspan_bc),
        ("Y-Span",  args.w_bending_yspan, setup_bending_yspan_bc),
        ("Twist",   args.w_twisting, setup_twisting_bc),
        ("Twist-A", args.w_twisting_alt, setup_twisting_alt_bc),
    ]
    w_lift_per = args.w_lifting / 4.0
    for i in range(4):
        static_configs.append((f"Lift-{i}", w_lift_per, lambda m, idx=i: setup_lifting_bc(m, idx)))

    from wht_solver.wht_solver import WHTSolver
    fea_solver = WHTSolver(model)
    load_cases_for_opt = []
    
    print("\n [3] 가중치 정규화: 모든 하중 케이스의 변형 에너지(C)를 동일 스케일로 조정")
    print("     (W_norm = User_Weight / Initial_Compliance)")
    
    # (A) 정적 케이스 정규화
    for name, user_w, setup_func in static_configs:
        if user_w <= 0: continue
        setup_func(model)
        lc = WHTLoadCase(name=name)
        lc.bcs = model.bcs; lc.forces = model.forces
        
        res = fea_solver.solve_static(lc)
        # Compliance C = f^T u = 2 * Strain Energy
        compliance = float(np.dot(res._u_aug, res._f_aug)) if hasattr(res, "_u_aug") else 1.0
        
        norm_w = user_w / (compliance + 1e-12)
        load_cases_for_opt.append((lc, norm_w))
        print(f"    - Static: {name:10s} | W={user_w:.2f} -> NormW={norm_w:.2e} (C={compliance:.2e})")

    # (B) 동적 스냅샷 정규화
    for snap in dynamic_snapshots:
        user_w = 1.0 # 동적 케이스 기본 가중치
        lc = snap['lc']; compliance = snap['compliance']
        norm_w = user_w / (compliance + 1e-12)
        load_cases_for_opt.append((lc, norm_w))
        print(f"    - Dynamic: {lc.name:10s} | W={user_w:.2f} -> NormW={norm_w:.2e} (C={compliance:.2e})")

    # 4. 최적화 실행
    optimizer = WHTopographySolver(
        model, 
        load_cases=load_cases_for_opt,
        bead_height_max=args.bead_height,
        bead_height_ratio=args.bead_area,
        min_width=args.min_width,
        sym_x=args.sym_x,
        bead_connect=args.bead_connect,
        connect_gap=args.connect_gap,
        bead_steps=args.height_steps
    )
    
    # 모니터링 GUI (옵션)
    ui_process = None; callback = None; stop_event = None
    if not args.no_gui:
        from wht_topo.monitor_ui import start_monitor_ui
        queue = multiprocessing.Queue()
        stop_event = multiprocessing.Event()
        ui_process = multiprocessing.Process(target=start_monitor_ui, args=(queue, stop_event))
        ui_process.daemon = False; ui_process.start(); callback = queue.put

    optimizer.solve(max_iter=args.iters, callback=callback, stop_event=stop_event)

    # 5. 결과 저장 및 시각화
    if args.export:
        model.export_to_solver('lsdyna', args.export, reorder=True)
        print(f"\n [5] 결과 저장 완료: {args.export}")
    
    if not args.no_viz:
        viz = WHTVisualizer(title="Industrial Topography Result")
        viz.show_result(model.to_wht_result_data())
        viz.plotter.app.exec_()

# ── 동적 하중 추출 로직 ──────────────────────────────────────────────────────

def extract_dynamic_snapshots(model, csv_path, t_start, grid_n=3, add_inertia=True, solver_method="scipy"):
    """
    영역별 변형 에너지(Strain Energy) 합을 기준으로 최악의 시점을 추출하여 정적 하중 케이스로 반환합니다.
    """
    from wht_solver.wht_dynamic_solver import WHTDynamicSolver
    from wht_solver.wht_dynamic_common import DampingSpec
    
    print(f"\n [1] 동적 해석 수행 (CSV: {csv_path}, Start: {t_start}s)")
    
    df = pd.read_csv(csv_path)
    df = df[df['Time'] >= t_start].reset_index(drop=True)
    time_arr = (df['Time'] - df['Time'].iloc[0]).to_numpy()
    
    local_z_dict, traj_mm = calculate_local_z_history(df, time_arr)
    corner_accels = calculate_corner_accelerations(traj_mm, time_arr[1]-time_arr[0])
    
    # RBE3/Master Node 기반 변위 경계 조건 설정
    corner_labels = ['C5', 'C6', 'C7', 'C8']
    bot_groups = model.get_corner_nodes(radius=150.0)
    load_groups = []; master_nids = []
    
    for idx, (_, cnids) in enumerate(bot_groups):
        pts = np.array([model.nodes[nid].coords() for nid in cnids])
        mnid = 900000 + idx
        model.add_node(mnid, *np.mean(pts, axis=0))
        model.add_rbe3(900000+idx, mnid, cnids, dofs=(0,1,2))
        master_nids.append(mnid)
        lg = _InterpLoadGroup(node_ids=[mnid], dof=2, time_arr=time_arr, disp_arr=local_z_dict[corner_labels[idx]])
        load_groups.append(lg)

    # 관성 하중 (-m*a) 설정
    all_pts = np.array([n.coords() for n in model.nodes.values() if n.id < 900000])
    xmin, xmax = all_pts[:,0].min(), all_pts[:,0].max()
    ymin, ymax = all_pts[:,1].min(), all_pts[:,1].max()
    dx, dy = (xmax-xmin) or 1.0, (ymax-ymin) or 1.0
    
    solver = WHTDynamicSolver(model)
    jm, sorted_nids, nid_to_idx = solver._build_jaxsso_model()
    m_diag = solver._assemble_lumped_mass(jm, jm.ndof, sorted_nids, nid_to_idx)
    
    if add_inertia:
        for nid, node in model.nodes.items():
            if nid >= 900000: continue
            ix = nid_to_idx.get(nid); nx, ny = (node.x-xmin)/dx, (node.y-ymin)/dy
            a = corner_accels[:,2]*(1-nx)*(1-ny) + corner_accels[:,1]*nx*(1-ny) + \
                corner_accels[:,0]*nx*ny + corner_accels[:,3]*(1-nx)*ny
            mass = m_diag[ix*6+2]
            if mass > 0:
                lg_i = _InterpLoadGroup(node_ids=[nid], dof=2, time_arr=time_arr, disp_arr=-mass*a)
                lg_i.load_type = "FORCE"; load_groups.append(lg_i)

    # 과도 응답 해석 실행
    print(f"     -> 과도 응답 해석 중... ({len(time_arr)} steps)")
    res = solver.solve_direct_dynamic(load_groups, time_arr, damping=DampingSpec(zeta=0.02), method=solver_method)
    
    # 영역 분할 및 변형 에너지 피크 시점 추출
    print(f" [2] 영역별 변형 에너지 합(Grid: {grid_n}x{grid_n}) 기반 피크 시점 탐색 중...")
    floor_nids = [nid for nid, n in model.nodes.items() if nid < 900000 and n.z < 5.0]
    gx_edges = np.linspace(xmin, xmax, grid_n + 1)
    gy_edges = np.linspace(ymin, ymax, grid_n + 1)
    
    snapshots = []; seen_times = set()
    K_base = solver._assemble_K_scipy(jm, sorted_nids, nid_to_idx)

    for i in range(grid_n):
        for j in range(grid_n):
            region_nids = [nid for nid in floor_nids if 
                           gx_edges[i] <= model.nodes[nid].x < gx_edges[i+1] and
                           gy_edges[j] <= model.nodes[nid].y < gy_edges[j+1]]
            if not region_nids: continue
            
            # 영역 내 변위 에너지 대용값(u^2)으로 피크 탐색 (정밀도와 속도 절충)
            u_region = res.displacements[:, [nid_to_idx[nid]*6+2 for nid in region_nids]]
            energy_proxy = np.sum(u_region**2, axis=1)
            
            peak_idx = np.argmax(energy_proxy)
            if peak_idx not in seen_times:
                seen_times.add(peak_idx)
                t_val = res.time[peak_idx]
                u_snap = res.displacements[peak_idx]
                
                # Snapshot을 정적 LoadCase로 변환
                lc = WHTLoadCase(name=f"Dyn_{t_val:.3f}s")
                for midx in range(4):
                    val = u_snap[nid_to_idx[900000+midx]*6 + 2]
                    lc.add_bc([900000+midx], dofs=(2,), vals=(val,))
                lc.add_bc([900000], dofs=(0,1)) # 안정성 구속
                
                # 실제 컴플라이언스(f^T u) 계산
                u_vec = res._u_aug[peak_idx] if hasattr(res, "_u_aug") else None
                comp = float(u_vec.T @ res._f_aug[peak_idx]) if u_vec is not None else energy_proxy[peak_idx]
                snapshots.append({'lc': lc, 'compliance': comp})
                
    # 모델 원상 복구
    for nid in range(900000, 900004): del model.nodes[nid]
    model.elements = {eid: e for eid, e in model.elements.items() if eid < 900000}
    return snapshots

# ── CLI 엔트리 포인트 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WHT Industrial Topography Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
[실행 예제]
  1. 기본 최적화 (정적 하중 케이스만 사용):
     python wht_topo/run_topo.py --iters 20 --sym-x --bead-area 0.25

  2. 동적 하중 통합 최적화 (실측 데이터 기반):
     python wht_topo/run_topo.py --dynamic-opts "wht_topo/sample_pos.csv, 1.6" --add-inertia

  3. JAX 기반 고속 동적-토포 통합 최적화:
     python wht_topo/run_topo.py --dynamic-opts "wht_topo/sample_pos.csv, 1.6" --add-inertia --solver-method jax

  4. 비드 이산화 및 연결 조건 강화:
     python wht_topo/run_topo.py --iters 30 --height-steps 2 --bead-connect --connect-gap 100.0
        """
    )
    # 최적화 제어
    parser.add_argument("--iters",        type=int,   default=15,    help="최대 반복 횟수")
    parser.add_argument("--bead-height",  type=float, default=10.0,  help="최대 비드 높이 (mm)")
    parser.add_argument("--min-width",    type=float, default=30.0,  help="최소 비드 폭 (필터 반경, mm)")
    parser.add_argument("--bead-area",    type=float, default=0.35,  help="비드 점유 면적 비율 (0~1)")
    parser.add_argument("--sym-x",         action="store_true", default=True, help="좌우 대칭 제약 조건")
    parser.add_argument("--bead-connect",  action="store_true", default=True, help="단절된 비드 자동 연결")
    parser.add_argument("--connect-gap",   type=float, default=120.0, help="비드 연결 최대 간격 (mm)")
    parser.add_argument("--height-steps",  type=int,   default=2,     help="비드 높이 단계 (2={0, h_max})")
    
    # 하중 가중치
    parser.add_argument("--w-bending",       type=float, default=1.0)
    parser.add_argument("--w-bending-xspan", type=float, default=0.8)
    parser.add_argument("--w-bending-yspan", type=float, default=0.8)
    parser.add_argument("--w-twisting",      type=float, default=1.5)
    parser.add_argument("--w-twisting-alt",  type=float, default=1.5)
    parser.add_argument("--w-lifting",       type=float, default=1.2)
    
    # 동적 하중 설정
    parser.add_argument("--dynamic-opts", type=str, help="CSV경로,시작시간 e.g. 'sample.csv,1.6'")
    parser.add_argument("--add-inertia",  action="store_true", help="동적 하중 추출 시 관성 하중(-ma) 포함")
    parser.add_argument("--dyn-grid",     type=int,   default=3,     help="동적 피크 탐색을 위한 그리드 분할 수 (N x N)")
    
    # 시스템 설정
    parser.add_argument("--mesh-size",    type=float, default=10.0)
    parser.add_argument("--solver-method", type=str, default="scipy", choices=["scipy", "jax"])
    parser.add_argument("--no-gui",        action="store_true", help="실시간 GUI 생략")
    parser.add_argument("--no-viz",        action="store_true", help="최종 시각화 생략")
    parser.add_argument("--export",        type=str, default="industrial_bead.k", help="결과 저장 파일명")
    
    args = parser.parse_args()
    run(args)
