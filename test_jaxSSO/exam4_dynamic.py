# -*- coding: utf-8 -*-
"""
[WHT_LightChassisModel] Exam 4: Implicit Dynamic Analysis (Large Rotation Handling)
================================================================================
대회전(Large Rotation) 및 낙하 충격을 포함한 CSV 실측 데이터 기반 동해석.

핵심 기능:
  - 대회전 대응 (Rigid Body Decoupling): 
    CSV의 글로벌 궤적에서 강체 회전을 분리하여 샤시 로컬 좌표계 기준의 순수 굽힘(Local Z)만 추출.
  - 자동 관성 하중 (Inertial Load): 
    낙하 시의 급격한 가속도 변화를 감지하여 샤시 본체에 관성력(F=-ma)을 인가. 
    (코너 변위만 인가했을 때 응력이 과소평가되는 문제 해결)
  - Nodal BC (SPCD): RBE3 마스터 노드에 로컬 벤딩 변위 인가.

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
    4 코너 기준으로 반경 내 + z 범위 노드 그룹 반환.
    순서: C5(+X,+Y), C6(+X,-Y), C7(-X,-Y), C8(-X,+Y)
    반환: [((cx, cy), [nid, ...]), ...]
    """
    all_xyz = np.array([v for v in node_db.values()])
    mask_z  = (all_xyz[:, 2] >= z_min) & (all_xyz[:, 2] <= z_max)
    xyz_z   = all_xyz[mask_z]

    if len(xyz_z) == 0:
        raise RuntimeError(f"z=[{z_min},{z_max}]mm 범위 내 노드가 없습니다.")

    x_min, x_max = xyz_z[:, 0].min(), xyz_z[:, 0].max()
    y_min, y_max = xyz_z[:, 1].min(), xyz_z[:, 1].max()
    print(f"     [Mesh Info] detected X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")

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
            raise RuntimeError(f"코너 ({cx:.0f},{cy:.0f}) 반경 {radius:.0f}mm 내 노드 없음.")
        groups.append(((cx, cy), nids))
    return groups


def calculate_local_z_history(csv_df, time_arr):
    """
    4개 코너(C5, C6, C7, C8)의 3D 궤적으로부터 강체 회전을 제거한 후,
    샤시 로컬 좌표계 기준의 순수 수직(Z) 변위만을 추출합니다.
    (로컬 X/Y는 0으로 고정하여 면내 변형 노이즈 제거)
    
    Returns:
        Dict[str, np.ndarray]: { 'C5': z_arr, 'C6': z_arr, ... } (T,)
    """
    n_steps = len(time_arr)
    corner_labels = ['C5', 'C6', 'C7', 'C8']
    
    # 1. (T, 4, 3) 궤적 데이터 구축 (mm 단위)
    traj = np.zeros((n_steps, 4, 3))
    for i, lbl in enumerate(corner_labels):
        for j, ax in enumerate(['X', 'Y', 'Z']):
            col = f"{lbl}_pos_{ax}"
            if col in csv_df.columns:
                traj[:, i, j] = csv_df[col].to_numpy(dtype=float) * 1000.0
            else:
                # 데이터 부재 시 0으로 채움 (추후 경고 필요할 수 있음)
                pass

    local_z_results = {lbl: np.zeros(n_steps) for lbl in corner_labels}
    p_loc_t0 = None

    print(f"    - [RigidBodyDecouple] Processing {n_steps} steps for local frame projection...")
    
    for t in range(n_steps):
        pts = traj[t] # (4, 3)
        
        # A. 중심점 및 로컬 좌표계(R) 구성
        origin = np.mean(pts, axis=0)
        p_c = pts - origin
        
        # C5(+X+Y), C6(+X-Y), C7(-X-Y), C8(-X+Y)
        # X축 방향: C7->C6 및 C8->C5의 평균
        v_x = ( (p_c[1] - p_c[2]) + (p_c[0] - p_c[3]) ) / 2.0
        # Y축 방향: C7->C8 및 C6->C5의 평균
        v_y = ( (p_c[3] - p_c[2]) + (p_c[0] - p_c[1]) ) / 2.0
        
        # 법선 벡터(Z) 산출 및 정규화
        z_loc = np.cross(v_x, v_y)
        z_norm = np.linalg.norm(z_loc)
        if z_norm < 1e-12:
            z_loc = np.array([0, 0, 1.0])
        else:
            z_loc /= z_norm
            
        # 전역 Z축 방향성 유지 (뒤집힘 방지)
        if z_loc @ np.array([0, 0, 1.0]) < 0:
            z_loc = -z_loc
            
        # 그람-슈미트 직교화
        x_loc = v_x / (np.linalg.norm(v_x) + 1e-12)
        y_loc = np.cross(z_loc, x_loc)
        x_loc = np.cross(y_loc, z_loc) # Re-ortho
        
        R = np.stack([x_loc, y_loc, z_loc], axis=1) # (3, 3)
        
        # B. 로컬 좌표계로 투영
        p_loc = p_c @ R # (4, 3)
        
        if t == 0:
            p_loc_t0 = p_loc.copy()
            
        # C. 초기 상태 대비 상대 변위 (Z 성분만 추출)
        delta_p_loc = p_loc - p_loc_t0
        for i, lbl in enumerate(corner_labels):
            local_z_results[lbl][t] = delta_p_loc[i, 2]

    return local_z_results, traj


def calculate_corner_accelerations(traj, dt):
    """
    (T, 4, 3) 궤적 데이터로부터 4개 코너 각각의 Z 가속도를 산출합니다.
    Returns:
        np.ndarray: (T, 4) - C5, C6, C7, C8 순서의 가속도 이력
    """
    n_steps = traj.shape[0]
    accels = np.zeros((n_steps, 4))
    
    def smooth(y, box_pts=5):
        box = np.ones(box_pts)/box_pts
        y_s = np.convolve(y, box, mode='same')
        y_s[:box_pts] = y[:box_pts]
        y_s[-box_pts:] = y[-box_pts:]
        return y_s

    for i in range(4):
        z = traj[:, i, 2]
        # 스무딩 -> 1차 미분 -> 스무딩 -> 2차 미분 -> 스무딩
        z_s = smooth(z)
        v_z = np.gradient(z_s, dt)
        v_s = smooth(v_z)
        a_z = np.gradient(v_s, dt)
        accels[:, i] = smooth(a_z)
        
    return accels


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
        
        # [NEW] t-start filtering
        t_start_limit = getattr(args, 't_start', 0.0)
        if t_start_limit > 0:
            csv_df = csv_df[csv_df['Time'] >= t_start_limit].copy()
            if csv_df.empty:
                raise ValueError(f"CSV에 {t_start_limit}s 이후의 데이터가 없습니다.")
            print(f"     [Filter] t >= {t_start_limit}s 적용 (시작점: {csv_df['Time'].iloc[0]:.4f}s)")

        time_arr_raw = csv_df['Time'].to_numpy(dtype=float)
        t_total_val = float(time_arr_raw[-1] - time_arr_raw[0])
        time_arr = time_arr_raw - time_arr_raw[0] # Shift to 0 for solver
        
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

    # 4. 하단 코너 SPCD 하중 그룹 구성 (로컬 변위 추출 방식) -------------------------
    load_groups = []

    if csv_df is not None:
        if getattr(args, 'use_global_z', False):
            print(f" [3] CSV 기반 글로벌 Z 변위 직접 인가 (기존 방식, t0 기준 상대값)")
            # 1. (T, 4, 3) 궤적 데이터 구축 (mm 단위)
            n_steps = len(time_arr)
            traj_mm = np.zeros((n_steps, 4, 3))
            corner_labels = ['C5', 'C6', 'C7', 'C8']
            for i, lbl in enumerate(corner_labels):
                for j, ax in enumerate(['X', 'Y', 'Z']):
                    col = f"{lbl}_pos_{ax}"
                    if col in csv_df.columns:
                        traj_mm[:, i, j] = csv_df[col].to_numpy(dtype=float) * 1000.0
            
            # 각 코너별 Z 변위 (t - t0)
            corner_z_data = {}
            for i, lbl in enumerate(corner_labels):
                corner_z_data[lbl] = traj_mm[:, i, 2] - traj_mm[0, i, 2]
        else:
            print(f" [3] CSV 기반 로컬 변위(Z) 추출 및 SPCD 하중 구성 (Large Rotation 대응)")
            # 로컬 좌표계 투영을 통한 순수 벤딩 변위(Z) 산출
            corner_z_data, traj_mm = calculate_local_z_history(csv_df, time_arr)
        
        CORNER_MAP = {
            0: 'C5', # (+X,+Y)
            1: 'C6', # (+X,-Y)
            2: 'C7', # (-X,-Y)
            3: 'C8', # (-X,+Y)
        }

        for idx, (_, corner_nids) in enumerate(bot_groups):
            cn_label = CORNER_MAP[idx]
            z_vals_rel = corner_z_data[cn_label]
            
            pts = np.array([model.nodes[nid].coords() for nid in corner_nids])
            center = np.mean(pts, axis=0)
            
            master_nid = 900000 + idx
            model.add_node(master_nid, center[0], center[1], center[2])
            
            rbe3_id = 900000 + idx
            model.add_rbe3(rbe3_id, master_nid, corner_nids, dofs=(0, 1, 2))
            
            # Z 변위 인가
            lg = _InterpLoadGroup(
                node_ids = [master_nid],
                dof      = 2, # Z
                time_arr = time_arr,
                disp_arr = z_vals_rel
            )
            load_groups.append(lg)
            
            max_z = np.max(np.abs(z_vals_rel))
            print(f"    - [RBE3/SPCD] Master Node={master_nid}, Corner={cn_label}, Z-DOF: Max Abs={max_z:.4f} mm.")

        # --- [WHT] 자동 관성 하중 (Inertial Load) 생성 (코너별 시차 충격 반영) ---
        if getattr(args, 'add_inertia', False):
            print(f" [4] 시차 충격 반영 관성 하중(Staggered Inertia) 생성 중...")
            dt_csv = time_arr[1] - time_arr[0]
            # 4개 코너별 가속도 (T, 4) -> 0:C5, 1:C6, 2:C7, 3:C8
            accels_4 = calculate_corner_accelerations(traj_mm, dt_csv)
            
            # 보간을 위한 가속도 매핑 (Bilinear Interpolation 준비)
            # C5(+X+Y), C6(+X-Y), C7(-X-Y), C8(-X+Y)
            # 바운딩 박스 확인
            all_pts = np.array([model.nodes[nid].coords() for nid in model.nodes if nid < 900000])
            x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
            y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
            dx = x_max - x_min if x_max != x_min else 1.0
            dy = y_max - y_min if y_max != y_min else 1.0

            # 1. 솔버를 이용해 노드별 질량 산출
            temp_solver = WHTDynamicSolver(model)
            jm, sorted_nids, nid_to_idx = temp_solver._build_jaxsso_model()
            m_diag = temp_solver._assemble_lumped_mass(jm, jm.ndof, sorted_nids, nid_to_idx)
            
            # 2. 모든 노드에 대해 Bilinear Interpolated F = -m * a(x,y) 추가
            n_added = 0
            for nid in model.nodes:
                if nid >= 900000: continue
                idx = nid_to_idx.get(nid)
                if idx is None: continue
                
                node = model.nodes[nid]
                nx = (node.x - x_min) / dx
                ny = (node.y - y_min) / dy
                
                # Bilinear 가속도 보간
                # a7(0,0), a6(1,0), a5(1,1), a8(0,1)
                a_local = (
                    accels_4[:, 2] * (1-nx)*(1-ny) + # C7
                    accels_4[:, 1] * (nx)*(1-ny)   + # C6
                    accels_4[:, 0] * (nx)*(ny)     + # C5
                    accels_4[:, 3] * (1-nx)*(ny)     # C8
                )
                
                node_mass = m_diag[idx * 6 + 2]
                if node_mass <= 0: continue
                
                f_history = -node_mass * a_local
                
                lg_i = _InterpLoadGroup(
                    node_ids = [nid],
                    dof      = 2,
                    time_arr = time_arr,
                    disp_arr = f_history
                )
                lg_i.load_type = "FORCE"
                load_groups.append(lg_i)
                n_added += 1
            
            avg_max_a = np.mean(np.max(np.abs(accels_4), axis=0))
            print(f"    - [Inertia] {n_added}개 노드에 시차 관성 하중 인가 완료. (Avg Max Accel={avg_max_a:.1f} mm/s^2)")

        # Stability SPC (X, Y 방향 강체 이동 방지)
        stab_nid = bot_groups[0][1][0]
        model.apply_spc([stab_nid], dofs=(0, 1)) # X, Y 구속
        print(f"    - [Stability] Node {stab_nid} constrained in X, Y.")
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
        method      = getattr(args, 'solver_method', 'scipy'),
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
    parser = argparse.ArgumentParser(
        description="Exam 4: Dynamic Analysis with Large Rotation Handling & Inertial Loads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
[실행 예제]
  1. 실측 데이터 기반 해석 (추천: 로컬 벤딩 + 관성 하중 적용)
     python test_jaxSSO/exam4_dynamic.py --pos-data wht_topo/sample_pos.csv --t-start 1.6 --add-inertia
     python test_jaxSSO/exam4_dynamic.py --pos-data wht_topo/sample_pos.csv --t-start 1.6 --use-global-z
     

  2. JAX 가속 솔버 사용 (대용량 모델 추천)
     python test_jaxSSO/exam4_dynamic.py --pos-data wht_topo/sample_pos.csv --t-start 1.6 --add-inertia --solver-method jax

  3. 가속도 노이즈가 심할 경우 (기본값: 관성 하중 미적용)
     python test_jaxSSO/exam4_dynamic.py --pos-data wht_topo/sample_pos.csv --t-start 1.6

[주의 사항]
  - --add-inertia 없이 변위만 인가할 경우, 4개 코너가 강체로 움직이면 응력이 0에 가깝게 나옵니다.
  - 실제 낙하 충격 응력을 보려면 --add-inertia 옵션 사용이 필수적입니다.
        """
    )
    parser.add_argument("--pos-data", type=str, help="CSV position data file path")
    parser.add_argument("--dt", type=float, help="Time step (s) [Default: 1e-4]")
    parser.add_argument("--t-start", type=float, default=0.0, help="Start time (s) to crop CSV data")
    parser.add_argument("--solver-method", type=str, default="scipy", choices=["scipy", "jax"], help="Solver method")
    parser.add_argument("--add-inertia", action="store_true", help="Apply automatic inertial loads (-m*a) derived from CSV acceleration")
    parser.add_argument("--use-global-z", action="store_true", help="Use raw global Z displacements instead of local frame projection")
    parser.add_argument("--no-viz", action="store_true", help="Skip the visualizer GUI after analysis")
    
    # Legacy arguments (ignored in Local Frame mode)
    parser.add_argument("--enforce-axes", type=str, default="Z", help="[Legacy] Enforced axes (now handled by local frame)")
    parser.add_argument("--rigid-xy", action="store_true", help="[Legacy] Apply mean X-Y (now handled by local frame)")
    
    args = parser.parse_args()
    run(args)
