# -*- coding: utf-8 -*-
"""
exam4_dynamic.py — 대회전 대응 동적 해석 및 Diversity-aware ESL 추출/검증
=========================================================================

[해석 파이프라인]
  [1] 메시 생성 (generate_shell_tray)
  [2] 코너 노드 탐색 (find_corner_nodes, 반경 100mm, z ≤ 2mm)
  [3] 하중 그룹 구성
      CSV 모드: 로컬 Z 변위 SPCD (마스터 노드 + RBE3) + 관성 하중 F=-ma (선택)
      합성 모드: Half-sine 4코너 시차 SPCD
  [4] Direct Newmark-β 과도 응답 해석 (ζ=2%)
  [5] 응력·변형률 이력 복원
  [6] ParaView HDF 저장
  [7] ESL 추출 및 정해석 검증 (--esl 지정 시)
      [7-1] SE 이력 계산 (½uᵀKu)
      [7-2] 윈도우별 SE 테이블 출력
      [7-3] Top-N Diversity-aware ESL 요약 테이블
      [7-4] 전체 ESL 케이스 정해석 수행
      [8]   동적 vs 정적 ESL 듀얼 시각화

[실행 예제]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
기본 실행 (CSV + 관성 하중 + ESL 추출, 권장):
  python test_jaxSSO/exam4_dynamic.py --pos-data wht_topo/sample_pos.csv

ESL 추출 생략 (동적 거동만 확인):
  python test_jaxSSO/exam4_dynamic.py --pos-data data.csv --no-verify

관성 하중 제외 (순수 변위 응답만):
  python test_jaxSSO/exam4_dynamic.py --pos-data data.csv --no-inertia

ESL 개수/윈도우 조정:
  python test_jaxSSO/exam4_dynamic.py --pos-data data.csv --n-windows 50 --n-top 20

합성 Half-sine 모드 (CSV 없이 알고리즘 검증용):
  python test_jaxSSO/exam4_dynamic.py --no-verify

특정 시점 이후만 분석 (충격 시작점 지정):
  python test_jaxSSO/exam4_dynamic.py --pos-data data.csv --t-start 1.6 --add-inertia
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

단위계: mm, N, tonne, s
"""

import re
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_dynamic_solver import WHTDynamicSolver
from wht_solver.wht_dynamic_common import DynamicLoadGroup, DampingSpec, DynamicResult
from wht_solver.load_cases import WHTLoadCase
from wht_modeler.wht_dynamic_utils import (
    find_corner_nodes,
    find_nodes_for_corners,
    parse_csv_header,
    calculate_local_z_history,
    calculate_corner_accelerations,
    InterpLoadGroup,
)
from wht_converter.wht_models import WHTMetadata, WHTResultData
from wht_converter.wht_exporters import VTKHDFExporter
from wht_visualizer.wht_visualizer import WHTVisualizer
from test_jaxSSO.mesh_utils import generate_shell_tray

# ─────────────────────────────────────────────────────────────────────────────
# 모듈 상수
# ─────────────────────────────────────────────────────────────────────────────

MAT = dict(E=210000.0, nu=0.3, rho=7.85e-9, t=1.2)

WIDTH, LENGTH, HEIGHT = 1600.0, 1200.0, 30.0
MESH_XY, MESH_Z      = 30.0, 10.0
DRAFT, FLANGE        = 10.0, 15.0

DT            = 2e-4   # s  기본 적분 스텝
T_TOTAL       = 0.06   # s  기본 해석 시간 (Half-sine 모드)
N_SAVE        = 200    # 저장 프레임 수 (Half-sine 모드)
N_SAVE_CSV    = 100    # 저장 프레임 수 (CSV 모드)
CORNER_RADIUS = 100.0  # mm  코너 탐색 반경

# Half-sine SPCD 기본값
U_AMP    = 10.0    # mm  처방 변위 진폭
T_PULSE  = 0.010   # s   half-sine 지속 시간
T_OFFSET = 0.005   # s   코너 간 시차

CORNER_MAP = {0: 'C5', 1: 'C6', 2: 'C7', 3: 'C8'}


# ─────────────────────────────────────────────────────────────────────────────
# 파이프라인 클래스
# ─────────────────────────────────────────────────────────────────────────────

class DynamicAnalysisPipeline:
    """
    CSV 실측 데이터(또는 합성 Half-sine) 기반 동적 해석 파이프라인.

    각 분석 단계를 독립 메서드로 분리하여 중간 결과를 인스턴스 속성으로 보관합니다.
    `run()` 메서드가 전체 단계를 순서대로 조율합니다.

    Attributes
    ----------
    cfg         : argparse.Namespace  — CLI 파라미터
    model       : WHTMeshModel        — FEM 모델
    node_db     : dict                — {nid: (x, y, z)}
    bot_groups  : list                — 코너 그룹 [(center, [nids]), ...]
    csv_df      : pd.DataFrame | None — t_start 필터 적용된 CSV
    time_arr    : np.ndarray  | None  — 0-based 상대 시간 배열 (s)
    load_groups : list                — DynamicLoadGroup / InterpLoadGroup 목록
    solver      : WHTDynamicSolver    — 동적 해석기
    dyn         : DynamicResult       — 과도 응답 결과
    wht_data    : WHTResultData       — ParaView 내보내기용 결과
    out_dir     : Path                — 결과 저장 디렉토리
    esl_cases   : list[WHTLoadCase]   — 추출된 ESL 로드케이스
    static_wht_data : WHTResultData   — ESL 정해석 결과 (시각화용)
    all_static_times: list[float]     — ESL 케이스별 대응 시각 (s)
    """

    def __init__(self, cfg):
        self.cfg = cfg

        # 단계별 출력 (run() 실행 전 None)
        self.model            : Optional[WHTMeshModel]       = None
        self.node_db          : Optional[dict]               = None
        self.bot_groups       : Optional[list]               = None  # [(name, [nids]), ...]
        self.csv_df           : Optional[pd.DataFrame]       = None
        self.csv_header       : dict                         = {}
        self.time_arr         : Optional[np.ndarray]         = None
        self.load_groups      : List                         = []
        self.solver           : Optional[WHTDynamicSolver]   = None
        self.dyn              : Optional[DynamicResult]      = None
        self.wht_data         : Optional[WHTResultData]      = None
        self.out_dir          : Optional[Path]               = None
        self.esl_cases        : Optional[List[WHTLoadCase]]  = None
        self.static_wht_data  : Optional[WHTResultData]      = None
        self.all_static_times : Optional[List[float]]        = None

    # ── 공개 진입점 ──────────────────────────────────────────────────────────

    def run(self) -> None:
        """전체 파이프라인을 순서대로 실행합니다."""
        self._load_csv()
        self._print_header()
        self._build_mesh()
        self._find_corners()
        self._build_load_groups()
        self._run_dynamic()
        self._recover_stress()
        self._export_results()
        if self.cfg.esl:
            self._extract_esl()
        self._visualize()

    # ── 단계 1: CSV 로드 ─────────────────────────────────────────────────────

    def _load_csv(self) -> None:
        """CSV를 읽고 # 헤더 파싱 → t_start 결정 → 필터를 적용합니다. pos_data 미지정 시 스킵."""
        if not self.cfg.pos_data:
            return

        csv_path = Path(self.cfg.pos_data)
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        print(f" [0] CSV 로드: {csv_path}")

        # # 주석 헤더 파싱 (코너 좌표, start_time)
        self.csv_header = parse_csv_header(str(csv_path))

        # t_start: CLI 인자 > CSV 헤더 > 0.0
        t_start_cli = getattr(self.cfg, 't_start', None)
        t_start = (t_start_cli if t_start_cli is not None
                   else self.csv_header.get('start_time') or 0.0)

        df = pd.read_csv(csv_path, comment='#', encoding='utf-8')
        df.columns = [c.replace('Chassis_', '') for c in df.columns]

        if t_start > 0:
            df = df[df['Time'] >= t_start].copy()
            if df.empty:
                raise ValueError(f"CSV에 t >= {t_start}s 데이터가 없습니다.")
            print(f"     [Filter] t >= {t_start}s 적용 (시작점: {df['Time'].iloc[0]:.4f}s)")

        time_raw      = df['Time'].to_numpy(dtype=float)
        self.csv_df   = df
        self.time_arr = time_raw - time_raw[0]
        print(f"     프레임: {len(self.time_arr)}, 총 시간: {self.time_arr[-1]:.4f}s")

    # ── 단계 2: 메시 생성 ────────────────────────────────────────────────────

    def _build_mesh(self) -> None:
        """트레이 셸 메시를 생성하고 재료·물성을 할당합니다."""
        print("\n [1] 메시 생성...")
        node_db, elem_db = generate_shell_tray(
            width=WIDTH, length=LENGTH, height=HEIGHT,
            mesh_size_xy=MESH_XY, mesh_size_z=MESH_Z,
            draft_angle=DRAFT, flange_width=FLANGE,
            origin='center',
        )
        model = WHTMeshModel.from_node_elem_db(node_db, elem_db,
                                               name="DynamicTray", is_solid=False)
        model.add_material(1, E=MAT['E'], nu=MAT['nu'], rho=MAT['rho'])
        model.add_property(1, "PSHELL", t=MAT['t'], mid=1)
        for eid in model.elements:
            model.elements[eid].pid = 1

        self.model   = model
        self.node_db = node_db
        print(f"     nodes={len(node_db)}, elements={len(elem_db)}")

    # ── 단계 3: 코너 노드 탐색 ───────────────────────────────────────────────

    def _find_corners(self) -> None:
        """CSV 헤더 코너 좌표 기준으로 C5~C8 각 3개 노드 탐색 (중복 없음)."""
        print(f" [2] 코너 노드 탐색 (각 3개, 중복 없음)...")
        header_corners = self.csv_header.get('corner_positions', {})
        c5c8 = {k: v for k, v in header_corners.items() if k in ('C5', 'C6', 'C7', 'C8')}
        if c5c8:
            corner_nodes = find_nodes_for_corners(self.node_db, c5c8, n_nodes=3)
            self.bot_groups = [
                (name, corner_nodes[name])
                for name in ['C5', 'C6', 'C7', 'C8'] if name in corner_nodes
            ]
            for name, nids in self.bot_groups:
                cx, cy, cz = c5c8[name]
                print(f"     {name} (ref {cx:+.0f},{cy:+.0f},{cz:+.0f}): {len(nids)}개 노드")
        else:
            # CSV 없거나 헤더에 코너 좌표 없으면 메시 기하 기반 탐색으로 fallback
            raw = find_corner_nodes(self.node_db, WIDTH, LENGTH, CORNER_RADIUS, z_min=0.0, z_max=2.0)
            self.bot_groups = [(CORNER_MAP[i], g[1]) for i, g in enumerate(raw)]
            print(f"     [WARN] CSV 헤더 코너 좌표 없음 → 메시 기하 기반 탐색 사용")

    # ── 단계 4: 하중 그룹 구성 ───────────────────────────────────────────────

    def _build_load_groups(self) -> None:
        """CSV 모드 또는 Half-sine 합성 모드로 하중 그룹을 구성합니다."""
        if self.csv_df is not None:
            self._build_load_groups_csv()
        else:
            self._build_load_groups_halfsine()

    def _build_load_groups_csv(self) -> None:
        """CSV 궤적 기반 SPCD + 관성 하중 그룹을 구성합니다."""
        if self.cfg.use_global_z:
            print(" [3] 글로벌 Z 변위 직접 인가")
            corner_z, traj_mm = self._extract_global_z()
        else:
            print(" [3] 로컬 Z 변위 추출 (Large Rotation 대응)")
            corner_z, traj_mm = calculate_local_z_history(self.csv_df, self.time_arr)

        # 마스터 노드 + RBE3 + SPCD
        for idx, (cname, corner_nids) in enumerate(self.bot_groups):
            pts    = np.array([self.model.nodes[nid].coords() for nid in corner_nids])
            center = np.mean(pts, axis=0)
            mnid   = 900000 + idx
            self.model.add_node(mnid, center[0], center[1], center[2])
            self.model.add_rbe3(mnid, mnid, corner_nids, dofs=(0, 1, 2))
            self.load_groups.append(InterpLoadGroup(
                node_ids=[mnid], dof=2,
                time_arr=self.time_arr, val_arr=corner_z[cname],
                load_type="SPCD",
            ))

        if self.cfg.add_inertia:
            self._add_inertia_loads(traj_mm)

        # 강체 이동 방지: C5 첫 번째 슬레이브 노드의 X, Y 고정
        self.model.apply_spc([self.bot_groups[0][1][0]], dofs=(0, 1))

    def _extract_global_z(self) -> Tuple[dict, np.ndarray]:
        """CSV에서 글로벌 Z 상대 변위와 (T,4,3) 궤적을 추출합니다."""
        n = len(self.time_arr)
        traj_mm = np.zeros((n, 4, 3))
        for i, lbl in enumerate(CORNER_MAP.values()):
            for j, ax in enumerate(['X', 'Y', 'Z']):
                col = (f"{lbl}_{ax}" if f"{lbl}_{ax}" in self.csv_df.columns
                       else f"{lbl}_pos_{ax}")
                if col in self.csv_df.columns:
                    traj_mm[:, i, j] = self.csv_df[col].to_numpy(dtype=float) * 1000.0
        corner_z = {lbl: traj_mm[:, i, 2] - traj_mm[0, i, 2]
                    for i, lbl in enumerate(CORNER_MAP.values())}
        return corner_z, traj_mm

    def _add_inertia_loads(self, traj_mm: np.ndarray) -> None:
        """
        이선형 보간으로 모든 내부 노드에 관성 하중 F = -m·a 를 인가합니다.

        가속도는 4코너 Z 궤적의 2차 미분이며, 노드 위치에 따라
        (C5, C6, C7, C8) 코너값을 이선형 보간합니다.
        """
        print(" [4] 관성 하중 생성 중 (F=-ma, 이선형 보간)...")
        dt_csv = self.time_arr[1] - self.time_arr[0]
        accels = calculate_corner_accelerations(traj_mm, dt_csv)  # (T, 4)

        all_pts = np.array([self.model.nodes[nid].coords()
                            for nid in self.model.nodes if nid < 900000])
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
        dx = max(x_max - x_min, 1.0)
        dy = max(y_max - y_min, 1.0)

        # 집중 질량 조립
        temp = WHTDynamicSolver(self.model)
        jm, s_nids, n2i = temp._build_jaxsso_model()
        m_diag = temp._assemble_lumped_mass(jm, jm.ndof, s_nids, n2i)

        n_inertia = 0
        for nid in self.model.nodes:
            if nid >= 900000:
                continue
            idx = n2i.get(nid)
            if idx is None:
                continue
            node = self.model.nodes[nid]
            nx = (node.x - x_min) / dx
            ny = (node.y - y_min) / dy
            # 이선형 형상함수: C5(+X+Y), C6(+X-Y), C7(-X-Y), C8(-X+Y)
            a_z = (
                accels[:, 0] * (nx)     * (ny)      +  # C5
                accels[:, 1] * (nx)     * (1 - ny)  +  # C6
                accels[:, 2] * (1 - nx) * (1 - ny)  +  # C7
                accels[:, 3] * (1 - nx) * (ny)         # C8
            )
            node_mass = m_diag[idx * 6 + 2]
            if node_mass > 0:
                self.load_groups.append(InterpLoadGroup(
                    [nid], 2, self.time_arr, -node_mass * a_z, load_type="FORCE"
                ))
                n_inertia += 1
        print(f"     -> {n_inertia}개 노드에 관성 하중 인가 완료.")

    def _build_load_groups_halfsine(self) -> None:
        """합성 Half-sine SPCD 하중을 구성합니다 (CSV 미지정 시)."""
        for i, (_, slave_nids) in enumerate(self.bot_groups):  # noqa: _ unused corner name
            self.load_groups.append(DynamicLoadGroup(
                node_ids  = slave_nids,
                dof       = 2,
                magnitude = U_AMP,
                time_func = "half_sine",
                load_type = "SPCD",
                t_pulse   = T_PULSE,
                t_start   = i * T_OFFSET,
                distribute= False,
            ))
        print(" [3] Half-Sine SPCD 하중 구성 완료 (4개 코너)")

    # ── 단계 5: 동해석 ───────────────────────────────────────────────────────

    def _run_dynamic(self) -> None:
        """Newmark-β 직접 적분 과도 응답 해석을 수행합니다."""
        dt_val      = (self.cfg.dt if self.cfg.dt else DT)
        t_total_val = (float(self.time_arr[-1]) if self.time_arr is not None else T_TOTAL)
        n_save_val  = (N_SAVE_CSV if self.csv_df is not None else N_SAVE)

        print(f"\n [4] Direct Newmark-β 동해석 (dt={dt_val:.1e}s, T={t_total_val:.3f}s)...")
        self.solver = WHTDynamicSolver(self.model)
        self.dyn = self.solver.solve_direct_dynamic(
            load_groups = self.load_groups,
            dt          = dt_val,
            T           = t_total_val,
            damping     = DampingSpec(mode="zeta", zeta=0.02),
            n_save      = n_save_val,
            method      = getattr(self.cfg, 'solver_method', 'scipy'),
        )
        print(f"\n     {self.dyn.summary()}")

    # ── 단계 6: 응력 복원 ────────────────────────────────────────────────────

    def _recover_stress(self) -> None:
        """응력·변형률 이력을 복원합니다."""
        print("\n [5] 응력·변형률 복원...")
        self.solver.recover_stress_history(self.dyn)

    # ── 단계 7: 결과 저장 ────────────────────────────────────────────────────

    def _export_results(self) -> None:
        """ParaView VTKHDF 파일로 결과를 저장합니다."""
        meta = WHTMetadata(
            solver_name="WHTDynamicSolver", solver_version="0.1.0",
            analysis_type="transient", coordinate_system="cartesian",
            unit_length="mm", unit_force="N",
        )
        self.wht_data = self.dyn.to_wht_result_data(meta, self.model)

        stamp        = datetime.now().strftime("D%Y%m%d_%H%M%S")
        self.out_dir = Path(__file__).resolve().parent.parent / "results" / stamp
        pv_dir       = self.out_dir / "paraview"
        pv_dir.mkdir(parents=True, exist_ok=True)

        hdf_path = str(pv_dir / "dynamic_result.hdf")
        VTKHDFExporter().export(self.wht_data, hdf_path)
        print(f"\n [6] ParaView HDF 저장: {hdf_path}")

    # ── 단계 8: ESL 추출 및 정해석 검증 ─────────────────────────────────────

    def _extract_esl(self) -> None:
        """
        SE 이력 계산 → 윈도우 테이블 → Top-N ESL 선정 → 정해석 수행을
        순서대로 실행하고 결과를 인스턴스 속성에 저장합니다.
        """
        n_win = self.cfg.n_windows
        n_top = self.cfg.n_top

        se_history = self._compute_se_history()
        self._print_window_table(se_history, n_win)

        print(f"\n [7-3] Top-{n_top} ESL 추출 중 (Diversity-aware)...")
        self.esl_cases = self.solver.extract_esl_advanced(
            self.dyn, n_windows=n_win, n_top=n_top
        )
        self._print_esl_summary(n_top)

        self._run_static_verification(n_top)

    def _compute_se_history(self) -> np.ndarray:
        """전체 저장 프레임에 대해 변형 에너지 SE = ½uᵀKu 를 계산합니다."""
        print("\n [7-1] 변형 에너지 이력 계산 중...")
        jm, s_nids, n2i = self.solver._build_jaxsso_model()
        K   = self.solver._assemble_K_scipy(jm, s_nids, n2i, stabilize=True)
        ndof = jm.ndof
        se  = np.zeros(self.dyn.n_save)
        for i in range(self.dyn.n_save):
            u_f  = self.dyn.u[i].flatten()[:ndof]
            se[i] = 0.5 * np.dot(u_f, K @ u_f)
        return se

    def _print_window_table(self, se_history: np.ndarray, n_win: int) -> None:
        """n_win 구간별 SE_Peak / SE_Sum 테이블을 출력합니다."""
        print(f"\n [7-2] 시간 구간(Window)별 변형 에너지 지표:")
        win_size = len(self.dyn.t_saved) // n_win
        rows = []
        for i in range(n_win):
            idx_s = i * win_size
            idx_e = (i + 1) * win_size if i < n_win - 1 else len(self.dyn.t_saved)
            if idx_s >= len(self.dyn.t_saved):
                break
            t_mid  = (self.dyn.t_saved[idx_s] + self.dyn.t_saved[idx_e - 1]) / 2.0
            se_win = se_history[idx_s:idx_e]
            rows.append({
                "Window":      i + 1,
                "Time_Mid(s)": t_mid,
                "SE_Peak":     float(np.max(se_win)) if len(se_win) > 0 else 0.0,
                "SE_Sum":      float(np.sum(se_win)) if len(se_win) > 0 else 0.0,
            })
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4e}"))

    def _print_esl_summary(self, n_top: int) -> None:
        """선정된 Top-N ESL 로드케이스 요약 테이블을 출력합니다."""
        print(f"\n [7-3] 선정된 Top-{n_top} ESL 로드케이스 (Diversity-aware):")
        rows = []
        for i, lc in enumerate(self.esl_cases):
            m_t  = re.search(r"t(\d+\.\d+)s",        lc.name)
            m_se = re.search(r"SE(\d+\.\d+e[+-]\d+)", lc.name)
            rows.append({
                "Rank":         i + 1,
                "Time(s)":      float(m_t.group(1))  if m_t  else 0.0,
                "StrainEnergy": float(m_se.group(1)) if m_se else 0.0,
                "Name":         lc.name,
            })
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4e}"))

    def _run_static_verification(self, n_top: int) -> None:
        """
        모든 ESL 케이스에 대해 정해석 및 응력 복원을 수행하고
        `self.static_wht_data` 와 `self.all_static_times` 에 결과를 저장합니다.
        """
        from wht_solver.wht_solver import WHTSolver
        from wht_solver.wht_stress_recovery import ElementStressRecovery

        print(f"\n [7-4] {n_top}개 ESL 케이스 정해석 수행 중...")
        static_solver = WHTSolver(self.model)
        sorted_nids   = self.solver.model.sorted_node_ids()

        disps, stresses, times = [], [], []
        n_cells = len(self.model.elements)

        for i, lc in enumerate(self.esl_cases):
            res   = static_solver.solve_static(lc)
            disps.append(res.displacement)

            rd_q = ElementStressRecovery.recover_quad4(self.model, res.displacement, sorted_nids)
            rd_t = ElementStressRecovery.recover_tria3(self.model, res.displacement, sorted_nids)
            shell_stress   = rd_q["Stress"] + rd_t["Stress"]
            padded = np.zeros((n_cells, 6))
            padded[:shell_stress.shape[0], :] = shell_stress
            stresses.append(padded)

            m_t = re.search(r"t(\d+\.\d+)s", lc.name)
            times.append(float(m_t.group(1)) if m_t else float(i))

        static_meta = WHTMetadata(
            solver_name="WHTStaticSolver", solver_version="0.1.0",
            analysis_type="static", coordinate_system="cartesian",
            unit_length="mm", unit_force="N",
        )
        rd = self.model.to_wht_result_data(static_meta)
        # time_values에 Rank 번호를 할당하여 시각화 슬라이더와 연동
        rd.time_values = np.arange(1, len(times) + 1, dtype=float)
        rd.point_data["Displacement"] = np.stack(disps,    axis=0)[:, :, :3]
        rd.cell_data["Stress"]        = np.stack(stresses, axis=0)

        self.static_wht_data  = rd
        self.all_static_times = times

    # ── 단계 9: 시각화 ───────────────────────────────────────────────────────

    def _visualize(self) -> None:
        """ESL 검증 여부에 따라 듀얼 또는 단일 시각화를 실행합니다."""
        if self.cfg.no_viz:
            return
        if self.cfg.esl and self.esl_cases:
            self._visualize_dual()
        else:
            self._visualize_single()

    def _visualize_single(self) -> None:
        """동적 응답 단독 시각화."""
        print(" [7] WHTVisualizer 실행...")
        title = "Exam 4: CSV Pos Dynamic" if self.cfg.pos_data else "Exam 4: Half-Sine Dynamic"
        viz   = WHTVisualizer(title=title)
        viz.show_result(self.wht_data, group_name="DynamicTray")
        viz.plotter.view_isometric()
        viz.plotter.reset_camera()
        if hasattr(viz.plotter, 'app'):
            viz.plotter.app.exec_()

    def _visualize_dual(self) -> None:
        """동적 응답 vs ESL 정해석 결과 듀얼 시각화."""
        n_top       = len(self.esl_cases)
        t_esl_top   = self.all_static_times[0]
        idx_esl_top = int(np.argmin(np.abs(self.dyn.t_saved - t_esl_top)))

        print(f" [8] 듀얼 시각화 (Dynamic vs {n_top} Static ESL Frames)...")

        viz_dyn = WHTVisualizer(
            title=f"Dynamic Analysis (Focus: Rank 1 Peak t={t_esl_top:.4f}s)"
        )
        viz_dyn.show_result(self.wht_data, group_name="Dynamic_Tray")
        viz_dyn.slider_time.setValue(idx_esl_top)
        viz_dyn.plotter.view_isometric()

        viz_static = WHTVisualizer(
            title=f"Static ESL Sequence (Frames 1-{n_top}, Diversity/Energy Sorted)"
        )
        viz_static.show_result(self.static_wht_data, group_name="Static_ESL_Tray")
        viz_static.plotter.view_isometric()

        if hasattr(viz_dyn.plotter, 'app'):
            viz_dyn.plotter.app.exec_()

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    def _print_header(self) -> None:
        mode = "CSV Position Data Analysis" if self.csv_df is not None \
               else "Synthetic Half-Sine Excitation"
        print("\n" + "=" * 65)
        print("  Exam 4: Dynamic Analysis with ESL Verification")
        print(f"  [Mode] {mode}")
        print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exam 4: 대회전 대응 동적 해석 및 Diversity-aware ESL 추출/검증",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 예제
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[권장] 실측 CSV + 관성 하중 + ESL 추출 (낙하 충격 표준 분석)
  python test_jaxSSO/exam4_dynamic.py --pos-data test_jaxSSO/structural_dynamics.csv
  -> t_start: CSV 헤더 start_time 자동 적용
  -> 관성 하중 활성 (기본), ESL 10개 추출 및 정해석 검증 포함

[권장] 충격 시작점 명시 + ESL 개수 확대 (고정밀 분석)
  python test_jaxSSO/exam4_dynamic.py --pos-data test_jaxSSO/structural_dynamics.csv --t-start 1.6 --n-windows 50 --n-top 15
  -> t_start 1.6s 이전의 준정적 구간을 제외하고 충격 구간만 분석
  -> 50개 창으로 더 촘촘히 피크 후보를 탐색, 상위 15개 ESL 선정

[빠른 확인] ESL 검증 생략 (동적 거동 파형만 빠르게 확인)
  python test_jaxSSO/exam4_dynamic.py \\
    --pos-data test_jaxSSO/structural_dynamics.csv \\
    --no-verify
  -> 정해석 및 듀얼 시각화 없이 동해석 + ParaView HDF 저장만 수행

[비교 분석] 로컬 프레임 vs 글로벌 Z 변위 비교
  python test_jaxSSO/exam4_dynamic.py \\
    --pos-data test_jaxSSO/structural_dynamics.csv \\
    --use-global-z --no-verify
  -> 강체 회전 제거 없이 글로벌 Z 궤적 직접 인가 (차이 확인용)

[순수 변위 응답] 관성 하중 제외 (SPCD 경계 조건 응답만)
  python test_jaxSSO/exam4_dynamic.py \\
    --pos-data test_jaxSSO/structural_dynamics.csv \\
    --no-inertia
  -> F=-ma 관성 분포 하중 없이 코너 변위 BC만으로 응답 계산
  -> 관성 하중 기여도 파악 목적 (결과를 --add-inertia 케이스와 비교)

[알고리즘 검증] CSV 없이 합성 Half-sine 모드
  python test_jaxSSO/exam4_dynamic.py --no-verify
  -> 4코너에 시차(5ms) Half-sine 변위 SPCD 적용
  -> 실측 데이터 없이 ESL 파이프라인 자체 동작 검증

[서버/헤드리스] GUI 없이 결과 파일만 저장
  python test_jaxSSO/exam4_dynamic.py \\
    --pos-data test_jaxSSO/structural_dynamics.csv \\
    --no-verify --no-viz
  -> 시각화 창 없이 ParaView HDF만 results/ 에 저장

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
옵션 상세 설명
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--pos-data      실측 4코너(C5~C8) 위치 CSV 경로.
                미지정 시 합성 Half-sine SPCD 모드로 동작.
                CSV 형식: # 주석 헤더(코너 좌표·start_time) + Frame,Time,C1_X...C8_Z

--t-start       CSV 분석 시작 시점(s).
                미지정 시 CSV 헤더의 "# start_time, X.X" 값을 자동 적용.
                헤더에도 없으면 0.0 (전체 구간 사용).

--add-inertia   CSV 궤적 2차 미분(가속도) 기반 F=-ma 관성 하중 계산.
  (기본: ON)    이선형 보간으로 모든 내부 노드에 분포 인가.
                낙하·충격처럼 가속도 기여가 큰 하중 케이스에 반드시 사용.

--no-inertia    관성 하중 제외 (순수 변위 SPCD 응답만 분석).

--esl           동해석 완료 후 ESL 추출 및 정해석 비교 검증 수행.
  (기본: ON)    SE 이력 → 윈도우 테이블 → Top-N 요약 → 정해석 → 듀얼 시각화

--no-verify     ESL 추출 및 검증 생략 (동적 거동·파형 확인만).

--n-windows     SE 이력을 분할하는 시간 창 수. 클수록 피크 후보를 촘촘히 탐색.
                기본: 30. 권장: 짧은 충격 구간 → 20, 긴 진동 구간 → 50

--n-top         최종 선정할 Diversity-aware ESL 개수.
                기본: 10. 최적화 정밀도↑ 원할 때 15~20 사용 (계산 시간 증가).

--use-global-z  로컬 프레임 투영 대신 글로벌 Z 궤적 직접 사용.
                강체 회전이 거의 없는 경우 또는 비교 분석 목적으로만 사용.

--solver-method scipy(기본) 또는 jax.
                scipy: 소규모·범용. jax: 대규모 모델 고속 GPU 처리.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    )

    # CSV 및 해석 설정
    parser.add_argument("--pos-data",      type=str,   default="wht_topo/sample_pos.csv",
                        help="실측 위치 데이터 CSV 경로 (기본: wht_topo/sample_pos.csv)")
    parser.add_argument("--dt",            type=float, default=None,
                        help="시간 적분 간격 s (기본: 2e-4)")
    parser.add_argument("--t-start",       type=float, default=None,
                        help="CSV 분석 시작 시점 s (기본: CSV 헤더 start_time 자동 적용)")
    parser.add_argument("--solver-method", type=str,   default="scipy",
                        choices=["scipy", "jax"],
                        help="동적 해석 솔버 (기본: scipy)")

    # 관성 하중
    parser.add_argument("--add-inertia",   action="store_true", default=True,
                        help="관성 하중(-ma) 인가 (기본: 활성)")
    parser.add_argument("--no-inertia",    action="store_false", dest="add_inertia",
                        help="관성 하중 제외")

    # 변위 모드
    parser.add_argument("--use-global-z",  action="store_true",
                        help="글로벌 Z 궤적 직접 사용 (기본: 로컬 프레임 투영)")

    # 시각화
    parser.add_argument("--no-viz",        action="store_true",
                        help="시각화 GUI 생략")

    # ESL 추출 및 검증
    parser.add_argument("--esl",           action="store_true", default=True,
                        help="ESL 추출 및 정해석 검증 (기본: 활성)")
    parser.add_argument("--no-verify",     action="store_false", dest="esl",
                        help="ESL 추출 및 검증 생략")
    parser.add_argument("--n-windows",     type=int, default=30,
                        help="피크 후보 탐색 시간 분할 수 (기본: 30)")
    parser.add_argument("--n-top",         type=int, default=10,
                        help="최종 선정 ESL 개수 (기본: 10)")

    DynamicAnalysisPipeline(parser.parse_args()).run()
