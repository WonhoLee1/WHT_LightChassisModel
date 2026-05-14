# -*- coding: utf-8 -*-
"""
run_topo.py — WHT 산업용 섀시 비드 최적화(Topography) 통합 도구
================================================================

[실행 모드]
  모드 A | 기본 정적 최적화 (--pos-data / --dynamic-opts 미지정)
    굽힘·비틀림·리프팅 등 표준 정적 하중 케이스 기반 비드 패턴 최적화.
    실행: python wht_topo/run_topo.py --iters 20 --sym-x

  모드 B | CSV 단독 동적 응답 해석 (--pos-data 지정, 최적화 생략)
    실측 4코너 위치 데이터로 과도 응답 해석을 수행하고 결과를 ParaView HDF로 저장.
    하중: 각 코너 마스터 노드에 로컬 Z 방향 SPCD 적용 (RBE3 연결).
    실행: python wht_topo/run_topo.py --pos-data wht_topo/structural_dynamics.csv

  모드 C | 동적 충격 통합 최적화 (--dynamic-opts 지정)
    Diversity-aware ESL 알고리즘으로 피크 스냅샷 n_top개를 추출하고
    정적 하중 케이스와 합산하여 최적화에 반영.
    실행: python wht_topo/run_topo.py --dynamic-opts "wht_topo/structural_dynamics.csv,1.6" --add-inertia --n-top 10

  모드 D | 고신뢰성 산업용 완전 제약 설계
    동적 하중 + 관성 하중 + 좌우 대칭 + 비드 연결 + 이산화(0/h_max)
    실행: python wht_topo/run_topo.py --dynamic-opts "data.csv,1.6" --add-inertia --sym-x --bead-connect --height-steps 2

[알고리즘 개요: Diversity-aware ESL]
  1. CSV에서 t_start 이후 구간 추출.
  2. 강체 회전 제거 → 로컬 Z 방향 순수 벤딩 변위 분리 (calculate_local_z_history).
  3. 4코너 마스터 노드(#900000~900003) + RBE3 → Z-SPCD 하중 그룹 구성.
  4. (옵션) 이선형 보간 + 집중 질량 → 관성 하중(F=-ma) 전 노드 인가.
  5. Newmark-β 직접 적분 과도 응답 해석 (ζ=2%).
  6. 변형 에너지 이력(SE = ½uᵀKu) 계산 → n_windows 구간 분할 → 피크 후보 추출.
  7. Greedy Max-Min Cosine Similarity 다양성 선별 → Top-n_top 스냅샷 선정.
  8. WHTLoadCase로 변환 → 정적 최적화 하중 케이스 풀에 추가.

단위계: mm, N, tonne, s
"""

import argparse
import re
import sys
import io
import multiprocessing
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_topo.loads import StochasticLoadManager
from wht_topo.solver import WHTopographySolver
from wht_visualizer.wht_visualizer import WHTVisualizer
from wht_converter.wht_models import WHTMetadata, WHTResultData
from wht_converter.wht_exporters import VTKHDFExporter
from test_jaxSSO.mesh_utils import generate_shell_tray
from wht_solver.load_cases import WHTLoadCase
from wht_solver.wht_dynamic_solver import WHTDynamicSolver
from wht_solver.wht_dynamic_common import DampingSpec, DynamicResult
from wht_modeler.wht_dynamic_utils import (
    find_corner_nodes,
    find_nodes_for_corners,
    parse_csv_header,
    calculate_local_z_history,
    calculate_corner_accelerations,
    InterpLoadGroup,
)

# ─────────────────────────────────────────────────────────────────────────────
# 모듈 상수
# ─────────────────────────────────────────────────────────────────────────────

TRAY_W, TRAY_L, TRAY_H = 1800.0, 1200.0, 35.0
MESH_XY, MESH_Z        = 40.0, 10.0
DRAFT_ANGLE            = 25.0
HOOK_SEQUENCE          = [(0.0, 12.0), (-5.0, 0.0), (0.0, -10.0), (-10.0, 0.0)]
MAT                    = dict(E=210000.0, nu=0.3, rho=7.85e-9, t=1.2)
CORNER_NAMES           = ['C5', 'C6', 'C7', 'C8']
CORNER_MAP             = dict(enumerate(CORNER_NAMES))   # {0:'C5', 1:'C6', ...}
ZETA                   = 0.02   # Rayleigh 감쇠비


# ─────────────────────────────────────────────────────────────────────────────
# 공유 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────

def _build_tray() -> Tuple[WHTMeshModel, dict]:
    """
    공통 트레이 셸 메시를 생성하고 WHTMeshModel을 반환합니다.

    Returns
    -------
    model   : WHTMeshModel
    node_db : dict  {nid: (x, y, z)}
    """
    node_db, elem_db = generate_shell_tray(
        width=TRAY_W, length=TRAY_L, height=TRAY_H,
        mesh_size_xy=MESH_XY, mesh_size_z=MESH_Z,
        draft_angle=DRAFT_ANGLE, flange_segments=HOOK_SEQUENCE,
        flanges=(False, True, True, True), mesh_type='quad4',
    )
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db,
                                           name="DynamicTray", is_solid=False)
    model.add_material(1, E=MAT['E'], nu=MAT['nu'], rho=MAT['rho'])
    model.add_property(1, "PSHELL", t=MAT['t'], mid=1)
    for eid in model.elements:
        model.elements[eid].pid = 1
    return model, node_db


def _load_csv(
    path_str: str,
    t_start: Optional[float] = None,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """
    CSV를 로드하고 'Chassis_' 접두사를 제거한 뒤 t_start 필터를 적용합니다.

    # 주석 헤더에서 코너 기준 좌표와 start_time을 파싱합니다.
    t_start 우선순위: 인자 > CSV 헤더 > 0.0

    Returns
    -------
    df       : pd.DataFrame  필터 후 DataFrame
    time_arr : np.ndarray    0-based 상대 시간 배열 (s)
    header   : dict          {'corner_positions': {...}, 'start_time': float|None}
    """
    csv_path = Path(path_str)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path

    header = parse_csv_header(str(csv_path))

    if t_start is None:
        t_start = header.get('start_time') or 0.0

    df = pd.read_csv(csv_path, comment='#', encoding='utf-8')
    df.columns = [c.replace('Chassis_', '') for c in df.columns]

    if t_start > 0:
        df = df[df['Time'] >= t_start].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"CSV에 t >= {t_start}s 데이터가 없습니다.")

    time_raw = df['Time'].to_numpy(dtype=float)
    return df, time_raw - time_raw[0], header


# ─────────────────────────────────────────────────────────────────────────────
# ESL 추출기  (TopographyPipeline 내부에서 사용)
# ─────────────────────────────────────────────────────────────────────────────

class ESLExtractor:
    """
    CSV 실측 데이터 → 동해석 → Diversity-aware ESL 스냅샷 추출.

    TopographyPipeline이 --dynamic-opts 지정 시 생성하여 사용합니다.
    모델에 마스터 노드/RBE3를 임시로 추가하고 추출 완료 후 원상 복구합니다.

    Attributes
    ----------
    model       : WHTMeshModel     대상 모델 (직접 수정 → 복구)
    node_db     : dict             {nid: (x, y, z)}
    csv_path    : str
    t_start     : float            충격 시작 시간 (s)
    n_windows   : int              SE 이력 분할 수
    n_top       : int              최종 ESL 개수
    add_inertia : bool
    use_global_z: bool
    """

    def __init__(
        self,
        model: WHTMeshModel,
        node_db: dict,
        csv_path: str,
        t_start: Optional[float] = None,
        n_windows: int = 30,
        n_top: int = 10,
        add_inertia: bool = True,
        use_global_z: bool = False,
    ):
        self.model        = model
        self.node_db      = node_db
        self.csv_path     = csv_path
        self.t_start      = t_start   # None → CSV 헤더 start_time 자동 적용
        self.n_windows    = n_windows
        self.n_top        = n_top
        self.add_inertia  = add_inertia
        self.use_global_z = use_global_z

        # 단계별 출력
        self._df          : Optional[pd.DataFrame]     = None
        self._time_arr    : Optional[np.ndarray]       = None
        self._csv_header  : dict                       = {}
        self._bot_groups  : Optional[list]             = None  # [(name, [nids]), ...]
        self._load_groups : List                       = []
        self._dyn_solver  : Optional[WHTDynamicSolver] = None
        self._dyn_res     : Optional[DynamicResult]    = None
        self._esl_cases   : Optional[list]             = None

    # ── 공개 진입점 ──────────────────────────────────────────────────────────

    def extract(self) -> List[Tuple[WHTLoadCase, float]]:
        """ESL 추출 전체 파이프라인을 실행하고 (WHTLoadCase, weight) 리스트를 반환합니다."""
        print(f"\n [ESL] Diversity-aware Snapshot Extraction 시작")
        print(f"       CSV={self.csv_path}")
        print(f"       n_windows={self.n_windows}, n_top={self.n_top}")
        print(f"       관성 하중={'포함' if self.add_inertia else '제외'}, "
              f"변위 모드={'글로벌 Z' if self.use_global_z else '로컬 프레임'}")

        self._load_csv()
        self._find_corners()
        self._build_spcd_groups()
        if self.add_inertia:
            self._add_inertia_loads()
        self._run_dynamic()
        self._extract_esl_cases()
        self._print_se_tables()
        self._cleanup_master_nodes()
        return self._build_snapshots()

    # ── 단계별 private 메서드 ────────────────────────────────────────────────

    def _load_csv(self) -> None:
        self._df, self._time_arr, self._csv_header = _load_csv(self.csv_path, self.t_start)
        # t_start가 None이었을 경우 헤더에서 결정된 값으로 동기화
        if self.t_start is None:
            self.t_start = self._csv_header.get('start_time') or 0.0
        dt = self._time_arr[1] - self._time_arr[0] if len(self._time_arr) > 1 else 1e-4
        print(f"     t_start={self.t_start}s, 프레임={len(self._time_arr)}, "
              f"T={self._time_arr[-1]:.3f}s, dt={dt:.2e}s")

    def _find_corners(self) -> None:
        """CSV 헤더 코너 좌표 기준으로 C5~C8 각 3개 노드 탐색 (중복 없음)."""
        header_corners = self._csv_header.get('corner_positions', {})
        c5c8 = {k: v for k, v in header_corners.items() if k in ('C5', 'C6', 'C7', 'C8')}
        if c5c8:
            corner_nodes = find_nodes_for_corners(self.node_db, c5c8, n_nodes=3)
            self._bot_groups = [
                (name, corner_nodes[name])
                for name in ['C5', 'C6', 'C7', 'C8'] if name in corner_nodes
            ]
            for name, nids in self._bot_groups:
                cx, cy, cz = c5c8[name]
                print(f"     {name} (ref {cx:+.0f},{cy:+.0f},{cz:+.0f}): {len(nids)}개 노드")
        else:
            # CSV 헤더에 코너 좌표 없으면 메시 기하 기반 탐색으로 fallback
            raw = find_corner_nodes(self.node_db, TRAY_W, TRAY_L, 150.0, z_min=0.0, z_max=2.0)
            self._bot_groups = [(CORNER_NAMES[i], g[1]) for i, g in enumerate(raw)]
            print("     [WARN] CSV 헤더에 코너 좌표 없음 → 메시 기하 기반 탐색 사용")

    def _build_spcd_groups(self) -> None:
        """각 코너에 마스터 노드(#900000+) + RBE3 + Z-SPCD 하중을 구성합니다."""
        if self.use_global_z:
            corner_z = self._extract_global_z()
        else:
            corner_z, _ = calculate_local_z_history(self._df, self._time_arr)

        for idx, (cname, cnids) in enumerate(self._bot_groups):
            center = np.mean([self.node_db[n] for n in cnids], axis=0)
            mnid   = 900000 + idx
            self.model.add_node(mnid, center[0], center[1], center[2])
            self.model.add_rbe3(mnid, mnid, cnids, dofs=(0, 1, 2))
            self._load_groups.append(InterpLoadGroup(
                [mnid], 2, self._time_arr, corner_z[cname], "SPCD"
            ))

    def _extract_global_z(self) -> dict:
        """CSV에서 글로벌 Z 상대 변위를 추출합니다."""
        corner_z = {}
        for lbl in CORNER_NAMES:
            col = f"{lbl}_Z" if f"{lbl}_Z" in self._df.columns else f"{lbl}_pos_Z"
            z = self._df[col].to_numpy(dtype=float) * 1000.0
            corner_z[lbl] = z - z[0]
        return corner_z

    def _add_inertia_loads(self) -> None:
        """이선형 보간으로 모든 내부 노드에 관성 하중 F = -m·a 를 인가합니다."""
        _, traj_mm = calculate_local_z_history(self._df, self._time_arr)
        dt         = self._time_arr[1] - self._time_arr[0] if len(self._time_arr) > 1 else 1e-4
        accels     = calculate_corner_accelerations(traj_mm, dt)  # (T, 4)

        all_xyz = np.array(list(self.node_db.values()))
        xmin, xmax = all_xyz[:, 0].min(), all_xyz[:, 0].max()
        ymin, ymax = all_xyz[:, 1].min(), all_xyz[:, 1].max()
        dx = max(xmax - xmin, 1.0)
        dy = max(ymax - ymin, 1.0)

        temp = WHTDynamicSolver(self.model)
        jm, s_nids, n2i = temp._build_jaxsso_model()
        m_diag = temp._assemble_lumped_mass(jm, jm.ndof, s_nids, n2i)

        n_inertia = 0
        for nid in self.node_db:
            ix = n2i.get(nid)
            if ix is None or nid >= 900000:
                continue
            x, y, _ = self.node_db[nid]
            nx, ny  = (x - xmin) / dx, (y - ymin) / dy
            a_z = (
                accels[:, 0] * (nx)     * (ny)      +  # C5: +X+Y
                accels[:, 1] * (nx)     * (1 - ny)  +  # C6: +X-Y
                accels[:, 2] * (1 - nx) * (1 - ny)  +  # C7: -X-Y
                accels[:, 3] * (1 - nx) * (ny)         # C8: -X+Y
            )
            node_mass = m_diag[ix * 6 + 2]
            if node_mass > 1e-12:
                self._load_groups.append(
                    InterpLoadGroup([nid], 2, self._time_arr, -node_mass * a_z, "FORCE")
                )
                n_inertia += 1
        print(f"     -> 관성 하중: {n_inertia}개 노드 인가 완료.")

    def _run_dynamic(self) -> None:
        dt      = self._time_arr[1] - self._time_arr[0] if len(self._time_arr) > 1 else 1e-4
        T_total = float(self._time_arr[-1])
        print(f"     -> 과도 응답 해석 (dt={dt:.2e}s, T={T_total:.3f}s)...")
        self._dyn_solver = WHTDynamicSolver(self.model)
        self._dyn_res    = self._dyn_solver.solve_direct_dynamic(
            self._load_groups, dt=dt, T=T_total, n_save=100,
            damping=DampingSpec(mode="zeta", zeta=ZETA),
        )

    def _extract_esl_cases(self) -> None:
        self._esl_cases = self._dyn_solver.extract_esl_advanced(
            self._dyn_res, n_windows=self.n_windows, n_top=self.n_top
        )

    def _print_se_tables(self) -> None:
        """SE 이력 계산 → 윈도우 테이블 → Top-N ESL 요약 테이블 출력."""
        # SE 이력
        print(f"\n [ESL-1] 변형 에너지 이력 계산 중...")
        jm, s_nids, n2i = self._dyn_solver._build_jaxsso_model()
        K    = self._dyn_solver._assemble_K_scipy(jm, s_nids, n2i, stabilize=True)
        ndof = jm.ndof
        se   = np.zeros(self._dyn_res.n_save)
        for i in range(self._dyn_res.n_save):
            u_f  = self._dyn_res.u[i].flatten()[:ndof]
            se[i] = 0.5 * np.dot(u_f, K @ u_f)

        # 윈도우별 테이블
        print(f"\n [ESL-2] 시간 구간(Window)별 변형 에너지 지표:")
        win_size = len(self._dyn_res.t_saved) // self.n_windows
        rows = []
        for i in range(self.n_windows):
            idx_s = i * win_size
            idx_e = (i + 1) * win_size if i < self.n_windows - 1 else len(self._dyn_res.t_saved)
            if idx_s >= len(self._dyn_res.t_saved):
                break
            t_mid  = (self._dyn_res.t_saved[idx_s] + self._dyn_res.t_saved[idx_e - 1]) / 2.0
            se_win = se[idx_s:idx_e]
            rows.append({
                "Window":      i + 1,
                "Time_Mid(s)": t_mid,
                "SE_Peak":     float(np.max(se_win)) if len(se_win) > 0 else 0.0,
                "SE_Sum":      float(np.sum(se_win)) if len(se_win) > 0 else 0.0,
            })
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4e}"))

        # Top-N 요약
        print(f"\n [ESL-3] 선정된 Top-{self.n_top} ESL 로드케이스 (Diversity-aware):")
        rows = []
        for i, lc in enumerate(self._esl_cases):
            m_t  = re.search(r"t(\d+\.\d+)s",        lc.name)
            m_se = re.search(r"SE(\d+\.\d+e[+-]\d+)", lc.name)
            rows.append({
                "Rank":         i + 1,
                "Time(s)":      float(m_t.group(1))  if m_t  else 0.0,
                "StrainEnergy": float(m_se.group(1)) if m_se else 0.0,
                "Name":         lc.name,
            })
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4e}"))

    def _cleanup_master_nodes(self) -> None:
        """동해석용으로 임시 추가한 마스터 노드와 RBE3를 제거합니다."""
        for mnid in range(900000, 900004):
            if mnid in self.model.nodes:
                del self.model.nodes[mnid]
        self.model.rbe3s = {k: v for k, v in self.model.rbe3s.items() if k < 900000}

    def _build_snapshots(self) -> List[Tuple[WHTLoadCase, float]]:
        """ESL 케이스에 안정화 BC를 추가하고 (lc, weight=1.0) 리스트를 반환합니다."""
        stab_nid  = self._bot_groups[0][1][0]
        snapshots = []
        for lc in self._esl_cases:
            lc.add_bc([stab_nid], dofs=(0, 1))
            snapshots.append((lc, 1.0))
        print(f"     -> {len(snapshots)}개 ESL 스냅샷 추출 완료.")
        return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# 모드 B: CSV 단독 동적 응답 해석 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

class PosDynamicPipeline:
    """
    CSV 위치 데이터 기반 단독 동적 응답 해석 파이프라인 (모드 B).

    최적화 없이 과도 응답을 해석하고 ParaView HDF로 저장합니다.

    Attributes
    ----------
    cfg      : argparse.Namespace
    model    : WHTMeshModel
    node_db  : dict
    dyn_res  : DynamicResult
    wht_data : WHTResultData
    out_dir  : Path
    """

    def __init__(self, cfg):
        self.cfg      = cfg
        self.model    : Optional[WHTMeshModel]     = None
        self.node_db  : Optional[dict]             = None
        self.dyn_res  : Optional[DynamicResult]    = None
        self.wht_data : Optional[WHTResultData]    = None
        self.out_dir  : Optional[Path]             = None

    def run(self) -> None:
        """파이프라인 전체 실행."""
        print(f"\n{'='*65}")
        print("  run_topo [모드 B]: CSV 단독 동적 응답 해석")
        print(f"{'='*65}\n")
        self._build_mesh()
        self._find_corners_and_load()
        self._run_dynamic()
        self._export()
        self._visualize()

    # ── 단계별 메서드 ────────────────────────────────────────────────────────

    def _build_mesh(self) -> None:
        print(" [1] 메시 생성...")
        self.model, self.node_db = _build_tray()
        print(f"     노드={len(self.node_db)}, 요소={len(self.model.elements)}")

    def _find_corners_and_load(self) -> None:
        """CSV 읽기 → 코너 탐색 → 마스터 노드 + RBE3 + SPCD 구성."""
        # CSV 로드 (t_start: CLI 인자 우선, 없으면 헤더, 없으면 0.0)
        t_start_arg = getattr(self.cfg, 't_start', None)
        print(f"\n [2] CSV 로드: {self.cfg.pos_data}")
        df, time_arr, header = _load_csv(self.cfg.pos_data, t_start_arg)
        t_start_used = t_start_arg if t_start_arg is not None else (header.get('start_time') or 0.0)
        dt_val  = float(time_arr[1] - time_arr[0]) if len(time_arr) > 1 else self.cfg.dt
        T_total = float(time_arr[-1])
        print(f"     t_start={t_start_used}s, 프레임={len(time_arr)}, "
              f"총 시간={T_total:.4f}s, dt={dt_val:.2e}s")

        # 코너 탐색: CSV 헤더 기준 좌표 → 최근접 3개 노드
        print(f"\n [3] 코너 노드 탐색 (각 3개, 중복 없음)...")
        header_corners = header.get('corner_positions', {})
        c5c8 = {k: v for k, v in header_corners.items() if k in ('C5', 'C6', 'C7', 'C8')}
        if c5c8:
            cmap = find_nodes_for_corners(self.node_db, c5c8, n_nodes=3)
            bot_groups = [(name, cmap[name]) for name in ['C5', 'C6', 'C7', 'C8'] if name in cmap]
            for name, bg in bot_groups:
                cx, cy, cz = c5c8[name]
                print(f"     {name} (ref {cx:+.0f},{cy:+.0f},{cz:+.0f}): {len(bg)}개 노드")
        else:
            raw = find_corner_nodes(self.node_db, TRAY_W, TRAY_L, self.cfg.corner_r, z_min=0.0, z_max=2.0)
            bot_groups = [(CORNER_NAMES[i], g[1]) for i, g in enumerate(raw)]
            print("     [WARN] CSV 헤더에 코너 좌표 없음 → 메시 기하 기반 탐색 사용")

        # 코너 변위 추출
        print(f"\n [4] 하중 그룹 구성 ({'글로벌 Z' if self.cfg.use_global_z else '로컬 프레임'})...")
        if self.cfg.use_global_z:
            corner_z = {}
            for lbl in CORNER_NAMES:
                col = f"{lbl}_Z" if f"{lbl}_Z" in df.columns else f"{lbl}_pos_Z"
                z = df[col].to_numpy(dtype=float) * 1000.0
                corner_z[lbl] = z - z[0]
        else:
            corner_z, _ = calculate_local_z_history(df, time_arr)

        # 마스터 노드 + RBE3 + SPCD
        load_groups = []
        for idx, (cname, corner_nids) in enumerate(bot_groups):
            center = np.mean([self.node_db[n] for n in corner_nids], axis=0)
            mnid   = 900000 + idx
            self.model.add_node(mnid, center[0], center[1], center[2])
            self.model.add_rbe3(mnid, mnid, corner_nids, dofs=(0, 1, 2))
            load_groups.append(InterpLoadGroup(
                [mnid], 2, time_arr, corner_z[cname], "SPCD"
            ))
        # 강체 이동 방지
        self.model.apply_spc([bot_groups[0][1][0]], dofs=(0, 1))

        # 인스턴스에 저장 (다음 단계에서 사용)
        self._load_groups = load_groups
        self._dt_val      = dt_val
        self._T_total     = T_total

    def _run_dynamic(self) -> None:
        print(f"\n [5] 과도 응답 해석 (Newmark-β, dt={self._dt_val:.2e}s, T={self._T_total:.4f}s)...")
        solver = WHTDynamicSolver(self.model)
        self.dyn_res = solver.solve_direct_dynamic(
            self._load_groups, dt=self._dt_val, T=self._T_total,
            damping=DampingSpec(mode="zeta", zeta=self.cfg.zeta),
            n_save=100,
        )
        print(f"\n     {self.dyn_res.summary()}")

    def _export(self) -> None:
        meta = WHTMetadata(
            solver_name="WHTDynamicSolver", solver_version="1.0.0",
            analysis_type="transient", coordinate_system="cartesian",
            unit_length="mm", unit_force="N", unit_mass="tonne",
        )
        self.wht_data = self.dyn_res.to_wht_result_data(meta, self.model)

        stamp       = datetime.now().strftime("D%Y%m%d_%H%M%S")
        self.out_dir = Path(__file__).resolve().parent.parent / "results" / stamp / "paraview"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        hdf_path = str(self.out_dir / "dynamic_result.hdf")
        VTKHDFExporter().export(self.wht_data, hdf_path)
        print(f"\n [6] ParaView HDF 저장: {hdf_path}")

    def _visualize(self) -> None:
        if self.cfg.no_viz:
            return
        print(" [7] WHTVisualizer 실행...")
        viz = WHTVisualizer(title="CSV Position Data - Dynamic Response")
        viz.show_result(self.wht_data, group_name="DynamicTray")
        viz.plotter.view_isometric()
        viz.plotter.reset_camera()
        if hasattr(viz.plotter, 'app'):
            viz.plotter.app.exec_()


# ─────────────────────────────────────────────────────────────────────────────
# 모드 A/C/D: 토포그래피 최적화 파이프라인
# ─────────────────────────────────────────────────────────────────────────────

class TopographyPipeline:
    """
    정적/동적 통합 토포그래피 최적화 파이프라인 (모드 A/C/D).

    --dynamic-opts 미지정 시 표준 정적 하중 케이스만 사용 (모드 A).
    --dynamic-opts 지정 시 ESLExtractor로 동적 스냅샷을 추출하여 병합 (모드 C/D).

    Attributes
    ----------
    cfg    : argparse.Namespace
    model  : WHTMeshModel
    solver : WHTopographySolver
    """

    def __init__(self, cfg):
        self.cfg    = cfg
        self.model  : Optional[WHTMeshModel]        = None
        self.node_db: Optional[dict]                = None
        self.solver : Optional[WHTopographySolver]  = None

    def run(self) -> None:
        """파이프라인 전체 실행."""
        print(f"\n{'='*80}")
        print(f" [wht_topo] Industrial Topography Optimization Pipeline")
        print(f"{'='*80}\n")
        self._build_mesh()
        load_cases = self._prepare_load_cases()
        self._build_solver(load_cases)
        self._run_optimizer()
        self._discretize()
        self._apply_shape()
        self._export()
        self._visualize()

    # ── 단계별 메서드 ────────────────────────────────────────────────────────

    def _build_mesh(self) -> None:
        print(" [1] 메시 생성...")
        self.model, self.node_db = _build_tray()
        print(f"     노드={len(self.node_db)}, 요소={len(self.model.elements)}")

    def _prepare_load_cases(self) -> Optional[list]:
        """
        정적 하중 케이스를 구성하고, --dynamic-opts 지정 시
        ESL 스냅샷을 추출하여 합산 리스트를 반환합니다.

        모드 A (dynamic_opts 없음): None 반환 → solver 내부에서 자동 생성.
        모드 C/D (dynamic_opts 있음): 정적 + 동적 ESL 합산 리스트 반환.
        """
        if not getattr(self.cfg, 'dynamic_opts', None):
            return None

        weights      = self._weights()
        load_manager = StochasticLoadManager(self.model)
        static_cases = load_manager.get_load_cases(
            mesh_size_z=self.cfg.mesh_size, weights=weights
        )

        opts    = [s.strip() for s in self.cfg.dynamic_opts.split(',')]
        t_start = float(opts[1]) if len(opts) > 1 else None  # None → CSV 헤더 자동 적용
        dyn_snaps = ESLExtractor(
            model        = self.model,
            node_db      = self.node_db,
            csv_path     = opts[0],
            t_start      = t_start,
            n_windows    = self.cfg.n_windows,
            n_top        = self.cfg.n_top,
            add_inertia  = self.cfg.add_inertia,
            use_global_z = self.cfg.use_global_z,
        ).extract()

        merged = static_cases + dyn_snaps
        print(f"\n [3] 하중 케이스 병합: 정적 {len(static_cases)} + "
              f"동적 ESL {len(dyn_snaps)} = 총 {len(merged)}개")
        return merged

    def _build_solver(self, load_cases: Optional[list]) -> None:
        print(f"\n [4] 최적화 준비 (h_max={self.cfg.bead_height}mm, "
              f"min_width={self.cfg.min_width}mm, "
              f"bead_area={self.cfg.bead_area*100:.0f}%)...")
        load_manager = StochasticLoadManager(self.model)
        self.solver  = WHTopographySolver(
            self.model, load_manager,
            bead_height_max  = self.cfg.bead_height,
            bead_height_ratio= self.cfg.bead_area,
            min_width        = self.cfg.min_width,
            draw_dir         = self.cfg.draw_dir,
            weights          = self._weights(),
            mesh_size_z      = self.cfg.mesh_size,
            sym_x            = self.cfg.sym_x,
            bead_connect     = self.cfg.bead_connect,
            connect_gap      = self.cfg.connect_gap,
            bead_steps       = self.cfg.height_steps,
            load_cases       = load_cases,
        )

    def _run_optimizer(self) -> None:
        """최적화를 실행합니다. GUI 지정 시 모니터링 프로세스를 병렬로 시작합니다."""
        ui_process = None
        callback   = None
        stop_event = None

        if not self.cfg.no_gui:
            from wht_topo.monitor_ui import start_monitor_ui
            queue      = multiprocessing.Queue()
            stop_event = multiprocessing.Event()
            ui_process = multiprocessing.Process(
                target=start_monitor_ui, args=(queue, stop_event)
            )
            ui_process.start()
            callback = queue.put

        print(f" [4] 최적화 실행 (max_iter={self.cfg.iters})...")
        self.solver.solve(max_iter=self.cfg.iters, callback=callback, stop_event=stop_event)

        if ui_process and ui_process.is_alive():
            queue.put("STOP")

    def _discretize(self) -> None:
        """height_steps >= 2 인 경우 비드 높이를 이산 레벨로 양자화합니다."""
        if self.cfg.height_steps < 2:
            return
        n      = self.cfg.height_steps
        levels = np.linspace(0.0, self.solver.h_max, n)
        self.solver.heights = levels[
            np.abs(self.solver.heights[:, None] - levels).argmin(axis=1)
        ]
        final = np.unique(np.round(self.solver.heights, 4))
        print(f"\n [5] 이산화: {n}단계 → {final} mm")

    def _apply_shape(self) -> None:
        """최종 비드 형상을 모델 노드 좌표에 적용합니다."""
        self.solver.apply_final_shape(skip_filter=(self.cfg.height_steps >= 2))

    def _export(self) -> None:
        if not self.cfg.export:
            return
        self.model.export_to_solver('lsdyna', self.cfg.export, reorder=True)
        print(f" [7] 결과 저장: {self.cfg.export}")

    def _visualize(self) -> None:
        if self.cfg.no_viz:
            return
        print(" [8] 시각화...")
        discrete     = self.cfg.height_steps >= 2
        result_data  = self.model.to_wht_result_data()
        heights_full = self.solver.get_full_heights(skip_filter=discrete)
        result_data.point_data["Bead_Height"] = (
            (heights_full / (self.solver.h_max + 1e-12)).reshape(1, -1, 1)
        )
        viz = WHTVisualizer(title="Industrial Topography Result")
        viz.load_results(result_data)
        viz.show()

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    def _weights(self) -> dict:
        return {
            "bending":       self.cfg.w_bending,
            "bending_xspan": self.cfg.w_bending_xspan,
            "bending_yspan": self.cfg.w_bending_yspan,
            "twisting":      self.cfg.w_twisting,
            "twisting_alt":  self.cfg.w_twisting_alt,
            "lifting":       self.cfg.w_lifting,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WHT 산업용 섀시 비드 최적화 도구 (Dynamic-ESL 통합형)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행 예제 (모드별)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[모드 A] 기본 정적 최적화
  python wht_topo/run_topo.py
  python wht_topo/run_topo.py --iters 20 --bead-height 12
  python wht_topo/run_topo.py --iters 30 --bead-area 0.30 --min-width 50 --height-steps 2

[모드 B] CSV 단독 동적 응답 해석 (최적화 생략)

  [권장] 실측 CSV + t_start 자동 적용
    python wht_topo/run_topo.py --pos-data wht_topo/structural_dynamics.csv
    -> CSV 헤더 "# start_time" 값 자동 적용

  [권장] t_start 명시 + 감쇠비 조정
    python wht_topo/run_topo.py \\
      --pos-data wht_topo/structural_dynamics.csv --t-start 1.6 --zeta 0.03
    -> 1.6s 이전 준정적 구간 제외, 감쇠 3% 적용

  [빠른 확인] 시각화 생략 (ParaView HDF만 저장)
    python wht_topo/run_topo.py \\
      --pos-data wht_topo/structural_dynamics.csv --no-viz

[모드 C] 동적 충격 통합 최적화 (ESL → 정적 + 동적 하중 병합)

  [권장] t_start CSV 헤더 자동 + 관성 하중 포함
    python wht_topo/run_topo.py \\
      --dynamic-opts "wht_topo/structural_dynamics.csv" --add-inertia
    -> CSV 헤더 start_time 자동 적용, ESL 10개 추출 후 정적 케이스와 병합

  [권장] t_start 명시 + ESL 개수 확대
    python wht_topo/run_topo.py \\
      --dynamic-opts "data.csv,1.6" --add-inertia --n-windows 50 --n-top 15
    -> 50개 창으로 고밀도 피크 탐색, 상위 15개 ESL 선정

  관성 하중 제외 (순수 변위 SPCD 응답 비교)
    python wht_topo/run_topo.py \\
      --dynamic-opts "data.csv,1.6" --n-top 10

[모드 D] 산업용 고신뢰성 완전 제약 설계

  [권장] 낙하 충격 + 관성 하중 + 대칭 + 비드 연결 + 이산화
    python wht_topo/run_topo.py \\
      --dynamic-opts "data.csv" --add-inertia \\
      --sym-x --bead-connect --height-steps 2 --export final.k

  [권장] 서버/헤드리스 + 결과 저장
    python wht_topo/run_topo.py \\
      --dynamic-opts "data.csv" --add-inertia \\
      --sym-x --bead-connect --height-steps 2 \\
      --export final.k --no-gui --no-viz

  ESL 정밀도 향상 (n-top 확대)
    python wht_topo/run_topo.py \\
      --dynamic-opts "data.csv" --add-inertia \\
      --n-windows 50 --n-top 20 \\
      --sym-x --bead-connect --height-steps 2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
옵션 상세 설명
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--pos-data      CSV 경로 지정 시 모드 B(동적 해석 단독)로 실행됩니다.
                CSV 형식: # 주석 헤더(코너 좌표·start_time) + Frame,Time,C1_X...C8_Z

--t-start       CSV 분석 시작 시점(s). (모드 B 전용)
                미지정 시 CSV 헤더 "# start_time" 자동 적용.
                헤더에도 없으면 0.0 (전체 구간 사용).

--dynamic-opts  "CSV경로" 또는 "CSV경로,시작시간(s)" 형식. (모드 C/D)
                시간 미지정 시 CSV 헤더 start_time 자동 적용.

--add-inertia   관성 하중 F=-ma를 모든 내부 노드에 이선형 보간으로 분포 인가.
                낙하·충격처럼 가속도 기여가 큰 하중 케이스에 반드시 사용.

--n-windows     SE 이력 분할 창 수. 클수록 피크 후보를 촘촘히 탐색.
                권장: 짧은 충격 구간 → 20, 긴 진동 구간 → 50 (기본: 30)

--n-top         최종 선정 ESL 개수. 기본: 10.
                정밀도↑ 원할 때 15~20 사용 (최적화 계산 시간 증가).

--height-steps  비드 이산화 레벨. 기본: 2 → {0, h_max} 이진화.
                생산성 우선 설계에 사용. 3 이상 지정 시 다단계 이산화.

--bead-connect  끊어진 비드를 형태학적 닫힘으로 자동 연결.
  (기본: ON)    --connect-gap 으로 연결 허용 최대 갭(mm)을 제어.

--sym-x         X축 기준 좌우 대칭 제약 활성화.
  (기본: ON)    --no-sym-x 로 해제.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    )

    # 최적화 기본 제어
    g = parser.add_argument_group("최적화 기본 설정")
    g.add_argument("--iters",       type=int,   default=15,   help="최대 반복 횟수 (기본: 15)")
    g.add_argument("--bead-height", type=float, default=10.0, help="최대 비드 높이 mm (기본: 10.0)")
    g.add_argument("--min-width",   type=float, default=30.0, help="최소 비드 폭 mm (기본: 30.0)")
    g.add_argument("--bead-area",   type=float, default=0.35, help="비드 점유 면적 비율 0~1 (기본: 0.35)")

    # 비드 형상 및 제조 제약
    g = parser.add_argument_group("비드 형상 및 제조 제약")
    g.add_argument("--sym-x",          action="store_true", default=True,  help="좌우 대칭 활성화 (기본: 활성)")
    g.add_argument("--no-sym-x",       action="store_false", dest="sym_x", help="좌우 대칭 해제")
    g.add_argument("--bead-connect",   action="store_true", default=True,  help="비드 자동 연결 활성화 (기본: 활성)")
    g.add_argument("--no-bead-connect",action="store_false", dest="bead_connect", help="비드 연결 비활성화")
    g.add_argument("--connect-gap",    type=float, default=120.0, help="비드 연결 최대 갭 mm (기본: 120.0)")
    g.add_argument("--height-steps",   type=int,   default=2,    help="비드 이산화 단계 (기본: 2 → {0, h_max})")
    g.add_argument("--draw-dir",       type=float, nargs=3, default=[0.0, 0.0, 1.0], help="비드 돌출 방향 (기본: 0 0 1)")

    # CSV 단독 동적 해석 (모드 B)
    g = parser.add_argument_group("CSV 단독 동적 응답 해석 (모드 B, 최적화 생략)")
    g.add_argument("--pos-data",  type=str,   default=None,  help="CSV 경로: 지정 시 동적 해석만 실행")
    g.add_argument("--t-start",   type=float, default=None,
                   help="분석 시작 시점 s (기본: CSV 헤더 start_time 자동 적용)")
    g.add_argument("--dt",        type=float, default=1e-4,  help="적분 시간 스텝 s (기본: 1e-4)")
    g.add_argument("--zeta",      type=float, default=0.02,  help="Rayleigh 감쇠비 (기본: 0.02)")
    g.add_argument("--corner-r",  type=float, default=150.0, help="코너 탐색 반경 mm (fallback 전용)")

    # 동적 ESL 통합 최적화 (모드 C/D)
    g = parser.add_argument_group("동적 ESL 통합 최적화 (모드 C/D)")
    g.add_argument("--dynamic-opts", type=str, default=None,  help="'CSV경로,시작시간(s)'")
    g.add_argument("--add-inertia",  action="store_true",     help="관성 하중(-ma) 인가")
    g.add_argument("--n-top",        type=int, default=10,    help="추출 ESL 개수 (기본: 10)")
    g.add_argument("--n-windows",    type=int, default=30,    help="시간 이력 분할 수 (기본: 30)")
    g.add_argument("--use-global-z", action="store_true",     help="글로벌 Z 궤적 직접 사용")

    # 하중 케이스 가중치
    g = parser.add_argument_group("정적 하중 케이스 가중치 (Weighted Sum)")
    g.add_argument("--w-bending",       type=float, default=1.0, help="중앙 굽힘 (기본: 1.0)")
    g.add_argument("--w-bending-xspan", type=float, default=0.8, help="X방향 스팬 굽힘 (기본: 0.8)")
    g.add_argument("--w-bending-yspan", type=float, default=0.8, help="Y방향 스팬 굽힘 (기본: 0.8)")
    g.add_argument("--w-twisting",      type=float, default=1.5, help="대각 비틀림 (기본: 1.5)")
    g.add_argument("--w-twisting-alt",  type=float, default=1.5, help="반전 대각 비틀림 (기본: 1.5)")
    g.add_argument("--w-lifting",       type=float, default=1.2, help="4코너 리프팅 합산 (기본: 1.2)")

    # 출력 및 시각화
    g = parser.add_argument_group("출력 및 시각화")
    g.add_argument("--export",    type=str,  default="industrial_bead.k", help="LS-DYNA .k 저장 경로")
    g.add_argument("--no-gui",    action="store_true", help="모니터링 GUI 비활성화")
    g.add_argument("--no-viz",    action="store_true", help="최종 3D 시각화 생략")
    g.add_argument("--mesh-size", type=float, default=10.0, help="BC 탐색 기준 메시 크기 mm")

    args = parser.parse_args()

    if args.pos_data:
        PosDynamicPipeline(args).run()
    else:
        TopographyPipeline(args).run()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
