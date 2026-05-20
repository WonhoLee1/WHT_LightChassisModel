# -*- coding: utf-8 -*-
"""
run_topo.py — WHT 산업용 섀시 비드 최적화(Topography) 통합 도구
================================================================

[실행 모드]
  모드 A | 기본 정적 최적화
    굽힘·비틀림·리프팅 표준 정적 하중 케이스로 비드 패턴 최적화.
    --dynamic-opts / --pos-data 미지정 시 자동 선택.
    실행: python wht_topo/run_topo.py --iters 20 --sym-x

  모드 B | CSV 단독 동적 응답 해석 (최적화 생략)
    실측 4코너 위치 데이터로 과도 응답 해석 후 ParaView HDF 저장.
    하중: 각 코너 마스터 노드(#900000~#900003)에 로컬 Z-SPCD 적용(RBE3 연결).
    실행: python wht_topo/run_topo.py \
      --pos-data wht_topo/structural_dynamics_rear.csv \
      --t-start 0.4 --t-end 0.6

  모드 C | 동적 충격 통합 최적화 — 반복 ESL (기본)
    매 이터레이션 현재 비드 형상에서 동해석을 재실행하여 ESL을 갱신.
    구조가 강성화될수록 동적 응답이 바뀌고 ESL도 함께 진화 → 정식 반복 ESL 절차.
    정적 하중(굽힘·비틀림·리프팅) + 동적 ESL이 함께 목적 함수에 반영됨.
    단일 시나리오: python wht_topo/run_topo.py \
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" --add-inertia
    복수 시나리오: python wht_topo/run_topo.py \
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \
      --w-dynamic 1.0 1.0 1.0 --add-inertia

  모드 C' | 동적 충격 통합 최적화 — 1회 ESL (--no-iterative-esl)
    최적화 전 초기 형상에서 ESL을 1회 추출하여 전체 최적화에 고정.
    계산 비용 낮음. 모드 C 결과와 비교 연구에 사용.
    실행: python wht_topo/run_topo.py \
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \
      --add-inertia --no-iterative-esl

  모드 D | 고신뢰성 산업용 완전 제약 설계
    반복 ESL + 관성 하중 + 좌우 대칭 + 비드 연결 + 이산화(0/h_max) + 배제 영역
    실행: python wht_topo/run_topo.py \
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \
      --add-inertia --sym-x --bead-connect --height-steps 2 \
      --exclude-rect 450,250,120,120 --exclude-rect 1350,250,120,120

[하중 케이스 구성 원칙]
  정적 하중과 동적 ESL은 독립적으로 가중치를 설정하여 목적 함수에 합산됩니다.
  동적 ESL 적용 시 정적 하중이 자동으로 꺼지지 않습니다.
  정적 하중을 완전히 끄려면 --no-static 을 사용하세요.

  하중 케이스 = 정적 케이스(w>0인 것)  +  동적 ESL 케이스
                ──────────────────────    ───────────────────────────
                --w-bending  W  F         시간분할 ESL (--w-dynamic W1 [W2 ...])
                --w-twisting W  F         요소별 피크 ESL (--w-peak)
                --w-lifting  W  F
                ...
                --no-static 으로 전체 제외 가능

[비드 배제 영역 — --exclude-rect / --exclude-poly]
  마운팅 홀, 슬롯, 보스 등 비드를 생성하면 안 되는 영역을 지정합니다.
  설계 요소의 도심 XY 좌표가 배제 영역 내에 포함되면 설계 변수에서 제외됩니다.

  사각형 배제 (중심 CX,CY + 가로W × 세로H):
    --exclude-rect CX,CY,W,H
    예) 마운팅 보스 4개소:
      --exclude-rect 450,250,120,120 --exclude-rect 1350,250,120,120 \
      --exclude-rect 450,950,120,120 --exclude-rect 1350,950,120,120

  다각형 배제 (꼭짓점 X1,Y1,X2,Y2,... 나열, 자동 닫힘):
    --exclude-poly X1,Y1,X2,Y2,...
    예) 장공/슬롯:
      --exclude-poly 600,400,900,400,900,500,600,500

  두 방식 혼용 가능:
    --exclude-rect 450,250,120,120 --exclude-poly 600,400,900,400,900,500,600,500

[목적함수 옵션 — --obj-type / --normalize-obj / --freq-penalty]

  기본 동작: f = (Σ w_i·C_i) / C_0   (Iter 0 총 컴플라이언스로 스케일)

  ① --normalize-obj
       f = Σ w_i·(C_i/C_i0)
       각 케이스를 Iter 0의 자기 값으로 정규화. 서로 크기가 다른 정적·동적 케이스가
       혼재할 때 큰 케이스가 작은 케이스를 압도하지 않도록 균등 반영.

  ② --obj-type max  (--obj-alpha α, 기본 10.0)
       f = (1/α)·log(Σ exp(α·w_i·C_i/C_i0))   (softmax)
       가장 성능이 나쁜 케이스에 집중하는 Min-Max 최적화.
       α가 클수록 hard-max에 근접 (→ 최악 케이스만 반영).
       α가 작을수록 soft 평균에 근접 (α=0 극한: Σ w_i·C_i/C_i0).

  ③ --obj-type sum+max
       f = 0.5·f_sum + 0.5·f_max
       평균 성능(sum)과 최악 케이스 방어(max)를 동시에 추구하는 절충안.

  ④ --freq-penalty W F0_HZ
       P = W·max(0, F0-f₁)²/F0²   를 위의 어떤 목적함수에도 추가 가능.
       f₁ < F0 일 때 비드 강성화 방향으로 패널티가 민감도에 기여.
       민감도는 모드 형상 φ₁을 활용한 φ₁ᵀ(∂K/∂h)φ₁ 식으로 계산
       (compliance 민감도와 동일한 JAX vmap 재사용, 추가 FEA 불필요).

  권장 조합:
    정적+동적 ESL 균등 반영  : --normalize-obj
    최악 케이스 방어          : --obj-type max --normalize-obj
    주파수+강성 동시 개선     : --normalize-obj --freq-penalty 5.0 50
    전면 산업 설계            : --obj-type sum+max --normalize-obj --freq-penalty 3.0 40

[동적 ESL 알고리즘 — 시간분할 스냅샷 (--w-dynamic)]
  1. CSV에서 [t_start, t_end] 구간 추출 (t_start: 인자 > CSV 헤더 > 0.0 우선순위).
  2. 강체 회전 제거 → 로컬 Z 방향 순수 벤딩 변위 분리 (calculate_local_z_history).
  3. 4코너 마스터 노드(#900000~900003) + RBE3 → Z-SPCD 하중 그룹 구성.
     CSV # 헤더의 C5~C8 좌표로 가장 가까운 FEM 노드 3개씩 자동 탐색.
  4. (옵션 --add-inertia) 이선형 보간 + 집중 질량 → 관성 하중(F=-ma) 전 노드 인가.
  5. Newmark-β 직접 적분 과도 응답 해석 (ζ=2%, --zeta 조정 가능).
  6. 전체 SE 이력(SE = ½uᵀKu) 계산 → n_windows 구간 분할 → 전역 피크 후보 추출.
  7. Greedy Max-Min Cosine Similarity 다양성 선별 → Top-n_top 스냅샷 선정.
  8. WHTLoadCase(SPCD 형태)로 변환 → 목적 함수 하중 케이스 풀에 추가.
  출력: results/D날짜_시간/esl_se_report[_iterNNN].png
        (SE 이력 + 윈도우 분할 + 선택 시점 마킹)

[동적 ESL 알고리즘 — 요소별 최대 SE 피크 (--w-peak)]
  동적 해석 완료 후 모든 쉘 요소(QUAD4/TRIA3)에 대해:
    SE_e(t) = ½ uₑ(t)ᵀ Kₑ uₑ(t)   — 요소별 변형에너지 이력
    t*_e    = argmax_t SE_e(t)       — 요소별 피크 시각 (요소마다 상이)
    f*_e    = Kₑ uₑ(t*_e)           — 피크 시각의 내력벡터
  전체 등가 하중:  F_eq = Σ_e Lₑᵀ f*_e  (전역 DOF Assembly)
  → 어떤 실제 순간에도 동시에 발생하지 않는 "최악의 포락선"을 보수적으로 커버.
  → 시간분할 ESL과 가중치를 독립 조정하여 병용 가능 (기본값 0.0 = 비활성).
  출력: results/D날짜_시간/esl_peak_report[_iterNNN].png
        (요소별 피크 SE 분포 + 피크 발생 시각 + 등가 노드력 상위 30)

[결과 파일 구조]
  results/
  └── D20260516_142301/          ← 실행 시작 시 1회 생성
      ├── paraview/
      │   ├── iter_000.hdf       ← 이터레이션별 변위·응력·모드 (ParaView)
      │   ├── iter_001.hdf
      │   └── ...
      ├── esl_se_report_iter000.png   ← 시간분할 ESL SE 이력 리포트
      ├── esl_peak_report_iter000.png ← 요소별 피크 ESL 리포트
      └── final.k                ← LS-DYNA 최적 비드 패턴 출력


단위계: mm, N, tonne, s
"""

import argparse
import math
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


class _Tee:
    """sys.stdout을 콘솔 + 파일에 동시 출력하는 래퍼."""
    def __init__(self, stream, log_path: Path):
        self._stream   = stream
        self._log_path = log_path
        self._file     = open(log_path, 'w', encoding='utf-8')
    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
    def flush(self):
        self._stream.flush()
        self._file.flush()
    def fileno(self):
        return self._stream.fileno()
    def close(self):
        self._file.close()
    @property
    def encoding(self):
        return self._stream.encoding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_topo.loads import StochasticLoadManager
from wht_topo.solver import WHTopographySolver
from wht_visualizer.wht_visualizer import WHTVisualizer
from wht_converter.wht_models import WHTMetadata, WHTResultData
from wht_converter.wht_exporters import VTKHDFExporter
from test_jaxSSO.mesh_utils import generate_shell_tray
from wht_solver.load_cases import WHTLoadCase, WHTForceEntry
from wht_solver.wht_quad4_element import _element_K_mitc4_plus
from wht_solver.wht_tria3_element import _element_K_tria3
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
MESH_XY, MESH_Z        = 30.0, 10.0
DRAFT_ANGLE            = 25.0
HOOK_SEQUENCE          = [(0.0, 12.0), (-5.0, 0.0), (0.0, -10.0), (-10.0, 0.0)]
MAT                    = dict(E=210000.0, nu=0.3, rho=7.85e-9, t=0.6)
CORNER_NAMES           = ['C5', 'C6', 'C7', 'C8']
CORNER_MAP             = dict(enumerate(CORNER_NAMES))   # {0:'C5', 1:'C6', ...}
ZETA                   = 0.02   # Rayleigh 감쇠비
G_MM_S2                = 9_810.0  # mm/s² per g
GRAVITY_MM             = 9_806.0  # mm/s²  표준 중력가속도 (9.80665 m/s² → 9806.65, 반올림 9806)


# ─────────────────────────────────────────────────────────────────────────────
# 터미널 출력 유틸 — 한글/CJK 2칸 표시 폭 보정
# ─────────────────────────────────────────────────────────────────────────────

_W = 100  # 박스 전체 표시 폭 (테두리 포함)
#
# 박스 폭 공식 검증 (단일 폭 문자 기준):
#   상단  : "  ┌" (3) + dashes + "┐" (1) = _W  →  dashes = _W - 4
#   행    : "  │" (3) + content + "│" (1) = _W  →  content = _W - 4
#   구분  : "  ├" (3) + dashes + "┤" (1) = _W  →  dashes = _W - 4
#   하단  : "  └" (3) + dashes + "┘" (1) = _W  →  dashes = _W - 4


def _dw(s: str) -> int:
    """터미널 표시 폭 계산: 한글·CJK = 2칸, 그 외 = 1칸."""
    n = 0
    for c in s:
        cp = ord(c)
        if (0xAC00 <= cp <= 0xD7A3   # 한글 완성형
                or 0x1100 <= cp <= 0x11FF   # 한글 자모
                or 0x3130 <= cp <= 0x318F   # 한글 호환 자모
                or 0x4E00 <= cp <= 0x9FFF   # CJK 통합 한자
                or 0x3000 <= cp <= 0x303F   # CJK 기호
                or 0xF900 <= cp <= 0xFAFF): # CJK 호환 한자
            n += 2
        else:
            n += 1
    return n


def _rpad(s: str, w: int) -> str:
    """표시 폭 기준 오른쪽 공백 패딩."""
    return s + ' ' * max(0, w - _dw(s))


def _hdr(title: str, ch: str = "═") -> str:
    """전체 폭 헤더 라인 (테두리 없음, 단순 장식선)."""
    inner = f"  {ch*2} {title} "
    n = max(0, _W - _dw(inner))  # 오른쪽 끝까지 채움
    return inner + ch * n


def _section(title: str, step: str = "") -> None:
    """박스 상단: ┌─ [step] title ─...─┐  (총 _W 표시 폭)."""
    prefix = f"[{step}] " if step else ""
    inner  = f"  ┌─ {prefix}{title} "
    # inner(_dw) + dashes + ┐(1) = _W  →  dashes = _W - _dw(inner) - 1
    n = max(2, _W - _dw(inner) - 1)
    print(f"\n{inner}{'─' * n}┐")


def _endsec() -> None:
    """박스 하단: └─...─┘  (총 _W 표시 폭)."""
    #   "  └"(3) + dashes + "┘"(1) = _W  →  dashes = _W - 4
    print(f"  └{'─' * (_W - 4)}┘")


def _row(label: str, value: str = "", lw: int = 30) -> None:
    """박스 행: │  <label padded lw> <value> <spaces> │  (총 _W 표시 폭).
    lw: 레이블 표시 폭 기준 (한글 포함)."""
    inner = f"  │  {_rpad(label, lw)} {value}"
    # inner(_dw) + spaces + │(1) = _W  →  spaces = _W - _dw(inner) - 1
    n = max(0, _W - _dw(inner) - 1)
    print(f"{inner}{' ' * n}│")


def _row_raw(content: str) -> None:
    """박스 행: 미리 구성된 문자열을 출력하고 우측 테두리를 맞춤."""
    inner = f"  │  {content}"
    n = max(0, _W - _dw(inner) - 1)
    print(f"{inner}{' ' * n}│")


def _sep() -> None:
    """박스 구분선: ├─...─┤  (총 _W 표시 폭)."""
    print(f"  ├{'─' * (_W - 4)}┤")


def _blank() -> None:
    """빈 행: │ spaces │  (총 _W 표시 폭)."""
    print(f"  │{' ' * (_W - 4)}│")


# ─────────────────────────────────────────────────────────────────────────────
# 공유 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────────────

def _build_tray(
    width: float       = TRAY_W,
    length: float      = TRAY_L,
    height: float      = TRAY_H,
    mesh_xy: float     = MESH_XY,
    mesh_z: float      = MESH_Z,
    draft_angle: float = DRAFT_ANGLE,
    E: float           = MAT['E'],
    nu: float          = MAT['nu'],
    rho: float         = MAT['rho'],
    t: float           = MAT['t'],
) -> Tuple[WHTMeshModel, dict]:
    """
    트레이 셸 메시를 생성하고 WHTMeshModel을 반환합니다.

    Parameters
    ----------
    width, length, height : 섀시 외형 치수 (mm)
    mesh_xy, mesh_z       : XY면·Z방향 메시 크기 (mm)
    draft_angle           : 측면 드래프트 각도 (deg)
    E, nu, rho            : 재료 물성 (MPa, -, tonne/mm³)
    t                     : 기본 판 두께 (mm)
    """
    node_db, elem_db = generate_shell_tray(
        width=width, length=length, height=height,
        mesh_size_xy=mesh_xy, mesh_size_z=mesh_z,
        draft_angle=draft_angle, flange_segments=HOOK_SEQUENCE,
        flanges=(False, True, True, True), mesh_type='quad4',
    )
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db,
                                           name="DynamicTray", is_solid=False)
    model.add_material(1, E=E, nu=nu, rho=rho)
    model.add_property(1, "PSHELL", t=t, mid=1)
    for eid in model.elements:
        model.elements[eid].pid = 1
    return model, node_db


def _parse_exclude_zones(
    rect_args: Optional[List[str]],
    poly_args: Optional[List[str]],
) -> List[dict]:
    """
    --exclude-rect / --exclude-poly CLI 인자를 solver가 소비하는 dict 리스트로 변환합니다.

    rect_args : ["cx,cy,w,h", ...]   — 중심점 + 크기
    poly_args : ["x1,y1,x2,y2,..."]  — 꼭짓점 나열 (짝수 개)

    Returns
    -------
    [{'type':'rect', 'cx':, 'cy':, 'w':, 'h':}, ...]
    [{'type':'poly', 'vertices':[(x,y),...]}, ...]
    """
    zones: List[dict] = []
    for s in (rect_args or []):
        v = [float(x) for x in s.split(',')]
        if len(v) != 4:
            raise ValueError(f"--exclude-rect 형식 오류 (cx,cy,w,h 4개 필요): '{s}'")
        zones.append({'type': 'rect', 'cx': v[0], 'cy': v[1], 'w': v[2], 'h': v[3]})
    for s in (poly_args or []):
        v = [float(x) for x in s.split(',')]
        if len(v) < 6 or len(v) % 2 != 0:
            raise ValueError(f"--exclude-poly 형식 오류 (짝수 개 좌표 ≥ 6 필요): '{s}'")
        verts = [(v[i], v[i + 1]) for i in range(0, len(v), 2)]
        zones.append({'type': 'poly', 'vertices': verts})
    return zones


def _load_csv(
    path_str: str,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Tuple[pd.DataFrame, np.ndarray, dict]:
    """
    CSV를 로드하고 'Chassis_' 접두사를 제거한 뒤 시간 범위 필터를 적용합니다.

    # 주석 헤더에서 코너 기준 좌표와 start_time을 파싱합니다.
    t_start 우선순위: 인자 > CSV 헤더 > 0.0
    t_end   미지정 시: CSV 마지막 프레임까지 사용

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

    if t_end is not None:
        df = df[df['Time'] <= t_end].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"CSV에 t <= {t_end}s 데이터가 없습니다.")

    time_raw = df['Time'].to_numpy(dtype=float)
    return df, time_raw - time_raw[0], header


def _apply_inertia_loads(
    df, time_arr, node_db, model, load_groups: list
) -> tuple:
    """
    코너 가속도 쌍선형 보간으로 모든 내부 노드에 관성 하중 F = -m·a 를 인가합니다.

    load_groups 리스트에 InterpLoadGroup(FORCE)을 직접 추가합니다.

    Returns
    -------
    total_mass   : float   총 섀시 질량 (tonne)
    corner_accels: ndarray (T, 4) 코너 가속도 [mm/s²]
    n_inertia    : int     관성 하중 인가 노드 수
    """
    _, traj_mm = calculate_local_z_history(df, time_arr)
    dt         = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 1e-4
    accels     = calculate_corner_accelerations(traj_mm, dt)  # (T, 4)

    all_xyz = np.array(list(node_db.values()))
    xmin, xmax = all_xyz[:, 0].min(), all_xyz[:, 0].max()
    ymin, ymax = all_xyz[:, 1].min(), all_xyz[:, 1].max()
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)

    temp   = WHTDynamicSolver(model)
    jm, s_nids, n2i = temp._build_jaxsso_model()
    m_diag = temp._assemble_lumped_mass(jm, jm.ndof, s_nids, n2i)

    total_mass = float(sum(
        m_diag[n2i[nid] * 6 + 2]
        for nid in node_db
        if n2i.get(nid) is not None and nid < 900000
        and m_diag[n2i[nid] * 6 + 2] > 1e-12
    ))

    n_inertia = 0
    for nid in node_db:
        ix = n2i.get(nid)
        if ix is None or nid >= 900000:
            continue
        x, y, _ = node_db[nid]
        nx, ny  = (x - xmin) / dx, (y - ymin) / dy
        a_z = (
            accels[:, 0] * (nx)     * (ny)      +  # C5: +X+Y
            accels[:, 1] * (nx)     * (1 - ny)  +  # C6: +X-Y
            accels[:, 2] * (1 - nx) * (1 - ny)  +  # C7: -X-Y
            accels[:, 3] * (1 - nx) * (ny)         # C8: -X+Y
        )
        node_mass = m_diag[ix * 6 + 2]
        if node_mass > 1e-12:
            load_groups.append(
                InterpLoadGroup([nid], 2, time_arr, -node_mass * a_z, "FORCE")
            )
            n_inertia += 1

    return total_mass, accels, n_inertia


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
    t_start     : float            분석 시작 시간 (s). None → CSV 헤더 start_time 자동 적용
    t_end       : float | None     분석 종료 시간 (s). None → CSV 마지막 프레임
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
        t_end: Optional[float] = None,
        n_windows: int = 30,
        n_top: int = 10,
        add_inertia: bool = True,
        use_global_z: bool = False,
        esl_weight: float = 1.0,
        w_peak: float = 0.0,
        iteration: int = -1,
        out_dir: Optional[Path] = None,
    ):
        self.model        = model
        self.node_db      = node_db
        self.csv_path     = csv_path
        self.t_start      = t_start   # None → CSV 헤더 start_time 자동 적용
        self.t_end        = t_end     # None → CSV 마지막 프레임까지
        self.n_windows    = n_windows
        self.n_top        = n_top
        self.add_inertia  = add_inertia
        self.use_global_z = use_global_z
        self.esl_weight   = esl_weight
        self.w_peak       = w_peak
        self.iteration    = iteration  # -1: 단독 실행, ≥0: 최적화 루프 내
        self.out_dir      = out_dir    # None이면 CSV 옆 디렉토리에 저장

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
        iter_tag = f"  Iter {self.iteration}" if self.iteration >= 0 else ""
        print(f"\n  {'─'*(_W-4)}")
        print(_hdr(f"ESL Extraction{iter_tag}  [{self.n_windows}win / top-{self.n_top}]", "─"))
        print(f"  CSV : {self.csv_path}")
        print(f"  관성 : {'포함 (--add-inertia)' if self.add_inertia else '제외'}"
              f"   변위 기준 : {'글로벌 Z' if self.use_global_z else '로컬 프레임 (강체 회전 제거)'}")
        print(f"  {'─'*(_W-4)}")

        self._load_csv()
        self._find_corners()
        self._build_spcd_groups()
        if self.add_inertia:
            self._add_inertia_loads()
        self._run_dynamic()
        self._extract_esl_cases()
        self._print_se_tables()
        self._cleanup_master_nodes()
        snapshots = self._build_snapshots()
        if self.w_peak > 0.0:
            _section("요소별 피크 SE 등가 하중 생성", "ESL-4")
            peak_lc = self._build_element_peak_lc()
            if peak_lc is not None:
                snapshots.append((peak_lc, self.w_peak))
                _row("피크 ESL", f"1개 추가  (가중치 w={self.w_peak})")
            _endsec()
        return snapshots

    # ── 단계별 private 메서드 ────────────────────────────────────────────────

    def _load_csv(self) -> None:
        self._df, self._time_arr, self._csv_header = _load_csv(
            self.csv_path, self.t_start, self.t_end
        )
        if self.t_start is None:
            self.t_start = self._csv_header.get('start_time') or 0.0
        dt   = self._time_arr[1] - self._time_arr[0] if len(self._time_arr) > 1 else 1e-4
        T    = float(self._time_arr[-1])
        t_end_actual = self.t_start + T
        t_end_str = f"{self.t_end:.3f} s (지정)" if self.t_end is not None else f"{t_end_actual:.3f} s"
        _section("CSV 로드 및 코너 변위 분석", "ESL-0")
        _row("파일", str(Path(self.csv_path).name))
        _row("기간", f"{self.t_start:.3f} s ~ {t_end_str}  "
                     f"({len(self._time_arr):,} frames,  dt={dt:.2e} s)")

        # 코너 변위 통계 (로컬 Z 기준)
        try:
            local_z, _ = calculate_local_z_history(self._df, self._time_arr)
            _sep()
            _row("코너", "  ΔZ peak (mm)       p-p (mm)   RMS (mm)")
            for lbl in CORNER_NAMES:
                z = local_z.get(lbl)
                if z is None: continue
                peak_pos = float(np.max(z)); peak_neg = float(np.min(z))
                pp = peak_pos - peak_neg; rms = float(np.sqrt(np.mean(z**2)))
                _row(f"  {lbl}", f"{peak_pos:+8.3f} / {peak_neg:+7.3f}   {pp:7.3f}   {rms:.3f}")
        except Exception:
            pass
        _endsec()

    def _find_corners(self) -> None:
        """CSV 헤더 코너 좌표 기준으로 C5~C8 각 3개 노드 탐색 (중복 없음)."""
        header_corners = self._csv_header.get('corner_positions', {})
        c5c8 = {k: v for k, v in header_corners.items() if k in ('C5', 'C6', 'C7', 'C8')}
        _section("FEM 코너 노드 탐색", "ESL-0b")
        if c5c8:
            corner_nodes = find_nodes_for_corners(self.node_db, c5c8, n_nodes=3)
            self._bot_groups = [
                (name, corner_nodes[name])
                for name in ['C5', 'C6', 'C7', 'C8'] if name in corner_nodes
            ]
            for name, nids in self._bot_groups:
                cx, cy, cz = c5c8[name]
                _row(f"  {name}", f"기준 ({cx:+.0f}, {cy:+.0f}, {cz:+.0f}) mm  →  노드 {len(nids)}개 할당")
        else:
            raw = find_corner_nodes(self.node_db, TRAY_W, TRAY_L, 150.0, z_min=0.0, z_max=2.0)
            self._bot_groups = [(CORNER_NAMES[i], g[1]) for i, g in enumerate(raw)]
            _row("경고", "CSV 헤더에 코너 좌표 없음 → 메시 기하 기반 탐색 사용")
        _endsec()

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
        total_mass, accels, n_inertia = _apply_inertia_loads(
            self._df, self._time_arr, self.node_db, self.model, self._load_groups
        )
        self._total_mass    = total_mass
        self._corner_accels = accels
        self._accel_dt      = self._time_arr[1] - self._time_arr[0] if len(self._time_arr) > 1 else 1e-4

        _section("관성 하중 (F = -m·a) 인가", "ESL-0c")
        _row("총 섀시 질량", f"{self._total_mass*1e6:.2f} kg  ({self._total_mass:.4e} tonne)")
        _row("관성 하중 인가 노드", f"{n_inertia:,}개")
        self._print_impact_summary()
        _endsec()

    def _print_impact_summary(self) -> None:
        """코너 가속도 기반 충격력 지표를 출력합니다 (CSV 간 비교용)."""
        if not hasattr(self, '_corner_accels') or self._corner_accels is None:
            return
        accels = self._corner_accels  # (T, 4)  mm/s²
        M      = getattr(self, '_total_mass', None)

        _sep()
        _row("코너 가속도 (로컬 Z)", "  peak [mm/s²]    peak [g]    RMS [g]")
        a_peaks_g = []
        for ci, lbl in enumerate(CORNER_NAMES):
            a   = accels[:, ci]
            pk  = float(np.max(np.abs(a)))
            rms = float(np.sqrt(np.mean(a**2)))
            g_pk  = pk  / G_MM_S2
            g_rms = rms / G_MM_S2
            a_peaks_g.append(g_pk)
            _row(f"  {lbl}", f"{pk:>12,.1f}   {g_pk:>8.3f} g   {g_rms:.3f} g")

        if M and M > 0:
            a_global_peak = float(np.max(np.abs(accels)))  # 전체 최대 (mm/s²)
            a_rms_global  = float(np.sqrt(np.mean(accels**2)))
            F_peak = M * a_global_peak          # N (tonne × mm/s² = N)
            F_rms  = M * a_rms_global
            g_peak = a_global_peak / G_MM_S2
            _sep()
            _row("▶ 충격력 지표 (M × a_peak)",
                 f"F_peak = {F_peak:,.1f} N   ({g_peak:.2f} g)")
            _row("  충격력 지표 (M × a_rms)",
                 f"F_rms  = {F_rms:,.1f} N")
            _row("  (이 값으로 다른 CSV 조건의 충격 심각도를 비교하세요)", "")

    def _run_dynamic(self) -> None:
        dt      = self._time_arr[1] - self._time_arr[0] if len(self._time_arr) > 1 else 1e-4
        T_total = float(self._time_arr[-1])
        _section(f"과도 응답 해석  (Newmark-β  ζ={ZETA*100:.0f}%)", "ESL-1")
        _row("적분 파라미터", f"dt={dt:.2e} s   T={T_total:.4f} s   저장 스냅샷=100")
        print()
        self._dyn_solver = WHTDynamicSolver(self.model)
        self._dyn_res    = self._dyn_solver.solve_direct_dynamic(
            self._load_groups, dt=dt, T=T_total, n_save=100,
            damping=DampingSpec(mode="zeta", zeta=ZETA),
        )
        _endsec()

    def _extract_esl_cases(self) -> None:
        self._esl_cases = self._dyn_solver.extract_esl_advanced(
            self._dyn_res, n_windows=self.n_windows, n_top=self.n_top
        )

    def _print_se_tables(self) -> None:
        """SE 이력 계산 → 윈도우 테이블 → Top-N ESL 요약 테이블 출력."""
        _section("변형에너지 이력 계산", "ESL-2")
        jm, s_nids, n2i = self._dyn_solver._build_jaxsso_model()
        K    = self._dyn_solver._assemble_K_scipy(jm, s_nids, n2i, stabilize=True)
        ndof = jm.ndof
        se   = np.zeros(self._dyn_res.n_save)
        for i in range(self._dyn_res.n_save):
            u_f   = self._dyn_res.u[i].flatten()[:ndof]
            se[i] = 0.5 * np.dot(u_f, K @ u_f)

        se_max = float(np.max(se)); se_mean = float(np.mean(se))
        se_rms = float(np.sqrt(np.mean(se**2)))
        t_peak = float(self._dyn_res.t_saved[int(np.argmax(se))])
        _row("SE 최대값", f"{se_max:.4e} J   @ t={t_peak:.4f} s")
        _row("SE 평균 / RMS", f"{se_mean:.4e} J   /   {se_rms:.4e} J")
        _sep()

        # 윈도우별 테이블 (4개씩 2열로 출력)
        win_size = len(self._dyn_res.t_saved) // self.n_windows
        rows = []
        for i in range(self.n_windows):
            idx_s = i * win_size
            idx_e = (i + 1) * win_size if i < self.n_windows - 1 else len(self._dyn_res.t_saved)
            if idx_s >= len(self._dyn_res.t_saved): break
            t_mid  = (self._dyn_res.t_saved[idx_s] + self._dyn_res.t_saved[idx_e - 1]) / 2.0
            se_win = se[idx_s:idx_e]
            rows.append((i+1, t_mid,
                          float(np.max(se_win)) if len(se_win) else 0.0,
                          float(np.sum(se_win)) if len(se_win) else 0.0))

        h_fmt = f"{'Win':>3}  {'T_mid(s)':>9}  {'SE_Peak(J)':>12}  {'SE_Sum(J)':>12}"
        _row_raw(f"{h_fmt}   {h_fmt}")

        half = (len(rows) + 1) // 2
        for i in range(half):
            r0 = rows[i]
            r1 = rows[i + half] if i + half < len(rows) else None
            left  = f"{r0[0]:>3}  {r0[1]:>9.4f}  {r0[2]:>12.4e}  {r0[3]:>12.4e}"
            right = (f"   {r1[0]:>3}  {r1[1]:>9.4f}  {r1[2]:>12.4e}  {r1[3]:>12.4e}"
                     if r1 else "")
            _row_raw(f"{left}{right}")
        _sep()

        # Top-N 요약
        _row_raw(f"Top-{self.n_top} ESL 로드케이스 (Diversity-aware Greedy Max-Min Cosine)")
        _row_raw(f"{'Rank':>4}  {'Time(s)':>9}  {'SE(J)':>12}  케이스명")
        for i, lc in enumerate(self._esl_cases):
            m_t  = re.search(r"t(\d+\.\d+)s",        lc.name)
            m_se = re.search(r"SE(\d+\.\d+e[+-]\d+)", lc.name)
            t_v  = float(m_t.group(1))  if m_t  else 0.0
            se_v = float(m_se.group(1)) if m_se else 0.0
            _row_raw(f"{i+1:>4}  {t_v:>9.4f}  {se_v:>12.4e}  {lc.name}")
        _endsec()
        self._plot_se_report(se)

    def _plot_se_report(self, se: np.ndarray) -> None:
        """SE 이력 그래프 → PNG 저장 (윈도우 분할 + ESL 선택 시점 마킹)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        t_arr    = self._dyn_res.t_saved
        n_t      = len(t_arr)
        win_size = n_t // self.n_windows

        fig, ax = plt.subplots(figsize=(12, 5))

        # 윈도우 배경 음영 + 경계선 + 레이블
        colors = ['#e8f4f8', '#f8ece8']
        for i in range(self.n_windows):
            idx_s = i * win_size
            idx_e = (i + 1) * win_size if i < self.n_windows - 1 else n_t
            if idx_s >= n_t:
                break
            t_s = t_arr[idx_s]
            t_e = t_arr[min(idx_e, n_t) - 1]
            ax.axvspan(t_s, t_e, color=colors[i % 2], alpha=0.6, lw=0)
            ax.axvline(t_s, color='#aaaaaa', lw=0.8, ls='--')
        ax.axvline(t_arr[-1], color='#aaaaaa', lw=0.8, ls='--')

        # SE 이력 선
        ax.plot(t_arr, se, color='steelblue', lw=1.2, label='Strain Energy')

        # 윈도우 레이블
        se_max = float(np.max(se)) if len(se) > 0 else 1.0
        for i in range(self.n_windows):
            idx_s = i * win_size
            idx_e = (i + 1) * win_size if i < self.n_windows - 1 else n_t
            if idx_s >= n_t:
                break
            t_s = t_arr[idx_s]
            t_e = t_arr[min(idx_e, n_t) - 1]
            ax.text((t_s + t_e) / 2, se_max * 1.02,
                    f"W{i+1}", ha='center', va='bottom', fontsize=7, color='#444444')

        # ESL 선택 시점 마킹
        for rank, lc in enumerate(self._esl_cases, 1):
            m = re.search(r"t(\d+\.\d+)s", lc.name)
            if not m:
                continue
            t_sel = float(m.group(1))
            idx   = np.argmin(np.abs(t_arr - t_sel))
            ax.scatter(t_arr[idx], se[idx], color='red', marker='*', s=120, zorder=5)
            ax.annotate(f"#{rank}", (t_arr[idx], se[idx]),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color='red')

        inertia_str = "inertia ON" if self.add_inertia else "inertia OFF"
        ax.set_title(
            f"ESL Strain Energy History  [{inertia_str} | {self.n_windows} windows | top-{self.n_top}]",
            fontsize=11
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain Energy (J)")
        ax.legend(fontsize=9)
        fig.tight_layout()

        iter_tag  = f"_iter{self.iteration:03d}" if self.iteration >= 0 else ""
        base_dir  = self.out_dir if self.out_dir else Path(self.csv_path).parent
        out_path  = base_dir / f"esl_se_report{iter_tag}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"     -> ESL SE 리포트 저장: {out_path}")

    def _cleanup_master_nodes(self) -> None:
        """동해석용으로 임시 추가한 마스터 노드와 RBE3를 제거합니다."""
        for mnid in range(900000, 900004):
            if mnid in self.model.nodes:
                del self.model.nodes[mnid]
        self.model.rbe3s = {k: v for k, v in self.model.rbe3s.items() if k < 900000}

    def _build_snapshots(self) -> List[Tuple[WHTLoadCase, float]]:
        """ESL 케이스에 안정화 BC를 추가하고 (lc, esl_weight) 리스트를 반환합니다."""
        stab_nid  = self._bot_groups[0][1][0]
        snapshots = []
        for lc in self._esl_cases:
            lc.add_bc([stab_nid], dofs=(0, 1))
            snapshots.append((lc, self.esl_weight))
        _section("ESL 스냅샷 확정", "ESL-3")
        _row("시간분할 스냅샷 수", f"{len(snapshots)}개  (가중치 w={self.esl_weight})")
        _endsec()
        return snapshots

    def _build_element_peak_lc(self) -> Optional[WHTLoadCase]:
        """
        요소별 최대 SE 발생 시각의 내력벡터를 조합하여 등가 정하중 케이스를 생성합니다.

        각 요소 e에 대해:
          t*_e = argmax_t  ½ uₑ(t)ᵀ Kₑ uₑ(t)
          f*_e = Kₑ uₑ(t*_e)   (피크 내력벡터)

        전체 등가 하중:  F_eq = Σ_e Lₑᵀ f*_e  (전역 DOF 기준 조합)
        """
        jm, s_nids, n2i = self._dyn_solver._build_jaxsso_model()
        ndof = jm.ndof
        t_arr = self._dyn_res.t_saved

        # 전 시간 변위 행렬: (n_save, ndof)
        U = np.vstack([self._dyn_res.u[i].flatten()[:ndof]
                       for i in range(self._dyn_res.n_save)])

        F_eq = np.zeros(ndof)

        # 시각화용 데이터 수집
        elem_centroids : List[np.ndarray] = []
        elem_se_peak   : List[float]      = []
        elem_t_peak    : List[float]      = []
        elem_node_xy   : List[np.ndarray] = []  # (n_corner, 2)

        for eid, elem in self.model.elements.items():
            etype = elem.type.upper()
            if etype not in ('QUAD4', 'QUAD', 'TRIA3', 'TRIA'):
                continue

            elem_dofs: List[int] = []
            valid = True
            for nid in elem.node_ids:
                if nid not in n2i:
                    valid = False
                    break
                idx = n2i[nid]
                elem_dofs.extend(range(idx * 6, idx * 6 + 6))
            if not valid:
                continue
            dof_idx = np.array(elem_dofs, dtype=int)

            prop = self.model.properties.get(elem.pid)
            mat  = self.model.materials.get(prop.mid) if prop else None
            t_th = prop.t   if prop else 1.0
            E    = mat.E    if mat  else 210000.0
            nu   = mat.nu   if mat  else 0.3
            coords = [np.array([self.model.nodes[nid].x,
                                self.model.nodes[nid].y,
                                self.model.nodes[nid].z])
                      for nid in elem.node_ids]

            if etype in ('QUAD4', 'QUAD'):
                Ke = _element_K_mitc4_plus(coords[0], coords[1], coords[2], coords[3], t_th, E, nu)
            else:
                Ke = _element_K_tria3(coords[0], coords[1], coords[2], t_th, E, nu)

            Ue     = U[:, dof_idx]
            se_e   = 0.5 * np.einsum('ti,ij,tj->t', Ue, Ke, Ue)
            t_star = int(np.argmax(se_e))
            F_eq[dof_idx] += Ke @ Ue[t_star]

            # 시각화 데이터
            centroid = np.mean(coords, axis=0)
            elem_centroids.append(centroid)
            elem_se_peak.append(float(se_e[t_star]))
            elem_t_peak.append(float(t_arr[t_star]))
            elem_node_xy.append(np.array([[c[0], c[1]] for c in coords]))

        # WHTLoadCase 생성 (비영 노드만 force 등록)
        lc = WHTLoadCase(name="ElementPeakSE_ESL")
        for nid, idx in n2i.items():
            fvec = F_eq[idx * 6:(idx + 1) * 6]
            if np.any(np.abs(fvec) > 1e-10):
                lc.forces.append(WHTForceEntry(nid, tuple(float(v) for v in fvec)))

        # 등가 노드력 크기 {nid: |F|}
        f_eq_mag: dict = {}
        for nid, idx in n2i.items():
            fvec = F_eq[idx * 6:(idx + 1) * 6]
            mag = float(np.linalg.norm(fvec))
            if mag > 1e-10:
                f_eq_mag[nid] = mag

        # 안정화 BC (시간분할 ESL과 동일)
        stab_nid = self._bot_groups[0][1][0]
        lc.add_bc([stab_nid], dofs=(0, 1))

        _row("처리 요소 / 비영 노드", f"{len(elem_se_peak):,}개  /  {len(lc.forces):,}개")

        self._plot_peak_lc_report(
            elem_node_xy  = elem_node_xy,
            elem_se_peak  = np.array(elem_se_peak),
            elem_t_peak   = np.array(elem_t_peak),
            f_eq_mag      = f_eq_mag,
        )
        return lc

    def _plot_peak_lc_report(
        self,
        elem_node_xy : List[np.ndarray],
        elem_se_peak : np.ndarray,
        elem_t_peak  : np.ndarray,
        f_eq_mag     : dict,
    ) -> None:
        """요소별 피크 SE / 피크 시각 / 등가 노드력 3패널 그래프를 PNG로 저장합니다."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"Element Peak ESL Report  [inertia={'ON' if self.add_inertia else 'OFF'}]",
            fontsize=13, fontweight='bold'
        )

        def _mesh_patch_collection(ax, values, cmap_name, label):
            """요소 폴리곤 패치 컬렉션을 그리고 colorbar를 반환."""
            norm  = Normalize(vmin=np.min(values), vmax=np.max(values))
            cmap  = matplotlib.colormaps[cmap_name]
            patches = []
            colors  = []
            for xy, val in zip(elem_node_xy, values):
                poly = mpatches.Polygon(xy, closed=True)
                patches.append(poly)
                colors.append(cmap(norm(val)))
            col = PatchCollection(patches, facecolors=colors, edgecolors='none', linewidths=0)
            ax.add_collection(col)
            # 축 범위
            all_xy = np.vstack(elem_node_xy)
            pad_x  = (all_xy[:, 0].max() - all_xy[:, 0].min()) * 0.03
            pad_y  = (all_xy[:, 1].max() - all_xy[:, 1].min()) * 0.03
            ax.set_xlim(all_xy[:, 0].min() - pad_x, all_xy[:, 0].max() + pad_x)
            ax.set_ylim(all_xy[:, 1].min() - pad_y, all_xy[:, 1].max() + pad_y)
            ax.set_aspect('equal')
            ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.02)
            cb.set_label(label, fontsize=9)
            return col

        # ── 패널 1: 요소별 피크 변형에너지 ─────────────────────────────────
        ax1 = axes[0]
        _mesh_patch_collection(ax1, elem_se_peak, 'hot_r', 'Peak SE (J)')
        ax1.set_title("Element Peak Strain Energy  SE*_e", fontsize=10)

        # ── 패널 2: 피크 발생 시각 분포 ────────────────────────────────────
        ax2 = axes[1]
        _mesh_patch_collection(ax2, elem_t_peak, 'coolwarm', 'Peak Time (s)')
        ax2.set_title("Peak SE Time  t*_e  (s)", fontsize=10)

        # ── 패널 3: 등가 노드력 크기 (상위 30개 바 차트) ────────────────────
        ax3 = axes[2]
        if f_eq_mag:
            sorted_nids = sorted(f_eq_mag, key=f_eq_mag.get, reverse=True)
            top_n  = min(30, len(sorted_nids))
            top_nids = sorted_nids[:top_n]
            top_vals = [f_eq_mag[n] for n in top_nids]
            # 노드 XY 위치를 레이블 대신 색으로 — 바 위에 nid 표시는 생략, 인덱스만
            ax3.bar(range(top_n), top_vals, color='steelblue', edgecolor='none')
            ax3.set_xticks(range(top_n))
            ax3.set_xticklabels([str(n) for n in top_nids],
                                rotation=75, fontsize=6, ha='right')
            ax3.set_xlabel("Node ID", fontsize=9)
            ax3.set_ylabel("|F_eq|  (N or N·mm)", fontsize=9)
            ax3.set_title(f"Equivalent Nodal Force Magnitude\n(Top {top_n} nodes)", fontsize=10)
            # 최대값 주석
            ax3.annotate(f"max={top_vals[0]:.2e}",
                         xy=(0, top_vals[0]), xytext=(2, top_vals[0] * 0.95),
                         fontsize=8, color='firebrick',
                         arrowprops=dict(arrowstyle='->', color='firebrick', lw=1.0))

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        iter_tag  = f"_iter{self.iteration:03d}" if self.iteration >= 0 else ""
        base_dir  = self.out_dir if self.out_dir else Path(self.csv_path).parent
        out_path  = base_dir / f"esl_peak_report{iter_tag}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"     -> 피크 ESL 리포트 저장: {out_path}")


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
        stamp = datetime.now().strftime("D%Y%m%d_%H%M%S")
        self.out_dir = Path(__file__).resolve().parent.parent / "results" / stamp
        self.out_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.out_dir / "run.log"
        tee = _Tee(sys.stdout, log_path)
        sys.stdout = tee
        try:
            print(f"\n  {'═'*(_W-2)}")
            print(_hdr("WHT  CSV Dynamic Response Analysis  [Mode B]"))
            print(f"  {'═'*(_W-2)}")
            print(f"  결과 디렉토리 : {self.out_dir}")
            print(f"  로그 파일     : {log_path}")
            print(f"  {'─'*(_W-2)}\n")
            self._build_mesh()
            self._find_corners_and_load()
            if getattr(self.cfg, 'add_inertia', False):
                total_mass, accels, n_inertia = _apply_inertia_loads(
                    self._df, self._time_arr,
                    self.node_db, self.model, self._load_groups
                )
                _section("관성 하중 (F = -m·a) 인가", "B-inertia")
                _row("총 섀시 질량", f"{total_mass*1e6:.2f} kg  ({total_mass:.4e} tonne)")
                _row("관성 하중 인가 노드", f"{n_inertia:,}개")
                _endsec()
            self._run_dynamic()
            self._export()
        finally:
            sys.stdout = tee._stream
            tee.close()
        self._visualize()

    # ── 단계별 메서드 ────────────────────────────────────────────────────────

    def _build_mesh(self) -> None:
        _section("메시 생성", "1")
        self.model, self.node_db = _build_tray(
            width       = self.cfg.tray_width,
            length      = self.cfg.tray_length,
            height      = self.cfg.tray_height,
            mesh_xy     = self.cfg.mesh_xy,
            mesh_z      = self.cfg.mesh_z,
            draft_angle = self.cfg.draft_angle,
            E           = self.cfg.mat_E,
            nu          = self.cfg.mat_nu,
            rho         = self.cfg.mat_rho,
            t           = self.cfg.mat_t,
        )
        _row("섀시 치수 (W × L × H)",
             f"{self.cfg.tray_width:.0f} × {self.cfg.tray_length:.0f} × {self.cfg.tray_height:.0f} mm")
        _row("재료", f"E={self.cfg.mat_E:.0f} MPa  ν={self.cfg.mat_nu}  "
                     f"ρ={self.cfg.mat_rho:.2e} t/mm³  t={self.cfg.mat_t} mm")
        _row("메시 크기 (XY / Z)", f"{self.cfg.mesh_xy:.0f} / {self.cfg.mesh_z:.0f} mm")
        _row("노드 / 요소", f"{len(self.node_db):,}  /  {len(self.model.elements):,}")
        _endsec()

    def _find_corners_and_load(self) -> None:
        """CSV 읽기 → 코너 탐색 → 마스터 노드 + RBE3 + SPCD 구성."""
        # CSV 로드 (t_start: CLI 인자 우선, 없으면 헤더, 없으면 0.0)
        t_start_arg = getattr(self.cfg, 't_start', None)
        t_end_arg   = getattr(self.cfg, 't_end',   None)
        print(f"\n [2] CSV 로드: {self.cfg.pos_data}")
        df, time_arr, header = _load_csv(self.cfg.pos_data, t_start_arg, t_end_arg)
        t_start_used = t_start_arg if t_start_arg is not None else (header.get('start_time') or 0.0)
        dt_val  = float(time_arr[1] - time_arr[0]) if len(time_arr) > 1 else self.cfg.dt
        T_total = float(time_arr[-1])
        t_end_str = f"{t_end_arg}s" if t_end_arg is not None else "end"
        print(f"     t=[{t_start_used}s ~ {t_end_str}], 프레임={len(time_arr)}, "
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
        self._df          = df
        self._time_arr    = time_arr

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
        paraview_dir = self.out_dir / "paraview"
        paraview_dir.mkdir(parents=True, exist_ok=True)
        hdf_path = str(paraview_dir / "dynamic_result.hdf")
        VTKHDFExporter().export(self.wht_data, hdf_path)
        _section("결과 파일 저장", "6")
        _row("ParaView HDF", hdf_path)
        _endsec()

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
        self.out_dir: Optional[Path]                = None

    def run(self) -> None:
        """파이프라인 전체 실행."""
        stamp = datetime.now().strftime("D%Y%m%d_%H%M%S")
        self.out_dir = Path(__file__).resolve().parent.parent / "results" / stamp
        self.out_dir.mkdir(parents=True, exist_ok=True)

        log_path = self.out_dir / "run.log"
        tee = _Tee(sys.stdout, log_path)
        sys.stdout = tee
        try:
            print(f"\n  {'═'*(_W-2)}")
            print(_hdr("WHT  Industrial Topography Optimization"))
            print(f"  {'═'*(_W-2)}")
            print(f"  결과 디렉토리 : {self.out_dir}")
            print(f"  로그 파일     : {log_path}")
            print(f"  {'─'*(_W-2)}\n")
            self._build_mesh()
            static_cases, esl_provider = self._prepare_load_cases()
            self._build_solver(static_cases, esl_provider)
            self._run_optimizer()
            self._discretize()
            self._apply_shape()
            self._export()
        finally:
            sys.stdout = tee._stream
            tee.close()
        self._visualize()

    # ── 단계별 메서드 ────────────────────────────────────────────────────────

    def _build_mesh(self) -> None:
        _section("메시 생성", "1")
        self.model, self.node_db = _build_tray(
            width       = self.cfg.tray_width,
            length      = self.cfg.tray_length,
            height      = self.cfg.tray_height,
            mesh_xy     = self.cfg.mesh_xy,
            mesh_z      = self.cfg.mesh_z,
            draft_angle = self.cfg.draft_angle,
            E           = self.cfg.mat_E,
            nu          = self.cfg.mat_nu,
            rho         = self.cfg.mat_rho,
            t           = self.cfg.mat_t,
        )
        _row("섀시 치수 (W × L × H)",
             f"{self.cfg.tray_width:.0f} × {self.cfg.tray_length:.0f} × {self.cfg.tray_height:.0f} mm")
        _row("재료", f"E={self.cfg.mat_E:.0f} MPa  ν={self.cfg.mat_nu}  "
                     f"ρ={self.cfg.mat_rho:.2e} t/mm³  t={self.cfg.mat_t} mm")
        _row("메시 크기 (XY / Z)", f"{self.cfg.mesh_xy:.0f} / {self.cfg.mesh_z:.0f} mm")
        _row("노드 / 요소", f"{len(self.node_db):,}  /  {len(self.model.elements):,}")
        _endsec()

    def _prepare_load_cases(self):
        """
        하중 케이스와 ESL provider를 준비합니다.

        Returns
        -------
        (static_cases, esl_provider)
          static_cases : List[(WHTLoadCase, w)] | None
              정적 하중 케이스. None이면 solver 내부 자동 생성.
          esl_provider : Callable[[int], list] | None
              동적 ESL 추출 콜백. --iterative-esl 시 각 이터레이션마다 호출.
              None이면 최적화 루프 내 동적 해석 없음.

        모드 A  (dynamic_opts 없음): (None, None)
        모드 C/D + 반복 추출 (기본, --iterative-esl): (static만, provider)
        모드 C/D + 1회 추출 (--no-iterative-esl): (static+ESL 병합, None)
        """
        if not getattr(self.cfg, 'dynamic_opts', None):
            return None, None

        # CSV 엔트리 파싱: "path" / "path,t_start" / "path,t_start,t_end"
        def _parse_entry(s: str):
            parts = [p.strip() for p in s.split(',')]
            csv_path = parts[0]
            t_start  = float(parts[1]) if len(parts) > 1 else None
            t_end    = float(parts[2]) if len(parts) > 2 else None
            return csv_path, t_start, t_end

        entries = [_parse_entry(e) for e in self.cfg.dynamic_opts]

        # --w-dynamic 가중치 목록: CSV 수보다 적으면 마지막 값으로 채움
        w_list_raw = list(self.cfg.w_dynamic) or [1.0]
        w_list = w_list_raw + [w_list_raw[-1]] * max(0, len(entries) - len(w_list_raw))

        if len(entries) > 1:
            print(f"\n [3] 동적 CSV 시나리오 {len(entries)}개:")
            for i, ((csv_path, t_start, t_end), w) in enumerate(zip(entries, w_list)):
                t_range = f"{t_start if t_start is not None else 'auto'} ~ {t_end if t_end is not None else 'end'}"
                print(f"     [{i+1}] {csv_path}  t=[{t_range}]s  w={w}")

        if getattr(self.cfg, 'no_static', False):
            static_cases = []
            print(f"\n [3] 정적 하중 케이스: 비활성 (--no-static).")
        else:
            w_bw = getattr(self.cfg, 'w_basic_weight', 1.0)
            raw_loads = self._loads()
            if w_bw != 0.0:
                # 자중 자동계산: 총 무게 * 스케일 * 입력 부호
                total_mass, W_N, x_cm, y_cm, z_cm, _, _ = self._calc_chassis_weight()
                scaled_loads = {
                    k: math.copysign(W_N * w_bw, v)
                    for k, v in raw_loads.items()
                }
                self._print_weight_report(total_mass, W_N, x_cm, y_cm, z_cm,
                                          scaled_loads)
            else:
                scaled_loads = raw_loads
                print(f"\n [3] --w-basic-weight 0: 입력 하중값 그대로 사용.")
            load_manager = StochasticLoadManager(self.model)
            static_cases = load_manager.get_load_cases(
                mesh_size_z=self.cfg.mesh_size,
                weights=self._weights(),
                loads=scaled_loads,
            )
            print(f"\n [3] 정적 하중 케이스 {len(static_cases)}개 구성 완료.")

        def _run_esl(iteration: int):
            all_snaps = []
            for (csv_path, t_start, t_end), w_dyn in zip(entries, w_list):
                snaps = ESLExtractor(
                    model        = self.model,
                    node_db      = self.node_db,
                    csv_path     = csv_path,
                    t_start      = t_start,
                    t_end        = t_end,
                    n_windows    = self.cfg.n_windows,
                    n_top        = self.cfg.n_top,
                    add_inertia  = self.cfg.add_inertia,
                    use_global_z = self.cfg.use_global_z,
                    esl_weight   = w_dyn,
                    w_peak       = self.cfg.w_peak,
                    iteration    = iteration,
                    out_dir      = self.out_dir,
                ).extract()
                all_snaps.extend(snaps)
            return all_snaps

        if getattr(self.cfg, 'iterative_esl', False):
            # ── 반복 추출 모드: provider를 solver에 전달 ──────────────────────
            def provider(iteration: int):
                print(f"\n [ESL-재추출] Iter {iteration} — 현재 비드 형상 기준 동적 해석")
                snaps = _run_esl(iteration)
                print(f"     -> Iter {iteration} ESL {len(snaps)}개 반환.")
                return snaps
            return static_cases, provider
        else:
            # ── 1회 추출 모드: 최적화 전 ESL을 미리 추출하여 정적 케이스에 병합 ──
            print(f"\n [3b] 동적 ESL 1회 추출 중...")
            dyn_snaps = _run_esl(iteration=-1)
            merged = static_cases + dyn_snaps
            print(f"\n [3b] 하중 케이스 병합: 정적 {len(static_cases)} + "
                  f"동적 ESL {len(dyn_snaps)} = 총 {len(merged)}개")
            return merged, None

    def _build_solver(self, load_cases: Optional[list], load_case_provider=None) -> None:
        print(f"\n [4] 최적화 준비 (h_max={self.cfg.bead_height}mm, "
              f"min_width={self.cfg.min_width}mm, "
              f"bead_area={self.cfg.bead_area*100:.0f}%)...")
        if load_case_provider is not None:
            print("     ESL 방식: [반복 추출] 매 이터레이션 현재 형상 기준 동해석 재실행.")
        else:
            print("     ESL 방식: [1회 추출] 최적화 전 고정 하중 케이스로 진행.")
        load_manager = StochasticLoadManager(self.model)
        freq_w, freq_f0 = getattr(self.cfg, 'freq_penalty', [0.0, 0.0])
        exclude_zones = _parse_exclude_zones(
            getattr(self.cfg, 'exclude_rect', None),
            getattr(self.cfg, 'exclude_poly', None),
        )
        self.solver  = WHTopographySolver(
            self.model, load_manager,
            bead_height_max   = self.cfg.bead_height,
            bead_height_ratio = self.cfg.bead_area,
            min_width         = self.cfg.min_width,
            draw_dir          = [float(v) for v in self.cfg.draw_dir.split(',')],
            weights           = self._weights(),
            mesh_size_z       = self.cfg.mesh_size,
            sym_x             = self.cfg.sym_x,
            bead_connect      = self.cfg.bead_connect,
            connect_gap       = self.cfg.connect_gap,
            bead_steps        = self.cfg.height_steps,
            load_cases        = load_cases,
            load_case_provider= load_case_provider,
            out_dir           = self.out_dir,
            normalize_obj     = self.cfg.normalize_obj,
            obj_type          = self.cfg.obj_type,
            obj_alpha         = self.cfg.obj_alpha,
            freq_weight       = freq_w,
            freq_target       = freq_f0,
            exclude_zones     = exclude_zones,
            n_workers         = self.cfg.n_workers,
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
                target=start_monitor_ui,
                args=(queue, stop_event, str(self.out_dir))
            )
            ui_process.start()
            callback = queue.put

        print(f" [4] 최적화 실행 (max_iter={self.cfg.iters})...")
        try:
            self.solver.solve(max_iter=self.cfg.iters, callback=callback, stop_event=stop_event)
        finally:
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
        # LS-DYNA .k 파일: out_dir 아래 저장 (--export 미지정 시 기본 파일명 사용)
        export_name = self.cfg.export or "final.k"
        export_path = Path(export_name)
        if not export_path.is_absolute() and self.out_dir:
            export_path = self.out_dir / export_path.name
        self.model.export_to_solver('lsdyna', str(export_path), reorder=True)
        print(f" [7] LS-DYNA .k 저장: {export_path}")

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
        """각 하중 케이스의 목적 함수 가중치 (nargs=2 인자의 첫 번째 값)."""
        return {
            "bending":       self.cfg.w_bending[0],
            "bending_xspan": self.cfg.w_bending_xspan[0],
            "bending_yspan": self.cfg.w_bending_yspan[0],
            "twisting":      self.cfg.w_twisting[0],
            "twisting_alt":  self.cfg.w_twisting_alt[0],
            "lifting":       self.cfg.w_lifting[0],
        }

    def _loads(self) -> dict:
        """각 하중 케이스의 적용 하중 크기 N (nargs=2 인자의 두 번째 값)."""
        return {
            "bending":       self.cfg.w_bending[1],
            "bending_xspan": self.cfg.w_bending_xspan[1],
            "bending_yspan": self.cfg.w_bending_yspan[1],
            "twisting":      self.cfg.w_twisting[1],
            "twisting_alt":  self.cfg.w_twisting_alt[1],
            "lifting":       self.cfg.w_lifting[1],
        }

    def _calc_chassis_weight(self) -> tuple:
        """
        루프형 질량 행렬로 샤시 총 무게·무게중심·지지점 반력 모멘트를 계산합니다.

        Returns
        -------
        total_mass : float   총 질량 [tonne]
        W_N        : float   총 자중 [N]   (total_mass * GRAVITY_MM)
        x_cm, y_cm, z_cm : float  무게중심 좌표 [mm]
        m_diag     : array   노드 질량 대각 벡터
        n2i        : dict    nid -> jaxsso index
        """
        temp = WHTDynamicSolver(self.model)
        jm, s_nids, n2i = temp._build_jaxsso_model()
        m_diag = temp._assemble_lumped_mass(jm, jm.ndof, s_nids, n2i)

        total_mass = 0.0
        mx = my = mz = 0.0
        for nid in self.node_db:
            ix = n2i.get(nid)
            if ix is None or nid >= 900_000:
                continue
            m_z = float(m_diag[ix * 6 + 2])
            if m_z <= 1e-12:
                continue
            total_mass += m_z
            x, y, z = self.node_db[nid]
            mx += m_z * x
            my += m_z * y
            mz += m_z * z

        if total_mass > 1e-12:
            x_cm = mx / total_mass
            y_cm = my / total_mass
            z_cm = mz / total_mass
        else:
            x_cm = y_cm = z_cm = 0.0

        W_N = total_mass * GRAVITY_MM
        return total_mass, W_N, x_cm, y_cm, z_cm, m_diag, n2i

    def _print_weight_report(self, total_mass: float, W_N: float,
                             x_cm: float, y_cm: float, z_cm: float,
                             scaled_loads: dict) -> None:
        """자중 계산 결과를 터미널에 상세 출력합니다."""
        all_xyz = np.array(list(self.node_db.values()))
        x_geo = (all_xyz[:, 0].min() + all_xyz[:, 0].max()) * 0.5
        y_geo = (all_xyz[:, 1].min() + all_xyz[:, 1].max()) * 0.5
        ecc_x = x_cm - x_geo
        ecc_y = y_cm - y_geo
        # 무게중심 편심에 의한 지지점 반력 모멘트 추정
        # (단순지지 4코너 가정: Mx = W * ecc_y, My = W * ecc_x)
        Mx_N_mm = W_N * ecc_y   # Y 편심 → X 축 모멘트
        My_N_mm = W_N * ecc_x   # X 편심 → Y 축 모멘트

        sep = "─" * 68
        print(f"\n  {sep}")
        print(f"  [자중 하중 자동계산]  (--w-basic-weight = {getattr(self.cfg, 'w_basic_weight', 1.0):.3g})")
        print(f"  {sep}")
        print(f"  {'항목':<28} {'값':>36}")
        print(f"  {'─'*28} {'─'*36}")
        print(f"  {'중력가속도':<28} {GRAVITY_MM:>30.1f} mm/s²")
        print(f"  {'총 질량':<28} {total_mass*1e6:>28.4f} kg  "
              f"({total_mass:.4e} tonne)")
        print(f"  {'총 자중 (Z방향 합력)':<28} {W_N:>28.2f} N  "
              f"({W_N/1000:.4f} kN)")
        print(f"  {'무게중심 X':<28} {x_cm:>33.2f} mm")
        print(f"  {'무게중심 Y':<28} {y_cm:>33.2f} mm")
        print(f"  {'무게중심 Z':<28} {z_cm:>33.2f} mm")
        print(f"  {'기하 중심 X':<28} {x_geo:>33.2f} mm")
        print(f"  {'기하 중심 Y':<28} {y_geo:>33.2f} mm")
        print(f"  {'X 편심 (x_cm - x_geo)':<28} {ecc_x:>33.2f} mm")
        print(f"  {'Y 편심 (y_cm - y_geo)':<28} {ecc_y:>33.2f} mm")
        print(f"  {'편심 X축 모멘트 Mx=W*ecc_y':<28} {Mx_N_mm:>26.2f} N·mm  "
              f"({Mx_N_mm/1000:.2f} N·m)")
        print(f"  {'편심 Y축 모멘트 My=W*ecc_x':<28} {My_N_mm:>26.2f} N·mm  "
              f"({My_N_mm/1000:.2f} N·m)")
        print(f"  {sep}")
        print(f"  [적용 하중 (자중 스케일 적용 후)]")
        print(f"  {'─'*28} {'─'*36}")
        case_labels = {
            "bending":       "중앙 굽힘",
            "bending_xspan": "X스팬 굽힘",
            "bending_yspan": "Y스팬 굽힘",
            "twisting":      "대각 비틀림",
            "twisting_alt":  "반전 비틀림",
            "lifting":       "4코너 리프팅",
        }
        for k, label in case_labels.items():
            F = scaled_loads.get(k, 0.0)
            print(f"  {label:<28} {F:>32.2f} N  ({F/1000:.4f} kN)")
        print(f"  {sep}\n")


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

  섀시 크기·재료 변경:
    python wht_topo/run_topo.py \\
      --tray-width 2000 --tray-length 1400 --tray-height 40 \\
      --mat-E 70000 --mat-nu 0.33 --mat-rho 2.7e-9 --mat-t 2.0
    -> 알루미늄 합금 섀시 (E=70 GPa, ρ=2.7 g/cm³)

[모드 B] CSV 단독 동적 응답 해석 (최적화 생략)

  [권장] 실측 CSV + t_start 자동 적용
    python wht_topo/run_topo.py --pos-data wht_topo/structural_dynamics.csv
    -> CSV 헤더 "# start_time" 값 자동 적용 / 결과: results/D날짜_시간/paraview/

  t_start 명시 + 감쇠비 조정
    python wht_topo/run_topo.py \\
      --pos-data wht_topo/structural_dynamics.csv --t-start 1.6 --zeta 0.03
    -> 1.6s 이전 준정적 구간 제외, 감쇠비 3% 적용

  시각화 생략 (HDF만 저장, 헤드리스)
    python wht_topo/run_topo.py \\
      --pos-data wht_topo/structural_dynamics.csv --no-viz

[모드 C] 동적 충격 통합 최적화 — 반복 ESL (기본)
  (정적 하중 + 동적 ESL 병합 / 매 이터레이션 ESL 재추출)
  3개 충격 시나리오: rear(0.4~0.6s) / c125(1.5~1.8s) / c235(1.5~1.8s)

  [권장] 복수 시나리오 반복 ESL + 관성 하중
    python wht_topo/run_topo.py \\
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \\
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \\
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \\
      --w-dynamic 1.0 1.0 1.0 --add-inertia
    -> 정적 케이스 유지, 매 이터레이션 3개 CSV 동해석 재실행 → ESL 갱신
    -> 결과: results/D날짜_시간/{paraview/, esl_se_report_iter000.png, ...}

  [권장] 시간분할 ESL + 요소별 피크 ESL 병용
    python wht_topo/run_topo.py \\
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \\
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \\
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \\
      --w-dynamic 1.0 1.0 1.0 --w-peak 0.5 --add-inertia
    -> 시나리오별 시간분할 10개(w=1.0) + 피크 1개(w=0.5) 이터레이션마다 재추출

  동적 ESL만 사용 (정적 하중 비활성)
    python wht_topo/run_topo.py \\
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \\
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \\
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \\
      --no-static --w-dynamic 1.0 --w-peak 0.5 --add-inertia
    -> 굽힘·비틀림·리프팅 정적 케이스 전체 제외

  피크 ESL만 단독 (시간분할 비활성)
    python wht_topo/run_topo.py \\
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \\
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \\
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \\
      --w-dynamic 0.0 --w-peak 1.0 --add-inertia

  ESL 고밀도 탐색
    python wht_topo/run_topo.py \\
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \\
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \\
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \\
      --w-dynamic 1.0 --add-inertia --n-windows 50 --n-top 15 --w-peak 0.5

[모드 C' — 비교용] 1회 ESL 추출 후 고정 (--no-iterative-esl)

  반복 ESL 결과와 비교 (동일 조건, ESL만 고정)
    python wht_topo/run_topo.py \\
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \\
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \\
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \\
      --no-iterative-esl --w-dynamic 1.0 --w-peak 0.5 --add-inertia
    -> 최적화 전 ESL 1회 추출, 이후 고정 / 계산 비용 낮음

[모드 D] 산업용 고신뢰성 완전 제약 설계

  [권장] 반복 ESL + 관성 하중 + 대칭 + 비드 연결 + 이산화
    python wht_topo/run_topo.py \\
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \\
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \\
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \\
      --w-dynamic 1.0 --w-peak 0.5 --add-inertia \\
      --sym-x --bead-connect --height-steps 2

  동적 ESL 전용 (정적 비활성)
    python wht_topo/run_topo.py --n-workers 4 --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" "wht_topo/structural_dynamics_c125.csv,1.5,1.8" "wht_topo/structural_dynamics_c235.csv,1.5,1.8"  --w-dynamic 1.0 --w-peak 1.0 --add-inertia  --sym-x --bead-connect --height-steps 2 --min-width 150 
    
  헤드리스 서버 실행
    python wht_topo/run_topo.py \\
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \\
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \\
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \\
      --w-dynamic 1.0 --w-peak 0.5 --add-inertia \\
      --n-windows 50 --n-top 20 \\
      --sym-x --bead-connect --height-steps 2 \\
      --no-gui --no-viz

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
옵션 상세 설명
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[섀시 형상]
--tray-width/length/height  섀시 외형 치수 mm (기본: 1800×1200×35)
--mesh-xy / --mesh-z        XY면·Z방향 메시 크기 mm (기본: 40 / 10)
--draft-angle               측면 드래프트 각도 deg (기본: 25)

[재료 물성]
--mat-E    탄성계수 MPa (기본: 210000, 강재)
--mat-nu   포아송비    (기본: 0.3)
--mat-rho  밀도 tonne/mm³ (기본: 7.85e-9, 강재)
--mat-t    기본 판 두께 mm (기본: 1.2)

[CSV 입력]
--pos-data      CSV 경로 → 모드 B(동적 단독 해석) 실행.
                형식: # 주석 헤더(코너 기준 좌표·start_time) + Frame,Time,C1_X...C8_Z
                코너 C5~C8 좌표를 헤더에 기재하면 FEM 노드를 자동 탐색.

--t-start       CSV 분석 시작 시점 s (모드 B 전용).
                미지정 → CSV 헤더 start_time 자동 적용 → 없으면 0.0.
--t-end         CSV 분석 종료 시점 s (모드 B 전용).
                미지정 → CSV 마지막 프레임까지 사용.

--dynamic-opts  시간 범위 지정 형식 (콤마 구분):
                  "CSV경로"                → t_start: 헤더/0.0,  t_end: 마지막 프레임
                  "CSV경로,시작(s)"        → t_start 지정,        t_end: 마지막 프레임
                  "CSV경로,시작(s),종료(s)"→ t_start/t_end 모두 지정
                복수 충격 시나리오 (공백 구분):
                  --dynamic-opts "drop.csv,1.6,3.0" "bump.csv,0.5,2.5" "sine.csv"

[ESL 추출 방식]
--iterative-esl    (기본: 활성) 매 이터레이션 현재 비드 형상에서 동해석 재실행.
                   구조 진화에 따른 동적 응답 변화를 하중 케이스에 반영.
                   이터레이션별 출력: results/.../esl_*_iterNNN.png

--no-iterative-esl ESL을 최적화 전 1회만 추출하여 고정.
                   계산 비용 낮음. --iterative-esl 결과와 비교 시 사용.

[하중 케이스 구성]
--no-static     정적 하중 케이스(굽힘·비틀림·리프팅) 전체 비활성.
                --dynamic-opts 병용 시에만 유효. 동적 ESL만 사용할 때 지정.

--w-bending/xspan/yspan  W F_N  중앙·X스팬·Y스팬 굽힘 가중치·하중 (w=0 → 제외)
--w-twisting/alt         W F_N  대각·반전 비틀림 가중치·하중
--w-lifting              W F_N  4코너 리프팅 가중치·하중

--add-inertia   관성 하중 F=-ma 전 노드 분포 인가.
                낙하·충격 하중 케이스에 반드시 사용.

--w-dynamic     시간분할 ESL 가중치 (기본: 1.0).
                복수 CSV 지정 시 CSV별 가중치를 공백 구분으로 입력:
                  --w-dynamic 1.0 0.8 0.5
                CSV 수보다 가중치 수가 적으면 마지막 값으로 나머지를 채움.
                출력: esl_se_report[_iterNNN].png
                      SE 이력 + 윈도우 분할 + 선택 시점 마킹

--w-peak        요소별 최대 SE 피크 ESL 가중치 (기본: 0.0=비활성 / 0.5~1.0 권장).
                각 요소 SE 최대 시각의 내력을 조합한 단일 보수적 케이스.
                출력: esl_peak_report[_iterNNN].png
                      요소 SE 분포 + 피크 발생 시각 + 등가 노드력 상위 30

--n-windows     SE 이력 분할 창 수 (기본: 30 / 충격→20, 장시간 진동→50)
--n-top         시간분할 ESL 최종 선정 개수 (기본: 10 / 고정밀→15~20)

[최적화 제약]
--sym-x / --no-sym-x        X축 좌우 대칭 (기본: 활성)
--bead-connect / --no-bead-connect  비드 형태학적 연결 (기본: 활성)
--connect-gap               비드 연결 최대 갭 mm (기본: 120)
--height-steps              비드 이산화 단계 (기본: 2 → {0, h_max})
--bead-height               최대 비드 높이 mm (기본: 10)
--bead-area                 비드 점유 면적 비율 0~1 (기본: 0.35)
--min-width                 최소 비드 폭 mm (기본: 30)

[비드 배제 영역]
--exclude-rect CX,CY,W,H    중심(CX,CY) 기준 W×H mm 사각형 영역 내 비드 생성 금지.
                            반복 지정으로 복수 영역 추가 가능:
                              --exclude-rect 500,300,100,80 --exclude-rect 200,400,60,60

--exclude-poly X1,Y1,X2,Y2,...
                            꼭짓점 좌표 나열(mm)로 정의한 임의 다각형 영역 내 비드 생성 금지.
                            꼭짓점은 순서대로 나열 (자동 닫힘), 최소 3개 꼭짓점(6개 좌표):
                              --exclude-poly 100,200,300,200,300,350,100,350
                            복수 영역: --exclude-poly "..." --exclude-poly "..."

  사용 예)
    구멍/마운팅 보스 주변 제외 (사각형 2개):
      --exclude-rect 450,250,120,120 --exclude-rect 1350,250,120,120

    장공/슬롯 주변 제외 (다각형):
      --exclude-poly 600,400,900,400,900,500,600,500

[출력]
결과 디렉토리: results/D날짜_시간/ (실행 시작 시 자동 생성)
  paraview/iter_NNN.hdf  — 이터레이션별 변위·응력·고유모드 (ParaView)
  esl_se_report_iterNNN.png   — 시간분할 ESL SE 이력 리포트
  esl_peak_report_iterNNN.png — 요소별 피크 ESL 리포트
  final.k (또는 --export 경로) — LS-DYNA 최적 비드 패턴

--no-gui   모니터링 GUI 비활성 (헤드리스 서버)
--no-viz   최종 3D 시각화 생략

[목적함수 옵션]
--normalize-obj
    각 케이스를 Iter 0 컴플라이언스로 정규화: f = Σ w_i·(C_i/C_i0).
    하중 크기가 서로 다른 정적·동적 케이스 혼재 시 권장.

--obj-type {sum|max|sum+max}  (기본: sum)
    sum     : f = Σ w_i·C_i  (또는 normalize-obj 시 Σ w_i·C_i/C_i0)
    max     : f = (1/α)·log(Σ exp(α·w_i·C_i/C_i0))   softmax 최악 케이스
    sum+max : f = 0.5·f_sum + 0.5·f_max               평균+최악 균형

--obj-alpha α  (기본: 10.0, obj-type=max/sum+max 시)
    클수록 hard-max (가장 나쁜 케이스만 반영).
    작을수록 soft 평균. 통상 5~50 범위 사용.

--freq-penalty W F0_HZ
    고유진동수 패널티: P = W·max(0,F0-f₁)²/F0²
    위 목적함수 어디에도 추가 가능.
    민감도는 첫 탄성 모드 형상 φ₁로 계산 (JAX vmap 재사용, 추가 FEA 없음):
      df₁/dh ≈ φ₁ᵀ(∂K/∂h)φ₁ / (4π²f₁)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    )

    # 섀시 형상 파라미터
    g = parser.add_argument_group("섀시 형상 (mm)")
    g.add_argument("--tray-width",   type=float, default=TRAY_W,      help=f"섀시 폭 mm (기본: {TRAY_W})")
    g.add_argument("--tray-length",  type=float, default=TRAY_L,      help=f"섀시 길이 mm (기본: {TRAY_L})")
    g.add_argument("--tray-height",  type=float, default=TRAY_H,      help=f"섀시 높이 mm (기본: {TRAY_H})")
    g.add_argument("--mesh-xy",      type=float, default=MESH_XY,     help=f"XY면 메시 크기 mm (기본: {MESH_XY})")
    g.add_argument("--mesh-z",       type=float, default=MESH_Z,      help=f"Z방향 메시 크기 mm (기본: {MESH_Z})")
    g.add_argument("--draft-angle",  type=float, default=DRAFT_ANGLE, help=f"측면 드래프트 각도 deg (기본: {DRAFT_ANGLE})")

    # 재료 물성
    g = parser.add_argument_group("재료 물성")
    g.add_argument("--mat-E",   type=float, default=MAT['E'],   help=f"탄성계수 MPa (기본: {MAT['E']})")
    g.add_argument("--mat-nu",  type=float, default=MAT['nu'],  help=f"포아송비 (기본: {MAT['nu']})")
    g.add_argument("--mat-rho", type=float, default=MAT['rho'], help=f"밀도 tonne/mm³ (기본: {MAT['rho']:.2e})")
    g.add_argument("--mat-t",   type=float, default=MAT['t'],   help=f"기본 판 두께 mm (기본: {MAT['t']})")

    # 최적화 기본 제어
    g = parser.add_argument_group("최적화 기본 설정")
    g.add_argument("--iters",       type=int,   default=15,   help="최대 반복 횟수 (기본: 15)")
    g.add_argument("--bead-height", type=float, default=10.0, help="최대 비드 높이 mm (기본: 10.0)")
    g.add_argument("--min-width",   type=float, default=30.0, help="최소 비드 폭 mm (기본: 30.0)")
    g.add_argument("--bead-area",   type=float, default=0.30, help="비드 점유 면적 비율 0~1 (기본: 0.35)")

    # 목적함수 옵션
    g = parser.add_argument_group("목적함수 옵션")
    g.add_argument("--normalize-obj",    action="store_true", default=False,
                   help="케이스별 초기 컴플라이언스로 정규화 (Σ w_i·C_i/C_i0). "
                        "하중 크기가 달라도 케이스 간 균등 반영.")
    g.add_argument("--no-normalize-obj", action="store_false", dest="normalize_obj",
                   help="정규화 비활성 (기본)")
    g.add_argument("--obj-type",    choices=["sum", "max", "sum+max"], default="sum+max",
                   help="목적함수 유형: sum=가중합(기본), max=softmax 최악케이스, "
                        "sum+max=0.5·sum+0.5·softmax")
    g.add_argument("--obj-alpha",   type=float, default=10.0,
                   help="softmax 온도 (--obj-type max/sum+max 시). "
                        "클수록 hard-max에 근접 (기본: 10.0)")
    g.add_argument("--freq-penalty", type=float, nargs=2, default=[0.0, 0.0],
                   metavar=("W", "F0_HZ"),
                   help="고유진동수 패널티: W·(max(0,F0-f1)/F0)². "
                        "예: --freq-penalty 5.0 50  → f1 < 50Hz 시 패널티 부과")

    # 병렬 해석 옵션
    g = parser.add_argument_group("병렬 해석 옵션")
    g.add_argument("--n-workers", type=int, default=4,
                   help="하중 케이스 병렬 해석 스레드 수 (기본: 4). "
                        "케이스 수보다 크게 지정해도 케이스 수로 자동 제한. "
                        "BLAS 멀티스레드와 충돌 방지를 위해 CPU 코어 수의 절반 권장.")

    # 비드 형상 및 제조 제약
    g = parser.add_argument_group("비드 형상 및 제조 제약")
    g.add_argument("--sym-x",          action="store_true", default=True,  help="좌우 대칭 활성화 (기본: 활성)")
    g.add_argument("--no-sym-x",       action="store_false", dest="sym_x", help="좌우 대칭 해제")
    g.add_argument("--bead-connect",   action="store_true", default=True,  help="비드 자동 연결 활성화 (기본: 활성)")
    g.add_argument("--no-bead-connect",action="store_false", dest="bead_connect", help="비드 연결 비활성화")
    g.add_argument("--connect-gap",    type=float, default=120.0, help="비드 연결 최대 갭 mm (기본: 120.0)")
    g.add_argument("--height-steps",   type=int,   default=1,    help="비드 이산화 단계 (기본: 2 → {0, h_max})")
    g.add_argument("--draw-dir", type=str, default="0,0,-1",
                   help="비드 돌출 방향 (쉼표 구분, 기본: 0,0,-1 = 아래). 예: 0,0,1  또는  0,0,-1")
    g.add_argument("--exclude-rect", type=str, action='append', default=None,
                   metavar="CX,CY,W,H",
                   help="비드 배제 사각형 영역 (중심 X,Y + 가로W 세로H, mm). "
                        "여러 영역: --exclude-rect 500,300,100,80 --exclude-rect 200,400,60,60")
    g.add_argument("--exclude-poly", type=str, action='append', default=None,
                   metavar="X1,Y1,X2,Y2,...",
                   help="비드 배제 다각형 영역 (꼭짓점 XY 좌표 나열, mm). "
                        "여러 영역: --exclude-poly 100,200,150,200,150,250,100,250")

    # CSV 단독 동적 해석 (모드 B)
    g = parser.add_argument_group("CSV 단독 동적 응답 해석 (모드 B, 최적화 생략)")
    g.add_argument("--pos-data",  type=str,   default=None,  help="CSV 경로: 지정 시 동적 해석만 실행")
    g.add_argument("--t-start",   type=float, default=None,
                   help="분석 시작 시점 s (기본: CSV 헤더 start_time 자동 적용)")
    g.add_argument("--t-end",     type=float, default=None,
                   help="분석 종료 시점 s (기본: CSV 마지막 프레임)")
    g.add_argument("--dt",        type=float, default=1e-4,  help="적분 시간 스텝 s (기본: 1e-4)")
    g.add_argument("--zeta",      type=float, default=0.02,  help="Rayleigh 감쇠비 (기본: 0.02)")
    g.add_argument("--corner-r",  type=float, default=150.0, help="코너 탐색 반경 mm (fallback 전용)")

    # 동적 ESL 통합 최적화 (모드 C/D)
    g = parser.add_argument_group("동적 ESL 통합 최적화 (모드 C/D)")
    g.add_argument("--dynamic-opts", type=str, nargs='+', default=None,
                   metavar="CSV[,T_START[,T_END]]",
                   help="'CSV경로' / 'CSV경로,시작(s)' / 'CSV경로,시작(s),종료(s)'. "
                        "복수 시나리오 지원: --dynamic-opts drop.csv,1.6,3.0 bump.csv,0.5")
    g.add_argument("--add-inertia",  action="store_true",     help="관성 하중(-ma) 인가")
    g.add_argument("--n-top",        type=int, default=10,    help="추출 ESL 개수 (기본: 10)")
    g.add_argument("--n-windows",    type=int, default=30,    help="시간 이력 분할 수 (기본: 30)")
    g.add_argument("--use-global-z",    action="store_true",      help="글로벌 Z 궤적 직접 사용")
    g.add_argument("--w-dynamic",       type=float, nargs='+', default=[1.0],
                   metavar="W",
                   help="시간분할 ESL 스냅샷 가중치 (기본: 1.0). CSV별 복수 지정 가능: --w-dynamic 1.0 0.8")
    g.add_argument("--w-peak",          type=float, default=0.0,  help="요소별 최대SE 피크 ESL 가중치 (기본: 0.0=비활성)")
    g.add_argument("--iterative-esl",   action="store_true", default=True,
                   help="매 이터레이션 ESL 재추출 (기본: 활성)")
    g.add_argument("--no-iterative-esl", action="store_false", dest="iterative_esl",
                   help="ESL 1회 추출 후 고정 (최적화 전 1회만 동해석)")
    g.add_argument("--no-static",        action="store_true", default=False,
                   help="정적 하중 케이스 전체 비활성 — 동적 ESL만 사용 (--dynamic-opts 병용 시)")

    # 하중 케이스 가중치 + 하중 크기 (W LOAD_N 두 값 입력)
    g = parser.add_argument_group(
        "정적 하중 케이스 설정 (가중치 하중N, 예: --w-bending 1.0 -10)"
    )
    g.add_argument("--w-bending",       type=float, nargs=2, default=[1.0, -10.0],
                   metavar=("W", "F_N"), help="중앙 굽힘        가중치·하중 (기본: 1.0 -5)")
    g.add_argument("--w-bending-xspan", type=float, nargs=2, default=[1.0, -10.0],
                   metavar=("W", "F_N"), help="X스팬 굽힘       가중치·하중 (기본: 1.0 -5)")
    g.add_argument("--w-bending-yspan", type=float, nargs=2, default=[1.0, -10.0],
                   metavar=("W", "F_N"), help="Y스팬 굽힘       가중치·하중 (기본: 1.0 -5)")
    g.add_argument("--w-twisting",      type=float, nargs=2, default=[1.0, -10.0],
                   metavar=("W", "F_N"), help="대각 비틀림      가중치·하중 (기본: 1.0 -5)")
    g.add_argument("--w-twisting-alt",  type=float, nargs=2, default=[1.0, -10.0],
                   metavar=("W", "F_N"), help="반전 대각 비틀림 가중치·하중 (기본: 1.0 -5)")
    g.add_argument("--w-lifting",       type=float, nargs=2, default=[1.0,  10.0],
                   metavar=("W", "F_N"), help="4코너 리프팅     가중치·하중 (기본: 1.0 5)")
    g.add_argument("--w-basic-weight",  type=float, default=1.0,
                   metavar="SCALE",
                   help="자중 기반 하중 스케일 (기본: 1.0). "
                        "0 이외의 값: 자중(total_mass*9806 N)*SCALE 을 각 하중값으로 대체. "
                        "0: 입력된 --w-xxx F_N 값을 그대로 사용.")

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
