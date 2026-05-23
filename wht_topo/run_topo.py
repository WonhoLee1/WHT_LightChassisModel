# -*- coding: utf-8 -*-
'''
# 여기는 삭제, 변경하지 않는다.

#모드 B | CSV 단독 동적 응답 해석 (최적화 생략)

python wht_topo/run_topo.py --pos-data wht_topo\structural_dynamics_c235.csv --add-inertia
python wht_topo/run_topo.py --pos-data wht_topo\structural_dynamics_c235.csv --add-inertia --use-global-z


#모드 D
python wht_topo/run_topo.py --dynamic-opts "wht_topo/structural_dynamics_rear.csv"  "wht_topo/structural_dynamics_c125.csv" "wht_topo/structural_dynamics_c235.csv" --add-inertia --sym-x --bead-connect 140 --height-steps 1 


# MST 방식 - 모든 섬을 Bresenham 직선으로 강제 연결
python wht_topo/run_topo.py ... --bead-connect 140 --bead-connect-alg mst

# Geodesic - 기존 비드를 최대 활용하는 자연스러운 경로
python wht_topo/run_topo.py ... --bead-connect 140 --bead-connect-alg geodesic

# Hybrid - closing으로 좁은 갭 먼저, 남은 섬은 MST로 처리
python wht_topo/run_topo.py ... --bead-connect 140 --bead-connect-alg hybrid


'''

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

  모드 E | 입력 디렉토리 일괄 실행 (--input-dir)
    단일 폴더에 옵션 파일(topo_arg.txt)과 복수 시나리오 CSV를 구성하여
    한 번의 명령으로 모드 C/D 설정을 완성합니다.
    폴더 구조:
      /input/
        topo_arg.txt              ← 기본 옵션 설정 (argparse 형식, # 주석 지원)
        scenario_A/               ← 폴더명이 loadcase 이름으로 사용됨
          motion_data.csv         ← 폴더 안의 첫 번째 .csv 파일을 자동 탐색
        scenario_B/
          bumpy_road.csv
        ...
    실행: python wht_topo/run_topo.py --input-dir /path/to/input
    우선순위: 명령줄 인자 > topo_arg.txt > 자동 발견 CSV

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

[자중 기반 정하중 자동계산 — --w-basic-weight]
  루프형 질량 행렬(mat-rho × 요소 면적 × 판 두께)로 총 자중을 산출하여
  각 정적 하중 케이스의 하중 크기를 물리적으로 의미 있는 값으로 자동 설정합니다.

  W_chassis = Σ(m_node) × 9806 mm/s²   [N]   (단위계: tonne·mm·s)

  --w-basic-weight 0     : 입력된 --w-xxx F_N 값을 그대로 적용
  --w-basic-weight 1.0   : F_i = sign(F_N_i) × W_chassis  (자중과 동일한 크기)
  --w-basic-weight 0.5   : F_i = sign(F_N_i) × W_chassis × 0.5  (자중의 50%)

  초기 실행 시 터미널에 자중·무게중심·편심·각 케이스 적용 하중을 상세 출력합니다.

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
  2. Kabsch 알고리즘(SVD)으로 강체 병진·회전 제거 → body-frame 3D 변형량 추출.
     충격 임펄스 순간(상대 Z가속도 > --contact-threshold)의 코너는 fit에서 제외.
     진단 그래프(X/Y/Z 방향 7행) → results 폴더 + CSV 폴더 동시 저장.
  3. 4코너 마스터 노드(#900000~900003) + RBE3 → X/Y/Z 3-DOF SPCD 하중 그룹 구성.
     CSV # 헤더의 C5~C8 좌표로 가장 가까운 FEM 노드 3개씩 자동 탐색.
  4. (옵션 --add-inertia) 이선형 보간 + 집중 질량 → 관성 하중(F=-ma) 전 노드 인가.
  5. 과도 응답 해석 (ζ=2%, --zeta 조정 가능).
     기본: Newmark-β 직접 적분. --n-modes N 지정 시 모달 중첩법(N개 모드) 사용.
     저장 스냅샷 수: --n-save-esl (기본 50, 최소 n_windows×2 자동 적용).
  6. 전체 SE 이력(SE = ½uᵀKu) 계산 → n_windows 구간 분할 → 전역 피크 후보 추출.
  7. Greedy Max-Min Cosine Similarity 다양성 선별 → Top-n_top 스냅샷 선정.
  8. WHTLoadCase(SPCD 형태)로 변환 → 목적 함수 하중 케이스 풀에 추가.
  출력: results/D날짜_시간/esl_se_report[_iterNNN].png
        (SE 이력 + 윈도우 분할 + 선택 시점 마킹)

[동적 해석 속도 최적화 옵션]
  --n-save-esl N        ESL 추출용 저장 스냅샷 수 (기본 50).
                        줄일수록 메모리·I/O 감소. n_windows×2 미만이면 자동 상향.
                        예) --n-windows 20 --n-save-esl 40

  --esl-skip-tol TOL    반복 ESL 재추출 스킵 임계값 (기본 0.0=항상 재실행).
                        이전 이터레이션 대비 Δh_rms < TOL 이면 동해석을 생략하고
                        직전 ESL 결과를 재사용. 수렴 후반부 동해석 횟수 절감.
                        예) --esl-skip-tol 0.05  (h_max=10mm 기준 0.5% 변화 이하)

  --parallel-scenarios N  동시 실행 프로세스 수 (기본: 1=순차).
                        2 이상 지정 시 ProcessPoolExecutor로 N개 시나리오를 동시 동해석.
                        시나리오당 모델 deep-copy+pickle 직렬화 발생 — 메모리 주의.
                        예) --dynamic-opts A.csv B.csv C.csv --parallel-scenarios 4

  --n-modes N           모달 중첩법 사용 모드 수 (기본: 0 = 직접 Newmark-β 적분).
                        0 초과 값을 지정하면 고유 모드를 N개 추출한 뒤 모달 좌표계
                        에서 ODE를 풀어 응답을 재합성—직접 적분 대비 수십~수백 배
                        빠르며 특히 자유도가 많은 메시에서 효과적.
                        권장값: 관심 주파수 대역을 커버하는 모드 수 (예: 20~50).
                        예) --n-modes 20

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
    run_kabsch_preprocessing,
    InterpLoadGroup,
    compute_vk_inertia_scale,
    print_vk_scale_report,
    compute_inertia_scale_via_fem,
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


def _dw(s: str) -> int:
    """터미널 표시 폭 계산: 한글·CJK = 2칸, 그 외 = 1칸."""
    n = 0
    for c in s:
        cp = ord(c)
        if (0xAC00 <= cp <= 0xD7A3
                or 0x1100 <= cp <= 0x11FF
                or 0x3130 <= cp <= 0x318F
                or 0x4E00 <= cp <= 0x9FFF
                or 0x3000 <= cp <= 0x303F
                or 0xF900 <= cp <= 0xFAFF):
            n += 2
        else:
            n += 1
    return n


def _rpad(s: str, w: int) -> str:
    """표시 폭 기준 오른쪽 공백 패딩."""
    return s + ' ' * max(0, w - _dw(s))


class _BoxPrinter:
    """박스 형식 터미널 출력 전담 클래스.

    모든 메서드는 표시 폭 _W 내에서 자동 줄바꿈을 보장합니다.
    한글·CJK 2칸 문자, 경로처럼 공백 없는 긴 문자열 모두 처리합니다.
    """

    def __init__(self, width: int = _W):
        self.W = width          # 박스 전체 표시 폭 (테두리 포함)
        self._inner = width - 4 # "  │" (3) + content + "│" (1)

    # ── 내부 유틸 ────────────────────────────────────────────────────────────

    def _split(self, s: str, max_w: int) -> list:
        """표시 폭 max_w를 초과하지 않도록 s를 줄 단위로 분리.

        공백이 있으면 단어 경계 우선, 없으면(경로 등) 하드-컷."""
        if _dw(s) <= max_w:
            return [s]
        lines, cur, cur_w = [], "", 0
        # 공백 기준 단어 분리 먼저 시도
        words = s.split(' ')
        if len(words) > 1:
            for word in words:
                sep = ' ' if cur else ''
                candidate = cur + sep + word
                if _dw(candidate) <= max_w:
                    cur = candidate
                else:
                    if cur:
                        lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
            # 단어 자체가 max_w 초과인 경우 하드-컷 재처리
            final = []
            for ln in lines:
                final.extend(self._hardcut(ln, max_w))
            return final
        return self._hardcut(s, max_w)

    def _hardcut(self, s: str, max_w: int) -> list:
        """표시 폭 기준 강제 절단."""
        lines, cur, cur_w = [], "", 0
        for ch in s:
            cw = _dw(ch)
            if cur_w + cw > max_w:
                lines.append(cur)
                cur, cur_w = ch, cw
            else:
                cur += ch
                cur_w += cw
        if cur:
            lines.append(cur)
        return lines or [s]

    def _print_line(self, content: str) -> None:
        """content를 박스 한 행으로 출력. 표시 폭을 맞춰 우측 테두리 정렬."""
        prefix = "  │  "
        inner = prefix + content
        n = max(0, self.W - _dw(inner) - 1)
        print(f"{inner}{' ' * n}│")

    # ── 공개 메서드 ───────────────────────────────────────────────────────────

    def hdr(self, title: str, ch: str = "═") -> str:
        """전체 폭 헤더 라인 (테두리 없음)."""
        inner = f"  {ch*2} {title} "
        n = max(0, self.W - _dw(inner))
        return inner + ch * n

    def section(self, title: str, step: str = "") -> None:
        """박스 상단: ┌─ [step] title ─...─┐"""
        prefix = f"[{step}] " if step else ""
        inner  = f"  ┌─ {prefix}{title} "
        n = max(2, self.W - _dw(inner) - 1)
        print(f"\n{inner}{'─' * n}┐")

    def endsec(self) -> None:
        """박스 하단: └─...─┘"""
        print(f"  └{'─' * (self.W - 4)}┘")

    def row(self, label: str, value: str = "", lw: int = 30) -> None:
        """박스 행. value가 길면 자동 줄바꿈하여 박스 안에서 출력."""
        label_part = _rpad(label, lw) + ' '
        # 값이 들어갈 수 있는 최대 표시 폭
        value_w = self._inner - 2 - _dw(label_part)  # "  "(2) + label + value
        lines = self._split(str(value), max(20, value_w))
        # 첫 줄: 레이블 + 값
        self._print_line(label_part + lines[0])
        # 이후 줄: 레이블 자리는 공백으로 채움
        cont_prefix = ' ' * (_dw(label_part))
        for ln in lines[1:]:
            self._print_line(cont_prefix + ln)

    def row_raw(self, content: str) -> None:
        """박스 행: 미리 구성된 문자열. 넘치면 줄바꿈."""
        lines = self._split(content, self._inner - 2)
        for ln in lines:
            self._print_line(ln)

    def sep(self) -> None:
        """박스 구분선: ├─...─┤"""
        print(f"  ├{'─' * (self.W - 4)}┤")

    def blank(self) -> None:
        """빈 행: │ spaces │"""
        print(f"  │{' ' * (self.W - 4)}│")


# 모듈 레벨 싱글턴 — 기존 호출부(_row, _section 등) 변경 없이 위임
_box = _BoxPrinter(_W)

def _hdr(title: str, ch: str = "═") -> str:        return _box.hdr(title, ch)
def _section(title: str, step: str = "") -> None:   _box.section(title, step)
def _endsec() -> None:                              _box.endsec()
def _row(label: str, value: str = "", lw: int = 30) -> None: _box.row(label, value, lw)
def _row_raw(content: str) -> None:                 _box.row_raw(content)
def _sep() -> None:                                 _box.sep()
def _blank() -> None:                               _box.blank()


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
    kabsch, time_arr, node_db, model, load_groups: list,
    corner_nids: list = None,
) -> tuple:
    """
    강체 가속도(T_arr 2차 미분, body-frame Z)로 모든 내부 노드에 관성 하중 F = -m·a 를 인가합니다.

    load_groups 리스트에 InterpLoadGroup(FORCE)을 직접 추가합니다.

    Returns
    -------
    total_mass   : float   총 섀시 질량 (tonne)
    a_body_z     : ndarray (T,) body-frame Z 강체 가속도 [mm/s²]
    n_inertia    : int     관성 하중 인가 노드 수
    """
    # world-frame 강체 가속도: 코너 accel_arr 평균 (T, 3)
    a_world  = kabsch.accel_arr.mean(axis=1)  # (T, 3)
    a_body   = np.einsum('tji,tj->ti', kabsch.R_arr, a_world)  # R.T @ a_world per step
    a_body_z = a_body[:, 2]  # (T,) body-frame Z 강체 가속도

    peak_g = float(np.max(np.abs(a_body_z))) / 9806.65
    rms_g  = float(np.sqrt(np.mean(a_body_z**2))) / 9806.65
    print(f"     강체 가속도 (body Z): peak={peak_g:.1f} g  RMS={rms_g:.1f} g")

    temp   = WHTDynamicSolver(model)
    jm, s_nids, n2i = temp._build_jaxsso_model()
    m_diag = temp._assemble_lumped_mass(jm, jm.ndof, s_nids, n2i)

    total_mass = float(sum(
        m_diag[n2i[nid] * 6 + 2]
        for nid in node_db
        if n2i.get(nid) is not None and nid < 900000
        and m_diag[n2i[nid] * 6 + 2] > 1e-12
    ))
    print(f"     총 질량 (lumped): {total_mass*1e3:.3f} kg")

    # Von Kármán 비선형 목표 처짐 계산
    all_xyz = np.array(list(node_db.values()))
    plate_a = float(all_xyz[:, 0].max() - all_xyz[:, 0].min())
    plate_b = float(all_xyz[:, 1].max() - all_xyz[:, 1].min())
    _, _, w_NL, vk_info = compute_vk_inertia_scale(
        E=MAT['E'], nu=MAT['nu'], t=MAT['t'],
        plate_a=plate_a, plate_b=plate_b,
        total_mass=total_mass, a_body_z=a_body_z,
    )
    print_vk_scale_report(vk_info)

    # FEM 역산: 코너 고정 정해석으로 실제 구조 강성 기반 보정계수 계산
    _corner_nids = corner_nids if corner_nids else []
    fem_scale, _ = compute_inertia_scale_via_fem(
        model=model,
        corner_nids=_corner_nids,
        n2i=n2i, m_diag=m_diag,
        a_body_z=a_body_z,
        w_NL_target=w_NL,
    )
    a_body_z_scaled = a_body_z * fem_scale

    n_inertia = 0
    for nid in node_db:
        ix = n2i.get(nid)
        if ix is None or nid >= 900000:
            continue
        node_mass = m_diag[ix * 6 + 2]
        if node_mass > 1e-12:
            load_groups.append(
                InterpLoadGroup([nid], 2, time_arr, -node_mass * a_body_z_scaled, "FORCE")
            )
            n_inertia += 1

    return total_mass, a_body_z_scaled, n_inertia


# ─────────────────────────────────────────────────────────────────────────────
# 병렬 ESL 추출 헬퍼 (ProcessPoolExecutor용 모듈 최상위 함수 — pickle 가능해야 함)
# ─────────────────────────────────────────────────────────────────────────────

def _esl_worker(args: tuple) -> list:
    """
    [병렬 단계] 동해석 + ESL 추출만 수행.
    prepare()는 메인 프로세스에서 순차적으로 완료된 후 호출된다.
    args에 ESLExtractor 인스턴스(prepare 완료 상태)를 전달한다.
    """
    extractor = args[0]
    return extractor.solve_and_extract()


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
        n_save_esl: int = 50,
        n_modes: int = 0,
        dt: Optional[float] = None,
        use_jax: bool = False,
        contact_threshold: float = 24516.6,
    ):
        self.model             = model
        self.node_db           = node_db
        self.csv_path          = csv_path
        self.t_start           = t_start   # None → CSV 헤더 start_time 자동 적용
        self.t_end             = t_end     # None → CSV 마지막 프레임까지
        self.n_windows         = n_windows
        self.n_top             = n_top
        self.add_inertia       = add_inertia
        self.use_global_z      = use_global_z
        self.contact_threshold = contact_threshold
        self.esl_weight   = esl_weight
        self.w_peak       = w_peak
        self.iteration    = iteration  # -1: 단독 실행, ≥0: 최적화 루프 내
        self.out_dir      = out_dir    # None이면 CSV 옆 디렉토리에 저장
        self.n_save_esl   = max(n_windows * 2, n_save_esl)  # 윈도우당 최소 2개 보장
        self.n_modes      = n_modes    # 0: 직접 Newmark-β, >0: 모달 중첩법
        self.dt           = dt         # None → CSV 샘플링 간격 그대로 사용
        self.use_jax      = use_jax    # True → JAX 직접 적분 (method='jax')

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
        """ESL 추출 전체 파이프라인 (순차 실행용 — prepare + solve_and_extract 통합)."""
        self.prepare()
        return self.solve_and_extract()

    def prepare(self) -> None:
        """
        [순차 단계] CSV 로드·코너 탐색·하중 그룹 구성까지 수행하고 출력.
        병렬 실행 시 이 단계를 먼저 순차적으로 완료한 뒤 solve_and_extract()를 병렬 실행한다.
        """
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

    def solve_and_extract(self) -> List[Tuple[WHTLoadCase, float]]:
        """
        [병렬 가능 단계] 동해석·ESL 추출·스냅샷 선정 수행.
        prepare() 완료 후 호출해야 한다.
        """
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
        """各 코너에 마스터 노드(#900000+) + RBE3 + 3-DOF SPCD 하중을 구성합니다.
        Kabsch 알고리즘으로 강체 운동을 제거한 body-frame 변형량(X,Y,Z)을 SPCD로 입력합니다.
        """
        corner_positions = self._csv_header.get('corner_positions', {})
        lc_name = Path(self.csv_path).parent.name  # loadcase 이름 = CSV 부모 폴더명

        diag_base = f"kabsch_{lc_name}"
        diag_paths = [str(self.out_dir / diag_base)] if self.out_dir else []

        self._kabsch = run_kabsch_preprocessing(
            self._df, self._time_arr,
            corner_positions=corner_positions if corner_positions else None,
            contact_accel_threshold=self.contact_threshold,
            diag_save_paths=diag_paths,
            loadcase_name=lc_name,
        )

        for idx, (cname, cnids) in enumerate(self._bot_groups):
            center = np.mean([self.node_db[n] for n in cnids], axis=0)
            mnid   = 900000 + idx
            self.model.add_node(mnid, center[0], center[1], center[2])
            self.model.add_rbe3(mnid, mnid, cnids, dofs=(0, 1, 2))
            d = self._kabsch.deformation.get(cname)
            if d is None:
                continue
            for dof_idx, dof in enumerate([0, 1, 2]):   # X, Y, Z
                self._load_groups.append(InterpLoadGroup(
                    [mnid], dof, self._time_arr, d[:, dof_idx], "SPCD"
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
        _c_nids = [nid for _, nids in self._bot_groups for nid in nids]
        total_mass, accels, n_inertia = _apply_inertia_loads(
            self._kabsch, self._time_arr, self.node_db, self.model, self._load_groups,
            corner_nids=_c_nids,
        )
        self._total_mass    = total_mass
        self._corner_accels = accels
        self._accel_dt      = self._time_arr[1] - self._time_arr[0] if len(self._time_arr) > 1 else 1e-4

        _section("관성 하중 (F = -m·a) 인가", "ESL-0c")
        _row("총 섀시 질량", f"{self._total_mass*1e3:.2f} kg  ({self._total_mass:.4e} tonne)")
        _row("관성 하중 인가 노드", f"{n_inertia:,}개")
        self._print_impact_summary()
        _endsec()

    def _print_impact_summary(self) -> None:
        """강체 가속도 기반 충격력 지표를 출력합니다 (CSV 간 비교용)."""
        if not hasattr(self, '_corner_accels') or self._corner_accels is None:
            return
        a    = self._corner_accels  # (T,) body-frame Z 강체 가속도 mm/s²
        M    = getattr(self, '_total_mass', None)
        pk   = float(np.max(np.abs(a)))
        rms  = float(np.sqrt(np.mean(a**2)))
        g_pk  = pk  / G_MM_S2
        g_rms = rms / G_MM_S2

        _sep()
        _row("강체 가속도 (body-frame Z)", f"peak {pk:,.1f} mm/s²  ({g_pk:.3f} g)  RMS {g_rms:.3f} g")

        if M and M > 0:
            F_peak = M * pk
            F_rms  = M * rms
            _sep()
            _row("▶ 충격력 지표 (M × a_peak)",
                 f"F_peak = {F_peak:,.1f} N   ({g_pk:.2f} g)")
            _row("  충격력 지표 (M × a_rms)",
                 f"F_rms  = {F_rms:,.1f} N")
            _row("  (이 값으로 다른 CSV 조건의 충격 심각도를 비교하세요)", "")

    def _run_dynamic(self) -> None:
        csv_dt  = self._time_arr[1] - self._time_arr[0] if len(self._time_arr) > 1 else 1e-4
        T_total = float(self._time_arr[-1])

        # --dt 적용: CSV 간격과 다르면 시계열 리샘플링
        if self.dt is not None and not np.isclose(self.dt, csv_dt, rtol=1e-6):
            n_new = max(2, int(round(T_total / self.dt)) + 1)
            t_new = np.linspace(0.0, T_total, n_new)
            for lg in self._load_groups:
                lg._u = np.interp(t_new, self._time_arr, lg._u)
                lg._t = t_new
            self._time_arr = t_new
            dt = float(t_new[1] - t_new[0])
            dt_note = f"dt={dt:.2e} s  (리샘플: CSV {csv_dt:.2e} s → 지정 {self.dt:.2e} s)"
        else:
            dt = csv_dt
            dt_note = f"dt={dt:.2e} s"

        try:
            _p = Path(self.csv_path)
            _folder = _p.parent.name
            csv_name = _folder if _folder else _p.stem
            if not csv_name:
                csv_name = _p.stem
        except Exception:
            csv_name = str(self.csv_path)
        self._dyn_solver = WHTDynamicSolver(self.model)

        if self.n_modes > 0:
            _section(
                f"과도 응답 해석  (모달 중첩법  modes={self.n_modes}  ζ={ZETA*100:.0f}%)",
                "ESL-1",
            )
            _row("하중 케이스", csv_name)
            _row("적분 파라미터",
                 f"{dt_note}   T={T_total:.4f} s   "
                 f"모드 수={self.n_modes}   ESL 저장 스냅샷={self.n_save_esl}")
            print()
            self._dyn_res = self._dyn_solver.solve_modal_dynamic(
                self._load_groups, dt=dt, T=T_total,
                n_modes  = self.n_modes,
                n_save   = self.n_save_esl,
                damping  = DampingSpec(mode="zeta", zeta=ZETA),
            )
        else:
            _section(f"과도 응답 해석  (직접 Newmark-β  ζ={ZETA*100:.0f}%)", "ESL-1")
            _row("하중 케이스", csv_name)
            _row("적분 파라미터",
                 f"{dt_note}   T={T_total:.4f} s   "
                 f"ESL 저장 스냅샷={self.n_save_esl}")
            print()
            self._dyn_res = self._dyn_solver.solve_direct_dynamic(
                self._load_groups, dt=dt, T=T_total, n_save=self.n_save_esl,
                damping=DampingSpec(mode="zeta", zeta=ZETA),
                label=csv_name,
                method='jax' if self.use_jax else 'scipy',
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
        self.cfg          = cfg
        self.model        : Optional[WHTMeshModel]     = None
        self.node_db      : Optional[dict]             = None
        self._bot_groups  : Optional[list]             = None
        self._kabsch      : Optional[object]           = None
        self._load_groups : list                       = []
        self.dyn_res      : Optional[DynamicResult]    = None
        self.wht_data     : Optional[WHTResultData]    = None
        self.out_dir      : Optional[Path]             = None

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
                _c_nids = [nid for _, nids in self._bot_groups for nid in nids]
                total_mass, accels, n_inertia = _apply_inertia_loads(
                    self._kabsch, self._time_arr,
                    self.node_db, self.model, self._load_groups,
                    corner_nids=_c_nids,
                )
                _section("관성 하중 (F = -m·a) 인가", "B-inertia")
                _row("총 섀시 질량", f"{total_mass*1e3:.2f} kg  ({total_mass:.4e} tonne)")
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
        csv_dt  = float(time_arr[1] - time_arr[0]) if len(time_arr) > 1 else 1e-3
        dt_val  = float(self.cfg.dt) if self.cfg.dt is not None else csv_dt
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

        self._bot_groups = bot_groups

        # Kabsch 전처리: 강체 제거 → body-frame 3D 변형량
        print(f"\n [4] Kabsch 전처리 및 3-DOF SPCD 하중 그룹 구성...")
        lc_name = Path(self.cfg.pos_data).parent.name or Path(self.cfg.pos_data).stem

        diag_paths = [str(self.out_dir / f"kabsch_{lc_name}")] if self.out_dir else []

        contact_thr = getattr(self.cfg, 'contact_threshold', 24516.6)
        self._kabsch = run_kabsch_preprocessing(
            df, time_arr,
            corner_positions=header_corners if header_corners else None,
            contact_accel_threshold=contact_thr,
            diag_save_paths=diag_paths,
            loadcase_name=lc_name,
        )

        # 마스터 노드 + RBE3 + X/Y/Z SPCD
        load_groups = []
        for idx, (cname, corner_nids) in enumerate(bot_groups):
            center = np.mean([self.node_db[n] for n in corner_nids], axis=0)
            mnid   = 900000 + idx
            self.model.add_node(mnid, center[0], center[1], center[2])
            self.model.add_rbe3(mnid, mnid, corner_nids, dofs=(0, 1, 2))
            d = self._kabsch.deformation.get(cname)
            if d is None:
                continue
            for dof_idx, dof in enumerate([0, 1, 2]):   # X, Y, Z
                load_groups.append(InterpLoadGroup(
                    [mnid], dof, time_arr, d[:, dof_idx], "SPCD"
                ))
        # 강체 이동 방지 (X, Y SPC는 SPCD가 직접 구속하므로 제거)
        # SPCD가 X,Y,Z 모두 구속 → 추가 SPC 불필요

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

        _n_par = getattr(self.cfg, 'parallel_scenarios', 1) or 1
        use_parallel = _n_par > 1 and len(entries) > 1

        def _make_extractor(csv_path, t_start, t_end, w_dyn, iteration, model):
            return ESLExtractor(
                model        = model,
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
                n_save_esl   = self.cfg.n_save_esl,
                n_modes      = getattr(self.cfg, 'n_modes', 0),
                dt                 = getattr(self.cfg, 'dt', None),
                use_jax            = getattr(self.cfg, 'jax_dynamic', False),
                contact_threshold  = getattr(self.cfg, 'contact_threshold', 24516.6),
            )

        def _run_esl(iteration: int):
            import copy
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # ── 1단계: prepare() 순차 실행 (출력이 섞이지 않도록) ────────────
            extractors = []
            for (csv_path, t_start, t_end), w_dyn in zip(entries, w_list):
                model_copy = copy.deepcopy(self.model) if use_parallel else self.model
                ext = _make_extractor(csv_path, t_start, t_end, w_dyn, iteration, model_copy)
                ext.prepare()
                extractors.append(ext)

            if not use_parallel:
                all_snaps = []
                for ext in extractors:
                    all_snaps.extend(ext.solve_and_extract())
                return all_snaps

            # ── 2단계: solve_and_extract() 병렬 실행 ─────────────────────────
            n_proc = min(len(extractors), _n_par)
            print(f"\n     [병렬 ESL] {len(extractors)}개 시나리오 동해석 병렬 시작... (workers={n_proc})")
            results = [None] * len(extractors)
            with ProcessPoolExecutor(max_workers=n_proc) as exe:
                fut_map = {exe.submit(_esl_worker, (ext,)): idx
                           for idx, ext in enumerate(extractors)}
                for fut in as_completed(fut_map):
                    idx = fut_map[fut]
                    results[idx] = fut.result()
            all_snaps = []
            for snaps in results:
                all_snaps.extend(snaps)
            return all_snaps

        if getattr(self.cfg, 'iterative_esl', False):
            # ── 반복 추출 모드: provider를 solver에 전달 ──────────────────────
            esl_skip_tol   = getattr(self.cfg, 'esl_skip_tol', 0.0)
            _prev_h        = [None]   # [np.ndarray | None]  직전 h_elem 저장
            _cached_snaps  = [None]   # [list | None]        직전 ESL 결과 캐시

            def provider(iteration: int, h_elem=None):
                nonlocal _prev_h, _cached_snaps
                # Δh_rms 스킵 판단
                if (esl_skip_tol > 0.0
                        and h_elem is not None
                        and _prev_h[0] is not None
                        and _cached_snaps[0] is not None):
                    delta = float(np.sqrt(np.mean((h_elem - _prev_h[0]) ** 2)))
                    if delta < esl_skip_tol:
                        print(f"\n [ESL-스킵] Iter {iteration} "
                              f"Δh_rms={delta:.4f} < tol={esl_skip_tol} "
                              f"→ 직전 ESL {len(_cached_snaps[0])}개 재사용.")
                        return _cached_snaps[0]

                print(f"\n [ESL-재추출] Iter {iteration} — 현재 비드 형상 기준 동적 해석")
                snaps = _run_esl(iteration)
                print(f"     -> Iter {iteration} ESL {len(snaps)}개 반환.")
                if h_elem is not None:
                    _prev_h[0]       = h_elem.copy()
                _cached_snaps[0] = snaps
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
            bead_connect_alg  = getattr(self.cfg, 'bead_connect_alg', 'closing'),
            bead_steps        = self.cfg.height_steps,
            filter_type       = getattr(self.cfg, 'filter_type', 'linear'),
            use_projection    = getattr(self.cfg, 'projection', 0.0) > 0,
            proj_beta_max     = getattr(self.cfg, 'projection', 0.0) or 32.0,
            load_cases        = load_cases,
            load_case_provider= load_case_provider,
            out_dir           = self.out_dir,
            normalize_obj     = self.cfg.normalize_obj > 0,
            weight_variation  = max(0.0, self.cfg.normalize_obj - 1.0),
            obj_type          = self.cfg.obj_type,
            obj_alpha         = self.cfg.obj_alpha,
            freq_weight       = freq_w,
            freq_target       = freq_f0,
            exclude_zones     = exclude_zones,
            n_workers         = self.cfg.n_workers,
        )
        n_designs = getattr(self.cfg, 'n_designs', 1)
        if n_designs > 1:
            self.solver.diversity_weight = getattr(self.cfg, 'diversity_weight', 0.3)
            self.solver.diversity_sigma  = getattr(self.cfg, 'diversity_sigma',  0.3)

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
                args=(queue, stop_event, str(self.out_dir)),
                kwargs={"num_modal_modes": self.cfg.modal_modes},
            )
            ui_process.start()
            callback = queue.put

        n_designs    = getattr(self.cfg, 'n_designs', 1)
        noise        = getattr(self.cfg, 'diversity_noise', 0.15)
        print(f" [4] 최적화 실행 (max_iter={self.cfg.iters}, n_designs={n_designs})...")
        try:
            for design_idx in range(n_designs):
                if n_designs > 1:
                    print(f"\n{'='*60}")
                    print(f"  [Design {design_idx+1}/{n_designs}]")
                    print(f"{'='*60}")
                    if design_idx > 0:
                        self.solver.reset_for_next_design(noise=noise)
                self.solver.solve(max_iter=self.cfg.iters, callback=callback,
                                  stop_event=stop_event)
                if n_designs > 1:
                    # 각 설계를 별도 파일로 저장
                    self._discretize()
                    self._apply_shape()
                    _stem = Path(self.cfg.export or "final.k").stem if self.cfg.export else "design"
                    _out  = (self.out_dir / f"{_stem}_d{design_idx+1:02d}.k") if self.out_dir \
                            else Path(f"{_stem}_d{design_idx+1:02d}.k")
                    self.model.export_to_solver('lsdyna', str(_out), reorder=True)
                    print(f" -> 설계 {design_idx+1} 저장: {_out}")
                    # 모델 좌표 원점 복원 후 다음 설계 준비
                    self.solver._restore_heights()
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
        print(f"  {'총 질량':<28} {total_mass*1e3:>28.4f} kg  "
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

def _resolve_input_dir(parser) -> list:
    """
    sys.argv 에서 --input-dir 를 먼저 추출합니다.

    1. topo_arg.txt 가 있으면 파일 내용을 기본 인자로 로드합니다.
       topo_arg.txt 내 --dynamic-opts / --pos-data 의 상대경로는 input_dir 기준으로 변환합니다.
    2. 하위 폴더를 탐색합니다. 폴더명이 loadcase 이름이 되고,
       폴더 안의 첫 번째 .csv 파일을 --dynamic-opts 항목으로 자동 추가합니다.
       (파일명은 chassis_corners.csv 고정 이름이 아니어도 됩니다.)
    3. 명령줄 인자(--input-dir 제외)가 topo_arg.txt 설정을 덮어씁니다.

    Returns
    -------
    list  parser.parse_args() 에 전달할 최종 argv 리스트
    """
    import shlex

    argv = sys.argv[1:]

    # --input-dir 값 추출 (따옴표 포함 경로 대응)
    input_dir: Optional[Path] = None
    clean_argv = []
    i = 0
    while i < len(argv):
        if argv[i] in ("--input-dir",):
            if i + 1 < len(argv):
                input_dir = Path(argv[i + 1].strip('"').strip("'"))
                i += 2
            else:
                i += 1
        elif argv[i].startswith("--input-dir="):
            input_dir = Path(argv[i].split("=", 1)[1].strip('"').strip("'"))
            i += 1
        else:
            clean_argv.append(argv[i])
            i += 1

    if input_dir is None:
        return clean_argv

    if not input_dir.is_dir():
        parser.error(f"--input-dir: 경로가 존재하지 않거나 디렉토리가 아닙니다: {input_dir}")

    print(f"\n  [input-dir] 입력 디렉토리: {input_dir}")

    # ── 1. topo_arg.txt 로드 ──────────────────────────────────────────────────
    base_argv: list = []
    arg_file = input_dir / "topo_arg.txt"
    if arg_file.exists():
        raw = arg_file.read_text(encoding="utf-8")
        lines = [l.strip() for l in raw.splitlines()]
        for line in lines:
            line = line.split("#")[0].strip()   # 주석 제거
            if not line:
                continue
            base_argv.extend(shlex.split(line))

        # topo_arg.txt 내 --dynamic-opts / --pos-data 의 상대경로를 input_dir 기준 절대경로로 변환.
        # CWD가 다른 위치일 때 "chassis_corners.csv" 같은 상대경로가 깨지는 문제 방지.
        _path_opts = {"--dynamic-opts", "--pos-data"}
        _in_path_opt = False
        for k, tok in enumerate(base_argv):
            if tok in _path_opts:
                _in_path_opt = True
                continue
            if _in_path_opt:
                if tok.startswith("--"):
                    _in_path_opt = False
                else:
                    # CSV 경로[,t_start[,t_end]] 형태에서 경로 부분만 절대경로 변환
                    parts = tok.split(",", 1)
                    p = Path(parts[0])
                    if not p.is_absolute():
                        parts[0] = str((input_dir / p).resolve())
                        base_argv[k] = ",".join(parts)

        print(f"  [input-dir] topo_arg.txt 로드: {len(base_argv)}개 토큰")
    else:
        print(f"  [input-dir] topo_arg.txt 없음 — 기본값 사용")

    # ── 2. 하위 폴더 탐색: 폴더명 = loadcase 이름, 폴더 내 첫 번째 .csv = 데이터 파일 ──
    # chassis_corners.csv 고정 이름 대신, 하위 폴더 안의 임의 .csv 파일을 자동 탐색한다.
    # 폴더명이 ESL extractor의 csv_name(_p.parent.name)으로 자동 반영되어 loadcase 이름이 된다.
    csv_entries: list = []
    for sub in sorted(input_dir.iterdir()):
        if not sub.is_dir():
            continue
        csv_files = sorted(sub.glob("*.csv"))
        if not csv_files:
            continue
        if len(csv_files) > 1:
            print(f"  [input-dir]   {sub.name}/: CSV {len(csv_files)}개 — 첫 번째 파일 사용: {csv_files[0].name}")
        csv_entries.append(str(csv_files[0]))

    if csv_entries:
        print(f"  [input-dir] 하위 폴더(loadcase) 발견: {len(csv_entries)}개")
        for p in csv_entries:
            _sub = Path(p)
            print(f"             [{_sub.parent.name}]  {_sub.name}")
    else:
        print(f"  [input-dir] CSV 파일을 포함하는 하위 폴더 없음")

    # ── 3. 최종 argv 조합 ─────────────────────────────────────────────────────
    # 우선순위: 명령줄(clean_argv) > 자동 발견 CSV(csv_entries) > topo_arg.txt(base_argv)
    #
    # 하위 폴더에서 CSV를 발견한 경우: topo_arg.txt 의 --dynamic-opts 를 제거하고
    # 발견된 CSV 목록으로 완전히 대체한다.
    # (topo_arg.txt 의 --dynamic-opts chassis_corners.csv 같은 잔류 항목과 충돌 방지)
    #
    # 명령줄(clean_argv) 에 --dynamic-opts 가 있으면 그것이 최우선 — 자동 발견 CSV도 무시.

    def _strip_dynamic_opts(argv: list) -> list:
        """argv 에서 --dynamic-opts 와 그 뒤 값 토큰을 모두 제거한다."""
        result = []
        skip = False
        for tok in argv:
            if tok == "--dynamic-opts":
                skip = True
                continue
            if skip and tok.startswith("--"):
                skip = False
            if not skip:
                result.append(tok)
        return result

    if csv_entries and "--dynamic-opts" not in clean_argv:
        # 하위 폴더 CSV 발견 + 명령줄에 --dynamic-opts 없음
        # → topo_arg.txt 의 --dynamic-opts 를 제거하고 발견된 CSV 로 대체
        base_argv_clean = _strip_dynamic_opts(base_argv)
        final_argv = base_argv_clean + clean_argv + ["--dynamic-opts"] + csv_entries
        if "--dynamic-opts" in base_argv:
            print(f"  [input-dir] topo_arg.txt 의 --dynamic-opts 를 하위 폴더 CSV {len(csv_entries)}개로 대체")
    else:
        final_argv = base_argv + clean_argv
        if csv_entries and "--dynamic-opts" in clean_argv:
            print(f"  [input-dir] 명령줄 --dynamic-opts 우선 적용 — 하위 폴더 CSV {len(csv_entries)}개 무시")
        elif csv_entries:
            # csv_entries 있지만 final_argv 에 --dynamic-opts 없는 경우 (base_argv 에도 없었음)
            final_argv += ["--dynamic-opts"] + csv_entries

    return final_argv


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
    python wht_topo/run_topo.py --n-workers 4 \\
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \\
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \\
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \\
      --w-dynamic 1.0 --w-peak 1.0 --add-inertia \\
      --sym-x --bead-connect --height-steps 2 --min-width 150

  헤드리스 서버 실행
    python wht_topo/run_topo.py \\
      --dynamic-opts "wht_topo/structural_dynamics_rear.csv,0.4,0.6" \\
                     "wht_topo/structural_dynamics_c125.csv,1.5,1.8" \\
                     "wht_topo/structural_dynamics_c235.csv,1.5,1.8" \\
      --w-dynamic 1.0 --w-peak 0.5 --add-inertia \\
      --n-windows 50 --n-top 20 \\
      --sym-x --bead-connect --height-steps 2 \\
      --no-gui --no-viz

[모드 E] 입력 디렉토리 일괄 실행

  폴더 내 topo_arg.txt + 하위 폴더 CSV 자동 수집 (폴더명 = loadcase 이름)
    python wht_topo/run_topo.py --input-dir "D:/data/session_01"

  폴더 구조 예시:
    session_01/
      topo_arg.txt          <- 기본 옵션 (# 주석 지원, 아래 옵션으로 덮어쓰기 가능)
      rear_impact/          <- loadcase 이름: "rear_impact"
        motion.csv          <- 폴더 안 첫 번째 .csv 파일 자동 사용 (파일명 무관)
      curb_bump_125/        <- loadcase 이름: "curb_bump_125"
        bumpy.csv
      curb_bump_235/        <- loadcase 이름: "curb_bump_235"
        bumpy.csv

  명령줄 인자로 topo_arg.txt 설정 일부 덮어쓰기
    python wht_topo/run_topo.py --input-dir "D:/data/session_01" --iters 30 --no-gui

  우선순위: 명령줄 인자 > topo_arg.txt > 자동 발견 CSV

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
옵션 상세 설명
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[섀시 형상]
--tray-width/length/height  섀시 외형 치수 mm (기본: 1800×1200×40)
--mesh-xy / --mesh-z        XY면·Z방향 메시 크기 mm (기본: 30 / 10)
--draft-angle               측면 드래프트 각도 deg (기본: 25)

[재료 물성]
--mat-E    탄성계수 MPa (기본: 210000, 강재)
--mat-nu   포아송비    (기본: 0.3)
--mat-rho  밀도 tonne/mm³ (기본: 7.85e-9, 강재)
--mat-t    기본 판 두께 mm (기본: 0.6)

[입력 디렉토리 (모드 E)]
--input-dir DIR
                디렉토리 경로를 지정하면 아래 두 동작을 자동 수행합니다.
                  1. DIR/topo_arg.txt 를 기본 옵션 파일로 로드 (# 주석 지원)
                  2. DIR 하위 폴더를 탐색 — 폴더명이 loadcase 이름이 되고,
                     폴더 안의 첫 번째 .csv 파일을 --dynamic-opts 항목으로 자동 추가
                     (파일명은 무관, chassis_corners.csv 고정 이름 불필요)
                우선순위: 명령줄 인자 > topo_arg.txt > 자동 발견 CSV
                경로에 공백이 있으면 따옴표로 감싸세요: --input-dir "D:/my path"

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
                --input-dir 사용 시 하위 폴더(폴더명=loadcase명) 내 첫 .csv 가 자동 추가됨.

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

--w-basic-weight SCALE
                자중 기반 하중 자동계산 스케일 (기본: 1.0).
                  0     : --w-xxx F_N 입력값을 그대로 사용
                  1.0   : 총 자중(M×9806 N)을 각 케이스 하중 크기로 자동 대체
                  0.5   : 자중의 50% 적용 (하중 민감도 분석 등에 활용)
                실행 초기 터미널에 총 질량·자중·무게중심·편심 모멘트·
                케이스별 적용 하중을 상세 출력합니다.

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
--n-save-esl    ESL 추출용 저장 스냅샷 수 (기본: 50).
                n_windows*2 미만이면 자동 상향. 줄일수록 동해석 메모리·I/O 감소.

--esl-skip-tol  반복 ESL 재추출 스킵 임계값 (기본: 0.0=항상 재실행).
                이전 이터 대비 Δh_rms < TOL 이면 직전 ESL 재사용.
                0.05 권장 (h_max 10mm 기준 약 0.5mm 이하 변화 시 스킵).

--parallel-scenarios N
                동시 실행 프로세스 수 (기본: 1=순차).
                2 이상 지정 시 ProcessPoolExecutor로 N개 동시 동해석.
                모델 pickle 직렬화 발생 → 메모리 N배 증가 주의.

--n-modes N     모달 중첩법 사용 모드 수 (기본: 0=직접 Newmark-β).
                0 초과 시 고유치 해석 후 모달 좌표계 ODE로 응답 합성.
                직접 적분 대비 수십~수백 배 빠름. 권장: 20~50.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
비드 형상 제어 (Topography Filter / Projection)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--min-width R mm  (기본: 80)
    공간 밀도 필터 반경. R 이내의 요소 높이를 가중 평균하여 비드가
    R 미만의 폭으로 형성되지 않도록 강제.
    체커보드 방지 필수 조건: R > 2 × 메시 간격.
    권장: 메시 간격의 3~5배 (메시 30mm → R=90~150).

--filter-type {linear|gaussian}  (기본: linear)
    linear  : w(d) = R - d  (거리에 비례하는 hat 커널)
              → 날카로운 필터 경계, 처음 시도에 적합
    gaussian: w(d) = exp(-d²/2σ²), σ=R/3
              → 부드러운 가우시안 커널. 필터 경계가 완만해
                 큼직한 덩어리 형태 패턴 유도에 유리.
                 체커보드 억제 효과가 linear보다 강함.
    권장: gaussian + --min-width 80~120

--projection β  (기본: 0=비활성)
    수렴형 Heaviside 투사(β-continuation).
    순전파: H_β(x;η=0.5) = [tanh(βη)+tanh(β(x-0.5))] / [tanh(βη)+tanh(β(0.5))]
    β=1 → 완만한 S-커브(부드러운 시작)
    β→∞ → 계단함수(완전한 0/1)
    최적화 진행에 따라 β를 1에서 지정값까지 선형 증가시켜
    초기에는 탐색 유연성을, 후기에는 선명한 비드 경계를 보장.
    수렴 후 비드 면적비 제약이 필터+Heaviside 조합으로 정확히 만족됨.
    권장: 32 (8~16: 완만한 수렴, 32: 표준, 64+: 불안정 위험)

--bead-connect N mm  (기본: 120, 0=비활성)
    형태학적 폐합(Morphological Closing: Dilation→Erosion)으로
    N mm 이하의 단절된 비드 구간을 자동 연결.
    ① bridge 노드 민감도를 이웃 최대값으로 승격 → MMA가 연결 유지
    ② bridge 추가로 인한 체적 증가를 사전에 vol_frac에서 차감
    연속된 비드 라인은 제조성(프레스 성형)과 강성 기여 모두에 필수.
    권장: 100~150 (메시 간격 30mm 기준 약 3~5 요소 간격)

--height-steps N  (기본: 1=연속)
    비드 높이 이산화 단계 수.
    1  : 연속 변수 (0~h_max 사이 임의값)
    2  : 이진({0, h_max}) → 완전한 ON/OFF 비드
    3+ : N개 등간격 레벨. tanh 계단 함수로 근사, β-continuation으로 점진 이산화.
    권장: 초기 탐색=1, 최종 제조 설계=2

--sym-x / --no-sym-x  (기본: 활성)
    X축 기준 좌우 대칭 강제. 설계 변수와 민감도를 좌우 평균하여 동기화.
    섀시 구조 특성상 대칭 강제가 물리적으로 타당하고 수렴을 가속함.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
다중 설계 탐색 (Multi-Design Exploration)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
동일 하중·제약 조건에서 위상적으로 다른 여러 설계안을 순차 생성.
각 설계 수렴 후 반발 패널티(Repulsion Penalty)를 추가해 다음 최적화가
이전 설계에서 멀어지도록 유도한다.

  수렴 설계 x*_k 등록 후 다음 목적함수:
    f_total = f_compliance + λ·Σ_k exp(-||x - x*_k||² / 2σ²)
    → x*_k 근방에서 목적함수가 높아져 MMA가 회피 방향으로 이동

--n-designs N  (기본: 1)
    생성할 설계 수. N>1이면 첫 설계 수렴 후 반발 패널티를 추가하며
    N-1번 재시작. 각 설계는 별도 파일(design_dNN.k)로 저장.

--diversity-weight λ  (기본: 0.3)
    반발 패널티 강도. 클수록 이전 설계에서 멀리 떨어진 새 형태 탐색.
    너무 크면 compliance가 나빠짐.
    권장: 0.1~0.5. 하중 케이스 수가 많거나 목적함수가 크면 낮춤.

--diversity-sigma σ  (기본: 0.3)
    반발 가우시안의 표준편차 (설계 변수 [0,1] 공간 기준).
    작으면(0.1~0.2) 국소 회피 → 거의 같은 형태에서 세부만 다름.
    크면(0.4~0.6) 넓은 회피 → 위상적으로 다른 구조 유도.
    권장: 0.25~0.4

--diversity-noise ε  (기본: 0.15)
    재시작 시 이전 수렴 설계에 추가하는 랜덤 노이즈 강도.
    반발 패널티와 함께 새로운 국소 최적해 탐색을 보조.
    너무 크면(>0.4) 체적 제약 위반으로 초기 이터 불안정.
    권장: 0.1~0.2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
목적함수 옵션
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--normalize-obj V  (기본: 0=비활성)
    0       : 비활성. f = C_total / C_0 (Iter 0 총 컴플라이언스로 스케일)
    1.0     : 케이스별 정규화. f = Σ w_i·(C_i/C_i0)
              하중 크기가 다른 정적·동적 케이스 혼재 시 권장.
              C_i0 = 각 케이스 Iter 0 컴플라이언스 → 케이스 간 크기 차이 제거.
    1.0~2.0 : 정규화 + 케이스별 가중치 랜덤 변동 (의외성 탐색).
              V=1.4 → 매 이터마다 각 케이스 가중치에 ±40% 균등 변동 적용.
              MMA가 이터마다 다른 케이스를 강조하여 단일 최소값에 고착되지 않음.
    권장: 다중 케이스 최적화 시 1.0, 다양성 탐색 시 1.2~1.5

--obj-type {sum|max|sum+max}  (기본: sum+max)
    sum     : f = Σ w_i·C_i. 모든 케이스 균등 최적화. 빠른 수렴.
    max     : f = (1/α)·log[Σ exp(α·w_i·C_i/C_i0)]
              KS(Kreisselmeier-Steinhauser) 함수 근사.
              log-sum-exp = 미분 가능한 softmax max 근사.
              α→∞: hard-max(최악 케이스만 반영), α→0: 단순 합산.
    sum+max : f = 0.5·f_sum + 0.5·f_max
              전체 균형과 최악 케이스 방어를 동시 달성. 권장.

--obj-alpha α  (기본: 10.0, obj-type=max/sum+max 시)
    KS softmax 온도 파라미터.
    1~3  : 모든 케이스 가중 평균에 가까움 (sum과 유사)
    10   : 최악 케이스에 ~60~80% 집중 (권장 시작값)
    50+  : hard-max에 근접, gradient가 나머지 케이스에서 거의 0 → 수렴 불안정
    권장: 5~20

--freq-penalty W F0_HZ
    고유진동수 패널티: P = W·max(0,F0-f₁)²/F0²
    f₁ < F0 인 경우에만 활성화 (단측 페널티).
    민감도는 첫 탄성 모드 형상 φ₁로 계산 (JAX vmap 재사용, 추가 FEA 없음):
      df₁/dh ≈ φ₁ᵀ(∂K/∂h)φ₁ / (4π²f₁)

[비드 배제 영역]
--exclude-rect CX,CY,W,H    중심(CX,CY) 기준 W×H mm 사각형 영역 내 비드 생성 금지.
                            반복 지정으로 복수 영역 추가 가능.

--exclude-poly X1,Y1,X2,Y2,...
                            꼭짓점 좌표(mm) 나열로 정의한 임의 다각형 영역.
                            꼭짓점은 순서대로 나열 (자동 닫힘), 최소 3개 꼭짓점.

[출력]
결과 디렉토리: results/D날짜_시간/ (실행 시작 시 자동 생성)
  paraview/iter_NNN.hdf        — 이터레이션별 변위·응력·고유모드 (ParaView)
  esl_se_report_iterNNN.png    — 시간분할 ESL SE 이력 리포트
  esl_peak_report_iterNNN.png  — 요소별 피크 ESL 리포트
  snapshots/iter_NNN.pkl       — 이터레이션 스냅샷 (재시작·후처리용)
  run.log                      — 전체 실행 로그
  final.k / design_dNN.k       — LS-DYNA 최적 비드 패턴 (다중 설계 시 번호별)

--no-gui   모니터링 GUI 비활성 (헤드리스 서버)
--no-viz   최종 3D 시각화 생략

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
추천 초기 설정
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[A] 빠른 탐색 (패턴 확인용)
  --iters 20 --min-width 80 --filter-type gaussian
  --bead-connect 120 --normalize-obj 1.0

[B] 표준 단일 설계 (권장 시작점)
  --iters 40 --min-width 100 --filter-type gaussian
  --projection 32 --bead-connect 120 --height-steps 2
  --sym-x --normalize-obj 1.0 --obj-type sum+max

[C] 체커보드 억제 강화 (픽셀화 문제 시)
  --iters 40 --min-width 120 --filter-type gaussian
  --projection 32 --bead-connect 150
  --sym-x --normalize-obj 1.0 --obj-type sum+max

[D] 다중 설계 탐색 (3개 다른 형태 생성)
  --iters 35 --n-designs 3
  --min-width 100 --filter-type gaussian --projection 32
  --bead-connect 120 --sym-x
  --normalize-obj 1.3 --obj-type sum+max
  --diversity-weight 0.3 --diversity-sigma 0.3 --diversity-noise 0.15

[E] 산업용 고신뢰성 (동적 ESL 포함)
  --iters 40 --min-width 100 --filter-type gaussian
  --projection 32 --bead-connect 120 --height-steps 2
  --sym-x --normalize-obj 1.0 --obj-type sum+max
  --w-dynamic 1.0 --w-peak 0.5 --add-inertia
  --n-windows 30 --n-top 10
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
    g.add_argument("--iters",       type=int,   default=20,   help="최대 반복 횟수 (기본: 15)")
    g.add_argument("--bead-height", type=float, default=15.0, help="최대 비드 높이 mm (기본: 10.0)")
    g.add_argument("--min-width",   type=float, default=120.0, help="최소 비드 폭 mm (기본: 30.0)")
    g.add_argument("--bead-area",   type=float, default=0.30, help="비드 점유 면적 비율 0~1 (기본: 0.30)")

    # 목적함수 옵션
    g = parser.add_argument_group("목적함수 옵션")
    g.add_argument("--normalize-obj",    type=float, default=1.0,
                   help="케이스별 정규화 활성화 및 가중치 변동 폭. "
                        "0=비활성(기본), 1.0=정규화(변동 없음), "
                        "1.4=정규화+케이스별 ±0.4 랜덤 가중치 부여 (의외성 유도)")
    g.add_argument("--obj-type",    choices=["sum", "max", "sum+max"], default="sum+max",
                   help="목적함수 유형: sum=가중합(기본), max=softmax 최악케이스, "
                        "sum+max=0.5·sum+0.5·softmax")
    g.add_argument("--obj-alpha",   type=float, default=10.0,
                   help="softmax 온도 (--obj-type max/sum+max 시). "
                        "1~3=케이스 평균에 가까움, 10=최악케이스 집중(기본), "
                        "50+=hard-max 근접(수렴 불안정 주의) (기본: 10.0)")
    g.add_argument("--freq-penalty", type=float, nargs=2, default=[0.0, 0.0],
                   metavar=("W", "F0_HZ"),
                   help="고유진동수 패널티: W·(max(0,F0-f1)/F0)². "
                        "예: --freq-penalty 5.0 50  → f1 < 50Hz 시 패널티 부과")

    # 다중 설계 탐색 옵션
    g = parser.add_argument_group("다중 설계 탐색 (Multi-Design Exploration)")
    g.add_argument("--n-designs",        type=int,   default=1,
                   help="생성할 설계 수 (기본: 1). >1이면 수렴 후 반발 패널티로 다른 설계 탐색")
    g.add_argument("--diversity-weight", type=float, default=0.3,
                   help="반발 패널티 강도 λ (기본: 0.3). 클수록 이전 설계에서 멀리 떨어짐")
    g.add_argument("--diversity-sigma",  type=float, default=0.3,
                   help="반발 범위 σ (기본: 0.3). 클수록 넓은 영역 회피")
    g.add_argument("--diversity-noise",  type=float, default=0.15,
                   help="설계 재시작 시 노이즈 강도 (기본: 0.15)")

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
    g.add_argument("--bead-connect",   type=float, default=120.0,
                   help="비드 자동 연결 최대 갭 mm. 0=비활성, >0=해당 갭 이하 단절 비드 연결 (기본: 120.0)")
    g.add_argument("--bead-connect-alg", type=str, default="closing",
                   choices=["closing", "mst", "geodesic", "hybrid"],
                   help="비드 연결 알고리즘 (기본: closing)")
    g.add_argument("--height-steps",   type=int,   default=1,    help="비드 이산화 단계 (기본: 2 → {0, h_max})")
    g.add_argument("--filter-type",    type=str,   default="linear", choices=["linear", "gaussian"],
                   help="공간 필터 커널. linear=hat(기본), gaussian=부드러운 덩어리 형태 유도")
    g.add_argument("--projection",     type=float, default=0.0,
                   help="Heaviside projection beta 최대값. 0=비활성, >0=활성화 (기본: 0). "
                        "8~16=부드러운 경계, 32=표준, 64+=hard 경계(수렴 불안정 주의)")
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
    g.add_argument("--dt",        type=float, default=0.005,  help="적분 시간 스텝 s (기본: None=CSV 샘플링 간격 그대로). ESL 모드에도 동일 적용.")
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
    g.add_argument("--n-save-esl",   type=int, default=30,
                   help="ESL 추출용 저장 스냅샷 수 (기본: 30). "
                        "n_windows*2 보다 작으면 자동으로 n_windows*2 로 상향. "
                        "줄일수록 동해석 메모리·I/O 감소.")
    g.add_argument("--use-global-z",    action="store_true",      help="글로벌 Z 궤적 직접 사용")
    g.add_argument("--contact-threshold", type=float, default=24516.6,
                   metavar="A",
                   help="Kabsch fit 제외 임계 상대 Z가속도 [mm/s²] (기본: 24517≈2.5g). "
                        "충격 임펄스 순간에 이 값을 초과하는 코너는 강체 fit에서 제외됩니다.")
    g.add_argument("--w-dynamic",       type=float, nargs='+', default=[1.0],
                   metavar="W",
                   help="시간분할 ESL 스냅샷 가중치 (기본: 1.0). CSV별 복수 지정 가능: --w-dynamic 1.0 0.8")
    g.add_argument("--w-peak",          type=float, default=0.0,  help="요소별 최대SE 피크 ESL 가중치 (기본: 0.0=비활성)")
    g.add_argument("--iterative-esl",   action="store_true", default=True,
                   help="매 이터레이션 ESL 재추출 (기본: 활성)")
    g.add_argument("--no-iterative-esl", action="store_false", dest="iterative_esl",
                   help="ESL 1회 추출 후 고정 (최적화 전 1회만 동해석)")
    g.add_argument("--parallel-scenarios", type=int, default=1, metavar="N",
                   help="복수 CSV 시나리오 동시 실행 프로세스 수 (기본: 1=순차). "
                        "2 이상 지정 시 ProcessPoolExecutor로 N개 시나리오를 동시 동해석. "
                        "CPU 코어 수 절반 권장. 예: --parallel-scenarios 4")
    g.add_argument("--esl-skip-tol",  type=float, default=0.0,
                   metavar="TOL",
                   help="반복 ESL 재추출 스킵 임계값 (기본: 0.0=항상 재실행). "
                        "이전 이터레이션 대비 Δh_rms < TOL 이면 동해석 생략하고 "
                        "직전 ESL을 재사용. 예: --esl-skip-tol 0.05")
    g.add_argument("--n-modes",       type=int, default=0,
                   metavar="N",
                   help="모달 중첩법 사용 모드 수 (기본: 0=직접 Newmark-β 적분). "
                        "0 초과 시 모달 중첩법으로 동해석 수행—직접 적분 대비 "
                        "수십~수백 배 빠름. 예: --n-modes 20")
    g.add_argument("--jax-dynamic",   action="store_true", default=False,
                   help="직접 Newmark-β를 JAX 가속 솔버로 실행 (기본: scipy)")
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
    g.add_argument("--no-gui",       action="store_true", help="모니터링 GUI 비활성화")
    g.add_argument("--no-viz",       action="store_true", help="최종 3D 시각화 생략")
    g.add_argument("--modal-modes",  type=int, default=20,
                   help="모달 재해석 모드 수 (기본: 20). 모니터 GUI Modal Analysis 탭에 반영됨")
    g.add_argument("--mesh-size", type=float, default=10.0, help="BC 탐색 기준 메시 크기 mm")

    # 입력 디렉토리 (topo_arg.txt + 하위 폴더 CSV 자동 수집)
    g = parser.add_argument_group("입력 디렉토리 (일괄 설정)")
    g.add_argument("--input-dir", type=str, default=None,
                   metavar="DIR",
                   help="입력 디렉토리 경로. 해당 폴더의 topo_arg.txt를 기본 옵션으로 로드하고, "
                        "하위 폴더(폴더명=loadcase 이름) 안의 첫 번째 .csv 파일을 "
                        "--dynamic-opts 항목으로 자동 추가. 파일명은 무관.")

    args = parser.parse_args(_resolve_input_dir(parser))

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
