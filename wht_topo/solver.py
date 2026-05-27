# -*- coding: utf-8 -*-
"""
solver.py
=========
Topography Optimization Solver — wht_solver 기반 정밀 구현.

[Topology vs Topography 구분]
    - Topology Optimization: 요소를 추가/제거, SIMP 밀도 변수 필요
    - Topography Optimization: 모든 요소 유지, 노드 Z 위치(비드 높이)를 변경
      → SIMP 불필요, WHTSolver.solve_static()을 그대로 사용 가능

핵심 아키텍처:
    - 설계 변수 : 바닥면 요소별 비드 높이 h_e ∈ [0, h_max]  (요소 기반)
                  요소의 4 노드가 동일 높이로 이동 → 평탄한 비드 평면 보장
                  노드 높이 = 인접 설계 요소 높이의 평균: h_n = mean(h_e for e adj n)
    - 민감도    : ∂C/∂h_e = Σ_{n∈e} (∂C/∂h_n) / n_adj(n)  (Chain-rule)
                  ∂C/∂h_n = Σ_{e∈N(n)} u_e^T (∂K_e/∂z_n) u_e  (Adjoint)
                  — QUAD4/TRIA3 모두 JAX Auto-Diff (vmap), 전체 K 재조립 불필요
    - 필터링    : 공간 필터(rmin)로 최소 비드 폭 제어 (요소 도심 기반)
    - 업데이트  : MMA (Method of Moving Asymptotes)

[목적함수 변형 — obj_type / normalize_obj / freq_weight]

  ① 기본 가중 합산 (obj_type='sum', normalize_obj=False, 기본값)
       f = (Σ w_i · C_i) / C_0
       · C_0 = 초기 총 컴플라이언스 (이터레이션 0 기준, 스케일 정규화용)
       · 민감도: df/dh = (Σ w_i · ∂C_i/∂h) / C_0
       · 주의: 하중 크기가 케이스마다 크게 다르면 큰 케이스가 목적함수를 지배.

  ② 케이스별 정규화 가중 합산 (normalize_obj=True)
       f = Σ w_i · (C_i / C_i0)
       · C_i0 = 이터레이션 0에서 케이스 i의 컴플라이언스 (케이스별 정규화)
       · 서로 다른 크기의 정적·동적 ESL 케이스를 균등하게 반영할 때 사용.
       · 민감도: df/dh = Σ (w_i/C_i0) · ∂C_i/∂h

  ③ Softmax 최악 케이스 (obj_type='max')
       f = (1/α) · log(Σ exp(α · w_i · C_i/C_i0))
       · α = obj_alpha (softmax 온도). 클수록 hard-max에 수렴, 작을수록 부드러운 평균.
       · 가장 나쁜 케이스에 집중 최적화 → Min-Max 강성화 효과.
       · 민감도: df/dh = Σ (softmax_weight_i · (w_i/C_i0) · ∂C_i/∂h)
         softmax_weight_i = exp(α·w_i·C_i/C_i0) / Σ exp(...)  (수치 안정: max-trick)

  ④ 혼합 목적함수 (obj_type='sum+max')
       f = 0.5 · f_sum + 0.5 · f_max
       · 평균 성능(sum)과 최악 케이스 방어(max)를 동시에 고려.

  ⑤ 고유진동수 패널티 (freq_weight > 0, freq_target > 0)
       P = freq_weight · max(0, freq_target - f₁)² / freq_target²
       · f₁ = 현재 비드 형상의 첫 번째 탄성 고유진동수 [Hz]
       · 위의 어떤 목적함수와도 병합 가능: f_total = f_obj + P
       · 민감도: dP/dh = -2·freq_weight·deficit/freq_target² · df₁/dh
         df₁/dh ≈ φ₁ᵀ(∂K/∂h)φ₁ / (4π²f₁)   — 단위 모달 질량 가정
         (φ₁ = 첫 번째 탄성 모드 형상, 동일한 JAX vmap 민감도 인프라 재사용)

[비드 패턴 형성 파이프라인 — 우주 구조 형성 유추]

  MMA 최적화는 초기 균일 밀도장에서 민감도(complinace gradient)를 따라
  비드 패턴을 진화시킨다. 우주 초기 밀도 요동이 중력 불안정으로 별과 은하를
  형성하는 과정과 구조적으로 유사하다.

  단계별 대응:
    초기 섭동(양자 요동)  ←→  --diversity-noise (초기 노이즈)
    Jeans length          ←→  --min-width R (필터 반경: 이 이하 파장의 요동 억제)
    중력 수렴             ←→  MMA 업데이트 (민감도 높은 곳으로 비드 집중)
    별 탄생 임계 밀도     ←→  --projection β (임계값 초과 시 비드로 확정)
    은하 팔 형성          ←→  --bead-connect-alg (분리된 비드 섬 연결 방식)

[비드 연결 알고리즘 — bead_connect_alg]

  MMA 수렴 후 비드가 분리된 섬(island)으로 나타날 수 있다.
  --bead-connect-alg 로 섬들을 잇는 경로 알고리즘을 선택한다.
  별들이 은하 팔로 이어지는 방식을 선택하는 것과 같다.

  closing   : 형태학적 폐합(Morphological Closing: Dilation → Erosion).
              binary 이미지를 팽창 후 수축하여 인근 섬 연결.
              단순하고 빠름. 섬 간격이 좁고 패턴이 균일할 때 적합.
              단점: 섬이 멀리 분산된 경우 미연결 잔존 가능.

  mst       : 최소 신장 트리(Minimum Spanning Tree, Kruskal).
              scipy.ndimage.label 로 섬 식별 → 섬 간 최단 픽셀 거리로 MST 구성
              → 각 엣지를 Bresenham 직선으로 브릿지.
              모든 섬의 연결을 수학적으로 보장. 경로가 직선적.
              권장: 섬이 많고 광범위하게 분산된 경우.

  geodesic  : 지오데식 최단 경로(multi-source Dijkstra).
              가장 큰 섬을 source로 비활성 요소(cost=1)를 통과하는
              최소 비용 경로로 각 섬을 연결. 기존 밀도장을 따라
              자연스러운 경로로 수렴하는 경향.
              권장: 비드 라인이 유기적으로 이어지길 원할 때.

  hybrid    : MST + Geodesic 순차 적용.
              MST로 전체 연결을 보장한 후 Geodesic으로 경로를 다듬음.
              두 알고리즘의 장점 결합. 계산 비용이 가장 높음.
              권장: 분산된 섬 + 자연스러운 경로 모두 원할 때.

wht_solver 활용:
    - WHTSolver.solve_static(WHTLoadCase)  → 각 하중 케이스별 변위 해석
    - wht_solver.wht_quad4_element._element_K_mitc4_plus  → 요소 K 계산
    - wht_solver.wht_tria3_element._element_K_tria3       → 요소 K 계산

Dependencies:
    wht_modeler, wht_solver, wht_topo.loads, wht_topo.mma
"""

import pickle
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from wht_modeler.wht_mesh_model import WHTMeshModel, WHTSPCEntry
from wht_solver.wht_solver import WHTSolver
from wht_topo.loads import StochasticLoadManager
from wht_topo.mma import MMAOptimizer

# JAX 민감도 함수 사전 정의 — QUAD4 (MITC4+)
from wht_solver.wht_quad4_element_jax import _element_K_mitc4_plus_jax

@jax.jit
def element_energy_jax(c1, c2, c3, c4, u_e, t, E, nu):
    K_e = _element_K_mitc4_plus_jax(c1, c2, c3, c4, t, E, nu)
    return jnp.dot(u_e, jnp.dot(K_e, u_e))

element_grad_jax      = jax.grad(element_energy_jax, argnums=(0, 1, 2, 3))
vmap_element_grad_jax = jax.jit(jax.vmap(element_grad_jax, in_axes=(0,0,0,0,0,0,0,0)))

# JAX 민감도 함수 사전 정의 — TRIA3 (MITC3+)
from wht_solver.wht_tria3_element_jax import _element_K_tria3_jax

@jax.jit
def element_energy_tria3_jax(c1, c2, c3, u_e, t, E, nu):
    K_e = _element_K_tria3_jax(c1, c2, c3, t, E, nu)
    return jnp.dot(u_e, jnp.dot(K_e, u_e))

element_grad_tria3_jax      = jax.grad(element_energy_tria3_jax, argnums=(0, 1, 2))
vmap_element_grad_tria3_jax = jax.jit(jax.vmap(element_grad_tria3_jax, in_axes=(0,0,0,0,0,0,0)))


def _von_mises_max(stress_arr: np.ndarray) -> float:
    """
    요소별 응력 배열에서 최대 Von-Mises 응력을 계산합니다.

    Parameters
    ----------
    stress_arr : (M, 6) ndarray
        [σx, σy, σz, τxy, τxz, τyz] 순서의 요소 응력 배열

    Returns
    -------
    float : 최대 Von-Mises 응력 [MPa]
    """
    sx  = stress_arr[:, 0]; sy  = stress_arr[:, 1]; sz  = stress_arr[:, 2]
    txy = stress_arr[:, 3]; txz = stress_arr[:, 4]; tyz = stress_arr[:, 5]
    vm = np.sqrt(0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2)
                 + 3.0 * (txy**2 + txz**2 + tyz**2))
    return float(np.max(vm))


def _bresenham_line(r0: int, c0: int, r1: int, c1: int):
    """Bresenham's line -- (row, col) 경로 반환."""
    points = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r0 < r1 else -1), (1 if c0 < c1 else -1)
    err = dr - dc
    r, c = r0, c0
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc
    return points


class WHTopographySolver:
    """
    물리적으로 올바른 Topography Optimization 엔진.

    설계 변수는 요소별 비드 높이 h_e이며, 요소 내 4 노드가 동일 높이로 이동합니다.
    노드 높이는 인접 설계 요소 높이의 평균으로 결정 → 울퉁불퉁한 형상 방지.
    SIMP/밀도 개념은 사용하지 않으며, WHTSolver.solve_static()을 직접 사용합니다.

    Parameters
    ----------
    model : WHTMeshModel
        유한요소 메시 모델.
    load_manager : StochasticLoadManager
        하중 케이스 및 경계 조건 생성기.
    bead_height_max : float
        최대 비드 높이 [mm]. 기본값 10.0.
    bead_height_ratio : float
        목표 비드 면적 비율 (bead height constraint). 기본값 0.3.
    min_width : float
        최소 비드 폭 [mm], 공간 필터 반경으로 사용. 기본값 80.0.
    draw_dir : tuple
        비드 돌출 방향 벡터 (정규화). 기본값 (0,0,1).
    weights : dict
        하중 케이스별 가중치 {"bending", "twisting", "lifting"}.
    mesh_size_z : float
        플랜지 노드 탐색 허용 오차 계산에 사용되는 메시 크기 [mm].
    fd_dz : float
        민감도 계산 시 Z 방향 중앙차분 섭동량 [mm]. 기본값 0.5.
    sym_x : bool
        좌우 대칭 제약 조건 활성화 여부. 기본값 False.
    normalize_obj : bool
        True 이면 케이스별 초기 컴플라이언스(C_i0)로 각 케이스를 정규화.
        f = Σ w_i·(C_i/C_i0). 정적·동적 ESL 케이스의 크기가 달라도 균등 반영.
        기본값 False (초기 총 컴플라이언스 C_0으로 단순 스케일링).
    obj_type : str
        목적함수 유형. 'sum'(기본), 'max'(softmax 최악 케이스), 'sum+max'(혼합).
        - 'sum'    : f = Σ w_i·C_i  (또는 normalize_obj=True 시 Σ w_i·C_i/C_i0)
        - 'max'    : f = (1/α)·log(Σ exp(α·w_i·C_i/C_i0))  — 최악 케이스 집중 최적화
        - 'sum+max': f = 0.5·f_sum + 0.5·f_max             — 평균+최악 균형
    obj_alpha : float
        Softmax 온도 파라미터 (obj_type='max'/'sum+max' 시 유효). 기본값 10.0.
        클수록 hard-max에 수렴(가장 나쁜 케이스만 반영), 작을수록 soft-average.
    freq_weight : float
        고유진동수 패널티 가중치 λ. 0이면 비활성. 기본값 0.0.
        P = λ·max(0, freq_target-f₁)²/freq_target²
    freq_target : float
        목표 최저 탄성 고유진동수 [Hz]. f₁ < freq_target 인 경우 패널티 부과.
        민감도는 JAX vmap을 통해 모드 형상 φ₁로 계산: df₁/dh ≈ φ₁ᵀ∂K/∂h φ₁/(4π²f₁).
    """

    # wht_solver와 동일한 쉘 요소 타입 집합
    _QUAD_TYPES  = frozenset({'QUAD4', 'QUAD'})
    _TRIA_TYPES  = frozenset({'TRIA3', 'TRIA'})
    _SHELL_TYPES = frozenset({'QUAD4', 'QUAD', 'TRIA3', 'TRIA'})

    def __init__(
        self,
        model: WHTMeshModel,
        load_manager: Optional[StochasticLoadManager] = None,
        constraints=None,
        bead_height_max: float = 10.0,
        bead_height_ratio: float = 0.3,
        min_width: float = 80.0,
        draw_dir: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        weights: Optional[Dict[str, float]] = None,
        mesh_size_z: float = 10.0,
        fd_dz: float = 0.5,
        sym_x: bool = False,
        bead_connect: float = 0.0,
        bead_connect_alg: str = "closing",
        bead_steps: int = 0,
        filter_type: str = "linear",
        use_projection: bool = False,
        proj_beta_max: float = 32.0,
        proj_eta: float = 0.5,
        load_cases: Optional[List[Tuple["WHTLoadCase", float]]] = None,
        load_case_provider=None,
        out_dir=None,
        normalize_obj: bool = False,
        weight_variation: float = 0.0,
        obj_type: str = "sum",
        obj_alpha: float = 10.0,
        freq_weight: float = 0.0,
        freq_target: float = 0.0,
        exclude_zones: Optional[List[dict]] = None,
        n_workers: int = 4,
        modal_modes: int = 20,
        vol_ramp_iters: int = 5,
        bidirectional: bool = False,
        min_width_init: float = -1.0,
        min_width_ramp_iters: int = 0,
    ):
        self.model          = model
        self.load_manager   = load_manager
        self.h_max          = bead_height_max
        self.h_ratio        = bead_height_ratio
        self.vol_ramp_iters = vol_ramp_iters
        self.bidirectional  = bidirectional    # True: x→(x-0.5)*2*h_max, ±방향 모두 허용
        _rmin_start         = min_width_init if min_width_init > min_width else min_width
        self.rmin           = _rmin_start
        self._rmin_start    = _rmin_start    # 루프 시작 시 rmin (init > final 이면 큰 값)
        self._rmin_final    = min_width      # 목표 최종 rmin (--min-width)
        self._rmin_ramp_n   = min_width_ramp_iters
        self._rmin_current  = _rmin_start
        self.mesh_size_z    = mesh_size_z
        self.fd_dz          = fd_dz
        self.sym_x          = sym_x
        self.bead_connect     = bead_connect        # 0=비활성, >0=연결 최대 갭(mm)
        self.connect_gap      = bead_connect        # 하위 호환용 alias
        self.bead_connect_alg = bead_connect_alg   # 'closing'|'mst'|'geodesic'|'hybrid'
        self.bead_steps     = bead_steps
        self.filter_type    = filter_type         # "linear" | "gaussian"
        self.use_projection = use_projection      # Heaviside projection 활성화
        self.proj_beta_max  = proj_beta_max       # beta 최대값 (continuation)
        # Heaviside 임계값 (0.5=중간)
        self.proj_eta       = proj_eta
        # 다중 설계 탐색용 반발 패널티
        self.diversity_weight      = 0.0   # λ: 반발 강도 (0=비활성)
        self.diversity_sigma       = 0.3   # σ: 반발 범위 (설계 변수 공간)
        self.diversity_start_iter  = -1    # -1=수렴 후 등록 방식, >=0=해당 iter부터 적용
        self._reference_designs: List[np.ndarray] = []  # 반발 기준 설계들
        # 동적 시나리오 정보 (monitor UI 재해석용) — run_topo.py에서 주입
        # 형식: [{"name": str, "csv_path": str, "t_start": float|None,
        #          "t_end": float|None, "add_inertia": bool,
        #          "use_global_z": bool, "n_modes": int}]
        self.dynamic_scenarios: List[dict] = []
        self.weights        = weights or {"bending": 1.0, "twisting": 1.5, "lifting": 1.2}
        self.load_cases_input    = load_cases
        self._load_case_provider = load_case_provider
        self._out_dir            = out_dir  # results/D날짜_시간/ — None이면 자동 생성
        self.normalize_obj       = normalize_obj
        self.weight_variation    = weight_variation  # >0이면 매 이터마다 케이스 가중치 랜덤 변동
        self.obj_type            = obj_type   # "sum" | "max" | "sum+max"
        self.obj_alpha           = obj_alpha  # softmax 온도 (클수록 hard-max에 가까움)
        self.freq_weight         = freq_weight
        self.freq_target         = freq_target  # Hz
        self._C_0_cases: Dict[str, float] = {}  # 케이스별 기준 컴플라이언스 (정규화용)
        self.exclude_zones       = exclude_zones or []  # 비드 배제 영역 목록
        self.n_workers           = max(1, n_workers)
        self.modal_modes         = max(1, int(modal_modes))

        # ... (중략) ...
        self._beta = 1.0
        
        self.constraints    = constraints or []
        self.freq_min       = 0.0
        self.freq_max       = 0.0
        for c in self.constraints:
            if hasattr(c, "min_freq"): self.freq_min = c.min_freq
            if hasattr(c, "max_freq"): self.freq_max = c.max_freq

        # 비드 돌출 방향 정규화
        d = np.array(draw_dir, dtype=np.float64)
        self.bead_dir = d / (np.linalg.norm(d) + 1e-10)

        # 재료/속성 기본값 보장
        self._ensure_material_properties()

        # 노드/요소 인덱스 구조 설정
        self.sorted_nids = model.sorted_node_ids()
        self.nid_to_idx  = {nid: i for i, nid in enumerate(self.sorted_nids)}
        self.elem_ids    = sorted(model.elements.keys())

        # 원본 좌표 저장 (최적화 도중 복원 및 3D 돌출용)
        self._coords_orig = {
            nid: np.array([model.nodes[nid].x, model.nodes[nid].y, model.nodes[nid].z])
            for nid in self.sorted_nids
        }

        # 설계 요소 식별 (바닥면 쉘 요소에서 플랜지 포함 요소 제외)
        # 설계 변수 = 요소별 비드 높이 h_e → 요소 4 노드가 동일 높이로 이동
        print(" -> [Solver] 설계 요소(비드 적용 가능 바닥면) 탐색 중...")
        self._design_elems, self._design_nids = self._find_design_elements()
        self._n_design          = len(self._design_elems)
        self._design_elem_to_idx = {eid: i for i, eid in enumerate(self._design_elems)}
        self._design_nid_to_idx  = {nid: i for i, nid in enumerate(self._design_nids)}
        n_int = len(self._design_nids)
        print(f"    - 설계 요소 수: {self._n_design}개  /  내부 노드 수: {n_int}개")

        # 집합(Aggregation) 배열 사전 구축
        # (elem_idx, node_internal_idx) 쌍 → elem→node 높이 분배, node→elem 민감도 집계
        aggr_src_list, aggr_dst_list = [], []
        node_adj_count = np.zeros(n_int, dtype=np.float64)
        for eidx, eid in enumerate(self._design_elems):
            for nid in self.model.elements[eid].node_ids:
                nidx = self._design_nid_to_idx.get(nid, -1)
                if nidx >= 0:
                    aggr_src_list.append(nidx)    # 내부 노드 인덱스
                    aggr_dst_list.append(eidx)    # 설계 요소 인덱스
                    node_adj_count[nidx] += 1.0
        self._aggr_src          = np.array(aggr_src_list, dtype=int)
        self._aggr_dst          = np.array(aggr_dst_list, dtype=int)
        self._node_adj_count_arr = node_adj_count
        # aggr_w[k] = 1 / n_adj(node) — 민감도 집계 및 노드 높이 역산에 사용
        self._aggr_w = 1.0 / (node_adj_count[self._aggr_src] + 1e-12)

        # 공간 필터 행렬 (최소 비드 폭 제어)
        self._H = self._build_filter()

        # 정적 하중 케이스 초기화 (이터레이션 간 고정)
        # load_cases_input is not None: 외부에서 명시적으로 주입 (빈 리스트 포함 — --no-static)
        if self.load_cases_input is not None:
            self._static_load_cases = list(self.load_cases_input)
            print(f" -> [Solver] 외부 주입 정적 하중 케이스 {len(self._static_load_cases)}개 로드됨"
                  f"{'  (--no-static: 정적 하중 없음)' if len(self._static_load_cases) == 0 else ''}.")
        elif self.load_manager:
            print(" -> [Solver] 하중 케이스 생성 중 (하중별 개별 BC 적용)...")
            self._static_load_cases = self.load_manager.get_load_cases(
                mesh_size_z=mesh_size_z,
                weights=self.weights,
                loads=getattr(self.load_manager, '_scaled_loads', None),
            )
        else:
            print(" -> [Error] 하중 케이스 정보가 없습니다.")
            self._static_load_cases = []

        if self._load_case_provider is not None:
            print(" -> [Solver] 동적 ESL provider 등록됨 — 매 이터레이션 재추출.")
            self._load_cases = list(self._static_load_cases)  # 초기값: 정적만
        else:
            self._load_cases = list(self._static_load_cases)

        # 좌우 대칭 매핑 (X-mid plane 기준)
        self._sym_map = None
        if self.sym_x:
            self._sym_map = self._build_symmetry_map()

        # 비드 연결 그리드 (Morphological Closing용)
        self._connect_grid = None
        if self.bead_connect > 0:
            self._connect_grid = self._build_connect_grid()
            print(f" -> [Solver] 비드 연결 활성화: alg={self.bead_connect_alg}  gap={self.bead_connect:.0f}mm")

        # 초기 비드 높이 — vol_frac 근방 + 큰 랜덤 분산
        # 균일 x=1.0으로 시작하면 OC 업데이트의 alpha clip이 모든 요소에 동시
        # 적용되어 공간 정보가 소실됨(vol_frac 도달까지 ~5 이터 균일 감소).
        # 대신 초기부터 [0, ~0.6] 범위에 크게 분산시키면 raw x 자체가 비드/비비드
        # 패턴을 가지므로, MMA는 mean(x)=vol_frac 제약 하에 패턴을 정제만 수행.
        rng = np.random.default_rng(0)
        x0 = rng.uniform(0.0, 1.0, self._n_design)
        if self.bidirectional:
            # 양방향: x=0.5 중심, 분산이 큰 초기값 (일부는 +, 일부는 -)
            # area = mean(|x-0.5| > 0.1) ≈ 1.0 에서 시작해 vol_ramp로 감소
            x0 = np.clip(x0, 0.01, 0.99)
            self.heights = (x0 - 0.5) * 2.0 * self.h_max
        else:
            x0 = np.clip(x0 * (bead_height_ratio / max(x0.mean(), 1e-6)), 0.01, 0.99)
            self.heights = x0 * self.h_max
        self.mma = MMAOptimizer(n_vars=self._n_design, vol_frac=bead_height_ratio)

        # 민감도 계산용 정적 데이터 사전 캐싱 (최적화 루프 내 재생성 방지)
        self._bead_dir_jnp = jnp.array(self.bead_dir, dtype=jnp.float64)
        self._build_sensitivity_cache()

    # ─────────────── 전처리 ───────────────

    def _build_sensitivity_cache(self):
        """
        민감도 계산에 필요한 정적 데이터를 사전에 배열로 구성합니다.
        최적화 루프 내 반복 생성을 방지하기 위해 __init__ 마지막에 한 번만 호출됩니다.

        구축 대상:
          QUAD4: _quad_nids, _quad_t/E/nu, _quad_dof_idx, _quad_node_design_idx
          TRIA3: _tria_nids, _tria_t/E/nu, _tria_dof_idx, _tria_node_design_idx
        """
        quad_nids_list, quad_t_list, quad_E_list, quad_nu_list = [], [], [], []
        quad_dof_list, quad_nd_idx_list = [], []
        tria_nids_list, tria_t_list, tria_E_list, tria_nu_list = [], [], [], []
        tria_dof_list, tria_nd_idx_list = [], []

        for eidx, eid in enumerate(self.elem_ids):
            elem = self.model.elements[eid]
            nids = elem.node_ids
            prop = self.model.properties[elem.pid]
            mat  = self.model.materials[prop.mid]
            dofs = [self.nid_to_idx[n] * 6 + d for n in nids for d in range(6)]
            nd_idx = [self._design_nid_to_idx.get(n, -1) for n in nids]

            if elem.type.upper() in self._QUAD_TYPES:
                quad_nids_list.append(nids)
                quad_t_list.append(prop.t)
                quad_E_list.append(mat.E)
                quad_nu_list.append(mat.nu)
                quad_dof_list.append(dofs)
                quad_nd_idx_list.append(nd_idx)
            elif elem.type.upper() in self._TRIA_TYPES:
                if any(i != -1 for i in nd_idx):
                    tria_nids_list.append(nids)
                    tria_t_list.append(prop.t)
                    tria_E_list.append(mat.E)
                    tria_nu_list.append(mat.nu)
                    tria_dof_list.append(dofs)
                    tria_nd_idx_list.append(nd_idx)

        n_quad = len(quad_nids_list)
        self._n_quad = n_quad
        if n_quad > 0:
            self._quad_nids            = np.array(quad_nids_list, dtype=int)
            self._quad_t               = np.array(quad_t_list,    dtype=np.float64)
            self._quad_E               = np.array(quad_E_list,    dtype=np.float64)
            self._quad_nu              = np.array(quad_nu_list,   dtype=np.float64)
            self._quad_dof_idx         = np.array(quad_dof_list,  dtype=int)  # (n_quad, 24)
            self._quad_node_design_idx = np.array(quad_nd_idx_list, dtype=int)  # (n_quad, 4)
        else:
            self._quad_nids = self._quad_t = self._quad_E = self._quad_nu = None
            self._quad_dof_idx = self._quad_node_design_idx = None

        n_tria = len(tria_nids_list)
        self._n_tria = n_tria
        if n_tria > 0:
            self._tria_nids            = np.array(tria_nids_list, dtype=int)   # (n_tria, 3)
            self._tria_t               = np.array(tria_t_list,    dtype=np.float64)
            self._tria_E               = np.array(tria_E_list,    dtype=np.float64)
            self._tria_nu              = np.array(tria_nu_list,   dtype=np.float64)
            self._tria_dof_idx         = np.array(tria_dof_list,  dtype=int)  # (n_tria, 18)
            self._tria_node_design_idx = np.array(tria_nd_idx_list, dtype=int)  # (n_tria, 3)
        else:
            self._tria_nids = self._tria_t = self._tria_E = self._tria_nu = None
            self._tria_dof_idx = self._tria_node_design_idx = None

        print(f"    - 민감도 캐시 구축 완료: QUAD4 {n_quad}개 / TRIA3 {n_tria}개 (설계 노드 인접)")

    def _ensure_material_properties(self):
        """모델에 재료 및 속성이 없으면 Steel 기본값을 추가합니다."""
        if not self.model.materials:
            print("    [경고] 재료 정보 없음 → Steel 기본값 적용 (E=210GPa, nu=0.3, rho=7.85e-9)")
            self.model.add_material(1, E=210000.0, nu=0.3, rho=7.85e-9)
        if not self.model.properties:
            print("    [경고] 속성 정보 없음 → PSHELL t=1.5mm 적용")
            self.model.add_property(1, "PSHELL", t=1.5, mid=1)
            for elem in self.model.elements.values():
                elem.pid = 1

    def _in_exclusion_zone(self, cx: float, cy: float) -> bool:
        """
        XY 평면상의 점 (cx, cy)이 배제 영역 중 하나에 포함되는지 판정합니다.

        지원 영역 유형
        --------------
        rect  : {'type':'rect', 'cx':, 'cy':, 'w':, 'h':}
                  cx±w/2, cy±h/2 사각형 내부 여부
        poly  : {'type':'poly', 'vertices': [(x,y), ...]}
                  임의 다각형 내부 여부 (Ray-casting)
        """
        for zone in self.exclude_zones:
            ztype = zone.get('type', '')
            if ztype == 'rect':
                if (abs(cx - zone['cx']) <= zone['w'] / 2.0 and
                        abs(cy - zone['cy']) <= zone['h'] / 2.0):
                    return True
            elif ztype == 'poly':
                verts = zone['vertices']
                n = len(verts)
                if n < 3:
                    continue
                inside = False
                j = n - 1
                for i in range(n):
                    xi, yi = verts[i]
                    xj, yj = verts[j]
                    if ((yi > cy) != (yj > cy)) and (
                            cx < (xj - xi) * (cy - yi) / (yj - yi + 1e-12) + xi):
                        inside = not inside
                    j = i
                if inside:
                    return True
        return False

    def _find_design_elements(self) -> Tuple[List[int], List[int]]:
        """
        비드 적용 가능한 바닥면 쉘 요소를 탐색합니다.

        선택 기준:
          - 쉘 요소(QUAD4/TRIA3)만 대상
          - 요소 도심 Z ≤ z_threshold (바닥면 기준)
          - 플랜지 노드를 하나라도 포함한 요소는 제외

        Returns
        -------
        design_elems : List[int]  — 설계 요소 ID 리스트 (정렬됨)
        design_nids  : List[int]  — 설계 요소에 속한 고유 노드 ID 리스트 (정렬됨)
        """
        if self.load_manager is not None:
            flange_nids = set(self.load_manager.get_boundary_nodes(
                mesh_size_z=self.mesh_size_z
            ))
        else:
            flange_nids = set()

        all_z = [self.model.nodes[nid].z for nid in self.sorted_nids]
        z_min = min(all_z)
        z_threshold = z_min + max(self.mesh_size_z * 0.5, 5.0)

        # ─── Pass 1: 둘레 1-ring 침식
        # 플랜지/경사면 노드를 포함하는 요소(ring 0)의 모든 노드를 "확장 둘레"로 등록.
        # Pass 2에서 이 확장 둘레 노드를 가진 요소는 설계 영역에서 제외 → 둘레로부터
        # 1개 요소 안쪽까지 모두 비드 생성 영역에서 빼냄 (경사 인접부 노이즈 방지).
        extended_boundary_nids = set(flange_nids)
        for eid in self.elem_ids:
            elem = self.model.elements[eid]
            if elem.type.upper() not in self._SHELL_TYPES:
                continue
            nids = elem.node_ids
            coords = [self.model.nodes[nid] for nid in nids]
            z_centroid = float(np.mean([n.z for n in coords]))
            if z_centroid > z_threshold:
                continue   # 바닥면 요소만 고려
            if any(nid in flange_nids for nid in nids):
                extended_boundary_nids.update(nids)

        design_elems: List[int] = []
        all_design_nids: set = set()
        n_excluded = 0
        n_eroded = 0

        for eid in self.elem_ids:
            elem = self.model.elements[eid]
            if elem.type.upper() not in self._SHELL_TYPES:
                continue
            nids = elem.node_ids
            if any(nid in extended_boundary_nids for nid in nids):
                if any(nid in flange_nids for nid in nids):
                    pass   # ring 0 (직접 둘레), 별도 카운트 안 함
                else:
                    n_eroded += 1   # ring 1 (내측 1칸)
                continue
            coords = [self.model.nodes[nid] for nid in nids]
            z_centroid = float(np.mean([n.z for n in coords]))
            if z_centroid > z_threshold:
                continue
            cx = float(np.mean([n.x for n in coords]))
            cy = float(np.mean([n.y for n in coords]))
            if self.exclude_zones and self._in_exclusion_zone(cx, cy):
                n_excluded += 1
                continue
            design_elems.append(eid)
            all_design_nids.update(nids)

        if n_eroded:
            print(f"    - 둘레 1-ring 침식: {n_eroded}개 내측 요소 추가 제외")
        if n_excluded:
            print(f"    - 배제 영역 필터: {n_excluded}개 요소 제외됨 ({len(self.exclude_zones)}개 영역)")

        return design_elems, sorted(all_design_nids)

    def _elem_centroids(self) -> np.ndarray:
        """설계 요소 도심 좌표 배열 (n_design, 3)을 반환합니다."""
        return np.array([
            np.mean([[self._coords_orig[nid][0],
                      self._coords_orig[nid][1],
                      self._coords_orig[nid][2]]
                     for nid in self.model.elements[eid].node_ids], axis=0)
            for eid in self._design_elems
        ])

    def _build_symmetry_map(self) -> np.ndarray:
        """
        X-mid plane을 기준으로 대칭 요소 인덱스 맵을 생성합니다.
        허용 오차는 메시 중위 간격의 60%로 자동 산정합니다.
        """
        from scipy.spatial import KDTree

        print(" -> [Solver] 좌우 대칭(Sym-X) 요소 매핑 중...")
        centroids = self._elem_centroids()
        x_mid = (centroids[:, 0].min() + centroids[:, 0].max()) / 2.0

        tree = KDTree(centroids)

        # 메시 간격 추정: 각 요소의 두 번째 최근접 거리(자기 자신 제외) 중위값
        dists_nn, _ = tree.query(centroids, k=2)
        median_spacing = float(np.median(dists_nn[:, 1]))
        tol = median_spacing * 0.6
        print(f"    x_mid={x_mid:.1f} mm   median_spacing={median_spacing:.1f} mm   tol={tol:.1f} mm")

        sym_map = np.arange(self._n_design)
        n_matched = 0
        for i in range(self._n_design):
            target = centroids[i].copy()
            target[0] = 2.0 * x_mid - target[0]
            dist, idx = tree.query(target)
            if dist < tol:
                sym_map[i] = idx
                n_matched += 1

        n_self = int(np.sum(sym_map == np.arange(self._n_design)))
        print(f"    매핑 결과: {n_matched}/{self._n_design} 쌍 매칭  |  자기매핑(대칭쌍 없음): {n_self}개")
        if n_self > self._n_design * 0.1:
            print(f"    [경고] 자기매핑 비율 {n_self/self._n_design*100:.0f}% — 메시가 X-mid 기준으로 비대칭이거나 tol이 부족합니다.")

        return sym_map

    def _build_connect_grid(self) -> dict:
        """
        Morphological Closing 연산을 위한 2D 그리드 인덱스를 사전 구축합니다.

        설계 요소 도심의 X, Y 좌표를 정규 격자에 매핑하여, 클로징 연산 시
        반복적인 좌표 변환 없이 O(1)으로 접근할 수 있게 합니다.

        Returns
        -------
        dict : {"gi": ndarray, "gj": ndarray, "nx": int, "ny": int, "spacing": float}
        """
        coords = self._elem_centroids()

        xs = np.sort(np.unique(np.round(coords[:, 0], 0)))
        ys = np.sort(np.unique(np.round(coords[:, 1], 0)))
        dx = float(np.median(np.diff(xs))) if len(xs) > 1 else self.rmin
        dy = float(np.median(np.diff(ys))) if len(ys) > 1 else self.rmin
        spacing = min(dx, dy)

        x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
        x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
        nx = int(np.ceil((x_max - x_min) / spacing)) + 2
        ny = int(np.ceil((y_max - y_min) / spacing)) + 2

        gi = np.clip(((coords[:, 0] - x_min) / spacing).astype(int), 0, nx - 1)
        gj = np.clip(((coords[:, 1] - y_min) / spacing).astype(int), 0, ny - 1)

        # 그리드 인덱스 → 설계 노드 인덱스 역매핑 (빠른 lookup용)
        grid_to_node = {}
        for k in range(self._n_design):
            grid_to_node[(gj[k], gi[k])] = k

        return {
            "gi": gi, "gj": gj, "nx": nx, "ny": ny,
            "spacing": spacing, "grid_to_node": grid_to_node,
        }

    # 8방향 이웃 오프셋 (대각 포함)
    _NEIGHBORS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def _apply_bead_connect(self, x: np.ndarray, threshold: float = 0.1):
        """알고리즘 선택 디스패처."""
        alg = getattr(self, 'bead_connect_alg', 'closing')
        if alg == 'mst':
            return self._apply_bead_connect_mst(x, threshold)
        elif alg == 'geodesic':
            return self._apply_bead_connect_geodesic(x, threshold)
        elif alg == 'hybrid':
            x1, bi1 = self._apply_bead_connect_closing(x, threshold)
            x2, bi2 = self._apply_bead_connect_mst(x1, threshold)
            return x2, bi1 + bi2
        else:  # 'closing' default
            return self._apply_bead_connect_closing(x, threshold)

    def _apply_bead_connect_closing(self, x: np.ndarray, threshold: float = 0.1):
        """
        Morphological Closing으로 단절된 비드 노드를 연결합니다.

        Parameters
        ----------
        x : (n_design,) 설계 변수 벡터 [0, 1]
        threshold : float
            활성 비드 판단 임계값 (기본 0.1)

        Returns
        -------
        x_new : (n_design,) 연결이 적용된 설계 변수 벡터
        bridge_idx : list[int] 이번에 새로 채워진 bridge 노드 인덱스
        """
        from scipy.ndimage import binary_dilation, binary_erosion

        cg       = self._connect_grid
        gi, gj   = cg["gi"], cg["gj"]
        nx, ny   = cg["nx"], cg["ny"]
        spacing  = cg["spacing"]
        g2n      = cg["grid_to_node"]

        # 1. 설계 변수를 2D 그리드에 래스터화
        grid = np.zeros((ny, nx), dtype=bool)
        for k in range(self._n_design):
            if x[k] > threshold:
                grid[gj[k], gi[k]] = True

        # 2. Morphological Closing: Dilation → Erosion (8-connectivity)
        #    n_iter = ceil(connect_gap / spacing) 셀만큼 팽창 후 동일하게 수축
        from scipy.ndimage import generate_binary_structure
        struct = generate_binary_structure(2, 2)  # 8-connectivity kernel
        n_iter = max(1, int(np.ceil(self.connect_gap / spacing)))
        dilated = grid.copy()
        for _ in range(n_iter):
            dilated = binary_dilation(dilated, structure=struct)
        closed = dilated.copy()
        for _ in range(n_iter):
            closed = binary_erosion(closed, structure=struct)

        # 3. bridge 노드 식별 및 값 승격
        bridge_mask = closed & ~grid  # 원래 비활성이었으나 채워진 셀
        x_new = x.copy()
        bridge_idx = []
        for k in range(self._n_design):
            if bridge_mask[gj[k], gi[k]]:
                bridge_idx.append(k)
                # 8방향 활성 이웃의 최대값으로 승격 (MMA가 유지하기 충분한 수준)
                row, col = gj[k], gi[k]
                neighbor_vals = []
                for dr, dc in self._NEIGHBORS_8:
                    nb = g2n.get((row + dr, col + dc))
                    if nb is not None and x[nb] > threshold:
                        neighbor_vals.append(x[nb])
                x_new[k] = max(x_new[k], max(neighbor_vals) if neighbor_vals else threshold + 0.1)
        return x_new, bridge_idx

    def _apply_bead_connect_mst(self, x: np.ndarray, threshold: float = 0.1):
        """
        MST(Minimum Spanning Tree) 기반 비드 섬 연결.

        [알고리즘]
        1. scipy.ndimage.label로 8-연결 컴포넌트 식별
        2. 컴포넌트 간 최근접 픽셀 쌍 거리 행렬 구성
        3. Kruskal MST로 최소 연결 트리 계산
        4. 각 MST 엣지: Bresenham 직선으로 bridge 경로 생성
        5. 경로 상 비활성 노드 승격

        [특징] 모든 섬을 반드시 연결. closing보다 공격적.
        """
        from scipy.ndimage import label as ndlabel
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree

        cg = self._connect_grid
        gi, gj = cg["gi"], cg["gj"]
        nx, ny = cg["nx"], cg["ny"]
        g2n    = cg["grid_to_node"]

        grid = np.zeros((ny, nx), dtype=bool)
        for k in range(self._n_design):
            if x[k] > threshold:
                grid[gj[k], gi[k]] = True

        struct = np.ones((3, 3), dtype=int)
        labeled, n_comp = ndlabel(grid, structure=struct)
        if n_comp <= 1:
            return x.copy(), []

        # 각 컴포넌트 픽셀 목록
        comp_pix = {}
        for r in range(ny):
            for c in range(nx):
                lbl = labeled[r, c]
                if lbl > 0:
                    comp_pix.setdefault(lbl, []).append((r, c))

        comp_ids = sorted(comp_pix.keys(), key=lambda cid: -len(comp_pix[cid]))
        n = len(comp_ids)

        # 컴포넌트 간 최근접 픽셀 쌍 거리 (샘플링으로 속도 확보)
        MAX_SAMPLE = 200

        def _sample(pix):
            if len(pix) <= MAX_SAMPLE:
                return np.array(pix)
            idx = np.random.choice(len(pix), MAX_SAMPLE, replace=False)
            return np.array(pix)[idx]

        dist_mat  = np.full((n, n), np.inf)
        near_pair = {}  # (i,j) -> (pa, pb)

        for i in range(n):
            arr_a = _sample(comp_pix[comp_ids[i]])
            for j in range(i + 1, n):
                arr_b = _sample(comp_pix[comp_ids[j]])
                dists = np.sum(
                    (arr_a[:, None, :] - arr_b[None, :, :]) ** 2, axis=2
                )
                mi = np.unravel_index(np.argmin(dists), dists.shape)
                d  = float(np.sqrt(dists[mi]))
                dist_mat[i, j] = dist_mat[j, i] = d
                near_pair[(i, j)] = (tuple(arr_a[mi[0]]), tuple(arr_b[mi[1]]))
                near_pair[(j, i)] = (tuple(arr_b[mi[1]]), tuple(arr_a[mi[0]]))

        mst = minimum_spanning_tree(csr_matrix(dist_mat))
        rows, cols = mst.nonzero()

        x_new = x.copy()
        bridge_idx = []
        for ei, ej in zip(rows, cols):
            pa, pb = near_pair[(ei, ej)]
            for r, c in _bresenham_line(pa[0], pa[1], pb[0], pb[1]):
                if 0 <= r < ny and 0 <= c < nx:
                    k = g2n.get((r, c))
                    if k is not None and x[k] <= threshold:
                        bridge_idx.append(k)
                        nbrs = [
                            x[g2n[(r+dr, c+dc)]]
                            for dr, dc in self._NEIGHBORS_8
                            if g2n.get((r+dr, c+dc)) is not None
                            and x[g2n[(r+dr, c+dc)]] > threshold
                        ]
                        x_new[k] = max(x_new[k], max(nbrs) if nbrs else threshold + 0.1)
        return x_new, bridge_idx

    def _apply_bead_connect_geodesic(self, x: np.ndarray, threshold: float = 0.1):
        """
        Dijkstra 최소비용 경로 기반 비드 섬 연결.

        [알고리즘]
        1. 컴포넌트 식별 (label)
        2. 가장 큰 컴포넌트를 '본체'로 지정
        3. 각 소형 섬 -> 본체까지 Dijkstra 최단 경로 탐색
           - 비활성 노드 비용 = 1.0 (통과 비용)
           - 활성 노드 비용  = 0.0 (이미 비드)
        4. 경로 상 비활성 노드 승격

        [특징] 기존 비드를 최대한 활용하는 자연스러운 경로.
               구불구불한 실제 구조에 적합.
        """
        from scipy.ndimage import label as ndlabel
        import heapq

        cg = self._connect_grid
        gi, gj = cg["gi"], cg["gj"]
        nx, ny = cg["nx"], cg["ny"]
        g2n    = cg["grid_to_node"]

        grid = np.zeros((ny, nx), dtype=bool)
        for k in range(self._n_design):
            if x[k] > threshold:
                grid[gj[k], gi[k]] = True

        struct = np.ones((3, 3), dtype=int)
        labeled, n_comp = ndlabel(grid, structure=struct)
        if n_comp <= 1:
            return x.copy(), []

        # 컴포넌트 크기 -> 가장 큰 것이 본체
        comp_sizes = {}
        for r in range(ny):
            for c in range(nx):
                lbl = labeled[r, c]
                if lbl > 0:
                    comp_sizes[lbl] = comp_sizes.get(lbl, 0) + 1
        main_comp = max(comp_sizes, key=lambda k: comp_sizes[k])

        # Multi-source Dijkstra: 본체 모든 픽셀을 source(cost=0)로 시작
        cost_grid = np.full((ny, nx), np.inf)
        prev_grid = {}
        heap = []

        for r in range(ny):
            for c in range(nx):
                if labeled[r, c] == main_comp:
                    cost_grid[r, c] = 0.0
                    heapq.heappush(heap, (0.0, r, c))

        while heap:
            d, r, c = heapq.heappop(heap)
            if d > cost_grid[r, c]:
                continue
            for dr, dc in self._NEIGHBORS_8:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < ny and 0 <= nc < nx):
                    continue
                step = 0.0 if grid[nr, nc] else 1.0
                nd = d + step
                if nd < cost_grid[nr, nc]:
                    cost_grid[nr, nc] = nd
                    prev_grid[(nr, nc)] = (r, c)
                    heapq.heappush(heap, (nd, nr, nc))

        # 컴포넌트별 최솟값 픽셀 탐색
        comp_min = {}  # comp_id -> (cost, r, c)
        for r in range(ny):
            for c in range(nx):
                lbl = labeled[r, c]
                if lbl == 0 or lbl == main_comp:
                    continue
                cost = cost_grid[r, c]
                if lbl not in comp_min or cost < comp_min[lbl][0]:
                    comp_min[lbl] = (cost, r, c)

        x_new = x.copy()
        bridge_idx = []
        for lbl, (_, r_start, c_start) in comp_min.items():
            cur = (r_start, c_start)
            while cur in prev_grid:
                r, c = cur
                if not grid[r, c]:
                    k = g2n.get((r, c))
                    if k is not None and x[k] <= threshold:
                        bridge_idx.append(k)
                        nbrs = [
                            x[g2n[(r+dr, c+dc)]]
                            for dr, dc in self._NEIGHBORS_8
                            if g2n.get((r+dr, c+dc)) is not None
                            and x[g2n[(r+dr, c+dc)]] > threshold
                        ]
                        x_new[k] = max(x_new[k], max(nbrs) if nbrs else threshold + 0.1)
                cur = prev_grid[cur]

        return x_new, bridge_idx

    # ─────────────── Discrete Projection (Staircase) ───────────────

    def _project_x(self, x: np.ndarray, beta: float) -> np.ndarray:
        """
        설계 변수를 이산 레벨로 부드럽게 투사합니다.

        bead_steps=1 → 2레벨 {0, 1}  (이진: 비드 없음 / 최대)
        bead_steps=2 → 3레벨 {0, 0.5, 1}
        bead_steps=N → N+1 레벨
        bead_steps=0 → 연속 (투사 없음)
        """
        if self.bead_steps < 1:
            return x

        n = self.bead_steps + 1          # 단계 수 → 레벨 수
        levels = np.linspace(0, 1, n)
        thresholds = (levels[:-1] + levels[1:]) * 0.5  # 인접 레벨의 중점

        x_proj = np.full_like(x, levels[0])
        for i in range(n - 1):
            diff = levels[i+1] - levels[i]
            x_proj += diff * 0.5 * (np.tanh(beta * (x - thresholds[i])) + 1.0)

        return x_proj

    def _project_x_grad(self, x: np.ndarray, beta: float) -> np.ndarray:
        """투사 함수의 미분값 (Chain-rule 용)."""
        if self.bead_steps < 1:
            return np.ones_like(x)

        n = self.bead_steps + 1
        levels = np.linspace(0, 1, n)
        thresholds = (levels[:-1] + levels[1:]) * 0.5  # 인접 레벨의 중점

        grad = np.zeros_like(x)
        for i in range(n - 1):
            diff = levels[i+1] - levels[i]
            grad += diff * 0.5 * beta * (1.0 - np.tanh(beta * (x - thresholds[i]))**2)

        return grad

    def _build_filter(self) -> np.ndarray:
        """
        공간 필터 행렬 H 조립 (최소 비드 폭 제어).

        filter_type="linear"  : w = rmin - dist  (hat kernel)
        filter_type="gaussian": w = exp(-dist²/2σ²), σ = rmin/3
          → 경계가 부드럽고 큼직한 덩어리 형태 유도에 유리
        """
        print(f" -> [Solver] 공간 필터 행렬 ({self.rmin}mm, {self.filter_type}) 조립 중...")
        n = self._n_design
        coords = self._elem_centroids()
        W = np.zeros((n, n))
        if self.filter_type == "gaussian":
            sigma = self.rmin / 3.0
            for i in range(n):
                dists = np.linalg.norm(coords - coords[i], axis=1)
                mask  = dists < self.rmin
                w     = np.exp(-dists[mask] ** 2 / (2.0 * sigma ** 2))
                W[i, mask] = w / (np.sum(w) + 1e-10)
        else:  # linear (기본)
            for i in range(n):
                dists = np.linalg.norm(coords - coords[i], axis=1)
                mask  = dists < self.rmin
                w     = self.rmin - dists[mask]
                W[i, mask] = w / (np.sum(w) + 1e-10)
        return W

    # ─────────────── 노드 좌표 조작 ───────────────

    def _apply_heights(self, h_elem: np.ndarray):
        """
        요소별 비드 높이 h_elem을 설계 노드 좌표에 반영합니다.

        노드 높이 = 인접 설계 요소 높이의 **최댓값**:
            h_n = max_{e adj n} h_e
        → 활성 비드 요소의 모든 노드가 비드 높이 전체로 올라옴 (평탄 plateau 보장).
           인접 비활성 요소와의 경계 노드는 비드 쪽으로 끌어올려져 노드 단위 스파이크
           ("뾰족 솟음")가 발생하지 않음. 비드 경계는 한 노드만큼 외측으로 확장.

        Parameters
        ----------
        h_elem : (n_design,) 요소별 비드 높이 배열 [mm]
        """
        n_int = len(self._design_nids)
        h_node = np.zeros(n_int)
        # in-place max: 각 노드에 인접한 요소들의 h_elem 중 최댓값으로 누적
        np.maximum.at(h_node, self._aggr_src, h_elem[self._aggr_dst])

        for i, nid in enumerate(self._design_nids):
            orig_xyz = self._coords_orig[nid]
            move_vec = float(h_node[i]) * self.bead_dir
            self.model.nodes[nid].x = orig_xyz[0] + move_vec[0]
            self.model.nodes[nid].y = orig_xyz[1] + move_vec[1]
            self.model.nodes[nid].z = orig_xyz[2] + move_vec[2]

    def _restore_heights(self):
        """모든 노드의 좌표를 원래 값으로 복원합니다."""
        for nid in self._design_nids:
            orig_xyz = self._coords_orig[nid]
            self.model.nodes[nid].x = orig_xyz[0]
            self.model.nodes[nid].y = orig_xyz[1]
            self.model.nodes[nid].z = orig_xyz[2]

    # ─────────────── 요소 강성 계산 (민감도용) ───────────────

    def _compute_element_K(self, eidx: int) -> np.ndarray:
        """
        현재 노드 좌표를 기반으로 단일 요소의 강성 행렬을 계산합니다.

        Parameters
        ----------
        eidx : int
            elem_ids 내 요소 인덱스

        Returns
        -------
        np.ndarray
            요소 강성 행렬 (18×18 TRIA3 또는 24×24 QUAD4)
        """
        from wht_solver.wht_quad4_element import _element_K_mitc4_plus
        from wht_solver.wht_tria3_element import _element_K_tria3

        eid  = self.elem_ids[eidx]
        elem = self.model.elements[eid]
        etype = elem.type.upper()

        prop = self.model.properties.get(elem.pid)
        mat  = self.model.materials.get(prop.mid) if prop else None
        if prop is None or mat is None:
            n_dof = len(elem.node_ids) * 6
            return np.zeros((n_dof, n_dof))

        crds = [
            np.array([self.model.nodes[nid].x,
                      self.model.nodes[nid].y,
                      self.model.nodes[nid].z])
            for nid in elem.node_ids
        ]
        try:
            if etype in self._QUAD_TYPES:
                return _element_K_mitc4_plus(
                    crds[0], crds[1], crds[2], crds[3], prop.t, mat.E, mat.nu
                )
            elif etype in self._TRIA_TYPES:
                return _element_K_tria3(
                    crds[0], crds[1], crds[2], prop.t, mat.E, mat.nu
                )
            else:
                n_dof = len(elem.node_ids) * 6
                return np.zeros((n_dof, n_dof))
        except Exception as exc:
            print(f"    [경고] 요소 {eid}({etype}) K 계산 실패: {exc}")
            n_dof = len(elem.node_ids) * 6
            return np.zeros((n_dof, n_dof))

    # ─────────────── 컴플라이언스 및 민감도 ───────────────

    def _compute_total_compliance(
        self, solver, iter_num: int = 0
    ) -> Tuple[float, dict, np.ndarray, Dict[str, np.ndarray]]:
        """
        현재 비드 형상에서의 총 컴플라이언스 및 설계 변수별 민감도를 계산합니다.

        병렬화 전략:
          1. K_base 1회 조립 (기하학만 의존 → 모든 하중 케이스 공유)
          2. 하중 케이스별 _augment_K + spsolve를 ThreadPoolExecutor로 병렬 실행
             (UMFPACK은 단일스레드 → 멀티코어에서 진정한 병렬화)
          3. JAX 민감도는 GIL 이슈로 순차 실행

        Parameters
        ----------
        solver : WHTSolver

        Returns
        -------
        total_C : float
            가중 합산 컴플라이언스 Σ w_i·C_i. 이터레이션 출력·수렴 판정에 사용.
        case_responses : dict
            {name: {'C': C_i, 'max_disp': ..., 'max_stress': ..., 'result': ...}}
        total_sens : (n_design,) ndarray
            가중 합산 민감도 Σ w_i·(∂C_i/∂h). obj_type='sum' 기본 경로에서 직접 사용.
        per_case_sens : dict[str, ndarray]
            케이스별 비가중 민감도 {name: ∂C_i/∂h (n_design,)}.
            정규화·softmax·패널티 목적함수 계산 시 solve() 에서 가중치를 별도 적용.

        Notes
        -----
        - 노드 좌표 배열 self._last_quad_c_jnp / self._last_tria_c_jnp 에 저장됨.
          고유진동수 패널티의 모드 형상 민감도(solve() 내부)가 이를 재사용.
        """
        from scipy.sparse.linalg import spsolve
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from wht_solver.wht_stress_recovery import ElementStressRecovery
        from wht_solver.wht_result import WHTSolverResult

        ndof           = len(self.sorted_nids) * 6
        total_C        = 0.0
        case_responses = {}
        total_sens     = np.zeros(self._n_design)
        per_case_sens: Dict[str, np.ndarray] = {}

        # ── 1. K_base 1회 조립 (BC 없는 기하학 전용) ─────────────────────────
        jm_base, _snids, _nidx = solver._build_jaxsso_model(load_case=None)
        K_base = solver._assemble_K_scipy(jm_base, _snids, _nidx, stabilize=True)
        print(f" -> [Solver] K_base 조립 완료 ({K_base.nnz} nnz) — {len(self._load_cases)}개 케이스 공유")

        # ── 2. 노드 좌표 배열 사전 추출 (JAX 민감도 공유, 이터레이션 내 고정) ──
        nodes = self.model.nodes
        if self._n_quad > 0:
            quad_c_jnp = jnp.array([
                [[nodes[nid].x, nodes[nid].y, nodes[nid].z] for nid in row]
                for row in self._quad_nids
            ], dtype=jnp.float64)  # (n_quad, 4, 3)
        if self._n_tria > 0:
            tria_c_jnp = jnp.array([
                [[nodes[nid].x, nodes[nid].y, nodes[nid].z] for nid in row]
                for row in self._tria_nids
            ], dtype=jnp.float64)  # (n_tria, 3, 3)

        # ── 3. BC 패턴별 그룹화 + 다중 RHS splu 공유 ────────────────────────
        # BC 패턴이 같은 케이스끼리 K_aug를 공유하고 LU 분해를 1회만 수행한다.
        # f_aug를 열 행렬로 묶어 lu.solve(F) 로 한 번에 풀면 LU 분해 횟수가
        # 케이스 수 → BC 패턴 수로 줄어든다.
        from scipy.sparse.linalg import splu

        # (jm_lc, K_aug, f_aug) 사전 준비
        lc_data = []   # [(orig_idx, weight, lc, jm_lc, K_aug_csc, f_aug), ...]
        for i, (lc, w) in enumerate(self._load_cases):
            jm_lc, _, _ = solver._build_jaxsso_model(load_case=lc)
            K_aug, f_aug = solver._augment_K_scipy(K_base, jm_lc)
            lc_data.append((i, w, lc, jm_lc, K_aug.tocsc(), f_aug))

        # BC 패턴 키: known_id 튜플로 그룹화
        from collections import defaultdict
        bc_groups: dict = defaultdict(list)
        for entry in lc_data:
            i, w, lc, jm_lc, K_aug_csc, f_aug = entry
            bc_key = tuple(sorted(jm_lc.known_id))
            bc_groups[bc_key].append(entry)

        n_groups = len(bc_groups)
        n_cases  = len(lc_data)
        print(f" -> [Solver] {n_cases}개 케이스 → {n_groups}개 BC 그룹으로 LU 공유 분해")

        def _extract_result(u_aug_np, lc, jm_lc):
            """u_aug 벡터 → displacement, cell_data, u_full, f_full, C_i."""
            n_nodes = len(self.sorted_nids)
            displacement = np.zeros((n_nodes, 6))
            for ii, nid in enumerate(self.sorted_nids):
                displacement[ii, :] = u_aug_np[self.nid_to_idx[nid] * 6:
                                                self.nid_to_idx[nid] * 6 + 6]

            # Lagrange multiplier 방식에서 prescribed DOF는 u_aug에 반영되지 않을 수 있음.
            # known_val(강제변위)이 있으면 해당 DOF의 변위를 명시적으로 복원한다.
            if jm_lc.known_val is not None:
                known_ids  = np.array(jm_lc.known_id,  dtype=np.int64)
                known_vals = np.array(jm_lc.known_val, dtype=np.float64)
                n_copy = min(len(known_ids), len(known_vals))
                for k in range(n_copy):
                    gid = known_ids[k]
                    nidx_local = gid // 6
                    dof_local  = gid %  6
                    # nidx_local은 sorted_nids 기준 0-based 인덱스
                    if 0 <= nidx_local < n_nodes:
                        displacement[nidx_local, dof_local] = known_vals[k]

            rd_q = ElementStressRecovery.recover_quad4(solver.model, displacement, self.sorted_nids)
            rd_t = ElementStressRecovery.recover_tria3(solver.model, displacement, self.sorted_nids)
            cell_data = {k: (rd_q[k] + rd_t[k])[np.newaxis, :, :] for k in rd_q}

            u_full = np.zeros(ndof)
            for ii, nid in enumerate(self.sorted_nids):
                u_full[self.nid_to_idx[nid] * 6:
                       self.nid_to_idx[nid] * 6 + 6] = displacement[ii, :]

            f_full = np.zeros(ndof)
            for force in lc.forces:
                idx = self.nid_to_idx.get(force.node_id)
                if idx is None: continue
                for d, fval in enumerate(force.load_vector):
                    if abs(fval) > 1e-12:
                        f_full[idx * 6 + d] += fval

            # C_i 계산: 순수 외력 케이스(BC 없음)는 f·u, 그 외(BC 포함/SPCD)는 u^T K u.
            # prescribed DOF 변위를 복원한 u_full을 사용하므로 SPCD 케이스도 올바른 값이 나온다.
            if lc.forces and not lc.bcs:
                C_i = float(np.dot(f_full, u_full))
            else:
                C_i = float(u_full @ (K_base @ u_full))

            result = WHTSolverResult("static", self.sorted_nids)
            result.displacement = displacement
            result.cell_data    = cell_data
            result._u_aug       = u_aug_np
            result._ndof        = ndof
            return lc.name, C_i, u_full, displacement, cell_data, result

        def _solve_group(entries):
            """BC 패턴이 같은 케이스 묶음을 다중 RHS로 한 번에 풀기."""
            _, _, _, _, K_aug_csc, _ = entries[0]
            lu = splu(K_aug_csc)
            # f_aug 열 행렬 조립 (augmented 시스템 크기 × 케이스 수)
            F = np.column_stack([e[5] for e in entries])
            U = lu.solve(F)   # LU 분해 1회, forward/back substitution N회
            results = []
            for col_idx, entry in enumerate(entries):
                i, w, lc, jm_lc, _, _ = entry
                u_aug_np = U[:, col_idx]
                results.append((i, w, _extract_result(u_aug_np, lc, jm_lc)))
            return results

        # ── 4. BC 그룹별 병렬 실행 ─────────────────────────────────────────────
        group_list = list(bc_groups.values())
        n_workers  = min(len(group_list), self.n_workers)
        solve_results = [None] * n_cases

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_solve_group, grp): grp for grp in group_list}
            with tqdm(total=n_cases, desc=f"  Iter {iter_num:>3d} 하중 케이스 해석",
                      unit="case", ncols=100) as pbar:
                for future in as_completed(futures):
                    for i, w, res in future.result():
                        solve_results[i] = (w, res)
                        pbar.update(1)

        # ── 5. 결과 수집 + JAX 민감도 (순차 — JAX GIL 이슈) ──────────────────
        # 좌표 배열을 인스턴스에 저장 → solve()의 주파수 민감도 계산에서 재사용
        self._last_quad_c_jnp = quad_c_jnp if self._n_quad > 0 else None
        self._last_tria_c_jnp = tria_c_jnp if self._n_tria > 0 else None

        for weight, (name, C_i, u_full, displacement, cell_data_r, result) in solve_results:
            total_C += weight * C_i
            stress_vals = cell_data_r["Stress"][0]
            case_responses[name] = {
                "C":          C_i,
                "max_disp":   float(np.max(np.linalg.norm(displacement[:, :3], axis=1))),
                "max_stress": _von_mises_max(stress_vals),
                "result":     result,
            }

            # 케이스별 비가중 민감도 — 노드 공간에서 계산 후 요소 공간으로 집계
            # ∂C_i/∂h_n : 내부 노드별 민감도 (n_int_nodes,)
            n_int = len(self._design_nids)
            case_sens_node = np.zeros(n_int)

            if self._n_quad > 0:
                quad_ue = u_full[self._quad_dof_idx]
                grads = vmap_element_grad_jax(
                    quad_c_jnp[:, 0], quad_c_jnp[:, 1],
                    quad_c_jnp[:, 2], quad_c_jnp[:, 3],
                    jnp.array(quad_ue),
                    jnp.array(self._quad_t), jnp.array(self._quad_E), jnp.array(self._quad_nu),
                )
                grad_stack = np.stack([np.array(g) for g in grads], axis=1)  # (n_quad, 4, 3)
                proj = np.einsum('qnd,d->qn', grad_stack, self.bead_dir)
                mask = self._quad_node_design_idx >= 0
                valid_nidx = np.where(mask, self._quad_node_design_idx, 0)
                np.add.at(case_sens_node, valid_nidx, np.where(mask, -proj, 0.0))

            if self._n_tria > 0:
                tria_ue = u_full[self._tria_dof_idx]
                grads_t = vmap_element_grad_tria3_jax(
                    tria_c_jnp[:, 0], tria_c_jnp[:, 1], tria_c_jnp[:, 2],
                    jnp.array(tria_ue),
                    jnp.array(self._tria_t), jnp.array(self._tria_E), jnp.array(self._tria_nu),
                )
                grad_stack_t = np.stack([np.array(g) for g in grads_t], axis=1)  # (n_tria, 3, 3)
                proj_t = np.einsum('qnd,d->qn', grad_stack_t, self.bead_dir)
                mask_t = self._tria_node_design_idx >= 0
                valid_nidx_t = np.where(mask_t, self._tria_node_design_idx, 0)
                np.add.at(case_sens_node, valid_nidx_t, np.where(mask_t, -proj_t, 0.0))

            # ∂C_i/∂h_e = Σ_{n∈e} (∂C_i/∂h_n) / n_adj(n)  (Chain-rule: h_n = mean h_e)
            case_sens = np.zeros(self._n_design)
            np.add.at(case_sens, self._aggr_dst,
                      case_sens_node[self._aggr_src] * self._aggr_w)

            per_case_sens[name] = case_sens
            total_sens += weight * case_sens

        return total_C, case_responses, total_sens, per_case_sens

    # ─────────────── 메인 최적화 루프 ───────────────

    def solve(self, max_iter: int = 30, tol: float = 1e-4, callback=None, stop_event=None) -> np.ndarray:
        """
        MMA 기반 Topography Optimization을 실행합니다.

        Parameters
        ----------
        max_iter : int
            최대 반복 수
        tol : float
            수렴 판정 기준 (비드 높이 변화 최대값)
        callback : Callable[[dict], None], optional
            각 반복마다 호출될 콜백 함수. 실시간 모니터링용.

        Returns
        -------
        heights : (n_design,) 최종 비드 높이 배열 [mm]
        """
        print(f"\n[wht_topo] Topography Optimization 시작 (max_iter={max_iter})")
        print(f"   - 설계 노드 수  : {self._n_design}개")
        print(f"   - 최대 비드 높이: {self.h_max}mm")
        print(f"   - 하중 케이스 수: {len(self._load_cases)}개")
        for lc, w in self._load_cases:
            print(f"     * {lc.name}: weight={w:.2f}, BC={len(lc.bcs)}개, F={len(lc.forces)}개")

        # MMA용 정규화 변수 x [0, 1]  (초기값: 최대 높이 = 1.0)
        x = self.heights.copy() / (self.h_max + 1e-12)

        # FEA 솔버 인스턴스 생성
        from wht_solver.wht_solver import WHTSolver
        fea_solver = WHTSolver(self.model)
        
        # ── 결과 저장용 디렉토리 및 Exporter 초기화 ──
        from datetime import datetime
        import os
        from wht_converter.wht_models import WHTMetadata
        from wht_converter.wht_exporters import VTKHDFExporter

        if self._out_dir is not None:
            export_dir = str(self._out_dir / "paraview")
        else:
            stamp = datetime.now().strftime("D%Y%m%d_%H%M%S")
            export_dir = os.path.join("results", stamp)
        os.makedirs(export_dir, exist_ok=True)
        exporter = VTKHDFExporter()
        meta = WHTMetadata(
            solver_name="JaxTopoSolver", solver_version="2.0",
            analysis_type="static", coordinate_system="cartesian",
            unit_length="mm", unit_force="N"
        )
        print(f" -> [Solver] 이터레이션별 통합 해석 결과 저장 디렉토리: {export_dir}")

        # ── 스냅샷 디렉토리 ──────────────────────────────────────────────────
        _snap_dir = Path(export_dir).parent / "snapshots"
        _snap_dir.mkdir(parents=True, exist_ok=True)

        # 설계 요소 메시 엣지 (Mesh View용 LineSegments: (E, 2, 2))
        _edge_set: set = set()
        for _eid in self._design_elems:
            _enids = list(self.model.elements[_eid].node_ids)
            for _k in range(len(_enids)):
                _edge_set.add(tuple(sorted([_enids[_k], _enids[(_k+1) % len(_enids)]])))
        mesh_edge_segs = np.array([
            [[self.model.nodes[_n1].x, self.model.nodes[_n1].y],
             [self.model.nodes[_n2].x, self.model.nodes[_n2].y]]
            for _n1, _n2 in _edge_set
        ], dtype=np.float32)

        # init.pkl — 모델 + 집합 배열 + 하중 케이스 (모니터 re-solve 용)
        _init_snap = {
            "model":              self.model,
            "static_load_cases":  list(self._static_load_cases),
            "design_elems":       self._design_elems,
            "design_nids":        self._design_nids,
            "aggr_src":           self._aggr_src,
            "aggr_dst":           self._aggr_dst,
            "aggr_w":             self._aggr_w,
            "sorted_nids":        self.sorted_nids,
            "bead_dir":           self.bead_dir,
            "h_max":              self.h_max,
            "orig_coords":        {nid: (self.model.nodes[nid].x,
                                         self.model.nodes[nid].y,
                                         self.model.nodes[nid].z)
                                   for nid in self._design_nids},
            "mesh_edge_segs":     mesh_edge_segs,
            "dynamic_scenarios":  list(self.dynamic_scenarios),
        }
        try:
            with open(_snap_dir / "init.pkl", "wb") as _f:
                pickle.dump(_init_snap, _f)
            print(f" -> [Snapshot] init.pkl 저장 완료: {_snap_dir}")
        except Exception as _e:
            print(f" -> [Snapshot] init.pkl 저장 실패: {_e}")

        C_0 = None
        f0_init = None   # Iter 0 목적함수값 (개선율 계산 기준)
        change_history: List[float] = []   # 정체(stagnation) 기반 수렴 감지용
        _mesh_edge_segs_sent = False

        # ── [Iter 0] 완전 평탄 초기 상태 평가 (UI/Monitor 요구사항) ─────────────
        h_phys_flat = np.zeros(self._n_design, dtype=np.float32)
        self._apply_heights(h_phys_flat)

        # ESL provider가 등록된 경우 평탄 초기 상태로 케이스 선추출 (compliance 기준값 확보)
        _dyn_cases_iter0: list = []
        if self._load_case_provider is not None:
            print("   [Iter 0] ESL 초기 케이스 추출 중 (평탄 상태 기준)...", flush=True)
            try:
                _dyn_cases_iter0 = self._load_case_provider(0, h_elem=h_phys_flat)
                self._load_cases = list(self._static_load_cases) + list(_dyn_cases_iter0)
                print(f"   [Iter 0] 하중 케이스: 정적 {len(self._static_load_cases)}개 "
                      f"+ ESL {len(_dyn_cases_iter0)}개 = 총 {len(self._load_cases)}개", flush=True)
            except Exception as _esl_err:
                print(f"   [경고] Iter 0 ESL 추출 실패: {_esl_err}", flush=True)

        C_total_0, C_responses_0, _, _ = self._compute_total_compliance(fea_solver, iter_num=0)
        
        if callback:
            coords = np.array([
                np.mean([[self.model.nodes[nid].x,
                          self.model.nodes[nid].y,
                          self.model.nodes[nid].z]
                         for nid in self.model.elements[eid].node_ids], axis=0)
                for eid in self._design_elems
            ])
            case_data = {}
            for name, res in C_responses_0.items():
                case_data[name] = {
                    "U": 0.5 * res["C"],
                    "max_disp": res["max_disp"],
                    "max_stress": res["max_stress"]
                }
            
            # ── Iter 0 모달 해석 (Ref. 주파수) ──────────────────────────────
            _ref_freqs = []
            try:
                _orig_spcs = list(fea_solver.model.spc_conditions)
                _spcd_nids = [nid for nid in fea_solver.model.nodes if nid >= 900000]
                _temp_spcs = [WHTSPCEntry(nid, (0,1,2,3,4,5), 0.0) for nid in _spcd_nids]
                # SPCD 노드가 있으면 해당 노드만 고정(free-free + SPCD 고정),
                # SPCD 노드가 없으면 원래 SPC 유지(강체 모드 방지)
                fea_solver.model.spc_conditions = _temp_spcs if _spcd_nids else _orig_spcs
                _ref_modal = fea_solver.solve_modal(
                    num_modes=self.modal_modes, exclude_rigid_body=False
                )
                fea_solver.model.spc_conditions = _orig_spcs
                _ref_freqs = _ref_modal.frequencies.tolist()
                print(f"   [Iter 0] 초기 고유진동수: "
                      f"{', '.join(f'{f:.2f}Hz' for f in _ref_freqs[:6])}"
                      f"{'...' if len(_ref_freqs) > 6 else ''}")
            except Exception as _e:
                print(f"   [경고] Iter 0 모달 해석 실패: {_e}")

            data_0 = {
                "iter": 0,
                "compliance": float(C_total_0),
                "area_ratio": 0.0,
                "cases": case_data,
                "frequencies": _ref_freqs,
                "avg_h": 0.0,
                "max_h": 0.0,
                "dx": 0.0,
                "coords": coords,
                "heights": h_phys_flat,
                "h_max": float(self.h_max),
                "bead_steps": int(self.bead_steps),
                "bead_dir_sign": 0 if self.bidirectional else int(np.sign(self.bead_dir[int(np.argmax(np.abs(self.bead_dir)))])),
                "snap_dir": str(_snap_dir),
                "min_width": float(self.rmin),
                "mesh_edge_segs": mesh_edge_segs,
            }
            callback(data_0)
            _mesh_edge_segs_sent = True

            try:
                with open(_snap_dir / "iter_000.pkl", "wb") as _f:
                    pickle.dump({
                        "iter": 0,
                        "h_elem": h_phys_flat.copy(),
                        "load_cases": [(lc.name, w, lc) for lc, w in self._load_cases],
                    }, _f)
            except Exception:
                pass
                
        self._restore_heights()

        for i in range(max_iter):
            # min-width 필터 반경 연속화 (filter continuation)
            # rmin을 _rmin_start(초기 큰 값)에서 _rmin_final(목표)까지 선형 감소
            if self._rmin_final < self._rmin_start and self._rmin_ramp_n > 0:
                _t_rmin = min(1.0, i / self._rmin_ramp_n)
                _rmin_i = self._rmin_start - _t_rmin * (self._rmin_start - self._rmin_final)
                if abs(_rmin_i - self._rmin_current) > 0.5:  # 0.5mm 이상 변화 시에만 재조립
                    _prev = self._rmin_current
                    self._rmin_current = _rmin_i
                    self.rmin = _rmin_i
                    self._H = self._build_filter()
                    print(f"   [filter-cont] rmin {_prev:.1f} → {_rmin_i:.1f}mm 필터 재조립")

            # 좌우 대칭 강제 (변수 동기화)
            if self.sym_x:
                x = 0.5 * (x + x[self._sym_map])

            x_old = x.copy()

            # ── 1. Filter → Discrete Projection → Heaviside (표준 SIMP) ──
            # 흐름: x → H@x → x_filt → project({0,0.5,1}) → heaviside → x_proj
            #     → *h_max → h_phys → FEA (필터가 x에 적용되어 h_phys가 곧 물리값)
            # min-width(필터 반경)가 직접 비드 폭을 제어. x_proj가 area_ratio 측정 대상.

            # (a) 공간 필터: 설계변수 평활화 → 최소 비드 폭 강제
            x_filt = self._H @ x

            # (b) 이산 투사: bead_steps 단계로 양자화
            # β-continuation: 초기 β=4로 시작 → max 50 도달
            if self.bead_steps >= 1:
                self._beta = min(50.0, 4.0 + (i / max_iter) * 80.0)
                x_proj_in = self._project_x(x_filt, self._beta)
            else:
                x_proj_in = x_filt

            # (c) Heaviside projection (use_projection=True 시)
            if self.use_projection:
                _beta_h = min(self.proj_beta_max, 1.0 + (i / max_iter) * (self.proj_beta_max - 1.0))
                _eta    = self.proj_eta
                _t1 = np.tanh(_beta_h * _eta)
                _t2 = np.tanh(_beta_h * (1.0 - _eta))
                x_proj = (_t1 + np.tanh(_beta_h * (x_proj_in - _eta))) / (_t1 + _t2 + 1e-12)
                x_proj = np.clip(x_proj, 0.0, 1.0)
            else:
                x_proj = x_proj_in

            # (d) 물리 높이 계산
            # 단방향: h = x_proj × h_max            (0 → h_max)
            # 양방향: h = (x_proj - 0.5) × 2 × h_max  (-h_max → +h_max)
            if self.bidirectional:
                h_phys = (x_proj - 0.5) * 2.0 * self.h_max
            else:
                h_phys = x_proj * self.h_max
            h_filtered = h_phys
            self._apply_heights(h_filtered)

            # 동적 ESL 재추출 (provider 등록 시 매 이터레이션 실행)
            if self._load_case_provider is not None:
                dyn_cases = self._load_case_provider(i, h_elem=h_filtered)
                self._load_cases = list(self._static_load_cases) + list(dyn_cases)
                print(f"     하중 케이스: 정적 {len(self._static_load_cases)}개 "
                      f"+ 동적 ESL {len(dyn_cases)}개 = 총 {len(self._load_cases)}개")

            # 컴플라이언스 및 민감도 계산 (솔버 인스턴스 공유)
            C_total, C_responses, dC_dh_base, per_case_sens = self._compute_total_compliance(fea_solver, iter_num=i)
            ndof = len(self.sorted_nids) * 6

            # 초기값 저장 및 정규화 (수렴 안정성 강화)
            if C_0 is None:
                C_0 = max(abs(float(C_total)), 1e-6)
                self._C_0_cases = {
                    name: max(abs(C_responses[name]['C']), 1e-10)
                    for name in C_responses
                }

            # ── 고유진동수 해석 (--modal-modes로 모드 수 제어) ──
            try:
                # [WHT] SPC 없이 완전 Free-Free 상태의 주파수를 구하기 위해 spc_conditions를 일시적으로 격리
                orig_spcs = list(fea_solver.model.spc_conditions)
                _spcd_nids = [nid for nid in fea_solver.model.nodes if nid >= 900000]
                _temp_spcs = [WHTSPCEntry(nid, (0,1,2,3,4,5), 0.0) for nid in _spcd_nids]
                fea_solver.model.spc_conditions = _temp_spcs if _spcd_nids else orig_spcs

                modal_results = fea_solver.solve_modal(num_modes=self.modal_modes, exclude_rigid_body=False)

                # [WHT] 모달 해석 이후 SPC 원상 복구
                fea_solver.model.spc_conditions = orig_spcs
                freqs = modal_results.frequencies  # Hz
                
                # 모드 해석 결과 터미널 출력 (주파수 및 질량 기여도)
                eff_mass, total_mass_cg = modal_results.calculate_effective_mass()
                if eff_mass is not None and total_mass_cg is not None:
                    print(f"\n{'='*60}")
                    print(f" [Internal Solver Modal Analysis Summary] : Iter {i}")
                    print(f"{'='*60}")
                    print(" MODE NO.   FREQUENCY(Hz)   X-MASS(%)   Y-MASS(%)   Z-MASS(%)")
                    total_mass_sum = total_mass_cg[:3].copy()
                    total_mass_sum[total_mass_sum < 1e-12] = 1.0 # prevent div by zero
                    for m_idx, f in enumerate(freqs):
                        x_pct = (eff_mass[m_idx, 0] / total_mass_sum[0]) * 100.0
                        y_pct = (eff_mass[m_idx, 1] / total_mass_sum[1]) * 100.0
                        z_pct = (eff_mass[m_idx, 2] / total_mass_sum[2]) * 100.0
                        print(f" {m_idx+1:7d}   {f:13.4f}   {x_pct:9.2f}   {y_pct:9.2f}   {z_pct:9.2f}")
                    print(f"{'='*60}\n")
                    
            except Exception as e:
                print(f"    [경고] 고유진동수 해석 실패 (Iter {i}): {e}")
                modal_results = None
                freqs = np.zeros(10)
            elastic_freqs = [f for f in freqs if f > 0.1]

            # ── 이터레이션 결과 저장 (ParaView 호환 VTKHDF) ──
            try:
                # 기본 WHTResultData 생성 (첫 번째 하중 케이스 기준)
                first_case_name = self._load_cases[0][0].name
                base_res = C_responses[first_case_name]["result"]
                wht_data = base_res.to_wht_result_data(meta, self.model)
                
                # 단일 Displacement/Stress 키 삭제 (이름 충돌 방지)
                wht_data.point_data.pop("Displacement", None)
                wht_data.cell_data.pop("Stress", None)
                
                # 1. 비드 높이 (Bead Height) 추가 — 요소 높이를 노드로 집계
                h_node_sum_e = np.zeros(len(self._design_nids))
                np.add.at(h_node_sum_e, self._aggr_src, h_filtered[self._aggr_dst])
                h_node_arr = h_node_sum_e / (self._node_adj_count_arr + 1e-12)
                h_current_full = np.zeros(len(self.sorted_nids))
                for i_dn, nid in enumerate(self._design_nids):
                    h_current_full[self.nid_to_idx[nid]] = h_node_arr[i_dn]
                wht_data.point_data["Bead_Height"] = h_current_full.reshape(1, -1, 1).astype(np.float32)

                # 2. 하중 케이스별 결과 병합
                for name, res_dict in C_responses.items():
                    res_data = res_dict["result"].to_wht_result_data(meta, self.model)
                    if "Displacement" in res_data.point_data:
                        wht_data.point_data[f"Disp_{name}"] = res_data.point_data["Displacement"]
                    if "Stress" in res_data.cell_data:
                        wht_data.cell_data[f"Stress_{name}"] = res_data.cell_data["Stress"]

                # 3. 고유 모드 형상 추가 및 주파수 정보 명시
                if modal_results is not None:
                    modal_data = modal_results.to_wht_result_data(meta, self.model)
                    if "Displacement" in modal_data.point_data:
                        mode_disp = modal_data.point_data["Displacement"] # (num_modes, N, 3)
                        for m_idx in range(mode_disp.shape[0]):
                            freq_hz = freqs[m_idx]
                            field_name = f"Mode_{m_idx+1}_{freq_hz:.2f}Hz"
                            wht_data.point_data[field_name] = mode_disp[m_idx:m_idx+1]
                
                if not hasattr(wht_data, "field_data") or wht_data.field_data is None:
                    wht_data.field_data = {}
                wht_data.field_data["Frequencies_Hz"] = np.array([freqs], dtype=np.float32)

                # 파일 출력 (모든 필드를 1개 파일에 병합)
                out_path = os.path.join(export_dir, f"iter_{i:03d}.hdf")
                exporter.export(wht_data, out_path)

                # LS-DYNA .k 저장 (이터레이션별 비드 패턴 기록)
                lsdyna_path = os.path.join(export_dir, f"iter_{i:03d}.k")
                try:
                    self.model.export_to_solver('lsdyna', lsdyna_path, reorder=True)
                except Exception as _ke:
                    print(f"    [경고] LS-DYNA .k 저장 실패 (Iter {i}): {_ke}")
            except Exception as e:
                print(f"    [경고] 해석 결과 저장 실패: {e}")

            # ── 목적함수 값 및 민감도 계산 (obj_type / normalize_obj 분기) ──────
            # weights_map: {케이스명 → w_i}  (self._load_cases에서 추출)
            # C0_cases   : {케이스명 → C_i0} (첫 이터레이션에서 고정)
            weights_map = {lc.name: w for lc, w in self._load_cases}
            if self.weight_variation > 0:
                # 매 이터마다 각 케이스 가중치에 ±variation 랜덤 변동 적용
                # uniform(-v, +v) → 케이스별 [w*(1-v), w*(1+v)] 범위
                for _n in list(weights_map.keys()):
                    _delta = self.weight_variation * (2.0 * np.random.rand() - 1.0)
                    weights_map[_n] = max(0.01, weights_map[_n] * (1.0 + _delta))
            C0_cases    = self._C_0_cases

            if self.obj_type == 'sum' and not self.normalize_obj:
                # ① 기본 가중 합산: f = (Σ w_i·C_i) / C_0
                #    C_0 = Iter 0의 총 컴플라이언스 → 이터레이션 간 스케일 안정화
                f0val = float(C_total) / C_0
                dC_dh = dC_dh_base / C_0

            elif self.obj_type == 'sum' and self.normalize_obj:
                # ② 케이스별 정규화 가중 합산: f = Σ w_i·(C_i/C_i0)
                #    C_i0 = 케이스 i의 Iter 0 컴플라이언스 → 케이스 간 크기 차이 보정
                #    하중이 서로 다른 정적·동적 ESL 케이스를 동등하게 최적화할 때 사용
                f0val = sum(
                    weights_map.get(n, 1.0) * (C_responses[n]['C'] / C0_cases.get(n, 1.0))
                    for n in C_responses
                )
                dC_dh = np.zeros(self._n_design)
                for n, cs in per_case_sens.items():
                    # df/dh = Σ (w_i/C_i0) · ∂C_i/∂h
                    dC_dh += weights_map.get(n, 1.0) / C0_cases.get(n, 1.0) * cs

            else:  # 'max' or 'sum+max' — softmax approximation
                # ③ Softmax 최악 케이스: f = (1/α)·log(Σ exp(α·w_i·C_i/C_i0))
                #    α→∞ : hard-max (최악 케이스만 반영)
                #    α→0 : 단순 합산에 수렴
                #    log-sum-exp max-trick으로 exp 오버플로 방지
                alpha     = self.obj_alpha
                names_ord = list(C_responses.keys())
                vals      = np.array([
                    weights_map.get(n, 1.0) * C_responses[n]['C'] / C0_cases.get(n, 1.0)
                    for n in names_ord
                ])
                v_shift = np.max(vals)                           # 수치 안정용 shift
                exps    = np.exp(alpha * (vals - v_shift))
                sum_exp = np.sum(exps)
                f0_max  = float(v_shift + (1.0 / alpha) * np.log(sum_exp + 1e-300))

                # 민감도: df/dh = Σ softmax_w_i · (w_i/C_i0) · ∂C_i/∂h
                #   softmax_w_i = exp(α·val_i) / Σ exp(α·val_j)
                sens_max = np.zeros(self._n_design)
                for k, n in enumerate(names_ord):
                    sw = exps[k] / (sum_exp + 1e-300)   # softmax weight for case i
                    sens_max += (weights_map.get(n, 1.0) / C0_cases.get(n, 1.0)) * sw * per_case_sens[n]

                if self.obj_type == 'max':
                    f0val = f0_max
                    dC_dh = sens_max
                else:  # ④ 'sum+max': f = 0.5·f_sum + 0.5·f_max
                    #    평균 성능(sum)과 최악 케이스 방어(max) 동시 고려
                    f0_sum = float(sum(
                        weights_map.get(n, 1.0) * (C_responses[n]['C'] / C0_cases.get(n, 1.0))
                        for n in C_responses
                    ))
                    sens_sum = np.zeros(self._n_design)
                    for n, cs in per_case_sens.items():
                        sens_sum += weights_map.get(n, 1.0) / C0_cases.get(n, 1.0) * cs
                    f0val = 0.5 * f0_sum + 0.5 * f0_max
                    dC_dh = 0.5 * sens_sum + 0.5 * sens_max

            # ── ⑤ 고유진동수 패널티 (∂K/∂h 기반 모드 형상 민감도) ──────────────
            # P = freq_weight · max(0, freq_target - f₁)² / freq_target²
            # 위 ①~④ 중 어떤 목적함수와도 병합: f_total = f_obj + P
            # 민감도: dP/dh = -2·λ·deficit/T² · df₁/dh
            #   df₁/dh ≈ φ₁ᵀ(∂K/∂h)φ₁ / (4π²f₁)   [단위 모달 질량 가정]
            #   φ₁(∂K/∂h)φ₁ 계산: compliance 민감도와 동일한 JAX vmap 재사용
            #                      (u_e → phi_e 대입만 변경)
            if (self.freq_weight > 0.0 and self.freq_target > 0.0
                    and modal_results is not None and len(elastic_freqs) > 0):
                f1      = elastic_freqs[0]
                deficit = max(0.0, self.freq_target - f1)
                if deficit > 0.0:
                    try:
                        # 첫 번째 탄성 모드 형상 → 전체 DOF 벡터 φ (ndof,)
                        # modal_results.mode_shapes: (n_modes, N, 6), sorted_nids 순
                        # elastic_freqs[0] 에 대응하는 실제 모드 인덱스를 찾아야 함
                        # (rigid body 모드가 앞에 있을 수 있으므로 mode_shapes[0] 은 오류)
                        _first_elastic_idx = next(
                            (k for k, f in enumerate(freqs) if f > 0.1), 0
                        )
                        phi_node = modal_results.mode_shapes[_first_elastic_idx]
                        phi = np.zeros(ndof)
                        for ii, nid in enumerate(self.sorted_nids):
                            phi[self.nid_to_idx[nid] * 6:self.nid_to_idx[nid] * 6 + 6] = phi_node[ii]
                        phi /= (np.linalg.norm(phi) + 1e-12)  # 단위 정규화

                        # φ^T (∂K_e/∂z_n) φ 계산 — vmap_element_grad_jax 재사용
                        # (compliance 민감도와 동일 구조, u_e 자리에 phi_e 대입)
                        # 결과는 노드 공간에서 집계 후 요소 공간으로 변환
                        n_int_phi = len(self._design_nids)
                        phi_sens_node = np.zeros(n_int_phi)
                        if self._n_quad > 0 and self._last_quad_c_jnp is not None:
                            quad_phie = phi[self._quad_dof_idx]
                            gp = vmap_element_grad_jax(
                                self._last_quad_c_jnp[:, 0], self._last_quad_c_jnp[:, 1],
                                self._last_quad_c_jnp[:, 2], self._last_quad_c_jnp[:, 3],
                                jnp.array(quad_phie),
                                jnp.array(self._quad_t), jnp.array(self._quad_E),
                                jnp.array(self._quad_nu),
                            )
                            gps = np.stack([np.array(g) for g in gp], axis=1)
                            prj = np.einsum('qnd,d->qn', gps, self.bead_dir)
                            mq  = self._quad_node_design_idx >= 0
                            vq  = np.where(mq, self._quad_node_design_idx, 0)
                            np.add.at(phi_sens_node, vq, np.where(mq, prj, 0.0))
                        if self._n_tria > 0 and self._last_tria_c_jnp is not None:
                            tria_phie = phi[self._tria_dof_idx]
                            gpt = vmap_element_grad_tria3_jax(
                                self._last_tria_c_jnp[:, 0], self._last_tria_c_jnp[:, 1],
                                self._last_tria_c_jnp[:, 2],
                                jnp.array(tria_phie),
                                jnp.array(self._tria_t), jnp.array(self._tria_E),
                                jnp.array(self._tria_nu),
                            )
                            gpst = np.stack([np.array(g) for g in gpt], axis=1)
                            prjt = np.einsum('qnd,d->qn', gpst, self.bead_dir)
                            mt   = self._tria_node_design_idx >= 0
                            vt   = np.where(mt, self._tria_node_design_idx, 0)
                            np.add.at(phi_sens_node, vt, np.where(mt, prjt, 0.0))
                        # 노드 → 요소 공간 집계
                        phi_sens = np.zeros(self._n_design)
                        np.add.at(phi_sens, self._aggr_dst,
                                  phi_sens_node[self._aggr_src] * self._aggr_w)

                        # df1/dh ≈ phi_sens / (4π²f1)  [단위 모달 질량 가정]
                        df1_dh = phi_sens / (4.0 * np.pi**2 * f1 + 1e-12)

                        # P = freq_weight * (deficit/freq_target)²
                        # dP/dh = -2 * freq_weight * deficit / freq_target² * df1/dh
                        t2 = self.freq_target**2 + 1e-12
                        P_norm = self.freq_weight * (deficit / (self.freq_target + 1e-12))**2
                        dP_dh  = -self.freq_weight * 2.0 * deficit / t2 * df1_dh

                        f0val += float(P_norm)
                        dC_dh  = dC_dh + dP_dh
                        print(f"     freq_penalty: f1={f1:.2f}Hz  target={self.freq_target:.2f}Hz"
                              f"  deficit={deficit:.2f}Hz  P={P_norm:.4f}")
                    except Exception as _fe:
                        print(f"    [경고] 고유진동수 민감도 계산 실패: {_fe}")

            if f0_init is None:
                f0_init = f0val

            # ── 민감도 역전파 (Chain-rule) ─────────────────────────────────────
            # 순전파: x → [H@] → x_filt → [project] → x_proj_in → [heaviside]
            #         → x_proj → *h_max → h_phys → FEA
            # 역전파:
            #   dC/dh         : FEA 직접 계산 (dC_dh)
            #   dC/dx_proj    = dC/dh · h_max
            #   dC/dx_proj_in = dC/dx_proj · dH/dx_in    (heaviside)
            #   dC/dx_filt    = dC/dx_proj_in · d_proj/dx_filt  (project at x_filt)
            #   dC/dx         = H^T · dC/dx_filt        (필터 전치 — 맨 마지막)

            # 1. 높이 스케일 역전파
            # 단방향: dh/dx_proj = h_max
            # 양방향: dh/dx_proj = 2 × h_max
            _h_scale = 2.0 * self.h_max if self.bidirectional else self.h_max
            dC_dx_proj = dC_dh * _h_scale

            # 2. Heaviside 역전파
            if self.use_projection:
                _beta_h = min(self.proj_beta_max, 1.0 + (i / max_iter) * (self.proj_beta_max - 1.0))
                _eta    = self.proj_eta
                _t1 = np.tanh(_beta_h * _eta)
                _t2 = np.tanh(_beta_h * (1.0 - _eta))
                dH_dx = _beta_h * (1.0 - np.tanh(_beta_h * (x_proj_in - _eta)) ** 2) / (_t1 + _t2 + 1e-12)
                dC_dx_proj_in = dC_dx_proj * dH_dx
            else:
                dC_dx_proj_in = dC_dx_proj

            # 3. 이산 투사 역전파: d_proj/dx_filt (x_filt 기준 — 순방향과 일관)
            if self.bead_steps >= 1:
                dC_dx_filt = dC_dx_proj_in * self._project_x_grad(x_filt, self._beta)
            else:
                dC_dx_filt = dC_dx_proj_in

            # 4. 공간 필터 역전파 (맨 마지막)
            dC_dx = self._H.T @ dC_dx_filt

            df0dx = (dC_dx / C_0).flatten()

            # ── 반발 패널티 (다중 설계 탐색) ─────────────────────────────────
            # diversity_start_iter >= 0: 해당 iter에 현재 x를 기준점으로 스냅샷 등록
            if (self.diversity_weight > 0
                    and self.diversity_start_iter >= 0
                    and i == self.diversity_start_iter):
                self._reference_designs.append(x.copy())
                print(f"   [Diversity] iter {i}: 기준 설계 스냅샷 등록 "
                      f"(총 {len(self._reference_designs)}개)")

            if self.diversity_weight > 0 and self._reference_designs:
                # start_iter 방식: iter 진행에 따라 패널티 강도 ramp-in (0→λ, 10 iter)
                if self.diversity_start_iter >= 0:
                    _ramp = min(1.0, (i - self.diversity_start_iter) / 10.0)
                    _w_eff = self.diversity_weight * max(0.0, _ramp)
                else:
                    _w_eff = self.diversity_weight
                if _w_eff > 0:
                    sig2 = self.diversity_sigma ** 2
                    for x_ref in self._reference_designs:
                        diff   = x - x_ref
                        dist2  = np.dot(diff, diff)
                        w_rep  = _w_eff * np.exp(-dist2 / (2.0 * sig2))
                        f0val += float(w_rep)
                        df0dx += w_rep * (-diff / sig2)

            # 좌우 대칭 강제 (민감도 동기화)
            if self.sym_x:
                df0dx = 0.5 * (df0dx + df0dx[self._sym_map])

            # ── vol_frac ramping: 1.0 → h_ratio (처음 vol_ramp_iters 구간) ──────
            _ramp_n = max(1, self.vol_ramp_iters)
            _t = min(1.0, i / _ramp_n)          # 0.0(iter=0) → 1.0(iter=ramp_n)
            _vol_target = 1.0 - _t * (1.0 - self.h_ratio)

            # ── 비드 연결 phase 1: df0dx 승격 + vol_frac 사전 보정 ────────────────
            # _apply_bead_connect를 1회만 호출해 두 가지를 동시 처리:
            #   a) bridge 노드 df0dx를 활성 이웃 최댓값으로 승격 (MMA가 bridge 유지하도록)
            #   b) bridge 추가로 인한 체적 팽창을 사전에 vol_frac에서 차감
            #      (미보정 시: MMA가 다음 이터에서 약한 노드를 0으로 압축 → 비드 수축)
            if self.bead_connect > 0:
                x_bridged, bridge_idx = self._apply_bead_connect(x)
                if bridge_idx:
                    cg = self._connect_grid
                    for k in bridge_idx:
                        row, col = cg["gj"][k], cg["gi"][k]
                        nb_sens = [
                            df0dx[nb]
                            for dr, dc in self._NEIGHBORS_8
                            if (nb := cg["grid_to_node"].get((row+dr, col+dc))) is not None
                            and x[nb] > 0.1
                        ]
                        if nb_sens:
                            df0dx[k] = max(df0dx[k], max(nb_sens))

                    # bridge 추가 후 체적 증가분 계산 → MMA 목표에서 선제 차감
                    # 비드 면적이 목표치(0.3)에 수축하는 버그 방지를 위해 선제 차감을 비활성화하고 h_ratio로 유지합니다.
                    # if self.bead_steps >= 1:
                    #     vol_before = float(np.mean(self._project_x(x, self._beta)))
                    #     vol_after  = float(np.mean(self._project_x(x_bridged, self._beta)))
                    # else:
                    #     vol_before = float(np.mean(x))
                    #     vol_after  = float(np.mean(x_bridged))
                    # bridge_excess = max(0.0, vol_after - vol_before)
                    # self.mma.vol_frac = max(0.01, self.h_ratio - bridge_excess)
                    self.mma.vol_frac = _vol_target
                else:
                    self.mma.vol_frac = _vol_target
            else:
                self.mma.vol_frac = _vol_target

            # 제약 조건: mean(H_area) = vol_target
            # 단방향: H_area(x_proj)         → x_proj > 0.1 기준
            # 양방향: H_area(|x_proj - 0.5|) → |x_proj - 0.5| > 0.1 기준
            #         (x=0.5 근방 = 비드 없음, x=0 or x=1 = 비드 있음)
            _beta_a  = 20.0
            _eta_a   = 0.1
            _t1a     = np.tanh(_beta_a * _eta_a)
            _t2a     = np.tanh(_beta_a * (1.0 - _eta_a))
            _denom_a = _t1a + _t2a + 1e-12

            def _smooth_area(xp):
                return (_t1a + np.tanh(_beta_a * (xp - _eta_a))) / _denom_a

            def _smooth_area_grad(xp):
                return _beta_a * (1.0 - np.tanh(_beta_a * (xp - _eta_a)) ** 2) / _denom_a

            if self.bidirectional:
                _dev      = np.abs(x_proj_in - 0.5)          # [0, 0.5]
                _dev_sign = np.sign(x_proj_in - 0.5)         # ±1
                _area_density = _smooth_area(_dev)
                _ha_grad_base = _smooth_area_grad(_dev) * _dev_sign  # chain: d|·|/dx_proj
            else:
                _area_density = _smooth_area(x_proj_in)
                _ha_grad_base = _smooth_area_grad(x_proj_in)

            fval = np.array([float(np.mean(_area_density)) - _vol_target])

            if self.bead_steps >= 1:
                _pg   = self._project_x_grad(x_filt, self._beta)
                dfdx  = self._H.T @ (_ha_grad_base * _pg / self._n_design)
            else:
                dfdx  = self._H.T @ (_ha_grad_base / self._n_design)

            # MMA 이분법도 smooth area 기준으로 체적 검사
            _H_ref    = self._H
            _beta_ref = self._beta
            _bs_ref   = self.bead_steps
            _bidir_ref = self.bidirectional
            def project_fn(x_t):
                xf  = _H_ref @ x_t
                xp  = self._project_x(xf, _beta_ref) if _bs_ref >= 1 else xf
                if _bidir_ref:
                    return _smooth_area(np.abs(xp - 0.5))
                return _smooth_area(xp)

            x = self.mma.update(
                x, f0val, df0dx,
                fval,
                dfdx,
                i + 1,
                project_fn=project_fn,
            )
            x = np.clip(x, 0.0, 1.0)

            # ── 비드 연결 phase 2: 물리적 closing 적용 ───────────────────────────
            if self.bead_connect > 0:
                x, _ = self._apply_bead_connect(x)

            # 수렴 판정
            change = float(np.abs(x - x_old).max())
            
            # 모니터링 콜백 호출
            if callback:
                # 설계 요소 도심 좌표 추출 (2D 시각화용)
                coords = np.array([
                    np.mean([[self.model.nodes[nid].x,
                              self.model.nodes[nid].y,
                              self.model.nodes[nid].z]
                             for nid in self.model.elements[eid].node_ids], axis=0)
                    for eid in self._design_elems
                ])
                # 변형 에너지는 컴플라이언스의 절반 (U = 0.5 * C)
                case_data = {}
                for name, res in C_responses.items():
                    case_data[name] = {
                        "U": 0.5 * res["C"],
                        "max_disp": res["max_disp"],
                        "max_stress": res["max_stress"]
                    }
                    
                data = {
                    "iter":       i + 1,
                    "compliance": float(C_total),
                    "area_ratio": float(np.mean(_area_density)),
                    "cases":      case_data,
                    "frequencies": freqs.tolist(),
                    "avg_h":      float(np.mean(h_phys)),
                    "max_h":      float(np.max(h_phys)),
                    "dx":         change,
                    "coords":     coords,
                    "heights":    h_phys * float(np.sign(
                        self.bead_dir[int(np.argmax(np.abs(self.bead_dir)))]
                    )),
                    "h_max":      float(self.h_max),
                    "bead_steps": int(self.bead_steps),
                    "bead_dir_sign": 0 if self.bidirectional else int(np.sign(self.bead_dir[int(np.argmax(np.abs(self.bead_dir)))])),
                    "snap_dir":   str(_snap_dir),
                    "min_width":  float(self.rmin),
                }
                if not _mesh_edge_segs_sent:
                    data["mesh_edge_segs"] = mesh_edge_segs
                    _mesh_edge_segs_sent = True
                callback(data)

                # 이터레이션 스냅샷 저장
                try:
                    with open(_snap_dir / f"iter_{i+1:03d}.pkl", "wb") as _f:
                        pickle.dump({
                            "iter":       i + 1,
                            "h_elem":     h_phys.copy(),
                            "load_cases": [(lc.name, w, lc)
                                           for lc, w in self._load_cases],
                        }, _f)
                except Exception:
                    pass

            # ── 이터레이션 요약 출력 ──────────────────────────────────────────
            _SW = 100
            f1_str = f"F1={elastic_freqs[0]:.1f}Hz" if elastic_freqs else "F1=--"
            # 수렴 판정: (1) 단일 이터 엄격 기준 dx<tol, OR
            #            (2) 정체 — 연속 5회 dx<tol*5 (MMA 미세 진동으로 수렴 못함 방지)
            # Iter 0~2는 초기 균일/대칭 상태에서 dx≈0 위양성 방지 위해 체크 스킵
            change_history.append(change)
            stagnant = (
                len(change_history) >= 5
                and all(c < tol * 5.0 for c in change_history[-5:])
            )
            converged = (i >= 3) and (change < tol or stagnant)
            stop_req  = stop_event and stop_event.is_set()

            obj_tag = (f"[{self.obj_type}{'·N' if self.normalize_obj else ''}]"
                       if self.obj_type != 'sum' or self.normalize_obj else "")

            # 목적함수 개선율 (Iter 0 대비)
            f0_impv = (f0_init - f0val) / (abs(f0_init) + 1e-12) * 100.0

            # 체적 제약 상태 (projected 기준 — fval과 동일 척도)
            vol_cur = float(np.mean(_area_density))
            vol_tgt = _vol_target
            vol_vio = vol_cur - vol_tgt   # >0: 초과, <0: 미달
            vol_mark = "OK" if abs(vol_vio) < 0.01 else ("HI" if vol_vio > 0 else "LO")

            print(f"\n  ┌─ Iter {i:03d} {'─' * (_SW - 15)}┐")
            print(f"     Obj: f0={f0val:.4f}{obj_tag}  "
                  f"impr={f0_impv:+.1f}%  (init={f0_init:.4f})"
                  f"   dx={change:.4f}")
            print(f"     Vol: {vol_cur*100:.1f}% / target {vol_tgt*100:.1f}%  [{vol_mark}]"
                  f"   Avg_h={np.mean(x_proj_in*self.h_max):.2f}mm"
                  f"   {f1_str}")
            for name, res in C_responses.items():
                c_ratio = res['C'] / max(self._C_0_cases.get(name, 1e-10), 1e-10)
                print(f"     [{name:28s}]  U={0.5*res['C']:.3e}J"
                      f"  u={res['max_disp']:.1f}mm  sv={res['max_stress']:.0f}MPa"
                      f"  ({c_ratio*100:.0f}%)")
            if len(elastic_freqs) >= 1:
                freq_str = "  ".join(f"f{k+1}={elastic_freqs[k]:.1f}Hz"
                                     for k in range(min(5, len(elastic_freqs))))
                freq_line = f"     Freq: {freq_str}"
                if self.freq_target > 0.0:
                    f1_cur = elastic_freqs[0]
                    freq_ok = "OK" if f1_cur >= self.freq_target else f"need {self.freq_target:.1f}Hz"
                    freq_line += f"  [{freq_ok}]"
                print(freq_line)
            if converged:
                if stagnant and change >= tol:
                    print(f"     >> Converged (stagnation: 5 iters all dx<{tol*5:.2e})")
                else:
                    print(f"     >> Converged  (dx={change:.2e} < tol={tol:.2e})")
            elif stop_req:
                print(f"     >> Stop requested")
            print(f"  └{'─' * (_SW - 4)}┘")

            if converged or stop_req:
                break

        # 원본 좌표 복원 후 최종 형상 적용
        self._restore_heights()
        # 최종 결과: filter → hard snap (β=∞ 등가 하드 threshold)
        # 최적화 루프는 연속 근사(tanh)로 진행했지만, 최종 형상은 각 레벨의
        # 경계(threshold)를 기준으로 가장 가까운 이산 레벨로 확정 snap한다.
        x_final_filt = self._H @ x
        if self.bead_steps >= 1:
            n = self.bead_steps + 1
            levels = np.linspace(0.0, 1.0, n)
            # argmin으로 가장 가까운 레벨 선택 (hard snap)
            idx = np.argmin(
                np.abs(x_final_filt[:, None] - levels[None, :]), axis=1
            )
            x_final_proj = levels[idx]
        else:
            x_final_proj = x_final_filt

        if self.bidirectional:
            self.heights = (x_final_proj - 0.5) * 2.0 * self.h_max
        else:
            self.heights = x_final_proj * self.h_max

        # 수렴 설계를 레퍼런스로 등록 (다음 solve() 호출 시 반발 대상)
        if self.diversity_weight > 0:
            self._reference_designs.append(x.copy())
            print(f" -> [Diversity] 설계 #{len(self._reference_designs)} 등록 "
                  f"(총 레퍼런스 {len(self._reference_designs)}개)")

        return self.heights

    def reset_for_next_design(self, noise: float = 0.15):
        """
        다음 설계 탐색을 위해 설계 변수를 초기화합니다.
        이전 수렴 결과에 noise를 주입해 다른 로컬 최적해로 유도합니다.

        Parameters
        ----------
        noise : float
            랜덤 노이즈 강도 [0, 1]. 기본 0.15.
        """
        self.mma = MMAOptimizer(n_vars=self._n_design, vol_frac=self.h_ratio)
        self._beta = 1.0
        # 이전 수렴 설계 + 노이즈에서 재시작
        if self._reference_designs:
            x_last = self._reference_designs[-1].copy()
            x_new  = x_last + noise * np.random.randn(self._n_design)
            self.heights = np.clip(x_new, 0.0, 1.0) * self.h_max
        else:
            self.heights = np.ones(self._n_design) * self.h_max
        print(f" -> [Diversity] 설계 변수 초기화 완료 (noise={noise:.2f})")

    def apply_final_shape(self, skip_filter: bool = False):
        """
        최적화 결과(self.heights)를 모델 노드 좌표에 영구 적용합니다.
        solve() 완료 후 export 전에 반드시 호출하십시오.

        Parameters
        ----------
        skip_filter : bool
            (deprecated) 표준 SIMP(filter→project) 순서 적용 후 self.heights는
            이미 필터를 거친 물리값이므로 재적용 안 함. 호환성 유지용 인자.
        """
        del skip_filter
        h = self.heights
        self._apply_heights(h)
        print(f" -> [Solver] 최종 비드 형상 적용 완료 "
              f"(Max: {np.max(h):.2f}mm, "
              f"Avg: {np.mean(h):.2f}mm)")

    def get_full_heights(self, skip_filter: bool = False) -> np.ndarray:
        """
        시각화용 전체 노드(sorted_nids 기준) 비드 높이 배열을 반환합니다.

        요소별 높이를 인접 요소 평균으로 노드에 집계합니다.
        skip_filter는 호환성 인자(no-op) — heights는 이미 필터 거친 물리값.
        """
        del skip_filter
        h_elem = self.heights
        h_node_sum = np.zeros(len(self._design_nids))
        np.add.at(h_node_sum, self._aggr_src, h_elem[self._aggr_dst])
        h_node = h_node_sum / (self._node_adj_count_arr + 1e-12)

        h_full = np.zeros(len(self.sorted_nids))
        for i, nid in enumerate(self._design_nids):
            h_full[self.nid_to_idx[nid]] = h_node[i]
        return h_full


# 하위 호환성을 위한 별칭
JaxTopoSolver = WHTopographySolver


if __name__ == "__main__":
    pass
