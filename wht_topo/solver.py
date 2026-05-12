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
    - 설계 변수 : 바닥면 노드별 비드 높이 h_n ∈ [0, h_max]
    - 목적 함수 : 멀티 케이스 가중 컴플라이언스 C = Σ w_i * F_i^T u_i 최소화
    - 민감도    : ∂C/∂h_n = Σ_{e∈N(n)} u_e^T (∂K_e/∂z_n) u_e  (Adjoint)
                  — QUAD4/TRIA3 모두 JAX Auto-Diff (vmap), 전체 K 재조립 불필요
    - 필터링    : 공간 필터(rmin)로 최소 비드 폭 제어
    - 업데이트  : MMA (Method of Moving Asymptotes)

wht_solver 활용:
    - WHTSolver.solve_static(WHTLoadCase)  → 각 하중 케이스별 변위 해석
    - wht_solver.wht_quad4_element._element_K_mitc4_plus  → 요소 K 계산
    - wht_solver.wht_tria3_element._element_K_tria3       → 요소 K 계산

Dependencies:
    wht_modeler, wht_solver, wht_topo.loads, wht_topo.mma
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from copy import deepcopy

from wht_modeler.wht_mesh_model import WHTMeshModel
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


class WHTopographySolver:
    """
    물리적으로 올바른 Topography Optimization 엔진.

    설계 변수는 노드별 비드 높이 h_n이며, SIMP/밀도 개념은 사용하지 않습니다.
    WHTSolver.solve_static()을 통해 각 하중 케이스를 직접 해석합니다.

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
        bead_connect: bool = False,
        connect_gap: float = 80.0,
        bead_steps: int = 0,
        load_cases: Optional[List[Tuple["WHTLoadCase", float]]] = None,
    ):
        self.model          = model
        self.load_manager   = load_manager
        self.h_max          = bead_height_max
        self.h_ratio        = bead_height_ratio
        self.rmin           = min_width
        self.mesh_size_z    = mesh_size_z
        self.fd_dz          = fd_dz
        self.sym_x          = sym_x
        self.bead_connect   = bead_connect
        self.connect_gap    = connect_gap 
        self.bead_steps     = bead_steps  
        self.weights        = weights or {"bending": 1.0, "twisting": 1.5, "lifting": 1.2}
        self.load_cases_input = load_cases # 외부 주입 하중 케이스

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

        # 설계 노드 식별 (바닥면 노드에서 플랜지 제외)
        print(" -> [Solver] 설계 노드(비드 적용 가능 바닥면) 탐색 중...")
        self._design_nids   = self._find_design_nodes()
        self._n_design      = len(self._design_nids)
        self._design_nid_to_idx = {nid: i for i, nid in enumerate(self._design_nids)}
        print(f"    - 설계 노드 수: {self._n_design}개")

        # 노드 → 인접 요소 인덱스 매핑 (민감도 계산 최적화용)
        self._node_to_elem_idx = self._build_node_elem_adjacency()

        # 공간 필터 행렬 (최소 비드 폭 제어)
        self._H = self._build_filter()

        # 하중 케이스 생성 (하중별 개별 BC 포함)
        if self.load_cases_input:
            print(f" -> [Solver] 외부 주입 하중 케이스 {len(self.load_cases_input)}개 로드됨.")
            self._load_cases = self.load_cases_input
        elif self.load_manager:
            print(" -> [Solver] 하중 케이스 생성 중 (하중별 개별 BC 적용)...")
            self._load_cases = self.load_manager.get_load_cases(
                mesh_size_z=mesh_size_z,
                weights=self.weights,
            )
        else:
            print(" -> [Error] 하중 케이스 정보가 없습니다.")
            self._load_cases = []

        # 좌우 대칭 매핑 (X-mid plane 기준)
        self._sym_map = None
        if self.sym_x:
            self._sym_map = self._build_symmetry_map()

        # 비드 연결 그리드 (Morphological Closing용)
        self._connect_grid = None
        if self.bead_connect:
            self._connect_grid = self._build_connect_grid()
            print(f" -> [Solver] 비드 연결(Bead Connect) 활성화: 간격 {self.connect_gap:.0f}mm 이연 갈라진 비드 연결함")

        # 초기 비드 높이 (0 = 평탄)
        self.heights = np.zeros(self._n_design)
        self.mma     = MMAOptimizer(n_vars=self._n_design, vol_frac=bead_height_ratio)

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

    def _find_design_nodes(self) -> List[int]:
        """
        비드 적용이 가능한 바닥면 노드를 탐색합니다.
        플랜지/경계 노드(최상단 고정 노드)는 제외합니다.

        Returns
        -------
        List[int]
            설계 변수로 사용할 노드 ID 리스트
        """
        if self.load_manager is not None:
            flange_nids = set(self.load_manager.get_boundary_nodes(
                mesh_size_z=self.mesh_size_z
            ))
        else:
            flange_nids = set()

        # 바닥면 노드 = Z가 최소에 가까운 노드
        all_z = [self.model.nodes[nid].z for nid in self.sorted_nids]
        z_min = min(all_z)
        z_threshold = z_min + max(self.mesh_size_z * 0.5, 5.0)

        design_nids = [
            nid for nid in self.sorted_nids
            if self.model.nodes[nid].z <= z_threshold and nid not in flange_nids
        ]
        return design_nids

    def _build_symmetry_map(self) -> np.ndarray:
        """
        X-mid plane을 기준으로 대칭 노드 인덱스 맵을 생성합니다.
        """
        print(" -> [Solver] 좌우 대칭(Sym-X) 노드 매핑 중...")
        coords = np.array([self._coords_orig[nid] for nid in self._design_nids])
        x_mid = (coords[:, 0].min() + coords[:, 0].max()) / 2.0
        
        from scipy.spatial import KDTree
        tree = KDTree(coords)
        
        sym_map = np.arange(self._n_design)
        for i in range(self._n_design):
            # 대칭 점 계산: (x, y, z) -> (2*x_mid - x, y, z)
            target = coords[i].copy()
            target[0] = 2.0 * x_mid - target[0]
            
            dist, idx = tree.query(target)
            if dist < 1.0: # 1mm 이내이면 대칭점으로 인정
                sym_map[i] = idx
        
        return sym_map

    def _build_connect_grid(self) -> dict:
        """
        Morphological Closing 연산을 위한 2D 그리드 인덱스를 사전 구축합니다.

        설계 노드의 X, Y 좌표를 정규 격자에 매핑하여, 클로징 연산 시
        반복적인 좌표 변환 없이 O(1)으로 접근할 수 있게 합니다.

        Returns
        -------
        dict : {"gi": ndarray, "gj": ndarray, "nx": int, "ny": int, "spacing": float}
        """
        coords = np.array([self._coords_orig[nid] for nid in self._design_nids])

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

    # ─────────────── Discrete Projection (Staircase) ───────────────

    def _project_x(self, x: np.ndarray, beta: float) -> np.ndarray:
        """
        설계 변수를 지정된 N단계 이산 레벨로 부드럽게 투사합니다.
        
        N=2 이면 {0, 1}, N=3 이면 {0, 0.5, 1} 등으로 수렴하도록 tanh 기반 계단 함수 적용.
        """
        if self.bead_steps < 2:
            return x
        
        n = self.bead_steps
        # 각 스텝 사이의 임계점(Thresholds)
        thresholds = np.linspace(0, 1, n * 2 + 1)[1:-1:2]
        levels = np.linspace(0, 1, n)
        
        # Staircase function: f(x) = l_0 + sum( (l_{k+1} - l_k) * 0.5 * (tanh(beta*(x - t_k)) + 1) )
        x_proj = np.full_like(x, levels[0])
        for i in range(n - 1):
            diff = levels[i+1] - levels[i]
            x_proj += diff * 0.5 * (np.tanh(beta * (x - thresholds[i])) + 1.0)
            
        return x_proj

    def _project_x_grad(self, x: np.ndarray, beta: float) -> np.ndarray:
        """
        투사 함수의 미분값 (Chain-rule 용).
        """
        if self.bead_steps < 2:
            return np.ones_like(x)
        
        n = self.bead_steps
        thresholds = np.linspace(0, 1, n * 2 + 1)[1:-1:2]
        levels = np.linspace(0, 1, n)
        
        # df/dx = sum( (l_{k+1} - l_k) * 0.5 * beta * (1 - tanh^2(beta*(x - t_k))) )
        grad = np.zeros_like(x)
        for i in range(n - 1):
            diff = levels[i+1] - levels[i]
            grad += diff * 0.5 * beta * (1.0 - np.tanh(beta * (x - thresholds[i]))**2)
            
        return grad

    def _build_node_elem_adjacency(self) -> Dict[int, List[int]]:
        """
        설계 노드별로 인접한 요소의 인덱스 리스트를 사전 계산합니다.
        민감도 계산 시 전체 K 재조립 없이 요소별 국부 계산에 사용합니다.

        Returns
        -------
        Dict[int, List[int]]
            {노드 ID : [인접 요소의 elem_ids 인덱스 리스트]}
        """
        design_nid_set = set(self._design_nids)
        node_to_eidx: Dict[int, List[int]] = {nid: [] for nid in self._design_nids}

        for eidx, eid in enumerate(self.elem_ids):
            for nid in self.model.elements[eid].node_ids:
                if nid in design_nid_set:
                    node_to_eidx[nid].append(eidx)

        return node_to_eidx

    def _build_filter(self) -> np.ndarray:
        """
        설계 노드 간 공간 필터 행렬 H를 조립합니다 (최소 비드 폭 제어).

        Returns
        -------
        np.ndarray
            (n_design, n_design) 필터 가중치 행렬
        """
        print(f" -> [Solver] 공간 필터 행렬 ({self.rmin}mm) 조립 중...")
        n = self._n_design
        coords = np.array([
            [self.model.nodes[nid].x, self.model.nodes[nid].y, self.model.nodes[nid].z]
            for nid in self._design_nids
        ])
        W = np.zeros((n, n))
        for i in range(n):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            mask  = dists < self.rmin
            w     = self.rmin - dists[mask]
            W[i, mask] = w / (np.sum(w) + 1e-10)
        return W

    # ─────────────── 노드 좌표 조작 ───────────────

    def _apply_heights(self, h: np.ndarray):
        """
        비드 높이 배열 h를 설계 노드의 3D 좌표에 반영합니다.

        Parameters
        ----------
        h : (n_design,) 비드 높이 배열 [mm]
        """
        for i, nid in enumerate(self._design_nids):
            orig_xyz = self._coords_orig[nid]
            move_vec = float(h[i]) * self.bead_dir
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

    def _compute_total_compliance(self, solver) -> Tuple[float, dict, np.ndarray]:
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
        total_C : float, case_responses : dict, total_sens : (n_design,)
        """
        from scipy.sparse.linalg import spsolve
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from wht_solver.wht_stress_recovery import ElementStressRecovery
        from wht_solver.wht_result import WHTSolverResult

        ndof       = len(self.sorted_nids) * 6
        total_C    = 0.0
        case_responses = {}
        total_sens = np.zeros(self._n_design)

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

        # ── 3. 하중 케이스별 FEA 워커 (스레드 내부, scipy만 사용) ─────────────
        def _solve_one(load_case: "WHTLoadCase"):
            jm_lc, _, _ = solver._build_jaxsso_model(load_case=load_case)
            K_aug, f_aug = solver._augment_K_scipy(K_base, jm_lc)
            u_aug_np = np.array(spsolve(K_aug.tocsc(), f_aug))

            n_nodes = len(self.sorted_nids)
            displacement = np.zeros((n_nodes, 6))
            for ii, nid in enumerate(self.sorted_nids):
                displacement[ii, :] = u_aug_np[self.nid_to_idx[nid] * 6:
                                                self.nid_to_idx[nid] * 6 + 6]

            rd_q = ElementStressRecovery.recover_quad4(solver.model, displacement, self.sorted_nids)
            rd_t = ElementStressRecovery.recover_tria3(solver.model, displacement, self.sorted_nids)
            cell_data = {k: (rd_q[k] + rd_t[k])[np.newaxis, :, :] for k in rd_q}

            u_full = np.zeros(ndof)
            for ii, nid in enumerate(self.sorted_nids):
                u_full[self.nid_to_idx[nid] * 6:
                       self.nid_to_idx[nid] * 6 + 6] = displacement[ii, :]

            f_full = np.zeros(ndof)
            for force in load_case.forces:
                idx = self.nid_to_idx.get(force.node_id)
                if idx is None: continue
                for d, fval in enumerate(force.load_vector):
                    if abs(fval) > 1e-12:
                        f_full[idx * 6 + d] += fval

            if load_case.forces and not load_case.bcs:
                # 순수 힘 주도 케이스: C = F · u  (equilibrium에서 u^T·K·u와 동일)
                C_i = float(np.dot(f_full, u_full))
            else:
                # 변위 BC 포함 케이스(SPCD/ESL): C = u^T · K · u (항상 양수인 변형 에너지)
                C_i = float(u_full @ (K_base @ u_full))

            result = WHTSolverResult("static", self.sorted_nids)
            result.displacement = displacement
            result.cell_data    = cell_data
            result._u_aug       = u_aug_np
            result._ndof        = ndof
            return load_case.name, C_i, u_full, displacement, cell_data, result

        # ── 4. ThreadPoolExecutor 병렬 실행 ────────────────────────────────────
        # UMFPACK(spsolve)은 단일스레드 → 멀티코어에서 진정한 병렬화
        # BLAS 과부하 방지를 위해 워커 수 4개 상한
        n_workers = min(len(self._load_cases), 4)
        solve_results = [None] * len(self._load_cases)
        lc_list = [(i, lc, w) for i, (lc, w) in enumerate(self._load_cases)]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_solve_one, lc): (i, w)
                       for i, lc, w in lc_list}
            for future in as_completed(futures):
                i, w = futures[future]
                solve_results[i] = (w, future.result())

        # ── 5. 결과 수집 + JAX 민감도 (순차 — JAX GIL 이슈) ──────────────────
        for weight, (name, C_i, u_full, displacement, cell_data_r, result) in solve_results:
            total_C += weight * C_i
            stress_vals = cell_data_r["Stress"][0]
            case_responses[name] = {
                "C":          C_i,
                "max_disp":   float(np.max(np.linalg.norm(displacement[:, :3], axis=1))),
                "max_stress": _von_mises_max(stress_vals),
                "result":     result,
            }

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
                valid_design_idx = np.where(mask, self._quad_node_design_idx, 0)
                np.add.at(total_sens, valid_design_idx, np.where(mask, -weight * proj, 0.0))

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
                valid_design_idx_t = np.where(mask_t, self._tria_node_design_idx, 0)
                np.add.at(total_sens, valid_design_idx_t, np.where(mask_t, -weight * proj_t, 0.0))

        return total_C, case_responses, total_sens

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

        # MMA용 정규화 변수 x [0, 1]
        x = self.heights.copy() / (self.h_max + 1e-12)
        if np.max(x) < 1e-6:
            x = np.full_like(x, 0.01)

        # FEA 솔버 인스턴스 생성
        from wht_solver.wht_solver import WHTSolver
        fea_solver = WHTSolver(self.model)
        
        # ── 결과 저장용 디렉토리 및 Exporter 초기화 ──
        from datetime import datetime
        import os
        from wht_converter.wht_models import WHTMetadata
        from wht_converter.wht_exporters import VTKHDFExporter

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

        C_0 = None 

        for i in range(max_iter):
            # 좌우 대칭 강제 (변수 동기화)
            if self.sym_x:
                x = 0.5 * (x + x[self._sym_map])

            x_old = x.copy()

            # ── 1. Discrete Projection & Filtering ──
            # beta continuation: 선형 증가, 전체 반복의 80%에서 max(50) 도달
            # (기존 100 → 60으로 완화: 패턴 안정화 전 조기 이산화 방지)
            if self.bead_steps >= 2:
                self._beta = min(50.0, 1.0 + (i / max_iter) * 60.0)
                x_proj = self._project_x(x, self._beta)
            else:
                x_proj = x

            # 현재 비드 높이 (물리량) 계산 및 적용
            h_phys     = x_proj * self.h_max
            h_filtered = self._H @ h_phys
            self._apply_heights(h_filtered)

            # 컴플라이언스 및 민감도 계산 (솔버 인스턴스 공유)
            C_total, C_responses, dC_dh = self._compute_total_compliance(fea_solver)
            
            # 초기값 저장 및 정규화 (수렴 안정성 강화)
            if C_0 is None:
                C_0 = float(C_total) if float(C_total) > 1e-6 else 1.0

            # ── 고유진동수 해석 (10차 모드) ──
            try:
                modal_results = fea_solver.solve_modal(num_modes=10, exclude_rigid_body=False)
                freqs = modal_results.frequencies  # Hz
            except Exception as e:
                print(f"    [경고] 고유진동수 해석 실패 (Iter {i}): {e}")
                modal_results = None
                freqs = np.zeros(10)

            # ── 이터레이션 결과 저장 (ParaView 호환 VTKHDF) ──
            try:
                # 기본 WHTResultData 생성 (첫 번째 하중 케이스 기준)
                first_case_name = self._load_cases[0][0].name
                base_res = C_responses[first_case_name]["result"]
                wht_data = base_res.to_wht_result_data(meta, self.model)
                
                # 단일 Displacement/Stress 키 삭제 (이름 충돌 방지)
                wht_data.point_data.pop("Displacement", None)
                wht_data.cell_data.pop("Stress", None)
                
                # 1. 비드 높이 (Bead Height) 추가
                h_current_full = np.zeros(len(self.sorted_nids))
                for idx_design, nid in enumerate(self._design_nids):
                    h_current_full[self.nid_to_idx[nid]] = h_filtered[idx_design]
                wht_data.point_data["Bead_Height"] = h_current_full.reshape(1, -1, 1).astype(np.float32)

                # 2. 하중 케이스별 결과 병합
                for name, res_dict in C_responses.items():
                    res_data = res_dict["result"].to_wht_result_data(meta, self.model)
                    if "Displacement" in res_data.point_data:
                        wht_data.point_data[f"Disp_{name}"] = res_data.point_data["Displacement"]
                    if "Stress" in res_data.cell_data:
                        wht_data.cell_data[f"Stress_{name}"] = res_data.cell_data["Stress"]

                # 3. 고유 모드 형상 추가 및 주파수 정보 명시
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
            except Exception as e:
                print(f"    [경고] 해석 결과 저장 실패: {e}")

            # 목적 함수 및 민감도 정규화 (MMA 엔진이 제약 조건을 더 잘 인식하도록 함)
            f0val = float(C_total) / C_0
            
            # 필터 및 투사 역전파 (Chain-rule)
            # dC/dx = (dx_proj/dx) * (dh/dx_proj) * (dH/dh) * (dC/dH)
            # 1. 필터 역전파: dC/dh_phys = H^T @ dC/dh_filtered
            dC_dh_phys = (self._H.T @ dC_dh)
            # 2. 물리적 변환 및 투사 역전파: dC/dx = (dC/dh_phys * h_max) * dx_proj/dx
            dC_dx = (dC_dh_phys * self.h_max)
            if self.bead_steps >= 2:
                dC_dx *= self._project_x_grad(x, self._beta)

            df0dx = (dC_dx / C_0).flatten()

            # 좌우 대칭 강제 (민감도 동기화)
            if self.sym_x:
                df0dx = 0.5 * (df0dx + df0dx[self._sym_map])

            # ── 비드 연결 phase 1: df0dx 승격 + vol_frac 사전 보정 ────────────────
            # _apply_bead_connect를 1회만 호출해 두 가지를 동시 처리:
            #   a) bridge 노드 df0dx를 활성 이웃 최댓값으로 승격 (MMA가 bridge 유지하도록)
            #   b) bridge 추가로 인한 체적 팽창을 사전에 vol_frac에서 차감
            #      (미보정 시: MMA가 다음 이터에서 약한 노드를 0으로 압축 → 비드 수축)
            if self.bead_connect:
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
                    if self.bead_steps >= 2:
                        vol_before = float(np.mean(self._project_x(x, self._beta)))
                        vol_after  = float(np.mean(self._project_x(x_bridged, self._beta)))
                    else:
                        vol_before = float(np.mean(x))
                        vol_after  = float(np.mean(x_bridged))
                    bridge_excess = max(0.0, vol_after - vol_before)
                    self.mma.vol_frac = max(0.01, self.h_ratio - bridge_excess)
                else:
                    self.mma.vol_frac = self.h_ratio
            else:
                self.mma.vol_frac = self.h_ratio

            # 제약 조건 민감도 (Chain-rule: df/dx = (1/N) * dx_proj/dx)
            dfdx = np.full(self._n_design, 1.0 / self._n_design)
            if self.bead_steps >= 2:
                dfdx *= self._project_x_grad(x, self._beta)

            fval = np.array([float(np.mean(x_proj)) - self.h_ratio])

            # MMA 업데이트
            # bead_steps >= 2: project_fn 전달 → 이분법이 mean(x_proj) = h_ratio 기준으로 수렴
            # bead_steps  < 2: project_fn=None → 기존 mean(x) 기준 (연속 변수이므로 동일)
            if self.bead_steps >= 2:
                project_fn = lambda xv: self._project_x(np.array(xv), self._beta)
            else:
                project_fn = None
            x = self.mma.update(
                x, f0val, df0dx,
                fval,
                dfdx,
                i + 1,
                project_fn=project_fn,
            )
            x = np.clip(x, 0.0, 1.0)

            # ── 비드 연결 phase 2: 물리적 closing 적용 ───────────────────────────
            if self.bead_connect:
                x, _ = self._apply_bead_connect(x)

            # 수렴 판정
            change = float(np.abs(x - x_old).max())
            
            # 모니터링 콜백 호출
            if callback:
                # 설계 노드 좌표 추출 (2D 시각화용)
                coords = np.array([
                    [self.model.nodes[nid].x, self.model.nodes[nid].y, self.model.nodes[nid].z]
                    for nid in self._design_nids
                ])
                # 비드 점유 면적 비율: 정규화 변수 x > 0.1 인 노드 비율
                area_ratio = float(np.mean(x > 0.1))
                
                # 변형 에너지는 컴플라이언스의 절반 (U = 0.5 * C)
                case_data = {}
                for name, res in C_responses.items():
                    case_data[name] = {
                        "U": 0.5 * res["C"],
                        "max_disp": res["max_disp"],
                        "max_stress": res["max_stress"]
                    }
                    
                data = {
                    "iter": i,
                    "compliance": float(C_total),
                    "area_ratio": float(np.mean(x_proj > 0.1)), # 투사된 변수 기준 면적
                    "cases": case_data,
                    "frequencies": freqs.tolist(),
                    "avg_h": float(np.mean(h_filtered)),
                    "max_h": float(np.max(h_filtered)),
                    "dx": change,
                    "coords": coords,
                    "heights": h_filtered.copy() # 시각화 시에도 필터링/투사된 최종 높이 전달
                }
                callback(data)

            if i % 5 == 0:
                print(f"  Iter {i:3d}: C_total={C_total:.4e}  ", end="")
                # 대표 진동수 출력
                elastic_freqs = [f for f in freqs if f > 0.1]
                if elastic_freqs:
                    print(f"F1={elastic_freqs[0]:.1f}Hz ", end="")
                
                for name, res in C_responses.items():
                    # 에너지(J), 최대 변위(mm), 최대 응력(MPa) 순차 출력
                    print(f"[{name}: {0.5*res['C']:.1e}J, {res['max_disp']:.1f}mm, {res['max_stress']:.0f}MPa] ", end="")
                print(f"Avg_h={np.mean(x*self.h_max):.2f}mm dx={change:.4f}")
            if change < tol:
                print(f"  -> 수렴 달성 (Iter {i})")
                break
                
            if stop_event and stop_event.is_set():
                print(f"  -> 사용자 중단 요청에 의한 조기 종료 (Iter {i})")
                break

        # 원본 좌표 복원 후 최종 형상 적용
        self._restore_heights()
        # 최종 결과도 투사 적용
        self.heights = self._project_x(x, self._beta) * self.h_max
        return self.heights

    def apply_final_shape(self, skip_filter: bool = False):
        """
        최적화 결과(self.heights)를 모델 노드 좌표에 영구 적용합니다.
        solve() 완료 후 export 전에 반드시 호출하십시오.

        Parameters
        ----------
        skip_filter : bool
            True이면 공간 필터(_H)를 재적용하지 않습니다.
            이산화(--height-steps) 후 호출 시 반드시 True로 설정해야
            양자화된 높이값이 블러링되지 않습니다.
        """
        h = self.heights if skip_filter else self._H @ self.heights
        self._apply_heights(h)
        print(f" -> [Solver] 최종 비드 형상 적용 완료 "
              f"(Max: {np.max(h):.2f}mm, "
              f"Avg: {np.mean(h):.2f}mm)")

    def get_full_heights(self, skip_filter: bool = False) -> np.ndarray:
        """
        시각화용 전체 노드(sorted_nids 기준) 비드 높이 배열을 반환합니다.
        """
        h_full = np.zeros(len(self.sorted_nids))
        h = self.heights if skip_filter else self._H @ self.heights
        for i, nid in enumerate(self._design_nids):
            idx = self.nid_to_idx[nid]
            h_full[idx] = h[i]
        return h_full


# 하위 호환성을 위한 별칭
JaxTopoSolver = WHTopographySolver


if __name__ == "__main__":
    pass
