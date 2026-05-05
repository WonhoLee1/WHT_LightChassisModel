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
                  — 요소 수준 중앙차분, ∂K_e/∂z_n만 재계산 (전체 K 재조립 불필요)
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

# JAX 민감도 함수 사전 정의
from wht_solver.wht_quad4_element_jax import _element_K_mitc4_plus_jax

@jax.jit
def element_energy_jax(c1, c2, c3, c4, u_e, t, E, nu):
    K_e = _element_K_mitc4_plus_jax(c1, c2, c3, c4, t, E, nu)
    return jnp.dot(u_e, jnp.dot(K_e, u_e))

# 4개의 노드 좌표(c1,c2,c3,c4)에 대한 자동 미분 (Auto-Diff)
element_grad_jax = jax.grad(element_energy_jax, argnums=(0, 1, 2, 3))
# vmap을 통한 모델 전체 요소 동시(병렬) 처리
vmap_element_grad_jax = jax.jit(jax.vmap(element_grad_jax, in_axes=(0,0,0,0,0,0,0,0)))


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
        load_manager: StochasticLoadManager,
        constraints=None,       # 향후 확장용 (현재 미사용)
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
        self.connect_gap    = connect_gap  # mm, 버드 연결 시 채울 최대 간격
        self.bead_steps     = bead_steps   # 0 = 연속, N >= 1 = N단계 이산 높이
        self.weights        = weights or {"bending": 1.0, "twisting": 1.5, "lifting": 1.2}

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
        print(" -> [Solver] 하중 케이스 생성 중 (하중별 개별 BC 적용)...")
        self._load_cases = load_manager.get_load_cases(
            mesh_size_z=mesh_size_z,
            weights=self.weights,
        )

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

    # ─────────────── 전처리 ───────────────

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
        flange_nids = set(self.load_manager.get_boundary_nodes(
            mesh_size_z=self.mesh_size_z
        ))

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

    def _apply_bead_connect(self, x: np.ndarray, threshold: float = 0.1) -> np.ndarray:
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
        """
        from scipy.ndimage import binary_dilation, binary_erosion

        cg  = self._connect_grid
        gi, gj   = cg["gi"], cg["gj"]
        nx, ny   = cg["nx"], cg["ny"]
        spacing  = cg["spacing"]
        g2n      = cg["grid_to_node"]

        # 1. 설계 변수를 2D 그리드에 래스터화
        grid = np.zeros((ny, nx), dtype=bool)
        for k in range(self._n_design):
            if x[k] > threshold:
                grid[gj[k], gi[k]] = True

        # 2. Morphological Closing: Dilation → Erosion
        #    n_iter = connect_gap / spacing 만큼 팽창 후 수축 → 갭 메움
        n_iter = max(1, int(np.ceil(self.connect_gap / spacing)))
        dilated = grid.copy()
        for _ in range(n_iter):
            dilated = binary_dilation(dilated)
        closed = dilated.copy()
        for _ in range(n_iter):
            closed = binary_erosion(closed)

        # 3. 클로징으로 새로 채워진 셀(브릿지 노드)을 최소 활성값으로 승격
        bridge_mask = closed & ~grid  # 원래 비활성이었으나 채워진 셀
        x_new = x.copy()
        for k in range(self._n_design):
            if bridge_mask[gj[k], gi[k]]:
                # 주변 활성 노드 평균값으로 승격 (부드러운 전환)
                row, col = gj[k], gi[k]
                neighbor_vals = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nb = g2n.get((row + dr, col + dc))
                    if nb is not None and x[nb] > threshold:
                        neighbor_vals.append(x[nb])
                if neighbor_vals:
                    x_new[k] = max(x_new[k], np.mean(neighbor_vals) * 0.7)
                else:
                    x_new[k] = max(x_new[k], threshold + 0.02)
        return x_new

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

        Parameters
        ----------
        solver : WHTSolver
            FEA 연산을 수행할 솔버 인스턴스

        Returns
        -------
        total_C : float
            가중 합산 컴플라이언스
        case_responses : dict
            하중별 응답 데이터 (C, max_disp, max_stress)
        total_sens : (n_design,) 설계 노드별 민감도
        """
        ndof = len(self.sorted_nids) * 6
        total_C    = 0.0
        case_responses = {} # {name: {"C": float, "max_disp": float, "max_stress": float}}
        total_sens = np.zeros(self._n_design)

        # JAX Vectorization용 데이터 컨테이너 (QUAD4)
        n_quad = sum(1 for eid in self.elem_ids if self.model.elements[eid].type.upper() in self._QUAD_TYPES)
        quad_c1 = np.zeros((n_quad, 3)); quad_c2 = np.zeros((n_quad, 3))
        quad_c3 = np.zeros((n_quad, 3)); quad_c4 = np.zeros((n_quad, 3))
        quad_ue = np.zeros((n_quad, 24))
        quad_t = np.zeros(n_quad); quad_E = np.zeros(n_quad); quad_nu = np.zeros(n_quad)
        quad_nids = np.zeros((n_quad, 4), dtype=int)
        
        # TRIA3 요소용 컨테이너
        n_tria = sum(1 for eid in self.elem_ids if self.model.elements[eid].type.upper() in self._TRIA_TYPES)
        tria_eidxs = [i for i, eid in enumerate(self.elem_ids) if self.model.elements[eid].type.upper() in self._TRIA_TYPES]

        for load_case, weight in self._load_cases:
            # ── 1. FEA 수행 ──────────────
            result = solver.solve_static(load_case)
            u_full = np.zeros(ndof)
            for i, nid in enumerate(self.sorted_nids):
                idx = self.nid_to_idx[nid]
                u_full[idx * 6: idx * 6 + 6] = result.displacement[i, :]

            # ── 2. 컴플라이언스 및 응답 ─────────────────────────────
            f_full = np.zeros(ndof)
            for force in load_case.forces:
                idx = self.nid_to_idx.get(force.node_id)
                if idx is None: continue
                for d, fval in enumerate(force.load_vector):
                    if abs(fval) > 1e-12:
                        f_full[idx * 6 + d] += fval

            C_i = float(np.dot(f_full, u_full))
            total_C += weight * C_i
            
            # 물리적 응답 추출 (최대 변위, 최대 응력)
            # result.displacement shape: (N, 6)
            # result.cell_data["Stress"] shape: (1, M, 6)
            disp_vals = result.displacement
            stress_vals = result.cell_data["Stress"][0]
            
            case_responses[load_case.name] = {
                "C": C_i,
                "max_disp": float(np.max(np.linalg.norm(disp_vals[:, :3], axis=1))),
                "max_stress": _von_mises_max(stress_vals)
            }

            # ── 3. QUAD4 민감도 (JAX) ─────────────────────
            q_idx = 0
            for eidx, eid in enumerate(self.elem_ids):
                elem = self.model.elements[eid]
                if elem.type.upper() in self._QUAD_TYPES:
                    nids = elem.node_ids
                    quad_c1[q_idx] = [self.model.nodes[nids[0]].x, self.model.nodes[nids[0]].y, self.model.nodes[nids[0]].z]
                    quad_c2[q_idx] = [self.model.nodes[nids[1]].x, self.model.nodes[nids[1]].y, self.model.nodes[nids[1]].z]
                    quad_c3[q_idx] = [self.model.nodes[nids[2]].x, self.model.nodes[nids[2]].y, self.model.nodes[nids[2]].z]
                    quad_c4[q_idx] = [self.model.nodes[nids[3]].x, self.model.nodes[nids[3]].y, self.model.nodes[nids[3]].z]
                    prop = self.model.properties[elem.pid]
                    mat = self.model.materials[prop.mid]
                    quad_t[q_idx] = prop.t
                    quad_E[q_idx] = mat.E
                    quad_nu[q_idx] = mat.nu
                    e_dofs = [self.nid_to_idx[n]*6+d for n in nids for d in range(6)]
                    quad_ue[q_idx] = u_full[e_dofs]
                    quad_nids[q_idx] = nids
                    q_idx += 1

            if q_idx > 0:
                grads = vmap_element_grad_jax(
                    jnp.array(quad_c1), jnp.array(quad_c2), jnp.array(quad_c3), jnp.array(quad_c4),
                    jnp.array(quad_ue), jnp.array(quad_t), jnp.array(quad_E), jnp.array(quad_nu)
                )
                for q in range(q_idx):
                    nids = quad_nids[q]
                    for i_local, grad_vec in enumerate([grads[0][q], grads[1][q], grads[2][q], grads[3][q]]):
                        idx = self._design_nid_to_idx.get(nids[i_local])
                        if idx is not None:
                            total_sens[idx] -= weight * float(np.dot(np.array(grad_vec), self.bead_dir))

            # ── 4. TRIA3 민감도 (3D 중앙차분) ────────────────────────
            for eidx in tria_eidxs:
                eid = self.elem_ids[eidx]
                elem = self.model.elements[eid]
                nids = elem.node_ids
                if not any(nid in self._design_nid_to_idx for nid in nids): continue
                
                for nid in nids:
                    idx = self._design_nid_to_idx.get(nid)
                    if idx is None: continue
                    
                    node = self.model.nodes[nid]
                    orig_xyz = np.array([node.x, node.y, node.z])
                    move = self.fd_dz * self.bead_dir
                    
                    node.x, node.y, node.z = orig_xyz + move
                    Ke_p = self._compute_element_K(eidx)
                    node.x, node.y, node.z = orig_xyz - move
                    Ke_m = self._compute_element_K(eidx)
                    node.x, node.y, node.z = orig_xyz # 복구
                    
                    dKe_dh = (Ke_p - Ke_m) / (2.0 * self.fd_dz)
                    e_dofs = [self.nid_to_idx[n]*6+d for n in nids for d in range(6)]
                    ue = u_full[e_dofs]
                    total_sens[idx] -= weight * float(ue @ dKe_dh @ ue)

        return total_C, case_responses, total_sens

    # ─────────────── 메인 최적화 루프 ───────────────

    def solve(self, max_iter: int = 30, tol: float = 1e-4, callback=None) -> np.ndarray:
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
        
        C_0 = None 

        for i in range(max_iter):
            # 좌우 대칭 강제 (변수 동기화)
            if self.sym_x:
                x = 0.5 * (x + x[self._sym_map])

            x_old = x.copy()

            # 현재 비드 높이 (물리량) 계산 및 적용
            h_phys = x * self.h_max
            h_filtered = self._H @ h_phys
            self._apply_heights(h_filtered)

            # 컴플라이언스 및 민감도 계산 (솔버 인스턴스 공유)
            C_total, C_responses, dC_dh = self._compute_total_compliance(fea_solver)
            
            # 초기값 저장 및 정규화 (수렴 안정성 강화)
            if C_0 is None:
                C_0 = float(C_total) if float(C_total) > 1e-6 else 1.0

            # ── 고유진동수 해석 (10차 모드) ──
            modal_results = fea_solver.solve_modal(num_modes=10, exclude_rigid_body=False)
            freqs = modal_results.frequencies  # Hz

            # 목적 함수 및 민감도 정규화 (MMA 엔진이 제약 조건을 더 잘 인식하도록 함)
            f0val = float(C_total) / C_0
            
            # 필터 역전파 및 정규화 체인룰 적용
            dC_dx = (self._H.T @ dC_dh) * self.h_max
            df0dx = (dC_dx / C_0).flatten()
            
            # 좌우 대칭 강제 (민감도 동기화)
            if self.sym_x:
                df0dx = 0.5 * (df0dx + df0dx[self._sym_map])

            # 제약 조건: Σ x_n / n_vars ≤ h_ratio  (1D 벡터, MMA 내부 브로드캐스트 요구)
            fval = np.array([float(np.mean(x)) - self.h_ratio])
            dfdx = np.full(self._n_design, 1.0 / self._n_design)  # (N,) — Bug #3 수정

            # MMA 업데이트
            x = self.mma.update(
                x, f0val, df0dx,
                fval,
                dfdx,
                i + 1,
            )
            x = np.clip(x, 0.0, 1.0)

            # 비드 연결: 단절된 비드 노드를 모폴로지 클로징으로 연결
            if self.bead_connect:
                x = self._apply_bead_connect(x)

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
                    "area_ratio": area_ratio,
                    "cases": case_data,
                    "frequencies": freqs.tolist(),
                    "avg_h": float(np.mean(x * self.h_max)),
                    "max_h": float(np.max(x * self.h_max)),
                    "dx": change,
                    "coords": coords,
                    "heights": (x * self.h_max).copy()
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

        # 원본 좌표 복원 후 최종 형상 적용
        self._restore_heights()
        self.heights = x * self.h_max
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
