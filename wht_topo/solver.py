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
    ):
        self.model          = model
        self.load_manager   = load_manager
        self.h_max          = bead_height_max
        self.h_ratio        = bead_height_ratio
        self.rmin           = min_width
        self.mesh_size_z    = mesh_size_z
        self.fd_dz          = fd_dz
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

    def _compute_total_compliance(self) -> Tuple[float, np.ndarray]:
        """
        현재 형상에서 모든 하중 케이스의 가중 컴플라이언스와
        설계 노드별 민감도를 계산합니다.

        WHTSolver.solve_static()을 각 하중 케이스에 직접 활용합니다.
        민감도는 Adjoint 공식으로 계산합니다:
            ∂C/∂h_n = Σ_{e∈N(n)} u_e^T (∂K_e/∂z_n) u_e

        Returns
        -------
        total_C : float
            가중 합산 컴플라이언스
        total_sens : (n_design,) 설계 노드별 민감도
        """
        ndof = len(self.sorted_nids) * 6
        total_C    = 0.0
        total_sens = np.zeros(self._n_design)

        solver = WHTSolver(self.model)

        # JAX Vectorization용 데이터 컨테이너 (QUAD4)
        n_quad = sum(1 for eid in self.elem_ids if self.model.elements[eid].type.upper() in self._QUAD_TYPES)
        quad_c1 = np.zeros((n_quad, 3)); quad_c2 = np.zeros((n_quad, 3))
        quad_c3 = np.zeros((n_quad, 3)); quad_c4 = np.zeros((n_quad, 3))
        quad_ue = np.zeros((n_quad, 24))
        quad_t = np.zeros(n_quad); quad_E = np.zeros(n_quad); quad_nu = np.zeros(n_quad)
        quad_nids = np.zeros((n_quad, 4), dtype=int)
        
        # TRIA3 요소용 컨테이너 (병렬화 없이 루프로 처리하되 설계 노드 인접성 활용)
        n_tria = sum(1 for eid in self.elem_ids if self.model.elements[eid].type.upper() in self._TRIA_TYPES)
        tria_eidxs = [i for i, eid in enumerate(self.elem_ids) if self.model.elements[eid].type.upper() in self._TRIA_TYPES]

        for load_case, weight in self._load_cases:
            # ── 1. WHTSolver.solve_static()으로 실제 FEA 수행 ──────────────
            result = solver.solve_static(load_case)

            # 변위를 전체 DOF 벡터로 전환 (N,6) → (N*6,)
            u_full = np.zeros(ndof)
            for i, nid in enumerate(self.sorted_nids):
                idx = self.nid_to_idx[nid]
                u_full[idx * 6: idx * 6 + 6] = result.displacement[i, :]

            # ── 2. 컴플라이언스 계산: C = F^T u ─────────────────────────────
            f_full = np.zeros(ndof)
            for force in load_case.forces:
                idx = self.nid_to_idx.get(force.node_id)
                if idx is None: continue
                for d, fval in enumerate(force.load_vector):
                    if abs(fval) > 1e-12:
                        f_full[idx * 6 + d] += fval

            C_i = float(np.dot(f_full, u_full))
            total_C += weight * C_i

            # ── 3. JAX 기반 초고속 자동 미분(Auto-Diff) ─────────────────────
            q_idx = 0
            for eidx, eid in enumerate(self.elem_ids):
                elem = self.model.elements[eid]
                etype = elem.type.upper()
                if etype in self._QUAD_TYPES:
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
                # JAX 병렬 처리 (수천 개의 요소를 단 한 번의 호출로 처리)
                grads = vmap_element_grad_jax(
                    jnp.array(quad_c1), jnp.array(quad_c2), jnp.array(quad_c3), jnp.array(quad_c4),
                    jnp.array(quad_ue), jnp.array(quad_t), jnp.array(quad_E), jnp.array(quad_nu)
                )
                
                grad_c1_np = np.array(grads[0])
                grad_c2_np = np.array(grads[1])
                grad_c3_np = np.array(grads[2])
                grad_c4_np = np.array(grads[3])

                # 계산된 그래디언트를 각 설계 노드에 분배 (컴플라이언스 최소화: - 부호 적용)
                # 3D 그래디언트 벡터와 bead_dir의 내적(Dot product)을 통해 정확한 방향 민감도 산출
                for q in range(q_idx):
                    nids = quad_nids[q]
                    for i_local, grad_vec in enumerate([grad_c1_np[q], grad_c2_np[q], grad_c3_np[q], grad_c4_np[q]]):
                        idx = self._design_nid_to_idx.get(nids[i_local])
                        if idx is not None:
                            total_sens[idx] -= weight * np.dot(grad_vec, self.bead_dir)

            # ── 4. TRIA3 요소 민감도 (중앙차분 루프) ────────────────────────
            # JAX화되지 않은 삼각형 요소도 빠짐없이 처리하여 메시 정합성 유지
            for eidx in tria_eidxs:
                eid = self.elem_ids[eidx]
                elem = self.model.elements[eid]
                nids = elem.node_ids
                
                # 설계 노드가 포함된 경우만 계산
                if not any(nid in self._design_nid_to_idx for nid in nids): continue
                
                # 각 노드별 중앙차분
                for nid in nids:
                    idx = self._design_nid_to_idx.get(nid)
                    if idx is None: continue
                    
                    node = self.model.nodes[nid]
                    z0 = node.z
                    
                    node.z = z0 + self.fd_dz
                    Ke_p = self._compute_element_K(eidx)
                    node.z = z0 - self.fd_dz
                    Ke_m = self._compute_element_K(eidx)
                    node.z = z0 # 복구
                    
                    dKe_dz = (Ke_p - Ke_m) / (2.0 * self.fd_dz)
                    e_dofs = [self.nid_to_idx[n]*6+d for n in nids for d in range(6)]
                    ue = u_full[e_dofs]
                    
                    # 삼각형 요소는 현재 Z방향 변위 기준 민감도로 근사 (향후 3D 확장 가능)
                    total_sens[idx] -= weight * float(ue @ dKe_dz @ ue) * self.bead_dir[2]

        return total_C, total_sens

    # ─────────────── 메인 최적화 루프 ───────────────

    def solve(self, max_iter: int = 30, tol: float = 1e-4) -> np.ndarray:
        """
        MMA 기반 Topography Optimization을 실행합니다.

        Parameters
        ----------
        max_iter : int
            최대 반복 수
        tol : float
            수렴 판정 기준 (비드 높이 변화 최대값)

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
        # 초기값으로 아주 작은 값(0.01)을 주어 최적화 초기 방향성 확보
        x = self.heights.copy() / (self.h_max + 1e-12)
        if np.max(x) < 1e-6:
            x = np.full_like(x, 0.01)

        for i in range(max_iter):
            x_old = x.copy()

            # 현재 비드 높이 (물리량) 계산 및 적용
            h_phys = x * self.h_max
            h_filtered = self._H @ h_phys
            self._apply_heights(h_filtered)

            # 컴플라이언스 및 민감도 계산
            C, dC_dh = self._compute_total_compliance()

            # 필터 역전파 및 정규화 체인룰 적용: ∂C/∂x = (H^T * ∂C/∂h_filtered) * h_max
            dC_dx = (self._H.T @ dC_dh) * self.h_max

            # 제약 조건: Σ x_n / n_vars ≤ h_ratio
            vol_constraint = float(np.mean(x)) - self.h_ratio
            dv_dx = np.ones(self._n_design) / self._n_design

            # MMA 업데이트 (내부적으로 [0, 1] 범위 최적화)
            x = self.mma.update(
                x, C, dC_dx,
                np.array([vol_constraint]),
                dv_dx,
                i + 1,
            )
            x = np.clip(x, 0.0, 1.0)

            # 수렴 판정
            change = float(np.abs(x - x_old).max())
            if i % 5 == 0:
                print(f"  Iter {i:3d}: C={C:.4e}  "
                      f"Avg_h={np.mean(x*self.h_max):.3f}mm  "
                      f"Max_h={np.max(x*self.h_max):.3f}mm  "
                      f"dx={change:.4f}")
            if change < tol:
                print(f"  -> 수렴 달성 (Iter {i})")
                break

        # 원본 좌표 복원 후 최종 형상 적용
        self._restore_heights()
        self.heights = x * self.h_max
        return self.heights

    def apply_final_shape(self):
        """
        최적화 결과(self.heights)를 모델 노드 좌표에 영구 적용합니다.
        solve() 완료 후 export 전에 반드시 호출하십시오.
        """
        h_filtered = self._H @ self.heights
        self._apply_heights(h_filtered)
        print(f" -> [Solver] 최종 비드 형상 적용 완료 "
              f"(Max: {np.max(h_filtered):.2f}mm, "
              f"Avg: {np.mean(h_filtered):.2f}mm)")

    def get_full_heights(self) -> np.ndarray:
        """
        시각화용 전체 노드(sorted_nids 기준) 비드 높이 배열을 반환합니다.
        """
        h_full = np.zeros(len(self.sorted_nids))
        h_filtered = self._H @ self.heights
        for i, nid in enumerate(self._design_nids):
            idx = self.nid_to_idx[nid]
            h_full[idx] = h_filtered[i]
        return h_full


# 하위 호환성을 위한 별칭
JaxTopoSolver = WHTopographySolver


if __name__ == "__main__":
    pass
