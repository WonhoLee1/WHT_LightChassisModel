# -*- coding: utf-8 -*-
"""
solver.py
=========
JAX 기반 위상 최적화 엔진 (Topography Optimization Solver) — 정밀 FEA 버전.

핵심 아키텍처:
    - WHTSolver(wht_solver)를 활용한 하중 케이스별 개별 실제 정적 FEA 수행
    - 각 하중 케이스는 물리적으로 다른 경계 조건(BC)과 하중을 가짐
    - Compliance(컴플라이언스) = u^T * F 로부터 엄밀한 요소별 민감도 계산
    - SIMP 페널티 밀도 기반 요소별 강성 스케일링
    - Heaviside Projection 기반 밀도 이산화
    - 공간 필터링(rmin)을 통한 최소 비드 폭 제어
    - MMA 업데이트 엔진

Dependencies:
    wht_modeler, wht_solver, wht_topo.loads, wht_topo.constraints, wht_topo.mma
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from scipy.sparse.linalg import spsolve

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_solver import WHTSolver
from wht_topo.loads import StochasticLoadManager
from wht_topo.constraints import DynamicConstraint, StressConstraint
from wht_topo.mma import MMAOptimizer


class JaxTopoSolver:
    """
    WHTSolver 기반 정밀 멀티 케이스 위상 최적화 엔진.

    각 하중 케이스(Bending, Twisting, Lifting)에 대해 실제 FEA를 수행하고,
    케이스별 컴플라이언스 민감도를 계산하여 비드 밀도를 최적화합니다.

    Parameters
    ----------
    model : WHTMeshModel
        유한요소 메시 모델.
    load_manager : StochasticLoadManager
        하중 패턴 및 경계 조건 생성기.
    constraints : list
        DynamicConstraint / StressConstraint 리스트 (현재는 향후 확장용).
    vol_frac : float
        목표 체적 분율 (0~1).
    simp_p : float
        SIMP 페널티 지수 (기본 3.0).
    min_width : float
        최소 비드 폭 (mm), 공간 필터 반경으로 사용.
    draw_angle : float
        제조 구배각 (도).
    draw_dir : tuple
        비드 돌출 방향 벡터.
    weights : dict
        하중 케이스별 가중치 {"bending", "twisting", "lifting"}.
    mesh_size_z : float
        플랜지 노드 탐색 허용 오차 계산에 사용되는 메시 크기 [mm].
    """

    def __init__(
        self,
        model: WHTMeshModel,
        load_manager: StochasticLoadManager,
        constraints: Optional[List[Union[DynamicConstraint, StressConstraint]]] = None,
        vol_frac: float = 0.3,
        simp_p: float = 3.0,
        min_width: float = 80.0,
        draw_angle: float = 25.0,
        draw_dir: Tuple[float, float, float] = (0, 0, 1),
        weights: Optional[Dict[str, float]] = None,
        mesh_size_z: float = 10.0,
    ):
        self.model = model
        self.load_manager = load_manager
        self.constraints = constraints or []
        self.vol_frac = vol_frac
        self.p = simp_p
        self.rmin = min_width
        self.draw_angle = draw_angle
        self.mesh_size_z = mesh_size_z
        self.weights = weights or {"bending": 1.0, "twisting": 1.5, "lifting": 1.2}

        # 방향 벡터 정규화
        d = np.array(draw_dir, dtype=np.float64)
        self.draw_dir = d / (np.linalg.norm(d) + 1e-10)

        # WHTSolver 인스턴스 (실제 FEA 수행)
        self._wht_solver = WHTSolver(model)

        # 재료 및 속성 기본값 설정 (없으면 Steel 기본값 사용)
        self._ensure_material_properties()

        # 전처리
        self.num_elements = len(model.elements)
        self.elem_ids = sorted(model.elements.keys())
        self.sorted_nids = model.sorted_node_ids()
        self.nid_to_idx = {nid: i for i, nid in enumerate(self.sorted_nids)}

        self._precompute_geometric_data()
        self._precompute_filter_matrix()

        # 하중 케이스 생성 (loads.py의 정밀 메서드 활용)
        print(" -> [Solver] 하중 케이스 생성 중 (하중별 개별 BC 적용)...")
        self._load_cases = load_manager.get_load_cases(
            mesh_size_z=mesh_size_z,
            weights=self.weights,
        )

        # JaxSSO 기반 기본 강성 행렬 조립 (밀도 페널티 계산의 기준)
        print(" -> [Solver] 기준 강성 행렬 조립 중...")
        self._K_elem_base = self._precompute_element_stiffness()

        # 초기 밀도 및 MMA 엔진
        self.density = np.full((self.num_elements,), vol_frac)
        self.mma = MMAOptimizer(n_vars=self.num_elements, vol_frac=vol_frac)

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

    def _precompute_geometric_data(self):
        """요소별 면적(Area), 기본 강성/질량 벡터를 사전 계산합니다."""
        print(" -> [Solver] 기하학적 데이터 사전 계산 중...")
        areas = []
        for eid in self.elem_ids:
            pts = np.array([
                [self.model.nodes[nid].x, self.model.nodes[nid].y, self.model.nodes[nid].z]
                for nid in self.model.elements[eid].node_ids
            ])
            if len(pts) == 4:
                area = 0.5 * np.linalg.norm(np.cross(pts[2] - pts[0], pts[3] - pts[1]))
            else:
                area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            areas.append(area)

        self.elem_areas = np.array(areas)
        self.total_area = np.sum(self.elem_areas)
        self.vol_grad_weights = self.elem_areas / self.total_area

    def _precompute_filter_matrix(self):
        """공간 필터 행렬 H를 조립 (면적 가중, 최소 비드 폭 제어)."""
        print(f" -> [Solver] 공간 필터 행렬 ({self.rmin}mm) 조립 중...")
        n = self.num_elements
        centers = np.array([
            np.mean([
                [self.model.nodes[nid].x, self.model.nodes[nid].y, self.model.nodes[nid].z]
                for nid in self.model.elements[eid].node_ids
            ], axis=0)
            for eid in self.elem_ids
        ])

        W = np.zeros((n, n))
        for i in range(n):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            mask = dists < self.rmin
            w = (self.rmin - dists[mask]) * self.elem_areas[mask]
            W[i, mask] = w / (np.sum(w) + 1e-10)
        self.H = W

    # wht_solver와 동일한 쉘 요소 타입 집합 (클래스 레벨 상수)
    _QUAD_TYPES = frozenset({'QUAD4', 'QUAD'})
    _TRIA_TYPES = frozenset({'TRIA3', 'TRIA'})
    _SHELL_TYPES = frozenset({'QUAD4', 'QUAD', 'TRIA3', 'TRIA'})

    def _precompute_element_stiffness(self) -> List[np.ndarray]:
        """
        각 요소의 기본 강성 행렬을 사전 계산합니다.
        SIMP 밀도 페널티 최적화 시 스케일링하여 사용합니다.

        요소 타입별 알고리즘:
            - QUAD4 / QUAD : MITC4+ (24×24 DOF)
            - TRIA3 / TRIA : MITC3+ Bathe et al.(2014) + Static Condensation (18×18 DOF)
            - 기타 (BEAM2 등) : 쉘 최적화 대상 외 → 영행렬 반환

        Returns
        -------
        List[np.ndarray]
            요소별 기준 강성 행렬 리스트 (elem_idx 순서)
        """
        from wht_solver.wht_quad4_element import _element_K_mitc4_plus
        from wht_solver.wht_tria3_element import _element_K_tria3

        quad_types  = self._QUAD_TYPES
        tria_types  = self._TRIA_TYPES
        shell_types = self._SHELL_TYPES

        k_elems = []
        for eid in self.elem_ids:
            elem  = self.model.elements[eid]
            etype = elem.type.upper() if hasattr(elem.type, 'upper') else str(elem.type)

            # 쉘 요소가 아닌 경우 (BEAM2 등) → 최적화 대상 외
            if etype not in shell_types:
                n_dof = len(elem.node_ids) * 6
                k_elems.append(np.zeros((n_dof, n_dof)))
                continue

            prop = self.model.properties.get(elem.pid)
            mat  = self.model.materials.get(prop.mid) if prop else None
            if prop is None or mat is None:
                n_dof = len(elem.node_ids) * 6
                k_elems.append(np.zeros((n_dof, n_dof)))
                continue

            crds = [
                np.array([self.model.nodes[nid].x,
                          self.model.nodes[nid].y,
                          self.model.nodes[nid].z])
                for nid in elem.node_ids
            ]
            try:
                if etype in quad_types:   # MITC4+
                    Ke = _element_K_mitc4_plus(
                        crds[0], crds[1], crds[2], crds[3],
                        prop.t, mat.E, mat.nu
                    )
                else:                     # MITC3+ (TRIA3 / TRIA)
                    Ke = _element_K_tria3(
                        crds[0], crds[1], crds[2],
                        prop.t, mat.E, mat.nu
                    )
            except Exception as exc:
                print(f"    [경고] 요소 {eid}({etype}) 강성 계산 실패: {exc}")
                n_dof = len(elem.node_ids) * 6
                Ke = np.zeros((n_dof, n_dof))
            k_elems.append(Ke)

        # 요소 타입 통계 출력
        n_quad = sum(1 for eid in self.elem_ids
                     if self.model.elements[eid].type.upper() in quad_types)
        n_tria = sum(1 for eid in self.elem_ids
                     if self.model.elements[eid].type.upper() in tria_types)
        n_other = self.num_elements - n_quad - n_tria
        print(f"    [조립] MITC4+(QUAD): {n_quad}개, "
              f"MITC3+(TRIA): {n_tria}개, 기타(skip): {n_other}개")
        return k_elems

    # ─────────────── Heaviside Projection ───────────────

    @staticmethod
    def _heaviside(x: np.ndarray, beta: float) -> np.ndarray:
        """Heaviside 투영 (eta=0.5 고정)."""
        eta = 0.5
        num = np.tanh(beta * eta) + np.tanh(beta * (x - eta))
        den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
        return num / den

    # ─────────────── 정밀 FEA 기반 컴플라이언스 계산 ───────────────

    def _assemble_penalized_K(self, rho_phys: np.ndarray):
        """
        SIMP 밀도 페널티를 적용한 전체 강성 행렬을 조립합니다.

        C_e(rho_e) = E_min + rho_e^p * (E_0 - E_min)
        여기서 E_min = 1e-9 * E_0 (수치적 안정성 확보)

        Parameters
        ----------
        rho_phys : (N_elem,) 물리 밀도 배열

        Returns
        -------
        K_penalized : sparse CSR matrix, 전체 DOF 크기
        """
        from scipy.sparse import csr_matrix

        ndof = len(self.sorted_nids) * 6
        rows, cols, vals = [], [], []

        for i, (eid, Ke) in enumerate(zip(self.elem_ids, self._K_elem_base)):
            rho_e = float(rho_phys[i])
            # SIMP 페널티: Void 영역에서도 최소 강성 유지 (수치적 안정성)
            E_scale = 1e-9 + (rho_e ** self.p) * (1.0 - 1e-9)

            elem = self.model.elements[eid]
            node_global_dofs = [
                self.nid_to_idx[nid] * 6 + d
                for nid in elem.node_ids for d in range(6)
            ]

            Ke_scaled = E_scale * Ke
            n_dof_e = len(node_global_dofs)
            for li in range(n_dof_e):
                for lj in range(n_dof_e):
                    rows.append(node_global_dofs[li])
                    cols.append(node_global_dofs[lj])
                    vals.append(Ke_scaled[li, lj])

        K = csr_matrix((vals, (rows, cols)), shape=(ndof, ndof))
        return K

    def _solve_load_case_compliance(
        self,
        rho_phys: np.ndarray,
        load_case,
        weight: float,
    ) -> Tuple[float, np.ndarray]:
        """
        하나의 하중 케이스에 대해 정밀 FEA를 수행하여
        컴플라이언스(C = u^T K u = F^T u)와 요소별 민감도를 계산합니다.

        Parameters
        ----------
        rho_phys : (N_elem,) 물리 밀도
        load_case : WHTLoadCase
        weight : float

        Returns
        -------
        compliance : float, 이 케이스의 가중 컴플라이언스
        sens : (N_elem,) 요소별 ∂C/∂ρ_e
        """
        from scipy.sparse import csr_matrix, diags, vstack, hstack, coo_matrix

        ndof = len(self.sorted_nids) * 6

        # 1. 페널티 강성 행렬 조립
        K = self._assemble_penalized_K(rho_phys)

        # 2. 하중 벡터 구성
        f = np.zeros(ndof)
        for force in load_case.forces:
            idx = self.nid_to_idx.get(force.node_id)
            if idx is None:
                continue
            for d, fval in enumerate(force.load_vector):
                if abs(fval) > 1e-12:
                    f[idx * 6 + d] += fval

        # 3. 하중 케이스별 BC 적용 (Lagrange Multiplier 방식)
        #    SPC DOF 수집
        bc_dof_set = {}  # {global_dof: value}
        for bc in load_case.bcs:
            idx = self.nid_to_idx.get(bc.node_id)
            if idx is None:
                continue
            for d in bc.dofs:
                gid = idx * 6 + d
                bc_dof_set[gid] = bc.value

        bc_dofs = sorted(bc_dof_set.keys())
        n_bc = len(bc_dofs)

        if n_bc == 0:
            # BC가 없으면 수치 안정성을 위해 첫 6 DOF 고정
            bc_dofs = list(range(6))
            n_bc = 6
            bc_values = np.zeros(n_bc)
        else:
            bc_values = np.array([bc_dof_set[gid] for gid in bc_dofs])

        # Lagrange Multiplier 시스템 구성: [K C^T; C 0] {u; λ} = {f; v}
        spc_rows = list(range(n_bc))
        spc_cols = bc_dofs
        spc_vals = [1.0] * n_bc
        C = coo_matrix(
            (spc_vals, (spc_rows, spc_cols)),
            shape=(n_bc, ndof)
        ).tocsr()

        K_aug = vstack([
            hstack([K, C.T]),
            hstack([C, csr_matrix((n_bc, n_bc))]),
        ]).tocsr()
        f_aug = np.concatenate([f, bc_values])

        # 4. 선형 시스템 풀기
        try:
            u_aug = spsolve(K_aug.tocsc(), f_aug)
        except Exception as e:
            print(f"    [경고] {load_case.name} 하중 케이스 해석 실패: {e}")
            return 0.0, np.zeros(self.num_elements)

        u = u_aug[:ndof]

        # 5. 컴플라이언스 계산: C = u^T f (외력에 대한 일)
        compliance = float(np.dot(f, u))

        # 6. 요소별 민감도 계산: ∂C/∂ρ_e = -p * ρ_e^(p-1) * u_e^T * K_e0 * u_e
        sens = np.zeros(self.num_elements)
        for i, (eid, Ke) in enumerate(zip(self.elem_ids, self._K_elem_base)):
            rho_e = float(rho_phys[i])
            elem = self.model.elements[eid]
            node_global_dofs = [
                self.nid_to_idx[nid] * 6 + d
                for nid in elem.node_ids for d in range(6)
            ]
            u_e = u[node_global_dofs]
            # ∂C/∂ρ_e = -p * ρ_e^(p-1) * u_e^T K_e0 u_e
            # (부호 음수: 컴플라이언스 최소화 = 강성 최대화)
            dC_drho = -self.p * (rho_e ** (self.p - 1)) * float(u_e @ Ke @ u_e)
            sens[i] = dC_drho

        return weight * compliance, weight * sens

    def compute_objective_and_gradient(
        self,
        rho: np.ndarray,
        beta: float,
    ) -> Tuple[float, np.ndarray]:
        """
        모든 하중 케이스에 대한 가중 합산 목적 함수와 구배를 계산합니다.

        Parameters
        ----------
        rho : (N_elem,) 설계 밀도 변수
        beta : float Heaviside 이산화 강도

        Returns
        -------
        total_obj : float 가중 합산 컴플라이언스
        total_grad : (N_elem,) 가중 합산 민감도 (필터 적용 후)
        """
        # Heaviside 투영 및 공간 필터 적용
        rho_tilde = self.H @ rho
        rho_phys = self._heaviside(rho_tilde, beta)

        total_obj = 0.0
        total_sens = np.zeros(self.num_elements)

        for load_case, weight in self._load_cases:
            c, dc = self._solve_load_case_compliance(rho_phys, load_case, weight)
            total_obj += c
            total_sens += dc

        # 필터 역전파 (체인 룰: ∂C/∂ρ = H^T ∂C/∂ρ_tilde)
        # Heaviside 구배
        eta = 0.5
        d_heaviside = (beta * (1.0 - np.tanh(beta * (rho_tilde - eta))**2)
                       / (np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))))
        total_sens_tilde = total_sens * d_heaviside
        total_grad = self.H.T @ total_sens_tilde

        return total_obj, total_grad

    # ─────────────── 메인 루프 ───────────────

    def solve(self, max_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
        """
        MMA 기반 멀티 케이스 정밀 위상 최적화를 실행합니다.

        Parameters
        ----------
        max_iter : int 최대 반복 수
        tol : float 수렴 판정 기준 (밀도 변화 최대값)

        Returns
        -------
        density : (N_elem,) 최종 밀도 필드
        """
        print(f"\n[wht_topo] 정밀 멀티 케이스 FEA 최적화 시작 (max_iter={max_iter})")
        print(f"   - 하중 케이스 수: {len(self._load_cases)}")
        for lc, w in self._load_cases:
            print(f"     * {lc.name}: weight={w:.2f}, BC={len(lc.bcs)}개, F={len(lc.forces)}개")

        rho = self.density.copy()
        beta = 1.0

        for i in range(max_iter):
            old_rho = rho.copy()

            # Heaviside continuation
            if i > 0 and i % 10 == 0 and beta < 8.0:
                beta = min(beta * 2.0, 8.0)
                print(f"  -> Beta 증가: {beta:.1f}")

            # 목적 함수 & 구배
            f0val, df0dx = self.compute_objective_and_gradient(rho, beta)

            # 면적 가중 체적 제약
            rho_tilde = self.H @ rho
            rho_phys = self._heaviside(rho_tilde, beta)
            current_vol = float(np.sum(rho_phys * self.vol_grad_weights))
            vol_val = current_vol - self.vol_frac

            # 체적 제약 민감도
            eta = 0.5
            d_hv = (beta * (1.0 - np.tanh(beta * (rho_tilde - eta))**2)
                    / (np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))))
            dv_drho = self.H.T @ (self.vol_grad_weights * d_hv)

            # MMA 업데이트
            rho = self.mma.update(
                rho, f0val, df0dx,
                np.array([vol_val]),
                dv_drho,
                i + 1,
            )

            # 수렴 판정
            change = np.abs(rho - old_rho).max()
            if i % 5 == 0:
                print(f"  Iter {i:3d}: Obj={f0val:.4e}  Vol={current_vol:.4f}"
                      f"  dRho={change:.4f}  Beta={beta:.1f}")
            if change < tol and beta >= 8.0:
                print(f"  -> 수렴 달성 (Iter {i})")
                break

        self.density = rho
        return self.density


if __name__ == "__main__":
    pass
