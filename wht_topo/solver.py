# -*- coding: utf-8 -*-
"""
solver.py
=========
JAX 기반 위상 최적화 엔진 (Topography Optimization Solver).

핵심 아키텍처:
    - SIMP 기반 밀도 페널티  +  Heaviside Projection
    - 공간 필터링(rmin)을 통한 최소 비드 폭 제어
    - 멀티 로드 컴플라이언스 (Bending / Twisting / Lifting)
    - MAC 기반 모드 트래킹을 통한 진동수 제약
    - MMA 업데이트 엔진

Dependencies:
    wht_modeler, wht_topo.loads, wht_topo.constraints, wht_topo.mma
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Optional, Tuple, Union

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_topo.loads import StochasticLoadManager
from wht_topo.constraints import DynamicConstraint, StressConstraint
from wht_topo.mma import MMAOptimizer


# ──────────────────────────────────────────────────────────────
#  Module-level pure function for JIT (self를 포함하지 않음)
# ──────────────────────────────────────────────────────────────
@jax.jit
def _compute_compliance_and_freq(
    rho_phys: jnp.ndarray,
    base_stiffness: jnp.ndarray,
    base_mass: jnp.ndarray,
    sens_b: jnp.ndarray,
    sens_t: jnp.ndarray,
    sens_l: jnp.ndarray,
    p: float,
) -> Tuple[Tuple[float, float, float], jnp.ndarray, jnp.ndarray]:
    """
    밀도 필드로부터 컴플라이언스와 고유 진동수를 계산하는 순수 함수.

    Parameters
    ----------
    rho_phys : (N,) 물리 밀도 (Heaviside 투영 후)
    base_stiffness : (N,) 요소별 기본 강성
    base_mass : (N,) 요소별 기본 질량
    sens_b/t/l : (N,) 벤딩/비틀림/리프팅 민감도
    p : float  SIMP 페널티 지수

    Returns
    -------
    compliances : (c_b, c_t, c_l) 각 하중 케이스별 컴플라이언스
    freqs : (N,) 주파수 배열
    modes : (10, N) 모드 형상 (Diagonal Approximation)
    """
    k_elements = (rho_phys ** p) * base_stiffness

    c_b = jnp.sum(sens_b / (k_elements + 1e-6))
    c_t = jnp.sum(sens_t / (k_elements + 1e-6))
    c_l = jnp.sum(sens_l / (k_elements + 1e-6))

    # Diagonal approximation 기반 고유치
    lambdas = k_elements / (rho_phys * base_mass + 1e-6)
    freqs = jnp.sqrt(jnp.sort(lambdas)) / (2.0 * jnp.pi)
    modes = jnp.tile(rho_phys, (10, 1))

    return (c_b, c_t, c_l), freqs, modes


class JaxTopoSolver:
    """
    SIMP + Heaviside + MMA 기반 상용급 위상 최적화 엔진.

    Parameters
    ----------
    model : WHTMeshModel
        유한요소 메시 모델.
    load_manager : StochasticLoadManager
        하중 패턴 생성기.
    constraints : list
        DynamicConstraint / StressConstraint 리스트.
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
    ):
        self.model = model
        self.load_manager = load_manager
        self.constraints = constraints or []
        self.vol_frac = vol_frac
        self.p = simp_p
        self.rmin = min_width
        self.draw_angle = draw_angle
        self.weights = weights or {"bending": 1.0, "twisting": 1.5, "lifting": 1.2}

        # 방향 벡터 정규화
        d = np.array(draw_dir, dtype=np.float64)
        self.draw_dir = d / (np.linalg.norm(d) + 1e-10)

        # 전처리 순서 중요: 면적 → 필터 (필터가 면적 사용)
        self.num_elements = len(model.elements)
        self._precompute_geometric_data()
        self._precompute_filter_matrix()
        self._precompute_load_sensitivities()

        # 초기 밀도 및 MMA 엔진
        self.density = jnp.full((self.num_elements,), vol_frac)
        self.mma = MMAOptimizer(n_vars=self.num_elements, vol_frac=vol_frac)

    # ─────────────── 전처리 ───────────────

    def _precompute_geometric_data(self):
        """요소별 면적(Area), 기본 강성/질량 벡터를 사전 계산."""
        print(" -> [Solver] 기하학적 데이터 사전 계산 중...")
        elem_ids = sorted(self.model.elements.keys())
        areas = []
        for eid in elem_ids:
            pts = np.array([
                [self.model.nodes[nid].x, self.model.nodes[nid].y, self.model.nodes[nid].z]
                for nid in self.model.elements[eid].node_ids
            ])
            if len(pts) == 4:
                area = 0.5 * np.linalg.norm(np.cross(pts[2] - pts[0], pts[3] - pts[1]))
            else:
                area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
            areas.append(area)

        self.elem_areas = jnp.array(areas)
        self.total_area = jnp.sum(self.elem_areas)
        self.vol_grad_weights = self.elem_areas / self.total_area

        # 기본 강성/질량 (면적에 비례)
        self.base_stiffness = self.elem_areas
        self.base_mass = self.elem_areas

    def _precompute_filter_matrix(self):
        """공간 필터 행렬 H를 조립 (면적 가중)."""
        print(f" -> [Solver] 공간 필터 행렬 ({self.rmin}mm) 조립 중...")
        elem_ids = sorted(self.model.elements.keys())
        n = len(elem_ids)
        centers = np.array([
            np.mean([
                [self.model.nodes[nid].x, self.model.nodes[nid].y, self.model.nodes[nid].z]
                for nid in self.model.elements[eid].node_ids
            ], axis=0)
            for eid in elem_ids
        ])

        W = np.zeros((n, n))
        areas_np = np.array(self.elem_areas)
        for i in range(n):
            dists = np.linalg.norm(centers - centers[i], axis=1)
            mask = dists < self.rmin
            w = (self.rmin - dists[mask]) * areas_np[mask]
            W[i, mask] = w / (np.sum(w) + 1e-10)
        self.H = jnp.array(W)

    def _precompute_load_sensitivities(self):
        """각 하중 패턴에 대한 요소별 민감도를 사전 계산 (시드 고정)."""
        print(" -> [Solver] 하중 민감도 계산 중...")
        np.random.seed(42)  # 결정적 시드
        self.sens_bending = self._pattern_to_sensitivity(
            self.load_manager._generate_bending_pattern()
        )
        self.sens_twisting = self._pattern_to_sensitivity(
            self.load_manager._generate_twisting_pattern()
        )
        self.sens_lifting = self._pattern_to_sensitivity(
            self.load_manager._generate_lifting_pattern()
        )

    def _pattern_to_sensitivity(self, load_pattern: np.ndarray) -> jnp.ndarray:
        """
        하중 패턴 → 요소별 민감도 변환.

        Parameters
        ----------
        load_pattern : (num_nodes, 3) 노드별 하중 벡터

        Returns
        -------
        sens : (num_elements,) 요소별 민감도
        """
        load_indices = np.where(np.abs(load_pattern[:, 2]) > 1e-3)[0]
        load_coords = self.load_manager.coords[load_indices]
        sens = np.zeros(self.num_elements)
        elem_ids = sorted(self.model.elements.keys())
        for i, eid in enumerate(elem_ids):
            ec = np.mean([
                [self.model.nodes[nid].x, self.model.nodes[nid].y, self.model.nodes[nid].z]
                for nid in self.model.elements[eid].node_ids
            ], axis=0)
            if len(load_coords) > 0:
                dist = np.min(np.linalg.norm(load_coords - ec, axis=1))
                sens[i] = 1.0 / (dist + 50.0)
        return jnp.array(sens)

    # ─────────────── Heaviside Projection ───────────────

    @staticmethod
    def _heaviside(x: jnp.ndarray, beta: float) -> jnp.ndarray:
        """Heaviside 투영 (eta=0.5 고정)."""
        eta = 0.5
        num = jnp.tanh(beta * eta) + jnp.tanh(beta * (x - eta))
        den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
        return num / den

    # ─────────────── 목적 함수 ───────────────

    def objective_fn(self, rho: jnp.ndarray, beta: float, ref_modes: List[jnp.ndarray]) -> float:
        """
        최적화 목적 함수 (Pure Function).

        Parameters
        ----------
        rho : (N,) 설계 변수 (밀도)
        beta : float  Heaviside 이산화 강도
        ref_modes : list  MAC 트래킹용 기준 모드 형상들
        """
        rho_tilde = jnp.dot(self.H, rho)
        rho_phys = self._heaviside(rho_tilde, beta)

        compliances, freqs, modes = _compute_compliance_and_freq(
            rho_phys,
            self.base_stiffness,
            self.base_mass,
            self.sens_bending,
            self.sens_twisting,
            self.sens_lifting,
            self.p,
        )
        c_b, c_t, c_l = compliances

        # 가중 멀티 로드 컴플라이언스
        total_compliance = (
            self.weights["bending"] * c_b
            + self.weights["twisting"] * c_t
            + self.weights["lifting"] * c_l
        )

        # 동적 제약 페널티
        dynamic_penalty = 0.0
        if self.constraints:
            for i, con in enumerate(self.constraints):
                if isinstance(con, DynamicConstraint) and i < len(ref_modes):
                    dynamic_penalty += con.get_penalty(freqs, modes, ref_modes[i])

        return total_compliance + dynamic_penalty

    # ─────────────── 메인 루프 ───────────────

    def solve(self, max_iter: int = 50, tol: float = 1e-3) -> jnp.ndarray:
        """
        최적화 실행.

        Parameters
        ----------
        max_iter : int  최대 반복 수
        tol : float  수렴 판정 기준 (밀도 변화 최대값)

        Returns
        -------
        density : (N,) 최종 밀도 필드
        """
        print(f"[wht_topo] MMA 기반 멀티 로드 최적화 시작 (max_iter={max_iter})")
        rho = self.density
        beta = 1.0

        # 초기 기준 모드
        ref_modes: List[jnp.ndarray] = []
        if self.constraints:
            _, _, initial_modes = _compute_compliance_and_freq(
                rho, self.base_stiffness, self.base_mass,
                self.sens_bending, self.sens_twisting, self.sens_lifting, self.p,
            )
            for con in self.constraints:
                if isinstance(con, DynamicConstraint):
                    ref_modes.append(initial_modes[con.tracker.target_mode_idx])

        grad_fn = jax.grad(self.objective_fn)

        for i in range(max_iter):
            old_rho = rho

            # Heaviside continuation
            if i > 0 and i % 10 == 0 and beta < 8.0:
                beta *= 2.0
                print(f"  -> Beta 증가: {beta:.1f}")

            # 목적 함수 & 구배
            f0val = self.objective_fn(rho, beta, ref_modes)
            df0dx = grad_fn(rho, beta, ref_modes)

            # 면적 가중 체적 제약
            current_vol = jnp.sum(rho * self.vol_grad_weights)
            vol_val = current_vol - self.vol_frac

            # MMA 업데이트
            rho = self.mma.update(
                rho, f0val, df0dx,
                jnp.array([vol_val]),
                self.vol_grad_weights,
                i + 1,
            )

            # MAC 기반 모드 트래킹 업데이트
            if self.constraints and ref_modes:
                rho_phys = self._heaviside(jnp.dot(self.H, rho), beta)
                _, _, current_modes = _compute_compliance_and_freq(
                    rho_phys, self.base_stiffness, self.base_mass,
                    self.sens_bending, self.sens_twisting, self.sens_lifting, self.p,
                )
                new_ref = []
                for j, con in enumerate(self.constraints):
                    if isinstance(con, DynamicConstraint) and j < len(ref_modes):
                        best = con.tracker.find_best_match(current_modes, ref_modes[j])
                        new_ref.append(current_modes[best])
                ref_modes = new_ref

            # 진행 상황 출력
            change = jnp.abs(rho - old_rho).max()
            if i % 5 == 0:
                print(f"  Iter {i:3d}: Obj={float(f0val):.4f}  Vol={float(current_vol):.4f}  dRho={float(change):.4f}  Beta={beta:.1f}")
            if change < tol and beta >= 8.0:
                print(f"  -> 수렴 달성 (Iter {i})")
                break

        self.density = rho
        return self.density


if __name__ == "__main__":
    pass
