# -*- coding: utf-8 -*-
"""
wht_esl_topo.py
===============
ESL (Equivalent Static Load) 기반 동하중 토포그라피 최적화.

Park (2006) 방법론:
    동해석 → ESL 추출 → 정적 토포 최적화 → 형상 업데이트 → 반복

수렴 판정:
    ||f_ESL(k+1) - f_ESL(k)|| / ||f_ESL(k)|| < tol

사용 흐름:
    optimizer = ESLTopoOptimizer(model, load_manager, dynamic_solver, ...)
    result    = optimizer.run(load_groups, dt=1e-4, T=0.1)
"""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_dynamic_solver import WHTDynamicSolver
from wht_solver.wht_dynamic_common import DampingSpec, DynamicLoadGroup, DynamicResult
from wht_solver.load_cases import WHTLoadCase
from wht_topo.loads import StochasticLoadManager
from wht_topo.solver import WHTopographySolver


# ─────────────────────────────────────────────────────────────────────────────
# 결과 컨테이너
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ESLTopoResult:
    """
    ESL 반복 최적화 결과.

    Attributes
    ----------
    heights         : 최종 비드 높이 배열 (n_design,) [mm]
    history         : 외부 반복별 이력 리스트
    converged       : 수렴 여부
    n_outer         : 실행된 외부 반복 수
    last_dynamic    : 마지막 동해석 결과 (DynamicResult)
    """
    heights:       np.ndarray
    history:       List[dict]  = field(default_factory=list)
    converged:     bool        = False
    n_outer:       int         = 0
    last_dynamic:  Optional[DynamicResult] = None


# ─────────────────────────────────────────────────────────────────────────────
# ESL 수렴 판정
# ─────────────────────────────────────────────────────────────────────────────

def _esl_convergence(
    prev_cases: Optional[List[WHTLoadCase]],
    curr_cases: List[WHTLoadCase],
    sorted_nids: List[int],
) -> float:
    """
    ESL 하중 벡터 변화율 계산.

    ||f_curr - f_prev|| / (||f_prev|| + 1e-12)

    Returns
    -------
    float : 상대 변화율 (0 = 변화 없음)
    """
    if prev_cases is None:
        return float("inf")

    def _to_vector(cases: List[WHTLoadCase]) -> np.ndarray:
        nid_idx = {nid: i for i, nid in enumerate(sorted_nids)}
        ndof    = len(sorted_nids) * 6
        f_vec   = np.zeros(ndof)
        for lc in cases:
            for fe in lc.forces:
                i = nid_idx.get(fe.node_id)
                if i is not None:
                    f_vec[i*6: i*6+6] += np.array(fe.load_vector)
        return f_vec

    f_prev = _to_vector(prev_cases)
    f_curr = _to_vector(curr_cases)
    return float(np.linalg.norm(f_curr - f_prev) / (np.linalg.norm(f_prev) + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# ESL 시점 자동 선택
# ─────────────────────────────────────────────────────────────────────────────

def select_esl_times(
    dynamic_result: DynamicResult,
    n_esl: int,
    include_peak: bool = True,
    method: str = "uniform",
) -> List[int]:
    """
    ESL 추출 시점 인덱스 선택.

    Parameters
    ----------
    n_esl        : 목표 ESL 시점 수
    include_peak : 최대 변위 시점 강제 포함
    method       : "uniform" — 균등 간격
                   "peaks"   — 국소 피크 자동 탐지

    Returns
    -------
    정렬된 인덱스 리스트
    """
    n_saved = dynamic_result.n_save
    peak_idx = dynamic_result.peak_time_index()

    if method == "peaks":
        # 각 저장 시점의 최대 변위 norm
        norms = np.max(np.abs(dynamic_result.u.reshape(n_saved, -1)), axis=1)
        # 국소 극대 탐지
        from scipy.signal import argrelmax
        local_max_idx = argrelmax(norms, order=max(1, n_saved // (n_esl * 2)))[0].tolist()
        # 크기 기준 상위 n_esl 선택
        local_max_idx = sorted(local_max_idx, key=lambda i: -norms[i])[:n_esl]
        idx_list = sorted(local_max_idx)
    else:
        idx_list = np.linspace(0, n_saved - 1, n_esl, dtype=int).tolist()

    if include_peak and peak_idx not in idx_list:
        idx_list.append(peak_idx)

    return sorted(set(idx_list))


# ─────────────────────────────────────────────────────────────────────────────
# 메인 ESL 반복 루프
# ─────────────────────────────────────────────────────────────────────────────

class ESLTopoOptimizer:
    """
    ESL 기반 동하중 토포그라피 최적화기.

    외부 루프: 동해석 → ESL 추출 → 형상 업데이트 → 수렴 판정
    내부 루프: WHTopographySolver.solve() (정적 토포 최적화)

    Parameters
    ----------
    model           : WHTMeshModel (공유 참조 — 내부 반복마다 좌표 갱신됨)
    load_manager    : StochasticLoadManager (WHTopographySolver 초기화용)
    dynamic_solver  : WHTDynamicSolver (동해석 수행)
    topo_kwargs     : WHTopographySolver 생성자 키워드 인수 dict
    n_outer         : 최대 외부 반복 수 (기본 5)
    tol_esl         : ESL 수렴 허용 오차 (기본 0.05 = 5%)
    n_esl           : ESL 시점 수 (기본 5)
    n_inner         : 내부 토포 최적화 반복 수 (기본 15)
    dynamic_method  : "modal" 또는 "direct" (기본 "modal")
    esl_time_method : "uniform" 또는 "peaks"
    """

    def __init__(
        self,
        model: WHTMeshModel,
        load_manager: StochasticLoadManager,
        dynamic_solver: WHTDynamicSolver,
        topo_kwargs: Optional[dict] = None,
        n_outer: int = 5,
        tol_esl: float = 0.05,
        n_esl: int = 5,
        n_inner: int = 15,
        dynamic_method: str = "modal",
        esl_time_method: str = "uniform",
    ):
        self.model           = model
        self.load_manager    = load_manager
        self.dyn_solver      = dynamic_solver
        self.topo_kwargs     = topo_kwargs or {}
        self.n_outer         = n_outer
        self.tol_esl         = tol_esl
        self.n_esl           = n_esl
        self.n_inner         = n_inner
        self.dynamic_method  = dynamic_method
        self.esl_time_method = esl_time_method

    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        load_groups: List[DynamicLoadGroup],
        dt: float,
        T: float,
        damping: Optional[DampingSpec] = None,
        n_modes: int = 20,
        n_save: int = 200,
        callback=None,
        stop_event=None,
    ) -> ESLTopoResult:
        """
        ESL 반복 최적화 실행.

        Parameters
        ----------
        load_groups : 동해석 하중 그룹 리스트
        dt          : 동해석 시간 스텝 [s]
        T           : 동해석 총 시간 [s]
        damping     : DampingSpec (기본 zeta=0.02)
        n_modes     : 모달 동해석 모드 수
        n_save      : 동해석 저장 시점 수
        callback    : 모니터 큐 콜백 (WHTopographySolver.solve와 동일)
        stop_event  : 외부 중단 이벤트

        Returns
        -------
        ESLTopoResult
        """
        if damping is None:
            damping = DampingSpec(mode="zeta", zeta=0.02)

        t_wall_total = time.time()
        history: List[dict] = []
        prev_esl_cases: Optional[List[WHTLoadCase]] = None
        curr_heights   = None
        last_dyn       = None
        converged      = False

        sorted_nids = self.model.sorted_node_ids()

        print(f"\n{'#'*65}", flush=True)
        print(f"  ESL Topography Optimization  (max_outer={self.n_outer})", flush=True)
        print(f"  Dynamic: {self.dynamic_method}  dt={dt:.2e}s  T={T:.2e}s", flush=True)
        print(f"{'#'*65}\n", flush=True)

        for outer in range(1, self.n_outer + 1):
            if stop_event and stop_event.is_set():
                print(" -> [ESL] 외부 반복 중단 (stop_event).", flush=True)
                break

            t_outer = time.time()
            print(f"\n{'─'*65}", flush=True)
            print(f"  [ESL Outer Loop {outer}/{self.n_outer}]", flush=True)
            print(f"{'─'*65}", flush=True)

            # ── Step 1: 동해석 ────────────────────────────────────────────
            if self.dynamic_method == "modal":
                dyn_result = self.dyn_solver.solve_modal_dynamic(
                    load_groups=load_groups,
                    dt=dt, T=T,
                    n_modes=n_modes,
                    damping=damping,
                    n_save=n_save,
                )
            else:
                dyn_result = self.dyn_solver.solve_direct_dynamic(
                    load_groups=load_groups,
                    dt=dt, T=T,
                    damping=damping,
                    n_save=n_save,
                    modal_modes=getattr(self, 'modal_modes', 20),
                )
            last_dyn = dyn_result

            # ── Step 2: ESL 추출 ─────────────────────────────────────────
            esl_idx   = select_esl_times(
                dyn_result, self.n_esl,
                include_peak=True,
                method=self.esl_time_method,
            )
            esl_cases = self.dyn_solver.extract_esl(
                dyn_result,
                n_esl=self.n_esl,
                include_peak=True,
            )

            # ── Step 3: ESL 수렴 판정 ────────────────────────────────────
            esl_change = _esl_convergence(prev_esl_cases, esl_cases, sorted_nids)
            print(f"    - ESL 변화율: {esl_change:.4f}  (tol={self.tol_esl})", flush=True)

            # ── Step 4: ESL → 토포 최적화용 하중 케이스 주입 ─────────────
            # ESL 케이스에 균등 가중치 부여 (총 합 = n_esl)
            esl_weight = 1.0
            esl_with_weights: List[Tuple[WHTLoadCase, float]] = [
                (lc, esl_weight) for lc in esl_cases
            ]

            # ── Step 5: WHTopographySolver 생성 및 _load_cases 주입 ───────
            topo_solver = WHTopographySolver(
                model=self.model,
                load_manager=self.load_manager,
                **self.topo_kwargs,
            )
            # 내부 _load_cases를 ESL 케이스로 교체
            topo_solver._load_cases = esl_with_weights

            # 이전 반복 높이 이어받기 (warm start)
            if curr_heights is not None:
                n_design = topo_solver._n_design
                if len(curr_heights) == n_design:
                    topo_solver.heights = curr_heights.copy()
                    topo_solver.mma.xold1 = curr_heights / (topo_solver.h_max + 1e-12)
                    topo_solver.mma.xold2 = topo_solver.mma.xold1.copy()

            # ── Step 6: 내부 토포 최적화 ─────────────────────────────────
            print(f"    - 내부 토포 최적화 ({self.n_inner}회) 시작...", flush=True)
            curr_heights = topo_solver.solve(
                max_iter=self.n_inner,
                callback=callback,
                stop_event=stop_event,
            )

            # ── Step 7: 형상을 모델 좌표에 반영 ──────────────────────────
            #   다음 외부 반복의 동해석에서 업데이트된 형상 사용
            topo_solver.apply_final_shape(skip_filter=False)

            # ── Step 8: 이력 기록 ────────────────────────────────────────
            peak_idx = dyn_result.peak_time_index()
            rec = {
                "outer":       outer,
                "esl_change":  esl_change,
                "peak_disp":   float(np.max(np.abs(dyn_result.u[peak_idx]))),
                "peak_t":      float(dyn_result.t_saved[peak_idx]),
                "n_esl_cases": len(esl_cases),
                "elapsed":     time.time() - t_outer,
            }
            history.append(rec)
            print(
                f"    - Outer {outer} 완료 | ESL Δ={esl_change:.4f} "
                f"peak={rec['peak_disp']:.3e}mm @ t={rec['peak_t']:.4f}s "
                f"({rec['elapsed']:.1f}s)",
                flush=True,
            )

            prev_esl_cases = esl_cases

            # ── 수렴 판정 ─────────────────────────────────────────────────
            if esl_change < self.tol_esl and outer > 1:
                converged = True
                print(
                    f"\n  [ESL] 수렴 완료 (outer={outer}, Δ={esl_change:.4f} < {self.tol_esl})",
                    flush=True,
                )
                break

        elapsed_total = time.time() - t_wall_total
        print(f"\n{'#'*65}", flush=True)
        print(
            f"  ESL 최적화 종료 | converged={converged} | "
            f"outer={len(history)} | {elapsed_total:.1f}s",
            flush=True,
        )
        print(f"{'#'*65}\n", flush=True)

        return ESLTopoResult(
            heights      = curr_heights if curr_heights is not None else np.array([]),
            history      = history,
            converged    = converged,
            n_outer      = len(history),
            last_dynamic = last_dyn,
        )
