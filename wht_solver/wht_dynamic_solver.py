# -*- coding: utf-8 -*-
"""
wht_dynamic_solver.py
=====================
WHT Dynamic Analysis — Implicit Newmark-β 동적 해석기

WHTSolver를 상속하여 기존 K/M 조립, BC 처리, 결과 포맷을 그대로 사용.

지원 해석:
  solve_modal_dynamic() — 모달 중첩법 (Modal Superposition)
  solve_direct_dynamic() — 직접 Newmark-β (Direct Integration)
  extract_esl()          — ESL 추출 (위상 최적화 연동)

경계조건 처리:
  정적/모달 해석의 Lagrange 증대계 대신 자유 DOF 직접 소거를 사용.
  (K_free = K[unknown_id, :][:, unknown_id] — solve_modal과 동일 방식)
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from .wht_solver import WHTSolver
from .load_cases import WHTLoadCase, WHTForceEntry
from .wht_dynamic_common import (
    DampingSpec,
    DynamicLoadGroup,
    DynamicResult,
    assemble_rayleigh_C,
    damping_from_zeta,
    newmark_coeffs,
)


class WHTDynamicSolver(WHTSolver):
    """
    WHT 암시적 동적 해석기.

    WHTSolver의 모든 기능(정적/모달 해석, K/M 조립, BC)을 상속하며
    추가로 동해석 두 가지 방법과 ESL 추출을 제공한다.

    Parameters
    ----------
    model          : WHTMeshModel
    stiffness_scale: RBE2 빔 강성 배율 (기본 1e3, WHTSolver 동일)

    Examples
    --------
    >>> from wht_solver.wht_dynamic_solver import WHTDynamicSolver
    >>> from wht_solver.wht_dynamic_common import DynamicLoadGroup, DampingSpec
    >>>
    >>> solver = WHTDynamicSolver(model)
    >>> loads = [
    ...     DynamicLoadGroup(
    ...         node_ids=[101, 102],
    ...         dof=2,
    ...         force_magnitude=5000.0,
    ...         time_func="half_sine",
    ...         t_pulse=0.02,
    ...     )
    ... ]
    >>> result = solver.solve_modal_dynamic(loads, dt=1e-4, T=0.1, n_modes=20)
    >>> print(result.summary())
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — 동해석
    # ─────────────────────────────────────────────────────────────────────────

    def solve_modal_dynamic(
        self,
        load_groups: List[DynamicLoadGroup],
        dt: float,
        T: float,
        n_modes: int = 20,
        damping: Optional[DampingSpec] = None,
        n_save: int = 100,
        newmark_beta: float = 0.25,
        newmark_gamma: float = 0.5,
    ) -> DynamicResult:
        """
        모달 중첩법 암시적 동해석.

        고유값 해석 결과(Φ, ω)로 운동방정식을 모달 좌표에서 비결합 SDOF로
        변환한 뒤 각 모드를 독립적으로 Newmark-β 적분. 물리 좌표로 복원.

        Parameters
        ----------
        load_groups  : 노드 그룹별 시간 하중 리스트
        dt           : 시간 스텝 [s]
        T            : 총 해석 시간 [s]
        n_modes      : 포함할 모드 수
        damping      : DampingSpec (기본: zeta=0.02)
        n_save       : 저장할 시점 수
        newmark_beta : Newmark β 파라미터 (0.25 = CAA, 무조건 안정)
        newmark_gamma: Newmark γ 파라미터 (0.5 = 수치 감쇠 없음)

        Returns
        -------
        DynamicResult
        """
        if damping is None:
            damping = DampingSpec(mode="zeta", zeta=0.02)

        print(f"\n{'='*60}", flush=True)
        print(f" [Modal Dynamic] dt={dt:.2e}s  T={T:.2e}s  modes={n_modes}", flush=True)
        print(f"{'='*60}", flush=True)
        t_wall = time.time()

        # ── 1. 고유값 해석 ──────────────────────────────────────────────────
        modal_result = self.solve_modal(
            num_modes=n_modes,
            exclude_rigid_body="cutoff:0.5",
        )
        freqs   = modal_result.frequencies           # (n_r,) Hz
        omegas  = 2.0 * np.pi * freqs                # rad/s
        n_r     = len(freqs)
        print(f"    - {n_r} structural modes (f1={freqs[0]:.2f} Hz ~ f{n_r}={freqs[-1]:.2f} Hz)", flush=True)

        # ── 2. K, M 조립 (자유 DOF) ─────────────────────────────────────────
        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model()
        ndof    = jm.ndof
        n_nodes = len(sorted_nids)
        K       = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=True)
        M_diag  = self._assemble_lumped_mass(jm, ndof, sorted_nids, nid_to_idx)
        free_id = np.array(jm.unknown_id, dtype=np.int64)
        M_free  = M_diag[free_id]                    # (ndof_free,)

        # ── 3. 모드 행렬 구성 (자유 DOF 기준) ──────────────────────────────
        # mode_shapes: (n_r, n_nodes, 6) → (n_r, ndof_total)
        Phi_full = modal_result.mode_shapes.reshape(n_r, n_nodes * 6)
        Phi      = Phi_full[:, free_id]              # (n_r, ndof_free)

        # ── 4. 모달 물리량 산출 ─────────────────────────────────────────────
        # 질량 정규화 여부 무관하게 명시적으로 계산
        m_r = np.array([Phi[r] @ (M_free * Phi[r]) for r in range(n_r)])  # (n_r,)
        k_r = omegas ** 2 * m_r                                            # (n_r,)

        # 모드별 감쇠비 ζ_r
        zeta_r = self._modal_zeta(damping, omegas, M_free, K, free_id)    # (n_r,)
        c_r    = 2.0 * zeta_r * omegas * m_r                              # (n_r,)

        # ── 5. Newmark 계수 및 유효 강성 (SDOF 스칼라) ─────────────────────
        nc      = newmark_coeffs(newmark_beta, newmark_gamma, dt)
        k_eff_r = k_r + nc["a0"] * m_r + nc["a1"] * c_r                  # (n_r,)

        # ── 6. 시간 적분 루프 ───────────────────────────────────────────────
        n_steps    = int(np.ceil(T / dt))
        save_every = max(1, n_steps // n_save)

        q   = np.zeros(n_r)
        qd  = np.zeros(n_r)
        qdd = np.zeros(n_r)

        t_saved, u_saved, v_saved, a_saved = [], [], [], []

        for step in range(n_steps + 1):
            t_cur  = step * dt
            f_full = self._build_load_vector(t_cur, load_groups, ndof, nid_to_idx)
            f_free = f_full[free_id]
            p_r    = Phi @ f_free            # (n_r,) 모달 하중

            if step == 0:
                # 초기 가속도: M q̈ = p - C q̇ - K q (q=qd=0)
                qdd = p_r / m_r
            else:
                p_eff = (p_r
                         + m_r * (nc["a0"]*q  + nc["a2"]*qd  + nc["a3"]*qdd)
                         + c_r * (nc["a1"]*q  + nc["a4"]*qd  + nc["a5"]*qdd))
                q_new   = p_eff / k_eff_r
                qdd_new = nc["a0"]*(q_new - q) - nc["a2"]*qd - nc["a3"]*qdd
                qd_new  = qd + dt * ((1 - newmark_gamma)*qdd + newmark_gamma*qdd_new)
                q, qd, qdd = q_new, qd_new, qdd_new

            # 저장 판단
            if step % save_every == 0 or step == n_steps:
                # 물리 좌표 복원
                u_free = Phi.T @ q    # (ndof_free,)
                v_free = Phi.T @ qd
                a_free = Phi.T @ qdd

                u_vec = np.zeros(ndof); u_vec[free_id] = u_free
                v_vec = np.zeros(ndof); v_vec[free_id] = v_free
                a_vec = np.zeros(ndof); a_vec[free_id] = a_free

                t_saved.append(t_cur)
                u_saved.append(u_vec.reshape(n_nodes, 6))
                v_saved.append(v_vec.reshape(n_nodes, 6))
                a_saved.append(a_vec.reshape(n_nodes, 6))

        result = DynamicResult(np.array(t_saved), sorted_nids)
        result.u           = np.array(u_saved)
        result.v           = np.array(v_saved)
        result.a           = np.array(a_saved)
        result.solver_type = "modal"
        result.solver_info = dict(
            n_modes=n_r, dt=dt, T=T,
            n_steps=n_steps, n_saved=len(t_saved),
            elapsed=time.time() - t_wall,
        )
        print(f" -> Done in {time.time()-t_wall:.1f}s | {result.summary()}", flush=True)
        return result

    # ─────────────────────────────────────────────────────────────────────────

    def solve_direct_dynamic(
        self,
        load_groups: List[DynamicLoadGroup],
        dt: float,
        T: float,
        damping: Optional[DampingSpec] = None,
        n_save: int = 100,
        newmark_beta: float = 0.25,
        newmark_gamma: float = 0.5,
    ) -> DynamicResult:
        """
        직접 Newmark-β 암시적 동해석 (전체 DOF).

        K_eff = K + a₀M + a₁C 를 1회 조립 후 LU 분해.
        각 스텝은 forward/back substitution만 수행.
        경계조건: 자유 DOF 직접 소거 (solve_modal과 동일, Lagrange 불필요).

        Parameters
        ----------
        load_groups  : 노드 그룹별 시간 하중 리스트
        dt           : 시간 스텝 [s]
        T            : 총 해석 시간 [s]
        damping      : DampingSpec (기본: zeta=0.02)
        n_save       : 저장할 시점 수
        newmark_beta : Newmark β (0.25 = CAA)
        newmark_gamma: Newmark γ (0.5 = 수치 감쇠 없음)

        Returns
        -------
        DynamicResult
        """
        from scipy.sparse import diags
        from scipy.sparse.linalg import splu

        if damping is None:
            damping = DampingSpec(mode="zeta", zeta=0.02)

        print(f"\n{'='*60}", flush=True)
        print(f" [Direct Dynamic] dt={dt:.2e}s  T={T:.2e}s", flush=True)
        print(f"{'='*60}", flush=True)
        t_wall = time.time()

        # ── 1. K, M 조립 ────────────────────────────────────────────────────
        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model()
        ndof    = jm.ndof
        n_nodes = len(sorted_nids)
        # Full K (shell + RBE2 stiff beams via JaxSSO)
        K       = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=True)
        # Shell-only K: Rayleigh 감쇠 기준 주파수 계산용 (RBE2 beam 제외)
        K_struct = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx,
                                          stabilize=True, include_beams=False)

        M_diag  = self._assemble_lumped_mass(jm, ndof, sorted_nids, nid_to_idx)
        
        # [WHT] Patch zero mass to avoid division-by-zero (NaN) during integration
        m_max = np.max(M_diag)
        if m_max > 0:
            M_diag = np.maximum(M_diag, max(m_max * 1e-8, 1e-10))
        else:
            M_diag = np.maximum(M_diag, 1e-10)

        # ── 2. DOF 분류: SPC(고정) / SPCD(시변 처방) / FREE ────────────────
        unknown_set = set(jm.unknown_id)

        # SPCD DOF → DynamicLoadGroup 매핑
        spcd_map: dict = {}   # global_dof -> DynamicLoadGroup
        for lg in load_groups:
            if lg.load_type.upper() != "SPCD":
                continue
            for nid in lg.node_ids:
                idx = nid_to_idx.get(nid)
                if idx is None:
                    continue
                gdof = idx * 6 + lg.dof
                if gdof in unknown_set:
                    spcd_map[gdof] = lg

        # RBE2 슬레이브 자동 확장은 코너 변형을 왜곡(수평 유지 강제)하므로 제거함.
        # SPCD는 오직 마스터 노드에만 적용되고, RBE2 beam 강성을 통해 슬레이브로 전달됨.
        spcd_id = np.array(sorted(spcd_map.keys()), dtype=np.int64)
        spcd_set = set(spcd_id.tolist())
        free_id  = np.array(
            [d for d in jm.unknown_id if d not in spcd_set], dtype=np.int64
        )
        n_free, n_spcd = len(free_id), len(spcd_id)

        if n_spcd:
            n_master = sum(1 for lg in load_groups if lg.load_type.upper() == "SPCD"
                           for _ in lg.node_ids)
            print(f"    - SPCD DOF: {n_spcd}개 (master {n_master})", flush=True)

        # ── 3. 서브행렬 추출 ────────────────────────────────────────────────
        K_ff = K[free_id, :][:, free_id].tocsr()
        M_f  = M_diag[free_id]
        M_mat_f = diags([M_f], [0], format="csr")

        # ── 4. 감쇠 ────────────────────────────────────────────────────────
        # RBE2 beam stiffness를 C와 K_eff_fs 계산에서 모두 배제
        # → beam K가 등가 힘을 수만 배 증폭하는 오류 방지
        K_struct_ff = K_struct[free_id, :][:, free_id].tocsr()
        alpha, beta_r = self._rayleigh_coeffs(damping, K_struct_ff, M_f)
        C_ff = assemble_rayleigh_C(alpha, beta_r, M_f, K_struct_ff)  # shell K만 사용
        print(f"    - Rayleigh damping: alpha={alpha:.4e}  beta={beta_r:.4e}", flush=True)

        # ── 5. K_eff 조립 + LU 분해 (1회) ──────────────────────────────────
        nc    = newmark_coeffs(newmark_beta, newmark_gamma, dt)
        K_eff_ff = K_ff + nc["a0"] * M_mat_f + nc["a1"] * C_ff  # 동역학은 full K
        print("    - LU decomposition ... ", end="", flush=True)
        lu = splu(K_eff_ff.tocsc())
        print("Done.", flush=True)

        # SPCD 커플링:
        # 정적 평형(K)은 RBE2 beam을 포함해야 변위가 올바르게 전달됨 (K_fs_full)
        # 감쇠(C)는 shell 구조물(K_struct_fs)만 사용 (C_fs_struct = beta_r * K_struct_fs)
        if n_spcd:
            K_fs_full   = K[free_id, :][:, spcd_id].tocsr()
            K_fs_struct = K_struct[free_id, :][:, spcd_id].tocsr()
        else:
            K_fs_full = K_fs_struct = None

        # ── 6. 시간 적분 루프 ─────────────────────────────────────────────
        n_steps    = int(np.ceil(T / dt))
        save_every = max(1, n_steps // n_save)

        u_f   = np.zeros(n_free)
        ud_f  = np.zeros(n_free)
        udd_f = np.zeros(n_free)

        t_saved, u_saved, v_saved, a_saved = [], [], [], []

        def _get_spcd_vecs(t_cur: float):
            """현재 시각의 SPCD 변위/속도/가속도 벡터."""
            u_s   = np.array([spcd_map[d].u_value(t_cur)   for d in spcd_id])
            ud_s  = np.array([spcd_map[d].ud_value(t_cur)  for d in spcd_id])
            udd_s = np.array([spcd_map[d].udd_value(t_cur) for d in spcd_id])
            return u_s, ud_s, udd_s

        for step in range(n_steps + 1):
            t_cur = step * dt

            # 외력 (FORCE 타입만)
            f_full = self._build_load_vector(t_cur, load_groups, ndof, nid_to_idx)
            f_f    = f_full[free_id]

            if n_spcd:
                u_s, ud_s, udd_s = _get_spcd_vecs(t_cur)
            else:
                u_s = ud_s = udd_s = None

            if step == 0:
                # 초기 가속도: M·ü = f − K·u  (u=0, SPCD 초기값 반영)
                rhs0 = f_f.copy()
                if n_spcd and K_fs_full is not None:
                    rhs0 -= K_fs_full @ u_s
                udd_f = rhs0 / M_f
            else:
                # Newmark 유효 하중
                f_eff = (f_f
                         + M_f  * (nc["a0"]*u_f  + nc["a2"]*ud_f  + nc["a3"]*udd_f)
                         + C_ff @ (nc["a1"]*u_f  + nc["a4"]*ud_f  + nc["a5"]*udd_f))
                if n_spcd and K_fs_full is not None:
                    f_eff -= K_fs_full @ u_s   # SPCD 처방 변위 기여분 (빔 포함)
                    f_eff -= (beta_r * K_fs_struct) @ ud_s  # SPCD 처방 속도 기여분 (빔 제외)

                u_f_new   = lu.solve(f_eff)
                udd_f_new = nc["a0"]*(u_f_new - u_f) - nc["a2"]*ud_f - nc["a3"]*udd_f
                ud_f_new  = ud_f + dt * ((1 - newmark_gamma)*udd_f + newmark_gamma*udd_f_new)
                u_f, ud_f, udd_f = u_f_new, ud_f_new, udd_f_new

            if step % save_every == 0 or step == n_steps:
                u_vec = np.zeros(ndof)
                v_vec = np.zeros(ndof)
                a_vec = np.zeros(ndof)
                u_vec[free_id] = u_f
                v_vec[free_id] = ud_f
                a_vec[free_id] = udd_f
                if n_spcd:
                    u_vec[spcd_id] = u_s          # 처방 변위를 결과에 직접 기록
                    a_vec[spcd_id] = udd_s

                t_saved.append(t_cur)
                u_saved.append(u_vec.reshape(n_nodes, 6))
                v_saved.append(v_vec.reshape(n_nodes, 6))
                a_saved.append(a_vec.reshape(n_nodes, 6))

        result = DynamicResult(np.array(t_saved), sorted_nids)
        result.u           = np.array(u_saved)
        result.v           = np.array(v_saved)
        result.a           = np.array(a_saved)
        result.solver_type = "direct"
        result.solver_info = dict(
            dt=dt, T=T, n_steps=n_steps,
            n_saved=len(t_saved), elapsed=time.time() - t_wall,
        )
        print(f" -> Done in {time.time()-t_wall:.1f}s | {result.summary()}", flush=True)
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — ESL 추출
    # ─────────────────────────────────────────────────────────────────────────

    def extract_esl(
        self,
        dynamic_result: DynamicResult,
        n_esl: int = 5,
        include_peak: bool = True,
    ) -> List[WHTLoadCase]:
        """
        동해석 결과로부터 등가 정하중(ESL) 추출.

        f_ESL(t*) = K · u(t*)  for selected time points t*

        Parameters
        ----------
        dynamic_result : DynamicResult
        n_esl          : 균등 간격 ESL 시점 수
        include_peak   : True이면 최대 변위 시점을 반드시 포함

        Returns
        -------
        list of WHTLoadCase  — WHTopographySolver에 직접 주입 가능
        """
        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model()
        K    = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=True)
        ndof = jm.ndof

        n_saved  = dynamic_result.n_save
        peak_idx = dynamic_result.peak_time_index()

        # 시점 인덱스 선택: 균등 분배 + 피크
        idx_list = np.linspace(0, n_saved - 1, n_esl, dtype=int).tolist()
        if include_peak and peak_idx not in idx_list:
            idx_list.append(peak_idx)
        idx_list = sorted(set(idx_list))

        load_cases: List[WHTLoadCase] = []
        for si in idx_list:
            t_val  = dynamic_result.t_saved[si]
            u_flat = dynamic_result.u[si].flatten()[:ndof]   # (ndof,)
            f_esl  = K @ u_flat                               # (ndof,)

            lc = WHTLoadCase(name=f"ESL_t{t_val:.5f}s")
            for i, nid in enumerate(sorted_nids):
                f_node = f_esl[i * 6: i * 6 + 6]
                if np.max(np.abs(f_node)) < 1e-12:
                    continue
                lc.forces.append(WHTForceEntry(nid, tuple(float(v) for v in f_node)))
            load_cases.append(lc)

        peak_t = dynamic_result.t_saved[peak_idx]
        print(
            f" -> [ESL] {len(load_cases)} load cases extracted "
            f"(peak @ t={peak_t:.4f}s, u_max="
            f"{float(np.max(np.abs(dynamic_result.u[peak_idx]))):.3e} mm)",
            flush=True,
        )
        return load_cases

    # ─────────────────────────────────────────────────────────────────────────
    # Public API — 응력/변형률 이력 복원
    # ─────────────────────────────────────────────────────────────────────────

    def recover_stress_history(
        self,
        dynamic_result: DynamicResult,
        fields: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> DynamicResult:
        """
        동해석 결과의 각 저장 스텝에 대해 응력/변형률을 복원합니다.

        정적 해석의 ElementStressRecovery와 동일한 로직을 적용하며,
        결과는 DynamicResult.stress_data에 (T, M, 6) 또는 (T, M) 형태로 저장됩니다.

        Parameters
        ----------
        dynamic_result : DynamicResult
            solve_direct_dynamic() 또는 solve_modal_dynamic()의 반환값.
        fields : list of str, optional
            복원할 필드명 리스트. None이면 주요 필드 전체 복원.
            예: [\"Stress\", \"Stress (Max Envelope)\", \"Strain\", \"Von Mises (Upper)\"]
        verbose : bool
            진행 상황 출력 여부.

        Returns
        -------
        DynamicResult
            stress_data가 채워진 동일 객체 (in-place).
        """
        from .wht_stress_recovery import ElementStressRecovery

        # 기본 복원 필드: 주요 6개 (전체 12개는 메모리 부담이 크므로 선택적)
        if fields is None:
            fields = [
                "Stress",               # Upper (+t/2) — 기본
                "Stress (Max Envelope)",# Upper/Lower 중 Von Mises 최대
                "Strain",               # Upper (+t/2)
                "Strain (Max Envelope)",# Upper/Lower 중 Max
            ]

        n_save  = dynamic_result.n_save
        n_nodes = len(dynamic_result.sorted_nids)
        sorted_nids = dynamic_result.sorted_nids

        if verbose:
            print(f"\n [Stress Recovery] {n_save} 스텝, 필드: {fields}", flush=True)
            t0 = time.time()

        # 첫 스텝으로 요소 수 파악
        u0 = dynamic_result.u[0]   # (n_nodes, 6)
        first = ElementStressRecovery.recover_quad4(self.model, u0, sorted_nids)
        # TRIA3가 혼재하면 추가
        tria = ElementStressRecovery.recover_tria3(self.model, u0, sorted_nids)
        # 두 결과를 합산 (요소별 중첩: 한 요소는 어느 한 쪽만 채워짐)
        sample_result = {k: first[k] + tria[k] for k in first}
        n_elem = sample_result["Stress"].shape[0]

        # (T, M, 6) 또는 (T, M) 버퍼 초기화
        stress_data: dict = {}
        for fld in fields:
            if fld not in sample_result:
                if verbose:
                    print(f"    [경고] 필드 '{fld}'를 찾을 수 없습니다. 건너뜁니다.")
                continue
            arr0 = sample_result[fld]
            if arr0.ndim == 2:          # (M, 6) → (T, M, 6)
                stress_data[fld] = np.zeros((n_save, n_elem, 6), dtype=np.float32)
            else:                        # (M,)   → (T, M)
                stress_data[fld] = np.zeros((n_save, n_elem), dtype=np.float32)

        # Von Mises 스칼라 필드 추가 (Stress가 복원될 경우)
        compute_vm_upper = "Stress" in stress_data
        compute_vm_envelope = "Stress (Max Envelope)" in stress_data
        if compute_vm_upper:
            stress_data["Von Mises (Upper)"] = np.zeros((n_save, n_elem), dtype=np.float32)
        if compute_vm_envelope:
            stress_data["Von Mises (Max Envelope)"] = np.zeros((n_save, n_elem), dtype=np.float32)

        def _von_mises(voigt: np.ndarray) -> np.ndarray:
            """Voigt (M, 6) → Von Mises (M,)."""
            s = voigt
            diff_sq = (s[:, 0]-s[:, 1])**2 + (s[:, 1]-s[:, 2])**2 + (s[:, 2]-s[:, 0])**2
            shear_sq = 6.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)
            return np.sqrt(0.5 * (diff_sq + shear_sq))

        # 각 스텝 루프
        for ti in range(n_save):
            u_frame = dynamic_result.u[ti]   # (n_nodes, 6)
            quad = ElementStressRecovery.recover_quad4(self.model, u_frame, sorted_nids)
            tria = ElementStressRecovery.recover_tria3(self.model, u_frame, sorted_nids)
            frame = {k: quad[k] + tria[k] for k in quad}

            for fld in stress_data:
                if fld.startswith("Von Mises"):
                    continue   # 아래에서 별도 처리
                if fld in frame:
                    stress_data[fld][ti] = frame[fld].astype(np.float32)

            if compute_vm_upper and "Stress" in frame:
                stress_data["Von Mises (Upper)"][ti] = _von_mises(frame["Stress"]).astype(np.float32)
            if compute_vm_envelope and "Stress (Max Envelope)" in frame:
                stress_data["Von Mises (Max Envelope)"][ti] = _von_mises(frame["Stress (Max Envelope)"]).astype(np.float32)

            if verbose and (ti + 1) % max(1, n_save // 10) == 0:
                print(f"    [{ti+1:4d}/{n_save}] ...", flush=True)

        dynamic_result.stress_data = stress_data
        if verbose:
            elapsed = time.time() - t0
            n_fields = len(stress_data)
            print(f" -> Stress Recovery 완료: {n_fields}개 필드, "
                  f"{elapsed:.1f}s ({n_save} steps × {n_elem} elements)", flush=True)

        return dynamic_result



    def _build_load_vector(
        self,
        t: float,
        load_groups: List[DynamicLoadGroup],
        ndof: int,
        nid_to_idx: dict,
    ) -> np.ndarray:
        """시각 t에서 전체 DOF 하중 벡터 구성."""
        f = np.zeros(ndof)
        for lg in load_groups:
            force_val = lg.evaluate(t)
            for nid in lg.node_ids:
                idx = nid_to_idx.get(nid)
                if idx is None:
                    continue
                g = idx * 6 + lg.dof
                if g < ndof:
                    f[g] += force_val
        return f

    def _rayleigh_coeffs(
        self,
        damping: DampingSpec,
        K_free,
        M_free: np.ndarray,
    ) -> tuple[float, float]:
        """DampingSpec → Rayleigh (α, β) 변환."""
        if damping.mode == "rayleigh":
            return damping.alpha, damping.beta

        # zeta 모드: 두 기준 주파수 자동 산출 또는 사용자 지정
        if damping.omega_ref1 > 0 and damping.omega_ref2 > 0:
            return damping_from_zeta(
                damping.zeta, damping.omega_ref1, damping.omega_ref2
            )

        # 자동: 저차 고유값 4개에서 1차/4차 선택
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh
        M_mat = diags([M_free], [0], format="csc")
        try:
            vals, _ = eigsh(
                K_free.tocsc(), k=4, M=M_mat,
                which="LM", sigma=-1.0, tol=1e-4, maxiter=20000,
            )
        except Exception:
            vals = np.array([1.0, 10.0, 100.0, 1000.0])  # fallback
        vals = np.sort(np.maximum(vals, 0.0))
        w_ref = np.sqrt(vals)
        nonzero = w_ref[w_ref > 2 * np.pi * 0.5]
        if len(nonzero) >= 2:
            return damping_from_zeta(damping.zeta, nonzero[0], nonzero[-1])
        elif len(nonzero) == 1:
            # 순수 강성 비례 감쇠
            beta = 2.0 * damping.zeta / nonzero[0]
            return 0.0, float(beta)
        else:
            return 0.0, 1e-4  # 최종 fallback

    def _modal_zeta(
        self,
        damping: DampingSpec,
        omegas: np.ndarray,
        M_free: np.ndarray,
        K,
        free_id: np.ndarray,
    ) -> np.ndarray:
        """모드별 감쇠비 ζ_r 배열 반환."""
        if damping.mode == "zeta":
            return np.full(len(omegas), damping.zeta)
        # Rayleigh → ζ_r = α/(2ω_r) + β ω_r/2
        alpha, beta = self._rayleigh_coeffs(
            damping,
            K[free_id, :][:, free_id],
            M_free,
        )
        return alpha / (2.0 * omegas) + beta * omegas / 2.0
