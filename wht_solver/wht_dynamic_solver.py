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

import os
# Force single-threading for BLAS/LAPACK to prevent threading crashes on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import jax
import jax.numpy as jnp
# High precision for structural dynamics
jax.config.update("jax_enable_x64", True)

import time
from typing import List, Optional, Dict, TYPE_CHECKING
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import splu
from tqdm import tqdm

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
        verbose: bool = True,
        method: str = 'scipy',
        label: str = '',
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
        verbose      : 출력 여부
        method       : 'scipy'(직접 해석) 또는 'jax'(JAX 가속)

        Returns
        -------
        DynamicResult
        """
        if damping is None:
            damping = DampingSpec(mode="zeta", zeta=0.02)

        if method.lower() == 'jax':
            return self._solve_direct_dynamic_jax(
                T, dt, load_groups, damping, newmark_beta, newmark_gamma, n_save, verbose
            )

        label_str = f"  [{label}]" if label else ""
        print(f"\n{'='*60}", flush=True)
        print(f" [Direct Dynamic]{label_str}  dt={dt:.2e}s  T={T:.2e}s", flush=True)
        print(f"{'='*60}", flush=True)
        t_wall = time.time()

        # ── 1. K, M 조립 ────────────────────────────────────────────────────
        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model()
        ndof    = jm.ndof
        n_nodes = len(sorted_nids)
        # Full K (shell + RBE2 stiff beams via JaxSSO)
        K       = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=True)
        # Shell-only K: Rayleigh 감쇠 기준 주파수 계산용 (RBE2 beam, RBE3 penalty 제외)
        K_struct = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx,
                                          stabilize=True, include_beams=False, include_rbe3=False)

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

        print(f"    - [Time Loop] Integrating {n_steps} steps...")
        pbar = tqdm(total=n_steps, desc="      Dynamic Solve", unit="step", leave=True)
        
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
            
            pbar.update(1)
        pbar.close()

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

    def _solve_direct_dynamic_jax(
        self,
        T: float,
        dt: float,
        load_groups: List[DynamicLoadGroup],
        damping: DampingSpec,
        newmark_beta: float = 0.25,
        newmark_gamma: float = 0.5,
        n_save: int = 100,
        verbose: bool = True,
    ) -> DynamicResult:
        """JAX-Accelerated Direct Dynamic Solve."""
        import jax
        import jax.numpy as jnp
        from jax.experimental import sparse
        from jax.scipy.sparse.linalg import cg
        
        if verbose:
            print(f"\n{'='*60}")
            print(f" [Direct Dynamic (JAX)] dt={dt:.2e}s  T={T:.2e}s")
            print(f"{'='*60}")
            print(f"    - [JAX] [1/5] Preparing model & assembling K/M matrices...", end="", flush=True)
            t_prep = time.time()

        # ── 1. 준비 ─────────────────────────────────────────────────────────
        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model()
        ndof    = jm.ndof
        n_nodes = len(sorted_nids)
        K       = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=True)
        K_struct = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, 
                                          stabilize=True, include_beams=False, include_rbe3=False)
        M_diag  = self._assemble_lumped_mass(jm, ndof, sorted_nids, nid_to_idx)
        
        # ── 1b. DOF 분류: SPCD → free_id에서 제거 (Scipy 솔버와 동일한 파티셔닝) ──
        #   [BUG FIX] 기존에는 free_id = jm.unknown_id 전체를 사용하여 SPCD DOF가
        #   free DOF에 포함되어 K_fs 커플링이 무효화되는 문제가 있었음.
        #   Scipy 솔버와 동일하게: SPCD DOF → spcd_id / 나머지 → free_id로 분리.
        spcd_map = {}
        unknown_set = set(int(x) for x in jm.unknown_id)
        for lg in load_groups:
            if lg.load_type == "SPCD":
                for nid in lg.node_ids:
                    idx = nid_to_idx.get(nid)
                    if idx is None:
                        continue
                    dof_gid = idx * 6 + lg.dof
                    if dof_gid in unknown_set:
                        spcd_map[dof_gid] = lg
        spcd_id = np.array(sorted(spcd_map.keys()), dtype=np.int32)
        spcd_set = set(spcd_id.tolist())
        n_spcd  = len(spcd_id)

        # SPCD DOF를 제외한 진정한 자유 DOF
        free_id = np.array(
            [d for d in jm.unknown_id if int(d) not in spcd_set], dtype=np.int32
        )
        n_free  = len(free_id)

        if n_spcd and verbose:
            print(f"    - [JAX] SPCD DOF: {n_spcd}개 | Free DOF: {n_free}개", flush=True)

        if verbose:
            print(f" Done ({time.time()-t_prep:.2f}s)")
            print(f"    - [JAX] [2/5] Converting to JAX Sparse (BCOO)...", end="", flush=True)
            t_conv_sparse = time.time()

        # ── 2. JAX Sparse & Dense 변환 ───────────────────────────────────────
        def to_jax_dense(scipy_mat):
            return jnp.array(scipy_mat.toarray(), dtype=jnp.float64)

        def to_jax_sparse(scipy_mat):
            coo = scipy_mat.tocoo()
            indices = jnp.stack([jnp.array(coo.row, dtype=jnp.int32), 
                                 jnp.array(coo.col, dtype=jnp.int32)], axis=1)
            return sparse.BCOO((jnp.array(coo.data, dtype=jnp.float64), indices), shape=coo.shape)

        # 시스템 행렬 구성
        # K_eff는 밀집 행렬로 변환하여 LU 분해 (안정성)
        # 10,000 DOF까지는 Dense LU가 CPU/GPU에서 더 안정적임 (특히 Penalty RBE3 사용 시)
        is_dense = (n_free < 10000)
        
        K_ff_jax_dense = to_jax_dense(K[free_id, :][:, free_id]) if is_dense else None
        K_ff_jax_sparse = to_jax_sparse(K[free_id, :][:, free_id])
        
        K_struct_ff_sparse = to_jax_sparse(K_struct[free_id, :][:, free_id])
        M_f  = jnp.array(M_diag[free_id], dtype=jnp.float64)
        
        alpha, beta_r = self._rayleigh_coeffs(damping, K_struct[free_id, :][:, free_id].tocsr(), M_diag[free_id])
        alpha = float(alpha)
        beta_r = float(beta_r)
        
        # C_ff = alpha*M + beta*K_struct (항상 Sparse로 유지하여 연산 속도 확보)
        diag_indices = jnp.stack([jnp.arange(n_free, dtype=jnp.int32), 
                                  jnp.arange(n_free, dtype=jnp.int32)], axis=1)
        M_sparse = sparse.BCOO((M_f, diag_indices), shape=(n_free, n_free))
        C_ff = alpha * M_sparse + beta_r * K_struct_ff_sparse
        
        raw_nc = newmark_coeffs(newmark_beta, newmark_gamma, dt)
        nc = {k: jnp.array(v, dtype=jnp.float64) for k, v in raw_nc.items()}
        
        if is_dense:
            # K_eff_ff (Dense) = K_ff + a0*M + a1*C
            K_eff_ff = K_ff_jax_dense + nc["a0"] * jnp.diag(M_f) + nc["a1"] * C_ff.todense()
        else:
            K_eff_ff = K_ff_jax_sparse + nc["a0"] * M_sparse + nc["a1"] * C_ff
        
        K_fs_full = to_jax_sparse(K[free_id, :][:, spcd_id]) if n_spcd else None
        K_fs_struct = to_jax_sparse(K_struct[free_id, :][:, spcd_id]) if n_spcd else None

        if verbose:
            print(f" Done ({time.time()-t_conv_sparse:.2f}s)")
            print(f"    - [JAX] [3/5] Pre-calculating SPCD trajectories...", end="", flush=True)
            t_traj = time.time()

        # ── 3. 시간 적분 루프 정의 (Nested Scan for Memory Efficiency) ────────────────
        n_steps = int(np.ceil(T / dt))
        save_every = max(1, n_steps // n_save)
        n_blocks = (n_steps + save_every - 1) // save_every
        total_steps_padded = n_blocks * save_every
        
        # Pre-calculate SPCD trajectories (Padded to match block size)
        times_padded = jnp.linspace(0, total_steps_padded * dt, total_steps_padded + 1, dtype=jnp.float64)
        u_s_all   = jnp.zeros((total_steps_padded + 1, n_spcd), dtype=jnp.float64)
        
        for i, gid in enumerate(spcd_id):
            lg = spcd_map[gid]
            t_data = getattr(lg, '_t', getattr(lg, 'time_arr', None))
            u_data = getattr(lg, '_u', getattr(lg, 'disp_arr', None))
            if t_data is not None and u_data is not None:
                # Use interp with padded times
                u_s_all = u_s_all.at[:, i].set(jnp.interp(times_padded, jnp.array(t_data), jnp.array(u_data)))
        
        ud_s_all  = jnp.gradient(u_s_all, dt, axis=0)
        udd_s_all = jnp.gradient(ud_s_all, dt, axis=0)
        f_f_all   = jnp.zeros((total_steps_padded + 1, n_free), dtype=jnp.float64)

        # Reshape inputs into blocks: (n_blocks, save_every, ...)
        # We exclude the last padded element for integration steps
        def reshape_to_blocks(arr):
            return arr[:total_steps_padded].reshape(n_blocks, save_every, -1)

        block_idxs  = reshape_to_blocks(jnp.arange(total_steps_padded, dtype=jnp.int32))
        block_u_s   = reshape_to_blocks(u_s_all)
        block_ud_s  = reshape_to_blocks(ud_s_all)
        block_udd_s = reshape_to_blocks(udd_s_all)
        block_f_f   = reshape_to_blocks(f_f_all)
        
        block_inputs = (block_idxs, block_u_s, block_ud_s, block_udd_s, block_f_f)

        # Jacobi preconditioner for CG (only for sparse)
        if not is_dense:
            diag_K_eff = K_eff_ff.todense().diagonal() 
            inv_diag = jnp.array(1.0, dtype=jnp.float64) / jnp.where(jnp.abs(diag_K_eff) > 1e-12, diag_K_eff, jnp.array(1.0, dtype=jnp.float64))
        else:
            inv_diag = None

        # Robust solve: Use dense direct solver with pre-factoring for medium sizes
        from jax.scipy.linalg import lu_factor, lu_solve
        if is_dense:
            K_eff_lu = lu_factor(K_eff_ff)
        else:
            K_eff_lu = None

        # Core Step Function (Same logic as before)
        def single_step(state, carry_in):
            u_f, ud_f, udd_f = state
            idx, u_s, ud_s, udd_s, f_f = carry_in
            f_eff = f_f + M_f * (nc["a0"]*u_f + nc["a2"]*ud_f + nc["a3"]*udd_f)
            f_eff = f_eff + C_ff @ (nc["a1"]*u_f + nc["a4"]*ud_f + nc["a5"]*udd_f)
            if n_spcd:
                f_eff -= K_fs_full @ u_s
                f_eff -= (beta_r * K_fs_struct) @ ud_s
            
            if is_dense:
                u_f_new = lu_solve(K_eff_lu, f_eff)
            else:
                from jax.scipy.sparse.linalg import cg
                u_f_new, _ = cg(lambda x: K_eff_ff @ x, f_eff, x0=u_f, tol=1e-8, maxiter=2000, M=lambda x: inv_diag * x)
            
            udd_f_new = nc["a0"]*(u_f_new - u_f) - nc["a2"]*ud_f - nc["a3"]*udd_f
            ud_f_new  = ud_f + dt * ((1.0 - newmark_gamma)*udd_f + newmark_gamma*udd_f_new)
            return (u_f_new, ud_f_new, udd_f_new), (u_f_new, ud_f_new, udd_f_new, u_s, udd_s)

        # Memory-Efficient Block Step
        def block_step(state, b_in):
            # b_in: (save_every, ...)
            # Inner scan: Only returns final state, doesn't store intermediate history!
            final_state, _ = jax.lax.scan(lambda s, ci: (single_step(s, ci)[0], None), state, b_in)
            
            # Extract last SPCD values for the block result
            u_s_last = b_in[1][-1]
            a_s_last = b_in[3][-1]
            return final_state, (final_state[0], final_state[1], final_state[2], u_s_last, a_s_last)

        init_state = (jnp.zeros(n_free, dtype=jnp.float64), 
                      jnp.zeros(n_free, dtype=jnp.float64), 
                      jnp.zeros(n_free, dtype=jnp.float64))

        if verbose:
            print(f" Done ({time.time()-t_traj:.2f}s)")
            print(f"    - [JAX] [4/5] JIT compiling & Block Integration ({n_blocks} blocks of {save_every} steps)...", flush=True)
            t0 = time.time()
        
        # ── 4. 실행 ──────────────────────────────────────────────────────────
        # outer scan: Only returns results for n_blocks frames!
        _, (u_f_hist, v_f_hist, a_f_hist, u_s_hist, a_s_hist) = jax.lax.scan(block_step, init_state, block_inputs)
        
        # [WHT] JAX 비동기 디스패치 완료 대기: block_until_ready()를 호출하지 않으면
        # "Time Integration Finished"가 JIT 컴파일 완료 시점(~1s)에 찍히고,
        # 실제 연산은 Step 5(np.array 호출)에서 수행되어 수 분이 소요되는 문제가 있음.
        u_f_hist = jax.block_until_ready(u_f_hist)
        
        if verbose:
            print(f"    - [JAX] Time Integration Finished ({time.time()-t0:.2f}s) [실제 연산 완료]")
            print(f"    - [JAX] [5/5] Converting {len(u_f_hist)} frames to Numpy...", end="", flush=True)
            t_conv = time.time()

        # [WHT] Memory Management: Delete large device arrays before conversion to avoid stalls
        del block_inputs, block_u_s, block_ud_s, block_udd_s, block_f_f
        del K_ff_jax_sparse, K_struct_ff_sparse, C_ff
        if is_dense: del K_eff_ff, K_eff_lu
        import gc; gc.collect()

        u_f_hist_np = np.array(jax.device_get(u_f_hist))
        v_f_hist_np = np.array(jax.device_get(v_f_hist))
        a_f_hist_np = np.array(jax.device_get(a_f_hist))
        if n_spcd:
            u_s_hist_np = np.array(jax.device_get(u_s_hist))
            a_s_hist_np = np.array(jax.device_get(a_s_hist))
        
        u_saved = []
        v_saved = []
        a_saved = []
        
        for i in range(n_blocks):
            u_vec = np.zeros(ndof)
            v_vec = np.zeros(ndof)
            a_vec = np.zeros(ndof)
            u_vec[free_id] = u_f_hist_np[i]
            v_vec[free_id] = v_f_hist_np[i]
            a_vec[free_id] = a_f_hist_np[i]
            if n_spcd:
                u_vec[spcd_id] = u_s_hist_np[i]
                a_vec[spcd_id] = a_s_hist_np[i]
            u_saved.append(u_vec.reshape(n_nodes, 6))
            v_saved.append(v_vec.reshape(n_nodes, 6))
            a_saved.append(a_vec.reshape(n_nodes, 6))

        if verbose:
            print(f" Done ({time.time()-t_conv:.2f}s)")

        # Result times for each block (at the end of the block)
        t_hist = np.arange(1, n_blocks + 1) * save_every * dt
        res = DynamicResult(t_hist, sorted_nids)
        res.u = np.array(u_saved)
        res.v = np.array(v_saved)
        res.a = np.array(a_saved)
        res.solver_type = "jax_direct"
        res.solver_info = dict(dt=dt, T=T, method="jax_cg", n_steps=n_steps)
        return res

    # Public API — ESL 추출
    # ─────────────────────────────────────────────────────────────────────────

    def extract_esl(
        self,
        dynamic_result: DynamicResult,
        n_esl: int = 5,
        include_peak: bool = True,
    ) -> List[WHTLoadCase]:
        """
        동해석 결과로부터 등가 정하중(ESL) 추출 (기존 방식: 균등 시점).

        f_ESL(t*) = K · u(t*)  for selected time points t*

        Parameters
        ----------
        dynamic_result : DynamicResult
        n_esl          : 균등 간격 ESL 시점 수
        include_peak   : True이면 최대 변위 시점을 반드시 포함

        Returns
        -------
        list of WHTLoadCase
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

            lc = WHTLoadCase(name=f"ESL_Uniform_t{t_val:.5f}s")
            for i, nid in enumerate(sorted_nids):
                f_node = f_esl[i * 6: i * 6 + 6]
                if np.max(np.abs(f_node)) < 1e-12:
                    continue
                lc.forces.append(WHTForceEntry(nid, tuple(float(v) for v in f_node)))
            load_cases.append(lc)

        return load_cases

    def extract_esl_advanced(
        self,
        dynamic_result: DynamicResult,
        n_windows: int = 30,
        n_top: int = 10,
        diversity_weight: float = 0.5,
    ) -> List[WHTLoadCase]:
        """
        [Method A+] 동해석 결과로부터 다양성을 고려한 고급 ESL 로드케이스 추출.
        
        1. 시계열을 n_windows로 분할하여 각 구간의 변형 에너지 피크(Candidate)를 추출합니다.
        2. Candidate들 중 서로 물리적인 변형 형상(Displacement Vector)이 가장 이질적인(Diverse) 
           상위 n_top개를 선택하여 정하중 케이스로 변환합니다. (Cosine Similarity 기반)

        Parameters
        ----------
        dynamic_result : DynamicResult
            동해석 결과
        n_windows : int
            시간 이력을 분할할 구간 수 (기본 30)
        n_top : int
            최종 추출할 로드케이스 수 (기본 10)
        diversity_weight : float
            다양성 가중치 (미사용, 현재는 Greedy Max-Min Similarity 방식 사용)

        Returns
        -------
        List[WHTLoadCase]
        """
        print(f"\n [ESL Advanced] Extracting {n_top} diverse snapshots from {n_windows} windows...")
        
        # 1. 기본 정보 및 Candidate 준비
        jm, sorted_nids, nid_to_idx = self._build_jaxsso_model()
        K = self._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=True)
        ndof = jm.ndof
        n_saved = dynamic_result.n_save
        
        # Strain Energy 계산 및 Binning
        strain_energies = np.zeros(n_saved)
        u_vectors = [] # Normalized displacement vectors for similarity check
        for i in range(n_saved):
            u_flat = dynamic_result.u[i].flatten()[:ndof]
            se = 0.5 * np.dot(u_flat, K @ u_flat)
            strain_energies[i] = se
            
            # Similarity 계산용 정규화 벡터 (L2 norm)
            norm = np.linalg.norm(u_flat)
            u_vectors.append(u_flat / (norm + 1e-12))
            
        # 2. 구간별 피크(Candidate Pool) 추출
        window_size = max(1, n_saved // n_windows)
        candidates = []
        for w in range(n_windows):
            start = w * window_size
            end = (w + 1) * window_size if w < n_windows - 1 else n_saved
            if start >= n_saved: break
            idx = start + np.argmax(strain_energies[start:end])
            candidates.append(idx)
        
        candidates = sorted(list(set(candidates))) # 중복 제거
        
        # 3. 다양성 기반 Greedy 선택 알고리즘 (Greedy Max-Min Similarity)
        # 목적: 이미 선택된 세트와 가장 '안 닮은' (Similarity가 가장 낮은) 후보를 순차적으로 추가
        
        # 첫 번째 선택: 전체 변형 에너지가 가장 큰 시점
        selected_indices = [candidates[np.argmax([strain_energies[c] for c in candidates])]]
        remaining = [c for c in candidates if c not in selected_indices]
        
        while len(selected_indices) < n_top and remaining:
            # 남은 후보들 중, 현재 선택된 세트와의 '최대 유사도'가 가장 '낮은' 후보 선택
            similarities = []
            for r_idx in remaining:
                u_r = u_vectors[r_idx]
                # 이미 선택된 것들과의 코사인 유사도 중 최댓값 (가장 닮은 정도)
                max_sim = max([np.dot(u_r, u_vectors[s_idx]) for s_idx in selected_indices])
                similarities.append(max_sim)
            
            # 유사도의 최댓값이 가장 작은 (즉, 가장 이질적인) 후보 선정
            best_idx = np.argmin(similarities)
            selected_indices.append(remaining.pop(best_idx))
            
        selected_indices = sorted(selected_indices)
        
        # 4. 로드케이스 생성 (Prescribed Displacement 적용)
        load_cases = []
        from .load_cases import WHTBCEntry
        for si in selected_indices:
            t_val = dynamic_result.t_saved[si]
            se_val = strain_energies[si]
            u_snap = dynamic_result.u[si]
            
            lc = WHTLoadCase(name=f"ESL_Peak_t{t_val:.5f}s_SE{se_val:.1e}")
            for i, nid in enumerate(sorted_nids):
                u_node = u_snap[i]
                lc.add_bc(nid, dofs=(0, 1, 2, 3, 4, 5), value=0.0)
                for d in range(6):
                    val = float(u_node[d])
                    if abs(val) > 1e-15:
                        lc.bcs.append(WHTBCEntry(nid, (d,), val))
            
            load_cases.append(lc)
            print(f"    - Selected Snapshot: t={t_val:.4f}s, SE={se_val:.3e}")

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
        동해석 결과의 각 저장 스텝에 대해 응력/변형률을 Nodal Point Data로 복원합니다.
        """
        from .wht_stress_recovery import ElementStressRecovery

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

        # [WHT] Nodal Averaging Setup
        M_total = len(self.model.elements)
        sorted_eids = sorted(self.model.elements.keys())
        
        node_count = np.zeros(n_nodes, dtype=np.int32)
        quad_to_node = np.zeros((M_total, 4), dtype=np.int64) - 1
        tria_to_node = np.zeros((M_total, 3), dtype=np.int64) - 1
        
        nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}
        
        for i, eid in enumerate(sorted_eids):
            elem = self.model.elements[eid]
            is_obj = hasattr(elem, 'type')
            etype = elem.type if is_obj else self.model.element_types.get(eid, "QUAD4")
            node_ids = elem.node_ids if is_obj else elem
            
            if etype in ["QUAD", "QUAD4"] and len(node_ids) == 4:
                idx_list = [nid_to_idx[n] for n in node_ids]
                quad_to_node[i] = idx_list
                for idx in idx_list:
                    node_count[idx] += 1
            elif etype in ["TRIA", "TRIA3"] and len(node_ids) == 3:
                idx_list = [nid_to_idx[n] for n in node_ids]
                tria_to_node[i] = idx_list
                for idx in idx_list:
                    node_count[idx] += 1

        node_count_safe = np.maximum(node_count, 1)[:, None]

        # 첫 스텝으로 필드 파악
        u0 = dynamic_result.u[0]
        quad_0 = ElementStressRecovery.recover_quad4_nodal(self.model, u0, sorted_nids)
        tria_0 = ElementStressRecovery.recover_tria3_nodal(self.model, u0, sorted_nids)

        stress_data: dict = {}
        for fld in fields:
            if fld not in quad_0:
                if verbose:
                    print(f"    [경고] 필드 '{fld}'를 찾을 수 없습니다. 건너뜁니다.")
                continue
            arr0 = quad_0[fld]
            if arr0.ndim == 3:          # (M, 4, 6) -> (T, N_nodes, 6)
                stress_data[fld] = np.zeros((n_save, n_nodes, 6), dtype=np.float32)
            else:                        
                stress_data[fld] = np.zeros((n_save, n_nodes), dtype=np.float32)

        compute_vm_upper = "Stress" in stress_data
        compute_vm_envelope = "Stress (Max Envelope)" in stress_data
        if compute_vm_upper:
            stress_data["Von Mises (Upper)"] = np.zeros((n_save, n_nodes), dtype=np.float32)
        if compute_vm_envelope:
            stress_data["Von Mises (Max Envelope)"] = np.zeros((n_save, n_nodes), dtype=np.float32)

        def _von_mises(voigt: np.ndarray) -> np.ndarray:
            s = voigt
            diff_sq = (s[:, 0]-s[:, 1])**2 + (s[:, 1]-s[:, 2])**2 + (s[:, 2]-s[:, 0])**2
            shear_sq = 6.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)
            return np.sqrt(0.5 * (diff_sq + shear_sq))

        if verbose:
            print(f"    - [Stress History] [Final Step] Recovering {n_save} frames to NODAL Data (PointData)...")
        pbar = tqdm(total=n_save, desc="      Stress History", unit="frame", leave=False)
        
        q_mask = quad_to_node[:, 0] >= 0
        t_mask = tria_to_node[:, 0] >= 0
        quad_nodes_mapped = quad_to_node[q_mask]
        tria_nodes_mapped = tria_to_node[t_mask]

        for ti in range(n_save):
            u_frame = dynamic_result.u[ti]
            quad = ElementStressRecovery.recover_quad4_nodal(self.model, u_frame, sorted_nids)
            tria = ElementStressRecovery.recover_tria3_nodal(self.model, u_frame, sorted_nids)

            for fld in stress_data:
                if fld.startswith("Von Mises"):
                    continue
                if fld in quad:
                    nodal_val = np.zeros((n_nodes, 6), dtype=np.float32)
                    q_val = quad[fld]
                    t_val = tria[fld]
                    
                    if np.any(q_mask):
                        np.add.at(nodal_val, quad_nodes_mapped, q_val[q_mask])
                    if np.any(t_mask):
                        np.add.at(nodal_val, tria_nodes_mapped, t_val[t_mask])
                        
                    nodal_val /= node_count_safe
                    stress_data[fld][ti] = nodal_val

            if compute_vm_upper and "Stress" in quad:
                stress_data["Von Mises (Upper)"][ti] = _von_mises(stress_data["Stress"][ti]).astype(np.float32)
            if compute_vm_envelope and "Stress (Max Envelope)" in quad:
                stress_data["Von Mises (Max Envelope)"][ti] = _von_mises(stress_data["Stress (Max Envelope)"][ti]).astype(np.float32)

            pbar.update(1)
        
        pbar.close()

        dynamic_result.stress_data = stress_data
        if verbose:
            elapsed = time.time() - t0
            n_fields = len(stress_data)
            print(f" -> Nodal Stress Recovery 완료: {n_fields}개 필드, "
                  f"{elapsed:.1f}s ({n_save} steps × {n_nodes} nodes)", flush=True)

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
