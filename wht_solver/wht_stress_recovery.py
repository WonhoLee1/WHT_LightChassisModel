"""
wht_stress_recovery.py
======================
WHT FEM Framework — Element Stress & Strain Recovery Module (v2)

Through-Thickness Integration Points 지원:
  - Upper (+t/2), Mid (0), Lower (-t/2)
  - Max Envelope (Upper/Lower 중 Von Mises 최대)

Stress & Strain 모두 Membrane/Bending 분리 출력.

Voigt Notation: (σ_xx, σ_yy, σ_zz, τ_xy, τ_xz, τ_yz)
  - Strain 전단 성분은 Engineering Shear (γ_xy = 2ε_xy)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


# ---------------------------------------------------------------------------
# Through-Thickness Integration Points 정의
# ---------------------------------------------------------------------------
# z_ratio: -1.0 = Lower (-t/2), 0.0 = Mid (0), +1.0 = Upper (+t/2)
DEFAULT_Z_RATIOS = {"lower": -1.0, "mid": 0.0, "upper": 1.0}


class ElementStressRecovery:
    """
    Shell 요소 응력/변형률 복원 엔진.

    Through-Thickness Integration Points를 지원하며,
    각 적분점별 응력/변형률을 분리 출력합니다.

    반환 구조:
        result_dict: Dict[str, np.ndarray]
            Keys: "Stress", "Stress (Mid)", "Stress (Lower)",
                  "Stress (Membrane)", "Stress (Bending)",
                  "Stress (Max Envelope)",
                  "Strain", "Strain (Mid)", "Strain (Lower)",
                  "Strain (Membrane)", "Strain (Bending)",
                  "Strain (Max Envelope)"
            Values: (M_total, 6) ndarray (Global Voigt)
    """

    @staticmethod
    def recover_quad4(
        wht_model,
        u_global_array: np.ndarray,
        sorted_nids: List[int],
    ) -> Dict[str, np.ndarray]:
        """
        QUAD4 요소의 중심(centroid)에서 응력/변형률을 복원합니다.

        Parameters
        ----------
        wht_model : WHTMeshModel
            메시 모델 인스턴스.
        u_global_array : np.ndarray, shape (N, 6)
            전역 절점 변위 벡터.
        sorted_nids : list of int
            정렬된 절점 ID 리스트 (행 인덱스 매핑).

        Returns
        -------
        dict of str -> np.ndarray
            각 적분점/분리 성분별 응력/변형률 배열 (M_total, 6).
        """
        nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}

        M_total = len(wht_model.elements)

        eid_list, node_idx_list, E_list, nu_list, t_list, row_map = [], [], [], [], [], []

        # 1. 필터링 및 프로퍼티 매핑
        for i, eid in enumerate(sorted(wht_model.elements.keys())):
            elem = wht_model.elements[eid]
            is_obj = hasattr(elem, 'type')
            etype = elem.type if is_obj else wht_model.element_types.get(eid, "QUAD4")
            node_ids = elem.node_ids if is_obj else elem

            if etype in ["QUAD", "QUAD4"] and len(node_ids) == 4:
                eid_list.append(eid)
                node_idx_list.append([nid_to_idx[n] for n in node_ids])

                pid = getattr(elem, 'pid', 0) if hasattr(elem, 'pid') else 0
                E, nu = 210000.0, 0.3
                t = 1.0

                if hasattr(wht_model, 'properties') and pid in wht_model.properties:
                    prop = wht_model.properties[pid]
                    t = getattr(prop, 't', 1.0)
                    mid = getattr(prop, 'mid', 0)
                    if hasattr(wht_model, 'materials') and mid in wht_model.materials:
                        mat = wht_model.materials[mid]
                        E = getattr(mat, 'E', 210000.0)
                        nu = getattr(mat, 'nu', 0.3)

                E_list.append(E)
                nu_list.append(nu)
                t_list.append(t)
                row_map.append(i)

        if not eid_list:
            return _empty_result_dict(M_total)

        idx_arr = np.array(node_idx_list)
        E_arr = np.array(E_list)
        nu_arr = np.array(nu_list)
        t_arr = np.array(t_list)
        row_arr = np.array(row_map)

        # 2. 로컬 좌표계 구성
        c_list = []
        for nid in sorted_nids:
            n = wht_model.nodes[nid]
            c_list.append([n.x, n.y, n.z] if hasattr(n, 'x') else n)
        c_all = np.array(c_list)
        C = c_all[idx_arr]     # (M_quad, 4, 3)
        U = u_global_array[idx_arr]  # (M_quad, 4, 6)

        V1 = C[:, 1, :] - C[:, 0, :]
        V2 = C[:, 3, :] - C[:, 0, :]

        norm_V1 = np.linalg.norm(V1, axis=1, keepdims=True)
        X_loc = V1 / norm_V1

        Z_raw = np.cross(X_loc, V2)
        norm_Z = np.linalg.norm(Z_raw, axis=1, keepdims=True)
        Z_loc = Z_raw / norm_Z
        Y_loc = np.cross(Z_loc, X_loc)

        T = np.stack([X_loc, Y_loc, Z_loc], axis=1)  # (M_quad, 3, 3)

        # 변위/회전 → 로컬 변환
        U_disp_loc = np.einsum('mij, mkj -> mki', T, U[:, :, :3])
        U_rot_loc = np.einsum('mij, mkj -> mki', T, U[:, :, 3:6])

        # 3. Jacobian 기반 형상함수 미분 (중심점 xi=0, eta=0)
        p0 = C[:, 0, :]
        C_loc = np.stack([
            np.einsum('mi,mi->m', C[:, k, :] - p0, X_loc) for k in range(4)
        ], axis=1)
        D_loc = np.stack([
            np.einsum('mi,mi->m', C[:, k, :] - p0, Y_loc) for k in range(4)
        ], axis=1)

        dNxi = np.array([-1.0, 1.0, 1.0, -1.0]) * 0.25
        dNeta = np.array([-1.0, -1.0, 1.0, 1.0]) * 0.25

        J11 = np.dot(C_loc, dNxi)
        J12 = np.dot(D_loc, dNxi)
        J21 = np.dot(C_loc, dNeta)
        J22 = np.dot(D_loc, dNeta)

        detJ = J11 * J22 - J12 * J21
        detJ_safe = np.where(np.abs(detJ) > 1e-12, detJ, 1e-12)

        invJ11 = J22 / detJ_safe
        invJ12 = -J12 / detJ_safe
        invJ21 = -J21 / detJ_safe
        invJ22 = J11 / detJ_safe

        dN_dx = np.outer(invJ11, dNxi) + np.outer(invJ12, dNeta)
        dN_dy = np.outer(invJ21, dNxi) + np.outer(invJ22, dNeta)

        u_x, u_y = U_disp_loc[:, :, 0], U_disp_loc[:, :, 1]
        th_x, th_y = U_rot_loc[:, :, 0], U_rot_loc[:, :, 1]

        # Membrane 변형률 (z-independent)
        eps_xx_m = np.sum(dN_dx * u_x, axis=1)
        eps_yy_m = np.sum(dN_dy * u_y, axis=1)
        gamma_xy_m = np.sum(dN_dy * u_x + dN_dx * u_y, axis=1)

        # Curvature (bending gradients, z-independent)
        kappa_xx = np.sum(dN_dx * th_y, axis=1)          # ∂θ_y/∂x
        kappa_yy = -np.sum(dN_dy * th_x, axis=1)         # -∂θ_x/∂y
        kappa_xy = np.sum(dN_dy * th_y - dN_dx * th_x, axis=1)

        # 각 적분점에서 응력/변형률 계산
        return _compute_all_layers(
            M_total, row_arr, E_arr, nu_arr, t_arr, T,
            eps_xx_m, eps_yy_m, gamma_xy_m,
            kappa_xx, kappa_yy, kappa_xy,
        )

    @staticmethod
    def recover_tria3(
        wht_model,
        u_global_array: np.ndarray,
        sorted_nids: List[int],
    ) -> Dict[str, np.ndarray]:
        """
        TRIA3 (CST) 요소의 중심(centroid)에서 응력/변형률을 복원합니다.

        Parameters
        ----------
        wht_model : WHTMeshModel
            메시 모델 인스턴스.
        u_global_array : np.ndarray, shape (N, 6)
            전역 절점 변위 벡터.
        sorted_nids : list of int
            정렬된 절점 ID 리스트 (행 인덱스 매핑).

        Returns
        -------
        dict of str -> np.ndarray
            각 적분점/분리 성분별 응력/변형률 배열 (M_total, 6).
        """
        nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}

        M_total = len(wht_model.elements)

        eid_list, node_idx_list, E_list, nu_list, t_list, row_map = [], [], [], [], [], []

        for i, eid in enumerate(sorted(wht_model.elements.keys())):
            elem = wht_model.elements[eid]
            is_obj = hasattr(elem, 'type')
            etype = elem.type if is_obj else wht_model.element_types.get(eid, "TRIA3")
            node_ids = elem.node_ids if is_obj else elem

            if etype in ["TRIA", "TRIA3"] and len(node_ids) == 3:
                eid_list.append(eid)
                node_idx_list.append([nid_to_idx[n] for n in node_ids])

                pid = getattr(elem, 'pid', 0) if hasattr(elem, 'pid') else 0
                E, nu = 210000.0, 0.3
                t = 1.0

                if hasattr(wht_model, 'properties') and pid in wht_model.properties:
                    prop = wht_model.properties[pid]
                    t = getattr(prop, 't', 1.0)
                    mid = getattr(prop, 'mid', 0)
                    if hasattr(wht_model, 'materials') and mid in wht_model.materials:
                        mat = wht_model.materials[mid]
                        E = getattr(mat, 'E', 210000.0)
                        nu = getattr(mat, 'nu', 0.3)

                E_list.append(E)
                nu_list.append(nu)
                t_list.append(t)
                row_map.append(i)

        if not eid_list:
            return _empty_result_dict(M_total)

        idx_arr = np.array(node_idx_list)
        E_arr = np.array(E_list)
        nu_arr = np.array(nu_list)
        t_arr = np.array(t_list)
        row_arr = np.array(row_map)

        # 로컬 좌표계 구성
        c_list = []
        for nid in sorted_nids:
            n = wht_model.nodes[nid]
            c_list.append([n.x, n.y, n.z] if hasattr(n, 'x') else n)
        c_all = np.array(c_list)
        C = c_all[idx_arr]
        U = u_global_array[idx_arr]

        V1 = C[:, 1, :] - C[:, 0, :]
        V2 = C[:, 2, :] - C[:, 0, :]

        norm_V1 = np.linalg.norm(V1, axis=1, keepdims=True)
        X_loc = V1 / norm_V1

        Z_raw = np.cross(X_loc, V2)
        norm_Z = np.linalg.norm(Z_raw, axis=1, keepdims=True)
        Z_loc = Z_raw / norm_Z
        Y_loc = np.cross(Z_loc, X_loc)

        T = np.stack([X_loc, Y_loc, Z_loc], axis=1)

        U_disp_loc = np.einsum('mij, mkj -> mki', T, U[:, :, :3])
        U_rot_loc = np.einsum('mij, mkj -> mki', T, U[:, :, 3:6])

        # CST 형상함수 미분
        x2 = norm_V1[:, 0]
        x3 = np.sum(V2 * X_loc, axis=1)
        y3 = np.sum(V2 * Y_loc, axis=1)

        two_A = np.maximum(np.abs(x2 * y3), 1e-12)

        dN_dx = np.stack([-y3, y3, np.zeros_like(y3)], axis=1) / two_A[:, None]
        dN_dy = np.stack([x3 - x2, -x3, x2], axis=1) / two_A[:, None]

        u_x, u_y = U_disp_loc[:, :, 0], U_disp_loc[:, :, 1]
        th_x, th_y = U_rot_loc[:, :, 0], U_rot_loc[:, :, 1]

        # Membrane
        eps_xx_m = np.sum(dN_dx * u_x, axis=1)
        eps_yy_m = np.sum(dN_dy * u_y, axis=1)
        gamma_xy_m = np.sum(dN_dy * u_x + dN_dx * u_y, axis=1)

        # Curvature
        kappa_xx = np.sum(dN_dx * th_y, axis=1)
        kappa_yy = -np.sum(dN_dy * th_x, axis=1)
        kappa_xy = np.sum(dN_dy * th_y - dN_dx * th_x, axis=1)

        return _compute_all_layers(
            M_total, row_arr, E_arr, nu_arr, t_arr, T,
            eps_xx_m, eps_yy_m, gamma_xy_m,
            kappa_xx, kappa_yy, kappa_xy,
        )


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _empty_result_dict(M_total: int) -> Dict[str, np.ndarray]:
    """빈 결과 딕셔너리를 반환합니다 (해당 요소 타입이 없을 때)."""
    z = np.zeros((M_total, 6))
    return {
        "Stress": z.copy(), "Stress (Mid)": z.copy(), "Stress (Lower)": z.copy(),
        "Stress (Membrane)": z.copy(), "Stress (Bending)": z.copy(),
        "Stress (Max Envelope)": z.copy(),
        "Strain": z.copy(), "Strain (Mid)": z.copy(), "Strain (Lower)": z.copy(),
        "Strain (Membrane)": z.copy(), "Strain (Bending)": z.copy(),
        "Strain (Max Envelope)": z.copy(),
    }


def _compute_at_z(
    z_dist: np.ndarray,
    E_arr: np.ndarray,
    nu_arr: np.ndarray,
    T: np.ndarray,
    eps_xx_m: np.ndarray,
    eps_yy_m: np.ndarray,
    gamma_xy_m: np.ndarray,
    kappa_xx: np.ndarray,
    kappa_yy: np.ndarray,
    kappa_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    특정 두께 위치(z_dist)에서 로컬 응력/변형률을 계산하고 전역 Voigt로 변환합니다.

    Parameters
    ----------
    z_dist : np.ndarray, shape (M,)
        두께 방향 위치 (z = z_ratio * t/2).
    E_arr : np.ndarray, shape (M,)
        영 계수.
    nu_arr : np.ndarray, shape (M,)
        포아송 비.
    T : np.ndarray, shape (M, 3, 3)
        로컬→전역 변환 행렬.
    eps_xx_m, eps_yy_m, gamma_xy_m : np.ndarray, shape (M,)
        면내(Membrane) 변형률 성분.
    kappa_xx, kappa_yy, kappa_xy : np.ndarray, shape (M,)
        굽힘 곡률 성분.

    Returns
    -------
    stress_voigt : np.ndarray, shape (M, 6)
        전역 Voigt 응력.
    strain_voigt : np.ndarray, shape (M, 6)
        전역 Voigt 변형률 (Engineering Shear).
    """
    M = len(E_arr)

    # Total strain at z = membrane + z * curvature
    eps_xx = eps_xx_m + z_dist * kappa_xx
    eps_yy = eps_yy_m + z_dist * kappa_yy
    gamma_xy = gamma_xy_m + z_dist * kappa_xy

    # Plane Stress 구성 방정식
    factor = E_arr / (1.0 - nu_arr ** 2)
    sig_xx = factor * (eps_xx + nu_arr * eps_yy)
    sig_yy = factor * (nu_arr * eps_xx + eps_yy)
    sig_xy = factor * ((1.0 - nu_arr) / 2.0) * gamma_xy

    # Plane Stress: eps_zz = -nu/(1-nu) * (eps_xx + eps_yy)
    eps_zz = -(nu_arr / (1.0 - nu_arr)) * (eps_xx + eps_yy)

    # 로컬 텐서(3x3) 구성
    sig_loc = np.zeros((M, 3, 3))
    sig_loc[:, 0, 0] = sig_xx
    sig_loc[:, 1, 1] = sig_yy
    sig_loc[:, 0, 1] = sig_xy
    sig_loc[:, 1, 0] = sig_xy

    eps_loc = np.zeros((M, 3, 3))
    eps_loc[:, 0, 0] = eps_xx
    eps_loc[:, 1, 1] = eps_yy
    eps_loc[:, 2, 2] = eps_zz
    eps_loc[:, 0, 1] = gamma_xy / 2.0
    eps_loc[:, 1, 0] = gamma_xy / 2.0

    # 전역 공간으로 역회전: σ_glob = T^T · σ_loc · T
    sig_glob = np.einsum('mji, mjk, mkl -> mil', T, sig_loc, T)
    eps_glob = np.einsum('mji, mjk, mkl -> mil', T, eps_loc, T)

    # Voigt 패킹 (xx, yy, zz, xy, xz, yz)
    stress_voigt = np.zeros((M, 6))
    stress_voigt[:, 0] = sig_glob[:, 0, 0]
    stress_voigt[:, 1] = sig_glob[:, 1, 1]
    stress_voigt[:, 2] = sig_glob[:, 2, 2]
    stress_voigt[:, 3] = sig_glob[:, 0, 1]
    stress_voigt[:, 4] = sig_glob[:, 0, 2]
    stress_voigt[:, 5] = sig_glob[:, 1, 2]

    strain_voigt = np.zeros((M, 6))
    strain_voigt[:, 0] = eps_glob[:, 0, 0]
    strain_voigt[:, 1] = eps_glob[:, 1, 1]
    strain_voigt[:, 2] = eps_glob[:, 2, 2]
    strain_voigt[:, 3] = eps_glob[:, 0, 1] * 2.0  # Engineering shear
    strain_voigt[:, 4] = eps_glob[:, 0, 2] * 2.0
    strain_voigt[:, 5] = eps_glob[:, 1, 2] * 2.0

    return stress_voigt, strain_voigt


def _von_mises_voigt(voigt: np.ndarray) -> np.ndarray:
    """
    Voigt 표기법 텐서로부터 Von Mises 등가 스칼라를 계산합니다.

    Parameters
    ----------
    voigt : np.ndarray, shape (M, 6)
        Voigt 순서 (xx, yy, zz, xy, xz, yz).

    Returns
    -------
    np.ndarray, shape (M,)
        Von Mises 등가값.
    """
    s = voigt
    diff_sq = (s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 + (s[:, 2] - s[:, 0]) ** 2
    shear_sq = 6.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2)
    return np.sqrt(0.5 * (diff_sq + shear_sq))


def _compute_all_layers(
    M_total: int,
    row_arr: np.ndarray,
    E_arr: np.ndarray,
    nu_arr: np.ndarray,
    t_arr: np.ndarray,
    T: np.ndarray,
    eps_xx_m: np.ndarray,
    eps_yy_m: np.ndarray,
    gamma_xy_m: np.ndarray,
    kappa_xx: np.ndarray,
    kappa_yy: np.ndarray,
    kappa_xy: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Upper/Mid/Lower 3개 적분점 및 Membrane/Bending 분리, Max Envelope을 계산합니다.

    Parameters
    ----------
    M_total : int
        전체 요소 수 (출력 배열 크기).
    row_arr : np.ndarray
        현재 요소 타입에 해당하는 행 인덱스.
    E_arr, nu_arr, t_arr : np.ndarray
        재질/두께 속성.
    T : np.ndarray, shape (M_elem, 3, 3)
        로컬→전역 변환 행렬.
    eps_xx_m, eps_yy_m, gamma_xy_m : np.ndarray
        면내 변형률 성분.
    kappa_xx, kappa_yy, kappa_xy : np.ndarray
        굽힘 곡률 성분.

    Returns
    -------
    dict of str -> np.ndarray
        12개 키를 가진 결과 딕셔너리. 각 값은 (M_total, 6).
    """
    result = _empty_result_dict(M_total)

    # --- Upper (+t/2) ---
    z_upper = t_arr / 2.0
    s_upper, e_upper = _compute_at_z(
        z_upper, E_arr, nu_arr, T,
        eps_xx_m, eps_yy_m, gamma_xy_m,
        kappa_xx, kappa_yy, kappa_xy,
    )
    result["Stress"][row_arr] = s_upper
    result["Strain"][row_arr] = e_upper

    # --- Mid (0) ---
    z_mid = np.zeros_like(t_arr)
    s_mid, e_mid = _compute_at_z(
        z_mid, E_arr, nu_arr, T,
        eps_xx_m, eps_yy_m, gamma_xy_m,
        kappa_xx, kappa_yy, kappa_xy,
    )
    result["Stress (Mid)"][row_arr] = s_mid
    result["Strain (Mid)"][row_arr] = e_mid

    # --- Lower (-t/2) ---
    z_lower = -t_arr / 2.0
    s_lower, e_lower = _compute_at_z(
        z_lower, E_arr, nu_arr, T,
        eps_xx_m, eps_yy_m, gamma_xy_m,
        kappa_xx, kappa_yy, kappa_xy,
    )
    result["Stress (Lower)"][row_arr] = s_lower
    result["Strain (Lower)"][row_arr] = e_lower

    # --- Membrane (z=0 에서의 순수 면내 성분) ---
    # Mid와 동일 (z=0이므로 bending 기여 없음)
    result["Stress (Membrane)"][row_arr] = s_mid
    result["Strain (Membrane)"][row_arr] = e_mid

    # --- Bending (Upper에서의 순수 굽힘 성분 = Total - Membrane) ---
    result["Stress (Bending)"][row_arr] = s_upper - s_mid
    result["Strain (Bending)"][row_arr] = e_upper - e_mid

    # --- Max Envelope (Upper와 Lower 중 Von Mises가 큰 쪽 선택) ---
    vm_upper = _von_mises_voigt(s_upper)
    vm_lower = _von_mises_voigt(s_lower)
    pick_upper = vm_upper >= vm_lower  # (M_elem,) bool

    s_envelope = np.where(pick_upper[:, None], s_upper, s_lower)
    e_envelope = np.where(pick_upper[:, None], e_upper, e_lower)
    result["Stress (Max Envelope)"][row_arr] = s_envelope
    result["Strain (Max Envelope)"][row_arr] = e_envelope

    return result