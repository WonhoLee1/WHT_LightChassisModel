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
import numba
from numba import njit
from typing import Dict, Tuple, List, Optional


# ---------------------------------------------------------------------------
# Through-Thickness Integration Points 정의
# ---------------------------------------------------------------------------
# z_ratio: -1.0 = Lower (-t/2), 0.0 = Mid (0), +1.0 = Upper (+t/2)
DEFAULT_Z_RATIOS = {"lower": -1.0, "mid": 0.0, "upper": 1.0}

_cache_quad4 = {}
_cache_tria3 = {}
_cache_quad4_geom = {}
_cache_quad4_geom_nodal = {}
_cache_tria3_geom = {}
_cache_tria3_geom_nodal = {}

class ElementStressRecoveryNumba:
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
        c_all: Optional[np.ndarray] = None
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

        cache_key = id(wht_model)
        if cache_key in _cache_quad4:
            idx_arr, E_arr, nu_arr, t_arr, row_arr, eid_list = _cache_quad4[cache_key]
        else:
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

            idx_arr = np.array(node_idx_list)
            E_arr = np.array(E_list)
            nu_arr = np.array(nu_list)
            t_arr = np.array(t_list)
            row_arr = np.array(row_map)
            _cache_quad4[cache_key] = (idx_arr, E_arr, nu_arr, t_arr, row_arr, eid_list)

        if len(eid_list) == 0:
            return _empty_result_dict(M_total)

        # 2. 로컬 좌표계 구성
        if c_all is None:
            c_list = []
            for nid in sorted_nids:
                n = wht_model.nodes[nid]
                c_list.append([n.x, n.y, n.z] if hasattr(n, 'x') else n)
            c_all = np.array(c_list)
        C = c_all[idx_arr]     # (M_quad, 4, 3)
        U = u_global_array[idx_arr]  # (M_quad, 4, 6)

        cache_key_geom = (id(wht_model), id(c_all)) if c_all is not None else None
        if cache_key_geom and cache_key_geom in _cache_quad4_geom:
            T, dN_dx, dN_dy = _cache_quad4_geom[cache_key_geom]
        else:
            V1 = C[:, 1, :] - C[:, 0, :]
            V2 = C[:, 3, :] - C[:, 0, :]

            norm_V1 = np.linalg.norm(V1, axis=1, keepdims=True)
            X_loc = V1 / norm_V1

            Z_raw = np.cross(X_loc, V2)
            norm_Z = np.linalg.norm(Z_raw, axis=1, keepdims=True)
            Z_loc = Z_raw / norm_Z
            Y_loc = np.cross(Z_loc, X_loc)

            T = np.stack([X_loc, Y_loc, Z_loc], axis=1)  # (M_quad, 3, 3)

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

            if cache_key_geom:
                _cache_quad4_geom[cache_key_geom] = (T, dN_dx, dN_dy)

        # 변위/회전 → 로컬 변환
        U_disp_loc = np.einsum('mij, mkj -> mki', T, U[:, :, :3])
        U_rot_loc = np.einsum('mij, mkj -> mki', T, U[:, :, 3:6])

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

        s_upper, e_upper, s_mid, e_mid, s_lower, e_lower, s_max_env, e_max_env = _compute_all_layers_numba(
            E_arr, nu_arr, t_arr, T,
            eps_xx_m, eps_yy_m, gamma_xy_m,
            kappa_xx, kappa_yy, kappa_xy,
        )
        res_p = {
            'Stress': s_upper, 'Strain': e_upper,
            'Stress (Mid)': s_mid, 'Strain (Mid)': e_mid,
            'Stress (Lower)': s_lower, 'Strain (Lower)': e_lower,
            'Stress (Max Envelope)': s_max_env, 'Strain (Max Envelope)': e_max_env,
            'Stress (Membrane)': s_mid, 'Strain (Membrane)': e_mid,
            'Stress (Bending)': s_upper - s_mid, 'Strain (Bending)': e_upper - e_mid
        }
        result_dict = _empty_result_dict(M_total)
        for k in res_p:
            result_dict[k][row_arr, :] = res_p[k]
        return result_dict


    @staticmethod
    def recover_quad4_nodal(
        wht_model,
        u_global_array: np.ndarray,
        sorted_nids: list,
        c_all: Optional[np.ndarray] = None,
        fields: Optional[list] = None
    ) -> dict:
        """
        QUAD4 요소의 4개 코너 노드 위치에서 응력/변형률을 복원합니다.
        반환 형태는 Dict[str, np.ndarray] 이며 값의 shape은 (M_total, 4, 6) 입니다.
        """
        nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}
        M_total = len(wht_model.elements)
        cache_key = id(wht_model)
        if cache_key in _cache_quad4:
            idx_arr, E_arr, nu_arr, t_arr, row_arr, eid_list = _cache_quad4[cache_key]
        else:
            eid_list, node_idx_list, E_list, nu_list, t_list, row_map = [], [], [], [], [], []

            for i, eid in enumerate(sorted(wht_model.elements.keys())):
                elem = wht_model.elements[eid]
                is_obj = hasattr(elem, 'type')
                etype = elem.type if is_obj else wht_model.element_types.get(eid, "QUAD4")
                node_ids = elem.node_ids if is_obj else elem

                if etype in ["QUAD", "QUAD4"] and len(node_ids) == 4:
                    eid_list.append(eid)
                    node_idx_list.append([nid_to_idx[n] for n in node_ids])
                    pid = getattr(elem, 'pid', 0) if hasattr(elem, 'pid') else 0
                    E, nu, t = 210000.0, 0.3, 1.0
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

            idx_arr = np.array(node_idx_list)
            E_arr = np.array(E_list)
            nu_arr = np.array(nu_list)
            t_arr = np.array(t_list)
            row_arr = np.array(row_map)
            _cache_quad4[cache_key] = (idx_arr, E_arr, nu_arr, t_arr, row_arr, eid_list)

        if len(eid_list) == 0:
            z = np.zeros((M_total, 4, 6))
            return {k: z.copy() for k in [
                "Stress", "Stress (Mid)", "Stress (Lower)", "Stress (Membrane)", "Stress (Bending)", "Stress (Max Envelope)",
                "Strain", "Strain (Mid)", "Strain (Lower)", "Strain (Membrane)", "Strain (Bending)", "Strain (Max Envelope)"
            ]}

        if c_all is None:
            c_list = [[wht_model.nodes[nid].x, wht_model.nodes[nid].y, wht_model.nodes[nid].z] for nid in sorted_nids]
            c_all = np.array(c_list)
        C = c_all[idx_arr]
        U = u_global_array[idx_arr]

        cache_key_geom = (id(wht_model), id(c_all)) if c_all is not None else None
        if cache_key_geom and cache_key_geom in _cache_quad4_geom_nodal:
            T, dN_dx_all, dN_dy_all = _cache_quad4_geom_nodal[cache_key_geom]
        else:
            V1 = C[:, 1, :] - C[:, 0, :]
            V2 = C[:, 3, :] - C[:, 0, :]
            X_loc = V1 / np.linalg.norm(V1, axis=1, keepdims=True)
            Z_raw = np.cross(X_loc, V2)
            Z_loc = Z_raw / np.linalg.norm(Z_raw, axis=1, keepdims=True)
            Y_loc = np.cross(Z_loc, X_loc)
            T = np.stack([X_loc, Y_loc, Z_loc], axis=1)

            p0 = C[:, 0, :]
            C_loc = np.stack([np.einsum('mi,mi->m', C[:, k, :] - p0, X_loc) for k in range(4)], axis=1)
            D_loc = np.stack([np.einsum('mi,mi->m', C[:, k, :] - p0, Y_loc) for k in range(4)], axis=1)

            xi_pts = np.array([-1.0, 1.0, 1.0, -1.0])
            eta_pts = np.array([-1.0, -1.0, 1.0, 1.0])

            dN_dx_all = []
            dN_dy_all = []

            for p in range(4):
                xi, eta = xi_pts[p], eta_pts[p]
                dNxi = np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)]) * 0.25
                dNeta = np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)]) * 0.25

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

                dN_dx_all.append(np.outer(invJ11, dNxi) + np.outer(invJ12, dNeta))
                dN_dy_all.append(np.outer(invJ21, dNxi) + np.outer(invJ22, dNeta))

            if cache_key_geom:
                _cache_quad4_geom_nodal[cache_key_geom] = (T, dN_dx_all, dN_dy_all)

        U_disp_loc = np.einsum('mij, mkj -> mki', T, U[:, :, :3])
        U_rot_loc = np.einsum('mij, mkj -> mki', T, U[:, :, 3:6])

        eps_xx_m_all = np.zeros((len(E_arr), 4))
        eps_yy_m_all = np.zeros((len(E_arr), 4))
        gamma_xy_m_all = np.zeros((len(E_arr), 4))
        kappa_xx_all = np.zeros((len(E_arr), 4))
        kappa_yy_all = np.zeros((len(E_arr), 4))
        kappa_xy_all = np.zeros((len(E_arr), 4))

        u_x, u_y = U_disp_loc[:, :, 0], U_disp_loc[:, :, 1]
        th_x, th_y = U_rot_loc[:, :, 0], U_rot_loc[:, :, 1]

        for p in range(4):
            dN_dx = dN_dx_all[p]
            dN_dy = dN_dy_all[p]

            eps_xx_m_all[:, p] = np.sum(dN_dx * u_x, axis=1)
            eps_yy_m_all[:, p] = np.sum(dN_dy * u_y, axis=1)
            gamma_xy_m_all[:, p] = np.sum(dN_dy * u_x + dN_dx * u_y, axis=1)
            
            kappa_xx_all[:, p] = np.sum(dN_dx * th_y, axis=1)
            kappa_yy_all[:, p] = -np.sum(dN_dy * th_x, axis=1)
            kappa_xy_all[:, p] = np.sum(dN_dy * th_y - dN_dx * th_x, axis=1)

        result_dict = {}
        for p in range(4):
            # U_rot_loc의 미분 (p번째 적분점)
            dN_dx, dN_dy = dN_dx_all[p], dN_dy_all[p]

            s_upper, e_upper, s_mid, e_mid, s_lower, e_lower, s_max_env, e_max_env = _compute_all_layers_numba(
                E_arr, nu_arr, t_arr, T,
                eps_xx_m_all[:, p], eps_yy_m_all[:, p], gamma_xy_m_all[:, p],
                kappa_xx_all[:, p], kappa_yy_all[:, p], kappa_xy_all[:, p],
            )
            res_p_raw = {
                'Stress': s_upper, 'Strain': e_upper,
                'Stress (Mid)': s_mid, 'Strain (Mid)': e_mid,
                'Stress (Lower)': s_lower, 'Strain (Lower)': e_lower,
                'Stress (Max Envelope)': s_max_env, 'Strain (Max Envelope)': e_max_env,
                'Stress (Membrane)': s_mid, 'Strain (Membrane)': e_mid,
                'Stress (Bending)': s_upper - s_mid, 'Strain (Bending)': e_upper - e_mid
            }
            res_p = {k: v for k, v in res_p_raw.items() if fields is None or k in fields}
            
            if not result_dict:
                for k in res_p:
                    result_dict[k] = np.zeros((M_total, 4, 6), dtype=np.float32)

            for k in res_p:
                result_dict[k][row_arr, p, :] = res_p[k]

        return result_dict



    @staticmethod
    def recover_tria3_nodal(
        wht_model,
        u_global_array: np.ndarray,
        sorted_nids: list,
        c_all: Optional[np.ndarray] = None,
        fields: Optional[list] = None
    ) -> dict:
        """
        TRIA3 (CST) 요소의 3개 코너 노드 위치에서 응력/변형률을 복원합니다.
        CST는 곡률이 없으므로 Centroid 값을 복제하여 반환합니다.
        """
        res_centroid = ElementStressRecoveryNumba.recover_tria3(wht_model, u_global_array, sorted_nids, c_all=c_all, fields=fields)
        M_total = len(wht_model.elements)
        result_dict = {}
        for k, v in res_centroid.items():
            arr_3 = np.zeros((M_total, 3, 6))
            for p in range(3):
                arr_3[:, p, :] = v
            result_dict[k] = arr_3
        return result_dict

    @staticmethod
    def recover_tria3(
        wht_model,
        u_global_array: np.ndarray,
        sorted_nids: List[int],
        c_all: Optional[np.ndarray] = None,
        fields: Optional[list] = None
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

        cache_key = id(wht_model)
        if cache_key in _cache_tria3:
            idx_arr, E_arr, nu_arr, t_arr, row_arr, eid_list = _cache_tria3[cache_key]
        else:
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

            idx_arr = np.array(node_idx_list)
            E_arr = np.array(E_list)
            nu_arr = np.array(nu_list)
            t_arr = np.array(t_list)
            row_arr = np.array(row_map)
            _cache_tria3[cache_key] = (idx_arr, E_arr, nu_arr, t_arr, row_arr, eid_list)

        if len(eid_list) == 0:
            return _empty_result_dict(M_total)

        # 로컬 좌표계 구성
        if c_all is None:
            c_list = []
            for nid in sorted_nids:
                n = wht_model.nodes[nid]
                c_list.append([n.x, n.y, n.z] if hasattr(n, 'x') else n)
            c_all = np.array(c_list)
        C = c_all[idx_arr]
        U = u_global_array[idx_arr]

        cache_key_geom = (id(wht_model), id(c_all)) if c_all is not None else None
        if cache_key_geom and cache_key_geom in _cache_tria3_geom:
            T, dN_dx, dN_dy = _cache_tria3_geom[cache_key_geom]
        else:
            V1 = C[:, 1, :] - C[:, 0, :]
            V2 = C[:, 2, :] - C[:, 0, :]

            norm_V1 = np.linalg.norm(V1, axis=1, keepdims=True)
            X_loc = V1 / norm_V1

            Z_raw = np.cross(X_loc, V2)
            norm_Z = np.linalg.norm(Z_raw, axis=1, keepdims=True)
            Z_loc = Z_raw / norm_Z
            Y_loc = np.cross(Z_loc, X_loc)

            T = np.stack([X_loc, Y_loc, Z_loc], axis=1)

            # CST 형상함수 미분
            x2 = norm_V1[:, 0]
            x3 = np.sum(V2 * X_loc, axis=1)
            y3 = np.sum(V2 * Y_loc, axis=1)

            two_A = np.maximum(np.abs(x2 * y3), 1e-12)

            dN_dx = np.stack([-y3, y3, np.zeros_like(y3)], axis=1) / two_A[:, None]
            dN_dy = np.stack([x3 - x2, -x3, x2], axis=1) / two_A[:, None]

            if cache_key_geom:
                _cache_tria3_geom[cache_key_geom] = (T, dN_dx, dN_dy)

        U_disp_loc = np.einsum('mij, mkj -> mki', T, U[:, :, :3])
        U_rot_loc = np.einsum('mij, mkj -> mki', T, U[:, :, 3:6])

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

        s_upper, e_upper, s_mid, e_mid, s_lower, e_lower, s_max_env, e_max_env = _compute_all_layers_numba(
            E_arr, nu_arr, t_arr, T,
            eps_xx_m, eps_yy_m, gamma_xy_m,
            kappa_xx, kappa_yy, kappa_xy,
        )
        res_p_raw = {
            'Stress': s_upper, 'Strain': e_upper,
            'Stress (Mid)': s_mid, 'Strain (Mid)': e_mid,
            'Stress (Lower)': s_lower, 'Strain (Lower)': e_lower,
            'Stress (Max Envelope)': s_max_env, 'Strain (Max Envelope)': e_max_env,
            'Stress (Membrane)': s_mid, 'Strain (Membrane)': e_mid,
            'Stress (Bending)': s_upper - s_mid, 'Strain (Bending)': e_upper - e_mid
        }
        res_p = {k: v for k, v in res_p_raw.items() if fields is None or k in fields}
        result_dict = {}
        for k in res_p:
            result_dict[k] = np.zeros((M_total, 6), dtype=np.float32)
            result_dict[k][row_arr, :] = res_p[k]
        return result_dict


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


@njit(cache=True)
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
):
    M = len(z_dist)
    stress_voigt = np.empty((M, 6), dtype=np.float64)
    strain_voigt = np.empty((M, 6), dtype=np.float64)

    for i in range(M):
        exx = eps_xx_m[i] + z_dist[i] * kappa_xx[i]
        eyy = eps_yy_m[i] + z_dist[i] * kappa_yy[i]
        gxy = gamma_xy_m[i] + z_dist[i] * kappa_xy[i]

        E = E_arr[i]
        nu = nu_arr[i]

        factor = E / (1.0 - nu**2)
        sxx = factor * (exx + nu * eyy)
        syy = factor * (nu * exx + eyy)
        sxy = factor * ((1.0 - nu) / 2.0) * gxy

        ezz = -(nu / (1.0 - nu)) * (exx + eyy)

        sig_loc = np.array([
            [sxx, sxy, 0.0],
            [sxy, syy, 0.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float64)

        eps_loc = np.array([
            [exx, gxy/2.0, 0.0],
            [gxy/2.0, eyy, 0.0],
            [0.0, 0.0, ezz]
        ], dtype=np.float64)

        Ti = T[i]
        Ti_T = Ti.T
        
        # Matrix multiply: Ti_T @ sig_loc @ Ti
        tmp_sig = np.dot(sig_loc, Ti)
        sig_glob = np.dot(Ti_T, tmp_sig)
        
        tmp_eps = np.dot(eps_loc, Ti)
        eps_glob = np.dot(Ti_T, tmp_eps)

        stress_voigt[i, 0] = sig_glob[0, 0]
        stress_voigt[i, 1] = sig_glob[1, 1]
        stress_voigt[i, 2] = sig_glob[2, 2]
        stress_voigt[i, 3] = sig_glob[0, 1]
        stress_voigt[i, 4] = sig_glob[0, 2]
        stress_voigt[i, 5] = sig_glob[1, 2]

        strain_voigt[i, 0] = eps_glob[0, 0]
        strain_voigt[i, 1] = eps_glob[1, 1]
        strain_voigt[i, 2] = eps_glob[2, 2]
        strain_voigt[i, 3] = eps_glob[0, 1] * 2.0
        strain_voigt[i, 4] = eps_glob[0, 2] * 2.0
        strain_voigt[i, 5] = eps_glob[1, 2] * 2.0

    return stress_voigt, strain_voigt



@njit(cache=True)
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


@njit(cache=True)
def _compute_all_layers_numba(
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
):
    z_upper = t_arr / 2.0
    s_upper, e_upper = _compute_at_z(
        z_upper, E_arr, nu_arr, T,
        eps_xx_m, eps_yy_m, gamma_xy_m,
        kappa_xx, kappa_yy, kappa_xy,
    )

    z_mid = np.zeros_like(t_arr)
    s_mid, e_mid = _compute_at_z(
        z_mid, E_arr, nu_arr, T,
        eps_xx_m, eps_yy_m, gamma_xy_m,
        kappa_xx, kappa_yy, kappa_xy,
    )

    z_lower = -t_arr / 2.0
    s_lower, e_lower = _compute_at_z(
        z_lower, E_arr, nu_arr, T,
        eps_xx_m, eps_yy_m, gamma_xy_m,
        kappa_xx, kappa_yy, kappa_xy,
    )

    vm_upper = _von_mises_voigt(s_upper)
    vm_lower = _von_mises_voigt(s_lower)
    
    # numba equivalent of np.where with broadcast
    N = len(vm_upper)
    s_max_env = np.empty_like(s_upper)
    e_max_env = np.empty_like(e_upper)
    for i in range(N):
        if vm_upper[i] >= vm_lower[i]:
            s_max_env[i] = s_upper[i]
            e_max_env[i] = e_upper[i]
        else:
            s_max_env[i] = s_lower[i]
            e_max_env[i] = e_lower[i]

    return s_upper, e_upper, s_mid, e_mid, s_lower, e_lower, s_max_env, e_max_env