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

_cache_quad4 = {}
_cache_tria3 = {}
_cache_quad4_geom = {}
_cache_quad4_geom_nodal = {}
_cache_tria3_geom = {}
_cache_tria3_geom_nodal = {}

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
            T, dN_dx, dN_dy, C_loc, D_loc = _cache_quad4_geom[cache_key_geom]
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
                _cache_quad4_geom[cache_key_geom] = (T, dN_dx, dN_dy, C_loc, D_loc)

        # 변위/회전 → 로컬 변환
        U_disp_loc = np.einsum('mij, mkj -> mki', T, U[:, :, :3])
        U_rot_loc = np.einsum('mij, mkj -> mki', T, U[:, :, 3:6])

        u_x, u_y = U_disp_loc[:, :, 0], U_disp_loc[:, :, 1]
        th_x, th_y = U_rot_loc[:, :, 0], U_rot_loc[:, :, 1]

        # Membrane 변형률 (z-independent)
        eps_xx_m = np.sum(dN_dx * u_x, axis=1)
        eps_yy_m = np.sum(dN_dy * u_y, axis=1)

        # MITC4+ gamma_xy: Ko-Lee-Bathe 2016 Eqn 27c
        # 특성 기하 벡터 (요소별 스칼라, (M,))
        XR0 = 0.25*(-C_loc[:,0]+C_loc[:,1]+C_loc[:,2]-C_loc[:,3])
        XR1 = 0.25*(-D_loc[:,0]+D_loc[:,1]+D_loc[:,2]-D_loc[:,3])
        XS0 = 0.25*(-C_loc[:,0]-C_loc[:,1]+C_loc[:,2]+C_loc[:,3])
        XS1 = 0.25*(-D_loc[:,0]-D_loc[:,1]+D_loc[:,2]+D_loc[:,3])
        XD0 = 0.25*(+C_loc[:,0]-C_loc[:,1]+C_loc[:,2]-C_loc[:,3])
        XD1 = 0.25*(+D_loc[:,0]-D_loc[:,1]+D_loc[:,2]-D_loc[:,3])
        det_RS = XR0*XS1 - XR1*XS0
        det_RS_safe = np.where(np.abs(det_RS)>1e-12, det_RS, 1e-12)
        c_r = (XD0*XS1 - XD1*XS0) / det_RS_safe
        c_s = (XD1*XR0 - XD0*XR1) / det_RS_safe
        d_val = c_r**2 + c_s**2 - 1.0
        use_mitc = np.abs(d_val) > 1e-10
        d_safe = np.where(use_mitc, d_val, 1.0)
        a_A = c_r*(c_r-1.0)/(2.0*d_safe)
        a_B = c_r*(c_r+1.0)/(2.0*d_safe)
        a_C = c_s*(c_s-1.0)/(2.0*d_safe)
        a_D = c_s*(c_s+1.0)/(2.0*d_safe)
        a_E = 2.0*c_r*c_s/d_safe

        def _dN_loc(xi, eta):
            dxi = np.array([-(1-eta),(1-eta),(1+eta),-(1+eta)])*0.25
            det = np.array([-(1-xi),-(1+xi),(1+xi),(1-xi)])*0.25
            j11 = C_loc@dxi; j12 = D_loc@dxi
            j21 = C_loc@det; j22 = D_loc@det
            dj = j11*j22-j12*j21
            dj_s = np.where(np.abs(dj)>1e-12, dj, 1e-12)
            i11=j22/dj_s; i12=-j12/dj_s; i21=-j21/dj_s; i22=j11/dj_s
            return np.outer(i11,dxi)+np.outer(i12,det), np.outer(i21,dxi)+np.outer(i22,det)

        dx_A, dy_A = _dN_loc(0.0,  1.0)   # A
        dx_B, dy_B = _dN_loc(0.0, -1.0)   # B
        dx_C, dy_C = _dN_loc(1.0,  0.0)   # C
        dx_D, dy_D = _dN_loc(-1.0, 0.0)   # D
        dx_E, dy_E = _dN_loc(0.0,  0.0)   # E = centroid

        # Eqn 27c 타잉점 기여: A,B → exx 행, C,D → eyy 행, E → exy 행
        exx_A = np.sum(dx_A*u_x, axis=1)
        exx_B = np.sum(dx_B*u_x, axis=1)
        eyy_C = np.sum(dy_C*u_y, axis=1)
        eyy_D = np.sum(dy_D*u_y, axis=1)
        gxy_E = np.sum(dy_E*u_x + dx_E*u_y, axis=1)

        gp = 1.0/np.sqrt(3.0)
        gamma_xy_m_mitc = np.zeros(len(E_arr))
        for R, S in ((-gp,-gp),(gp,-gp),(gp,gp),(-gp,gp)):
            wA = 0.25*(R + 4.0*a_A*R*S)
            wB = 0.25*(-R + 4.0*a_B*R*S)
            wC = 0.25*(S + 4.0*a_C*R*S)
            wD = 0.25*(-S + 4.0*a_D*R*S)
            wE = 1.0 + a_E*R*S
            gamma_xy_m_mitc += wA*exx_A + wB*exx_B + wC*eyy_C + wD*eyy_D + wE*gxy_E
        gamma_xy_m_mitc /= 4.0

        gamma_xy_m_std = np.sum(dN_dy*u_x + dN_dx*u_y, axis=1)
        gamma_xy_m = np.where(use_mitc, gamma_xy_m_mitc, gamma_xy_m_std)

        # Curvature (bending gradients, z-independent)
        kappa_xx = np.sum(dN_dx * th_y, axis=1)          # ∂θ_y/∂x
        kappa_yy = -np.sum(dN_dy * th_x, axis=1)         # -∂θ_x/∂y
        kappa_xy = np.sum(dN_dy * th_y - dN_dx * th_x, axis=1)

        res_p = _compute_all_layers(
            M_total, row_arr, E_arr, nu_arr, t_arr, T,
            eps_xx_m, eps_yy_m, gamma_xy_m,
            kappa_xx, kappa_yy, kappa_xy,
        )
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

            res_p = _compute_all_layers(
                M_total, row_arr, E_arr, nu_arr, t_arr, T,
                eps_xx_m_all[:, p], eps_yy_m_all[:, p], gamma_xy_m_all[:, p],
                kappa_xx_all[:, p], kappa_yy_all[:, p], kappa_xy_all[:, p],
                fields=fields
            )
            
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
        res_centroid = ElementStressRecovery.recover_tria3(wht_model, u_global_array, sorted_nids, c_all=c_all, fields=fields)
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

        res_p = _compute_all_layers(
            M_total, row_arr, E_arr, nu_arr, t_arr, T,
            eps_xx_m, eps_yy_m, gamma_xy_m,
            kappa_xx, kappa_yy, kappa_xy,
            fields=fields
        )
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
    T_T = T.transpose(0, 2, 1)
    sig_glob = np.matmul(T_T, np.matmul(sig_loc, T))
    eps_glob = np.matmul(T_T, np.matmul(eps_loc, T))

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
    fields: Optional[list] = None
) -> Dict[str, np.ndarray]:
    """
    Upper/Mid/Lower 3개 적분점 및 Membrane/Bending 분리, Max Envelope을 계산합니다.
    """
    result = {}

    need_mid = fields is None or any("Mid" in f or "Membrane" in f or "Bending" in f for f in fields)
    need_lower = fields is None or any("Lower" in f or "Max Envelope" in f for f in fields)

    # --- Upper (+t/2) ---
    z_upper = t_arr / 2.0
    s_upper, e_upper = _compute_at_z(
        z_upper, E_arr, nu_arr, T,
        eps_xx_m, eps_yy_m, gamma_xy_m,
        kappa_xx, kappa_yy, kappa_xy,
    )
    if fields is None or "Stress" in fields:
        result["Stress"] = s_upper
    if fields is None or "Strain" in fields:
        result["Strain"] = e_upper

    # --- Mid (0) ---
    s_mid, e_mid = None, None
    if need_mid:
        z_mid = np.zeros_like(t_arr)
        s_mid, e_mid = _compute_at_z(
            z_mid, E_arr, nu_arr, T,
            eps_xx_m, eps_yy_m, gamma_xy_m,
            kappa_xx, kappa_yy, kappa_xy,
        )
        if fields is None or "Stress (Mid)" in fields:
            result["Stress (Mid)"] = s_mid
        if fields is None or "Strain (Mid)" in fields:
            result["Strain (Mid)"] = e_mid
        if fields is None or "Stress (Membrane)" in fields:
            result["Stress (Membrane)"] = s_mid
        if fields is None or "Strain (Membrane)" in fields:
            result["Strain (Membrane)"] = e_mid
        if fields is None or "Stress (Bending)" in fields:
            result["Stress (Bending)"] = s_upper - s_mid
        if fields is None or "Strain (Bending)" in fields:
            result["Strain (Bending)"] = e_upper - e_mid

    # --- Lower (-t/2) ---
    s_lower, e_lower = None, None
    if need_lower:
        z_lower = -t_arr / 2.0
        s_lower, e_lower = _compute_at_z(
            z_lower, E_arr, nu_arr, T,
            eps_xx_m, eps_yy_m, gamma_xy_m,
            kappa_xx, kappa_yy, kappa_xy,
        )
        if fields is None or "Stress (Lower)" in fields:
            result["Stress (Lower)"] = s_lower
        if fields is None or "Strain (Lower)" in fields:
            result["Strain (Lower)"] = e_lower

    # --- Max Envelope ---
    if fields is None or "Stress (Max Envelope)" in fields or "Strain (Max Envelope)" in fields:
        if s_lower is not None:
            vm_upper = _von_mises_voigt(s_upper)
            vm_lower = _von_mises_voigt(s_lower)
            pick_upper = vm_upper >= vm_lower
            if fields is None or "Stress (Max Envelope)" in fields:
                result["Stress (Max Envelope)"] = np.where(pick_upper[:, None], s_upper, s_lower)
            if fields is None or "Strain (Max Envelope)" in fields:
                result["Strain (Max Envelope)"] = np.where(pick_upper[:, None], e_upper, e_lower)
        else:
            if fields is None or "Stress (Max Envelope)" in fields:
                result["Stress (Max Envelope)"] = s_upper
            if fields is None or "Strain (Max Envelope)" in fields:
                result["Strain (Max Envelope)"] = e_upper

    return result

    return result