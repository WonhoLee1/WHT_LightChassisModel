"""
wht_stress_recovery.py
======================
WHT FEM Framework — Element Stress & Strain Recovery Module

Calculates global Voigt stress/strain tensors at the centroid of elements
from global nodal displacements using highly vectorized NumPy operations.
"""

import numpy as np

class ElementStressRecovery:
    
    @staticmethod
    def recover_quad4(wht_model, u_global_array, sorted_nids):
        """
        Vectorized recovery of centroidal membrane stress/strain for QUAD4 elements.
        
        :param wht_model: WHTMeshModel instance.
        :param u_global_array: (N, 6) global nodal displacements.
        :param sorted_nids: List of node IDs defining the row order.
        :return: (stresses, strains) as (M, 6) global Voigt arrays.
        """
        nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}
        
        M_total = len(wht_model.elements)
        out_stresses = np.zeros((M_total, 6))
        out_strains = np.zeros((M_total, 6))
        
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
            return out_stresses, out_strains, out_strains, out_strains
            
        idx_arr = np.array(node_idx_list)  # (M_quad, 4)
        E_arr = np.array(E_list)           # (M_quad,)
        nu_arr = np.array(nu_list)         # (M_quad,)
        t_arr = np.array(t_list)           # (M_quad,)
        row_arr = np.array(row_map)        # (M_quad,)
        
        # 2. 전역 좌표 및 벡터 변환 (Local Coordinate System)
        c_list = []
        for nid in sorted_nids:
            n = wht_model.nodes[nid]
            c_list.append([n.x, n.y, n.z] if hasattr(n, 'x') else n)
        c_all = np.array(c_list)
        C = c_all[idx_arr]                 # (M_quad, 4, 3)
        U = u_global_array[idx_arr]        # (M_quad, 4, 6)
        
        V1 = C[:, 1, :] - C[:, 0, :]
        V2 = C[:, 3, :] - C[:, 0, :]
        
        norm_V1 = np.linalg.norm(V1, axis=1, keepdims=True)
        X_loc = V1 / norm_V1
        
        Z_raw = np.cross(X_loc, V2)
        norm_Z = np.linalg.norm(Z_raw, axis=1, keepdims=True)
        Z_loc = Z_raw / norm_Z
        Y_loc = np.cross(Z_loc, X_loc)
        
        T = np.stack([X_loc, Y_loc, Z_loc], axis=1) # (M_quad, 3, 3)
        
        # 변위(Translational) 및 회전(Rotational) 성분 각각 로컬로 변환
        U_disp_loc = np.einsum('mij, mkj -> mki', T, U[:, :, :3])
        U_rot_loc  = np.einsum('mij, mkj -> mki', T, U[:, :, 3:6])
        
        # 3. Membrane & Bending 변형률 계산 (요소 윗면 z = +t/2 기준)
        a = norm_V1[:, 0] / 2.0
        b = np.einsum('mi,mi->m', V2, Y_loc) / 2.0  # local y-projection of V2
        
        u_x, u_y = U_disp_loc[:, :, 0], U_disp_loc[:, :, 1]
        th_x, th_y = U_rot_loc[:, :, 0], U_rot_loc[:, :, 1]
        
        dN_dx = np.array([-1, 1, 1, -1]) / (4 * a[:, None])
        dN_dy = np.array([-1, -1, 1, 1]) / (4 * b[:, None])
        
        # Membrane (u,v at mid-plane)
        eps_xx_m = np.sum(dN_dx * u_x, axis=1)
        eps_yy_m = np.sum(dN_dy * u_y, axis=1)
        gamma_xy_m = np.sum(dN_dy * u_x + dN_dx * u_y, axis=1)
        
        # Bending at z = t/2 (Mindlin-Reissner Shell Assumption)
        # Curvature kappa_x = theta_y,x. eps_x = z * kappa_x
        z_dist = t_arr / 2.0
        # Correct Factor: Sum(dN/dx * theta_y) gives the gradient.
        # Ensure units match (mm^-1 * rad = mm^-1).
        eps_xx_b =  z_dist * np.sum(dN_dx * th_y, axis=1)
        eps_yy_b = -z_dist * np.sum(dN_dy * th_x, axis=1)
        gamma_xy_b = z_dist * np.sum(dN_dy * th_y - dN_dx * th_x, axis=1)
        
        eps_xx = eps_xx_m + eps_xx_b
        eps_yy = eps_yy_m + eps_yy_b
        gamma_xy = gamma_xy_m + gamma_xy_b
        
        factor = E_arr / (1 - nu_arr**2)
        sig_xx = factor * (eps_xx + nu_arr * eps_yy)
        sig_yy = factor * (nu_arr * eps_xx + eps_yy)
        sig_xy = factor * ((1 - nu_arr) / 2.0) * gamma_xy
        
        # 4. 로컬 텐서(3x3) 구성 및 전역 공간(Global Tensor)으로 역회전
        # Plane Stress: eps_zz = -nu/(1-nu) * (eps_xx + eps_yy)
        eps_zz = -(nu_arr / (1.0 - nu_arr)) * (eps_xx + eps_yy)
        
        sig_loc, eps_loc = np.zeros((len(E_arr), 3, 3)), np.zeros((len(E_arr), 3, 3))
        sig_loc[:, 0, 0], sig_loc[:, 1, 1], sig_loc[:, 0, 1], sig_loc[:, 1, 0] = sig_xx, sig_yy, sig_xy, sig_xy
        eps_loc[:, 0, 0], eps_loc[:, 1, 1], eps_loc[:, 2, 2] = eps_xx, eps_yy, eps_zz
        eps_loc[:, 0, 1], eps_loc[:, 1, 0] = gamma_xy/2.0, gamma_xy/2.0
        
        sig_glob = np.einsum('mji, mjk, mkl -> mil', T, sig_loc, T)
        eps_glob = np.einsum('mji, mjk, mkl -> mil', T, eps_loc, T)
        
        # 5. Global Voigt Notation (11, 22, 33, 12, 13, 23) 패킹
        # Note: Strain components in Voigt are (e_xx, e_yy, e_zz, gamma_xy, gamma_xz, gamma_yz)
        out_stresses[row_arr, 0], out_stresses[row_arr, 1], out_stresses[row_arr, 2] = sig_glob[:,0,0], sig_glob[:,1,1], sig_glob[:,2,2]
        out_stresses[row_arr, 3], out_stresses[row_arr, 4], out_stresses[row_arr, 5] = sig_glob[:,0,1], sig_glob[:,0,2], sig_glob[:,1,2]
        
        # Prepare individual components for diagnosis
        # Total
        out_strains[row_arr, 0], out_strains[row_arr, 1], out_strains[row_arr, 2] = eps_glob[:,0,0], eps_glob[:,1,1], eps_glob[:,2,2]
        out_strains[row_arr, 3], out_strains[row_arr, 4], out_strains[row_arr, 5] = eps_glob[:,0,1]*2.0, eps_glob[:,0,2]*2.0, eps_glob[:,1,2]*2.0
        
        # Membrane (z=0)
        eps_loc_m = np.zeros_like(eps_loc)
        eps_loc_m[:, 0, 0], eps_loc_m[:, 1, 1] = eps_xx_m, eps_yy_m
        eps_loc_m[:, 0, 1], eps_loc_m[:, 1, 0] = gamma_xy_m/2.0, gamma_xy_m/2.0
        # eps_zz_m = -nu/(1-nu) * (eps_xx_m + eps_yy_m)
        eps_loc_m[:, 2, 2] = -(nu_arr / (1.0 - nu_arr)) * (eps_xx_m + eps_yy_m)
        eps_glob_m = np.einsum('mji, mjk, mkl -> mil', T, eps_loc_m, T)
        
        out_strains_m = np.zeros_like(out_strains)
        out_strains_m[row_arr, 0], out_strains_m[row_arr, 1], out_strains_m[row_arr, 2] = eps_glob_m[:,0,0], eps_glob_m[:,1,1], eps_glob_m[:,2,2]
        out_strains_m[row_arr, 3], out_strains_m[row_arr, 4], out_strains_m[row_arr, 5] = eps_glob_m[:,0,1]*2.0, eps_glob_m[:,0,2]*2.0, eps_glob_m[:,1,2]*2.0
        
        # Bending (z=t/2)
        eps_loc_b = np.zeros_like(eps_loc)
        eps_loc_b[:, 0, 0], eps_loc_b[:, 1, 1] = eps_xx_b, eps_yy_b
        eps_loc_b[:, 0, 1], eps_loc_b[:, 1, 0] = gamma_xy_b/2.0, gamma_xy_b/2.0
        # eps_zz_b = -nu/(1-nu) * (eps_xx_b + eps_yy_b)
        eps_loc_b[:, 2, 2] = -(nu_arr / (1.0 - nu_arr)) * (eps_xx_b + eps_yy_b)
        eps_glob_b = np.einsum('mji, mjk, mkl -> mil', T, eps_loc_b, T)
        
        out_strains_b = np.zeros_like(out_strains)
        out_strains_b[row_arr, 0], out_strains_b[row_arr, 1], out_strains_b[row_arr, 2] = eps_glob_b[:,0,0], eps_glob_b[:,1,1], eps_glob_b[:,2,2]
        out_strains_b[row_arr, 3], out_strains_b[row_arr, 4], out_strains_b[row_arr, 5] = eps_glob_b[:,0,1]*2.0, eps_glob_b[:,0,2]*2.0, eps_glob_b[:,1,2]*2.0

        return out_stresses, out_strains, out_strains_m, out_strains_b

    @staticmethod
    def recover_tria3(wht_model, u_global_array, sorted_nids):
        """
        Vectorized recovery of centroidal membrane stress/strain for TRIA3 (CST) elements.
        
        :param wht_model: WHTMeshModel instance.
        :param u_global_array: (N, 6) global nodal displacements.
        :param sorted_nids: List of node IDs defining the row order.
        :return: (stresses, strains) as (M, 6) global Voigt arrays.
        """
        nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}
        
        M_total = len(wht_model.elements)
        out_stresses = np.zeros((M_total, 6))
        out_strains = np.zeros((M_total, 6))
        
        eid_list, node_idx_list, E_list, nu_list, t_list, row_map = [], [], [], [], [], []
        
        # 1. 필터링 및 프로퍼티 매핑
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
            return out_stresses, out_strains, out_strains, out_strains
            
        idx_arr = np.array(node_idx_list)  # (M_tria, 3)
        E_arr = np.array(E_list)           # (M_tria,)
        nu_arr = np.array(nu_list)         # (M_tria,)
        t_arr = np.array(t_list)           # (M_tria,)
        row_arr = np.array(row_map)        # (M_tria,)
        
        # 2. 전역 좌표 및 벡터 변환 (Local Coordinate System)
        c_list = []
        for nid in sorted_nids:
            n = wht_model.nodes[nid]
            c_list.append([n.x, n.y, n.z] if hasattr(n, 'x') else n)
        c_all = np.array(c_list)
        C = c_all[idx_arr]                 # (M_tria, 3, 3)
        U = u_global_array[idx_arr]        # (M_tria, 3, 6)
        
        V1 = C[:, 1, :] - C[:, 0, :]
        V2 = C[:, 2, :] - C[:, 0, :]
        
        norm_V1 = np.linalg.norm(V1, axis=1, keepdims=True)
        X_loc = V1 / norm_V1
        
        Z_raw = np.cross(X_loc, V2)
        norm_Z = np.linalg.norm(Z_raw, axis=1, keepdims=True)
        Z_loc = Z_raw / norm_Z
        Y_loc = np.cross(Z_loc, X_loc)
        
        T = np.stack([X_loc, Y_loc, Z_loc], axis=1) # (M_tria, 3, 3)
        
        U_disp_loc = np.einsum('mij, mkj -> mki', T, U[:, :, :3])
        U_rot_loc  = np.einsum('mij, mkj -> mki', T, U[:, :, 3:6])
        
        # 3. CST Membrane + Bending (z = +t/2)
        x2 = norm_V1[:, 0]
        # Coordinates of Node 3 relative to local frame P1(0,0), P2(x2, 0)
        x3 = np.sum(V2 * X_loc, axis=1)
        y3 = np.sum(V2 * Y_loc, axis=1)
        
        # Area calculation (Safety guard for near-zero areas)
        two_A = np.maximum(np.abs(x2 * y3), 1e-12)
        
        # CST Derivatives (Standard Linear Triangular Functions)
        dN_dx = np.stack([-y3, y3, np.zeros_like(y3)], axis=1) / two_A[:, None]
        dN_dy = np.stack([x3 - x2, -x3, x2], axis=1) / two_A[:, None]
        
        u_x, u_y = U_disp_loc[:, :, 0], U_disp_loc[:, :, 1]
        th_x, th_y = U_rot_loc[:, :, 0], U_rot_loc[:, :, 1]
        
        # Components Separation
        eps_xx_m = np.sum(dN_dx * u_x, axis=1)
        eps_yy_m = np.sum(dN_dy * u_y, axis=1)
        gamma_xy_m = np.sum(dN_dy * u_x + dN_dx * u_y, axis=1)
        
        eps_xx_b = (t_arr / 2.0) * np.sum(dN_dx * th_y, axis=1)
        eps_yy_b = -(t_arr / 2.0) * np.sum(dN_dy * th_x, axis=1)
        gamma_xy_b = (t_arr / 2.0) * np.sum(dN_dy * th_y - dN_dx * th_x, axis=1)

        eps_xx = eps_xx_m + eps_xx_b
        eps_yy = eps_yy_m + eps_yy_b
        gamma_xy = gamma_xy_m + gamma_xy_b
        
        factor = E_arr / (1 - nu_arr**2)
        sig_xx = factor * (eps_xx + nu_arr * eps_yy)
        sig_yy = factor * (nu_arr * eps_xx + eps_yy)
        sig_xy = factor * ((1 - nu_arr) / 2.0) * gamma_xy
        
        # 4. 로컬 텐서 구성 및 역회전
        # Plane Stress: eps_zz = -nu/(1-nu) * (eps_xx + eps_yy)
        eps_zz = -(nu_arr / (1.0 - nu_arr)) * (eps_xx + eps_yy)
        
        sig_loc, eps_loc = np.zeros((len(E_arr), 3, 3)), np.zeros((len(E_arr), 3, 3))
        sig_loc[:, 0, 0], sig_loc[:, 1, 1], sig_loc[:, 0, 1], sig_loc[:, 1, 0] = sig_xx, sig_yy, sig_xy, sig_xy
        eps_loc[:, 0, 0], eps_loc[:, 1, 1], eps_loc[:, 2, 2] = eps_xx, eps_yy, eps_zz
        eps_loc[:, 0, 1], eps_loc[:, 1, 0] = gamma_xy/2.0, gamma_xy/2.0
        
        sig_glob = np.einsum('mji, mjk, mkl -> mil', T, sig_loc, T)
        eps_glob = np.einsum('mji, mjk, mkl -> mil', T, eps_loc, T)
        
        # 5. Global Voigt Notation 패킹
        out_stresses[row_arr, 0], out_stresses[row_arr, 1], out_stresses[row_arr, 2] = sig_glob[:,0,0], sig_glob[:,1,1], sig_glob[:,2,2]
        out_stresses[row_arr, 3], out_stresses[row_arr, 4], out_stresses[row_arr, 5] = sig_glob[:,0,1], sig_glob[:,0,2], sig_glob[:,1,2]
        
        # Total Strain
        out_strains[row_arr, 0], out_strains[row_arr, 1], out_strains[row_arr, 2] = eps_glob[:,0,0], eps_glob[:,1,1], eps_glob[:,2,2]
        out_strains[row_arr, 3], out_strains[row_arr, 4], out_strains[row_arr, 5] = eps_glob[:,0,1]*2.0, eps_glob[:,0,2]*2.0, eps_glob[:,1,2]*2.0
        
        # Membrane Strain (z=0)
        eps_loc_m = np.zeros_like(eps_loc)
        eps_loc_m[:, 0, 0], eps_loc_m[:, 1, 1] = eps_xx_m, eps_yy_m
        eps_loc_m[:, 0, 1], eps_loc_m[:, 1, 0] = gamma_xy_m/2.0, gamma_xy_m/2.0
        eps_loc_m[:, 2, 2] = -(nu_arr / (1.0 - nu_arr)) * (eps_xx_m + eps_yy_m)
        eps_glob_m = np.einsum('mji, mjk, mkl -> mil', T, eps_loc_m, T)
        
        out_strains_m = np.zeros_like(out_strains)
        out_strains_m[row_arr, 0], out_strains_m[row_arr, 1], out_strains_m[row_arr, 2] = eps_glob_m[:,0,0], eps_glob_m[:,1,1], eps_glob_m[:,2,2]
        out_strains_m[row_arr, 3], out_strains_m[row_arr, 4], out_strains_m[row_arr, 5] = eps_glob_m[:,0,1]*2.0, eps_glob_m[:,0,2]*2.0, eps_glob_m[:,1,2]*2.0
        
        # Bending Strain (z=t/2)
        eps_loc_b = np.zeros_like(eps_loc)
        eps_loc_b[:, 0, 0], eps_loc_b[:, 1, 1] = eps_xx_b, eps_yy_b
        eps_loc_b[:, 0, 1], eps_loc_b[:, 1, 0] = gamma_xy_b/2.0, gamma_xy_b/2.0
        eps_loc_b[:, 2, 2] = -(nu_arr / (1.0 - nu_arr)) * (eps_xx_b + eps_yy_b)
        eps_glob_b = np.einsum('mji, mjk, mkl -> mil', T, eps_loc_b, T)
        
        out_strains_b = np.zeros_like(out_strains)
        out_strains_b[row_arr, 0], out_strains_b[row_arr, 1], out_strains_b[row_arr, 2] = eps_glob_b[:,0,0], eps_glob_b[:,1,1], eps_glob_b[:,2,2]
        out_strains_b[row_arr, 3], out_strains_b[row_arr, 4], out_strains_b[row_arr, 5] = eps_glob_b[:,0,1]*2.0, eps_glob_b[:,0,2]*2.0, eps_glob_b[:,1,2]*2.0

        return out_stresses, out_strains, out_strains_m, out_strains_b