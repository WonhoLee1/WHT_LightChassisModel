"""
wht_tria3_element.py
====================
WHT FEM Framework — Industrial-Grade MITC3+ Shell Element (18 DOF)
Based on Bathe et al. (2014) and IvyFEM Specifications.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import pinv

def tri_area(c1, c2, c3):
    """Calculates triangle area using cross-product for maximum consistency."""
    return 0.5 * np.linalg.norm(np.cross(c2 - c1, c3 - c1))

def _local_csys(c1, c2, c3):
    """Calculates local basis and side lengths relative to node 1."""
    v12 = c2 - c1; v13 = c3 - c1
    x_loc = v12 / (np.linalg.norm(v12) + 1e-15)
    z_raw = np.cross(x_loc, v13)
    z_loc = z_raw / (np.linalg.norm(z_raw) + 1e-15)
    y_loc = np.cross(z_loc, x_loc)
    T = np.stack([x_loc, y_loc, z_loc], axis=0) # Basis matrix
    
    # Coordinates relative to node 1
    x2 = np.dot(v12, x_loc)
    x3 = np.dot(v13, x_loc); y3 = np.dot(v13, y_loc)
    area = 0.5 * x2 * y3
    return T, x2, x3, y3, area

def _element_K_tria3(c1, c2, c3, t, E, nu):
    T, x2, x3, y3, A = _local_csys(c1, c2, c3)
    if A < 1e-10: return np.zeros((18, 18)) # Guard for degenerate mesh
    
    G = E / (2.0 * (1.0 + nu)); k_shear = 5.0 / 6.0; inv_nu2 = 1.0 / (1.0 - nu**2)
    
    # 1. Membrane (CST): Standard [u, v]
    # Robust dN/dx, dN/dy for general triangle coordinates
    # N1 = 1 - xi - eta, N2 = xi, N3 = eta
    # x = x2*xi + x3*eta, y = y2*xi + y3*eta
    # J = x2*y3 - x3*y2 = 2*A (Note: y1=x1=0, y2=0 in local frame)
    y2 = 0.0
    invJ = 1.0 / (2.0 * A)
    # dN/dx = [-(y3-y2), y3, -y2] * invJ
    # dN/dy = [(x3-x2), -x3, x2] * invJ
    dN_dx = np.array([-(y3-y2), y3, -y2]) * invJ
    dN_dy = np.array([(x3-x2), -x3, x2]) * invJ

    Dm = (E * t * inv_nu2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Bm = np.zeros((3, 18))
    for i in range(3):
        Bm[0, 6*i]   = dN_dx[i]
        Bm[1, 6*i+1] = dN_dy[i]
        Bm[2, 6*i]   = dN_dy[i]
        Bm[2, 6*i+1] = dN_dx[i]
    K_loc = (Bm.T @ Dm @ Bm) * A

    K_full = np.zeros((21, 21))
    Db = (E * t**3 / 12.0 * inv_nu2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Ds = (k_shear * G * t) * np.eye(2)
    
    # Pre-calculate Tying Operators at mid-edges
    # P1: (0.5, 0), P2: (0, 0.5), P3: (0.5, 0.5)
    def get_tying_B(xi, eta, edge_vec):
        # Calculates covariant shear strain component along edge_vec
        # gamma_v = v_x * gamma_xz + v_y * gamma_yz
        # On flat plate: gamma_v = del_v(w) + v_x * theta_y - v_y * theta_x
        L = np.array([1-xi-eta, xi, eta])
        B = np.zeros(21)
        # 1. w part: gradient of w along edge_vec
        # w_v = (w_j - w_i) for edge ij
        # But we need consistent operator for any node i,j,k
        # Standard MITC3 tying at mid-edges:
        if xi == 0.5 and eta == 0.0: # Edge 1-2
            B[6*0+2] = -1.0; B[6*1+2] = 1.0
            B[6*0+4] = 0.5*x2; B[6*1+4] = 0.5*x2
        elif xi == 0.0 and eta == 0.5: # Edge 1-3
            B[6*0+2] = -1.0; B[6*2+2] = 1.0
            B[6*0+3] = -0.5*y3; B[6*2+3] = -0.5*y3
            B[6*0+4] = 0.5*x3;  B[6*2+4] = 0.5*x3
        elif xi == 0.5 and eta == 0.5: # Edge 2-3
            B[6*1+2] = -1.0; B[6*2+2] = 1.0
            B[6*1+3] = -0.5*y3; B[6*2+3] = -0.5*y3
            B[6*1+4] = 0.5*(x3-x2); B[6*2+4] = 0.5*(x3-x2)
        return B

    Bx1 = get_tying_B(0.5, 0.0, None) # gamma_rz(P1)
    Be2 = get_tying_B(0.0, 0.5, None) # gamma_sz(P2)
    Bq3 = get_tying_B(0.5, 0.5, None) # gamma_qz(P3)
    
    # c coefficient: c = gamma_sz(P2) - gamma_rz(P1) - gamma_qz(P3)
    B_c = Be2 - Bx1 - Bq3
    
    # 3-point Gauss Integration
    gauss_pts = [(1/6, 1/6), (2/3, 1/6), (1/6, 2/3)]
    for xi, eta in gauss_pts:
        phi_b = 27.0 * xi * eta * (1-xi-eta)
        dphi_dxi = 27.0 * (eta - 2*xi*eta - eta**2)
        dphi_deta = 27.0 * (xi - xi**2 - 2*xi*eta)
        dphi_dx = (dphi_dxi * y3 + dphi_deta * (x2-x3)) / (x2*y3)
        dphi_dy = (dphi_deta * x2) / (x2*y3) # Corrected mapping
        
        # Bending
        Bb = np.zeros((3, 21))
        for i in range(3):
            Bb[0, 6*i+4] = dN_dx[i]; Bb[1, 6*i+3] = -dN_dy[i]; Bb[2, 6*i+3] = -dN_dx[i]; Bb[2, 6*i+4] = dN_dy[i]
        Bb[0, 20] = dphi_dx; Bb[1, 19] = -dphi_dy; Bb[2, 19] = -dphi_dx; Bb[2, 20] = dphi_dy
        
        # MITC3 Interpolated Covariant Strains
        B_gamma_rz = Bx1 + B_c * eta
        B_gamma_sz = Be2 - B_c * xi
        
        # Mapping back to Cartesian [gamma_xz, gamma_yz]
        # {gamma_x, gamma_y} = J^-T * {gamma_rz, gamma_sz}
        B_gamma_hat = np.zeros((2, 21))
        B_gamma_hat[0, :] = (y3 * B_gamma_rz - x3 * B_gamma_sz) / (x2*y3)
        B_gamma_hat[1, :] = (x2 * B_gamma_sz) / (x2*y3)
        
        # Add Bubble contribution
        B_gamma_hat[0, 18] += dphi_dx; B_gamma_hat[0, 20] += phi_b
        B_gamma_hat[1, 18] += dphi_dy; B_gamma_hat[1, 19] -= phi_b
        
        K_full += (Bb.T @ Db @ Bb + B_gamma_hat.T @ Ds @ B_gamma_hat) * (A / 3.0)

    # Static Condensation: 21 -> 18
    Kbb = K_full[0:18, 0:18]; Kii = K_full[18:21, 18:21]
    Kib = K_full[18:21, 0:18]; Kbi = K_full[0:18, 18:21]
    if np.abs(np.trace(Kii)) > 1e-12:
        K_cond = Kbb - Kbi @ np.linalg.pinv(Kii, rcond=1e-12) @ Kib
    else:
        K_cond = Kbb
    K_loc += K_cond

    # 3. Drilling Stabilization (Standard OpenSees Penalty)
    # [WHT] Relative penalty preserves rigid body modes while providing stability.
    #       Must involve all 3 nodes in a single constraint to maintain objectivity.
    #
    # [WHT-TUNING] 드릴링 페널티 선정 근거:
    #   - QUAD4: Ktt = 1.0  * G * t  (기준치)
    #   - TRIA3: Ktt = 1e-2 * G * t  (현재값)
    #
    #   이전 값(1e-4)에서는 109 Hz 근처에 ~40개의 드릴링 모드가 군집을 형성하여
    #   122.9 Hz 굽힘 모드를 마스킹하는 문제가 있었음.
    #
    #   1e-2로 상향 시, 드릴링 모드 군집은 ~345 Hz 이상으로 이동하여
    #   모든 검증 타겟 주파수(~250 Hz 이하)와 충분히 분리됨.
    #
    #   QUAD4(1.0) 대비 100배 약하므로 굽힘·전단·막(membrane) 거동에는 영향 없음.
    Ktt = 1.0e-2 * G * t
    Bd = np.zeros((1, 18))
    for i in range(3):
        Bd[0, 6*i] = -0.5 * dN_dy[i]; Bd[0, 6*i+1] = 0.5 * dN_dx[i]; Bd[0, 6*i+5] = -1.0/3.0
    K_loc += (Bd.T @ Bd) * Ktt * A

        
    # Global Transformation
    T_18 = np.zeros((18, 18))
    for i in range(3):
        T_18[6*i:6*i+3, 6*i:6*i+3] = T; T_18[6*i+3:6*i+6, 6*i+3:6*i+6] = T
    return T_18.T @ K_loc @ T_18

def K_tria3_scipy(wht_model, sorted_nids, nid_to_idx) -> csr_matrix:
    ndof = len(sorted_nids) * 6; rows, cols, data = [], [], []
    nid_arr = list(sorted_nids); nid_to_crds = {nid: [wht_model.nodes[nid].x, wht_model.nodes[nid].y, wht_model.nodes[nid].z] for nid in nid_arr}
    for eid, elem in wht_model.elements.items():
        if elem.type not in ('TRIA3', 'TRIA'): continue
        nids = elem.node_ids; pid = elem.pid; prop = wht_model.properties.get(pid); mat = wht_model.materials.get(prop.mid) if prop else None
        if not prop: continue
        # Default E/nu if Material is missing
        t = prop.t; E = mat.E if mat else 210000.0; nu = mat.nu if mat else 0.3
        K_e = _element_K_tria3(np.array(nid_to_crds[nids[0]]), np.array(nid_to_crds[nids[1]]), np.array(nid_to_crds[nids[2]]), t, E, nu)
        dofs = np.array([nid_to_idx[nid] * 6 + d for nid in nids for d in range(6)])
        rr, cc = np.meshgrid(dofs, dofs, indexing='ij'); rows.append(rr.ravel()); cols.append(cc.ravel()); data.append(K_e.ravel())
    if not rows: return csr_matrix((ndof, ndof))
    return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof)).tocsr()

def M_tria3_lumped(wht_model, ndof: int, sorted_nids, nid_to_idx) -> np.ndarray:
    M_diag = np.zeros(ndof)
    # Unit System Note: Steel Density 7.85e-9 ton/mm^3 (consistent with MPa and mm)
    # If using kg/mm^3, change to 7.85e-6.
    nid_arr = list(sorted_nids); nid_to_crds = {nid: [wht_model.nodes[nid].x, wht_model.nodes[nid].y, wht_model.nodes[nid].z] for nid in nid_arr}
    for eid, elem in wht_model.elements.items():
        if elem.type not in ("TRIA3", "TRIA"): continue
        pid = elem.pid; prop = wht_model.properties.get(pid); mat = wht_model.materials.get(prop.mid) if prop else None
        if not prop: continue
        t = prop.t; rho = mat.rho if mat else 7.85e-9
        c1, c2, c3 = [np.array(nid_to_crds[nid]) for nid in elem.node_ids]
        area = tri_area(c1, c2, c3)
        m_node = (area * t * rho) / 3.0
        # Characteristic length for rotational inertia calculation
        L_char = np.sqrt(4.0 * area / np.sqrt(3.0))
        rot_inert = max(m_node * (L_char**2) / 12.0, 1e-8)
        for nid in elem.node_ids:
            idx = nid_to_idx[nid] * 6; M_diag[idx:idx+3] += m_node; M_diag[idx+3:idx+6] += rot_inert
    return M_diag
