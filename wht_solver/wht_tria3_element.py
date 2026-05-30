# -*- coding: utf-8 -*-
"""
wht_tria3_element.py
====================
WHT FEM Framework — CalculiX US3 (CS-DSG + ANDES) Shell Element Formulation.
100% numerically identical to CalculiX's US3 3-node flat shell element.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def tri_area(c1, c2, c3):
    """Calculates triangle area using cross-product."""
    return 0.5 * np.linalg.norm(np.cross(c2 - c1, c3 - c1))

def us3_csys_cr(xg):
    """Calculates local basis tm (3,3) and tmg (18,18) relative to node 1."""
    e1 = xg[1] - xg[0]
    dl1 = np.linalg.norm(e1) + 1e-15
    e1 = e1 / dl1
    
    e2 = xg[2] - xg[0]
    e3 = np.cross(e1, e2)
    dl3 = np.linalg.norm(e3) + 1e-15
    e3 = e3 / dl3
    
    e2 = np.cross(e3, e1)
    dl2 = np.linalg.norm(e2) + 1e-15
    e2 = e2 / dl2
    
    tm = np.stack([e1, e2, e3], axis=0)
    
    tmg = np.zeros((18, 18))
    for i in range(6):
        tmg[3*i:3*i+3, 3*i:3*i+3] = tm
    return tm, tmg

def us3_linel_Qi(E, nu):
    """Generates material elasticity matrices Qin (3,3) and Qs (2,2)."""
    Qin = np.zeros((3, 3))
    q1 = E / (1.0 - nu**2)
    Qin[0, 0] = q1
    Qin[0, 1] = q1 * nu
    Qin[1, 0] = q1 * nu
    Qin[1, 1] = q1
    Qin[2, 2] = q1 * (1.0 - nu) / 2.0
    
    Qs = np.zeros((2, 2))
    kap = 5.0 / 6.0
    q1_s = E / (2.0 * (1.0 + nu))
    Qs[0, 0] = q1_s * kap
    Qs[1, 1] = q1_s * kap
    return Qin, Qs

def us3_CS(X, Y):
    """Cell-based smoothing sub-triangle B-matrix."""
    x21 = X[1] - X[0]
    x13 = X[0] - X[2]
    y31 = Y[2] - Y[0]
    y12 = Y[0] - Y[1]
    x32 = X[2] - X[1]
    y23 = Y[1] - Y[2]
    
    Ae = 0.5 * (x21 * y31 - x13 * y12)
    Ae_safe = Ae if abs(Ae) > 1e-12 else 1e-12
    
    a1 = 0.5 * y12 * x13
    a2 = 0.5 * y31 * x21
    a3 = 0.5 * x21 * x13
    a4 = 0.5 * y12 * y31
    
    bs1 = np.zeros((2, 6))
    bs1[0, 2] = 0.5 * x32 / Ae_safe
    bs1[0, 3] = -0.5
    bs1[1, 2] = 0.5 * y23 / Ae_safe
    bs1[1, 4] = 0.5
    
    bs2 = np.zeros((2, 6))
    bs2[0, 2] = 0.5 * x13 / Ae_safe
    bs2[0, 3] = 0.5 * a1 / Ae_safe
    bs2[0, 4] = 0.5 * a3 / Ae_safe
    bs2[1, 2] = 0.5 * y31 / Ae_safe
    bs2[1, 3] = 0.5 * a4 / Ae_safe
    bs2[1, 4] = 0.5 * a2 / Ae_safe
    
    bs3 = np.zeros((2, 6))
    bs3[0, 2] = 0.5 * x21 / Ae_safe
    bs3[0, 3] = -0.5 * a2 / Ae_safe
    bs3[0, 4] = -0.5 * a3 / Ae_safe
    bs3[1, 2] = 0.5 * y12 / Ae_safe
    bs3[1, 3] = -0.5 * a4 / Ae_safe
    bs3[1, 4] = -0.5 * a1 / Ae_safe
    
    return bs1, bs2, bs3, Ae

def us3_Bs(X):
    """CS-DSG shear strain B-matrix assembly."""
    x1, x2, x3 = X[0, 0], X[1, 0], X[2, 0]
    y1, y2, y3 = X[0, 1], X[1, 1], X[2, 1]
    x31 = x3 - x1
    y31 = y3 - y1
    y12 = y1 - y2
    x21 = x2 - x1
    
    Ae = 0.5 * (x21 * y31 - x31 * (-y12))
    Ae_safe = Ae if abs(Ae) > 1e-12 else 1e-12
    
    x0 = (x1 + x2 + x3) / 3.0
    y0 = (y1 + y2 + y3) / 3.0
    
    X_1 = np.array([x0, x1, x2])
    Y_1 = np.array([y0, y1, y2])
    X_2 = np.array([x0, x2, x3])
    Y_2 = np.array([y0, y2, y3])
    X_3 = np.array([x0, x3, x1])
    Y_3 = np.array([y0, y3, y1])
    
    a3 = 1.0 / 3.0
    
    # sub-tri 1
    bs1, bs2, bs3, Ai = us3_CS(X_1, Y_1)
    B1 = np.zeros((2, 18))
    B1[:, 0:6] = a3 * bs1 + bs2
    B1[:, 6:12] = a3 * bs1 + bs3
    B1[:, 12:18] = a3 * bs1
    B1 = B1 * Ai
    
    # sub-tri 2
    bs1, bs2, bs3, Ai = us3_CS(X_2, Y_2)
    B2 = np.zeros((2, 18))
    B2[:, 0:6] = a3 * bs1
    B2[:, 6:12] = a3 * bs1 + bs2
    B2[:, 12:18] = a3 * bs1 + bs3
    B2 = B2 * Ai
    
    # sub-tri 3
    bs1, bs2, bs3, Ai = us3_CS(X_3, Y_3)
    B3 = np.zeros((2, 18))
    B3[:, 0:6] = a3 * bs1 + bs3
    B3[:, 6:12] = a3 * bs1
    B3[:, 12:18] = a3 * bs1 + bs2
    B3 = B3 * Ai
    
    Bs = (1.0 / Ae_safe) * (B1 + B2 + B3)
    return Bs

def us3_Bb(X, Y):
    """Plate curvature B-matrix."""
    x21 = X[1] - X[0]
    x31 = X[2] - X[0]
    x32 = X[2] - X[1]
    y23 = Y[1] - Y[2]
    y31 = Y[2] - Y[0]
    y12 = Y[0] - Y[1]
    
    Ae = 0.5 * (x21 * y31 - x31 * (-y12))
    Ae_safe = Ae if abs(Ae) > 1e-12 else 1e-12
    
    dNdx1 = y23 / (2.0 * Ae_safe)
    dNdy1 = x32 / (2.0 * Ae_safe)
    dNdx2 = y31 / (2.0 * Ae_safe)
    dNdy2 = -x31 / (2.0 * Ae_safe)
    dNdx3 = y12 / (2.0 * Ae_safe)
    dNdy3 = x21 / (2.0 * Ae_safe)
    
    bb = np.zeros((3, 18))
    bb[0, 4] = dNdx1
    bb[0, 10] = dNdx2
    bb[0, 16] = dNdx3
    
    bb[1, 3] = -dNdy1
    bb[1, 9] = -dNdy2
    bb[1, 15] = -dNdy3
    
    bb[2, 3] = -dNdx1
    bb[2, 4] = dNdy1
    bb[2, 9] = -dNdx2
    bb[2, 10] = dNdy2
    bb[2, 15] = -dNdx3
    bb[2, 16] = dNdy3
    return bb

def us3_Kp(X, Db, Ds):
    """CS-DSG plate bending stiffness assembly."""
    x21 = X[1, 0] - X[0, 0]
    x31 = X[2, 0] - X[0, 0]
    y31 = X[2, 1] - X[0, 1]
    y12 = X[0, 1] - X[1, 1]
    Ae = 0.5 * (x21 * y31 - x31 * (-y12))
    
    Bs = us3_Bs(X)
    Bb = us3_Bb(X[:, 0], X[:, 1])
    
    Kp = (Bs.T @ Ds @ Bs + Bb.T @ Db @ Bb) * Ae
    return Kp

def us3_Km(X, Qin, h):
    """ANDES membrane stiffness matrix assembly."""
    alpha = 1.0 / 8.0
    ab = alpha / 6.0
    b0 = (alpha**2) / 4.0
    
    b1, b2, b3 = 1.0, 2.0, 1.0
    b4, b5, b6 = 0.0, 1.0, -1.0
    b7, b8, b9 = -1.0, -1.0, -2.0
    
    x12 = X[0, 0] - X[1, 0]
    x23 = X[1, 0] - X[2, 0]
    x31 = X[2, 0] - X[0, 0]
    y12 = X[0, 1] - X[1, 1]
    y23 = X[1, 1] - X[2, 1]
    y31 = X[2, 1] - X[0, 1]
    
    x21 = -x12
    x32 = -x23
    x13 = -x31
    y21 = -y12
    y32 = -y23
    y13 = -y31
    
    Ae = 0.5 * (x21 * y31 - x13 * (-y12))
    Ae_safe = Ae if abs(Ae) > 1e-12 else 1e-12
    
    A2 = 2.0 * Ae
    A4 = 4.0 * Ae
    h2 = 0.5 * h
    V = Ae * h
    
    LL21 = x21**2 + y21**2
    LL32 = x32**2 + y32**2
    LL13 = x13**2 + y13**2
    
    L = np.zeros((9, 3))
    L[0, 0] = h2 * y23
    L[0, 2] = h2 * x32
    L[1, 1] = h2 * x32
    L[1, 2] = h2 * y23
    L[2, 0] = h2 * y23 * (y13 - y21) * ab
    L[2, 1] = h2 * x32 * (x31 - x12) * ab
    L[2, 2] = h2 * (x31 * y13 - x12 * y21) * 2.0 * ab
    
    L[3, 0] = h2 * y31
    L[3, 2] = h2 * x13
    L[4, 1] = h2 * x13
    L[4, 2] = h2 * y31
    L[5, 0] = h2 * y31 * (y21 - y32) * ab
    L[5, 1] = h2 * x13 * (x12 - x23) * ab
    L[5, 2] = h2 * (x12 * y21 - x23 * y32) * 2.0 * ab
    
    L[6, 0] = h2 * y12
    L[6, 2] = h2 * x21
    L[7, 1] = h2 * x21
    L[7, 2] = h2 * y12
    L[8, 0] = h2 * y12 * (y32 - y13) * ab
    L[8, 1] = h2 * x21 * (x23 - x31) * ab
    L[8, 2] = h2 * (x23 * y32 - x31 * y13) * 2.0 * ab
    
    Kb = (L @ Qin @ L.T) / V
    
    T0 = np.zeros((3, 9))
    T0[0, 0] = x32 / A4
    T0[0, 1] = y32 / A4
    T0[0, 2] = 1.0
    T0[0, 3] = x13 / A4
    T0[0, 4] = y13 / A4
    T0[0, 6] = x21 / A4
    T0[0, 7] = y21 / A4
    
    T0[1, 0] = x32 / A4
    T0[1, 1] = y32 / A4
    T0[1, 3] = x13 / A4
    T0[1, 4] = y13 / A4
    T0[1, 5] = 1.0
    T0[1, 6] = x21 / A4
    T0[1, 7] = y21 / A4
    
    T0[2, 0] = x32 / A4
    T0[2, 1] = y32 / A4
    T0[2, 3] = x13 / A4
    T0[2, 4] = y13 / A4
    T0[2, 6] = x21 / A4
    T0[2, 7] = y21 / A4
    T0[2, 8] = 1.0
    
    A14 = 1.0 / (Ae_safe * A4)
    Te = np.zeros((3, 3))
    Te[0, 0] = A14 * y23 * y13 * LL21
    Te[0, 1] = A14 * y31 * y21 * LL32
    Te[0, 2] = A14 * y12 * y32 * LL13
    Te[1, 0] = A14 * x23 * x13 * LL21
    Te[1, 1] = A14 * x31 * x21 * LL32
    Te[1, 2] = A14 * x12 * x32 * LL13
    Te[2, 0] = A14 * (y23 * x31 + x32 * y13) * LL21
    Te[2, 1] = A14 * (y31 * x12 + x13 * y21) * LL32
    Te[2, 2] = A14 * (y12 * x23 + x21 * y32) * LL13
    
    A14_q = A2 / 3.0
    Q1 = np.zeros((3, 3))
    Q1[0, 0] = A14_q * b1 / LL21
    Q1[0, 1] = A14_q * b2 / LL21
    Q1[0, 2] = A14_q * b3 / LL21
    Q1[1, 0] = A14_q * b4 / LL32
    Q1[1, 1] = A14_q * b5 / LL32
    Q1[1, 2] = A14_q * b6 / LL32
    Q1[2, 0] = A14_q * b7 / LL13
    Q1[2, 1] = A14_q * b8 / LL13
    Q1[2, 2] = A14_q * b9 / LL13
    
    Q2 = np.zeros((3, 3))
    Q2[0, 0] = A14_q * b9 / LL21
    Q2[0, 1] = A14_q * b7 / LL21
    Q2[0, 2] = A14_q * b8 / LL21
    Q2[1, 0] = A14_q * b3 / LL32
    Q2[1, 1] = A14_q * b1 / LL32
    Q2[1, 2] = A14_q * b2 / LL32
    Q2[2, 0] = A14_q * b6 / LL13
    Q2[2, 1] = A14_q * b4 / LL13
    Q2[2, 2] = A14_q * b5 / LL13
    
    Q3 = np.zeros((3, 3))
    Q3[0, 0] = A14_q * b5 / LL21
    Q3[0, 1] = A14_q * b6 / LL21
    Q3[0, 2] = A14_q * b4 / LL21
    Q3[1, 0] = A14_q * b8 / LL32
    Q3[1, 1] = A14_q * b9 / LL32
    Q3[1, 2] = A14_q * b7 / LL32
    Q3[2, 0] = A14_q * b2 / LL13
    Q3[2, 1] = A14_q * b3 / LL13
    Q3[2, 2] = A14_q * b1 / LL13
    
    Q4 = (Q1 + Q2) * 0.5
    Q5 = (Q2 + Q3) * 0.5
    Q6 = (Q3 + Q1) * 0.5
    
    Enat = Te.T @ Qin @ Te
    
    KO = (3.0 / 4.0) * b0 * V * (Q4.T @ Enat @ Q4 + Q5.T @ Enat @ Q5 + Q6.T @ Enat @ Q6)
    Kh = T0.T @ KO @ T0
    Km_9x9 = Kb + Kh
    
    K = np.zeros((18, 18))
    for i in range(3):
        r_f = 6 * i
        r_m = 3 * i
        
        def map_9_to_18(row_9):
            res = np.zeros(18)
            res[0:2] = row_9[0:2]
            res[5] = row_9[2]
            res[6:8] = row_9[3:5]
            res[11] = row_9[5]
            res[12:14] = row_9[6:8]
            res[17] = row_9[8]
            return res
            
        K[r_f] = map_9_to_18(Km_9x9[r_m])
        K[r_f+1] = map_9_to_18(Km_9x9[r_m+1])
        K[r_f+5] = map_9_to_18(Km_9x9[r_m+2])
        
    return K

def us3_M(X, h, rho):
    """Consistent mass matrix assembly."""
    M = np.zeros((18, 18))
    points3 = np.array([
        [1.0/6.0, 1.0/6.0],
        [4.0/6.0, 1.0/6.0],
        [1.0/6.0, 4.0/6.0]
    ])
    w3 = 1.0 / 3.0
    
    x12 = X[0, 0] - X[1, 0]
    x23 = X[1, 0] - X[2, 0]
    x31 = X[2, 0] - X[0, 0]
    y12 = X[0, 1] - X[1, 1]
    y23 = X[1, 1] - X[2, 1]
    y31 = X[2, 1] - X[0, 1]
    
    x21 = -x12
    x32 = -x23
    x13 = -x31
    y21 = -y12
    y32 = -y23
    y13 = -y31
    
    Ae = 0.5 * (x21 * y31 - x13 * (-y12))
    
    m_3t = np.zeros((6, 6))
    q1 = rho * h
    m_3t[0, 0] = q1
    m_3t[1, 1] = q1
    m_3t[2, 2] = q1
    q1_rot = (rho * h**3) / 12.0
    m_3t[3, 3] = q1_rot
    m_3t[4, 4] = q1_rot
    
    for i in range(3):
        r = points3[i, 0]
        s = points3[i, 1]
        Nrs = np.array([1.0 - r - s, r, s])
        
        N_u = np.zeros((6, 18))
        for j in range(3):
            N_u[0, 0 + j*6] = Nrs[j]
            N_u[1, 1 + j*6] = Nrs[j]
            N_u[2, 2 + j*6] = Nrs[j]
            N_u[3, 3 + j*6] = Nrs[j]
            N_u[4, 4 + j*6] = Nrs[j]
            
        M += (N_u.T @ m_3t @ N_u) * Ae * w3
    return M

def _element_K_tria3(c1, c2, c3, t, E, nu) -> np.ndarray:
    """Single TRIA3 flat shell element stiffness (18x18) in global coords."""
    xg = np.stack([c1, c2, c3], axis=0)
    tm, tmg = us3_csys_cr(xg)
    x = np.stack([tm @ xg[0], tm @ xg[1], tm @ xg[2]], axis=0)
    Qin, Qs = us3_linel_Qi(E, nu)
    Db = Qin * (t**3) / 12.0
    Ds = Qs * t
    Kshell = us3_Km(x, Qin, t) + us3_Kp(x, Db, Ds)
    return tmg.T @ Kshell @ tmg


def K_tria3_scipy(wht_model, sorted_nids, nid_to_idx) -> csr_matrix:
    """Assembles sparse global stiffness matrix using CalculiX US3 flat shell formulation."""
    ndof = len(sorted_nids) * 6; rows, cols, data = [], [], []
    nid_arr = list(sorted_nids); nid_to_crds = {nid: [wht_model.nodes[nid].x, wht_model.nodes[nid].y, wht_model.nodes[nid].z] for nid in nid_arr}
    for eid, elem in wht_model.elements.items():
        e_type = elem.type.upper()
        if e_type not in ('TRIA3', 'TRIA'): continue
        nids = elem.node_ids; pid = elem.pid; prop = wht_model.properties.get(pid); mat = wht_model.materials.get(prop.mid) if prop else None
        if not prop: continue
        t = prop.t; E = mat.E if mat else 210000.0; nu = mat.nu if mat else 0.3
        
        c1 = np.array(nid_to_crds[nids[0]])
        c2 = np.array(nid_to_crds[nids[1]])
        c3 = np.array(nid_to_crds[nids[2]])
        
        # 1. Coordinate transformation
        xg = np.stack([c1, c2, c3], axis=0)
        tm, tmg = us3_csys_cr(xg)
        
        # 2. Local coordinates
        x = np.zeros((3, 3))
        x[0] = tm @ xg[0]
        x[1] = tm @ xg[1]
        x[2] = tm @ xg[2]
        
        # 3. Elastic tangent matrices
        Qin, Qs = us3_linel_Qi(E, nu)
        
        # 4. Thickness integration (membrane, bending, shear)
        Dm = Qin * t
        Db = Qin * (t**3) / 12.0
        Ds = Qs * t
        
        # 5. Kp (plate CS-DSG) & Km (membrane ANDES)
        Kp = us3_Kp(x, Db, Ds)
        Km = us3_Km(x, Qin, t)
        
        Kshell = Km + Kp
        
        # 6. Global transformation
        K_e = tmg.T @ Kshell @ tmg
        
        dofs = np.array([nid_to_idx[nid] * 6 + d for nid in nids for d in range(6)])
        rr, cc = np.meshgrid(dofs, dofs, indexing='ij')
        rows.append(rr.ravel()); cols.append(cc.ravel()); data.append(K_e.ravel())
        
    if not rows: return csr_matrix((ndof, ndof))
    return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof)).tocsr()

def M_tria3_lumped(wht_model, ndof: int, sorted_nids, nid_to_idx) -> np.ndarray:
    """Assembles global mass diagonal vector from globalized consistent mass matrix (Row Sum Lumping)."""
    M_diag = np.zeros(ndof)
    nid_arr = list(sorted_nids); nid_to_crds = {nid: [wht_model.nodes[nid].x, wht_model.nodes[nid].y, wht_model.nodes[nid].z] for nid in nid_arr}
    for eid, elem in wht_model.elements.items():
        if elem.type not in ("TRIA3", "TRIA"): continue
        pid = getattr(elem, "pid", None)
        if pid is None or pid == 0: continue
        prop = wht_model.properties.get(pid)
        if not prop: continue
        mat = wht_model.materials.get(prop.mid)
        t = prop.t; rho = mat.rho if mat else 7.85e-9
        
        c1, c2, c3 = [np.array(nid_to_crds[nid]) for nid in elem.node_ids]
        xg = np.stack([c1, c2, c3], axis=0)
        tm, tmg = us3_csys_cr(xg)
        x = np.zeros((3, 3))
        x[0] = tm @ xg[0]
        x[1] = tm @ xg[1]
        x[2] = tm @ xg[2]
        
        # Consistent mass matrix us3_M
        M_local = us3_M(x, t, rho)
        
        # Global transformation
        M_e = tmg.T @ M_local @ tmg
        
        # Row Sum Lumping 적용하여 전체 질량 완벽 보존
        row_sums = np.sum(M_e, axis=1)
        for i, nid in enumerate(elem.node_ids):
            idx = nid_to_idx[nid] * 6
            M_diag[idx:idx+6] += row_sums[6*i : 6*i+6]
            
    return M_diag
