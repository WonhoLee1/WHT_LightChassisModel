"""
wht_tria3_element.py
====================
WHT FEM Framework — TRIA3 Flat Shell Element (18 DOF)

Formulation
-----------
* Membrane  : CST (Constant Strain Triangle)  — exact, no shear locking
* Bending   : Mindlin-Reissner with 1-point centroid integration
              (thin-plate Kirchhoff limit is recovered automatically)
* Drilling  : weak rotational spring (same strategy as JaxSSO MITC4)

Local DOF order (per node): [u, v, w, θx, θy, θz]
Global DOF index for node i : i*6 ... i*6+5

Public API
----------
    K_tria3_scipy(model, sorted_nids, nid_to_idx) -> scipy.sparse.csr_matrix
        Returns the assembled TRIA3 stiffness contribution in CSR format.
        Add to the JaxSSO QUAD4 K before solving.

    M_tria3_lumped(model, ndof, sorted_nids, nid_to_idx) -> np.ndarray (ndof,)
        Returns the TRIA3 lumped mass diagonal vector.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


# ---------------------------------------------------------------------------
# Element-level stiffness (single element, pure numpy)
# ---------------------------------------------------------------------------

def _local_csys(c1, c2, c3):
    """
    Build local coordinate system [X_loc, Y_loc, Z_loc] for a TRIA3.
    Returns (T_3x3, x2_loc, x3_loc, y3_loc, area)
      where T rows are local unit vectors expressed in global frame.
    """
    v12 = c2 - c1
    v13 = c3 - c1

    x_loc = v12 / np.linalg.norm(v12)
    z_raw = np.cross(x_loc, v13)
    z_loc = z_raw / np.linalg.norm(z_raw)
    y_loc = np.cross(z_loc, x_loc)

    # Local 2-D coordinates
    x2 = np.dot(v12, x_loc)          # always > 0
    x3 = np.dot(v13, x_loc)
    y3 = np.dot(v13, y_loc)          # always > 0 for CCW ordering

    area = 0.5 * x2 * y3             # = 0.5 * |det([v12_loc, v13_loc])|

    T = np.stack([x_loc, y_loc, z_loc], axis=0)  # (3, 3), rows = local axes
    return T, x2, x3, y3, area


def _element_K_tria3(c1, c2, c3, t, E, nu):
    """
    Compute the 18×18 element stiffness matrix in the **global** frame.

    Parameters
    ----------
    c1, c2, c3 : (3,) arrays  — global nodal coordinates
    t          : float        — shell thickness
    E          : float        — Young's modulus
    nu         : float        — Poisson's ratio

    Returns
    -------
    K_global : (18, 18) ndarray
    """
    T, x2, x3, y3, A = _local_csys(c1, c2, c3)
    two_A = 2.0 * A

    # ---- Shape-function derivatives in local 2-D ----
    # Standard CST:  N1=1-ξ-η, N2=ξ, N3=η  with mapping x=N1*0+N2*x2+N3*x3, y=N1*0+N2*0+N3*y3
    dN_dx = np.array([-y3, y3, 0.0]) / two_A
    dN_dy = np.array([x3 - x2, -x3, x2]) / two_A

    # ---- Plane-stress material matrix (membrane + bending) ----
    fac = E / (1.0 - nu ** 2)
    Cm = fac * np.array([[1.0, nu, 0.0],
                          [nu, 1.0, 0.0],
                          [0.0, 0.0, (1.0 - nu) / 2.0]])

    # ---- (1) Membrane stiffness K_m (6×6 in local u,v DOFs) ----
    # B_m (3×6): columns = [u1, v1, u2, v2, u3, v3]
    B_m = np.array([
        [dN_dx[0], 0.0,      dN_dx[1], 0.0,      dN_dx[2], 0.0     ],
        [0.0,      dN_dy[0], 0.0,      dN_dy[1], 0.0,      dN_dy[2]],
        [dN_dy[0], dN_dx[0], dN_dy[1], dN_dx[1], dN_dy[2], dN_dx[2]],
    ])
    K_m_loc = t * A * (B_m.T @ Cm @ B_m)      # (6, 6)

    # ---- (2) Bending stiffness K_b (9×9 in local w, θx, θy DOFs) ----
    # B_kappa (3×9): κxx=∂θy/∂x, κyy=-∂θx/∂y, κxy=∂θy/∂y-∂θx/∂x
    # DOF order: [w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3]
    B_kappa = np.array([
        [0.0,  0.0,       dN_dx[0],  0.0,  0.0,       dN_dx[1],  0.0,  0.0,       dN_dx[2]],
        [0.0, -dN_dy[0],  0.0,       0.0, -dN_dy[1],  0.0,       0.0, -dN_dy[2],  0.0     ],
        [0.0, -dN_dx[0],  dN_dy[0],  0.0, -dN_dx[1],  dN_dy[1],  0.0, -dN_dx[2],  dN_dy[2]],
    ])

    D_plate = (E * t ** 3) / (12.0 * (1.0 - nu ** 2))
    C_b = D_plate * np.array([[1.0, nu, 0.0],
                               [nu, 1.0, 0.0],
                               [0.0, 0.0, (1.0 - nu) / 2.0]])

    k_shear_factor = 5.0 / 6.0
    G = E / (2.0 * (1.0 + nu))
    C_s = k_shear_factor * G * t * np.eye(2)

    # Shear K_s: 3-point Gauss integration to eliminate spurious zero-energy mode.
    # 1-point (centroid) Mindlin triangle has a rank-deficient K_b (null-space dim=4
    # instead of 3) because one spurious twist mode has zero shear at the centroid.
    # Evaluating B_s at 3 interior Gauss points makes that mode penalized => rank 12.
    gp_xi  = np.array([2.0/3.0, 1.0/6.0, 1.0/6.0])
    gp_eta = np.array([1.0/6.0, 2.0/3.0, 1.0/6.0])
    w_gp   = 1.0 / 3.0

    K_s_loc = np.zeros((9, 9))
    for xi, eta in zip(gp_xi, gp_eta):
        N_gp = np.array([1.0 - xi - eta, xi, eta])
        B_s_gp = np.array([
            [dN_dx[0],  0.0,    N_gp[0], dN_dx[1],  0.0,    N_gp[1], dN_dx[2],  0.0,    N_gp[2]],
            [dN_dy[0], -N_gp[0], 0.0,   dN_dy[1], -N_gp[1], 0.0,    dN_dy[2], -N_gp[2], 0.0   ],
        ])
        K_s_loc += A * w_gp * (B_s_gp.T @ C_s @ B_s_gp)

    K_b_loc = A * (B_kappa.T @ C_b @ B_kappa) + K_s_loc  # (9, 9)

    # ---- (3) Expand to 18×18 local matrix ----
    # Node-level DOF layout: [u, v, w, θx, θy, θz]  →  [0,1,2,3,4,5]
    # For 3 nodes: global index = node*6 + local_dof
    K_local = np.zeros((18, 18))

    # Membrane: (u,v) → local DOF [0,1] per node → global 0,1,6,7,12,13
    m_dofs = np.array([0, 1, 6, 7, 12, 13])
    K_local[np.ix_(m_dofs, m_dofs)] += K_m_loc

    # Bending: (w,θx,θy) → local DOF [2,3,4] per node → global 2,3,4,8,9,10,14,15,16
    b_dofs = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16])
    K_local[np.ix_(b_dofs, b_dofs)] += K_b_loc

    # Drilling weak spring: use the same strategy as JaxSSO MITC4
    # k_rz = min diagonal bending term / 1000
    bending_diag = np.abs(np.diag(K_b_loc)[[1, 2, 4, 5, 7, 8]])  # θx, θy terms
    k_rz = np.min(bending_diag[bending_diag > 0]) / 100.0 if np.any(bending_diag > 0) else 1e-6
    drill_dofs = [5, 11, 17]
    for d in drill_dofs:
        K_local[d, d] += k_rz

    # ---- (4) Build 18×18 transformation matrix ----
    # T_18 is block-diagonal: 3 × diag(T_3x3, T_3x3)
    T_18 = np.zeros((18, 18))
    for n in range(3):
        s = n * 6
        T_18[s:s+3, s:s+3] = T          # translation
        T_18[s+3:s+6, s+3:s+6] = T      # rotation

    # ---- (5) Transform to global ----
    K_global = T_18.T @ K_local @ T_18

    return K_global


# ---------------------------------------------------------------------------
# Assembled stiffness (all TRIA3 elements in the model)
# ---------------------------------------------------------------------------

def K_tria3_scipy(wht_model, sorted_nids, nid_to_idx) -> csr_matrix:
    """
    Assemble TRIA3 global stiffness matrix in scipy CSR format.

    Only processes elements whose type is 'TRIA3' or 'TRIA'.
    Elements with other types are skipped silently.
    """
    ndof = len(sorted_nids) * 6

    # Gather node coordinates array (ordered by sorted_nids)
    c_all = np.array([
        [wht_model.nodes[nid].x, wht_model.nodes[nid].y, wht_model.nodes[nid].z]
        if hasattr(wht_model.nodes[nid], 'x')
        else list(wht_model.nodes[nid])
        for nid in sorted_nids
    ])

    rows_list, cols_list, data_list = [], [], []

    for eid in sorted(wht_model.elements.keys()):
        elem = wht_model.elements[eid]
        is_obj = hasattr(elem, 'type')
        etype = elem.type if is_obj else wht_model.element_types.get(eid, '')
        node_ids = elem.node_ids if is_obj else elem

        if etype not in ('TRIA3', 'TRIA') or len(node_ids) != 3:
            continue

        # Material / property lookup
        pid = getattr(elem, 'pid', 0) if is_obj else 0
        E, nu, t = 210000.0, 0.3, 1.0
        if hasattr(wht_model, 'properties') and pid in wht_model.properties:
            prop = wht_model.properties[pid]
            t = getattr(prop, 't', 1.0)
            mid = getattr(prop, 'mid', 0)
            if hasattr(wht_model, 'materials') and mid in wht_model.materials:
                mat = wht_model.materials[mid]
                E  = getattr(mat, 'E',  210000.0)
                nu = getattr(mat, 'nu', 0.3)

        # Global node indices
        idx = [nid_to_idx[n] for n in node_ids]
        c1, c2, c3 = c_all[idx[0]], c_all[idx[1]], c_all[idx[2]]

        K_e = _element_K_tria3(c1, c2, c3, t, E, nu)   # (18, 18)

        # DOF scatter indices
        dofs = np.array([i * 6 + d for i in idx for d in range(6)])  # (18,)
        rr, cc = np.meshgrid(dofs, dofs, indexing='ij')
        rows_list.append(rr.ravel())
        cols_list.append(cc.ravel())
        data_list.append(K_e.ravel())

    if not rows_list:
        return csr_matrix((ndof, ndof))

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.concatenate(data_list)

    return coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()


# ---------------------------------------------------------------------------
# Lumped mass vector
# ---------------------------------------------------------------------------

def M_tria3_lumped(wht_model, ndof: int, sorted_nids, nid_to_idx) -> np.ndarray:
    """
    Build TRIA3 contribution to the lumped mass diagonal vector.

    Uses area × thickness × density, evenly distributed to the 3 nodes.
    Rotational inertia: same near-zero stub as QUAD4 (avoids spurious modes).
    """
    M_diag = np.zeros(ndof)

    c_all = np.array([
        [wht_model.nodes[nid].x, wht_model.nodes[nid].y, wht_model.nodes[nid].z]
        if hasattr(wht_model.nodes[nid], 'x')
        else list(wht_model.nodes[nid])
        for nid in sorted_nids
    ])

    for eid in sorted(wht_model.elements.keys()):
        elem = wht_model.elements[eid]
        is_obj = hasattr(elem, 'type')
        etype = elem.type if is_obj else wht_model.element_types.get(eid, '')
        node_ids = elem.node_ids if is_obj else elem

        if etype not in ('TRIA3', 'TRIA') or len(node_ids) != 3:
            continue

        pid = getattr(elem, 'pid', 0) if is_obj else 0
        t, rho = 1.0, 7.85e-9
        if hasattr(wht_model, 'properties') and pid in wht_model.properties:
            prop = wht_model.properties[pid]
            t = getattr(prop, 't', 1.0)
            mid = getattr(prop, 'mid', 0)
            if hasattr(wht_model, 'materials') and mid in wht_model.materials:
                mat = wht_model.materials[mid]
                rho = getattr(mat, 'rho', 7.85e-9)

        idx = [nid_to_idx[n] for n in node_ids]
        c1, c2, c3 = c_all[idx[0]], c_all[idx[1]], c_all[idx[2]]
        v12, v13 = c2 - c1, c3 - c1
        area = 0.5 * np.linalg.norm(np.cross(v12, v13))

        elem_mass = area * t * rho
        m_node = elem_mass / 3.0
        rot_inert = max(m_node * area / 12.0, 1e-8)   # area-based, matches QUAD4 m*L²/12

        for i in idx:
            base = i * 6
            M_diag[base:base+3]   += m_node
            M_diag[base+3:base+6] += rot_inert

    return M_diag
