"""
wht_quad4_element.py
====================
WHT FEM Framework — MITC4+ High-Fidelity Shell Element (24 DOF)

Formulation
-----------
* Reference: "The MITC4+ shell element and its performance", K.J. Bathe et al. (2006)
* Shear Tying: MITC (Mixed Interpolation of Tensorial Components) using 2 points for g13 and 2 for g23.
* Stability: Drilling stabilization using min eigenvalues of membrane (OpenSees rationale).
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def _get_shape_functions(xi, eta):
    return 0.25 * np.array([
        (1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)
    ])

def _get_shape_derivatives(xi, eta):
    dN_dxi = 0.25 * np.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
    dN_deta = 0.25 * np.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
    return dN_dxi, dN_deta

def _get_mb_matrices(xi, eta, coords):
    """Membrane and Bending B-matrices at (xi, eta)."""
    dN_dxi, dN_deta = _get_shape_derivatives(xi, eta)
    J = np.array([
        [np.dot(dN_dxi, coords[:,0]), np.dot(dN_dxi, coords[:,1])],
        [np.dot(dN_deta, coords[:,0]), np.dot(dN_deta, coords[:,1])]
    ])
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    
    dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
    dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
    
    Bm = np.zeros((3, 24))
    for i in range(4):
        Bm[0, 6*i]   = dN_dx[i]
        Bm[1, 6*i+1] = dN_dy[i]
        Bm[2, 6*i]   = dN_dy[i]
        Bm[2, 6*i+1] = dN_dx[i]
        
    Bb = np.zeros((3, 24))
    for i in range(4):
        Bb[0, 6*i+4] =  dN_dx[i]
        Bb[1, 6*i+3] = -dN_dy[i]
        Bb[2, 6*i+4] =  dN_dy[i]
        Bb[2, 6*i+3] = -dN_dx[i]
        
    return Bm, Bb, detJ

def _get_Bs_raw_13(xi, eta, coords):
    """Row of B for gamma_13."""
    Bs_row = np.zeros(24)
    N = _get_shape_functions(xi, eta)
    dN_dxi, dN_deta = _get_shape_derivatives(xi, eta)
    J = np.array([
        [np.dot(dN_dxi, coords[:, 0]), np.dot(dN_dxi, coords[:, 1])],
        [np.dot(dN_deta, coords[:, 0]), np.dot(dN_deta, coords[:, 1])]
    ])
    invJ = np.linalg.inv(J)
    # gamma_13 = w,x_local + theta_y_local
    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    for i in range(4):
        Bs_row[6*i+2] = dN_dx[i]
        Bs_row[6*i+4] = N[i]
    return Bs_row

def _get_Bs_raw_23(xi, eta, coords):
    """Row of B for gamma_23."""
    Bs_row = np.zeros(24)
    N = _get_shape_functions(xi, eta)
    dN_dxi, dN_deta = _get_shape_derivatives(xi, eta)
    J = np.array([
        [np.dot(dN_dxi, coords[:, 0]), np.dot(dN_dxi, coords[:, 1])],
        [np.dot(dN_deta, coords[:, 0]), np.dot(dN_deta, coords[:, 1])]
    ])
    invJ = np.linalg.inv(J)
    # gamma_23 = w,y_local - theta_x_local
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    for i in range(4):
        Bs_row[6*i+2] = dN_dy[i]
        Bs_row[6*i+3] = -N[i]
    return Bs_row

def _element_K_mitc4_plus(c1, c2, c3, c4, t, E, nu):
    # Standard local basis for Shell
    v12 = c2 - c1
    v14 = c4 - c1
    x_loc = v12 / (np.linalg.norm(v12) + 1e-12)
    z_raw = np.cross(x_loc, v14)
    z_loc = z_raw / (np.linalg.norm(z_raw) + 1e-12)
    y_loc = np.cross(z_loc, x_loc)
    T_mat = np.stack([x_loc, y_loc, z_loc], axis=0)
    
    # RELATIVE projection to get purely local 2D (x,y)
    p2 = c2 - c1; p3 = c3 - c1; p4 = c4 - c1
    coords_2d = np.array([
        [0.0, 0.0],
        [np.dot(p2, x_loc), np.dot(p2, y_loc)],
        [np.dot(p3, x_loc), np.dot(p3, y_loc)],
        [np.dot(p4, x_loc), np.dot(p4, y_loc)]
    ])
    
    # Material
    G = E / (2.0 * (1.0 + nu))
    k_shear = 5.0 / 6.0
    inv_nu2 = 1.0 / (1.0 - nu**2)
    Dm = (E * t * inv_nu2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Db = (E * t**3 / 12.0 * inv_nu2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Ds = (k_shear * G * t) * np.eye(2)
    
    # Assembly
    K_loc = np.zeros((24, 24))
    K_drill = np.zeros((24, 24))
    gp = [-1.0/np.sqrt(3), 1.0/np.sqrt(3)]
    Ktt = 0.01 * G # Penalty parameter (OpenSees rationale)

    for xi_g in gp:
        for eta_g in gp:
            Bm, Bb, detJ = _get_mb_matrices(xi_g, eta_g, coords_2d)
            
            # (1) B-matrix for Drilling (gamma_z = theta_z - 0.5*(v,x - u,y))
            # OpenSees/Bathe: B_drill = [ -0.5*dN/dy, 0.5*dN/dx, 0, 0, 0, -N ]
            N_list = _get_shape_functions(xi_g, eta_g)
            dN_dxi, dN_deta = _get_shape_derivatives(xi_g, eta_g)
            J = np.array([
                [np.dot(dN_dxi, coords_2d[:,0]), np.dot(dN_dxi, coords_2d[:,1])],
                [np.dot(dN_deta, coords_2d[:,0]), np.dot(dN_deta, coords_2d[:,1])]
            ])
            invJ = np.linalg.inv(J)
            detJ = np.linalg.det(J)
            if detJ < 1e-11: continue # Skip degenerate integration point
            
            dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
            dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
            
            Bd = np.zeros((1, 24))
            for i in range(4):
                Bd[0, 6*i]   = -0.5 * dN_dy[i] # u component
                Bd[0, 6*i+1] =  0.5 * dN_dx[i] # v component
                Bd[0, 6*i+5] = -N_list[i]      # theta_z component
            
            K_drill += (Bd.T @ Bd) * Ktt * detJ

            # MITC4 Shear (Julia Rationale)
            Bs13_A = _get_Bs_raw_13(0.0, -1.0, coords_2d)
            Bs13_B = _get_Bs_raw_13(0.0,  1.0, coords_2d)
            Bs13 = 0.5 * (1.0 - eta_g) * Bs13_A + 0.5 * (1.0 + eta_g) * Bs13_B
            
            Bs23_C = _get_Bs_raw_23(-1.0, 0.0, coords_2d)
            Bs23_D = _get_Bs_raw_23( 1.0, 0.0, coords_2d)
            Bs23 = 0.5 * (1.0 - xi_g) * Bs23_C + 0.5 * (1.0 + xi_g) * Bs23_D
            
            Bs = np.vstack([Bs13, Bs23])
            
            K_loc += (Bm.T @ Dm @ Bm + Bb.T @ Db @ Bb + Bs.T @ Ds @ Bs) * detJ

    # Add integrated drilling stiffness
    K_loc += K_drill
        
    # Global Transform
    T_24 = np.zeros((24, 24))
    for i in range(4):
        T_24[6*i:6*i+3, 6*i:6*i+3] = T_mat
        T_24[6*i+3:6*i+6, 6*i+3:6*i+6] = T_mat
    return T_24.T @ K_loc @ T_24

def K_quad4_scipy(wht_model, sorted_nids, nid_to_idx) -> csr_matrix:
    ndof = len(sorted_nids) * 6
    rows, cols, data = [], [], []
    nid_arr = list(sorted_nids)
    nid_to_crds = {nid: [wht_model.nodes[nid].x, wht_model.nodes[nid].y, wht_model.nodes[nid].z] for nid in nid_arr}
    
    for eid, elem in wht_model.elements.items():
        if elem.type not in ('QUAD4', 'QUAD'): continue
        nids = elem.node_ids
        pid = elem.pid
        prop = wht_model.properties.get(pid); mat = wht_model.materials.get(prop.mid) if prop else None
        t = prop.t if prop else 1.0; E = mat.E if mat else 210000.0; nu = mat.nu if mat else 0.3
        
        K_e = _element_K_mitc4_plus(
            np.array(nid_to_crds[nids[0]]), np.array(nid_to_crds[nids[1]]),
            np.array(nid_to_crds[nids[2]]), np.array(nid_to_crds[nids[3]]),
            t, E, nu
        )
        dofs = np.array([nid_to_idx[nid] * 6 + d for nid in nids for d in range(6)])
        rr, cc = np.meshgrid(dofs, dofs, indexing='ij')
        rows.append(rr.ravel()); cols.append(cc.ravel()); data.append(K_e.ravel())
        
    if not rows: return csr_matrix((ndof, ndof))
    return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof)).tocsr()

def M_quad4_lumped(wht_model, ndof: int, sorted_nids, nid_to_idx) -> np.ndarray:
    M_diag = np.zeros(ndof)
    # Unit System: 7.85e-9 ton/mm^3 (Consistent with MPa, N, s)
    # If standard kg/mm^3 is required, use 7.85e-6.
    for eid, elem in wht_model.elements.items():
        if elem.type not in ("QUAD4", "QUAD"): continue
        pid = elem.pid; prop = wht_model.properties.get(pid); mat = wht_model.materials.get(prop.mid) if prop else None
        if not prop: continue
        t = prop.t; rho = mat.rho if mat else 7.85e-9
        
        # Consistent area: sum of two triangles
        p = [np.array([wht_model.nodes[nid].x, wht_model.nodes[nid].y, wht_model.nodes[nid].z]) for nid in elem.node_ids]
        a1 = 0.5 * np.linalg.norm(np.cross(p[1]-p[0], p[2]-p[0]))
        a2 = 0.5 * np.linalg.norm(np.cross(p[2]-p[0], p[3]-p[0]))
        area = a1 + a2
        
        m_node = (area * t * rho) / 4.0
        # Characteristic length based on area for rotational inertia consistency
        L_char = np.sqrt(area)
        rot_inert = max(m_node * (L_char**2) / 12.0, 1e-8)
        for nid in elem.node_ids:
            base = nid_to_idx[nid] * 6
            M_diag[base:base+3] += m_node
            M_diag[base+3:base+6] += rot_inert
    return M_diag
