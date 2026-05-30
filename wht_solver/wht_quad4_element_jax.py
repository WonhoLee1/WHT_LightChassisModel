"""
wht_quad4_element_jax.py
========================
MITC4+ 요소 강성 행렬 JAX 구현.

_element_K_mitc4_plus_jax  : 단일 요소 순수 함수 (jit 적용 가능)
K_quad4_jax                : vmap 배치 조립 — 전체 요소를 단일 XLA 커널로 처리
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

@jax.jit
def _get_shape_functions(xi, eta):
    return 0.25 * jnp.array([
        (1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)
    ])

@jax.jit
def _get_shape_derivatives(xi, eta):
    dN_dxi = 0.25 * jnp.array([-(1-eta), (1-eta), (1+eta), -(1+eta)])
    dN_deta = 0.25 * jnp.array([-(1-xi), -(1+xi), (1+xi), (1-xi)])
    return dN_dxi, dN_deta

@jax.jit
def _get_mb_matrices(xi, eta, coords):
    dN_dxi, dN_deta = _get_shape_derivatives(xi, eta)
    J = jnp.array([
        [jnp.dot(dN_dxi, coords[:,0]), jnp.dot(dN_dxi, coords[:,1])],
        [jnp.dot(dN_deta, coords[:,0]), jnp.dot(dN_deta, coords[:,1])]
    ])
    detJ = jnp.linalg.det(J)
    invJ = jnp.linalg.inv(J)
    
    dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
    dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
    
    Bm = jnp.zeros((3, 24))
    for i in range(4):
        Bm = Bm.at[0, 6*i].set(dN_dx[i])
        Bm = Bm.at[1, 6*i+1].set(dN_dy[i])
        Bm = Bm.at[2, 6*i].set(dN_dy[i])
        Bm = Bm.at[2, 6*i+1].set(dN_dx[i])
        
    Bb = jnp.zeros((3, 24))
    for i in range(4):
        Bb = Bb.at[0, 6*i+4].set(dN_dx[i])
        Bb = Bb.at[1, 6*i+3].set(-dN_dy[i])
        Bb = Bb.at[2, 6*i+4].set(dN_dy[i])
        Bb = Bb.at[2, 6*i+3].set(-dN_dx[i])
        
    return Bm, Bb, detJ

@jax.jit
def _get_Bs_raw_13(xi, eta, coords):
    Bs_row = jnp.zeros(24)
    N = _get_shape_functions(xi, eta)
    dN_dxi, dN_deta = _get_shape_derivatives(xi, eta)
    J = jnp.array([
        [jnp.dot(dN_dxi, coords[:, 0]), jnp.dot(dN_dxi, coords[:, 1])],
        [jnp.dot(dN_deta, coords[:, 0]), jnp.dot(dN_deta, coords[:, 1])]
    ])
    invJ = jnp.linalg.inv(J)
    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    for i in range(4):
        Bs_row = Bs_row.at[6*i+2].set(dN_dx[i])
        Bs_row = Bs_row.at[6*i+4].set(N[i])
    return Bs_row

@jax.jit
def _get_Bs_raw_23(xi, eta, coords):
    Bs_row = jnp.zeros(24)
    N = _get_shape_functions(xi, eta)
    dN_dxi, dN_deta = _get_shape_derivatives(xi, eta)
    J = jnp.array([
        [jnp.dot(dN_dxi, coords[:, 0]), jnp.dot(dN_dxi, coords[:, 1])],
        [jnp.dot(dN_deta, coords[:, 0]), jnp.dot(dN_deta, coords[:, 1])]
    ])
    invJ = jnp.linalg.inv(J)
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    for i in range(4):
        Bs_row = Bs_row.at[6*i+2].set(dN_dy[i])
        Bs_row = Bs_row.at[6*i+3].set(-N[i])
    return Bs_row

def _element_K_mitc4_plus_jax(c1, c2, c3, c4, t, E, nu, beta):
    v12 = c2 - c1
    v14 = c4 - c1
    x_loc = v12 / (jnp.linalg.norm(v12) + 1e-12)
    z_raw = jnp.cross(x_loc, v14)
    z_loc = z_raw / (jnp.linalg.norm(z_raw) + 1e-12)
    y_loc = jnp.cross(z_loc, x_loc)
    T_mat = jnp.stack([x_loc, y_loc, z_loc], axis=0)
    
    p2 = c2 - c1; p3 = c3 - c1; p4 = c4 - c1
    coords_2d = jnp.array([
        [0.0, 0.0],
        [jnp.dot(p2, x_loc), jnp.dot(p2, y_loc)],
        [jnp.dot(p3, x_loc), jnp.dot(p3, y_loc)],
        [jnp.dot(p4, x_loc), jnp.dot(p4, y_loc)]
    ])
    
    G = E / (2.0 * (1.0 + nu))
    k_shear = 5.0 / 6.0
    inv_nu2 = 1.0 / (1.0 - nu**2)
    Dm = (E * t * inv_nu2) * jnp.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Db = (E * t**3 / 12.0 * inv_nu2) * jnp.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Ds = (k_shear * G * t) * jnp.eye(2)
    
    K_loc = jnp.zeros((24, 24))
    K_drill = jnp.zeros((24, 24))
    gp = jnp.array([-1.0/jnp.sqrt(3), 1.0/jnp.sqrt(3)])
    # Ktt = 1.0 * G * t 

    # Membrane locking fix: Selective Reduced Integration (SRI)
    Bm_0, Bb_0, detJ_0 = _get_mb_matrices(0.0, 0.0, coords_2d)
    Km_1pt = (Bm_0.T @ Dm @ Bm_0) * (4.0 * detJ_0)
    
    Km_full = jnp.zeros((24, 24))
    Kb_full = jnp.zeros((24, 24))
    Ks_full = jnp.zeros((24, 24))

    for xi_g in gp:
        for eta_g in gp:
            Bm, Bb, detJ_mb = _get_mb_matrices(xi_g, eta_g, coords_2d)
            
            N_list = _get_shape_functions(xi_g, eta_g)
            dN_dxi, dN_deta = _get_shape_derivatives(xi_g, eta_g)
            J = jnp.array([
                [jnp.dot(dN_dxi, coords_2d[:,0]), jnp.dot(dN_dxi, coords_2d[:,1])],
                [jnp.dot(dN_deta, coords_2d[:,0]), jnp.dot(dN_deta, coords_2d[:,1])]
            ])
            invJ = jnp.linalg.inv(J)
            detJ2 = jnp.linalg.det(J)
            
            valid_mask = jnp.where(detJ2 < 1e-11, 0.0, 1.0)
            
            dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
            dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
            
            Bd = jnp.zeros((1, 24))
            for i in range(4):
                Bd = Bd.at[0, 6*i].set(-0.5 * dN_dy[i])
                Bd = Bd.at[0, 6*i+1].set(0.5 * dN_dx[i])
                Bd = Bd.at[0, 6*i+5].set(-N_list[i])
            
            # Dynamic drilling stiffness scaling matrix (symmetrical)
            beta_dof = jnp.repeat(beta, 6)
            beta_mat = jnp.sqrt(beta_dof[:, None] * beta_dof[None, :])
            C_drill = 24.9766  # Symmetrically Optimized to match CalculiX
            K_drill += valid_mask * ((Bd.T @ Bd) * beta_mat * (C_drill * G * t) * detJ2)

            Bs13_A = _get_Bs_raw_13(0.0, -1.0, coords_2d)
            Bs13_B = _get_Bs_raw_13(0.0,  1.0, coords_2d)
            Bs13 = 0.5 * (1.0 - eta_g) * Bs13_A + 0.5 * (1.0 + eta_g) * Bs13_B
            
            Bs23_C = _get_Bs_raw_23(-1.0, 0.0, coords_2d)
            Bs23_D = _get_Bs_raw_23( 1.0, 0.0, coords_2d)
            Bs23 = 0.5 * (1.0 - xi_g) * Bs23_C + 0.5 * (1.0 + xi_g) * Bs23_D
            
            Bs = jnp.vstack([Bs13, Bs23])
            
            Km_full += valid_mask * ((Bm.T @ Dm @ Bm) * detJ_mb)
            Kb_full += valid_mask * ((Bb.T @ Db @ Bb) * detJ_mb)
            Ks_full += valid_mask * ((Bs.T @ Ds @ Bs) * detJ_mb)

    alpha = 1e-4
    Km_eff = Km_full
    K_md = Km_eff + K_drill

    # [WHT] Exact Rigid Body Projection for Warped Elements
    coords = jnp.vstack([c1, c2, c3, c4])
    coords_loc = (coords - coords[0]) @ T_mat.T
    
    # Build R_rbm
    R_rbm = jnp.zeros((24, 6))
    for i in range(4):
        px, py, pz = coords_loc[i]
        # Translations
        R_rbm = R_rbm.at[6*i+0, 0].set(1.0)
        R_rbm = R_rbm.at[6*i+1, 1].set(1.0)
        R_rbm = R_rbm.at[6*i+2, 2].set(1.0)
        # Rotations
        R_rbm = R_rbm.at[6*i+0, 4].set(pz)
        R_rbm = R_rbm.at[6*i+0, 5].set(-py)
        R_rbm = R_rbm.at[6*i+1, 3].set(-pz)
        R_rbm = R_rbm.at[6*i+1, 5].set(px)
        R_rbm = R_rbm.at[6*i+2, 3].set(py)
        R_rbm = R_rbm.at[6*i+2, 4].set(-px)
        R_rbm = R_rbm.at[6*i+3, 3].set(1.0)
        R_rbm = R_rbm.at[6*i+4, 4].set(1.0)
        R_rbm = R_rbm.at[6*i+5, 5].set(1.0)
        
    Q, _ = jnp.linalg.qr(R_rbm)
    P_proj = jnp.eye(24) - Q @ Q.T
    
    # Project ONLY membrane and drilling to avoid softening bending
    K_md_proj = P_proj.T @ K_md @ P_proj
    
    K_loc = K_md_proj + Kb_full + Ks_full

    T_24 = jnp.kron(jnp.eye(8), T_mat)
    return T_24.T @ K_loc @ T_24


# ── vmap으로 jit 적용한 배치 버전 ─────────────────────────────────────────────
# in_axes=(0,0,0,0,0,0,0,0): 첫 번째 축(요소 축)으로 배치
_element_K_batch = jax.jit(
    jax.vmap(_element_K_mitc4_plus_jax, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))
)


def K_quad4_jax(wht_model, sorted_nids, nid_to_idx, node_beta=None) -> csr_matrix:
    """
    JAX vmap 배치 조립 — 전체 QUAD4 요소를 단일 XLA 커널로 처리.

    모든 요소의 좌표·재료 배열을 미리 준비한 뒤 _element_K_batch로 (n_elem, 24, 24)
    강성 텐서를 한 번에 계산하고 COO 포맷으로 전역 행렬을 조립한다.
    """
    ndof = len(sorted_nids) * 6

    # ── 노드 좌표 NumPy 배열 구성 (기존 파이썬 루프 제거하고 nodes_array() 활용) ───────────
    node_crds_arr = wht_model.nodes_array()

    # ── 재료/물성 캐시 (리스트 컴프리헨션으로 최적화) ──────────────────────────────
    materials = wht_model.materials
    pid_prop_cache = {
        pid: (
            prop.t if prop else 1.0,
            materials[prop.mid].E if (prop and prop.mid in materials) else 210000.0,
            materials[prop.mid].nu if (prop and prop.mid in materials) else 0.3,
        )
        for pid, prop in wht_model.properties.items()
    }

    # ── QUAD4 요소만 사전 필터링 ────────────────────────────────
    quad_elements = [
        elem for elem in wht_model.elements.values()
        if elem.type.upper() in ('QUAD4', 'QUAD')
    ]

    if not quad_elements:
        return csr_matrix((ndof, ndof))

    # ── 연결 정보 및 물성 일괄 추출 ───────────────────────────
    get_prop = pid_prop_cache.get
    default_prop = (1.0, 210000.0, 0.3)

    conn_list = [[nid_to_idx[nid] for nid in elem.node_ids] for elem in quad_elements]
    props_list = [get_prop(elem.pid, default_prop) for elem in quad_elements]

    # ── NumPy/JAX 어레이 일괄 구성 ────────────────────────────
    conn_arr = np.array(conn_list, dtype=np.int32)   # (n_elem, 4)
    props_arr = np.array(props_list, dtype=np.float64) # (n_elem, 3)

    # 좌표 추출: fancy indexing
    c1 = jnp.array(node_crds_arr[conn_arr[:, 0]], dtype=jnp.float64)
    c2 = jnp.array(node_crds_arr[conn_arr[:, 1]], dtype=jnp.float64)
    c3 = jnp.array(node_crds_arr[conn_arr[:, 2]], dtype=jnp.float64)
    c4 = jnp.array(node_crds_arr[conn_arr[:, 3]], dtype=jnp.float64)

    t_arr  = jnp.array(props_arr[:, 0], dtype=jnp.float64)
    E_arr  = jnp.array(props_arr[:, 1], dtype=jnp.float64)
    nu_arr = jnp.array(props_arr[:, 2], dtype=jnp.float64)

    # beta 추출
    if node_beta is not None:
        beta_arr = jnp.array(np.asarray(node_beta)[conn_arr], dtype=jnp.float64)
    else:
        beta_arr = jnp.full((len(quad_elements), 4), 1e-4, dtype=jnp.float64)

    K_all = np.array(_element_K_batch(c1, c2, c3, c4, t_arr, E_arr, nu_arr, beta_arr))
    # K_all shape: (n_elem, 24, 24)

    # ── COO 어셈블리 ─────────────────────────────────────────────────────────
    dof_arr = (conn_arr[:, :, None] * 6 + np.arange(6, dtype=np.int32)[None, None, :]).reshape(-1, 24)
    n_elem  = conn_arr.shape[0]

    # 각 요소의 24×24 인덱스 쌍을 벡터화로 생성
    ii = np.repeat(dof_arr, 24, axis=1).reshape(n_elem, 24, 24)   # row
    jj = np.tile(dof_arr[:, None, :], (1, 24, 1))                  # col

    rows = ii.ravel()
    cols = jj.ravel()
    data = K_all.ravel()

    return coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
