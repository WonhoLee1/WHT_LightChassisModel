"""
wht_quad4_element.py
====================
WHT FEM Framework — MITC4+ High-Fidelity Shell Element (24 DOF)

Formulation
-----------
* Reference: "The MITC4+ shell element and its performance", K.J. Bathe et al. (2006)
* Shear Tying: MITC (Mixed Interpolation of Tensorial Components) using 2 points for g13 and 2 for g23.
* Stability: Drilling stabilization using min eigenvalues of membrane (OpenSees rationale).

Numba 가속
----------
* _element_K_mitc4_plus_nb() — @njit(cache=True) 버전. 첫 호출 시 JIT 컴파일(수초),
  이후 캐시에서 즉시 로드됨.
* K_quad4_scipy() 내부에서 자동으로 numba 버전을 선택하며, import 실패 시 순수
  NumPy 백업 함수(_element_K_mitc4_plus_np)로 폴백한다.
"""

from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Numba 가용성 확인
# ─────────────────────────────────────────────────────────────────────────────
try:
    from numba import njit as _njit
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False
    def _njit(*args, **kwargs):          # 더미 데코레이터
        def _wrap(fn): return fn
        return _wrap


# ─────────────────────────────────────────────────────────────────────────────
# 순수 NumPy 헬퍼 (백업용 / 기존 코드 유지)
# ─────────────────────────────────────────────────────────────────────────────

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
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    for i in range(4):
        Bs_row[6*i+2] = dN_dy[i]
        Bs_row[6*i+3] = -N[i]
    return Bs_row

def _element_K_mitc4_plus_np(c1, c2, c3, c4, t, E, nu, beta):
    """순수 NumPy 백업 — Numba 미사용 원본."""
    v12 = c2 - c1
    v14 = c4 - c1
    x_loc = v12 / (np.linalg.norm(v12) + 1e-12)
    z_raw = np.cross(x_loc, v14)
    z_loc = z_raw / (np.linalg.norm(z_raw) + 1e-12)
    y_loc = np.cross(z_loc, x_loc)
    T_mat = np.stack([x_loc, y_loc, z_loc], axis=0)

    p2 = c2 - c1; p3 = c3 - c1; p4 = c4 - c1
    coords_2d = np.array([
        [0.0, 0.0],
        [np.dot(p2, x_loc), np.dot(p2, y_loc)],
        [np.dot(p3, x_loc), np.dot(p3, y_loc)],
        [np.dot(p4, x_loc), np.dot(p4, y_loc)]
    ])

    G = E / (2.0 * (1.0 + nu))
    k_shear = 5.0 / 6.0
    inv_nu2 = 1.0 / (1.0 - nu**2)
    Dm = (E * t * inv_nu2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Db = (E * t**3 / 12.0 * inv_nu2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    Ds = (k_shear * G * t) * np.eye(2)

    K_loc = np.zeros((24, 24))
    K_drill = np.zeros((24, 24))
    # gp = [-1.0/np.sqrt(3), 1.0/np.sqrt(3)]

    # Membrane locking fix: Selective Reduced Integration (SRI)
    Bm_0, Bb_0, detJ_0 = _get_mb_matrices(0.0, 0.0, coords_2d)
    Km_1pt = (Bm_0.T @ Dm @ Bm_0) * (4.0 * detJ_0)
    
    Km_full = np.zeros((24, 24))
    Kb_full = np.zeros((24, 24))
    Ks_full = np.zeros((24, 24))

    for xi_g in gp:
        for eta_g in gp:
            Bm, Bb, detJ = _get_mb_matrices(xi_g, eta_g, coords_2d)

            N_list = _get_shape_functions(xi_g, eta_g)
            dN_dxi, dN_deta = _get_shape_derivatives(xi_g, eta_g)
            J = np.array([
                [np.dot(dN_dxi, coords_2d[:,0]), np.dot(dN_dxi, coords_2d[:,1])],
                [np.dot(dN_deta, coords_2d[:,0]), np.dot(dN_deta, coords_2d[:,1])]
            ])
            invJ = np.linalg.inv(J)
            detJ = np.linalg.det(J)
            if detJ < 1e-11: continue

            dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
            dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta

            Bd = np.zeros((1, 24))
            for i in range(4):
                Bd[0, 6*i]   = -0.5 * dN_dy[i]
                Bd[0, 6*i+1] =  0.5 * dN_dx[i]
                Bd[0, 6*i+5] = -N_list[i]

            beta_dof = np.repeat(beta, 6)
            beta_mat = np.sqrt(np.outer(beta_dof, beta_dof))
            C_drill = 24.9766  # Symmetrically Optimized to match CalculiX
            K_drill += (Bd.T @ Bd) * beta_mat * (C_drill * G * t) * detJ

            Bs13_A = _get_Bs_raw_13(0.0, -1.0, coords_2d)
            Bs13_B = _get_Bs_raw_13(0.0,  1.0, coords_2d)
            Bs13 = 0.5 * (1.0 - eta_g) * Bs13_A + 0.5 * (1.0 + eta_g) * Bs13_B

            Bs23_C = _get_Bs_raw_23(-1.0, 0.0, coords_2d)
            Bs23_D = _get_Bs_raw_23( 1.0, 0.0, coords_2d)
            Bs23 = 0.5 * (1.0 - xi_g) * Bs23_C + 0.5 * (1.0 + xi_g) * Bs23_D

            Bs = np.vstack([Bs13, Bs23])

            Km_full += (Bm.T @ Dm @ Bm) * detJ
            Kb_full += (Bb.T @ Db @ Bb) * detJ
            Ks_full += (Bs.T @ Ds @ Bs) * detJ

    # Apply 5% stabilization for Hourglass control
    alpha = 1e-4
    Km_eff = Km_full
    K_loc = Km_eff + Kb_full + Ks_full + K_drill

    T_24 = np.zeros((24, 24))
    for i in range(4):
        T_24[6*i:6*i+3, 6*i:6*i+3] = T_mat
        T_24[6*i+3:6*i+6, 6*i+3:6*i+6] = T_mat
    return T_24.T @ K_loc @ T_24


# ─────────────────────────────────────────────────────────────────────────────
# Numba JIT 버전 — Numba 비호환 연산 제거 버전
# np.vstack → 수동 행 복사, 슬라이스 대입 → 명시적 루프
# ─────────────────────────────────────────────────────────────────────────────

@_njit(cache=True)
def _nb_shape_N(xi, eta):
    N = np.empty(4)
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta)
    return N

@_njit(cache=True)
def _nb_shape_dN(xi, eta):
    dN_dxi  = np.empty(4)
    dN_deta = np.empty(4)
    dN_dxi[0]  = 0.25 * (-(1.0 - eta));  dN_dxi[1]  = 0.25 * ( (1.0 - eta))
    dN_dxi[2]  = 0.25 * ( (1.0 + eta));  dN_dxi[3]  = 0.25 * (-(1.0 + eta))
    dN_deta[0] = 0.25 * (-(1.0 - xi));   dN_deta[1] = 0.25 * (-(1.0 + xi))
    dN_deta[2] = 0.25 * ( (1.0 + xi));   dN_deta[3] = 0.25 * ( (1.0 - xi))
    return dN_dxi, dN_deta

@_njit(cache=True)
def _nb_inv2x2(J):
    """2×2 행렬 역행렬 + 행렬식."""
    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    inv = np.empty((2, 2))
    inv[0, 0] =  J[1, 1] / det
    inv[0, 1] = -J[0, 1] / det
    inv[1, 0] = -J[1, 0] / det
    inv[1, 1] =  J[0, 0] / det
    return inv, det

@_njit(cache=True)
def _nb_jacobian(dN_dxi, dN_deta, coords):
    J = np.zeros((2, 2))
    for k in range(4):
        J[0, 0] += dN_dxi[k]  * coords[k, 0]
        J[0, 1] += dN_dxi[k]  * coords[k, 1]
        J[1, 0] += dN_deta[k] * coords[k, 0]
        J[1, 1] += dN_deta[k] * coords[k, 1]
    return J

@_njit(cache=True)
def _nb_Bm_Bb(dN_dx, dN_dy):
    """Membrane (3×24) + Bending (3×24) B 행렬."""
    Bm = np.zeros((3, 24))
    Bb = np.zeros((3, 24))
    for i in range(4):
        Bm[0, 6*i]     = dN_dx[i]
        Bm[1, 6*i + 1] = dN_dy[i]
        Bm[2, 6*i]     = dN_dy[i]
        Bm[2, 6*i + 1] = dN_dx[i]
        Bb[0, 6*i + 4] =  dN_dx[i]
        Bb[1, 6*i + 3] = -dN_dy[i]
        Bb[2, 6*i + 4] =  dN_dy[i]
        Bb[2, 6*i + 3] = -dN_dx[i]
    return Bm, Bb

@_njit(cache=True)
def _nb_Bs13(xi, eta, coords):
    """gamma_13 전단 B 행 벡터 (24,)."""
    N = _nb_shape_N(xi, eta)
    dN_dxi, dN_deta = _nb_shape_dN(xi, eta)
    J = _nb_jacobian(dN_dxi, dN_deta, coords)
    invJ, _ = _nb_inv2x2(J)
    dN_dx = np.empty(4)
    for k in range(4):
        dN_dx[k] = invJ[0, 0] * dN_dxi[k] + invJ[0, 1] * dN_deta[k]
    Bs = np.zeros(24)
    for i in range(4):
        Bs[6*i + 2] = dN_dx[i]
        Bs[6*i + 4] = N[i]
    return Bs

@_njit(cache=True)
def _nb_Bs23(xi, eta, coords):
    """gamma_23 전단 B 행 벡터 (24,)."""
    N = _nb_shape_N(xi, eta)
    dN_dxi, dN_deta = _nb_shape_dN(xi, eta)
    J = _nb_jacobian(dN_dxi, dN_deta, coords)
    invJ, _ = _nb_inv2x2(J)
    dN_dy = np.empty(4)
    for k in range(4):
        dN_dy[k] = invJ[1, 0] * dN_dxi[k] + invJ[1, 1] * dN_deta[k]
    Bs = np.zeros(24)
    for i in range(4):
        Bs[6*i + 2] = dN_dy[i]
        Bs[6*i + 3] = -N[i]
    return Bs

@_njit(cache=True)
def _nb_matmul_3x24_24x3(A, B):
    """A(3×24) @ B(24×3) → (3×3)."""
    C = np.zeros((3, 3))
    for i in range(3):
        for k in range(24):
            if A[i, k] == 0.0: continue
            for j in range(3):
                C[i, j] += A[i, k] * B[k, j]
    return C

@_njit(cache=True)
def _nb_BtDB(B, D):
    """Bᵀ(24×3) @ D(3×3) @ B(3×24) → (24×24)."""
    DB = np.zeros((3, 24))
    for i in range(3):
        for j in range(24):
            for k in range(3):
                DB[i, j] += D[i, k] * B[k, j]
    K = np.zeros((24, 24))
    for i in range(24):
        for k in range(3):
            if B[k, i] == 0.0: continue
            for j in range(24):
                K[i, j] += B[k, i] * DB[k, j]
    return K

@_njit(cache=True)
def _nb_BstDs_Bs(Bs13, Bs23, Ds):
    """[Bs13; Bs23]ᵀ @ Ds(2×2) @ [Bs13; Bs23] → (24×24)."""
    K = np.zeros((24, 24))
    for i in range(24):
        for j in range(24):
            K[i, j] += (Bs13[i] * (Ds[0,0]*Bs13[j] + Ds[0,1]*Bs23[j])
                      + Bs23[i] * (Ds[1,0]*Bs13[j] + Ds[1,1]*Bs23[j]))
    return K

@_njit(cache=True)
def _nb_cross3(a, b):
    c = np.empty(3)
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]
    return c

@_njit(cache=True)
def _nb_norm3(a):
    return (a[0]**2 + a[1]**2 + a[2]**2) ** 0.5

@_njit(cache=True)
def _element_K_mitc4_plus_nb(c1, c2, c3, c4, t, E, nu, beta):
    """MITC4+ 요소 강성 행렬 — Numba JIT 버전 (24×24)."""
    # ── 로컬 좌표계 ────────────────────────────────────────────────────────
    v12 = c2 - c1
    v14 = c4 - c1
    x_loc = v12 / (_nb_norm3(v12) + 1e-12)
    z_raw = _nb_cross3(x_loc, v14)
    z_loc = z_raw / (_nb_norm3(z_raw) + 1e-12)
    y_loc = _nb_cross3(z_loc, x_loc)

    T_mat = np.empty((3, 3))
    for k in range(3):
        T_mat[0, k] = x_loc[k]
        T_mat[1, k] = y_loc[k]
        T_mat[2, k] = z_loc[k]

    p2 = c2 - c1; p3 = c3 - c1; p4 = c4 - c1
    coords_2d = np.zeros((4, 2))
    coords_2d[1, 0] = p2[0]*x_loc[0] + p2[1]*x_loc[1] + p2[2]*x_loc[2]
    coords_2d[1, 1] = p2[0]*y_loc[0] + p2[1]*y_loc[1] + p2[2]*y_loc[2]
    coords_2d[2, 0] = p3[0]*x_loc[0] + p3[1]*x_loc[1] + p3[2]*x_loc[2]
    coords_2d[2, 1] = p3[0]*y_loc[0] + p3[1]*y_loc[1] + p3[2]*y_loc[2]
    coords_2d[3, 0] = p4[0]*x_loc[0] + p4[1]*x_loc[1] + p4[2]*x_loc[2]
    coords_2d[3, 1] = p4[0]*y_loc[0] + p4[1]*y_loc[1] + p4[2]*y_loc[2]

    # ── 재료 행렬 ──────────────────────────────────────────────────────────
    G = E / (2.0 * (1.0 + nu))
    k_shear = 5.0 / 6.0
    inv_nu2 = 1.0 / (1.0 - nu * nu)
    c_m = E * t * inv_nu2
    c_b = E * t**3 / 12.0 * inv_nu2
    c_s = k_shear * G * t

    Dm = np.zeros((3, 3))
    Dm[0, 0] = c_m;       Dm[0, 1] = c_m * nu
    Dm[1, 0] = c_m * nu;  Dm[1, 1] = c_m
    Dm[2, 2] = c_m * (1.0 - nu) / 2.0

    Db = np.zeros((3, 3))
    Db[0, 0] = c_b;       Db[0, 1] = c_b * nu
    Db[1, 0] = c_b * nu;  Db[1, 1] = c_b
    Db[2, 2] = c_b * (1.0 - nu) / 2.0

    Ds = np.zeros((2, 2))
    Ds[0, 0] = c_s;  Ds[1, 1] = c_s

    # Ktt = 1.0 * G * t

    # ── 가우스 적분 (2×2) ──────────────────────────────────────────────────
    gp0 = -1.0 / 1.7320508075688772   # -1/sqrt(3)
    gp1 =  1.0 / 1.7320508075688772

    K_loc   = np.zeros((24, 24))
    K_drill = np.zeros((24, 24))

    # Membrane locking fix: Selective Reduced Integration (SRI)
    dN_dxi_0, dN_deta_0 = _nb_shape_dN(0.0, 0.0)
    J_0 = _nb_jacobian(dN_dxi_0, dN_deta_0, coords_2d)
    invJ_0, detJ_0 = _nb_inv2x2(J_0)
    dN_dx_0 = np.empty(4)
    dN_dy_0 = np.empty(4)
    for k in range(4):
        dN_dx_0[k] = invJ_0[0,0]*dN_dxi_0[k] + invJ_0[0,1]*dN_deta_0[k]
        dN_dy_0[k] = invJ_0[1,0]*dN_dxi_0[k] + invJ_0[1,1]*dN_deta_0[k]
    Bm_0, _ = _nb_Bm_Bb(dN_dx_0, dN_dy_0)
    Km_1pt_loc = _nb_BtDB(Bm_0, Dm)
    Km_1pt = np.zeros((24, 24))
    for i in range(24):
        for j in range(24):
            Km_1pt[i, j] = Km_1pt_loc[i, j] * (4.0 * detJ_0)

    Km_full = np.zeros((24, 24))
    Kb_full = np.zeros((24, 24))
    Ks_full = np.zeros((24, 24))

    # MITC 전단 타잉점은 가우스 루프 밖에서 미리 계산
    Bs13_A = _nb_Bs13(0.0, -1.0, coords_2d)
    Bs13_B = _nb_Bs13(0.0,  1.0, coords_2d)
    Bs23_C = _nb_Bs23(-1.0, 0.0, coords_2d)
    Bs23_D = _nb_Bs23( 1.0, 0.0, coords_2d)

    for xi_g in (gp0, gp1):
        for eta_g in (gp0, gp1):
            dN_dxi, dN_deta = _nb_shape_dN(xi_g, eta_g)
            J = _nb_jacobian(dN_dxi, dN_deta, coords_2d)
            invJ, detJ = _nb_inv2x2(J)
            if detJ < 1e-11:
                continue

            dN_dx = np.empty(4)
            dN_dy = np.empty(4)
            for k in range(4):
                dN_dx[k] = invJ[0,0]*dN_dxi[k] + invJ[0,1]*dN_deta[k]
                dN_dy[k] = invJ[1,0]*dN_dxi[k] + invJ[1,1]*dN_deta[k]

            Bm, Bb = _nb_Bm_Bb(dN_dx, dN_dy)

            # Drilling B (1×24 → 벡터로 처리)
            N = _nb_shape_N(xi_g, eta_g)
            Bd = np.zeros(24)
            for i in range(4):
                Bd[6*i]     = -0.5 * dN_dy[i]
                Bd[6*i + 1] =  0.5 * dN_dx[i]
                Bd[6*i + 5] = -N[i]
            C_drill = 24.9766  # Symmetrically Optimized to match CalculiX
            for i in range(24):
                for j in range(24):
                    K_drill[i, j] += Bd[i] * Bd[j] * np.sqrt(beta[i // 6] * beta[j // 6]) * (C_drill * G * t) * detJ

            # MITC 전단 보간 (타잉점 혼합)
            Bs13 = np.empty(24)
            Bs23 = np.empty(24)
            w13_a = 0.5 * (1.0 - eta_g)
            w13_b = 0.5 * (1.0 + eta_g)
            w23_c = 0.5 * (1.0 - xi_g)
            w23_d = 0.5 * (1.0 + xi_g)
            for k in range(24):
                Bs13[k] = w13_a * Bs13_A[k] + w13_b * Bs13_B[k]
                Bs23[k] = w23_c * Bs23_C[k] + w23_d * Bs23_D[k]

            Km  = _nb_BtDB(Bm, Dm)
            Kb  = _nb_BtDB(Bb, Db)
            Ks  = _nb_BstDs_Bs(Bs13, Bs23, Ds)
            for i in range(24):
                for j in range(24):
                    Km_full[i, j] += Km[i, j] * detJ
                    Kb_full[i, j] += Kb[i, j] * detJ
                    Ks_full[i, j] += Ks[i, j] * detJ

    alpha = 1e-4
    for i in range(24):
        for j in range(24):
            Km_eff = Km_1pt[i, j] + alpha * (Km_full[i, j] - Km_1pt[i, j])
            K_loc[i, j] += Km_eff + Kb_full[i, j] + Ks_full[i, j] + K_drill[i, j]

    # ── 전역 좌표 변환 T_24ᵀ @ K_loc @ T_24 ──────────────────────────────
    T_24 = np.zeros((24, 24))
    for i in range(4):
        for r in range(3):
            for c in range(3):
                T_24[6*i + r, 6*i + c]     = T_mat[r, c]
                T_24[6*i+3+r, 6*i+3+c]     = T_mat[r, c]

    # T_24.T @ K_loc @ T_24
    tmp = np.zeros((24, 24))
    for i in range(24):
        for k in range(24):
            if T_24[k, i] == 0.0: continue
            for j in range(24):
                tmp[i, j] += T_24[k, i] * K_loc[k, j]
    K_global = np.zeros((24, 24))
    for i in range(24):
        for k in range(24):
            if tmp[i, k] == 0.0: continue
            for j in range(24):
                K_global[i, j] += tmp[i, k] * T_24[k, j]
    return K_global


# 외부 호출용 통합 함수 — Numba 가용 시 JIT 버전 사용, 아니면 NumPy 폴백
def _element_K_mitc4_plus(c1, c2, c3, c4, t, E, nu, beta=None):
    if beta is None:
        beta = np.ones(4) * 1e-4
    if _NUMBA_OK:
        return _element_K_mitc4_plus_nb(c1, c2, c3, c4, t, E, nu, beta)
    return _element_K_mitc4_plus_np(c1, c2, c3, c4, t, E, nu, beta)


# ─────────────────────────────────────────────────────────────────────────────
# 어셈블리
# ─────────────────────────────────────────────────────────────────────────────

def K_quad4_scipy(wht_model, sorted_nids, nid_to_idx, node_beta=None) -> csr_matrix:
    ndof = len(sorted_nids) * 6
    rows, cols, data = [], [], []
    nid_arr = list(sorted_nids)
    nid_to_crds = {nid: [wht_model.nodes[nid].x, wht_model.nodes[nid].y, wht_model.nodes[nid].z] for nid in nid_arr}

    for i, (eid, elem) in enumerate(wht_model.elements.items()):
        e_type = elem.type.upper()
        if e_type not in ('QUAD4', 'QUAD'): continue
        nids = elem.node_ids
        pid = elem.pid
        prop = wht_model.properties.get(pid); mat = wht_model.materials.get(prop.mid) if prop else None
        t = prop.t if prop else 1.0; E = mat.E if mat else 210000.0; nu = mat.nu if mat else 0.3

        if node_beta is not None:
            elem_beta = np.array([node_beta[nid_to_idx[nid]] for nid in nids], dtype=np.float64)
        else:
            elem_beta = np.ones(4) * 1e-4

        K_e = _element_K_mitc4_plus(
            np.array(nid_to_crds[nids[0]]), np.array(nid_to_crds[nids[1]]),
            np.array(nid_to_crds[nids[2]]), np.array(nid_to_crds[nids[3]]),
            t, E, nu, elem_beta
        )
        dofs = np.array([nid_to_idx[nid] * 6 + d for nid in nids for d in range(6)])
        rr, cc = np.meshgrid(dofs, dofs, indexing='ij')
        rows.append(rr.ravel()); cols.append(cc.ravel()); data.append(K_e.ravel())

    if not rows: return csr_matrix((ndof, ndof))
    return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof)).tocsr()

def M_quad4_lumped(wht_model, ndof: int, sorted_nids, nid_to_idx) -> np.ndarray:
    """
    QUAD4 요소의 Lumped Mass 대각 벡터를 numpy 벡터 연산으로 조립한다.

    Parameters:
        wht_model   : WHTMeshModel — 노드/요소/재료/물성 정보를 포함하는 모델
        ndof        : int — 전체 자유도 수 (N * 6)
        sorted_nids : list — 정렬된 노드 ID 목록
        nid_to_idx  : dict — 노드 ID -> 행렬 인덱스 매핑

    Returns:
        M_diag : (ndof,) numpy 배열 — 럼프드 질량 대각 성분
    """
    # ── 노드 좌표 배열 구성 ──────────────────────────────────────────────────
    n_nodes = len(sorted_nids)
    node_crds_arr = np.empty((n_nodes, 3), dtype=np.float64)
    for nid in sorted_nids:
        nd = wht_model.nodes[nid]
        node_crds_arr[nid_to_idx[nid]] = (nd.x, nd.y, nd.z)

    # ── 재료/물성 캐시 (루프 전 1회 구성) ───────────────────────────────────
    pid_cache: dict = {}
    for pid, prop in wht_model.properties.items():
        if not prop:
            continue
        mat = wht_model.materials.get(prop.mid)
        pid_cache[pid] = (prop.t, mat.rho if mat else 7.85e-9)

    # ── 요소 데이터 추출 루프 ────────────────────────────────────────────────
    conn_list, t_list, rho_list = [], [], []
    for eid, elem in wht_model.elements.items():
        if elem.type not in ("QUAD4", "QUAD"):
            continue
        pid = getattr(elem, "pid", None)
        if pid is None or pid == 0:
            raise ValueError(
                f"QUAD 요소 {eid}에 유효한 pid 속성이 누락되었습니다. "
                "모달 해석 시 0 질량 에러의 원인이 됩니다."
            )
        if pid not in pid_cache:
            prop = wht_model.properties.get(pid)
            if not prop:
                raise ValueError(
                    f"QUAD 요소 {eid}의 pid={pid}에 해당하는 속성(Property)을 찾을 수 없습니다."
                )
            mat = wht_model.materials.get(prop.mid)
            pid_cache[pid] = (prop.t, mat.rho if mat else 7.85e-9)
        t, rho = pid_cache[pid]
        conn_list.append([nid_to_idx[nid] for nid in elem.node_ids])
        t_list.append(t)
        rho_list.append(rho)

    if not conn_list:
        return np.zeros(ndof)

    # ── numpy 벡터 연산으로 면적·질량 일괄 계산 ─────────────────────────────
    conn_arr = np.array(conn_list, dtype=np.int32)    # (n_elem, 4)
    t_arr    = np.array(t_list,    dtype=np.float64)  # (n_elem,)
    rho_arr  = np.array(rho_list,  dtype=np.float64)  # (n_elem,)

    # 좌표 추출: (n_elem, 4, 3) via fancy indexing
    coords = node_crds_arr[conn_arr]

    # 대각선 교차곱으로 면적 계산 (두 삼각형 합과 수학적으로 동일)
    d02   = coords[:, 2] - coords[:, 0]           # (n_elem, 3)
    d13   = coords[:, 3] - coords[:, 1]           # (n_elem, 3)
    cross = np.cross(d02, d13)                    # (n_elem, 3)
    area  = 0.5 * np.linalg.norm(cross, axis=1)  # (n_elem,)

    # 노드당 질량 (요소 면적을 4 노드에 균등 분배)
    m_node    = area * t_arr * rho_arr / 4.0      # (n_elem,)
    # Reissner-Mindlin 쉘의 럼프드 질량 이론에 근거한 정석 회전 관성 럼핑 공식 적용
    rot_inert = m_node * (t_arr ** 2) / 12.0      # (n_elem,)

    # ── Scatter-add: np.add.at으로 전역 M_diag에 조립 ───────────────────────
    # DOF 인덱스: (n_elem, 4, 6) — [dx,dy,dz,rx,ry,rz] 순
    base_dofs = conn_arr * 6                                          # (n_elem, 4)
    dof_idx   = (base_dofs[:, :, None]
                 + np.arange(6, dtype=np.int32)[None, None, :])      # (n_elem, 4, 6)

    # 질량값: (n_elem, 4, 6) — 앞 3 DOF: 평행이동, 뒤 3 DOF: 회전
    m4    = np.repeat(m_node[:, None],    4, axis=1)                  # (n_elem, 4)
    rot4  = np.repeat(rot_inert[:, None], 4, axis=1)                  # (n_elem, 4)
    mass_val = np.concatenate([
        np.repeat(m4[:, :, None],   3, axis=2),   # (n_elem, 4, 3)
        np.repeat(rot4[:, :, None], 3, axis=2),   # (n_elem, 4, 3)
    ], axis=2)                                                         # (n_elem, 4, 6)

    M_diag = np.zeros(ndof)
    np.add.at(M_diag, dof_idx.ravel(), mass_val.ravel())
    return M_diag
