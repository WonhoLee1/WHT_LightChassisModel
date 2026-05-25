# -*- coding: utf-8 -*-
"""
wht_tria3_element_jax.py
========================
JAX pure-function implementation of the MITC3+ shell element (18 DOF).

Mirrors _element_K_tria3 in wht_tria3_element.py exactly, converted for
JAX auto-differentiation. Used exclusively by wht_topo sensitivity analysis.

Key differences from the numpy version:
  - Python conditionals on traced values replaced with jnp.where / unrolled
  - get_tying_B 3-branch if/elif unrolled to 3 separate constant-index builds
  - Static condensation guard (trace > 1e-12) removed — always apply pinv
  - Degenerate element guard (A < 1e-10) replaced with scalar jnp.where mask
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def _element_K_tria3_jax(c1, c2, c3, t, E, nu):
    """
    MITC3+ element stiffness matrix (18x18) as a JAX pure function.

    Parameters
    ----------
    c1, c2, c3 : (3,) jnp arrays  — global node coordinates
    t, E, nu   : scalar            — thickness, Young's modulus, Poisson's ratio

    Returns
    -------
    (18, 18) jnp array  — global stiffness matrix
    """
    # ── Local coordinate system ─────────────────────────────────────────────
    v12 = c2 - c1
    v13 = c3 - c1
    x_loc = v12 / (jnp.linalg.norm(v12) + 1e-15)
    z_raw = jnp.cross(x_loc, v13)
    z_loc = z_raw / (jnp.linalg.norm(z_raw) + 1e-15)
    y_loc = jnp.cross(z_loc, x_loc)
    T = jnp.stack([x_loc, y_loc, z_loc], axis=0)  # (3, 3) row-basis

    x2 = jnp.dot(v12, x_loc)
    x3 = jnp.dot(v13, x_loc)
    y3 = jnp.dot(v13, y_loc)
    A  = 0.5 * x2 * y3  # signed area in local frame
    A_safe = jnp.where(jnp.abs(A) < 1e-10, 1e-10, A)  # degenerate guard before invJ

    # ── Material constants ──────────────────────────────────────────────────
    G       = E / (2.0 * (1.0 + nu))
    k_shear = 5.0 / 6.0
    inv_nu2 = 1.0 / (1.0 - nu ** 2)

    Dm = (E * t * inv_nu2) * jnp.array(
        [[1., nu, 0.], [nu, 1., 0.], [0., 0., (1. - nu) / 2.]]
    )
    Db = (E * t ** 3 / 12.0 * inv_nu2) * jnp.array(
        [[1., nu, 0.], [nu, 1., 0.], [0., 0., (1. - nu) / 2.]]
    )
    Ds = (k_shear * G * t) * jnp.eye(2)

    # ── Shape function derivatives in local frame (y2 = 0) ─────────────────
    invJ   = 1.0 / (2.0 * A_safe)
    dN_dx  = jnp.array([-(y3), y3, 0.0]) * invJ        # [N1,N2,N3]
    dN_dy  = jnp.array([(x3 - x2), -x3, x2]) * invJ

    # ── Membrane stiffness (constant strain triangle) ───────────────────────
    Bm = jnp.zeros((3, 18))
    for i in range(3):
        Bm = Bm.at[0, 6*i  ].set(dN_dx[i])
        Bm = Bm.at[1, 6*i+1].set(dN_dy[i])
        Bm = Bm.at[2, 6*i  ].set(dN_dy[i])
        Bm = Bm.at[2, 6*i+1].set(dN_dx[i])
    K_loc = (Bm.T @ Dm @ Bm) * A  # (18, 18)

    # ── MITC3+ Tying Operators (fixed tying points — no conditional needed) ─
    # P1: (xi=0.5, eta=0.0)  Edge 1-2
    Bx1 = jnp.zeros(21)
    Bx1 = Bx1.at[2].set(-1.0).at[8].set(1.0)
    Bx1 = Bx1.at[4].set(0.5 * x2).at[10].set(0.5 * x2)

    # P2: (xi=0.0, eta=0.5)  Edge 1-3
    Be2 = jnp.zeros(21)
    Be2 = Be2.at[2].set(-1.0).at[14].set(1.0)
    Be2 = Be2.at[3].set(-0.5 * y3).at[15].set(-0.5 * y3)
    Be2 = Be2.at[4].set(0.5 * x3).at[16].set(0.5 * x3)

    # P3: (xi=0.5, eta=0.5)  Edge 2-3
    Bq3 = jnp.zeros(21)
    Bq3 = Bq3.at[8].set(-1.0).at[14].set(1.0)
    Bq3 = Bq3.at[9].set(-0.5 * y3).at[15].set(-0.5 * y3)
    Bq3 = Bq3.at[10].set(0.5 * (x3 - x2)).at[16].set(0.5 * (x3 - x2))

    B_c = Be2 - Bx1 - Bq3  # (21,)

    # ── 3-point Gauss integration (bending + transverse shear) ─────────────
    K_full = jnp.zeros((21, 21))

    for xi, eta in [(1./6., 1./6.), (2./3., 1./6.), (1./6., 2./3.)]:
        phi_b     = 27.0 * xi * eta * (1 - xi - eta)
        dphi_dxi  = 27.0 * (eta - 2*xi*eta - eta**2)
        dphi_deta = 27.0 * (xi  - xi**2  - 2*xi*eta)
        dphi_dx   = (dphi_dxi * y3 + dphi_deta * (x2 - x3)) / (x2 * y3)
        dphi_dy   = (dphi_deta * x2) / (x2 * y3)

        # Bending B-matrix (3, 21)
        Bb = jnp.zeros((3, 21))
        for i in range(3):
            Bb = Bb.at[0, 6*i+4].set(dN_dx[i])
            Bb = Bb.at[1, 6*i+3].set(-dN_dy[i])
            Bb = Bb.at[2, 6*i+3].set(-dN_dx[i])
            Bb = Bb.at[2, 6*i+4].set(dN_dy[i])
        Bb = Bb.at[0, 20].set(dphi_dx)
        Bb = Bb.at[1, 19].set(-dphi_dy)
        Bb = Bb.at[2, 19].set(-dphi_dx)
        Bb = Bb.at[2, 20].set(dphi_dy)

        # MITC3 interpolated covariant shear strains
        B_gamma_rz = Bx1 + B_c * eta  # (21,)
        B_gamma_sz = Be2 - B_c * xi   # (21,)

        # Map covariant → Cartesian [gamma_xz, gamma_yz]  (2, 21)
        row0 = (y3 * B_gamma_rz - x3 * B_gamma_sz) / (x2 * y3)
        row1 = (x2 * B_gamma_sz) / (x2 * y3)
        B_gamma_hat = jnp.stack([row0, row1], axis=0)

        # Bubble DOF contributions to shear
        B_gamma_hat = B_gamma_hat.at[0, 18].add(dphi_dx)
        B_gamma_hat = B_gamma_hat.at[0, 20].add(phi_b)
        B_gamma_hat = B_gamma_hat.at[1, 18].add(dphi_dy)
        B_gamma_hat = B_gamma_hat.at[1, 19].add(-phi_b)

        K_full = K_full + (
            Bb.T @ Db @ Bb + B_gamma_hat.T @ Ds @ B_gamma_hat
        ) * (A / 3.0)

    # ── Static condensation: 21 → 18 ───────────────────────────────────────
    Kbb = K_full[:18, :18]
    Kii = K_full[18:,  18:]
    Kib = K_full[18:, :18]
    Kbi = K_full[:18,  18:]
    K_loc = K_loc + Kbb - Kbi @ jnp.linalg.pinv(Kii) @ Kib

    # ── Drilling stabilization ──────────────────────────────────────────────
    Ktt = 1.0e-5 * G * t
    Bd = jnp.zeros((1, 18))
    for i in range(3):
        Bd = Bd.at[0, 6*i  ].set(-0.5 * dN_dy[i])
        Bd = Bd.at[0, 6*i+1].set(0.5  * dN_dx[i])
        Bd = Bd.at[0, 6*i+5].set(-1.0 / 3.0)
    K_loc = K_loc + (Bd.T @ Bd) * Ktt * A

    # ── Global transformation ────────────────────────────────────────────────
    T_18 = jnp.zeros((18, 18))
    for i in range(3):
        T_18 = T_18.at[6*i  :6*i+3, 6*i  :6*i+3].set(T)
        T_18 = T_18.at[6*i+3:6*i+6, 6*i+3:6*i+6].set(T)

    result = T_18.T @ K_loc @ T_18

    # Degenerate element guard (A ~ 0 → zero matrix)
    valid = jnp.where(A < 1e-10, 0.0, 1.0)
    return valid * result
