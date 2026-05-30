# -*- coding: utf-8 -*-
"""
wht_tria3_element_jax.py
========================
JAX pure-function implementation of the CalculiX US3 (CS-DSG + ANDES) flat shell element.
Mirrors wht_tria3_element.py exactly, converted for JAX auto-differentiation.
Used exclusively by wht_topo sensitivity analysis.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

@jax.jit
def us3_csys_cr_jax(xg):
    e1 = xg[1] - xg[0]
    dl1 = jnp.linalg.norm(e1) + 1e-15
    e1 = e1 / dl1
    
    e2 = xg[2] - xg[0]
    e3 = jnp.cross(e1, e2)
    dl3 = jnp.linalg.norm(e3) + 1e-15
    e3 = e3 / dl3
    
    e2 = jnp.cross(e3, e1)
    dl2 = jnp.linalg.norm(e2) + 1e-15
    e2 = e2 / dl2
    
    tm = jnp.stack([e1, e2, e3], axis=0)
    
    tmg = jnp.zeros((18, 18))
    for i in range(6):
        tmg = tmg.at[3*i:3*i+3, 3*i:3*i+3].set(tm)
    return tm, tmg

@jax.jit
def us3_linel_Qi_jax(E, nu):
    Qin = jnp.zeros((3, 3))
    q1 = E / (1.0 - nu**2)
    Qin = Qin.at[0, 0].set(q1)
    Qin = Qin.at[0, 1].set(q1 * nu)
    Qin = Qin.at[1, 0].set(q1 * nu)
    Qin = Qin.at[1, 1].set(q1)
    Qin = Qin.at[2, 2].set(q1 * (1.0 - nu) / 2.0)
    
    Qs = jnp.zeros((2, 2))
    kap = 5.0 / 6.0
    q1_s = E / (2.0 * (1.0 + nu))
    Qs = Qs.at[0, 0].set(q1_s * kap)
    Qs = Qs.at[1, 1].set(q1_s * kap)
    return Qin, Qs

@jax.jit
def us3_CS_jax(X, Y):
    x21 = X[1] - X[0]
    x13 = X[0] - X[2]
    y31 = Y[2] - Y[0]
    y12 = Y[0] - Y[1]
    x32 = X[2] - X[1]
    y23 = Y[1] - Y[2]
    
    Ae = 0.5 * (x21 * y31 - x13 * y12)
    Ae_safe = jnp.where(jnp.abs(Ae) < 1e-12, 1e-12, Ae)
    
    a1 = 0.5 * y12 * x13
    a2 = 0.5 * y31 * x21
    a3 = 0.5 * x21 * x13
    a4 = 0.5 * y12 * y31
    
    bs1 = jnp.zeros((2, 6))
    bs1 = bs1.at[0, 2].set(0.5 * x32 / Ae_safe)
    bs1 = bs1.at[0, 3].set(-0.5)
    bs1 = bs1.at[1, 2].set(0.5 * y23 / Ae_safe)
    bs1 = bs1.at[1, 4].set(0.5)
    
    bs2 = jnp.zeros((2, 6))
    bs2 = bs2.at[0, 2].set(0.5 * x13 / Ae_safe)
    bs2 = bs2.at[0, 3].set(0.5 * a1 / Ae_safe)
    bs2 = bs2.at[0, 4].set(0.5 * a3 / Ae_safe)
    bs2 = bs2.at[1, 2].set(0.5 * y31 / Ae_safe)
    bs2 = bs2.at[1, 3].set(0.5 * a4 / Ae_safe)
    bs2 = bs2.at[1, 4].set(0.5 * a2 / Ae_safe)
    
    bs3 = jnp.zeros((2, 6))
    bs3 = bs3.at[0, 2].set(0.5 * x21 / Ae_safe)
    bs3 = bs3.at[0, 3].set(-0.5 * a2 / Ae_safe)
    bs3 = bs3.at[0, 4].set(-0.5 * a3 / Ae_safe)
    bs3 = bs3.at[1, 2].set(0.5 * y12 / Ae_safe)
    bs3 = bs3.at[1, 3].set(-0.5 * a4 / Ae_safe)
    bs3 = bs3.at[1, 4].set(-0.5 * a1 / Ae_safe)
    
    return bs1, bs2, bs3, Ae

@jax.jit
def us3_Bs_jax(X):
    x1, x2, x3 = X[0, 0], X[1, 0], X[2, 0]
    y1, y2, y3 = X[0, 1], X[1, 1], X[2, 1]
    x31 = x3 - x1
    y31 = y3 - y1
    y12 = y1 - y2
    x21 = x2 - x1
    
    Ae = 0.5 * (x21 * y31 - x31 * (-y12))
    Ae_safe = jnp.where(jnp.abs(Ae) < 1e-12, 1e-12, Ae)
    
    x0 = (x1 + x2 + x3) / 3.0
    y0 = (y1 + y2 + y3) / 3.0
    
    X_1 = jnp.array([x0, x1, x2])
    Y_1 = jnp.array([y0, y1, y2])
    X_2 = jnp.array([x0, x2, x3])
    Y_2 = jnp.array([y0, y2, y3])
    X_3 = jnp.array([x0, x3, x1])
    Y_3 = jnp.array([y0, y3, y1])
    
    a3 = 1.0 / 3.0
    
    # sub-tri 1
    bs1, bs2, bs3, Ai = us3_CS_jax(X_1, Y_1)
    B1 = jnp.zeros((2, 18))
    B1 = B1.at[:, 0:6].set(a3 * bs1 + bs2)
    B1 = B1.at[:, 6:12].set(a3 * bs1 + bs3)
    B1 = B1.at[:, 12:18].set(a3 * bs1)
    B1 = B1 * Ai
    
    # sub-tri 2
    bs1, bs2, bs3, Ai = us3_CS_jax(X_2, Y_2)
    B2 = jnp.zeros((2, 18))
    B2 = B2.at[:, 0:6].set(a3 * bs1)
    B2 = B2.at[:, 6:12].set(a3 * bs1 + bs2)
    B2 = B2.at[:, 12:18].set(a3 * bs1 + bs3)
    B2 = B2 * Ai
    
    # sub-tri 3
    bs1, bs2, bs3, Ai = us3_CS_jax(X_3, Y_3)
    B3 = jnp.zeros((2, 18))
    B3 = B3.at[:, 0:6].set(a3 * bs1 + bs3)
    B3 = B3.at[:, 6:12].set(a3 * bs1)
    B3 = B3.at[:, 12:18].set(a3 * bs1 + bs2)
    B3 = B3 * Ai
    
    Bs = (1.0 / Ae_safe) * (B1 + B2 + B3)
    return Bs

@jax.jit
def us3_Bb_jax(X, Y):
    x21 = X[1] - X[0]
    x31 = X[2] - X[0]
    x32 = X[2] - X[1]
    y23 = Y[1] - Y[2]
    y31 = Y[2] - Y[0]
    y12 = Y[0] - Y[1]
    
    Ae = 0.5 * (x21 * y31 - x31 * (-y12))
    Ae_safe = jnp.where(jnp.abs(Ae) < 1e-12, 1e-12, Ae)
    
    dNdx1 = y23 / (2.0 * Ae_safe)
    dNdy1 = x32 / (2.0 * Ae_safe)
    dNdx2 = y31 / (2.0 * Ae_safe)
    dNdy2 = -x31 / (2.0 * Ae_safe)
    dNdx3 = y12 / (2.0 * Ae_safe)
    dNdy3 = x21 / (2.0 * Ae_safe)
    
    bb = jnp.zeros((3, 18))
    bb = bb.at[0, 4].set(dNdx1)
    bb = bb.at[0, 10].set(dNdx2)
    bb = bb.at[0, 16].set(dNdx3)
    
    bb = bb.at[1, 3].set(-dNdy1)
    bb = bb.at[1, 9].set(-dNdy2)
    bb = bb.at[1, 15].set(-dNdy3)
    
    bb = bb.at[2, 3].set(-dNdx1)
    bb = bb.at[2, 4].set(dNdy1)
    bb = bb.at[2, 9].set(-dNdx2)
    bb = bb.at[2, 10].set(dNdy2)
    bb = bb.at[2, 15].set(-dNdx3)
    bb = bb.at[2, 16].set(dNdy3)
    return bb

@jax.jit
def us3_Kp_jax(X, Db, Ds):
    x21 = X[1, 0] - X[0, 0]
    x31 = X[2, 0] - X[0, 0]
    y31 = X[2, 1] - X[0, 1]
    y12 = X[0, 1] - X[1, 1]
    Ae = 0.5 * (x21 * y31 - x31 * (-y12))
    
    Bs = us3_Bs_jax(X)
    Bb = us3_Bb_jax(X[:, 0], X[:, 1])
    
    Kp = (Bs.T @ Ds @ Bs + Bb.T @ Db @ Bb) * Ae
    return Kp

@jax.jit
def us3_Km_jax(X, Qin, h):
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
    Ae_safe = jnp.where(jnp.abs(Ae) < 1e-12, 1e-12, Ae)
    
    A2 = 2.0 * Ae
    A4 = 4.0 * Ae
    h2 = 0.5 * h
    V = Ae * h
    
    LL21 = x21**2 + y21**2
    LL32 = x32**2 + y32**2
    LL13 = x13**2 + y13**2
    
    L = jnp.zeros((9, 3))
    L = L.at[0, 0].set(h2 * y23)
    L = L.at[0, 2].set(h2 * x32)
    L = L.at[1, 1].set(h2 * x32)
    L = L.at[1, 2].set(h2 * y23)
    L = L.at[2, 0].set(h2 * y23 * (y13 - y21) * ab)
    L = L.at[2, 1].set(h2 * x32 * (x31 - x12) * ab)
    L = L.at[2, 2].set(h2 * (x31 * y13 - x12 * y21) * 2.0 * ab)
    
    L = L.at[3, 0].set(h2 * y31)
    L = L.at[3, 2].set(h2 * x13)
    L = L.at[4, 1].set(h2 * x13)
    L = L.at[4, 2].set(h2 * y31)
    L = L.at[5, 0].set(h2 * y31 * (y21 - y32) * ab)
    L = L.at[5, 1].set(h2 * x13 * (x12 - x23) * ab)
    L = L.at[5, 2].set(h2 * (x12 * y21 - x23 * y32) * 2.0 * ab)
    
    L = L.at[6, 0].set(h2 * y12)
    L = L.at[6, 2].set(h2 * x21)
    L = L.at[7, 1].set(h2 * x21)
    L = L.at[7, 2].set(h2 * y12)
    L = L.at[8, 0].set(h2 * y12 * (y32 - y13) * ab)
    L = L.at[8, 1].set(h2 * x21 * (x23 - x31) * ab)
    L = L.at[8, 2].set(h2 * (x23 * y32 - x31 * y13) * 2.0 * ab)
    
    Kb = (L @ Qin @ L.T) / V
    
    T0 = jnp.zeros((3, 9))
    T0 = T0.at[0, 0].set(x32 / A4)
    T0 = T0.at[0, 1].set(y32 / A4)
    T0 = T0.at[0, 2].set(1.0)
    T0 = T0.at[0, 3].set(x13 / A4)
    T0 = T0.at[0, 4].set(y13 / A4)
    T0 = T0.at[0, 6].set(x21 / A4)
    T0 = T0.at[0, 7].set(y21 / A4)
    
    T0 = T0.at[1, 0].set(x32 / A4)
    T0 = T0.at[1, 1].set(y32 / A4)
    T0 = T0.at[1, 3].set(x13 / A4)
    T0 = T0.at[1, 4].set(y13 / A4)
    T0 = T0.at[1, 5].set(1.0)
    T0 = T0.at[1, 6].set(x21 / A4)
    T0 = T0.at[1, 7].set(y21 / A4)
    
    T0 = T0.at[2, 0].set(x32 / A4)
    T0 = T0.at[2, 1].set(y32 / A4)
    T0 = T0.at[2, 3].set(x13 / A4)
    T0 = T0.at[2, 4].set(y13 / A4)
    T0 = T0.at[2, 6].set(x21 / A4)
    T0 = T0.at[2, 7].set(y21 / A4)
    T0 = T0.at[2, 8].set(1.0)
    
    A14 = 1.0 / (Ae_safe * A4)
    Te = jnp.zeros((3, 3))
    Te = Te.at[0, 0].set(A14 * y23 * y13 * LL21)
    Te = Te.at[0, 1].set(A14 * y31 * y21 * LL32)
    Te = Te.at[0, 2].set(A14 * y12 * y32 * LL13)
    Te = Te.at[1, 0].set(A14 * x23 * x13 * LL21)
    Te = Te.at[1, 1].set(A14 * x31 * x21 * LL32)
    Te = Te.at[1, 2].set(A14 * x12 * x32 * LL13)
    Te = Te.at[2, 0].set(A14 * (y23 * x31 + x32 * y13) * LL21)
    Te = Te.at[2, 1].set(A14 * (y31 * x12 + x13 * y21) * LL32)
    Te = Te.at[2, 2].set(A14 * (y12 * x23 + x21 * y32) * LL13)
    
    A14_q = A2 / 3.0
    Q1 = jnp.zeros((3, 3))
    Q1 = Q1.at[0, 0].set(A14_q * b1 / LL21)
    Q1 = Q1.at[0, 1].set(A14_q * b2 / LL21)
    Q1 = Q1.at[0, 2].set(A14_q * b3 / LL21)
    Q1 = Q1.at[1, 0].set(A14_q * b4 / LL32)
    Q1 = Q1.at[1, 1].set(A14_q * b5 / LL32)
    Q1 = Q1.at[1, 2].set(A14_q * b6 / LL32)
    Q1 = Q1.at[2, 0].set(A14_q * b7 / LL13)
    Q1 = Q1.at[2, 1].set(A14_q * b8 / LL13)
    Q1 = Q1.at[2, 2].set(A14_q * b9 / LL13)
    
    Q2 = jnp.zeros((3, 3))
    Q2 = Q2.at[0, 0].set(A14_q * b9 / LL21)
    Q2 = Q2.at[0, 1].set(A14_q * b7 / LL21)
    Q2 = Q2.at[0, 2].set(A14_q * b8 / LL21)
    Q2 = Q2.at[1, 0].set(A14_q * b3 / LL32)
    Q2 = Q2.at[1, 1].set(A14_q * b1 / LL32)
    Q2 = Q2.at[1, 2].set(A14_q * b2 / LL32)
    Q2 = Q2.at[2, 0].set(A14_q * b6 / LL13)
    Q2 = Q2.at[2, 1].set(A14_q * b4 / LL13)
    Q2 = Q2.at[2, 2].set(A14_q * b5 / LL13)
    
    Q3 = jnp.zeros((3, 3))
    Q3 = Q3.at[0, 0].set(A14_q * b5 / LL21)
    Q3 = Q3.at[0, 1].set(A14_q * b6 / LL21)
    Q3 = Q3.at[0, 2].set(A14_q * b4 / LL21)
    Q3 = Q3.at[1, 0].set(A14_q * b8 / LL32)
    Q3 = Q3.at[1, 1].set(A14_q * b9 / LL32)
    Q3 = Q3.at[1, 2].set(A14_q * b7 / LL32)
    Q3 = Q3.at[2, 0].set(A14_q * b2 / LL13)
    Q3 = Q3.at[2, 1].set(A14_q * b3 / LL13)
    Q3 = Q3.at[2, 2].set(A14_q * b1 / LL13)
    
    Q4 = (Q1 + Q2) * 0.5
    Q5 = (Q2 + Q3) * 0.5
    Q6 = (Q3 + Q1) * 0.5
    
    Enat = Te.T @ Qin @ Te
    
    KO = (3.0 / 4.0) * b0 * V * (Q4.T @ Enat @ Q4 + Q5.T @ Enat @ Q5 + Q6.T @ Enat @ Q6)
    Kh = T0.T @ KO @ T0
    Km_9x9 = Kb + Kh
    
    K = jnp.zeros((18, 18))
    for i in range(3):
        r_f = 6 * i
        r_m = 3 * i
        
        row_9 = Km_9x9[r_m]
        res0 = jnp.zeros(18)
        res0 = res0.at[0:2].set(row_9[0:2]).at[5].set(row_9[2])
        res0 = res0.at[6:8].set(row_9[3:5]).at[11].set(row_9[5])
        res0 = res0.at[12:14].set(row_9[6:8]).at[17].set(row_9[8])
        K = K.at[r_f].set(res0)
        
        row_9_1 = Km_9x9[r_m+1]
        res1 = jnp.zeros(18)
        res1 = res1.at[0:2].set(row_9_1[0:2]).at[5].set(row_9_1[2])
        res1 = res1.at[6:8].set(row_9_1[3:5]).at[11].set(row_9_1[5])
        res1 = res1.at[12:14].set(row_9_1[6:8]).at[17].set(row_9_1[8])
        K = K.at[r_f+1].set(res1)
        
        row_9_2 = Km_9x9[r_m+2]
        res2 = jnp.zeros(18)
        res2 = res2.at[0:2].set(row_9_2[0:2]).at[5].set(row_9_2[2])
        res2 = res2.at[6:8].set(row_9_2[3:5]).at[11].set(row_9_2[5])
        res2 = res2.at[12:14].set(row_9_2[6:8]).at[17].set(row_9_2[8])
        K = K.at[r_f+5].set(res2)
        
    return K

@jax.jit
def _element_K_tria3_jax(c1, c2, c3, t, E, nu):
    """
    JAX pure-function implementation of the CalculiX US3 3-node flat shell element.
    """
    xg = jnp.stack([c1, c2, c3], axis=0)
    tm, tmg = us3_csys_cr_jax(xg)
    
    x = jnp.zeros((3, 3))
    x = x.at[0].set(tm @ xg[0])
    x = x.at[1].set(tm @ xg[1])
    x = x.at[2].set(tm @ xg[2])
    
    Qin, Qs = us3_linel_Qi_jax(E, nu)
    
    Dm = Qin * t
    Db = Qin * (t**3) / 12.0
    Ds = Qs * t
    
    Kp = us3_Kp_jax(x, Db, Ds)
    Km = us3_Km_jax(x, Qin, t)
    
    Kshell = Km + Kp
    
    result = tmg.T @ Kshell @ tmg
    
    # Degenerate element guard
    Ae = 0.5 * ((x[1, 0] - x[0, 0]) * (x[2, 1] - x[0, 1]) - (x[2, 0] - x[0, 0]) * (x[0, 1] - x[1, 1]))
    valid = jnp.where(jnp.abs(Ae) < 1e-10, 0.0, 1.0)
    return valid * result
