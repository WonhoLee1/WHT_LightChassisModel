# -*- coding: utf-8 -*-
"""
Extended Patch Tests for ElementStressRecovery
================================================
Test 4: 45-degree rotated QUAD4 - pure local tension -> global tensor rotation
Test 5: TRIA3 pure membrane tension
Test 6: TRIA3 pure shear
Test 7: TRIA3 pure bending
Test 8: QUAD4 combined membrane + bending
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_solver.wht_stress_recovery import ElementStressRecovery

# -------------------------------------------------------------------------
# Minimal stubs
# -------------------------------------------------------------------------

class Node:
    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z

class Elem:
    def __init__(self, etype, node_ids, pid=1):
        self.type = etype; self.node_ids = node_ids; self.pid = pid

class Prop:
    def __init__(self, t, mid): self.t = t; self.mid = mid; self.type = "PSHELL"

class Mat:
    def __init__(self, E, nu, rho=7.85e-9): self.E = E; self.nu = nu; self.rho = rho

class MockModel:
    def __init__(self, nodes, elements, E=210000.0, nu=0.3, t=1.0):
        self.nodes = {i+1: Node(*c) for i, c in enumerate(nodes)}
        self.elements = {i+1: e for i, e in enumerate(elements)}
        self.properties = {1: Prop(t=t, mid=1)}
        self.materials = {1: Mat(E=E, nu=nu)}

def run(label, stresses, strains, checks):
    print(f"\n--- {label} ---")
    ok = True
    for name, got, exp, tol in checks:
        err = abs(got - exp)
        flag = "OK" if err < tol else "FAIL"
        if flag == "FAIL": ok = False
        print(f"  {name:30s}: got={got:10.4f}, exp={exp:10.4f}, err={err:.2e}  [{flag}]")
    print("  => ALL OK" if ok else "  => FAILED")
    return ok


# =========================================================================
# Test 4: 45-degree rotated QUAD4, pure local-X tension
# =========================================================================
def test4_rotated_quad4():
    c = s = 1.0 / np.sqrt(2)
    E, nu, t, e0 = 210000.0, 0.3, 1.0, 0.001

    # 1-unit QUAD4 rotated 45 degrees around Z
    # Nodes in global: n0=(0,0), n1=(c,s,0), n2=(c-s, s+c, 0), n3=(-s,c,0)
    n0 = [0.0, 0.0, 0.0]
    n1 = [c,   s,   0.0]
    n2 = [c-s, s+c, 0.0]
    n3 = [-s,  c,   0.0]

    # Pure local-X displacement: u_loc_x = e0 * x_loc
    # Node local coords: n0=(0,0), n1=(1,0), n2=(1,1), n3=(0,1)
    # u_loc_x: 0, e0, e0, 0  |  u_loc_y: 0, 0, 0, 0
    # Transform to global: u_glob = T^T * u_loc
    # T rows = [X_loc, Y_loc, Z_loc] = [[c,s,0],[-s,c,0],[0,0,1]]
    # u_glob = [c*u_lx + (-s)*u_ly, s*u_lx + c*u_ly, 0]
    u_global = np.zeros((4, 6))
    for i, u_lx in enumerate([0.0, e0, e0, 0.0]):
        u_global[i, 0] = c * u_lx   # global X
        u_global[i, 1] = s * u_lx   # global Y

    model = MockModel(
        nodes=[n0, n1, n2, n3],
        elements=[Elem("QUAD4", [1, 2, 3, 4], pid=1)],
        E=E, nu=nu, t=t,
    )
    sorted_nids = [1, 2, 3, 4]
    stresses, strains = ElementStressRecovery.recover_quad4(model, u_global, sorted_nids)

    # Expected local stress
    fac = E / (1 - nu**2)
    sig_loc_xx = fac * e0                 # ε_yy = 0 in test
    sig_loc_yy = fac * nu * e0

    # Tensor rotation: σ_glob = T^T * σ_loc * T
    # For 45-deg rotation with only σ_loc_xx and σ_loc_yy:
    # σ_GXX = c^2*σ_xx + s^2*σ_yy
    # σ_GYY = s^2*σ_xx + c^2*σ_yy   (same value for 45 deg)
    # σ_GXY = c*s*(σ_xx - σ_yy)
    sig_GXX = c**2 * sig_loc_xx + s**2 * sig_loc_yy
    sig_GYY = s**2 * sig_loc_xx + c**2 * sig_loc_yy
    sig_GXY = c * s * (sig_loc_xx - sig_loc_yy)

    tol = 0.1
    return run("Test 4: 45-deg QUAD4 local tension -> global rotation", stresses, strains, [
        ("sigma_XX (global)", stresses[0, 0], sig_GXX, tol),
        ("sigma_YY (global)", stresses[0, 1], sig_GYY, tol),
        ("sigma_XY (global)", stresses[0, 3], sig_GXY, tol),
        ("sigma_ZZ (should=0)", stresses[0, 2], 0.0, tol),
    ])


# =========================================================================
# Test 5: TRIA3 pure membrane tension
# =========================================================================
def test5_tria3_tension():
    E, nu, t, e0 = 210000.0, 0.3, 1.0, 0.001
    # Right triangle: nodes at (0,0,0), (2,0,0), (0,2,0)
    model = MockModel(
        nodes=[[0,0,0],[2,0,0],[0,2,0]],
        elements=[Elem("TRIA3", [1, 2, 3], pid=1)],
        E=E, nu=nu, t=t,
    )
    # u_x = e0 * x  (nodes: 0, 2*e0, 0)
    u_global = np.zeros((3, 6))
    u_global[1, 0] = 2.0 * e0

    sorted_nids = [1, 2, 3]
    stresses, strains = ElementStressRecovery.recover_tria3(model, u_global, sorted_nids)

    fac = E / (1 - nu**2)
    sig_xx_exp = fac * e0
    eps_xx_exp = e0

    tol = 1.0
    return run("Test 5: TRIA3 pure membrane tension", stresses, strains, [
        ("eps_xx",    strains[0, 0], eps_xx_exp, 1e-6),
        ("sigma_xx",  stresses[0, 0], sig_xx_exp, tol),
        ("sigma_yy",  stresses[0, 1], fac * nu * e0, tol),
        ("sigma_xy",  stresses[0, 3], 0.0, tol),
    ])


# =========================================================================
# Test 6: TRIA3 pure shear
# =========================================================================
def test6_tria3_shear():
    E, nu, t, g0 = 210000.0, 0.3, 1.0, 0.002
    model = MockModel(
        nodes=[[0,0,0],[2,0,0],[0,2,0]],
        elements=[Elem("TRIA3", [1, 2, 3], pid=1)],
        E=E, nu=nu, t=t,
    )
    # u_x = g0/2 * y, u_y = g0/2 * x  (symmetric shear)
    # Nodes: n1=(0,0)->u=(0,0), n2=(2,0)->u=(0, g0), n3=(0,2)->u=(g0, 0)
    u_global = np.zeros((3, 6))
    u_global[1, 1] = g0            # node2: u_y = g0/2 * 2 = g0
    u_global[2, 0] = g0            # node3: u_x = g0/2 * 2 = g0

    sorted_nids = [1, 2, 3]
    stresses, strains = ElementStressRecovery.recover_tria3(model, u_global, sorted_nids)

    G = E / (2 * (1 + nu))
    sig_xy_exp = G * g0

    tol = 1.0
    return run("Test 6: TRIA3 pure shear", stresses, strains, [
        ("gamma_xy",  strains[0, 3], g0, 1e-6),
        ("sigma_xy",  stresses[0, 3], sig_xy_exp, tol),
        ("sigma_xx",  stresses[0, 0], 0.0, tol),
        ("sigma_yy",  stresses[0, 1], 0.0, tol),
    ])


# =========================================================================
# Test 7: TRIA3 pure bending
# =========================================================================
def test7_tria3_bending():
    E, nu, t, k0 = 210000.0, 0.3, 2.0, 0.001
    # k0 = curvature (∂θy/∂x = k0) -> ε_xx_bend = z * k0, z = t/2
    model = MockModel(
        nodes=[[0,0,0],[2,0,0],[0,2,0]],
        elements=[Elem("TRIA3", [1, 2, 3], pid=1)],
        E=E, nu=nu, t=t,
    )
    # θy = k0 * x (rotation about y-axis varies linearly with x)
    # Nodes: n1=(0,0)->θy=0, n2=(2,0)->θy=k0*2, n3=(0,2)->θy=0
    u_global = np.zeros((3, 6))
    u_global[1, 4] = k0 * 2.0     # node2: θy (DOF index 4 = rotation around global y)

    sorted_nids = [1, 2, 3]
    stresses, strains = ElementStressRecovery.recover_tria3(model, u_global, sorted_nids)

    z = t / 2.0
    fac = E / (1 - nu**2)
    eps_xx_bend_exp = z * k0
    sig_xx_bend_exp = fac * eps_xx_bend_exp

    tol = 1.0
    return run("Test 7: TRIA3 pure bending", stresses, strains, [
        ("eps_xx_bend", strains[0, 0], eps_xx_bend_exp, 1e-6),
        ("sigma_xx",    stresses[0, 0], sig_xx_bend_exp, tol),
        ("sigma_yy",    stresses[0, 1], fac * nu * eps_xx_bend_exp, tol),
    ])


# =========================================================================
# Test 8: QUAD4 combined membrane + bending
# =========================================================================
def test8_quad4_combined():
    E, nu, t = 210000.0, 0.3, 2.0
    # 2x2 QUAD4 at (0,0)~(2,2)
    model = MockModel(
        nodes=[[0,0,0],[2,0,0],[2,2,0],[0,2,0]],
        elements=[Elem("QUAD4", [1, 2, 3, 4], pid=1)],
        E=E, nu=nu, t=t,
    )
    # Membrane: ε_xx = 0.001
    e_m = 0.001
    # Bending:  curvature κ = 0.001 (∂θy/∂x = κ -> θy = κ*x)
    kappa = 0.001

    # Nodes at x=0: u_x=0, θy=0
    # Nodes at x=2: u_x=e_m*2, θy=kappa*2
    u_global = np.zeros((4, 6))
    u_global[1, 0] = e_m * 2         # node2: u_x
    u_global[2, 0] = e_m * 2         # node3: u_x
    u_global[1, 4] = kappa * 2       # node2: θy
    u_global[2, 4] = kappa * 2       # node3: θy

    sorted_nids = [1, 2, 3, 4]
    stresses, strains = ElementStressRecovery.recover_quad4(model, u_global, sorted_nids)

    fac = E / (1 - nu**2)
    z = t / 2.0
    eps_m = e_m
    eps_b = z * kappa
    eps_tot = eps_m + eps_b

    sig_xx_exp = fac * eps_tot
    sig_yy_exp = fac * nu * eps_tot

    tol = 1.0
    return run("Test 8: QUAD4 membrane + bending combined", stresses, strains, [
        ("eps_xx_total",  strains[0, 0], eps_tot, 1e-6),
        ("sigma_xx",      stresses[0, 0], sig_xx_exp, tol),
        ("sigma_yy",      stresses[0, 1], sig_yy_exp, tol),
        ("sigma_xy",      stresses[0, 3], 0.0, tol),
    ])


# =========================================================================
if __name__ == "__main__":
    results = [
        test4_rotated_quad4(),
        test5_tria3_tension(),
        test6_tria3_shear(),
        test7_tria3_bending(),
        test8_quad4_combined(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Extended Patch Tests: {passed}/{total} passed")
    if passed == total:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
