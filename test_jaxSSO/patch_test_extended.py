# -*- coding: utf-8 -*-
"""
Extended Patch Tests for ElementStressRecovery (v2 API)
=======================================================
Test 4: 45-degree rotated QUAD4 - pure local tension -> global tensor rotation
Test 5: TRIA3 pure membrane tension
Test 6: TRIA3 pure shear
Test 7: TRIA3 pure bending
Test 8: QUAD4 combined membrane + bending
Test 9: Mid-plane symmetry (순수 굽힘에서 Mid=0 확인)
Test 10: Upper/Lower 대칭성 (순수 굽힘에서 Upper = -Lower)
Test 11: 순수 인장에서 Upper = Mid = Lower 확인
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

def run(label, result_dict, checks):
    """검증 함수. result_dict는 새 API의 Dict[str, ndarray] 반환값."""
    print(f"\n--- {label} ---")
    ok = True
    for name, got, exp, tol in checks:
        err = abs(got - exp)
        flag = "OK" if err < tol else "FAIL"
        if flag == "FAIL": ok = False
        print(f"  {name:40s}: got={got:12.6f}, exp={exp:12.6f}, err={err:.2e}  [{flag}]")
    print("  => ALL OK" if ok else "  => FAILED")
    return ok


# =========================================================================
# Test 4: 45-degree rotated QUAD4, pure local-X tension
# =========================================================================
def test4_rotated_quad4():
    c = s = 1.0 / np.sqrt(2)
    E, nu, t, e0 = 210000.0, 0.3, 1.0, 0.001

    n0 = [0.0, 0.0, 0.0]
    n1 = [c,   s,   0.0]
    n2 = [c-s, s+c, 0.0]
    n3 = [-s,  c,   0.0]

    u_global = np.zeros((4, 6))
    for i, u_lx in enumerate([0.0, e0, e0, 0.0]):
        u_global[i, 0] = c * u_lx
        u_global[i, 1] = s * u_lx

    model = MockModel(
        nodes=[n0, n1, n2, n3],
        elements=[Elem("QUAD4", [1, 2, 3, 4], pid=1)],
        E=E, nu=nu, t=t,
    )
    sorted_nids = [1, 2, 3, 4]
    rd = ElementStressRecovery.recover_quad4(model, u_global, sorted_nids)
    stresses = rd["Stress"]

    fac = E / (1 - nu**2)
    sig_loc_xx = fac * e0
    sig_loc_yy = fac * nu * e0

    sig_GXX = c**2 * sig_loc_xx + s**2 * sig_loc_yy
    sig_GYY = s**2 * sig_loc_xx + c**2 * sig_loc_yy
    sig_GXY = c * s * (sig_loc_xx - sig_loc_yy)

    tol = 0.1
    return run("Test 4: 45-deg QUAD4 local tension -> global rotation", rd, [
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
    model = MockModel(
        nodes=[[0,0,0],[2,0,0],[0,2,0]],
        elements=[Elem("TRIA3", [1, 2, 3], pid=1)],
        E=E, nu=nu, t=t,
    )
    u_global = np.zeros((3, 6))
    u_global[1, 0] = 2.0 * e0

    sorted_nids = [1, 2, 3]
    rd = ElementStressRecovery.recover_tria3(model, u_global, sorted_nids)
    stresses = rd["Stress"]
    strains = rd["Strain"]

    fac = E / (1 - nu**2)
    sig_xx_exp = fac * e0
    eps_xx_exp = e0

    tol = 1.0
    return run("Test 5: TRIA3 pure membrane tension", rd, [
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
    u_global = np.zeros((3, 6))
    u_global[1, 1] = g0
    u_global[2, 0] = g0

    sorted_nids = [1, 2, 3]
    rd = ElementStressRecovery.recover_tria3(model, u_global, sorted_nids)
    stresses = rd["Stress"]
    strains = rd["Strain"]

    G = E / (2 * (1 + nu))
    sig_xy_exp = G * g0

    tol = 1.0
    return run("Test 6: TRIA3 pure shear", rd, [
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
    model = MockModel(
        nodes=[[0,0,0],[2,0,0],[0,2,0]],
        elements=[Elem("TRIA3", [1, 2, 3], pid=1)],
        E=E, nu=nu, t=t,
    )
    u_global = np.zeros((3, 6))
    u_global[1, 4] = k0 * 2.0

    sorted_nids = [1, 2, 3]
    rd = ElementStressRecovery.recover_tria3(model, u_global, sorted_nids)
    stresses = rd["Stress"]
    strains = rd["Strain"]

    z = t / 2.0
    fac = E / (1 - nu**2)
    eps_xx_bend_exp = z * k0
    sig_xx_bend_exp = fac * eps_xx_bend_exp

    tol = 1.0
    return run("Test 7: TRIA3 pure bending", rd, [
        ("eps_xx_bend (Upper)", strains[0, 0], eps_xx_bend_exp, 1e-6),
        ("sigma_xx (Upper)",    stresses[0, 0], sig_xx_bend_exp, tol),
        ("sigma_yy (Upper)",    stresses[0, 1], fac * nu * eps_xx_bend_exp, tol),
    ])


# =========================================================================
# Test 8: QUAD4 combined membrane + bending
# =========================================================================
def test8_quad4_combined():
    E, nu, t = 210000.0, 0.3, 2.0
    model = MockModel(
        nodes=[[0,0,0],[2,0,0],[2,2,0],[0,2,0]],
        elements=[Elem("QUAD4", [1, 2, 3, 4], pid=1)],
        E=E, nu=nu, t=t,
    )
    e_m = 0.001
    kappa = 0.001

    u_global = np.zeros((4, 6))
    u_global[1, 0] = e_m * 2
    u_global[2, 0] = e_m * 2
    u_global[1, 4] = kappa * 2
    u_global[2, 4] = kappa * 2

    sorted_nids = [1, 2, 3, 4]
    rd = ElementStressRecovery.recover_quad4(model, u_global, sorted_nids)
    stresses = rd["Stress"]
    strains = rd["Strain"]

    fac = E / (1 - nu**2)
    z = t / 2.0
    eps_m = e_m
    eps_b = z * kappa
    eps_tot = eps_m + eps_b

    sig_xx_exp = fac * eps_tot
    sig_yy_exp = fac * nu * eps_tot

    tol = 1.0
    return run("Test 8: QUAD4 membrane + bending combined", rd, [
        ("eps_xx_total (Upper)", strains[0, 0], eps_tot, 1e-6),
        ("sigma_xx (Upper)",     stresses[0, 0], sig_xx_exp, tol),
        ("sigma_yy (Upper)",     stresses[0, 1], sig_yy_exp, tol),
        ("sigma_xy (Upper)",     stresses[0, 3], 0.0, tol),
    ])


# =========================================================================
# Test 9: 순수 굽힘에서 Mid-plane 응력 = 0 확인
# =========================================================================
def test9_midplane_zero_in_bending():
    E, nu, t, k0 = 210000.0, 0.3, 2.0, 0.001
    model = MockModel(
        nodes=[[0,0,0],[2,0,0],[0,2,0]],
        elements=[Elem("TRIA3", [1, 2, 3], pid=1)],
        E=E, nu=nu, t=t,
    )
    u_global = np.zeros((3, 6))
    u_global[1, 4] = k0 * 2.0

    sorted_nids = [1, 2, 3]
    rd = ElementStressRecovery.recover_tria3(model, u_global, sorted_nids)
    stress_mid = rd["Stress (Mid)"]
    strain_mid = rd["Strain (Mid)"]

    tol = 1e-10
    return run("Test 9: Pure bending -> Mid-plane stress/strain = 0", rd, [
        ("sigma_xx (Mid)", stress_mid[0, 0], 0.0, tol),
        ("sigma_yy (Mid)", stress_mid[0, 1], 0.0, tol),
        ("sigma_xy (Mid)", stress_mid[0, 3], 0.0, tol),
        ("eps_xx (Mid)",   strain_mid[0, 0], 0.0, tol),
    ])


# =========================================================================
# Test 10: 순수 굽힘에서 Upper = -Lower 대칭성
# =========================================================================
def test10_upper_lower_symmetry():
    E, nu, t, k0 = 210000.0, 0.3, 2.0, 0.001
    model = MockModel(
        nodes=[[0,0,0],[2,0,0],[0,2,0]],
        elements=[Elem("TRIA3", [1, 2, 3], pid=1)],
        E=E, nu=nu, t=t,
    )
    u_global = np.zeros((3, 6))
    u_global[1, 4] = k0 * 2.0

    sorted_nids = [1, 2, 3]
    rd = ElementStressRecovery.recover_tria3(model, u_global, sorted_nids)
    s_upper = rd["Stress"]
    s_lower = rd["Stress (Lower)"]

    tol = 1e-6
    return run("Test 10: Pure bending -> Upper = -Lower symmetry", rd, [
        ("sigma_xx: Upper + Lower = 0", s_upper[0, 0] + s_lower[0, 0], 0.0, tol),
        ("sigma_yy: Upper + Lower = 0", s_upper[0, 1] + s_lower[0, 1], 0.0, tol),
        ("sigma_xy: Upper + Lower = 0", s_upper[0, 3] + s_lower[0, 3], 0.0, tol),
    ])


# =========================================================================
# Test 11: 순수 인장에서 Upper = Mid = Lower (두께 무관)
# =========================================================================
def test11_tension_all_layers_equal():
    E, nu, t, e0 = 210000.0, 0.3, 1.0, 0.001
    model = MockModel(
        nodes=[[0,0,0],[2,0,0],[0,2,0]],
        elements=[Elem("TRIA3", [1, 2, 3], pid=1)],
        E=E, nu=nu, t=t,
    )
    u_global = np.zeros((3, 6))
    u_global[1, 0] = 2.0 * e0

    sorted_nids = [1, 2, 3]
    rd = ElementStressRecovery.recover_tria3(model, u_global, sorted_nids)
    s_upper = rd["Stress"]
    s_mid = rd["Stress (Mid)"]
    s_lower = rd["Stress (Lower)"]

    tol = 1e-6
    return run("Test 11: Pure tension -> Upper = Mid = Lower", rd, [
        ("Upper_xx == Mid_xx",   s_upper[0, 0] - s_mid[0, 0], 0.0, tol),
        ("Upper_xx == Lower_xx", s_upper[0, 0] - s_lower[0, 0], 0.0, tol),
        ("Upper_yy == Mid_yy",   s_upper[0, 1] - s_mid[0, 1], 0.0, tol),
        ("Upper_yy == Lower_yy", s_upper[0, 1] - s_lower[0, 1], 0.0, tol),
    ])


# =========================================================================
if __name__ == "__main__":
    results = [
        test4_rotated_quad4(),
        test5_tria3_tension(),
        test6_tria3_shear(),
        test7_tria3_bending(),
        test8_quad4_combined(),
        test9_midplane_zero_in_bending(),
        test10_upper_lower_symmetry(),
        test11_tension_all_layers_equal(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Extended Patch Tests (v2): {passed}/{total} passed")
    if passed == total:
        print("All tests PASSED.")
    else:
        print("Some tests FAILED.")
