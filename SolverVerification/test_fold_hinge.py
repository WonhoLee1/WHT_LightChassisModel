# -*- coding: utf-8 -*-
"""
test_fold_hinge.py
==================
Verification tests for _add_fold_hinge_stiffness.

Theory
------
V-beam: two flat panels of width W joined at a fold angle phi_deg (measured
from flat, i.e. 0 = coplanar, 90 = L-section, 180 = folded back).
Simply supported at both ends of the fold line (length L).

First bending mode (z-direction, symmetric) analytical solution:

    f1 = omega1 / (2*pi)
    omega1 = (pi/L)^2 * sqrt(EI_eff / m_per_length)

where the dominant (membrane / parallel-axis) contribution is:

    I_eff = t * W^3 * sin^2(phi_deg/2) / 6

    m_per_length = rho * 2*W * t

For phi_deg = 0  (flat plate): I_eff = 0  -- plate bending D term dominates.
For phi_deg = 90 (L-section) : I_eff = t*W^3/12, much larger than D term.

Three test scenarios
--------------------
1. Flat plate (phi=0): fold spring must add zero stiffness -- regression check.
2. L-section (phi=90): fold spring should push FEM toward analytical beam freq.
3. Alpha sweep (phi=90): scan alpha 0.1..5.0, find value that minimises error.
"""

from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_entities import WHTSPCEntry
from wht_solver.wht_solver import WHTSolver


# ---------------------------------------------------------------------------
# Material / geometry constants
# ---------------------------------------------------------------------------
E   = 210000.0    # MPa
NU  = 0.3
RHO = 7.85e-9     # ton/mm^3


# ---------------------------------------------------------------------------
# Mesh builder
# ---------------------------------------------------------------------------

def _build_v_beam(
    L: float,
    W: float,
    t: float,
    phi_deg: float,
    nx: int = 21,
    nw: int = 6,
) -> WHTMeshModel:
    """
    Two QUAD4 panels (each nx×nw elements) joined at a fold angle phi_deg.

    Coordinate layout
    -----------------
    - x: along beam length [0, L]
    - Fold line: along x at y=0, z=0
    - Right panel: extends to  y = +W*cos(phi_r), z = +W*sin(phi_r)
    - Left  panel: extends to  y = -W*cos(phi_r), z = +W*sin(phi_r)
    where phi_r = phi_deg/2 (each panel at phi_r from the horizontal z-axis).

    Simply-supported BC: fold-line nodes (y=0, z=0) at x=0 and x=L are
    fixed in z (DOF 2).  One corner node also fixed in x, y, rz to remove
    rigid body modes.
    """
    phi_r = np.deg2rad(phi_deg / 2.0)   # half angle from horizontal
    cos_r = np.cos(phi_r)
    sin_r = np.sin(phi_r)

    # Each panel: nx points along x, nw points along panel width direction
    # For right panel: parametric point (ix, iw):
    #   x = ix * dx,  y = +iw*dw*cos_r,  z = iw*dw*sin_r
    dx = L / (nx - 1)
    dw = W / (nw - 1)

    nodes = []   # (x, y, z)
    nid_map = {}  # (panel, ix, iw) -> nid

    nid = 0

    # Panel 0 (right): ix in [0..nx-1], iw in [0..nw-1]
    # Panel 1 (left) : ix in [0..nx-1], iw in [0..nw-1]
    # Fold-line nodes (iw==0) are SHARED between panels -> same nid

    # Create fold-line nodes first (iw=0)
    fold_nids = {}
    for ix in range(nx):
        fold_nids[ix] = nid
        nodes.append((ix * dx, 0.0, 0.0))
        nid_map[(0, ix, 0)] = nid
        nid_map[(1, ix, 0)] = nid  # shared
        nid += 1

    # Right panel inner nodes (iw > 0)
    for iw in range(1, nw):
        for ix in range(nx):
            nid_map[(0, ix, iw)] = nid
            nodes.append((ix * dx, +iw * dw * cos_r, iw * dw * sin_r))
            nid += 1

    # Left panel inner nodes (iw > 0)
    for iw in range(1, nw):
        for ix in range(nx):
            nid_map[(1, ix, iw)] = nid
            nodes.append((ix * dx, -iw * dw * cos_r, iw * dw * sin_r))
            nid += 1

    # Elements: right panel then left panel
    elems = []
    for panel in range(2):
        for iw in range(nw - 1):
            for ix in range(nx - 1):
                n1 = nid_map[(panel, ix,     iw    )]
                n2 = nid_map[(panel, ix + 1, iw    )]
                n3 = nid_map[(panel, ix + 1, iw + 1)]
                n4 = nid_map[(panel, ix,     iw + 1)]
                elems.append([n1, n2, n3, n4])

    # Build model
    model = WHTMeshModel(name="V_Beam")
    model.add_material(1, E, NU, RHO)
    model.add_property(1, "PSHELL", t, 1)

    for i, (x, y, z) in enumerate(nodes):
        model.add_node(i, x, y, z)

    for i, nids in enumerate(elems):
        model.add_element(i, nids, 'QUAD4', pid=1)

    # BC: simply supported at x=0 and x=L, fix Z on fold-line nodes
    for ix in range(nx):
        nid_fold = fold_nids[ix]
        node = model.nodes[nid_fold]
        if abs(node.x) < 1e-6 or abs(node.x - L) < 1e-6:
            model.spc_conditions.append(WHTSPCEntry(nid_fold, (2,)))  # fix z

    # Fix rigid body modes at node 0
    model.spc_conditions.append(WHTSPCEntry(0, (0, 1, 5)))

    return model


# ---------------------------------------------------------------------------
# Analytical formula
# ---------------------------------------------------------------------------

def v_beam_analytical_f1(L, W, t, phi_deg):
    """
    First bending mode (z-direction) of a simply-supported V-beam.

    Uses the membrane (parallel-axis) contribution only, which dominates
    for W >> t.  Plate bending contribution (D/EI ratio ~ (t/W)^2) is added
    for completeness but is negligible.

    Returns f1 [Hz].
    """
    phi_r = np.deg2rad(phi_deg / 2.0)   # each panel's angle from horizontal
    sin_r = np.sin(phi_r)

    # Moment of inertia about the centroidal y-axis (bending in z)
    # Membrane contribution (dominant):
    I_membrane = t * W**3 * sin_r**2 / 6.0

    # Plate bending contribution of each panel (secondary):
    # Each panel's I_plate about its own neutral plane ≈ W*t^3/12, projected
    I_plate = 2.0 * (W * t**3 / 12.0) * sin_r**2

    I_eff = I_membrane + I_plate

    EI_eff = E * I_eff
    m_per_length = RHO * 2.0 * W * t  # ton/mm

    omega1 = (np.pi / L)**2 * np.sqrt(EI_eff / m_per_length)
    return omega1 / (2.0 * np.pi)


# ---------------------------------------------------------------------------
# FEM runner
# ---------------------------------------------------------------------------

def _run_modal(model, fold_alpha=0.0, num_modes=20):
    solver = WHTSolver(
        model,
        k_backend='numpy',
        fold_alpha=fold_alpha,
        fold_phi_min_deg=2.0,
    )
    result = solver.solve_modal(num_modes=num_modes)
    return result.frequencies


def _pick_first_bending(freqs, f_theory, search_window=0.5):
    """
    Find the FEM mode closest to f_theory within search_window * f_theory.
    Returns (f_fem, mode_index) or (None, -1) if not found.
    """
    lo = f_theory * (1.0 - search_window)
    hi = f_theory * (1.0 + search_window)
    candidates = [(i, f) for i, f in enumerate(freqs) if lo <= f <= hi]
    if not candidates:
        # Widen search
        candidates = [(i, f) for i, f in enumerate(freqs)]
        candidates.sort(key=lambda x: abs(x[1] - f_theory))
    else:
        candidates.sort(key=lambda x: abs(x[1] - f_theory))
    return (candidates[0][1], candidates[0][0]) if candidates else (None, -1)


# ---------------------------------------------------------------------------
# Test 1: flat plate - fold spring must not add stiffness
# ---------------------------------------------------------------------------

def test_flat_plate_no_change():
    """
    phi=0 → sin(phi)=0 → k_spring=0 for all edges.
    Frequencies must be identical with and without fold spring.
    """
    print("\n" + "="*60)
    print("Test 1: Flat plate (phi=0) - fold spring regression check")
    print("="*60)

    L, W, t = 400.0, 50.0, 1.0
    model = _build_v_beam(L, W, t, phi_deg=0.0, nx=17, nw=5)

    freqs_base  = _run_modal(model, fold_alpha=0.0)
    freqs_hinge = _run_modal(model, fold_alpha=1.0)

    max_diff_pct = np.max(np.abs(freqs_base[:10] - freqs_hinge[:10]) /
                          np.abs(freqs_base[:10] + 1e-12)) * 100.0

    print(f"  Frequencies (base)  : {freqs_base[:6]}")
    print(f"  Frequencies (hinge) : {freqs_hinge[:6]}")
    print(f"  Max deviation (modes 1-10): {max_diff_pct:.6f}%")

    passed = max_diff_pct < 0.001
    print(f"  {'[PASS]' if passed else '[FAIL]'} Flat plate: fold spring adds "
          f"{'zero (correct)' if passed else f'{max_diff_pct:.4f}% (unexpected)'} stiffness")
    return passed


# ---------------------------------------------------------------------------
# Test 2: L-section (phi=90°) - analytical vs FEM comparison
# ---------------------------------------------------------------------------

def test_l_section_frequency():
    """
    phi=90 degrees (L-section).
    Compare:
      - WHT without fold spring
      - WHT with fold spring (alpha sweep)
      - Analytical V-beam formula
    """
    print("\n" + "="*60)
    print("Test 2: L-section (phi=90°) - frequency comparison")
    print("="*60)

    L, W, t, phi_deg = 400.0, 50.0, 1.0, 90.0
    model = _build_v_beam(L, W, t, phi_deg=phi_deg, nx=21, nw=7)

    f_theory = v_beam_analytical_f1(L, W, t, phi_deg)
    print(f"  Analytical f1 = {f_theory:.2f} Hz")

    # Without fold spring
    freqs_base = _run_modal(model, fold_alpha=0.0, num_modes=30)
    f_base, idx_base = _pick_first_bending(freqs_base, f_theory)
    err_base = (f_base - f_theory) / f_theory * 100.0 if f_base else None
    print(f"  FEM (no spring): f1 = {f_base:.2f} Hz  "
          f"(mode {idx_base+1}, err = {err_base:+.2f}%)")

    # With fold spring at several alpha values
    print("\n  Alpha sweep:")
    print(f"  {'alpha':>8}  {'f1_fem [Hz]':>12}  {'err [%]':>8}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*8}")

    best_alpha, best_err, best_f = None, np.inf, None
    for alpha in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        freqs = _run_modal(model, fold_alpha=alpha, num_modes=30)
        f_fem, _ = _pick_first_bending(freqs, f_theory)
        if f_fem is None:
            continue
        err = (f_fem - f_theory) / f_theory * 100.0
        print(f"  {alpha:>8.2f}  {f_fem:>12.2f}  {err:>+8.2f}%")
        if abs(err) < abs(best_err):
            best_alpha, best_err, best_f = alpha, err, f_fem

    print(f"\n  Best alpha: {best_alpha}  ->  f1 = {best_f:.2f} Hz  "
          f"(err = {best_err:+.2f}%)")

    improvement = abs(err_base) - abs(best_err) if err_base else 0
    print(f"  Error reduction vs no-spring: {improvement:.2f} percentage points")
    return best_alpha, best_err


# ---------------------------------------------------------------------------
# Test 3: phi sweep - verify sin²(phi) scaling
# ---------------------------------------------------------------------------

def test_phi_sweep():
    """
    For alpha=1.0, check that the frequency increase with phi follows
    the expected sin^2 trend (increasing fold angle → increasing stiffness).
    """
    print("\n" + "="*60)
    print("Test 3: phi sweep - stiffness scaling with fold angle")
    print("="*60)

    L, W, t = 400.0, 50.0, 1.0
    print(f"\n  {'phi [deg]':>10}  {'f_theory [Hz]':>14}  "
          f"{'f_no_spring [Hz]':>17}  {'f_spring (a=1) [Hz]':>20}  "
          f"{'err_spring [%]':>14}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*17}  {'-'*20}  {'-'*14}")

    mono_check = []
    for phi_deg in [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]:
        model = _build_v_beam(L, W, t, phi_deg=phi_deg, nx=17, nw=6)
        f_theory = v_beam_analytical_f1(L, W, t, phi_deg)

        freqs_base  = _run_modal(model, fold_alpha=0.0, num_modes=30)
        freqs_hinge = _run_modal(model, fold_alpha=1.0, num_modes=30)

        f_base,  _ = _pick_first_bending(freqs_base,  f_theory)
        f_hinge, _ = _pick_first_bending(freqs_hinge, f_theory)

        err_hinge = (f_hinge - f_theory) / f_theory * 100.0 if f_hinge else float('nan')
        print(f"  {phi_deg:>10.1f}  {f_theory:>14.2f}  "
              f"{f_base:>17.2f}  {f_hinge:>20.2f}  {err_hinge:>+14.2f}%")
        mono_check.append(f_hinge)

    # Monotone check: frequency should increase with phi
    mono_pass = all(mono_check[i] <= mono_check[i+1]
                    for i in range(len(mono_check) - 1))
    print(f"\n  Monotone increase with phi: {'[PASS]' if mono_pass else '[FAIL]'}")
    return mono_pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  _add_fold_hinge_stiffness  Verification Suite")
    print("="*60)

    p1 = test_flat_plate_no_change()
    best_alpha, best_err = test_l_section_frequency()
    p3 = test_phi_sweep()

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    print(f"  Test 1 (flat plate no-change)   : {'PASS' if p1 else 'FAIL'}")
    print(f"  Test 2 (L-section best alpha)   : {best_alpha} -> err={best_err:+.2f}%")
    print(f"  Test 3 (phi sweep monotone)     : {'PASS' if p3 else 'FAIL'}")
    print()
    print("  Recommended alpha from Test 2 sweep - use this value when")
    print("  constructing WHTSolver(model, fold_alpha=<alpha>, fold_phi_min_deg=3.0)")
