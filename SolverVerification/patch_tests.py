# -*- coding: utf-8 -*-
import os
import sys
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

# Add workspace to path
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_entities import WHTSPCEntry
from wht_solver.wht_solver import WHTSolver
from wht_solver.load_cases import WHTLoadCase
import SolverVerification.analytical_solutions as analytical

@dataclass
class TestResult:
    name: str
    quantity: str
    element_type: str
    theory: float
    fem: float
    exec_time_ms: float = 0.0
    tol_pct: float = 5.0
    details: str = ""

    @property
    def error_pct(self) -> float:
        if abs(self.theory) < 1e-15:
            return 0.0 if abs(self.fem) < 1e-15 else 100.0
        return abs(self.fem - self.theory) / abs(self.theory) * 100.0

    @property
    def passed(self) -> bool:
        return self.error_pct <= self.tol_pct

def _mesh_plate(Lx, Ly, nx, ny, quads=True):
    """Simple grid mesh generator."""
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    xv, yv = np.meshgrid(x, y)
    nodes = np.stack([xv.flatten(), yv.flatten(), np.zeros_like(xv.flatten())], axis=1)
    
    elems = []
    if quads:
        for j in range(ny - 1):
            for i in range(nx - 1):
                n1 = j * nx + i
                n2 = j * nx + i + 1
                n3 = (j + 1) * nx + i + 1
                n4 = (j + 1) * nx + i
                # ID is 1-based for WHTMeshModel elements usually, but here we'll use 0-based for convenience
                elems.append([n1, n2, n3, n4])
    else:
        for j in range(ny - 1):
            for i in range(nx - 1):
                n1, n2, n3, n4 = j*nx+i, j*nx+i+1, (j+1)*nx+i, (j+1)*nx+i+1
                elems.append([n1, n2, n4])
                elems.append([n2, n3, n4])
    return nodes, elems

class PatchTestRunner:
    def __init__(self, k_backend: str = 'auto'):
        self.E = 210000.0
        self.nu = 0.3
        self.rho = 7.85e-9
        self.k_backend = k_backend

    def build_wht_model(self, nodes, elems, etype, t):
        model = WHTMeshModel(name=f"Test_{etype}")
        model.add_material(1, self.E, self.nu, self.rho)
        model.add_property(1, "PSHELL", t, 1)
        
        for i, (x, y, z) in enumerate(nodes):
            model.add_node(i, x, y, z)
        
        for i, nids in enumerate(elems):
            model.add_element(i, nids, etype, pid=1)
        return model

    def test_3pt_bending(self, etype='QUAD4'):
        L, w, t, P = 100.0, 10.0, 2.0, 100.0
        nx, ny = 21, 5
        nodes, elems = _mesh_plate(L, w, nx, ny, quads=(etype=='QUAD4'))
        model = self.build_wht_model(nodes, elems, etype, t)
        
        # BCs: Simply supported at x=0 and x=L
        for nid, node in model.nodes.items():
            if abs(node.x - 0.0) < 1e-5 or abs(node.x - L) < 1e-5:
                model.spc_conditions.append(WHTSPCEntry(nid, (2,))) # Fix Z
            if nid == 0:
                model.spc_conditions.append(WHTSPCEntry(nid, (0, 1, 5))) # Fix rigid body
        
        lc = WHTLoadCase("PointLoad")
        mid_node_id = int(np.argmin(np.abs(nodes[:, 0] - L/2.0) + np.abs(nodes[:, 1] - w/2.0)))
        lc.add_force(mid_node_id, (2,), (-P,))
        
        solver = WHTSolver(model, k_backend=self.k_backend)
        t0 = time.perf_counter()
        result = solver.solve_static(lc)
        t_ms = (time.perf_counter() - t0) * 1000.0
        
        I = (w * t**3) / 12.0
        th_w = analytical.beam_3point_bending_deflection(L, self.E, I, P)
        fe_w = np.max(np.abs(result.displacement[:, 2]))
        
        th_s = analytical.beam_3point_bending_stress(L, w, t, P)
        fe_s = np.max(np.abs(result.cell_data['Stress'][0, :, 0])) # Max Sigma_x
        
        return [
            TestResult("3-pt Bending", "Max Deflection", etype, th_w, fe_w, t_ms, tol_pct=5.0),
            TestResult("3-pt Bending", "Max Stress (Sx)", etype, th_s, fe_s, t_ms, tol_pct=15.0)
        ]

    def test_frequency(self, etype='QUAD4'):
        L, t = 1000.0, 10.0
        # Increased mesh density for higher mode stability (Mode 5 verification)
        nx, ny = 21, 21 
        nodes, elems = _mesh_plate(L, L, nx, ny, quads=(etype=='QUAD4'))
        model = self.build_wht_model(nodes, elems, etype, t)
        
        # BCs: Simply supported on all 4 edges (Z fixed)
        for nid, node in model.nodes.items():
            if (abs(node.x - 0.0) < 1e-5 or abs(node.x - L) < 1e-5 or 
                abs(node.y - 0.0) < 1e-5 or abs(node.y - L) < 1e-5):
                model.spc_conditions.append(WHTSPCEntry(nid, (2,)))
            if nid == 0:
                model.spc_conditions.append(WHTSPCEntry(nid, (0, 1, 5)))
        
        solver = WHTSolver(model, k_backend=self.k_backend)
        
        # Diagnostic: Total Mass
        ndof = len(model.nodes) * 6
        sorted_nids = sorted(model.nodes.keys())
        nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}
        from wht_solver.wht_quad4_element import M_quad4_lumped
        from wht_solver.wht_tria3_element import M_tria3_lumped
        M_diag = M_quad4_lumped(model, ndof, sorted_nids, nid_to_idx)
        M_diag += M_tria3_lumped(model, ndof, sorted_nids, nid_to_idx)
        total_mass = np.sum(M_diag[0::6]) # Only translation DOFs
        print(f"    [Diagnostic] Total Mesh Mass ({etype}): {total_mass:.6f} ton")
        
        t0 = time.perf_counter()
        # TRIA3 has more low-frequency in-plane modes than QUAD4, requiring
        # more modes to reach the bending targets (122.9+ Hz).
        result = solver.solve_modal(num_modes=30)
        t_ms = (time.perf_counter() - t0) * 1000.0
        
        print(f"    [Diagnostic] First 10 Frequencies ({etype}): {result.frequencies[:10]}")
        
        # Multiple mode comparison: Mode 1-5
        # Theory modes (m,n) combos: (1,1), (1,2), (2,1), (2,2), (1,3)...
        theory_modes = [
            ((1,1), 49.17), ((1,2), 122.9), ((2,1), 122.9), ((2,2), 196.7), ((1,3), 245.8)
        ]
        
        # Recalculate accurately based on current parameters
        results = []
        for i, ((m, n), _) in enumerate(theory_modes):
            th_f = analytical.kirchhoff_frequency(L, L, self.E, t, self.nu, self.rho, m=m, n=n)
            
            # Match FEM mode to theory (since spurious modes might shift indices)
            # Find closest FEM frequency within 30% tolerance
            diffs = [abs(fe_f - th_f) for fe_f in result.frequencies]
            best_idx = np.argmin(diffs)
            best_fe = result.frequencies[best_idx]
            
            # If closest match is still > 30% away, it's a mismatch
            is_valid = (abs(best_fe - th_f) / th_f < 0.3)
            fe_f_use = best_fe if is_valid else 0.0
            
            results.append(
                TestResult("Natural Frequency", f"Mode {i+1} ({m},{n}) [Hz]", etype, th_f, fe_f_use, t_ms, tol_pct=10.0)
            )

        return results

    def test_membrane_uniaxial(self, etype='QUAD4'):
        L, w, t, p_stress = 100.0, 10.0, 2.0, 100.0
        # Optimal mesh for CST accuracy (Aspect Ratio close to 1:1)
        # L/nx = 100/20 = 5, w/ny = 10/2 = 5
        nx, ny = 21, 3 
        nodes, elems = _mesh_plate(L, w, nx, ny, quads=(etype=='QUAD4'))
        model = self.build_wht_model(nodes, elems, etype, t)
        
        lc = WHTLoadCase("membrane_tension")
        
        # 1. Fixed BC at x=0: Fix X, Z and all rotations. 
        # For Y, only fix ONE node to prevent rigid body motion but allow Poisson contraction.
        mid_y = w / 2.0
        y_constrained = False
        
        for nid, node in model.nodes.items():
            if abs(node.x - 0.0) < 1e-5:
                # Fix translation X, Z and all rotations (Dofs 0, 2, 3, 4, 5)
                # Dof 1 (Y) is left free except for the central node
                if not y_constrained and abs(node.y - mid_y) < (w/ny + 1e-5):
                    lc.add_bc(nid, (0, 1, 2, 3, 4, 5)) 
                    y_constrained = True
                else:
                    lc.add_bc(nid, (0, 2, 3, 4, 5))
        
        # 2. Tension at x=L: Consistent load vector
        right_nodes = sorted([nid for nid in model.nodes if abs(model.nodes[nid].x - L) < 1e-5], 
                            key=lambda nid: model.nodes[nid].y)
        
        force_total = p_stress * w * t
        num_seg = len(right_nodes) - 1
        if num_seg > 0:
            force_per_seg = force_total / num_seg
            for i, nid in enumerate(right_nodes):
                if i == 0 or i == len(right_nodes) - 1:
                    weight = 0.5
                else:
                    weight = 1.0
                lc.add_force(nid, (0,), (force_per_seg * weight,))
        
        # 3. Solve
        solver = WHTSolver(model, k_backend=self.k_backend)
        t0 = time.perf_counter()
        result = solver.solve_static(lc)
        t_ms = (time.perf_counter() - t0) * 1000.0
        
        # Theory: eps_x = sigma / E. Displacement_x = eps_x * L
        th_ux = (p_stress / self.E) * L
        fe_ux = np.max(result.displacement[:, 0])
        
        # Theory stress: sigma_x = p_stress
        fe_sx = np.mean(result.cell_data['Stress'][0, :, 0])
        
        tol_val = 1.0 if etype == 'QUAD4' else 2.0
        return [
            TestResult("Membrane Tension", "Max Displacement X", etype, th_ux, fe_ux, t_ms, tol_pct=tol_val),
            TestResult("Membrane Tension", "Avg Stress Sx", etype, p_stress, fe_sx, t_ms, tol_pct=tol_val)
        ]

    def test_4pt_bending(self, etype='QUAD4'):
        L, w, t, P = 1000.0, 100.0, 10.0, 1000.0
        nx, ny = 21, 5
        nodes, elems = _mesh_plate(L, w, nx, ny, quads=(etype=='QUAD4'))
        model = self.build_wht_model(nodes, elems, etype, t)
        
        # 4-pt bending: supports at ends, loads at L/3, 2L/3
        from wht_solver.load_cases import LoadCaseLibrary, WHTLoadCase
        supp_nodes = []
        for nid, node in model.nodes.items():
            if abs(node.x - 0.0) < 1e-5 or abs(node.x - L) < 1e-5:
                supp_nodes.append(nid)
        
        lc = WHTLoadCase("4pt_bending")
        for nid in supp_nodes:
            lc.add_bc(nid, (0, 1, 2))
        
        load_nodes_l3 = [nid for nid, node in model.nodes.items() if abs(node.x - L/3.0) < L/nx]
        load_nodes_2l3 = [nid for nid, node in model.nodes.items() if abs(node.x - 2*L/3.0) < L/nx]
        # Just pick the center-y nodes for load
        load_nodes = [n for n in load_nodes_l3 + load_nodes_2l3 if abs(model.nodes[n].y - w/2.0) < 1e-5]
        
        n = len(load_nodes)
        for nid in load_nodes:
            lc.add_force(nid, (2,), (-P/n,))
            
        solver = WHTSolver(model, k_backend=self.k_backend)
        t0 = time.perf_counter()
        result = solver.solve_static(lc)
        t_ms = (time.perf_counter() - t0) * 1000.0
        
        # Theory: Max Deflection for 4-pt bending at inner span
        # a = L/3. w = (P_pt*a/(24*E*I)) * (3*L^2 - 4*a^2)
        # where P_pt is the point load at EACH span point (Total force = 2*P_pt)
        P_pt = P / 2.0 
        I = (w * t**3) / 12.0
        a = L/3.0
        th_w = (P_pt * a / (24.0 * self.E * I)) * (3.0 * L**2 - 4.0 * a**2)
        fe_w = np.max(np.abs(result.displacement[:, 2]))
        
        return [
            TestResult("4-pt Bending", "Max Deflection", etype, th_w, fe_w, t_ms, tol_pct=5.0)
        ]

    def test_twisting(self, etype='QUAD4'):
        L, w, t, P = 1000.0, 1000.0, 10.0, 1000.0
        nx, ny = 11, 11
        nodes, elems = _mesh_plate(L, w, nx, ny, quads=(etype=='QUAD4'))
        model = self.build_wht_model(nodes, elems, etype, t)
        
        # Fix 3 corners, apply P at 4th
        n_corner = { (0,0): -1, (L,0): -1, (L,w): -1, (0,w): -1 }
        for nid, node in model.nodes.items():
            for c in n_corner:
                if abs(node.x - c[0]) < 1e-5 and abs(node.y - c[1]) < 1e-5:
                    n_corner[c] = nid
        
        lc = WHTLoadCase("twisting")
        lc.add_bc(n_corner[(0,0)], (2,))
        lc.add_bc(n_corner[(L,0)], (2,))
        lc.add_bc(n_corner[(L,w)], (2,))
        # Load at (0,w)
        lc.add_force(n_corner[(0,w)], (2,), (-P,))
        
        # Also fix rigid body in XY
        lc.add_bc(n_corner[(0,0)], (0, 1, 5))
        
        solver = WHTSolver(model, k_backend=self.k_backend)
        t0 = time.perf_counter()
        result = solver.solve_static(lc)
        t_ms = (time.perf_counter() - t0) * 1000.0
        
        # Theory twisting: Pure twisting of a square plate
        # For a 3-corner-fix, 1-corner-load P, the deflection is w = P*L*w / (2*D*(1-nu))
        D = (self.E * t**3) / (12.0 * (1.0 - self.nu**2))
        th_w = (P * L * w) / (2.0 * D * (1.0 - self.nu))
        fe_w = np.max(np.abs(result.displacement[:, 2]))
        
        return [
            TestResult("Plate Twisting", "Corner Deflection", etype, th_w, fe_w, t_ms, tol_pct=5.0)
        ]


# ---------------------------------------------------------------------------
# 백엔드 비교 실행
# ---------------------------------------------------------------------------

def run_benchmark(backends=('jax', 'numpy', 'numba')) -> List[dict]:
    """
    고유진동수 해석 벤치마크: 규모가 있는 shell 모델에 대해
    각 백엔드별 K 조립 시간 + 고유해 시간을 측정한다.

    모델: 1000x1000mm, t=5mm 평판, simply-supported, QUAD4 51x51 mesh
          (50x50 = 2500 elements, 2601 nodes, 15606 DOF)
    측정 항목:
      - K assembly time [ms]
      - Modal solve time [ms] (10 modes, ARPACK)
      - First 5 frequencies [Hz]
    """
    from wht_solver.wht_quad4_element import _NUMBA_OK
    try:
        from wht_solver.wht_quad4_element_jax import K_quad4_jax  # noqa
        _jax_ok = True
    except Exception:
        _jax_ok = False

    # Build benchmark model once
    L, t_plate = 1000.0, 5.0
    nx, ny = 51, 51
    E_b, nu_b, rho_b = 210000.0, 0.3, 7.85e-9

    nodes, elems = _mesh_plate(L, L, nx, ny, quads=True)
    runner = PatchTestRunner()
    model = runner.build_wht_model(nodes, elems, 'QUAD4', t_plate)
    model.materials[1].E   = E_b
    model.materials[1].nu  = nu_b
    model.materials[1].rho = rho_b

    # Simply supported on all 4 edges
    for nid, node in model.nodes.items():
        if (abs(node.x) < 1e-5 or abs(node.x - L) < 1e-5 or
                abs(node.y) < 1e-5 or abs(node.y - L) < 1e-5):
            model.spc_conditions.append(WHTSPCEntry(nid, (2,)))
    model.spc_conditions.append(WHTSPCEntry(0, (0, 1, 5)))

    n_nodes = len(model.nodes)
    n_elems = len(model.elements)
    ndof    = n_nodes * 6

    print(f"\n{'='*60}")
    print(f"  Benchmark: {nx-1}x{ny-1} QUAD4 plate  "
          f"({n_nodes} nodes / {n_elems} elems / ~{ndof} DOF)")
    print(f"{'='*60}")
    print(f"  {'Backend':>8}  {'K assemble [ms]':>16}  "
          f"{'Modal solve [ms]':>16}  {'Total [ms]':>10}  Freq 1-5 [Hz]")
    print(f"  {'-'*8}  {'-'*16}  {'-'*16}  {'-'*10}  {'-'*30}")

    records = []
    for backend in backends:
        if backend == 'jax'   and not _jax_ok:
            print(f"  {backend:>8}  [SKIP - JAX unavailable]")
            continue
        if backend == 'numba' and not _NUMBA_OK:
            print(f"  {backend:>8}  [SKIP - Numba unavailable]")
            continue

        solver = WHTSolver(model, k_backend=backend)

        # Warm-up call for Numba JIT (first call compiles)
        if backend == 'numba':
            _ = WHTSolver(runner.build_wht_model(*_mesh_plate(100, 100, 5, 5),
                                                  'QUAD4', t_plate),
                          k_backend='numba').solve_modal(num_modes=6)

        # --- K assembly only ---
        sorted_nids  = sorted(model.nodes.keys())
        nid_to_idx   = {nid: i for i, nid in enumerate(sorted_nids)}
        t0_k = time.perf_counter()
        if backend == 'jax':
            from wht_solver.wht_quad4_element_jax import K_quad4_jax as _K_jax
            K_q = _K_jax(model, sorted_nids, nid_to_idx, None)
        elif backend == 'numba':
            from wht_solver.wht_quad4_element import K_quad4_scipy
            K_q = K_quad4_scipy(model, sorted_nids, nid_to_idx, None, backend='numba')
        else:
            from wht_solver.wht_quad4_element import K_quad4_scipy
            K_q = K_quad4_scipy(model, sorted_nids, nid_to_idx, None, backend='numpy')
        t_k_ms = (time.perf_counter() - t0_k) * 1000.0

        # --- full modal solve ---
        t0_m = time.perf_counter()
        result = solver.solve_modal(num_modes=10)
        t_m_ms = (time.perf_counter() - t0_m) * 1000.0

        freqs5 = result.frequencies[1:6]  # skip rigid-body mode 0
        freq_str = "  ".join(f"{f:.1f}" for f in freqs5)

        print(f"  {backend:>8}  {t_k_ms:>16.1f}  {t_m_ms:>16.1f}  "
              f"{t_k_ms+t_m_ms:>10.1f}  {freq_str}")

        records.append({
            'backend':       backend,
            'n_nodes':       n_nodes,
            'n_elems':       n_elems,
            'ndof':          ndof,
            't_k_ms':        t_k_ms,
            't_modal_ms':    t_m_ms,
            't_total_ms':    t_k_ms + t_m_ms,
            'frequencies':   result.frequencies[:10].tolist(),
        })

    return records


def _run_backend(backend: str, etypes=('QUAD4', 'TRIA3')) -> List[TestResult]:
    runner = PatchTestRunner(k_backend=backend)
    results = []
    for etype in etypes:
        results += runner.test_3pt_bending(etype)
        results += runner.test_membrane_uniaxial(etype)
        results += runner.test_4pt_bending(etype)
        results += runner.test_twisting(etype)
        results += runner.test_frequency(etype)
    return results


def run_all_backends(backends=('jax', 'numpy', 'numba')) -> dict:
    """
    jax / numpy / numba 세 백엔드에 대해 전체 패치 테스트를 실행하고
    결과를 비교 출력한다.

    Returns
    -------
    dict[str, List[TestResult]]
    """
    from wht_solver.wht_quad4_element import _NUMBA_OK
    try:
        from wht_solver.wht_quad4_element_jax import K_quad4_jax  # noqa
        _jax_ok = True
    except Exception:
        _jax_ok = False

    all_results = {}
    for backend in backends:
        if backend == 'jax' and not _jax_ok:
            print(f"\n[SKIP] backend=jax — JAX 미설치")
            continue
        if backend == 'numba' and not _NUMBA_OK:
            print(f"\n[SKIP] backend=numba — Numba 미설치")
            continue

        print(f"\n{'='*60}")
        print(f"  Backend: {backend.upper()}")
        print(f"{'='*60}")
        results = _run_backend(backend)
        all_results[backend] = results

        passed = sum(1 for r in results if r.passed)
        total  = len(results)
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name} | {r.quantity} | {r.element_type}"
                  f" | theory={r.theory:.4g}  fem={r.fem:.4g}  err={r.error_pct:.2f}%")
        print(f"\n  결과: {passed}/{total} 통과")

    # 백엔드 간 비교 요약
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  백엔드 비교 요약")
        print(f"{'='*60}")
        ref_backend = next(iter(all_results))
        ref = {(r.name, r.quantity, r.element_type): r for r in all_results[ref_backend]}
        for backend, results in all_results.items():
            if backend == ref_backend:
                continue
            max_diff = 0.0
            for r in results:
                key = (r.name, r.quantity, r.element_type)
                if key in ref and abs(ref[key].theory) > 1e-15:
                    diff = abs(r.fem - ref[key].fem) / abs(ref[key].fem) * 100
                    max_diff = max(max_diff, diff)
            print(f"  {ref_backend.upper()} vs {backend.upper()} 최대 편차: {max_diff:.4f}%")

    return all_results


if __name__ == '__main__':
    run_all_backends()
    run_benchmark()
