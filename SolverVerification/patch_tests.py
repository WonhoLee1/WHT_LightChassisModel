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
    def __init__(self):
        self.E = 210000.0
        self.nu = 0.3
        self.rho = 7.85e-9

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
        
        solver = WHTSolver(model)
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
        
        solver = WHTSolver(model)
        
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
        result = solver.solve_modal(num_modes=15)
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
        solver = WHTSolver(model)
        t0 = time.perf_counter()
        result = solver.solve_static(lc)
        t_ms = (time.perf_counter() - t0) * 1000.0
        
        # Theory: eps_x = sigma / E. Displacement_x = eps_x * L
        th_ux = (p_stress / self.E) * L
        fe_ux = np.max(result.displacement[:, 0])
        
        # Theory stress: sigma_x = p_stress
        fe_sx = np.mean(result.cell_data['Stress'][0, :, 0])
        
        return [
            TestResult("Membrane Tension", "Max Displacement X", etype, th_ux, fe_ux, t_ms, tol_pct=1.0),
            TestResult("Membrane Tension", "Avg Stress Sx", etype, p_stress, fe_sx, t_ms, tol_pct=1.0)
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
            
        solver = WHTSolver(model)
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
        
        solver = WHTSolver(model)
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
