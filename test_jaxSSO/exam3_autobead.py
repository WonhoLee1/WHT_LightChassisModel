# -*- coding: utf-8 -*-
"""
exam3_autobead.py
=================
WHT FEM Framework — Unified Modal Analysis with Topography Auto-Beads

Overview:
---------
This script performs a high-fidelity modal analysis on a structural shell/solid tray
enhanced with "Topography Beads" for stiffness optimization.

Key Features:
-------------
- Hook-Fold Flange Structure (Rise -> Inward -> Fall -> Inward).
- Auto-Bead Generation: Random symmetric bead patterns for floor reinforcement.
- Dual Solver Support: JaxSSO (Standard Shell) & jax-fem (High-fidelity Solid Hexa).
- Visualizer Integration: Preview bead heights and results with WHT Visualizer.
"""

import os
import sys
import argparse
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import jax
import jax.numpy as jnp

# JAX float64 필수 (고유치 해석 수치 안정성 보장)
jax.config.update("jax_enable_x64", True) 

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_entities import WHTSPCEntry
from wht_solver.wht_solver import WHTSolver
from wht_converter.wht_models import WHTMetadata, WHTResultData
from wht_converter.wht_adapters import JaxFEMAdapter
from wht_visualizer.wht_visualizer import WHTVisualizer
from mesh_utils import generate_shell_tray, generate_solid_hexa_tray, apply_auto_beads


# ==============================================================================
# 0. CONFIGURATION & SCHEMA
# ==============================================================================

@dataclass
class PipelineConfig:
    """
    Unified settings for the Shell/Solid Tray with Auto-Beads.
    """
    # 1. Geometry Dimensions [mm]
    width:  float = 1800.0
    length: float = 1200.0
    height: float = 35.0
    thickness: float = 0.6  # Default shell thickness
    
    # 2. Mesh Control
    # 'shell' uses JaxSSO, 'solid' uses jax-fem (HEXA8)
    solve_mode: str = 'shell' 
    mesh_size_xy: float = 40.0
    mesh_size_z:  float = 10.0
    draft_angle:  float = 25.0
    wall_layers:  int = 2    # Only for solid hexa
    
    # 3. Flange Configuration (Hook-Fold Sequence)
    flanges: tuple = (False, True, True, True) # Bottom disabled
    flange_segments: List[Tuple[float, float]] = field(default_factory=list)
    
    # 4. Auto-Bead Configuration (Topography)
    bead_mode: str = 'random'         # Default to 'random' as requested
    bead_margin: float = 50.0
    bead_target_ratio: float = 0.4  # 50% bead coverage
    bead_min_depth: float = 3.0     # Min vertical morphing [mm]
    bead_max_depth: float = 10.0    # Max vertical morphing [mm]
    bead_min_size: float = 50.0     # Min square size [mm]
    bead_max_size: float = 200.0    # Max square size [mm]
    
    # 5. Solver & Analysis Settings
    num_modes: int = 8
    solver_method: str = 'auto'
    
    # 6. Material (Steel)
    E:   float = 210000.0
    nu:  float = 0.3
    rho: float = 7.85e-9


# ==============================================================================
# 1. CORE PIPELINE FUNCTIONS
# ==============================================================================

def build_base_geometry(cfg: PipelineConfig) -> Tuple[Dict, Dict]:
    """Generates the raw tray geometry before morphing."""
    if cfg.solve_mode == 'shell':
        print(f" -> Generating Base Shell Tray Mesh (size={cfg.mesh_size_xy})...")
        node_db, elem_db = generate_shell_tray(
            width=cfg.width, length=cfg.length, height=cfg.height,
            mesh_size_xy=cfg.mesh_size_xy, mesh_size_z=cfg.mesh_size_z,
            draft_angle=cfg.draft_angle, flange_segments=cfg.flange_segments,
            flanges=cfg.flanges, mesh_type='quad4',
            origin='center'
        )
    else:
        print(f" -> Generating Base Solid Hexa Tray Mesh (layers={cfg.wall_layers})...")
        node_db, elem_db = generate_solid_hexa_tray(
            width=cfg.width, length=cfg.length, height=cfg.height,
            mesh_size_xy=cfg.mesh_size_xy, mesh_size_z=cfg.mesh_size_z,
            draft_angle=cfg.draft_angle, wall_layers=cfg.wall_layers,
            flange_segments=cfg.flange_segments, flanges=cfg.flanges,
            origin='center'
        )
    return node_db, elem_db


def apply_topography_beads(node_db: Dict, cfg: PipelineConfig) -> Dict:
    """Applies random bead patterns to the flat sections of the tray."""
    print(" -> Applying Symmetric Auto-Beads (Morphed Topography)...")
    node_db_morphed = apply_auto_beads(
        node_db=node_db, 
        width=cfg.width, length=cfg.length,
        margin=cfg.bead_margin,
        target_ratio=cfg.bead_target_ratio,
        min_depth=cfg.bead_min_depth,
        max_depth=cfg.bead_max_depth,
        min_size=cfg.bead_min_size,
        max_size=cfg.bead_max_size,
        origin='center',
        mode=cfg.bead_mode
    )
    return node_db_morphed


def build_wht_model(node_db: Dict, elem_db: Dict, cfg: PipelineConfig) -> WHTMeshModel:
    """Converts raw DB to WHT Model and applies BCs/Materials."""
    is_solid = (cfg.solve_mode == 'solid')
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db, is_solid=is_solid)
    model.name = f"{cfg.solve_mode.capitalize()}_Beaded_Tray"
    
    # 1. Material & Property
    model.add_material(1, cfg.E, cfg.nu, cfg.rho)
    if is_solid:
        model.add_property(1, "PSOLID", 0.0, 1)
    else:
        model.add_property(1, "PSHELL", cfg.thickness, 1)
    
    for elem in model.elements.values():
        elem.pid = 1
        
    # 2. Boundary Conditions (Fixed Top Flange)
    # The top flange end height is the sum of height + all flange dz
    max_z = cfg.height + sum([seg[1] for seg in cfg.flange_segments])
    fixed_count = 0
    for nid, node in model.nodes.items():
        if abs(node.z - max_z) < 0.2:
            model.apply_spc(nid, (0, 1, 2, 3, 4, 5))
            fixed_count += 1
    print(f"    [BC] Constrained {fixed_count} nodes at top flange rim.")
    
    return model


def evaluate_modal_response(model: WHTMeshModel, cfg: PipelineConfig):
    """Executes the specialized solver based on mesh type."""
    if cfg.solve_mode == 'shell':
        print(f" -> Solving Modal Problem (JaxSSO Sparse, modes={cfg.num_modes})...")
        solver = WHTSolver(model)
        return solver.solve_modal(num_modes=cfg.num_modes, method=cfg.solver_method)
    else:
        return _solve_solid_jaxfem(model, cfg)


# ==============================================================================
# 2. INTERNAL SOLVER (SOLID/JAX-FEM)
# ==============================================================================

def _solve_solid_jaxfem(model: WHTMeshModel, cfg: PipelineConfig):
    """Specialized modal logic for Solid Hexa using jax-fem wrapper."""
    print(f" -> Initializing jax-fem pipeline for Solid Hexa (modes={cfg.num_modes})...")
    try:
        from jax_fem.generate_mesh import Mesh
        from jax_fem.problem import Problem
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import diags, csr_matrix
    except ImportError:
        raise ImportError("jax-fem is required for solid mode analysis.")

    # Convert WHT to jax-fem mesh
    points = model.nodes_array()
    nid_to_idx = model.node_id_to_index()
    cells = np.array([[nid_to_idx[nid] for nid in model.elements[eid].node_ids] 
                     for eid in sorted(model.elements.keys()) if model.elements[eid].type == "HEXA8"])
    mesh = Mesh(points, cells)
    
    class ModalSolidProblem(Problem):
        def get_tensor_map(self):
            def constitutive(u_grad):
                mu = cfg.E / (2.0 * (1.0 + cfg.nu))
                lmbda = cfg.E * cfg.nu / ((1.0 + cfg.nu) * (1.0 - 2.0 * cfg.nu))
                eps = 0.5 * (u_grad + u_grad.T)
                return lmbda * jnp.trace(eps) * jnp.eye(3) + 2.0 * mu * eps
            return constitutive

    max_z = cfg.height + sum([seg[1] for seg in cfg.flange_segments])
    def flange_filter(x): return jnp.isclose(x[2], max_z, atol=0.2)
    
    dirichlet_bc = [[flange_filter]*3, [0, 1, 2], [lambda x: 0.0]*3]
    prob = ModalSolidProblem(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc)
    
    # Assembly
    print("    [Solid] Assembling K (jax-fem Stiff) and M (Lumped Mass)...")
    prob.newton_update(prob.unflatten_fn_sol_list(np.zeros(prob.num_total_dofs_all_vars)))
    K = csr_matrix((np.array(prob.V), (prob.I, prob.J)), shape=(prob.num_total_dofs_all_vars, prob.num_total_dofs_all_vars))
    
    # Lumped mass calculation logic
    import pyvista as pv
    grid = pv.UnstructuredGrid(np.hstack([np.full((cells.shape[0], 1), 8), cells]).flatten(), 
                               np.full(cells.shape[0], 12, dtype=np.uint8), points)
    nodal_mass = np.zeros(len(points))
    cell_volumes = grid.compute_cell_sizes()["Volume"]
    for i, cell in enumerate(cells):
        m_cell = cell_volumes[i] * cfg.rho
        nodal_mass[cell] += m_cell / 8.0
    M = diags([np.repeat(nodal_mass, 3)], [0], shape=K.shape, dtype=K.dtype).tocsr()

    # Solve
    vals, vecs = eigsh(K, k=cfg.num_modes, M=M, which='LM', sigma=-0.1)
    res_vecs = vecs.reshape((len(points), 3, cfg.num_modes)).transpose(2, 0, 1)
    return (prob, {"eigvecs": res_vecs, "eigvals": vals})


# ==============================================================================
# 3. POST-PROCESSING & VISUALIZATION
# ==============================================================================

def inject_bead_metadata(wht_data: WHTResultData, model: WHTMeshModel, n_orig: Dict, n_new: Dict):
    """Adds the bead height (DZ) data as a viewable result field in visualizer."""
    dz_list = []
    for nid in model.sorted_node_ids():
        dz = n_new[nid][2] - n_orig[nid][2]
        dz_list.append(dz)
        
    if wht_data.point_data is None: wht_data.point_data = {}
    
    # Define num_t based on existing context
    num_t = len(wht_data.time_values) if wht_data.time_values is not None else 1
    
    # NEW: Merge into a single 3D Vector field named 'Bead_Height'
    # This supports both Magnitude coloring and Deform warping.
    dz_vector = np.zeros((num_t, len(dz_list), 3), dtype=np.float32)
    dz_vector[:, :, 2] = np.array(dz_list, dtype=np.float32).reshape(1, -1)
    
    wht_data.point_data["Bead_Height"] = dz_vector
    return wht_data


# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Modal Analysis with Auto Beads")
    parser.add_argument("--solve", type=str, choices=['shell', 'solid'], default='shell', help="Solver type")
    parser.add_argument("--mode", type=str, choices=['grid', 'random'], default='grid', help="Bead pattern mode")
    parser.add_argument("--modes", type=int, default=8, help="Number of modes to calculate")
    parser.add_argument("--min", type=float, default=3.0, help="Min bead depth [mm]")
    parser.add_argument("--max", type=float, default=10.0, help="Max bead depth [mm]")
    parser.add_argument("--min_size", type=float, default=50.0, help="Min bead size [mm]")
    parser.add_argument("--max_size", type=float, default=200.0, help="Max bead size [mm]")
    parser.add_argument("--preview", action="store_true", help="Preview beads and exit")
    args = parser.parse_args()

    # 1. Pipeline Sequence (Hook-Fold Flange)
    hook_sequence = [(0.0, 12.0), (-5.0, 0.0), (0.0, -10.0), (-10.0, 0.0)]
    
    cfg = PipelineConfig(
        solve_mode=args.solve,
        bead_mode=args.mode,
        num_modes=args.modes,
        bead_min_depth=args.min,
        bead_max_depth=args.max,
        bead_min_size=args.min_size,
        bead_max_size=args.max_size,
        flange_segments=hook_sequence
    )

    print("\n" + "="*80)
    print(f" [ANALYSIS RUN] {cfg.solve_mode.upper()} TRAY WITH AUTO-BEADS")
    print("="*80)

    try:
        # A. Geometry Generation
        node_db_orig, elem_db = build_base_geometry(cfg)
        node_db_morphed = apply_topography_beads(node_db_orig, cfg)
        
        # B. Model Assembly
        model = build_wht_model(node_db_morphed, elem_db, cfg)
        
        if args.preview:
            print("\n -> [PREVIEW MODE] Launching Bead Visualizer...")
            # [WHT] Metadata 필수 파라미터 (solver_name, version, coord_sys, unit_force) 누락 해결
            meta = WHTMetadata(
                solver_name="WHT_Geometry_Preview",
                solver_version="1.0.0",
                analysis_type="modal",
                coordinate_system="cartesian",
                unit_length="mm",
                unit_force="N"
            )
            wht_data = model.to_wht_result_data(meta)
            wht_data = inject_bead_metadata(wht_data, model, node_db_orig, node_db_morphed)
            
            viz = WHTVisualizer(title="Topography Bead Preview (Morphed Floor)", show=True)
            viz.load_results(wht_data)
            viz.show(); return

        # C. Solver Execution (해석 수행)
        results = evaluate_modal_response(model, cfg)
        
        # D. Reporting & Visualization (결과 보고 및 시각화)
        if cfg.solve_mode == 'shell':
            print("\n" + "#"*80)
            print(" #  SHELL MODAL FREQUENCIES (Hz) - (JaxSSO Sparse Solver)               #")
            print("#"*80)
            for i, f in enumerate(results.frequencies): 
                print(f"  Mode {i+1:2d}: {f:8.3f} Hz")
            
            meta = WHTMetadata(
                solver_name="JaxSSO", solver_version="2.1.0",
                analysis_type="modal", coordinate_system="cartesian",
                unit_length="mm", unit_force="N"
            )
            wht_data = results.to_wht_result_data(meta, model)
        else:
            prob, res_dict = results
            adapter = JaxFEMAdapter()
            meta = WHTMetadata(
                solver_name="jax-fem", solver_version="0.1.0",
                analysis_type="modal", coordinate_system="cartesian",
                unit_length="mm", unit_force="N"
            )
            wht_data = adapter.convert(prob, res_dict, "modal", meta)
            print("\n -> [Solid Result] Modal solution extracted. Launching viewer...")

        # Inject original bead data (비드 높이 데이터를 시각화 모듈에 주입)
        wht_data = inject_bead_metadata(wht_data, model, node_db_orig, node_db_morphed)
        
        viz = WHTVisualizer(title=f"Beaded Tray Modal Shapes [{cfg.solve_mode.upper()}]", show=True)
        viz.load_results(wht_data, color="black")
        viz.plotter.view_isometric()
        if hasattr(viz.plotter, 'app'): viz.plotter.app.exec_()
            
    except Exception:
        print("\n" + "!"*80)
        print(" [CRITICAL ERROR] Auto-Bead Pipeline failed.")
        print("!"*80)
        traceback.print_exc()


if __name__ == "__main__":
    main()
