# -*- coding: utf-8 -*-
"""
exam2_shell_jaxSSO_load.py
==========================
WHT FEM Framework — Shell Tray Static Analysis Example

Overview:
---------
This script executes a high-fidelity static analysis for a structural shell tray
under a centralized area load. It utilizes the same advanced Hook-Fold flange 
geometry as the modal example.

Key Features:
-------------
- Centralized Area Load distribution across multiple nodes.
- High-fidelity static response (Displacement & Von-Mises Stress).
- Comparison between different mesh types (QUAD4, TRIA3_FREE).
"""

import numpy as np
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_solver import WHTSolver
from wht_solver.load_cases import WHTLoadCase
from wht_visualizer.wht_visualizer import WHTVisualizer
from wht_converter.wht_models import WHTMetadata
from mesh_utils import generate_shell_tray


# ==============================================================================
# 0. CONFIGURATION & SCHEMA
# ==============================================================================

@dataclass
class PipelineConfig:
    """
    Unified settings for the Shell Tray Static Pipeline.
    """
    # Geometry Dimensions [mm]
    width:  float = 1800.0
    length: float = 1200.0
    height: float = 35.0
    thickness: float = 0.5
    
    # Mesh Control [mm]
    mesh_type: str = 'quad4'
    mesh_size_xy: float = 60.0
    mesh_size_z:  float = 10.0
    draft_angle:  float = 35.0
    
    # Flange Configuration
    flanges: tuple = (False, True, True, True) 
    flange_segments: List[Tuple[float, float]] = field(default_factory=list)
    
    # Static Load Settings
    load_area_size: float = 200.0     # Square area side length [mm]
    total_force: float = 100.0        # Total downward force [N]
    
    # Material (Steel)
    E: float = 210000.0
    nu: float = 0.3
    rho: float = 7.85e-9


# ==============================================================================
# 1. CORE PIPELINE FUNCTIONS
# ==============================================================================

def build_structural_model(cfg: PipelineConfig) -> Tuple[WHTMeshModel, Dict]:
    """Generates geometry and applies boundary conditions for static analysis."""
    print(f" -> Generating Geometry [{cfg.mesh_type.upper()}] with Hook-Flanges...")
    
    # 1.1 Generate Raw Mesh
    node_db, elem_db = generate_shell_tray(
        width=cfg.width, length=cfg.length, height=cfg.height,
        mesh_size_xy=cfg.mesh_size_xy, mesh_size_z=cfg.mesh_size_z,
        draft_angle=cfg.draft_angle, flange_segments=cfg.flange_segments,
        flanges=cfg.flanges, mesh_type=cfg.mesh_type
    )
    
    # 1.2 Convert to WHT Model
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db)
    MID, PID = 1, 1
    model.add_material(MID, E=cfg.E, nu=cfg.nu, rho=cfg.rho)
    model.add_property(PID, "PSHELL", cfg.thickness, MID)
    
    for elem in model.elements.values():
        elem.pid = PID
    
    # 1.3 Apply Boundary Conditions (Fix all DOFs at the top rim/flange)
    # The rim points end at cumulative height.
    fixed_count = 0
    max_z = cfg.height + sum([seg[1] for seg in cfg.flange_segments])
    for nid, node in model.nodes.items():
        if abs(node.z - max_z) < 0.1:
            model.apply_spc(nid, (0, 1, 2, 3, 4, 5))
            fixed_count += 1
    print(f"    [BC] Constrained {fixed_count} nodes at top flange.")
            
    return model, node_db


def evaluate_static_response(model: WHTMeshModel, cfg: PipelineConfig):
    """Defines area load and solves the static system."""
    solver = WHTSolver(model)
    lc = WHTLoadCase("AreaLoad")
    
    # 2.1 Find nodes in center area for distributed load
    area_dim = cfg.load_area_size
    target_nodes = []
    # If using 'center' origin in generation, x0/y0 are handled. 
    # Here we assume current generate_shell_tray uses default origin.
    # We look for nodes near (width/2, length/2)
    for nid, node in model.nodes.items():
        if (abs(node.x - cfg.width/2) <= area_dim/2 and 
            abs(node.y - cfg.length/2) <= area_dim/2 and 
            node.z < 1.0):
            target_nodes.append(nid)
    
    if not target_nodes:
        # Fallback to single center node
        center_nid = min(model.nodes.keys(), key=lambda n: (model.nodes[n].x - cfg.width/2)**2 + (model.nodes[n].y - cfg.length/2)**2)
        target_nodes = [center_nid]
        print(f" <!> No nodes in {area_dim}mm area. Falling back to center point.")
    
    force_per_node = -cfg.total_force / len(target_nodes)
    print(f" -> Applying {cfg.total_force}N over {len(target_nodes)} nodes (Area: {area_dim}sq).")
    
    for nid in target_nodes:
        lc.add_force(nid, (2,), (force_per_node,))
    
    return solver.solve_static(lc)


def evaluate_static_response_twist(model: WHTMeshModel, cfg: PipelineConfig):
    """
    Executes a torsion test (Twist) using RBE2 master nodes.
    - Clears existing rim BCs to allow torsion.
    - Left side: Rigidly connected to a master node, all 6 DOFs fixed.
    - Right side: Rigidly connected to a master node, rotated 10 deg around X-axis.
    """
    print("\n" + "-"*80)
    print(" -> [Twist Scenario] Setting up Torsion Test with RBE2 Master Nodes...")
    print("-"*80)
    
    # [CRITICAL] Clear existing SPCs (like Rim BC) so the load can pass through the plate
    old_spc_count = len(model.spc_conditions)
    model.spc_conditions = []
    print(f"    [Setup] Cleared {old_spc_count} existing SPCs to enable pure torsion.")
    
    # 1. Identify Left and Right boundary nodes (Xmin, Xmax)
    nodes_arr = model.nodes_array()
    xmin, xmax = nodes_arr[:, 0].min(), nodes_arr[:, 0].max()
    
    left_slave_nids = [nid for nid, node in model.nodes.items() if abs(node.x - xmin) < 0.1]
    right_slave_nids = [nid for nid, node in model.nodes.items() if abs(node.x - xmax) < 0.1]
    
    # 2. Add Master Nodes at centers of X-ends
    # We'll use high IDs for master nodes to avoid conflict
    max_id = max(model.nodes.keys())
    m_left_id = max_id + 1
    m_right_id = max_id + 2
    
    y_mid = (nodes_arr[:, 1].max() + nodes_arr[:, 1].min()) / 2.0
    z_mid = (nodes_arr[:, 2].max() + nodes_arr[:, 2].min()) / 2.0
    
    model.add_node(m_left_id, xmin, y_mid, z_mid)
    model.add_node(m_right_id, xmax, y_mid, z_mid)
    
    # 3. Add RBE2 Rigid Elements
    # rbe2_id starts from a high range
    model.add_rbe2(900, m_left_id, left_slave_nids)
    model.add_rbe2(901, m_right_id, right_slave_nids)
    
    # 4. Setup Load Case with Prescribed Twist
    lc = WHTLoadCase("Twist_10deg_X")
    
    # Left Master: All 6 DOFs fixed to zero
    lc.add_bc(m_left_id, (0, 1, 2, 3, 4, 5), 0.0)
    
    # Right Master: Twist 10 deg around X (DOF 3)
    # Note: DOF indices are 0..5. Rotation around X is DOF 3.
    # Degrees to Radians: 10 * pi / 180
    angle_rad = 10.0 * np.pi / 180.0
    
    # Request says: "오른쪽 노드는 x축 회전으로 10도 비튼다. 다른 자유도는 고정한다."
    # So DOFs 0,1,2, 4,5 are fixed to 0.0, and DOF 3 is fixed to angle_rad.
    lc.add_bc(m_right_id, (0, 1, 2, 4, 5), 0.0)
    lc.add_bc(m_right_id, (3,), angle_rad)
    
    print(f"    [Setup] RBE2 Left: master {m_left_id} with {len(left_slave_nids)} slaves.")
    print(f"    [Setup] RBE2 Right: master {m_right_id} with {len(right_slave_nids)} slaves.")
    print(f"    [BC] Applied +10 deg rotation to master {m_right_id}.")

    solver = WHTSolver(model)
    result = solver.solve_static(lc)
    
    # 5. Output Reaction Forces (6-DOFs)
    reac_l = result.reaction_force(m_left_id)
    reac_r = result.reaction_force(m_right_id)
    
    print("\n" + "="*80)
    print(f" {'[WHT] Torsion Test Reaction Forces':^78}")
    print("="*80)
    labels = ["Fx(N)", "Fy(N)", "Fz(N)", "Mx(Nmm)", "My(Nmm)", "Mz(Nmm)"]
    
    print(f" Master Node Path | {' | '.join([f'{l:>10}' for l in labels])}")
    print("-" * 110)
    row_l = f" Left (Fixed)     | " + " | ".join([f"{v:10.2e}" for v in reac_l])
    row_r = f" Right (Twisted)  | " + " | ".join([f"{v:10.2e}" for v in reac_r])
    print(row_l)
    print(row_r)
    print("="*110 + "\n")
    
    return result


# ==============================================================================
# 2. ANALYSIS EXECUTION & REPORTING
# ==============================================================================

def print_result_table(all_results: Dict[str, any]):
    print("\n" + "#"*95)
    print(" #  STATIC ANALYSIS COMPARISON SUMMARY                                                        #")
    print("#"*95)
    header = f"  {'Mesh Type':<15} | {'Nodes':<10} | {'Max Uz (mm)':<15} | {'Max Stress (MPa)':<15} | {'Diff%'}"
    print(header)
    print("-" * 95)
    
    ref_val = None
    for mtype, res in all_results.items():
        if res is None: continue
        
        max_uz = np.abs(res.displacement[:, 2]).max()
        # Access stress diagnostic if available
        max_vm = getattr(res, '_max_vm_diagnostic', 0.0)
        
        if ref_val is None:
            ref_val = max_uz
            diff_pct = 0.0
        else:
            diff_pct = (max_uz - ref_val) / ref_val * 100.0
            
        print(f"  {mtype.upper():<15} | {res.displacement.shape[0]:<10} | {max_uz:15.5f} | {max_vm:15.5f} | {diff_pct:8.2f}%")
    print("-" * 95)


def main():
    # --- STEP 1: Hook-Fold Design ---
    hook_sequence = [
        ( 0.0,  12.0), # 1. 상승
        (-5.0,   0.0), # 2. 내측
        ( 0.0, -10.0), # 3. 하락
        (-10.0,  0.0)  # 4. 내측
    ]
    
    # --- STEP 2: Scenarios ---
    test_suite = [
        PipelineConfig(mesh_type='quad4', flange_segments=hook_sequence),
        PipelineConfig(mesh_type='tria3_free', flange_segments=hook_sequence),
    ]
    
    all_summary = {}
    vis_queue = []
    
    for cfg in test_suite:
        print("\n" + "="*80)
        print(f" [RUN] Static Load Analysis - {cfg.mesh_type.upper()}")
        print("="*80)
        try:
            model, _ = build_structural_model(cfg)
            result = evaluate_static_response(model, cfg)
            all_summary[cfg.mesh_type] = result
            
            # Prepare for visualization
            meta = WHTMetadata(
                solver_name="JaxSSO", solver_version="2.1.0",
                analysis_type="static", coordinate_system="cartesian",
                unit_length="mm", unit_force="N"
            )
            vis_queue.append((cfg, result, model, meta))
            
        except Exception:
            print(f"\n[ERROR] {cfg.mesh_type} failed.")
            traceback.print_exc()
            all_summary[cfg.mesh_type] = None

    # --- STEP 3: Reporting ---
    print_result_table(all_summary)

    # --- STEP 4: Twist Scenario (New Request) ---
    cfg_twist = PipelineConfig(mesh_type='quad4', mesh_size_xy=60.0) 
    model_twist, _ = build_structural_model(cfg_twist)
    result_twist = evaluate_static_response_twist(model_twist, cfg_twist)
    
    # Add to visualization queue
    meta_twist = WHTMetadata(
        solver_name="JaxSSO", solver_version="2.1.0",
        analysis_type="static", coordinate_system="cartesian",
        unit_length="mm", unit_force="N"
    )
    vis_queue.append((cfg_twist, result_twist, model_twist, meta_twist))

    # --- STEP 5: Visualization ---
    if vis_queue:
        print("\n -> Launching WHT Visualizer for results...")
        # We'll visualize the Twist result (last in queue)
        cfg_v, res_v, model_v, meta_v = vis_queue[-1] 
        wht_data = res_v.to_wht_result_data(meta_v, model_v)
        
        viz = WHTVisualizer(title=f"Static Analysis - {cfg_v.mesh_type.upper()} ({res_v.analysis_type})", show=True)
        viz.load_results(wht_data, color="black")
        viz.plotter.view_isometric()
        if hasattr(viz.plotter, 'app'):
            viz.plotter.app.exec_()


if __name__ == "__main__":
    main()
