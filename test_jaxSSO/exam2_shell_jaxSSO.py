# -*- coding: utf-8 -*-
"""
exam2_shell_jaxSSO.py
=====================
WHT FEM Framework — Shell Tray Modal Analysis Example (Advanced Flange Version)

Overview:
---------
This script executes a high-fidelity modal analysis for a structural shell tray
featuring a complex "Hook-like" tiered flange structure:
Sequence: Rise -> Inward -> Fall -> Inward.

Standardization:
----------------
- Mapping: (Index 0: Bottom/Ymin, 1: Right/Xmax, 2: Top/Ymax, 3: Left/Xmin)
- Mesh: High-fidelity QUAD4 with Mass Participation filtering.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple
import numpy as np
import traceback

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_solver import WHTSolver
from wht_visualizer.wht_visualizer import WHTVisualizer
from wht_converter.wht_models import WHTMetadata
from mesh_utils import generate_shell_tray


# ==============================================================================
# 0. CONFIGURATION & SCHEMA
# ==============================================================================

@dataclass
class PipelineConfig:
    """
    Unified settings for the Shell Tray Analysis Pipeline.
    """
    # Geometry Dimensions [mm]
    width:  float = 1800.0
    length: float = 1200.0
    height: float = 35.0
    thickness: float = 0.5
    
    # Mesh Control [mm]
    # Note: mesh_size_xy is reduced to capture small flange details (10-15mm)
    mesh_type: str = 'quad4'
    mesh_size_xy: float = 40.0
    mesh_size_z:  float = 10.0
    draft_angle:  float = 35.0
    
    # Flange Configuration
    # Mapping: (Index 0: Bottom, 1: Right, 2: Top, 3: Left)
    flanges: tuple = (False, True, True, True) # Bottom side flange = False
    flange_segments: List[Tuple[float, float]] = field(default_factory=list)
    
    # Modal Solver Settings
    num_modes: int = 12
    solver_method: str = 'auto'
    exclude_rigid_body: Union[bool, str] = 'mass'
    shift_hz: Optional[float] = None


# ==============================================================================
# 1. CORE PIPELINE FUNCTIONS
# ==============================================================================

def build_structural_model(cfg: PipelineConfig) -> Tuple[WHTMeshModel, Dict]:
    """Generates geometry and applies physical material properties."""
    print(f" -> Generating Geometry [{cfg.mesh_type.upper()}] with Hook-Flanges...")
    
    # Generate Raw Mesh 
    node_db, elem_db = generate_shell_tray(
        width=cfg.width, length=cfg.length, height=cfg.height,
        mesh_size_xy=cfg.mesh_size_xy, mesh_size_z=cfg.mesh_size_z,
        draft_angle=cfg.draft_angle, flange_segments=cfg.flange_segments,
        flanges=cfg.flanges, mesh_type=cfg.mesh_type
    )
    print(f"    [MeshInfo] Nodes: {len(node_db)} | Elements: {len(elem_db)}")

    # Convert to WHT Object-Oriented Mesh Model
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db)
    
    # Add Physics (Steel)
    MID, PID = 1, 1
    model.add_material(MID, E=210000.0, nu=0.3, rho=7.85e-9)
    model.add_property(PID, "PSHELL", cfg.thickness, MID)
    
    for elem in model.elements.values():
        elem.pid = PID
        
    return model, node_db


def evaluate_modal_response(model: WHTMeshModel, cfg: PipelineConfig):
    """Executes the modal solver and applies the chosen filtering strategy."""
    print(f" -> Solving Modal Problem (Target: {cfg.num_modes} elastic modes)...")
    solver = WHTSolver(model)
    results = solver.solve_modal(
        num_modes=cfg.num_modes, 
        method=cfg.solver_method,
        exclude_rigid_body=cfg.exclude_rigid_body,
        shift_hz=cfg.shift_hz
    )
    return results


# ==============================================================================
# 2. MAIN EXECUTION
# ==============================================================================

def main():
    # --- STEP 1: Complex Flange Design (The "Hook-Fold" Sequence) ---
    # Width (seg_w): Change in horizontal offset (+: Outward, -: Inward)
    # Height (seg_dz): Change in vertical position (+: Up, -: Down)
    
    hook_sequence = [
        ( 0.0,  12.0), # 1. 상승 (Rise): w=0, dz=15
        (-5.0,  0.0), # 2. 내측이동 (Inward): w=-20, dz=0
        ( 0.0, -10.0), # 3. 하락 (Fall): w=0, dz=-15
        (-10.0,  0.0)  # 4. 내측이동 (Inward): w=-20, dz=0
    ]
    
    # Target Configuration
    cfg = PipelineConfig(
        #mesh_type='quad4',
        mesh_type='tria3_free',
        flanges=(False, True, True, True), # Bottom Disabled
        flange_segments=hook_sequence,
        num_modes=10
    )

    print("\n" + "="*80)
    print(f" [ANALYSIS RUN] COMPLEX HOOK-FLANGE TRAY (Bottom: False)")
    print("="*80)
    
    try:
        # Pipeline execution
        model, _ = build_structural_model(cfg)
        results = evaluate_modal_response(model, cfg)
        
        # 3. Report Results
        print("\n" + "#"*80)
        print(" #  ELASTIC NATURAL FREQUENCIES (Hz) - (Mass Participation Filtered)   #")
        print("#"*80)
        for i, f in enumerate(results.frequencies):
            print(f"  Mode {i+1:2d}: {f:8.3f} Hz")
        print("-" * 80)

        # 4. Visualization
        print("\n -> Launching Visualizer for structural inspection...")
        meta = WHTMetadata(
            solver_name="JaxSSO", solver_version="2.1.0",
            analysis_type="modal", coordinate_system="cartesian",
            unit_length="mm", unit_force="N"
        )
        wht_data = results.to_wht_result_data(meta, model)
        
        viz = WHTVisualizer(title="WHT Visualizer - Hook-Flange Modal Shapes", show=True)
        viz.load_results(wht_data, color="black")
        viz.plotter.view_isometric()
        
        if hasattr(viz.plotter, 'app'):
            viz.plotter.app.exec_()
            
    except Exception:
        print("\n[CRITICAL ERROR] Pipeline failed.")
        traceback.print_exc()


if __name__ == "__main__":
    main()
