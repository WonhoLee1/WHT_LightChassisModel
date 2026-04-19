# -*- coding: utf-8 -*-
"""
[WHT_LightChassisModel] Exam 1: Natural Frequency Analysis (Standard Template)
=============================================================================
1. Geometry: Shell Tray with Draft Angle & Flange
2. Data Model: WHTMeshModel (Standard Interface)
3. Solver: JaxSSO (Standard Modal Wrapper)
4. Pipeline: Unified to_wht_result_data() Path
5. Units: MPa, ton, mm (Industrial Standard)
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_entities import WHTSPCEntry
from wht_solver.wht_solver import WHTSolver
from wht_converter.wht_models import WHTMetadata
from wht_converter.wht_exporters import VTUPVDExporter, VTKHDFExporter
from wht_visualizer.wht_visualizer import WHTVisualizer
from test_jaxSSO.mesh_utils import generate_shell_tray

# --- Industrial Material: Steel (MPa, ton, mm) ---
MAT_STEEL = {
    'E': 210000.0,   # Elastic Modulus (MPa)
    'nu': 0.3,       # Poisson's ratio
    'rho': 7.85e-9,  # Density (ton/mm^3)
    't': 0.6         # Standard sheet thickness (mm)
}

def run_standard_modal_pipeline(width=1800.0, length=1400.0, height=30.0, 
                               draft_angle=15.0, flange_width=10.0, num_modes=10):
    """
    Standardizes the modal analysis flow for shell structures.
    """
    print("\n" + "="*60)
    print(f" [Exam 1] Standard Shell Modal Pipeline: {width}x{length}")
    print("="*60 + "\n")

    # 1. Mesh Generation
    print(f" -> Generating Mesh (Draft: {draft_angle}nd, Flange: {flange_width}mm)...")
    node_db, elem_db = generate_shell_tray(
        width=width, length=length, height=height, 
        mesh_size_xy=40.0, mesh_size_z=10.0,
        draft_angle=draft_angle, flange_width=flange_width, origin='center'
    )
    
    # 2. Build WHTMeshModel
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db, name="StandardTray", is_solid=False)
    
    # Add Material & Property
    model.add_material(1, E=MAT_STEEL['E'], nu=MAT_STEEL['nu'], rho=MAT_STEEL['rho'])
    model.add_property(1, "PSHELL", t=MAT_STEEL['t'], mid=1)
    for eid in model.elements: 
        model.elements[eid].pid = 1
    
    # 3. Apply Boundary Conditions (Fix top flange)
    print(" -> Applying Boundary Conditions (Fixed Top Flange)...")
    fixed_nids = []
    for nid, coords in node_db.items():
        if abs(coords[2] - height) < 0.1: # Top flange location
            model.apply_spc(nid, dofs=(0, 1, 2, 3, 4, 5))
            fixed_nids.append(nid)
    print(f"    Nodes constrained: {len(fixed_nids)}")

    # 4. Initialize Solver & Mass Sanity Check
    print(" -> Initializing Solver & Verifying Physical Mass...")
    solver = WHTSolver(model)
    
    # Internal check of the mass matrix
    jm, _, _ = solver._build_jaxsso_model()
    jm.model_ready()
    nid_to_idx = {nid: i for i, nid in enumerate(model.sorted_node_ids())}
    M_diag = solver._assemble_lumped_mass(jm, jm.ndof, model.sorted_node_ids(), nid_to_idx)
    total_mass = np.sum(M_diag[::6]) # Physical mass from translation DOFs
    print(f"    Total Model Mass: {total_mass:.6f} tons (~{total_mass*1000:.2f} kg)")

    # 5. Solve Modal
    print(f" -> Solving for {num_modes} Natural Frequencies...")
    results = solver.solve_modal(num_modes=num_modes)
    
    print("\n[RESULT] Natural Frequencies (Hz):")
    for i, f in enumerate(results.frequencies):
        print(f"  Mode {i+1:02d}: {f:8.2f} Hz")

    # 6. Unified Data Conversion & Export
    print("\n -> Converting results via Unified Result Path...")
    meta = WHTMetadata(
        solver_name="JaxSSO", solver_version="1.0.0", analysis_type="modal",
        coordinate_system="cartesian", unit_length="mm", unit_force="N"
    )
    wht_data = results.to_wht_result_data(meta, model)
    
    # Organize into Parts (Floor vs Walls) for demonstration
    floor_indices = []
    wall_indices = []
    for i, eid in enumerate(model.sorted_element_ids()):
        # Floor has all nodes near Z=0 (since extruded from Z=0 in mesh_utils)
        nids = model.elements[eid].node_ids
        if all(abs(node_db[nid][2] - 0.0) < 1e-3 for nid in nids):
            floor_indices.append(i)
        else:
            wall_indices.append(i)
            
    wht_data.element_sets = {
        "Tray_Floor": np.array(floor_indices),
        "Tray_Walls": np.array(wall_indices)
    }

    # Export
    stamp = datetime.now().strftime("D%Y%m%d-%H%M%S")
    out_dir = Path(__file__).resolve().parent.parent / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    VTKHDFExporter().export(wht_data, str(out_dir/"standard_modal.hdf"))
    print(f" -> Exported result to: {out_dir}")

    # 7. Professional Visualization
    print(" -> Launching WHT Visualizer...")
    viz = WHTVisualizer(title="Exam 1: Standard Modal Analysis", show=True)
    viz.load_results(wht_data, color="white", label="Modal Shape")
    viz.plotter.view_xy()
    viz.plotter.reset_camera()
    
    if hasattr(viz.plotter, 'app'):
        viz.plotter.app.exec_()

if __name__ == "__main__":
    # Standard benchmark: 1m x 1m steel tray with inclined walls
    run_standard_modal_pipeline(
        width=1800.0, 
        length=1200.0, 
        height=10.0, 
        draft_angle=10.0, 
        flange_width=10.0
    )
