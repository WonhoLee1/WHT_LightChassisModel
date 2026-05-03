# -*- coding: utf-8 -*-
"""
test_morphing.py
================
Tests the WHTMorpher by generating a base tray (Mesh A), creating a mock bead
model (Mesh B), aligning them, and morphing Mesh A to B. Finally, computes modal
frequencies for both original and morphed meshes.
"""

import sys
import os
import math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_morphing import WHTMorpher
from wht_solver.wht_solver import WHTSolver
from wht_converter.wht_models import WHTMetadata
from wht_visualizer.wht_visualizer import WHTVisualizer
from mesh_utils import generate_shell_tray
from wht_modeler.wht_selector import WHTSelector

def generate_mock_bead_model_k(filepath: str, width=1000.0, length=800.0, nx=150, ny=120):
    """Generates a high-res mesh with complex beads and saves as LS-DYNA .k."""
    with open(filepath, 'w') as f:
        f.write("*KEYWORD\n*NODE\n")
        dx = width / nx
        dy = length / ny
        nid = 1
        
        # Center at origin for easy generation
        cx = width / 2.0
        cy = length / 2.0
        
        nodes = {}
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = i * dx - cx
                y = j * dy - cy
                
                # Smoother composite beads (Increased divisors for stability)
                z = 10.0 * math.sin(x / 80.0) * math.cos(y / 80.0)
                z += 3.0 * math.sin(x / 50.0) 
                z += 2.0 * math.cos(y / 45.0) 
                
                # Outer flange region (flat)
                if abs(x) > (cx - 100) or abs(y) > (cy - 100):
                    z = 0.0
                
                f.write(f"{nid:8d}{x:16.5f}{y:16.5f}{z:16.5f}\n")
                nodes[(i, j)] = nid
                nid += 1
                
        f.write("*ELEMENT_SHELL\n")
        eid = 1
        pid = 1
        for j in range(ny):
            for i in range(nx):
                n1 = nodes[(i, j)]
                n2 = nodes[(i+1, j)]
                n3 = nodes[(i+1, j+1)]
                n4 = nodes[(i, j+1)]
                f.write(f"{eid:8d}{pid:8d}{n1:8d}{n2:8d}{n3:8d}{n4:8d}\n")
                eid += 1
        f.write("*END\n")
    print(f" -> Generated smooth mock target mesh: {filepath}")

def build_base_model(mesh_size: float = 40.0) -> WHTMeshModel:
    """Builds Mesh A (Chassis Tray) using mesh_utils with specified size."""
    print(f" -> Generating Base Mesh (Mesh A, size={mesh_size})...")
    width = 1000.0
    length = 800.0
    node_db, elem_db = generate_shell_tray(
        width=width, length=length, height=30.0,
        mesh_size_xy=mesh_size, mesh_size_z=10.0,
        draft_angle=30.0, flanges=(True, True, True, True),
        flange_segments=[(20.0, 0.0)], origin='center'
    )
    print(f"    - Mesh Created: {len(node_db)} nodes, {len(elem_db)} elements.")
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db)
    
    # Add material and properties for solver
    MID, PID = 1, 1
    model.add_material(MID, E=210000.0, nu=0.3, rho=7.85e-9)
    model.add_property(PID, "PSHELL", 1.5, MID)
    for elem in model.elements.values():
        elem.pid = PID
        
    return model

def solve_modal(model: WHTMeshModel, title: str):
    """Runs modal analysis and returns the first 5 natural frequencies."""
    print(f" -> Solving Modal Analysis: {title}")
    
    # Boundary Conditions: Fix all flange outer edges
    rim_nids = WHTSelector(model).by_box(z=(29.0, 31.0)).get_ids()
    model.apply_spc(rim_nids, (0, 1, 2, 3, 4, 5))
    
    solver = WHTSolver(model)
    meta = WHTMetadata("JaxSSO", "v1.0", "modal", "cartesian", "mm", "N")
    result = solver.solve_modal(num_modes=5, method='sparse')
    
    if result and hasattr(result, "frequencies"):
        print(f"    [Result] {title} Frequencies: {[round(f, 2) for f in result.frequencies]} Hz")
    
    # Needs to return WHTResultData for visualizer
    return result.to_wht_result_data(meta, model)

def main():
    target_file = "bead_model.k"
    # Target Mesh (Mesh B) - Fine
    if not os.path.exists(target_file):
        generate_mock_bead_model_k(target_file)
        
    # 1. Original Reference (Fine - 8mm)
    res_orig = solve_modal(build_base_model(8.0), "Fine Original (8mm)")
    
    # 2. Base Model to Morph (Rough - 50mm)
    rough_model = build_base_model(50.0)
    
    # 3. Morphing the Rough Model
    print(" -> Initializing Morpher for Rough Model...")
    morpher = WHTMorpher(rough_model)
    morpher.load_target_mesh(target_file)
    morpher.align_meshes(z_offset=0.0, align_centers=True)
    
    print(" -> Morphing floor region (Rough)...")
    morpher.morph(
        bounding_box=(-490, 490, -390, 390, -5, 5), 
        direction=np.array([0, 0, 1]),
        max_dist=20.0, 
        area_tolerance=0.1
    )
    
    # 4. Solve Morphed Rough Model
    res_morphed = solve_modal(rough_model, "Rough Morphed (50mm)")
    
    # 5. Visualize
    print(" -> Launching Visualizer (Fine Original vs Rough Morphed)...")
    app = WHTVisualizer()
    app.load_results(res_orig, group_name="Original_Fine", clear=True)
    app.load_results(res_morphed, group_name="Morphed_Rough", clear=False)
    
    # Adjust appearance
    if "Original_Fine_Mesh_Model" in app.parts:
        app.parts["Original_Fine_Mesh_Model"]["actor"].prop.opacity = 0.3
        app.parts["Original_Fine_Mesh_Model"]["actor"].prop.color = "grey"
        
    if "Morphed_Rough_Mesh_Model" in app.parts:
        app.parts["Morphed_Rough_Mesh_Model"]["actor"].prop.opacity = 0.8
        app.parts["Morphed_Rough_Mesh_Model"]["actor"].prop.color = "gold"

    app.show()

if __name__ == "__main__":
    main()
