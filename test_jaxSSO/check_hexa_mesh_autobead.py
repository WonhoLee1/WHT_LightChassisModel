# -*- coding: utf-8 -*-
"""
Check 3D Solid Hexa Tray Mesh with Auto Beads
"""

import os
import sys
import numpy as np
import pyvista as pv

# Add workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mesh_utils import generate_solid_hexa_tray, apply_auto_beads
from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_selectors import apply_named_sets_by_recipe

def check_hexa_mesh_autobead():
    print(" -> Generating 3D Solid Hexa Tray Mesh...")
    width = 1800.0
    length = 1200.0
    height = 30.0
    thickness = 0.6
    
    # 1. Create the base flat tray (Centered)
    node_db, elem_db = generate_solid_hexa_tray(
        width=width, length=length, height=height, thickness=thickness, 
        mesh_size_xy=20.0, mesh_size_z=10.0, draft_angle=10.0, wall_layers=2,
        flange_width=10.0, origin='center'
    )
    
    print(f" -> Base Mesh: {len(node_db)} nodes, {len(elem_db)} HEXA8 elements")
    
    # 2. Apply Autobead Morphing
    node_db = apply_auto_beads(
        node_db, width=width, length=length, 
        margin=20.0, target_ratio=0.60, max_depth=10.0,
        min_len=50.0, max_len=300.0,
        min_width=50.0, max_width=150.0,
        smoothing_sigma=1.5, origin='center'
    )
    
    # 3. Initialize WHTMeshModel
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db, is_solid=True)

    # 4. Define Selection Recipe (Control point for Speaker/PCB/Flange)
    recipe = {
        "set_node-flange": {
            "type": "box", 
            "z_range": (height - 0.1, height + 0.1)
        },
        "set_node-spk_A": {
            "type": "cylinder", 
            "data": [(-300, -200, 40.0), (-700, -200, 40.0), (-300, -500, 40.0), (-700, -500, 40.0)],
            "z_range": (-thickness-50.1, 50.1)
        },
        "set_node-spk_B": {
            "type": "cylinder", 
            "data": [(300, -200, 40.0), (700, -200, 40.0), (300, -500, 40.0), (700, -500, 40.0)],
            "z_range": (-thickness-50.1, 50.1)
        },
        "set_node-pcb_A": {
            "type": "cylinder", 
            "data": [(-200, 50, 40.0), (-600, 50, 40.0), (-200, 200, 40.0), (-600, 200, 40.0)],
            "z_range": (-thickness-50.1, 50.1)
        },
        "set_node-pcb_B": {
            "type": "cylinder", 
            "data": [(200, 50, 40.0), (600, 50, 40.0), (200, 200, 40.0), (600, 200, 40.0)],
            "z_range": (-thickness-50.1, 50.1)
        }
    }
    
    print(" -> Applying Named Sets by Recipe...")
    apply_named_sets_by_recipe(model, recipe)
    
    # Print Diagnostics
    for set_name in recipe.keys():
        try:
            nids = model.get_nodes_by_set_name(set_name)
            print(f"    - '{set_name}': {len(nids)} nodes found.")
        except:
            print(f"    - '{set_name}': Set not found.")
    
    # 5. Build Visualization Grid
    ir = model.to_wht_result_data()
    cells = []
    for i in range(len(ir.offsets) - 1):
        c_nodes = ir.connectivity[ir.offsets[i]:ir.offsets[i+1]]
        cells.append(len(c_nodes))
        cells.extend(c_nodes)
    grid = pv.UnstructuredGrid(np.array(cells, dtype=np.int64), ir.cell_types, ir.nodes)
    
    # Visualize
    plotter = pv.Plotter(title="3D Solid Hexa Tray with Auto Beads & Sets")
    plotter.set_background("black")
    
    # Full mesh
    plotter.add_mesh(grid, show_edges=True, edge_color='gray', color='lightblue', smooth_shading=False, opacity=0.8)
    
    # Highlight sets
    colors = {
        "set_node-flange": "red", 
        "set_node-spk_A": "cyan", "set_node-spk_B": "blue",
        "set_node-pcb_A": "magenta", "set_node-pcb_B": "yellow"
    }
    nid_to_idx = model.node_id_to_index()
    sorted_nids = model.sorted_node_ids()
    
    for set_name, color in colors.items():
        try:
            nids = model.get_nodes_by_set_name(set_name)
            pdata = pv.PolyData(ir.nodes[[nid_to_idx[n] for n in nids]])
            plotter.add_mesh(pdata, color=color, point_size=5, render_points_as_spheres=True, label=set_name)
        except: pass

    # --- Interactive Point Picking ---
    def pick_callback(point):
        # Find the nearest node index by calculating distances to all nodes
        dists = np.linalg.norm(ir.nodes - point, axis=1)
        index = np.argmin(dists)
        
        real_nid = sorted_nids[index]
        coord_str = f"({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})"
        label = f"Node ID: {real_nid}\n{coord_str}"
        plotter.add_point_labels([point], [label], name='picked_label', font_size=12, text_color='white', shape_color='black', shadow=True)
        print(f" -> [Picked] Node ID: {real_nid} at {coord_str}")

    plotter.enable_point_picking(callback=pick_callback, show_message=True, font_size=10)
    
    plotter.add_legend()
    plotter.add_axes(line_width=3, color='white')
    plotter.camera_position = 'iso'
    print(" -> Opening PyVista Window...")
    plotter.show()

if __name__ == "__main__":
    check_hexa_mesh_autobead()
