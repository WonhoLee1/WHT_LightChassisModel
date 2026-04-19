"""
check_tray_sets.py
==================
Verification script for Automated Set Generation (Flange, Speaker, PCB)
on a centered Tray model.
"""

import sys
import os
import numpy as np
import pyvista as pv

# Add workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from test_jaxSSO.mesh_utils import generate_solid_hexa_tray
from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_selectors import apply_named_sets_by_recipe

def main():
    width, length, height = 1800.0, 1200.0, 30.0
    thickness = 0.6
    
    print(" -> Generating Centered Solid Tray Mesh...")
    node_db, elem_db = generate_solid_hexa_tray(
        width=width, length=length, height=height, thickness=thickness, 
        mesh_size_xy=20.0, mesh_size_z=10.0, draft_angle=10.0, wall_layers=2,
        flange_width=10.0, origin='center'
    )
    
    # Initialize WHTMeshModel
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db, is_solid=True)
    
    # Define selection recipe based on user request
    recipe = {
        "set_node-flange": {
            "type": "box", 
            "z_range": (height - 0.1, height + 0.1)
        },
        "set_node-spk_A": {
            "type": "cylinder", 
            "data": [(-100, -400, 20.0), (-400, -400, 20.0), (100, -500, 20.0), (400, -500, 20.0)],
            "z_range": (-thickness-0.1, 0.1) # Flat floor area
        },
        "set_node-pcb_A": {
            "type": "cylinder", 
            "data": [(-100, 50, 20.0), (-400, 50, 20.0), (100, 100, 20.0), (400, 100, 20.0)],
            "z_range": (-thickness-0.1, 0.1)
        }
    }
    
    print(" -> Applying Named Sets by Recipe...")
    apply_named_sets_by_recipe(model, recipe)
    
    # Report findings
    for name in recipe.keys():
        try:
            nids = model.get_nodes_by_set_name(name)
            print(f"    - '{name}': {len(nids)} nodes found.")
        except KeyError:
            print(f"    - '{name}': NOT FOUND.")
            
    # Visualization using PyVista
    print(" -> Opening PyVista for visual confirmation...")
    plotter = pv.Plotter(title="WHT Tray Sets Validation")
    
    # Convert model to UnstructuredGrid (via converter IR)
    ir = model.to_wht_result_data()
    
    # Build PyVista-compatible cell array: [n1, p1, p2..., n2, p3, p4...]
    cells = []
    for i in range(len(ir.offsets) - 1):
        c_nodes = ir.connectivity[ir.offsets[i]:ir.offsets[i+1]]
        cells.append(len(c_nodes))
        cells.extend(c_nodes)
    cells = np.array(cells, dtype=np.int64)
    
    grid = pv.UnstructuredGrid(cells, ir.cell_types, ir.nodes)
    
    # Base mesh
    plotter.add_mesh(grid, color='white', opacity=0.3, show_edges=True, edge_color='gray')
    
    # Highlight sets
    colors = {"set_node-flange": "red", "set_node-spk_A": "cyan", "set_node-pcb_A": "magenta"}
    
    for set_name, color in colors.items():
        try:
            nids = model.get_nodes_by_set_name(set_name)
            # Find indices in the grid (which uses sorted node IDs from WHTMeshModel)
            nid_to_idx = model.node_id_to_index()
            indices = [nid_to_idx[nid] for nid in nids]
            
            pdata = pv.PolyData(ir.nodes[indices])
            plotter.add_mesh(pdata, color=color, point_size=10.0, render_points_as_spheres=True, label=set_name)
        except:
            pass
            
    plotter.add_legend()
    plotter.add_axes()
    plotter.show()

if __name__ == "__main__":
    main()
