# -*- coding: utf-8 -*-
"""
Check 2D Shell Tray Mesh with Auto Beads
"""

import os
import sys
import numpy as np
import pyvista as pv

# Add workspace root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mesh_utils import generate_shell_tray, apply_auto_beads
from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_selectors import apply_named_sets_by_recipe

def check_shell_mesh_autobead():
    print(" -> Generating Shell Tray Mesh...")
    width = 1800.0
    length = 1200.0
    height = 30.0
    thickness = 0.6
    
    # 1. Create the base flat tray (Centered)
    node_db, elem_db = generate_shell_tray(
        width=width, length=length, height=height,
        mesh_size_xy=20.0, mesh_size_z=10.0, draft_angle=10.0, flange_width=10.0,
        origin='center'
    )
    
    # 2. Apply Autobead Morphing
    node_db = apply_auto_beads(
        node_db, width=width, length=length, 
        margin=20.0, target_ratio=0.60, max_depth=10.0,
        min_len=50.0, max_len=300.0,
        min_width=50.0, max_width=150.0,
        smoothing_sigma=1.5, origin='center'
    )
    
    # 3. Initialize WHTMeshModel
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db, is_solid=False)

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
    
    # 5. Prepare WHTResultData & Dummy Load
    ir = model.to_wht_result_data()
    ir.time_values = np.array([0.0])
    
    nid_to_idx = model.node_id_to_index()
    sorted_nids = model.sorted_node_ids()
    
    # Load Marker(초록색 구슬) 기능을 확인하기 위해 가상의 하중 데이터 주입
    dummy_load = np.zeros((1, ir.nodes.shape[0], 3), dtype=np.float32)
    try:
        # spk_A 위치에 Z방향 하중(-100)이 가해진다고 가정
        spk_nids = model.get_nodes_by_set_name("set_node-spk_A")
        spk_idxs = [nid_to_idx[nid] for nid in spk_nids]
        dummy_load[0, spk_idxs, 2] = -100.0  
    except Exception: pass
    ir.point_data = {"Applied_Load": dummy_load}
    
    # 6. Launch WHTVisualizer
    from wht_visualizer.wht_visualizer import WHTVisualizer
    viz = WHTVisualizer(title="2D Shell Tray with Auto Beads & Sets", show=True)
    viz.load_results(ir)
    
    # WHTVisualizer가 자동 처리하는 Flange(BC), Spk_A(Load)를 제외한 나머지 셋 하이라이트 유지
    colors = {
        "set_node-spk_B": "blue",
        "set_node-pcb_A": "magenta", "set_node-pcb_B": "yellow"
    }
    
    for set_name, color in colors.items():
        try:
            nids = model.get_nodes_by_set_name(set_name)
            pdata = pv.PolyData(ir.nodes[[nid_to_idx[n] for n in nids]])
            viz.plotter.add_mesh(pdata, color=color, point_size=5, render_points_as_spheres=True, label=set_name)
        except: pass

    # --- Interactive Point Picking ---
    def pick_callback(point):
        # Find the nearest node index by calculating distances to all nodes
        dists = np.linalg.norm(ir.nodes - point, axis=1)
        index = np.argmin(dists)
        
        real_nid = sorted_nids[index]
        coord_str = f"({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})"
        label = f"Node ID: {real_nid}\n{coord_str}"
        viz.plotter.add_point_labels([point], [label], name='picked_label', font_size=12, text_color='white', shape_color='black', shadow=True)
        print(f" -> [Picked] Node ID: {real_nid} at {coord_str}")

    viz.plotter.enable_point_picking(callback=pick_callback, show_message=True, font_size=10)
    
    viz.plotter.add_legend()
    viz.plotter.view_isometric()
    print(" -> Opening WHTVisualizer Window...")
    viz.show()

if __name__ == "__main__":
    check_shell_mesh_autobead()
