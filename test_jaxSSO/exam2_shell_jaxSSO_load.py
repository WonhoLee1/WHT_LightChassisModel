import os
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# Add workspace to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_solver import WHTSolver
from wht_solver.load_cases import WHTLoadCase
from wht_visualizer.wht_visualizer import visualize, export_to_wht_result
from test_jaxSSO.exam2_shell_jaxSSO import generate_mesh

@dataclass
class PipelineConfig:
    mesh_type: str = 'quad4'          # 'quad4' or 'tria3'
    width: float = 1800.0
    length: float = 1200.0
    height: float = 10.0
    thickness: float = 0.6
    mesh_size_xy: float = 40.0
    draft_angle: float = 15.0
    flange_width: float = 10.0
    E: float = 210.0e3                # MPa
    nu: float = 0.3
    rho: float = 7.85e-9              # ton/mm^3
    num_modes: int = 16
    
    # Static Load Specifics
    load_area_size: float = 200.0     # Square area side length
    total_force: float = 1000.0       # Total downward force (N)

def build_model(node_db, elem_db, cfg):
    model = WHTMeshModel()
    # Add nodes
    for nid, coords in node_db.items():
        model.add_node(nid, coords[0], coords[1], coords[2])
    
    # Add Material & Property
    mid = 1
    pid = 1
    model.add_material(mid, cfg.E, cfg.nu, cfg.rho)
    model.add_property(pid, "PSHELL", cfg.thickness, mid)
    
    # Add elements
    for eid, nids in elem_db.items():
        # Derive type from node count
        etype = "TRIA3" if len(nids) == 3 else "QUAD4"
        model.add_element(eid, nids, etype, pid=pid)
        
    # Boundary Conditions: Simply supported on all 4 edges (Z constraint) at the FLANGE
    # Top flange is at z == height
    fixed_count = 0
    for nid, node in model.nodes.items():
        # Fix the flange (top perimeter)
        if abs(node.z - cfg.height) < 0.1:
            model.apply_spc(nid, (0, 1, 2, 3, 4, 5))
            fixed_count += 1
    print(f"    Nodes constrained (flange): {fixed_count}")
            
    return model

def solve_static_load(model, cfg):
    solver = WHTSolver(model)
    lc = WHTLoadCase("AreaLoad")
    
    # Diagnostic: Print mesh coordinate ranges
    nodes_xyz = np.array([n.coords() for n in model.nodes.values()])
    x_range = (nodes_xyz[:,0].min(), nodes_xyz[:,0].max())
    y_range = (nodes_xyz[:,1].min(), nodes_xyz[:,1].max())
    z_range = (nodes_xyz[:,2].min(), nodes_xyz[:,2].max())
    print(f"    Mesh Ranges: X={x_range}, Y={y_range}, Z={z_range}")

    # Find nodes in center area for distributed load
    area_dim = cfg.load_area_size
    center_nids = []
    for nid, node in model.nodes.items():
        # Stay near bottom (z < 1.0) and within square
        if (abs(node.x) <= area_dim/2 and 
            abs(node.y) <= area_dim/2 and 
            node.z < 1.0):
            center_nids.append(nid)
    
    if not center_nids:
        # Fallback to single closest node if area is too small for mesh size
        min_dist = float('inf')
        fallback_nid = -1
        for nid, node in model.nodes.items():
            dist = np.sqrt(node.x**2 + node.y**2 + node.z**2)
            if dist < min_dist:
                min_dist = dist
                fallback_nid = nid
        center_nids = [fallback_nid]
        print(f" <!> No nodes in {area_dim}mm area. Falling back to single node.")
    
    num_nodes = len(center_nids)
    force_per_node = -cfg.total_force / num_nodes
    print(f" -> Applying Wide Load: {area_dim}x{area_dim} mm area ({num_nodes} nodes, Total F={cfg.total_force}N)")
    
    # Apply distributed load
    for nid in center_nids:
        lc.add_force(nid, (2,), (force_per_node,))
    
    result = solver.solve_static(lc)
    return result

def print_summary_table(all_results):
    print("\n" + "="*95)
    print(f" {'Static Displacement & Stress Results':^93}")
    print("="*95)
    header = f"{'Mesh Type':<15} | {'Points':<8} | {'Max Uz (mm)':<15} | {'Max Stress (MPa)':<15} | {'Diff (%)':<10}"
    print(header)
    print("-" * 95)
    
    ref_val = None
    for mtype, res in all_results.items():
        if res is None:
            print(f"{mtype.upper():<15} | {'FAILED':<8} | {'-':<15} | {'-':<15} | {'-':<10}")
            continue
            
        # Max Z-displacement (Uz is index 2)
        uz_data = res.displacement[:, 2]
        max_uz = np.abs(uz_data).max()
        
        # Max Stress from diagnostic
        max_vm = getattr(res, '_max_vm_diagnostic', 0.0)
        
        if ref_val is None:
            ref_val = max_uz
            diff_pct = 0.0
        else:
            diff_pct = (max_uz - ref_val) / ref_val * 100.0
            
        print(f"{mtype.upper():<15} | {res.displacement.shape[0]:<8} | {max_uz:15.4f} | {max_vm:15.4f} | {diff_pct:8.2f}%")
    print("="*95)

def main():
    cfgs = [
        PipelineConfig(mesh_type='quad4', load_area_size=100, total_force=23),
        #PipelineConfig(mesh_type='tria3', load_area_size=100, total_force=23),
        #PipelineConfig(mesh_type='mixed', load_area_size=100, total_force=23),
        PipelineConfig(mesh_type='tria3_free', load_area_size=100, total_force=23),
    ]
    
    all_summary = {}
    all_vis_data = [] # Collect data for bulk visualization
    
    for cfg in cfgs:
        print(f"\n[{cfg.mesh_type.upper()}] Static Analysis Pipeline Started...")
        try:
            node_db, elem_db = generate_mesh(cfg)
            model = build_model(node_db, elem_db, cfg)
            result = solve_static_load(model, cfg)
            all_summary[cfg.mesh_type] = result
            
            # Export for potential visualization
            from wht_visualizer.wht_visualizer import export_to_wht_result
            wht_data = export_to_wht_result(model, result)
            all_vis_data.append(wht_data)
            
        except Exception as e:
            print(f"[ERROR] {cfg.mesh_type} failed: {e}")
            import traceback
            traceback.print_exc()
            all_summary[cfg.mesh_type] = None
            
    print_summary_table(all_summary)

    # Now open all visualizer windows at once
    if all_vis_data:
        print(f"\n -> Opening {len(all_vis_data)} visualization windows...")
        from wht_visualizer.wht_visualizer import visualize
        for i, data in enumerate(all_vis_data):
            # Only block on the last window to keep all windows open
            is_last = (i == len(all_vis_data) - 1)
            visualize(data, block=is_last)

if __name__ == "__main__":
    main()
