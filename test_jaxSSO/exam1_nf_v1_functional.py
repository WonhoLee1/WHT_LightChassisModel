# -*- coding: utf-8 -*-
"""
[WHT_LightChassisModel] Exam 1: Natural Frequency Analysis using JaxSSO
======================================================================
1. Geometry: 100x100x10 Shell Tray (Gmsh)
2. Solver: JaxSSO (Stiffness) + Custom Lumped Mass Matrix
3. BC: Fixed tray corner edges
4. Output: Mode shapes (Natural Frequencies) visualized via PyVista
"""

import os
import sys
import numpy as np
import jax.numpy as jnp
from JaxSSO.model import Model
from JaxSSO import assemblemodel
import pyvista as pv
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

# Local project imports
from mesh_utils import generate_shell_tray, get_nodes_in_box, apply_fixed_bc

def run_nf_analysis():
    print("--- [WHT] Exam 1: Natural Frequency Analysis (JaxSSO) ---")
    
    # 1. Generate Mesh (Gmsh)
    width, length, height = 1000.0, 1000.0, 5.0
    mesh_size_xy = 50.0
    mesh_size_z = 5.0
    node_db, elem_db = generate_shell_tray(width, length, height, mesh_size_xy, mesh_size_z)
    print(f" -> Mesh Generated: {len(node_db)} nodes, {len(elem_db)} elements")

    # 2. Material & Property Definition (User Requested: 1000.0 MPa)
    mat = {
        'E': 1000.0,      # Young's Modulus (MPa)
        'nu': 0.4,        # Poisson's ratio
        'rho': 1.0e-9,    # Density (t/mm^3)
        't': 2.0          # Thickness (mm)
    }
    print(f" -> Material Defined: E={mat['E']} MPa, rho={mat['rho']} t/mm3")
    
    # 3. Build JaxSSO Model
    model = Model()
    
    # Add Nodes
    for nid, coords in node_db.items():
        model.add_node(nid, coords[0], coords[1], coords[2])
    
    # Add Elements (MITC4 Quads)
    quad_count = 0
    tria_count = 0
    for eid, nids in elem_db.items():
        if len(nids) == 4:
            model.add_quad(eid, nids[0], nids[1], nids[2], nids[3], 
                           mat['t'], mat['E'], mat['nu'])
            quad_count += 1
        else:
            tria_count += 1
            
    print(f" -> Elements Added: {quad_count} Quads")
    if tria_count > 0:
        print(f" > [WARNING] {tria_count} triangles detected. JaxSSO does not support Tria shells.")

    # 4. Apply Boundary Conditions (Fixed Nodes)
    rim_node_ids = get_nodes_in_box(node_db, z_range=(height-0.1, height+0.1))
    print(f" -> Applying Fixed BC to {len(rim_node_ids)} nodes at Z={height} (Rim)")
    apply_fixed_bc(model, rim_node_ids, dofs=(0, 1, 2, 3, 4, 5))
    
    base_corners = get_nodes_in_box(node_db, z_range=(-0.1, 0.1), x_range=(-0.1, width+0.1), y_range=(-0.1, length+0.1))
    actual_corners = []
    for nid in base_corners:
        x, y, z = node_db[nid]
        if (x < 0.1 or x > width-0.1) and (y < 0.1 or y > length-0.1):
            actual_corners.append(nid)
    print(f" -> Fixing {len(actual_corners)} base corner nodes for stability")
    apply_fixed_bc(model, actual_corners, dofs=(0, 1, 2, 3, 4, 5))

    # 5. Assemble K (JaxSSO)
    model.model_ready()
    K_bcoo = assemblemodel.model_K(model)
    ndof = model.get_dofs()
    K_scipy = csr_matrix((np.array(K_bcoo.data), (np.array(K_bcoo.indices[:,0]), np.array(K_bcoo.indices[:,1]))), shape=(ndof, ndof))

    # 6. Assemble M (Hand-made Lumped Mass)
    print(" -> Assembling Lumped Mass Matrix...")
    M_diag = np.zeros(ndof)
    for eid, nids in elem_db.items():
        if len(nids) == 4:
            c = [node_db[n] for n in nids]
            v1 = c[1] - c[0]
            v2 = c[3] - c[0]
            area = np.linalg.norm(np.cross(v1, v2))
            
            elem_mass = area * mat['t'] * mat['rho']
            mass_per_node = elem_mass / 4.0
            
            for nid in nids:
                M_diag[nid*6 : nid*6+3] += mass_per_node
                M_diag[nid*6+3 : nid*6+6] += mass_per_node * (mesh_size_xy**2) / 12.0 + 1e-15 # small epsilon
                
    # 7. Solve Eigenvalues
    known_id = model.known_indices
    all_dofs = np.arange(ndof)
    unknown_id = np.setdiff1d(all_dofs, known_id)
    
    K_free = K_scipy[unknown_id, :][:, unknown_id]
    M_free = M_diag[unknown_id]
    
    num_modes = 10
    print(f" -> Solving Eigenvalues for {num_modes} modes...")
    vals, vecs_free = eigsh(K_free, k=num_modes, M=csr_matrix(np.diag(M_free)), which='LM', sigma=0)
    
    freqs = np.sqrt(np.maximum(vals, 0)) / (2 * np.pi)
    print("\n[RESULT] Natural Frequencies (Hz):")
    for i, f in enumerate(freqs):
        print(f"  Mode {i+1:02d}: {f:8.2f} Hz")

    # 8. Visualization (PyVista) - [WHT] Debugging Distortion & Mixed Mesh
    print("\n -> Opening PyVista Visualization (Geometry Check)...")
    mode_idx = 0
    u_full = np.zeros(ndof)
    u_full[unknown_id] = vecs_free[:, mode_idx]
    
    # [WHT] Robust Indexing to prevent "Twisting"
    sorted_node_ids = sorted(node_db.keys())
    nodes_array = np.array([node_db[nid] for nid in sorted_node_ids])
    nid_to_idx = {nid: i for i, nid in enumerate(sorted_node_ids)}
    
    cells = []
    cell_types = []
    for eid, nids in elem_db.items():
        n_node = len(nids)
        if n_node in [3, 4]:
            cells.append(n_node)
            # Ensure CCW ordering from Gmsh is preserved
            cells.extend([nid_to_idx[n] for n in nids])
            # 5: Tria, 9: Quad
            cell_types.append(5 if n_node == 3 else 9)
            
    grid = pv.UnstructuredGrid(cells, cell_types, nodes_array)
    
    # Process Displacement
    disp = u_full.reshape(-1, 6)[:, :3]
    # Align displacement with sorted nodes
    node_disp = np.array([disp[nid] for nid in sorted_node_ids])
    
    max_d = np.max(np.linalg.norm(node_disp, axis=1))
    norm_disp = node_disp / (max_d + 1e-12)
    
    grid.point_data["Mode_Vector"] = norm_disp
    grid.point_data["Disp_Magnitude"] = np.linalg.norm(node_disp, axis=1)
    
    # Warp by mode shape
    warped = grid.warp_by_vector("Mode_Vector", factor=15.0) 
    
    plotter = pv.Plotter(title=f"JaxSSO Modal: Mode 1 ({freqs[mode_idx]:.1f} Hz)")
    plotter.set_background("black")
    
    # Show Original Mesh in Wireframe for comparison
    plotter.add_mesh(grid, style='wireframe', color='grey', opacity=0.2, label='Undeformed')
    
    # Show Warped Mesh
    plotter.add_mesh(warped, scalars="Disp_Magnitude", cmap="jet", 
                     show_edges=True, edge_color="grey", line_width=1,
                     smooth_shading=False, label='Deformed Mode Shape') 

    # --- [WHT] Visualize Boundary Conditions ---
    bc_node_indices = np.unique(np.array(known_id) // 6)
    bc_coords = nodes_array[bc_node_indices]
    bc_points = pv.PolyData(bc_coords)
    plotter.add_mesh(bc_points, color='red', point_size=10.0, 
                     render_points_as_spheres=True, label='Fixed BC (Points)')
    
    plotter.add_scalar_bar(title="Displacement", title_font_size=14, label_font_size=14, color='white')
    plotter.add_axes(line_width=2, color='white')
    plotter.add_legend(size=(0.2, 0.2), bcolor=None, face=None) # Legend size also slightly adjusted
    plotter.camera_position = 'iso'
    plotter.show()

if __name__ == "__main__":
    run_nf_analysis()
