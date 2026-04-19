# -*- coding: utf-8 -*-
"""
[WHT_LightChassisModel] Exam 2: 3D Solid Hexa Modal Analysis using jax-fem
==========================================================================
1. Geometry: 3D Hexahedral Shell Tray (Gmsh)
2. Data Model: WHTMeshModel (is_solid=True)
3. Solver: jax-fem (Modal)
4. Pipeline: JaxFEMAdapter -> VTKHDFExporter (ParaView)
"""

import os
import sys
import math
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pyvista as pv

# jax-fem components
from jax_fem.problem import Problem
from jax_fem.generate_mesh import Mesh

# Project components
# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_jaxSSO.mesh_utils import generate_solid_hexa_tray
from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_converter.wht_adapters import JaxFEMAdapter
from wht_converter.wht_models import WHTMetadata
from wht_converter.wht_exporters import VTKHDFExporter, VTUPVDExporter
from wht_visualizer.wht_visualizer import WHTVisualizer

def run_exam2():
    print("\n" + "="*60)
    print(" [Exam 2] Solid Modal Analysis Pipeline (jax-fem)")
    print("="*60 + "\n")
    
    # 1. Geometry conditions
    width, length, height = 1800.0, 1200.0, 10.0
    thickness = 0.6
    mesh_size_xy = 50.0 
    draft_angle = 15.0
    flange_width = 80.0
    mesh_size_z = thickness # ensures exactly wall_layers elements through thickness
    
    # 2. Generate Solid Hexa Mesh (Order 1 = HEX8)
    print(f" -> Generating Refined Solid Hexa Tray Mesh (T={thickness}, Order=1)...", flush=True)
    node_db, elem_db = generate_solid_hexa_tray(
        width=width, length=length, height=height, thickness=thickness,   
        mesh_size_xy=mesh_size_xy, mesh_size_z=mesh_size_z, draft_angle=draft_angle, 
        wall_layers=3, flange_width=flange_width, origin='center', order=1
    )
    
    from wht_converter.wht_utils import node_dict_to_array, remap_connectivity
    points, id_map = node_dict_to_array(node_db)
    cells = remap_connectivity(elem_db, id_map)
    
    print(f"    Nodes: {len(points)}, Elements: {len(cells)}")
    
    from jax_fem.generate_mesh import Mesh
    from jax_fem.problem import Problem
    import scipy.sparse as sps
    import numpy as onp # Fix for onp.array calls
    
    mesh = Mesh(cells=cells, points=points, ele_type='HEX8')
    
    # 3. jax-fem Problem Setup (HEX8) purely for shape functions
    E, nu, rho = 210.0e3, 0.3, 7.85e-9 # consistent units (MPa, ton, mm)

    print(" -> Building jax-fem Problem (HEX8) for shape functions...", flush=True)
    problem = Problem(mesh, vec=3, dim=3, ele_type='HEX8')
    
    # 4. Matrix Assembly (Custom JAX Integration)
    print(f" -> Assembling Stiffness Matrix (Custom JAX Integration)...", flush=True)
    
    # Construct 3D isotropic elasticity D matrix (6x6)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = jnp.zeros((6, 6))
    D = D.at[0:3, 0:3].set(lmbda)
    for i in range(3): 
        D = D.at[i, i].set(lmbda + 2.0*mu)
    for i in range(3, 6): 
        D = D.at[i, i].set(mu)
    
    @jax.jit
    def get_element_stiffness(shape_grad_q, jxw_q):
        # shape_grad_q: (num_quads, 8, 3)
        # jxw_q: (num_quads,)
        
        def compute_quad_stiffness(sg, jxw):
            # sg: (8, 3)
            # D: (6, 6) global
            
            # Construct B matrix (6, 24)
            B = jnp.zeros((6, 24))
            for i in range(8):
                idx = i * 3
                # Normal strains
                B = B.at[0, idx  ].set(sg[i, 0])
                B = B.at[1, idx+1].set(sg[i, 1])
                B = B.at[2, idx+2].set(sg[i, 2])
                # Shear strains (engineering)
                B = B.at[3, idx+1].set(sg[i, 2])
                B = B.at[3, idx+2].set(sg[i, 1])
                
                B = B.at[4, idx  ].set(sg[i, 2])
                B = B.at[4, idx+2].set(sg[i, 0])
                
                B = B.at[5, idx  ].set(sg[i, 1])
                B = B.at[5, idx+1].set(sg[i, 0])
            
            return B.T @ D @ B * jxw
            
        ke_q = jax.vmap(compute_quad_stiffness)(shape_grad_q, jxw_q) # (num_quads, 24, 24)
        return jnp.sum(ke_q, axis=0) # (24, 24)
    
    # Extract element properties
    # problem.shape_grads: (num_cells, num_quads, num_nodes, dim)
    # problem.JxW: (num_cells, 1, num_quads). Remove the middle axis.
    sg_all = problem.shape_grads
    jxw_all = problem.JxW[:, 0, :]
    
    print(" -> Computing all element stiffness matrices via vmap...", flush=True)
    ke_all = jax.vmap(get_element_stiffness)(sg_all, jxw_all) # (num_cells, 24, 24)
    
    # Construct CSR Matrix
    V = onp.array(ke_all.reshape(-1))
    I, J = problem.I, problem.J
    print(f" -> V.shape: {V.shape}, I.shape: {I.shape}, J.shape: {J.shape}")

    K_full = sps.csr_matrix((V, (I, J)), 
                            shape=(problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars))

    def dirichlet_bc_info_fn(points):
        # Fix Top Flange (rim support at Z=10)
        # points is a single point (3,) due to jax.vmap
        return jnp.abs(points[2] - 10.0) < 1e-3

    # location_fns, vecs (component Indices), value_fns
    # 5. Manual Boundary Conditions
    print(f" -> Applying Dirichlet BCs manually (Fixed Top Flange at Z=10)...", flush=True)
    from mesh_utils import get_nodes_in_box
    # The rim support is at top face (Z=10.0)
    top_node_ids = get_nodes_in_box(node_db, z_range=(9.99, 10.01))
    
    fixed_dofs = []
    for nid in top_node_ids:
        # id_map converts GMSH tags to 0-indexed points array index
        local_idx = id_map[nid]
        # Fix all 3 translations
        fixed_dofs.extend([local_idx*3, local_idx*3+1, local_idx*3+2])
    
    fixed_dofs = np.array(fixed_dofs, dtype=int)
    all_dofs = np.arange(K_full.shape[0])
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
    
    print(f"    Fixed Nodes: {len(top_node_ids)}, Fixed DOFs: {len(fixed_dofs)}")
    print(f"    System Size: Total={K_full.shape[0]}, Free={len(free_dofs)}")

    # 6. Mass Matrix (Lumped - Accurate Volume based)
    print(f" -> Calculating Lumped Mass Matrix (Rho={rho})...", flush=True)
    import pyvista as pv
    grid = pv.UnstructuredGrid(
        onp.hstack([onp.full((cells.shape[0], 1), 8), cells]).flatten(), 
        onp.full(cells.shape[0], 12, dtype=onp.uint8), 
        points
    )
    cell_volumes = grid.compute_cell_sizes()["Volume"]
    
    nodal_mass = onp.zeros(len(points))
    for i, cell in enumerate(cells):
        m_cell = cell_volumes[i] * rho
        nodal_mass[cell] += m_cell / 8.0
        
    dof_mass = onp.repeat(nodal_mass, 3)
    
    # 7. Condensation and Eigen-Solve
    M_free = sps.diags(dof_mass[free_dofs]).tocsc()
    K_free = K_full[free_dofs, :][:, free_dofs].tocsc()
    
    print(f" -> [Sanity Check] Total Mass: {onp.sum(nodal_mass):.6f} tons")
    
    num_modes = 12
    print(f" -> Solving for {num_modes} Modes (Shift-Invert, sigma=0.01)...", flush=True)
    from scipy.sparse.linalg import eigsh
    try:
        vals, vecs_free = eigsh(K_free, k=num_modes, M=M_free, which='LM', sigma=0.01)
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs_free = vecs_free[:, idx]
    except Exception as e:
        print(f" -> [Error] Eigensolver failed: {e}")
        return

    freqs = np.sqrt(np.abs(vals)) / (2.0 * np.pi)
    
    # Sort frequencies ascending
    sorted_idx = np.argsort(freqs)
    freqs = freqs[sorted_idx]
    
    print("\n[Final Result] Solid HEX8 Natural Frequencies (Hz):")
    for i, f in enumerate(freqs):
        print(f"  Mode {i+1:2d}: {f:8.2f} Hz", flush=True)

    # 8. Reconstruct and Visualize
    vecs_full = np.zeros((K_full.shape[0], num_modes))
    vecs_full[free_dofs, :] = vecs_free
    
    # [WHT] Normalize Mode Shapes to Max Displacement = 1.0 directly like JaxSSO
    # User requested to disable normalization so modes remain mass-normalized natively.
    # for m in range(num_modes):
    #     max_disp = np.max(np.abs(vecs_full[:, m].reshape(-1, 3)))
    #     if max_disp > 1e-12:
    #         vecs_full[:, m] /= max_disp
            
    # Export results
    vecs_reshaped = vecs_full.reshape((len(points), 3, num_modes)).transpose(2, 0, 1)
    results = {"eigvecs": vecs_reshaped, "eigvals": vals}
    
    adapter = JaxFEMAdapter()
    meta = WHTMetadata(solver_name="jax-fem", solver_version="HEX8-Refined", analysis_type="modal", 
                        coordinate_system="cartesian", unit_length="mm", unit_force="N")
    wht_data = adapter.convert(problem, results, "modal", meta)
    
    # 9. Clear result data from problem to free memory
    del problem.physical_quad_points, problem.shape_grads, problem.JxW
    
    from datetime import datetime
    stamp = datetime.now().strftime("D%Y%m%d-%H%M%S")
    out_dir = Path(__file__).resolve().parent.parent / "results" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = str(out_dir/"solid_hexa_modal.hdf")
    VTKHDFExporter().export(wht_data, h5_path)
    print(f" -> Results saved to: {h5_path}")

    viz = WHTVisualizer(title="Solid Hexa Modal Analysis (HEX8 Refined)", show=True)
    viz.load_results(wht_data, color="black", label="Solid Hexa Modal Shape")
    viz.plotter.show_axes()
    viz.plotter.camera_position = [(3000, 2000, 2000), (0, 0, 0), (0, 0, 1)]
    if hasattr(viz.plotter, 'app'): viz.plotter.app.exec_()

if __name__ == "__main__":
    run_exam2()

if __name__ == "__main__":
    run_exam2()