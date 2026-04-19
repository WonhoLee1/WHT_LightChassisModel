# -*- coding: utf-8 -*-
"""
Check 3D Solid Hexa Tray Mesh Generation
========================================
Generates a structured Hexahedral tray mesh and visualizes it using PyVista.
Includes a clipping plane to verify the hollow interior and wall thickness.
"""

import numpy as np
import pyvista as pv
from mesh_utils import generate_solid_hexa_tray

def check_hexa_mesh():
    print(" -> Generating 3D Solid Hexa Tray Mesh...")
    # 벽면 두께 방향으로 3개의 레이어(wall_layers=3)와 5도 경사(draft_angle=5.0) 적용
    node_db, elem_db = generate_solid_hexa_tray(
        width=1000.0, length=1000.0, height=100.0, thickness=30.0, 
        mesh_size_xy=50.0, mesh_size_z=10.0, draft_angle=5.0, wall_layers=3
    )
    
    print(f" -> Mesh Generated: {len(node_db)} nodes, {len(elem_db)} HEXA8 elements")
    
    # 노드 딕셔너리를 NumPy 배열로 변환
    sorted_nids = sorted(node_db.keys())
    nodes_array = np.array([node_db[nid] for nid in sorted_nids])
    
    # PyVista 셀(연결) 정보 구성
    cells = []
    cell_types = []
    
    for eid, nids in elem_db.items():
        if len(nids) == 8:
            cells.append(8)            # 해당 요소의 노드 개수
            cells.extend(nids)         # 0-based node indices
            cell_types.append(12)      # 12 = VTK_HEXAHEDRON
            
    # UnstructuredGrid 객체 생성
    grid = pv.UnstructuredGrid(cells, cell_types, nodes_array)
    
    # 시각화 설정
    plotter = pv.Plotter(title="3D Solid Hexa Tray Mesh Check (Draft: 5°, Layers: 3)")
    plotter.set_background("black")
    
    # 전체 메쉬의 윤곽선(Edges) 표시 (반투명하게)
    plotter.add_mesh(grid.extract_all_edges(), color='white', opacity=0.1)
    
    # X=500 평면을 기준으로 절반을 잘라내어(Clip) 내부 구조 확인
    clipped_grid = grid.clip(normal='x', origin=(500.0, 0.0, 0.0))
    plotter.add_mesh(clipped_grid, show_edges=True, edge_color='black', color='lightblue', smooth_shading=False)
    
    plotter.add_axes(line_width=3, color='white')
    plotter.camera_position = 'iso'
    print(" -> Opening PyVista Window...")
    plotter.show()

def check_hexa_fem():
    """
    Performs Natural Frequency (Modal) analysis on the Hexahedral tray using jax-fem.
    """
    print("\n--- [WHT] jax-fem Modal Analysis Test ---")
    print(" -> Generating 3D Solid Hexa Tray Mesh...")
    node_db, elem_db = generate_solid_hexa_tray(
        width=1000.0, length=1000.0, height=100.0, thickness=30.0, 
        mesh_size_xy=50.0, mesh_size_z=10.0, draft_angle=5.0, wall_layers=3
    )
    
    sorted_nids = sorted(node_db.keys())
    points = np.array([node_db[nid] for nid in sorted_nids])
    
    cells = []
    for eid, nids in elem_db.items():
        if len(nids) == 8:
            cells.append(nids)
    cells = np.array(cells)
    
    print(f" -> Mesh Extracted: {points.shape[0]} nodes, {cells.shape[0]} HEXA8 elements")

    try:
        import jax.numpy as jnp
        from jax_fem.generate_mesh import Mesh
        from jax_fem.problem import Problem
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import diags
    except ImportError as e:
        print(f"[!] ImportError encountered: {e}")
        print("    Please install it to run this test (pip install jax-fem).")
        return

    # 1. Create jax-fem mesh
    mesh = Mesh(points, cells)

    # 2. Define Material & Constitutive Law
    E = 1000.0
    nu = 0.4
    rho = 1.0e-9

    class HexaModalProblem(Problem):
        def get_tensor_map(self):
            def constitutive_equation(u_grad):
                mu = E / (2.0 * (1.0 + nu))
                lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                epsilon = 0.5 * (u_grad + u_grad.T)
                return lmbda * jnp.trace(epsilon) * jnp.eye(3) + 2.0 * mu * epsilon
            return constitutive_equation

        def get_mass_map(self):
            def mass_equation(u, x):
                return rho * u
            return mass_equation

    # 3. Boundary Conditions (Free-Free Analysis)
    dirichlet_bc_info = [[], [], []]

    print(" -> Initializing jax-fem HexaModalProblem (Free-Free, Assembling Matrices)...")
    try:
        problem = HexaModalProblem(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
        
        # In jax-fem, tangent stiffness matrix V is populated via newton_update
        dofs_zero = np.zeros(problem.num_total_dofs_all_vars)
        sol_list = problem.unflatten_fn_sol_list(dofs_zero)
        problem.newton_update(sol_list)
        
        # Assemble scipy CSR matrix from problem.V
        K = diags([0], [0], shape=(problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars)).tocsr()
        import scipy.sparse as sps
        K = sps.csr_matrix((np.array(problem.V), (problem.I, problem.J)), 
                           shape=(problem.num_total_dofs_all_vars, problem.num_total_dofs_all_vars))
        
        # 4. Extract or Approximate Mass Matrix (M)
        if hasattr(problem, 'M'):
            M = problem.M
        else:
            print(" -> [Warning] Exact Mass Matrix not directly exposed. Using exact PyVista volume for Lumped Mass...")
            ndof = K.shape[0]
            
            # Calculate exact volume from the mesh using PyVista
            pv_cells_vol = np.hstack([np.full((cells.shape[0], 1), 8), cells]).flatten()
            pv_types_vol = np.full(cells.shape[0], 12, dtype=np.uint8)
            grid_vol = pv.UnstructuredGrid(pv_cells_vol, pv_types_vol, points)
            total_vol = grid_vol.volume
            
            print(f" -> Exact Mesh Volume calculated via PyVista: {total_vol:.2f} mm^3")
            mass_per_dof = (total_vol * rho) / ndof
            print(f" -> Applied Total Mass: {total_vol * rho:.4e} tons (Distributed to {ndof} DOFs)")
            M = diags([np.full(ndof, mass_per_dof, dtype=K.dtype)], [0], shape=(ndof, ndof)).tocsr()

        # 5. Solve Eigenvalue Problem (Free-Free requires Shift-Invert for singular K)
        num_modes = 16  # 6 rigid body modes + 10 elastic modes
        print(f" -> Solving for {num_modes} Natural Frequencies (Shift-Invert)...")
        vals, vecs = eigsh(K, k=num_modes, M=M, which='LM', sigma=-0.1)
        
        # Sort eigenvalues to ensure ascending order
        sorted_indices = np.argsort(vals)
        vals = vals[sorted_indices]
        vecs = vecs[:, sorted_indices]  # 벡터도 동일한 순서로 정렬!
        
        freqs = np.sqrt(np.maximum(vals, 0)) / (2 * np.pi)
        
        print("\n[RESULT] Natural Frequencies (Hz) via jax-fem (Flexible Modes Only):")
        mode_idx = 1
        first_elastic_idx = -1
        for i, f in enumerate(freqs):
            if f > 0.1:  # Filter out rigid body modes (~0 Hz)
                print(f"  Mode {mode_idx:02d}: {f:8.2f} Hz")
                if first_elastic_idx == -1:
                    first_elastic_idx = i
                mode_idx += 1
                
        # 6. Visualize the first elastic mode
        if first_elastic_idx != -1:
            print(f"\n -> Visualizing First Elastic Mode (Mode 1: {freqs[first_elastic_idx]:.2f} Hz)...")
            
            # 1D 모드 형상 벡터를 3D 노드 변위로 리쉐이프
            mode_shape_flat = vecs[:, first_elastic_idx]
            mode_shape_3d = mode_shape_flat.reshape(-1, 3)
            
            # 시각화를 위해 변위 정규화
            max_disp = np.max(np.linalg.norm(mode_shape_3d, axis=1))
            norm_disp = mode_shape_3d / (max_disp + 1e-12)
            
            # PyVista UnstructuredGrid 생성용 데이터 조립
            pv_cells = []
            pv_cell_types = []
            for nids in cells:
                pv_cells.append(8)
                pv_cells.extend(nids)
                pv_cell_types.append(12)  # VTK_HEXAHEDRON
                
            grid = pv.UnstructuredGrid(pv_cells, pv_cell_types, points)
            grid.point_data["Mode_Vector"] = norm_disp
            grid.point_data["Disp_Magnitude"] = np.linalg.norm(mode_shape_3d, axis=1)
            
            # 변위 벡터를 기준으로 메쉬를 크게 뒤틀기(Warp)
            warped = grid.warp_by_vector("Mode_Vector", factor=150.0) 
            
            plotter = pv.Plotter(title=f"jax-fem Modal: First Elastic Mode ({freqs[first_elastic_idx]:.1f} Hz)")
            plotter.set_background("black")
            
            # 원래 형상(투명한 와이어프레임)과 변형된 형상 함께 표시
            plotter.add_mesh(grid, style='wireframe', color='grey', opacity=0.1, label='Undeformed')
            plotter.add_mesh(warped, scalars="Disp_Magnitude", cmap="jet", 
                             show_edges=True, edge_color="grey", line_width=1,
                             smooth_shading=False)
            
            plotter.add_scalar_bar(title="Relative Disp", color="white")
            plotter.add_axes(line_width=3, color='white')
            plotter.camera_position = 'iso'
            plotter.show()

    except Exception as e:
        print(f"\n[!] Error during jax-fem execution: {e}")

if __name__ == "__main__":
    # check_hexa_mesh() # PyVista 시각화 실행 시 활성화
    check_hexa_fem()    # jax-fem 모달 해석 실행