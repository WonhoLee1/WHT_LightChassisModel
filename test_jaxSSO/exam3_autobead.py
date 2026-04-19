# -*- coding: utf-8 -*-
"""
[WHT_LightChassisModel] Exam 3: Unified Modal Analysis with Auto Beads
======================================================================
1. Solver: JaxSSO (Shell) & jax-fem (Solid Hexa)
2. Topography: Random symmetric beads (Auto Beads)
3. Boundary Condition: Fixed at 'flange' nodes
4. Export: Industrial formats (.k, .rad, .fem) with ID options
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True) # JAX float64 필수 (고유치 해석 수치 안정성 보장)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_jaxSSO.mesh_utils import generate_shell_tray, generate_solid_hexa_tray, apply_auto_beads
from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_selectors import apply_named_sets_by_recipe
from wht_modeler.wht_entities import WHTSPCEntry
from wht_solver.wht_solver import WHTSolver
from wht_converter.wht_models import WHTMetadata, WHTResultData
from wht_converter.wht_adapters import JaxSSOAdapter, JaxFEMAdapter
from wht_visualizer.wht_visualizer import WHTVisualizer

# --- Material and Geometry Configuration ---
MAT_STEEL = {'E': 210000.0, 'nu': 0.3, 'rho': 7.85e-9, 't': 0.6}
TRAY_GEOM = {'width': 1800.0, 'length': 1200.0, 'height': 30.0, 'thickness': 0.6}
BEAD_PARAMS = {'margin': 30.0, 'target_ratio': 0.5, 'max_depth': 10.0}

def inject_bead_data(wht_data, model, node_db_orig, node_db_new):
    """모든 타임스텝(모드)에 대해 비드 높이와 변위(Warping) 데이터를 주입합니다."""
    dz_list = []
    disp_list = []
    for nid in model.sorted_node_ids():
        z_old = node_db_orig[nid][2]
        z_new = node_db_new[nid][2]
        dz = z_new - z_old
        dz_list.append(dz)
        disp_list.append([0.0, 0.0, dz])
        
    T = max(1, len(wht_data.time_values) if wht_data.time_values is not None else 1)
    dz_array = np.array(dz_list, dtype=np.float32).reshape(1, -1, 1)
    disp_array = np.array(disp_list, dtype=np.float32).reshape(1, -1, 3)
    
    # 모달 해석 시 N개의 모드에 대해 형태를 유지할 수 있도록 반복 복사
    dz_array = np.repeat(dz_array, T, axis=0)
    disp_array = np.repeat(disp_array, T, axis=0)
    
    if getattr(wht_data, 'point_data', None) is None: wht_data.point_data = {}
    wht_data.point_data["Bead_Height"] = dz_array
    wht_data.point_data["Bead_Displacement"] = disp_array
    
    return wht_data

def show_bead_preview(model, node_db_orig, node_db_new):
    print("\n -> [Preview] Launching Bead Height Viewer...", flush=True)
    ir = model.to_wht_result_data()
    ir.time_values = np.array([0.0])
    ir = inject_bead_data(ir, model, node_db_orig, node_db_new)
    
    viz = WHTVisualizer(title="Auto Bead Height Preview", show=True)
    viz.load_results(ir)
    viz.show()

def run_shell_pipeline(num_modes=5, preview=False):
    print("\n>>> [Shell Analysis] JaxSSO Pipeline Start", flush=True)
    
    # 1. Mesh & Beads
    shell_geom = {k: v for k, v in TRAY_GEOM.items() if k != 'thickness'}
    node_db, elem_db = generate_shell_tray(
        **shell_geom, mesh_size_xy=40.0, mesh_size_z=10.0, draft_angle=15.0, flange_width=10.0, origin='center'
    )
    node_db_orig = {nid: np.copy(coords) for nid, coords in node_db.items()}
    node_db = apply_auto_beads(node_db, TRAY_GEOM['width'], TRAY_GEOM['length'], origin='center', **BEAD_PARAMS)
    
    # 2. Build Model (exam2_shell_jaxSSO.py 방식)
    model = WHTMeshModel(name="Shell_Tray_Exam3")
    
    # Add Material & Property
    model.add_material(1, MAT_STEEL['E'], MAT_STEEL['nu'], MAT_STEEL['rho'])
    model.add_property(1, "PSHELL", MAT_STEEL['t'], 1)

    for nid, coords in node_db.items():
        model.add_node(nid, *coords)
        
    for eid, nids in elem_db.items():
        model.add_element(eid, nids, "QUAD4" if len(nids)==4 else "TRIA3", pid=1)

    # 3. Apply Boundary Conditions (Fixed Top Flange)
    print(" -> Applying Boundary Conditions (Fixed Top Flange)...", flush=True)
    fixed_count = 0
    for nid, node in model.nodes.items():
        if abs(node.z - TRAY_GEOM['height']) < 0.1: # Top flange location
            model.spc_conditions.append(WHTSPCEntry(nid, (0, 1, 2, 3, 4, 5)))
            fixed_count += 1
    print(f"    Nodes constrained: {fixed_count}", flush=True)

    if preview:
        # 해석을 수행하지 않고 비드 미리보기만 실행 후 종료
        show_bead_preview(model, node_db_orig, node_db)
        return model, None, None, None

    # 4. Solve
    print(" -> Initializing JaxSSO Solver & Solving Modal...", flush=True)
    solver = WHTSolver(model)
    
    # Sanity check: Total mass
    jm_temp, _, _ = solver._build_jaxsso_model()
    jm_temp.model_ready() # Must call model_ready before accessing ndof
    M_diag = solver._assemble_lumped_mass(jm_temp, jm_temp.ndof, model.sorted_node_ids(), {nid: i for i, nid in enumerate(model.sorted_node_ids())})
    total_m_shell = np.sum(M_diag[::6])
    print(f" -> [Sanity Check] Total Shell Mass from M: {total_m_shell:.6f} tons", flush=True)

    print(f" -> Solving for {num_modes} modes via JaxSSO...", flush=True)
    try:
        res = solver.solve_modal(num_modes=num_modes)
    except Exception as e:
        print(f" -> [Error] JaxSSO solver failed: {e}", flush=True)
        return model, None, None, None

    print("\n[Shell Result] Natural Frequencies (Hz):", flush=True)
    for i, f in enumerate(res.frequencies):
        print(f"  Mode {i+1}: {f:8.2f} Hz", flush=True)
        
    return model, res, node_db_orig, node_db

def run_solid_pipeline(num_modes=5, preview=False):
    print("\n>>> [Solid Analysis] jax-fem Pipeline Start", flush=True)
    
    # 1. Mesh & Beads
    node_db, elem_db = generate_solid_hexa_tray(
        **TRAY_GEOM, mesh_size_xy=40.0, mesh_size_z=10.0, draft_angle=15.0, wall_layers=2, flange_width=10.0, origin='center'
    )
    node_db_orig = {nid: np.copy(coords) for nid, coords in node_db.items()}
    node_db = apply_auto_beads(node_db, TRAY_GEOM['width'], TRAY_GEOM['length'], origin='center', **BEAD_PARAMS)
    
    # 2. Build Model & Sets
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db, is_solid=True, name="Solid_Tray")
    recipe = {
        "set_node-flange": {"type": "box", "z_range": (TRAY_GEOM['height'] - 0.1, TRAY_GEOM['height'] + 0.1)}
    }
    apply_named_sets_by_recipe(model, recipe)
    
    if preview:
        # 해석을 수행하지 않고 비드 미리보기만 실행 후 종료
        show_bead_preview(model, node_db_orig, node_db)
        return model, None, None, None

    # 3. Solver Setup (jax-fem uses its own classes)
    try:
        import jax.numpy as jnp
        from jax_fem.generate_mesh import Mesh
        from jax_fem.problem import Problem
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import diags
    except ImportError as e:
        print(f" -> [Error] jax-fem not installed: {e}", flush=True)
        return model, None, None, None

    points = model.nodes_array()
    nid_to_idx = model.node_id_to_index()
    cells = np.array([[nid_to_idx[n] for n in model.elements[eid].node_ids] 
                     for eid in sorted(model.elements.keys()) if model.elements[eid].type == "HEXA8"])
    
    mesh = Mesh(points, cells)
    E, nu, rho = MAT_STEEL['E'], MAT_STEEL['nu'], MAT_STEEL['rho']
    
    class ModalProblem(Problem):
        def get_tensor_map(self):
            def constitutive_equation(u_grad):
                mu = E / (2.0 * (1.0 + nu)); lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                eps = 0.5 * (u_grad + u_grad.T)
                return lmbda * jnp.trace(eps) * jnp.eye(3) + 2.0 * mu * eps
            return constitutive_equation

    # Boundary: Fixed at flange
    def dirichlet_val(point): return 0.0
    z_top = TRAY_GEOM['height']
    def flange_filter(x): return jnp.isclose(x[2], z_top, atol=0.2)
    
    # Correct structure for jax-fem: [location_fns, vecs, value_fns]
    dirichlet_bc_info = [
        [flange_filter, flange_filter, flange_filter],
        [0, 1, 2],
        [dirichlet_val, dirichlet_val, dirichlet_val]
    ]

    print(" -> Assembling K & M matrices (jax-fem)...", flush=True)
    prob = ModalProblem(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)
    sol_list = prob.unflatten_fn_sol_list(np.zeros(prob.num_total_dofs_all_vars))
    prob.newton_update(sol_list)
    import scipy.sparse as sps
    K = sps.csr_matrix((np.array(prob.V), (prob.I, prob.J)), shape=(prob.num_total_dofs_all_vars, prob.num_total_dofs_all_vars))
    
    # Simple lumped mass for Demo (Total mass distributed)
    import pyvista as pv
    grid = pv.UnstructuredGrid(np.hstack([np.full((cells.shape[0], 1), 8), cells]).flatten(), np.full(cells.shape[0], 12, dtype=np.uint8), points)
    total_m = grid.volume * rho
    # Accurate volume-based lumped mass distribution
    cell_volumes = grid.compute_cell_sizes()["Volume"]
    nodal_mass = np.zeros(len(points))
    for i, cell in enumerate(cells):
        m_cell = cell_volumes[i] * rho
        nodal_mass[cell] += m_cell / 8.0
    
    dof_mass = np.repeat(nodal_mass, 3)
    M = diags([dof_mass], [0], shape=K.shape, dtype=K.dtype).tocsr()
    
    # Safety Check
    if not np.all(np.isfinite(K.data)) or not np.all(np.isfinite(M.data)):
        print(" -> [Error] K or M matrix contains non-finite values (NaN/Inf).", flush=True)
        return model, None, None, None

    print(f" -> Solving for {num_modes} modes via eigsh (Improved Mass Matrix)...", flush=True)
    try:
        vals, vecs = eigsh(K, k=num_modes, M=M, which='LM', sigma=-0.1)
    except Exception as e:
        print(f" -> [Error] Eigenvalue solver failed: {e}", flush=True)
        return model, None, None, None
        
    freqs = np.sqrt(np.abs(vals)) / (2 * np.pi)
    
    print("\n[Solid Result] Natural Frequencies (Hz):", flush=True)
    for i, f in enumerate(sorted(freqs)):
        print(f"  Mode {i+1}: {f:8.2f} Hz", flush=True)
        
    # Format for visualization
    # [WHT] Normalize Mode Shapes to Max Displacement = 1.0 directly like JaxSSO
    # User requested to disable normalization so modes remain mass-normalized natively.
    # for m in range(num_modes):
    #     max_disp = np.max(np.abs(vecs[:, m]))
    #     if max_disp > 1e-12:
    #         vecs[:, m] /= max_disp
            
    res_vecs = vecs.reshape((len(points), 3, num_modes)).transpose(2, 0, 1)
    res_data = {"eigvecs": res_vecs, "eigvals": vals}
    return model, (prob, res_data), node_db_orig, node_db

def export_model(model):
    print("\n--- Export Module ---", flush=True)
    ans = input(" -> Export model to industrial formats? (y/n): ").lower()
    if ans != 'y': return
    
    reorder_ans = input(" -> Reorder IDs sequentially (1..N)? (y/n) [Default: Keep Original]: ").lower()
    reorder = (reorder_ans == 'y')
    
    formats = {'1': ('lsdyna', '.k'), '2': ('radioss', '.rad'), '3': ('optistruct', '.fem')}
    print(" Select formats to export (e.g. 1,2):")
    for k, v in formats.items(): print(f"  {k}. {v[0].upper()}")
    
    choices = input(" Choice: ").split(',')
    for c in choices:
        c = c.strip()
        if c in formats:
            stype, ext = formats[c]
            fname = f"{model.name}_morphed{ext}"
            model.export_to_solver(stype, fname, reorder=reorder)

if __name__ == "__main__":
    # =========================================================================
    # 실행 방법 (Usage Examples):
    # 1. 기본 Shell 해석 (JaxSSO, 5개 모드): 
    #    python exam3_autobead.py
    # 2. Solid 해석 (jax-fem, 10개 모드):
    #    python exam3_autobead.py --mode solid --modes 10
    # 3. 해석 후 모델 파일(K, RAD 등) 내보내기 묻기:
    #    python exam3_autobead.py --export
    # 4. 비드 높이 미리보기 (해석 건너뜀):
    #    python exam3_autobead.py --preview
    # =========================================================================
    parser = argparse.ArgumentParser(description="Unified Modal Analysis with Auto Beads")
    parser.add_argument("--mode", type=str, choices=['shell', 'solid'], default='shell', help="Solver type")
    parser.add_argument("--modes", type=int, default=5, help="Number of modes to calculate")
    parser.add_argument("--export", action="store_true", help="Prompt to export model to industrial formats")
    parser.add_argument("--preview", action="store_true", help="Preview auto bead height and exit before solving")
    args = parser.parse_args()
    
    if args.mode == 'shell':
        model, result, n_orig, n_new = run_shell_pipeline(num_modes=args.modes, preview=args.preview)
        if result:
            if args.export:
                export_model(model)
            meta = WHTMetadata(
                solver_name="JaxSSO", solver_version="0.1.0", 
                analysis_type="modal", coordinate_system="cartesian",
                unit_length="mm", unit_force="N"
            )
            wht_data = result.to_wht_result_data(meta, model)
            wht_data = inject_bead_data(wht_data, model, n_orig, n_new)
            viz = WHTVisualizer(title="Shell Modal Analysis", show=True)
            viz.load_results(wht_data)
            viz.show()
    else:
        model, result, n_orig, n_new = run_solid_pipeline(num_modes=args.modes, preview=args.preview)
        if result:
            if args.export:
                export_model(model)
            prob, res_dict = result
            adapter = JaxFEMAdapter()
            meta = WHTMetadata(
                solver_name="jax-fem", solver_version="0.1.0",
                analysis_type="modal", coordinate_system="cartesian",
                unit_length="mm", unit_force="N"
            )
            wht_data = adapter.convert(prob, res_dict, "modal", meta)
            wht_data = inject_bead_data(wht_data, model, n_orig, n_new)
            viz = WHTVisualizer(title="Solid Modal Analysis", show=True)
            viz.load_results(wht_data)
            viz.show()
