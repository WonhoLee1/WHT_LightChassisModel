import os
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Add current dir to path
sys.path.insert(0, str(Path.cwd()))

from test_jaxSSO.exam2_shell_jaxSSO import PipelineConfig, generate_mesh, build_model
from wht_solver.wht_solver import WHTSolver
from wht_solver.wht_tria3_element import tri_area

def run_diag():
    cfg = PipelineConfig(mesh_type='tria3_free')
    print(f"Generating mesh for {cfg.mesh_type}...")
    node_db, elem_db = generate_mesh(cfg)
    model = build_model(node_db, elem_db, cfg)
    print(f"Nodes: {len(model.nodes)}, Elements: {len(model.elements)}")

    # 1. Connectivity Check
    # Map nids to 0-indexed
    snids = model.sorted_node_ids()
    nid_to_i = {nid: i for i, nid in enumerate(snids)}
    n_nodes = len(snids)
    
    rows, cols = [], []
    for eid, elem in model.elements.items():
        nids = elem.node_ids
        for i in range(len(nids)):
            for j in range(i+1, len(nids)):
                r, c = nid_to_i[nids[i]], nid_to_i[nids[j]]
                rows.extend([r, c])
                cols.extend([c, r])
    
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(adj)
    print(f"Connected components: {n_comp}")
    if n_comp > 1:
        comp_sizes = [np.sum(labels == i) for i in range(n_comp)]
        print(f"Component sizes: {comp_sizes}")
        
    # 2. Area Check
    zero_areas = []
    for eid, elem in model.elements.items():
        if elem.type == 'TRIA3':
            c = [np.array(model.nodes[nid].coords()) for nid in elem.node_ids]
            a = tri_area(*c)
            if a < 1e-10:
                zero_areas.append((eid, a))
    print(f"Zero-area elements: {len(zero_areas)}")

    # 3. Solver check
    solver = WHTSolver(model)
    jm, sn, n2i = solver._build_jaxsso_model()
    print(f"JaxSSO model built. NDOF: {jm.ndof}")
    
    # 4. Mass check
    M_diag = solver._assemble_lumped_mass(jm, jm.ndof, sn, n2i)
    zero_mass_dofs = np.where(M_diag < 1e-15)[0]
    print(f"Zero mass DOFs: {len(zero_mass_dofs)}")
    
    # 5. Stiffness check
    K = solver._assemble_K_scipy(model, jm, n2i)
    k_diag = K.diagonal()
    zero_k_dofs = np.where(np.abs(k_diag) < 1e-15)[0]
    print(f"Zero stiffness DOFs (before boundary/AUTOSPC): {len(zero_k_dofs)}")

if __name__ == "__main__":
    run_diag()
