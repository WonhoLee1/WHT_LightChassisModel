import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from wht_modeler.wht_mesh_model import WHTMeshModel
from test_jaxSSO.mesh_utils import generate_shell_tray
from wht_modeler.wht_dynamic_utils import find_corner_nodes
from wht_topo.loads import StochasticLoadManager

def setup_model_and_lcs():
    WIDTH = 1241.0
    LENGTH = 1641.0
    MESH_SIZE = 30.0
    MESH_Z = 20.0
    
    node_db, elem_db = generate_shell_tray(WIDTH, LENGTH, MESH_Z, MESH_SIZE)
    model = WHTMeshModel()
    
    for nid, coords in node_db.items():
        model.add_node(nid, coords[0], coords[1], coords[2])
    for eid, nids in elem_db.items():
        if len(nids) == 4:
            model.add_element(eid, nids, "QUAD4")
        elif len(nids) == 3:
            model.add_element(eid, nids, "TRIA3")
            
    model.add_material(1, E=200000.0, nu=0.29, rho=7.6e-9)
    model.add_property(1, "PSHELL", t=0.5, mid=1)
    for eid in model.elements:
        model.elements[eid].pid = 1
        
    weights = {"bending": 1.0, "twisting": 1.0, "lifting": 1.0}
    loads_val = {"bending": -500.0, "twisting": -3000.0, "lifting": 2000.0}
    manager = StochasticLoadManager(model)
    lcs = manager.get_load_cases(mesh_size_z=MESH_Z, weights=weights, loads=loads_val)
    return model, lcs
