import sys
from pathlib import Path
import numpy as np

# add WHT path
wht_dir = Path("d:/PythonCodeStudy/WHT_LightChassisModel")
if str(wht_dir) not in sys.path:
    sys.path.append(str(wht_dir))

from wht_modeler.wht_mesh_model import WHTMeshModel, WHTNode, WHTElement, WHTProperty, WHTMaterial
from wht_solver.wht_solver import WHTSolver

# create wht model
model = WHTMeshModel()
mat = WHTMaterial(1)
mat.name = "STEEL"
mat.E = 210000.0
mat.nu = 0.3
mat.rho = 7.85e-9
model.materials[1] = mat

prop = WHTProperty(1, "PSHELL")
prop.name = "PSHELL1"
prop.mid = 1
prop.t = 2.0
model.properties[1] = prop

# nodes
nodes = {
    1: (0.0,   0.0,   0.0),
    2: (50.0,  0.0,   0.0),
    3: (100.0, 0.0,   0.0),
    4: (0.0,   25.0,  0.0),
    5: (50.0,  25.0,  0.0),
    6: (100.0, 25.0,  0.0),
    7: (0.0,   50.0,  0.0),
    8: (50.0,  50.0,  0.0),
    9: (100.0, 50.0,  0.0),
}
for nid, coords in nodes.items():
    model.nodes[nid] = WHTNode(nid, *coords)

# elements
elements = [
    (1, "QUAD4", [1, 2, 5, 4], 1),
    (2, "QUAD4", [2, 3, 6, 5], 1),
    (3, "QUAD4", [4, 5, 8, 7], 1),
    (4, "QUAD4", [5, 6, 9, 8], 1),
]
for eid, etype, nids, pid in elements:
    elem = WHTElement(eid, etype, nids)
    elem.pid = pid
    model.elements[eid] = elem

# create solver
solver = WHTSolver(model)

print(f"Elements: {len(model.elements)}")
print(f"Properties: {model.properties.keys()}")
print(f"Materials: {model.materials.keys()}")
for eid, elem in model.elements.items():
    print(f"Elem {eid}: type={elem.type}, pid={elem.pid}")
for pid, prop in model.properties.items():
    print(f"Prop {pid}: t={getattr(prop, 't', 'N/A')}, mid={getattr(prop, 'mid', 'N/A')}")


# Set Boundary Conditions for Modal Analysis (Free-Free)
# from wht_modeler.wht_entities import WHTSPCEntry
# for nid in [1, 4, 7]:
#     bc = WHTSPCEntry(nid, [0,1,2,3,4,5], 0.0)
#     model.spc_conditions.append(bc)

print("--- WHT Solver Modal Analysis ---")
# Build diagnostic info
jm, sorted_nids, nid_to_idx = solver._build_jaxsso_model()
K_scipy = solver._assemble_K_scipy(jm, sorted_nids, nid_to_idx, stabilize=False)
M_all = solver._assemble_lumped_mass(jm, jm.ndof, sorted_nids, nid_to_idx)
print(f"Diagnostics: Max K_diag = {np.max(K_scipy.diagonal()):.4e}, Max M_diag = {np.max(M_all):.4e}")

res = solver.solve_modal(num_modes=10, method='dense', exclude_rigid_body=False)
print("WHT Frequencies:")
for i, f in enumerate(res.frequencies):
    print(f"  Mode {i+1}: {f:.3f} Hz")

print("\n--- Saving Modal Report in CalculiX format ---")
test_report_path = "d:/PythonCodeStudy/WHT_LightChassisModel/scratch/wht_modal_report.dat"
res.save_modal_report(test_report_path)

print("\n--- CCX Modal Analysis ---")
import sys
if "D:/PythonCodeStudy/AutoCalculix" not in sys.path:
    sys.path.append("D:/PythonCodeStudy/AutoCalculix")
from src.autocalculix_api import run_calculix_analysis
import os
scratch_dir = Path(os.getcwd()) / "scratch" / "test_workspace"
scratch_dir.mkdir(parents=True, exist_ok=True)
bcs_ccx = []
print(f"Elements: {len(model.elements)}")
print(f"Properties: {model.properties.keys()}")
print(f"Materials: {model.materials.keys()}")

# format nodes, elements, properties for CCX
nodes_ccx = {n.nid: (n.x, n.y, n.z) for n in model.nodes.values()}
elements_ccx = [(e.eid, e.type, e.node_ids, e.pid) for e in model.elements.values()]
properties_ccx = {1: (2.0, 210000.0, 0.3, 7.85e-9)}

ccx_res = run_calculix_analysis(
    nodes=nodes_ccx,
    elements=elements_ccx,
    properties=properties_ccx,
    analysis_type="modal",
    analysis_config={"job_name": "test_modal_bc", "num_modes": 10},
    bcs=bcs_ccx,
    workspace_dir=str(scratch_dir)
)
print("CCX Frequencies:")
for f in ccx_res.get("frequencies", []):
    print(f"  Mode {f['mode']:d}: {f['hz']:.3f} Hz")
