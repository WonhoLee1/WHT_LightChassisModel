# -*- coding: utf-8 -*-
"""
Exporter Verification Script
============================
Tests the card formats for LS-DYNA, Radioss, and OptiStruct.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_jaxSSO.mesh_utils import generate_shell_tray
from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_selectors import apply_named_sets_by_recipe

def test_export_workflow():
    print(">>> Testing Industrial Exporters...")
    
    # 1. Create a simple model
    node_db, elem_db = generate_shell_tray(width=100, length=100, height=10, mesh_size_xy=50, mesh_size_z=10, origin='center')
    model = WHTMeshModel.from_node_elem_db(node_db, elem_db, name="Test_Model")
    
    # 2. Apply a set
    recipe = {"set_node-test": {"type": "box", "z_range": (-1, 1)}}
    apply_named_sets_by_recipe(model, recipe)
    
    # 3. Export to all
    out_dir = Path("export_test")
    out_dir.mkdir(exist_ok=True)
    
    solvers = ['lsdyna', 'radioss', 'optistruct']
    exts = ['.k', '.rad', '.fem']
    
    for s, e in zip(solvers, exts):
        p = out_dir / f"test_export{e}"
        print(f" -> Exporting to {s.upper()}...")
        model.export_to_solver(s, str(p), reorder=True) # Test reordering too
        
        # Basic validation: Check if file is not empty and contains keywords
        content = p.read_text(encoding='utf-8')
        if s == 'lsdyna' and "*NODE" in content and "*SET_NODE_LIST" in content:
            print(f"    [OK] LS-DYNA card format looks valid.")
        elif s == 'radioss' and "/NODE" in content and "/GRNOD" in content:
            print(f"    [OK] Radioss card format looks valid.")
        elif s == 'optistruct' and "GRID" in content and "SET" in content:
            print(f"    [OK] OptiStruct card format looks valid.")
        else:
            print(f"    [FAIL] {s.upper()} file check failed.")

if __name__ == "__main__":
    test_export_workflow()
