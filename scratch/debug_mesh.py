# -*- coding: utf-8 -*-
from mesh_utils import generate_shell_tray
import gmsh

def test_mesh():
    complex_rim = [(10.0, 0.0), (5.0, 5.0), (10.0, 0.0)]
    custom_flanges = (True, False, True, False)
    
    print("Testing QUAD4 mesh generation with stepped transition...")
    try:
        nodes, elems = generate_shell_tray(
            width=1800.0, length=1200.0, height=35.0, low_height=10.0,
            mesh_size_xy=100.0, mesh_size_z=10.0, draft_angle=5.0,
            flange_segments=complex_rim, flanges=custom_flanges,
            mesh_type='quad4'
        )
        print(f"Success! Nodes: {len(nodes)}, Elements: {len(elems)}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mesh()
