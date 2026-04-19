# -*- coding: utf-8 -*-
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_jaxSSO.mesh_utils import generate_solid_hexa_tray

def diagnose_mesh(thickness, wall_layers, mesh_size_xy):
    print(f"\n>>> Diagnosing Mesh: Thickness={thickness}, Wall Layers={wall_layers}, Mesh Size XY={mesh_size_xy}")
    node_db, elem_db = generate_solid_hexa_tray(
        width=1800.0, length=1200.0, height=30.0, thickness=thickness,   
        mesh_size_xy=mesh_size_xy, mesh_size_z=10.0, draft_angle=5.0, wall_layers=wall_layers
    )
    
    # Calculate Aspect Ratio of wall elements
    # Let's find an element that is part of the wall (peripheral)
    aspect_ratios = []
    for eid, nids in elem_db.items():
        pts = [node_db[n] for n in nids]
        # Rough calc: Max edge / Min edge
        edges = []
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                d = np.linalg.norm(pts[i] - pts[j])
                edges.append(d)
        ar = max(edges) / min(edges)
        aspect_ratios.append(ar)
        
    print(f" -> Max Aspect Ratio: {max(aspect_ratios):.2f}")
    print(f" -> Min Aspect Ratio: {min(aspect_ratios):.2f}")
    print(f" -> Avg Aspect Ratio: {np.mean(aspect_ratios):.2f}")
    
    if max(aspect_ratios) > 100:
        print(" [!] WARNING: Extremely high aspect ratio detected. Results will be numerically unstable.")

if __name__ == "__main__":
    diagnose_mesh(thickness=10.6, wall_layers=3, mesh_size_xy=50.0)
    diagnose_mesh(thickness=0.6, wall_layers=3, mesh_size_xy=50.0)
