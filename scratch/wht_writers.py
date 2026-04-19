"""
wht_writers.py
==============
WHT Universal FEM Framework — Writer Layer

Exports WHTMeshModel to solver-specific files.
"""

from .wht_mesh_model import WHTMeshModel


class LSDYNAWriter:
    """
    Writer for LS-DYNA keyword files (.k).
    """
    def write(self, model: WHTMeshModel, file_path: str):
        content = ["*KEYWORD\n"]
        
        # --- Nodes ---
        content.append("*NODE\n")
        # Format: I8, 3E16.0
        for nid, coords in model.nodes.items():
            content.append(f"{nid:>8}{coords[0]:>16.8e}{coords[1]:>16.8e}{coords[2]:>16.8e}\n")
            
        # --- Elements ---
        content.append("*ELEMENT_SHELL\n")
        # Format: 2I8, 4I8
        for eid, node_ids in model.elements.items():
            # Default Part ID = 1 if not specified
            content.append(f"{eid:>8}{1:>8}")
            for nid in node_ids:
                content.append(f"{nid:>8}")
            # Pad with zeros if fewer than 4 nodes (Triangles)
            if len(node_ids) < 4:
                for _ in range(4 - len(node_ids)):
                    content.append(f"{0:>8}")
            content.append("\n")
            
        # --- Node Sets ---
        for sid, nset in model.node_sets.items():
            content.append("*SET_NODE_LIST\n")
            content.append(f"{sid:>10}\n")
            # 8 nodes per line
            for i, nid in enumerate(nset.node_ids):
                content.append(f"{nid:>10}")
                if (i + 1) % 8 == 0:
                    content.append("\n")
            if len(nset.node_ids) % 8 != 0:
                content.append("\n")
        
        content.append("*END\n")
        
        with open(file_path, 'w') as f:
            f.writelines(content)
