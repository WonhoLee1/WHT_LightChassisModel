# -*- coding: utf-8 -*-
import gmsh
import numpy as np

def generate_shell_tray(width=100.0, length=100.0, height=10.0, mesh_size=5.0):
    """
    Generates a Shell Tray mesh using Gmsh (OpenCASCADE).
    Returns: nodes (Dict: id -> [x,y,z]), elements (Dict: id -> [n1, n2, n3, n4])
    """
    gmsh.initialize()
    gmsh.model.add("ShellTray")
    
    # Bottom Surface
    p1 = gmsh.model.occ.addPoint(0, 0, 0, mesh_size)
    p2 = gmsh.model.occ.addPoint(width, 0, 0, mesh_size)
    p3 = gmsh.model.occ.addPoint(width, length, 0, mesh_size)
    p4 = gmsh.model.occ.addPoint(0, length, 0, mesh_size)
    
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)
    
    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    s1 = gmsh.model.occ.addPlaneSurface([cl])
    
    # Side Walls (Extrude edges in Z)
    # Wall 1 (at Y=0)
    gmsh.model.occ.extrude([(1, l1)], 0, 0, height)
    # Wall 2 (at X=width)
    gmsh.model.occ.extrude([(1, l2)], 0, 0, height)
    # Wall 3 (at Y=length)
    gmsh.model.occ.extrude([(1, l3)], 0, 0, height)
    # Wall 4 (at X=0)
    # [WHT] Revert to Stable Mesh Settings to Fix Distortion
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)             # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 0)  # Disable subdivision to prevent indexing issues
    gmsh.option.setNumber("Mesh.Smoothing", 5)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    
    gmsh.model.occ.synchronize()
    
    # Set mesh size globally
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    
    # Generate 2D Mesh
    gmsh.model.mesh.generate(2)
    
    # Extract Nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = node_coords.reshape(-1, 3)
    # [WHT] JaxSSO expects 0-based contiguous IDs for index-based DOF mapping
    tag_to_idx = {int(tag): i for i, tag in enumerate(node_tags)}
    node_id_to_coords = {i: nodes[i] for i in range(len(node_tags))}
    
    # Extract Elements (Quads and Triangles)
    elem_dict = {}
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        num_v = 3 if etype == 2 else 4
        enodes_reshaped = enodes.reshape(-1, num_v)
        for i, tag in enumerate(etags):
            # Map connectivity to new 0-based IDs
            elem_dict[int(tag)] = [tag_to_idx[int(n)] for n in enodes_reshaped[i]]
            
    gmsh.finalize()
    return node_id_to_coords, elem_dict

def get_nodes_in_box(nodes_dict, x_range=None, y_range=None, z_range=None):
    """
    Returns a list of node IDs within a specified bounding box.
    """
    selected_ids = []
    for nid, coords in nodes_dict.items():
        x, y, z = coords
        in_x = (x_range[0] <= x <= x_range[1]) if x_range else True
        in_y = (y_range[0] <= y <= y_range[1]) if y_range else True
        in_z = (z_range[0] <= z <= z_range[1]) if z_range else True
        if in_x and in_y and in_z:
            selected_ids.append(nid)
    return selected_ids

def apply_fixed_bc(model, node_ids, dofs=(0, 1, 2, 3, 4, 5)):
    """
    Utility to apply fixed BCs to a list of node IDs in a JaxSSO model.
    JaxSSO uses add_support(nodeTag, active_supports=[1,1,1,1,1,1])
    """
    for nid in node_ids:
        # Create support vector (1=fixed, 0=free)
        support_vec = [0, 0, 0, 0, 0, 0]
        for d in dofs:
            support_vec[d] = 1
        model.add_support(nid, support_vec)
