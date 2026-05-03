"""
wht_morphing.py
===============
WHT Morphing Engine.
Morphs a base mesh (Mesh A) to a target geometry (Mesh B) using ray tracing.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import pyvista as pv

from .wht_mesh_model import WHTMeshModel
from .io import LSDYNAReader, OptistructReader, RadiossReader

class WHTMorpher:
    """
    Morphs a WHTMeshModel (Base) onto a Target Mesh using normal ray tracing.
    """
    def __init__(self, base_model: WHTMeshModel):
        self.base_model = base_model
        self.target_pv: Optional[pv.PolyData] = None
        self.target_model: Optional[WHTMeshModel] = None

    def load_target_mesh(self, filepath: str) -> None:
        """Loads Mesh B and converts it to PyVista PolyData for ray tracing."""
        ext = Path(filepath).suffix.lower()
        if ext in ['.k', '.key']:
            reader = LSDYNAReader()
        elif ext in ['.fem', '.bdf']:
            reader = OptistructReader()
        elif ext in ['.rad']:
            reader = RadiossReader()
        else:
            raise ValueError(f"Unsupported target mesh extension: {ext}")
            
        self.target_model = reader.read(filepath)
        self.target_pv = self._to_pyvista(self.target_model)

    def align_meshes(self, z_offset: Optional[float] = None, align_centers: bool = True) -> None:
        """
        Aligns the target mesh to the base mesh.
        If z_offset is None, matches Z-min.
        If align_centers is True, matches X,Y centers.
        Modifies target_pv in place.
        """
        if self.target_pv is None:
            raise RuntimeError("Target mesh not loaded.")

        # Base bounds
        base_pv = self._to_pyvista(self.base_model)
        b_xmin, b_xmax, b_ymin, b_ymax, b_zmin, b_zmax = base_pv.bounds
        
        # Target bounds
        t_xmin, t_xmax, t_ymin, t_ymax, t_zmin, t_zmax = self.target_pv.bounds
        
        dx = dy = dz = 0.0
        
        if align_centers:
            dx = (b_xmin + b_xmax)/2.0 - (t_xmin + t_xmax)/2.0
            dy = (b_ymin + b_ymax)/2.0 - (t_ymin + t_ymax)/2.0
            
        if z_offset is not None:
            dz = z_offset
        else:
            dz = b_zmin - t_zmin

        self.target_pv.translate([dx, dy, dz], inplace=True)
        print(f" [Morpher] Aligned Target Mesh: Translated by ({dx:.2f}, {dy:.2f}, {dz:.2f})")

    def morph(self, bounding_box: Optional[Tuple[float, float, float, float, float, float]] = None, 
              direction: Optional[np.ndarray] = None,
              max_dist: float = 10.0,
              area_tolerance: float = 0.1) -> None:
        """
        Morphs base nodes within bounding_box to the target mesh.
        
        Parameters
        ----------
        bounding_box : tuple, optional
            (xmin, xmax, ymin, ymax, zmin, zmax)
        direction : np.ndarray, optional
            Projection direction vector. If None, uses surface normals.
        max_dist : float
            Maximum distance to search for target surface along the direction.
        area_tolerance : float
            Minimum ratio of morphed area vs original area (e.g. 0.1) to prevent inversion.
        """
        if self.target_pv is None:
            raise RuntimeError("Target mesh not loaded.")

        base_pv = self._to_pyvista(self.base_model)
        # Compute normals for base mesh points
        base_pv.compute_normals(cell_normals=False, point_normals=True, inplace=True, auto_orient_normals=True)
        normals = base_pv.point_data['Normals']

        n_morphed = 0
        n_rejected = 0
        
        # Create a mapping from node id to elements for safety checks
        node_to_elems = self._build_node_to_elem_map()
        
        original_coords = {nid: (n.x, n.y, n.z) for nid, n in self.base_model.nodes.items()}
        
        # Ray trace parameters
        for i, (nid, node) in enumerate(self.base_model.nodes.items()):
            if bounding_box:
                xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
                if not (xmin <= node.x <= xmax and ymin <= node.y <= ymax and zmin <= node.z <= zmax):
                    continue

            # Determine projection direction
            if direction is not None:
                nrm = direction / np.linalg.norm(direction)
            else:
                nrm = normals[i]
                
            p0 = np.array([node.x, node.y, node.z])
            start_pt = p0 - nrm * max_dist
            end_pt = p0 + nrm * max_dist
            
            # Cast ray
            points, cells = self.target_pv.ray_trace(start_pt, end_pt)
            
            if points.shape[0] > 0:
                # Find closest intersection
                dists = np.linalg.norm(points - p0, axis=1)
                best_idx = np.argmin(dists)
                new_p = points[best_idx]
                
                # Check safety (Area change)
                if self._check_element_safety(nid, new_p, node_to_elems, original_coords, area_tolerance):
                    node.x, node.y, node.z = new_p[0], new_p[1], new_p[2]
                    n_morphed += 1
                else:
                    n_rejected += 1

        print(f" [Morpher] Morphed {n_morphed} nodes. Rejected {n_rejected} due to safety limits.")

    def _to_pyvista(self, model: WHTMeshModel) -> pv.PolyData:
        """Converts shell nodes/elements of WHTMeshModel to PyVista PolyData."""
        nid_to_idx = {}
        points = []
        for idx, (nid, node) in enumerate(model.nodes.items()):
            nid_to_idx[nid] = idx
            points.append([node.x, node.y, node.z])
            
        faces = []
        for eid, elem in model.elements.items():
            if elem.type in ['QUAD4', 'TRIA3']:
                npts = len(elem.node_ids)
                faces.append(npts)
                faces.extend([nid_to_idx[nid] for nid in elem.node_ids])
                
        return pv.PolyData(np.array(points), np.array(faces))

    def _build_node_to_elem_map(self) -> Dict[int, List[int]]:
        n2e = {nid: [] for nid in self.base_model.nodes}
        for eid, elem in self.base_model.elements.items():
            for nid in elem.node_ids:
                if nid in n2e:
                    n2e[nid].append(eid)
        return n2e

    def _calc_elem_area(self, eid: int, temp_node_pos: Dict[int, np.ndarray], orig_pos: Dict[int, Tuple[float, float, float]]) -> float:
        elem = self.base_model.elements[eid]
        pts = []
        for nid in elem.node_ids:
            if nid in temp_node_pos:
                pts.append(temp_node_pos[nid])
            else:
                pts.append(np.array(orig_pos[nid]))
                
        if len(pts) == 3:
            return 0.5 * np.linalg.norm(np.cross(pts[1]-pts[0], pts[2]-pts[0]))
        elif len(pts) == 4:
            a1 = 0.5 * np.linalg.norm(np.cross(pts[1]-pts[0], pts[2]-pts[0]))
            a2 = 0.5 * np.linalg.norm(np.cross(pts[2]-pts[0], pts[3]-pts[0]))
            return a1 + a2
        return 0.0

    def _check_element_safety(self, nid: int, new_p: np.ndarray, n2e: Dict[int, List[int]], orig_pos: Dict[int, Tuple[float,float,float]], tol: float) -> bool:
        """Returns True if it is safe to move node nid to new_p."""
        for eid in n2e[nid]:
            orig_area = self._calc_elem_area(eid, {}, orig_pos)
            new_area = self._calc_elem_area(eid, {nid: new_p}, orig_pos)
            if orig_area > 1e-9:
                if new_area / orig_area < tol:
                    return False
        return True
