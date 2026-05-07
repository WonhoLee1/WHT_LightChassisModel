# -*- coding: utf-8 -*-
"""
wht_selector.py
===============
WHT FEM Framework ??Advanced Selection Engine

?곸슜 CAE Preprocessor(HyperMesh, ANSA ?????좏깮 湲곕뒫??踰ㅼ튂留덊궧?섏뿬 援ы쁽??
?몃뱶 諛??붿냼 ?좏깮 ?대옒?ㅼ엯?덈떎. 怨〓쪧 湲곕컲 ?좏깮, 怨듦컙 ?꾪꽣留? 洹몃━怨?
?대윭??議곗옉?ㅼ쓣 吏곷젹(Serial)濡??곌껐?섏뿬 蹂듭옟???꾪꽣留곸씠 媛?ν븯?꾨줉 ?ㅺ퀎?섏뿀?듬땲??
"""

import numpy as np
from typing import List, Set, Tuple, Optional, Union, Dict, Iterable
from .wht_mesh_model import WHTMeshModel


class WHTSelector:
    """
    怨좉툒 ?몃뱶/?붿냼 ?좏깮湲??대옒?ㅼ엯?덈떎.
    硫붿꽌??泥댁씠?앹쓣 ?듯빐 ?꾪꽣留?怨쇱젙??吏곴??곸쑝濡?湲곗닠?????덉뒿?덈떎.
    """

    def __init__(self, model: WHTMeshModel, initial_nids: Optional[Iterable[int]] = None):
        self.model = model
        # ?꾩옱 ?좏깮???몃뱶 ID?ㅼ쓽 吏묓빀
        if initial_nids is not None:
            self.selected_nids = set(initial_nids)
        else:
            self.selected_nids = set(model.nodes.keys())
        
        self._adj = None  # Lazy adjacency
        self._elem_normals = None # Lazy normals

    # ------------------------------------------------------------------
    # 1. 吏묓빀 ?곗궛 (Set Operations / Chaining)
    # ------------------------------------------------------------------

    def reset(self, to_all: bool = True) -> 'WHTSelector':
        """?좏깮 ?곸뿭??珥덇린?뷀빀?덈떎."""
        if to_all:
            self.selected_nids = set(self.model.nodes.keys())
        else:
            self.selected_nids = set()
        return self

    def add(self, other: Union['WHTSelector', List[int], Set[int]]) -> 'WHTSelector':
        """湲곗〈 ?좏깮 ?곸뿭???덈줈???몃뱶?ㅼ쓣 ?⑹묩?덈떎 (OR ?곗궛)."""
        ids = other.selected_nids if isinstance(other, WHTSelector) else set(other)
        self.selected_nids.update(ids)
        return self

    def filter(self, other: Union['WHTSelector', List[int], Set[int]]) -> 'WHTSelector':
        """湲곗〈 ?좏깮 ?곸뿭 以?議곌굔??留욌뒗 ?몃뱶?ㅻ쭔 ?④퉩?덈떎 (AND ?곗궛)."""
        ids = other.selected_nids if isinstance(other, WHTSelector) else set(other)
        self.selected_nids.intersection_update(ids)
        return self

    def remove(self, other: Union['WHTSelector', List[int], Set[int]]) -> 'WHTSelector':
        """湲곗〈 ?좏깮 ?곸뿭?먯꽌 ?뱀젙 ?몃뱶?ㅼ쓣 ?쒖쇅?⑸땲??(NOT ?곗궛)."""
        ids = other.selected_nids if isinstance(other, WHTSelector) else set(other)
        self.selected_nids.difference_update(ids)
        return self

    def get_ids(self) -> List[int]:
        """理쒖쥌 ?좏깮???몃뱶 ID 由ъ뒪?몃? 諛섑솚?⑸땲??"""
        return sorted(list(self.selected_nids))

    # ------------------------------------------------------------------
    # 2. 怨듦컙 湲곕컲 ?좏깮 (Spatial Selection)
    # ------------------------------------------------------------------

    def by_box(self, x: Tuple[float, float] = None, 
               y: Tuple[float, float] = None, 
               z: Tuple[float, float] = None) -> 'WHTSelector':
        """Bounding Box ?곸뿭 ?댁쓽 ?몃뱶?ㅼ쓣 ?꾪꽣留곹빀?덈떎."""
        new_set = set()
        for nid in self.selected_nids:
            node = self.model.nodes[nid]
            if (x is None or x[0] <= node.x <= x[1]) and \
               (y is None or y[0] <= node.y <= y[1]) and \
               (z is None or z[0] <= node.z <= z[1]):
                new_set.add(nid)
        self.selected_nids = new_set
        return self

    def by_sphere(self, center: Tuple[float, float, float], radius: float) -> 'WHTSelector':
        """援?Sphere) ?곸뿭 ?댁쓽 ?몃뱶?ㅼ쓣 ?꾪꽣留곹빀?덈떎."""
        cx, cy, cz = center
        r2 = radius ** 2
        new_set = set()
        for nid in self.selected_nids:
            n = self.model.nodes[nid]
            if (n.x - cx)**2 + (n.y - cy)**2 + (n.z - cz)**2 <= r2:
                new_set.add(nid)
        self.selected_nids = new_set
        return self

    # ------------------------------------------------------------------
    # 3. 吏묓빀 湲곕컲 ?좏깮 (Set-based Selection)
    # ------------------------------------------------------------------

    def by_ids(self, nids: Iterable[int]) -> 'WHTSelector':
        """?뱀젙 ?몃뱶 ID 由ъ뒪?몃? 湲곕컲?쇰줈 ?좏깮 ?곸뿭???ㅼ젙?⑸땲??"""
        self.selected_nids = set(nids)
        return self

    def by_set(self, sid_or_name: Union[int, str]) -> 'WHTSelector':
        """Node Set ID ?먮뒗 ?대쫫??湲곕컲?쇰줈 ?몃뱶?ㅼ쓣 ?좏깮?⑸땲??"""
        try:
            if isinstance(sid_or_name, int):
                nids = self.model.get_nodes_by_set(sid_or_name)
            else:
                nids = self.model.get_nodes_by_set_name(sid_or_name)
            self.selected_nids = set(nids)
        except KeyError:
            self.selected_nids = set()
        return self

    def by_elem_set(self, sid_or_name: Union[int, str]) -> 'WHTSelector':
        """Element Set ?댁쓽 ?붿냼?ㅼ씠 ?ы븿?섎뒗 紐⑤뱺 ?몃뱶?ㅼ쓣 ?좏깮?⑸땲??"""
        try:
            if isinstance(sid_or_name, int):
                nids = self.model.get_nodes_from_elem_set(sid_or_name)
            else:
                nids = self.model.get_nodes_from_elem_set_name(sid_or_name)
            self.selected_nids = set(nids)
        except KeyError:
            self.selected_nids = set()
        return self

    # ------------------------------------------------------------------
    # 4. ?꾩긽 諛?怨〓쪧 湲곕컲 ?좏깮 (Topology & Curvature Selection)
    # ------------------------------------------------------------------

    def by_path(self, seed_nid: int, angle_limit_deg: float = 20.0) -> 'WHTSelector':
        """
        ?먯? 諛⑺뼢 蹂?붾웾(怨〓쪧)??湲곕컲?쇰줈 ?곗냽??由?Rim) 寃쎈줈 ?몃뱶瑜??좏깮?⑸땲??
        (HyperMesh??'by Path' 湲곕뒫怨??좎궗)
        """
        adj = self._get_adj()
        if seed_nid not in self.model.nodes: return self
        
        visited = {seed_nid}
        path = [seed_nid]
        current = seed_nid
        prev_vec = None
        
        while True:
            next_node = None
            min_angle = 180.0
            
            for neighbor in adj.get(current, []):
                if neighbor in visited: continue
                
                c_coords = np.array(self.model.nodes[current].coords())
                n_coords = np.array(self.model.nodes[neighbor].coords())
                curr_vec = n_coords - c_coords
                curr_vec = curr_vec / (np.linalg.norm(curr_vec) + 1e-9)
                
                if prev_vec is None:
                    # ?쒖옉?먯뿉?쒕뒗 媛??二쇰맂 諛⑺뼢(?? X, Y, Z異?以??섎굹)???좏샇?섎룄濡?媛?뺥븯嫄곕굹 ?꾩쓽 ?좏깮
                    angle = 0.0 
                else:
                    angle = np.degrees(np.arccos(np.clip(np.dot(prev_vec, curr_vec), -1.0, 1.0)))
                
                if angle < angle_limit_deg and angle < min_angle:
                    min_angle = angle
                    next_node = neighbor
                    best_vec = curr_vec
            
            if next_node is None: break
            path.append(next_node)
            visited.add(next_node)
            prev_vec = best_vec
            current = next_node
            
        self.selected_nids = set(path)
        return self

    def by_face(self, seed_nid: int, angle_limit_deg: float = 30.0) -> 'WHTSelector':
        """
        硫?怨〓쪧(Face Curvature)??湲곕컲?쇰줈 ?몄젒???몃뱶?ㅼ쓣 ?좏깮?⑸땲??
        ?붿냼 踰뺤꽑(Normal) 踰≫꽣??蹂?붾웾???꾧퀎移??대궡???곸뿭??Flood fill ?⑸땲??
        (HyperMesh??'by Face' 湲곕뒫怨??좎궗)
        """
        if seed_nid not in self.model.nodes: return self
        
        adj = self._get_adj()
        elem_normals = self._get_elem_normals()
        
        # ?몃뱶? ?곌껐???붿냼??留??앹꽦
        node_to_elems = self._get_node_to_elems()
        
        visited_nodes = {seed_nid}
        queue = [seed_nid]
        
        # ?쒖옉 ?몃뱶 二쇰? ?붿냼?ㅼ쓽 ?됯퇏 踰뺤꽑??湲곗? 踰뺤꽑?쇰줈 ?ㅼ젙
        seed_elems = node_to_elems.get(seed_nid, [])
        if not seed_elems: return self
        
        # ?쒖옉 ?몃뱶??湲곗? 踰뺤꽑 怨꾩궛
        ref_normal = np.mean([elem_normals[eid] for eid in seed_elems], axis=0)
        ref_normal /= (np.linalg.norm(ref_normal) + 1e-9)
        
        while queue:
            curr_nid = queue.pop(0)
            for neighbor in adj.get(curr_nid, []):
                if neighbor in visited_nodes: continue
                
                # ?댁썐 ?몃뱶???됯퇏 踰뺤꽑 怨꾩궛
                neighbor_elems = node_to_elems.get(neighbor, [])
                if not neighbor_elems: continue
                
                neighbor_normal = np.mean([elem_normals[eid] for eid in neighbor_elems], axis=0)
                neighbor_normal /= (np.linalg.norm(neighbor_normal) + 1e-9)
                
                # ??踰뺤꽑 ?ъ씠??媛곷룄 怨꾩궛
                dot = np.clip(np.dot(ref_normal, neighbor_normal), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))
                
                if angle <= angle_limit_deg:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)
                    # ?좏깮?곸쑝濡?ref_normal???낅뜲?댄듃?섏뿬 ?먯쭊??怨〓쪧 ???媛??(?ш린?쒕뒗 怨좎젙 湲곗? ?ъ슜)
        
        self.selected_nids = visited_nodes
        return self

    def expand_by_face(self, angle_limit_deg: float = 30.0, z_min: Optional[float] = None) -> 'WHTSelector':
        """
        ?꾩옱 ?좏깮???몃뱶?ㅻ줈遺??硫?怨〓쪧(Face Curvature)??湲곕컲?쇰줈 ?좏깮 ?곸뿭???뺤옣?⑸땲??(Flood Fill).
        
        Args:
            angle_limit_deg: 踰뺤꽑 踰≫꽣 蹂?붾웾 ?덉슜移?
            z_min: ?뺤옣 ??怨좊젮??理쒖냼 ?믪씠 ?쒗븳 (?꾩슂??.
        """
        if not self.selected_nids: return self
        
        adj = self._get_adj()
        elem_normals = self._get_elem_normals()
        node_to_elems = self._get_node_to_elems()
        
        queue = list(self.selected_nids)
        final_selection = set(self.selected_nids)
        
        while queue:
            curr_nid = queue.pop(0)
            # 湲곗? 踰뺤꽑: ?꾩옱 ?몃뱶???됯퇏 踰뺤꽑
            curr_elems = node_to_elems.get(curr_nid, [])
            if not curr_elems: continue
            ref_normal = np.mean([elem_normals[eid] for eid in curr_elems], axis=0)
            ref_normal /= (np.linalg.norm(ref_normal) + 1e-9)
            
            for neighbor in adj.get(curr_nid, []):
                if neighbor in final_selection: continue
                if z_min is not None and self.model.nodes[neighbor].z < z_min: continue
                
                neighbor_elems = node_to_elems.get(neighbor, [])
                if not neighbor_elems: continue
                neighbor_normal = np.mean([elem_normals[eid] for eid in neighbor_elems], axis=0)
                neighbor_normal /= (np.linalg.norm(neighbor_normal) + 1e-9)
                
                dot = np.clip(np.dot(ref_normal, neighbor_normal), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))
                
                if angle <= angle_limit_deg:
                    final_selection.add(neighbor)
                    queue.append(neighbor)
                    
        self.selected_nids = final_selection
        return self

    # ------------------------------------------------------------------
    # 4. ?대? ?ы띁 (Internal Helpers)
    # ------------------------------------------------------------------

    def _get_adj(self):
        if self._adj is None:
            self._adj = self.model.get_adjacency()
        return self._adj

    def _get_node_to_elems(self) -> Dict[int, List[int]]:
        """?몃뱶 ID瑜?怨듭쑀?섎뒗 ?붿냼 ID 由ъ뒪??留듭쓣 ?앹꽦?⑸땲??"""
        mapping = {nid: [] for nid in self.model.nodes.keys()}
        for eid, elem in self.model.elements.items():
            for nid in elem.node_ids:
                if nid in mapping:
                    mapping[nid].append(eid)
        return mapping

    def _get_elem_normals(self) -> Dict[int, np.ndarray]:
        """紐⑤뱺 ???붿냼??踰뺤꽑 踰≫꽣瑜?怨꾩궛?⑸땲??"""
        if self._elem_normals is not None:
            return self._elem_normals
            
        normals = {}
        for eid, elem in self.model.elements.items():
            nids = elem.node_ids
            if len(nids) < 3:
                normals[eid] = np.array([0.0, 0.0, 1.0])
                continue
            
            # 理쒖냼 3媛쒖쓽 ?몃뱶瑜??ъ슜?섏뿬 踰뺤꽑 怨꾩궛 (踰≫꽣 ?몄쟻)
            p0 = np.array(self.model.nodes[nids[0]].coords())
            p1 = np.array(self.model.nodes[nids[1]].coords())
            p2 = np.array(self.model.nodes[nids[2]].coords())
            
            v1 = p1 - p0
            v2 = p2 - p0
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            normals[eid] = n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
            
        self._elem_normals = normals
        return normals

    def __repr__(self) -> str:
        return f"WHTSelector(selected={len(self.selected_nids)} / total={len(self.model.nodes)})"
