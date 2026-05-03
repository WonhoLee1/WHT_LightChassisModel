# -*- coding: utf-8 -*-
"""
wht_selector.py
===============
WHT FEM Framework — Advanced Selection Engine

상용 CAE Preprocessor(HyperMesh, ANSA 등)의 선택 기능을 벤치마킹하여 구현된 
노드 및 요소 선택 클래스입니다. 곡률 기반 선택, 공간 필터링, 그리고 
이러한 조작들을 직렬(Serial)로 연결하여 복잡한 필터링이 가능하도록 설계되었습니다.
"""

import numpy as np
from typing import List, Set, Tuple, Optional, Union, Dict, Iterable
from .wht_mesh_model import WHTMeshModel


class WHTSelector:
    """
    고급 노드/요소 선택기 클래스입니다.
    메서드 체이닝을 통해 필터링 과정을 직관적으로 기술할 수 있습니다.
    """

    def __init__(self, model: WHTMeshModel, initial_nids: Optional[Iterable[int]] = None):
        self.model = model
        # 현재 선택된 노드 ID들의 집합
        if initial_nids is not None:
            self.selected_nids = set(initial_nids)
        else:
            self.selected_nids = set(model.nodes.keys())
        
        self._adj = None  # Lazy adjacency
        self._elem_normals = None # Lazy normals

    # ------------------------------------------------------------------
    # 1. 집합 연산 (Set Operations / Chaining)
    # ------------------------------------------------------------------

    def reset(self, to_all: bool = True) -> 'WHTSelector':
        """선택 영역을 초기화합니다."""
        if to_all:
            self.selected_nids = set(self.model.nodes.keys())
        else:
            self.selected_nids = set()
        return self

    def add(self, other: Union['WHTSelector', List[int], Set[int]]) -> 'WHTSelector':
        """기존 선택 영역에 새로운 노드들을 합칩니다 (OR 연산)."""
        ids = other.selected_nids if isinstance(other, WHTSelector) else set(other)
        self.selected_nids.update(ids)
        return self

    def filter(self, other: Union['WHTSelector', List[int], Set[int]]) -> 'WHTSelector':
        """기존 선택 영역 중 조건에 맞는 노드들만 남깁니다 (AND 연산)."""
        ids = other.selected_nids if isinstance(other, WHTSelector) else set(other)
        self.selected_nids.intersection_update(ids)
        return self

    def remove(self, other: Union['WHTSelector', List[int], Set[int]]) -> 'WHTSelector':
        """기존 선택 영역에서 특정 노드들을 제외합니다 (NOT 연산)."""
        ids = other.selected_nids if isinstance(other, WHTSelector) else set(other)
        self.selected_nids.difference_update(ids)
        return self

    def get_ids(self) -> List[int]:
        """최종 선택된 노드 ID 리스트를 반환합니다."""
        return sorted(list(self.selected_nids))

    # ------------------------------------------------------------------
    # 2. 공간 기반 선택 (Spatial Selection)
    # ------------------------------------------------------------------

    def by_box(self, x: Tuple[float, float] = None, 
               y: Tuple[float, float] = None, 
               z: Tuple[float, float] = None) -> 'WHTSelector':
        """Bounding Box 영역 내의 노드들을 필터링합니다."""
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
        """구(Sphere) 영역 내의 노드들을 필터링합니다."""
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
    # 3. 집합 기반 선택 (Set-based Selection)
    # ------------------------------------------------------------------

    def by_set(self, sid_or_name: Union[int, str]) -> 'WHTSelector':
        """Node Set ID 또는 이름을 기반으로 노드들을 선택합니다."""
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
        """Element Set 내의 요소들이 포함하는 모든 노드들을 선택합니다."""
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
    # 4. 위상 및 곡률 기반 선택 (Topology & Curvature Selection)
    # ------------------------------------------------------------------

    def by_path(self, seed_nid: int, angle_limit_deg: float = 20.0) -> 'WHTSelector':
        """
        에지 방향 변화량(곡률)을 기반으로 연속된 림(Rim) 경로 노드를 선택합니다.
        (HyperMesh의 'by Path' 기능과 유사)
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
                    # 시작점에서는 가장 주된 방향(예: X, Y, Z축 중 하나)을 선호하도록 가정하거나 임의 선택
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
        면 곡률(Face Curvature)을 기반으로 인접한 노드들을 선택합니다.
        요소 법선(Normal) 벡터의 변화량이 임계치 이내인 영역을 Flood fill 합니다.
        (HyperMesh의 'by Face' 기능과 유사)
        """
        if seed_nid not in self.model.nodes: return self
        
        adj = self._get_adj()
        elem_normals = self._get_elem_normals()
        
        # 노드와 연결된 요소들 맵 생성
        node_to_elems = self._get_node_to_elems()
        
        visited_nodes = {seed_nid}
        queue = [seed_nid]
        
        # 시작 노드 주변 요소들의 평균 법선을 기준 법선으로 설정
        seed_elems = node_to_elems.get(seed_nid, [])
        if not seed_elems: return self
        
        # 시작 노드의 기준 법선 계산
        ref_normal = np.mean([elem_normals[eid] for eid in seed_elems], axis=0)
        ref_normal /= (np.linalg.norm(ref_normal) + 1e-9)
        
        while queue:
            curr_nid = queue.pop(0)
            for neighbor in adj.get(curr_nid, []):
                if neighbor in visited_nodes: continue
                
                # 이웃 노드의 평균 법선 계산
                neighbor_elems = node_to_elems.get(neighbor, [])
                if not neighbor_elems: continue
                
                neighbor_normal = np.mean([elem_normals[eid] for eid in neighbor_elems], axis=0)
                neighbor_normal /= (np.linalg.norm(neighbor_normal) + 1e-9)
                
                # 두 법선 사이의 각도 계산
                dot = np.clip(np.dot(ref_normal, neighbor_normal), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))
                
                if angle <= angle_limit_deg:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)
                    # 선택적으로 ref_normal을 업데이트하여 점진적 곡률 대응 가능 (여기서는 고정 기준 사용)
        
        self.selected_nids = visited_nodes
        return self

    def expand_by_face(self, angle_limit_deg: float = 30.0, z_min: Optional[float] = None) -> 'WHTSelector':
        """
        현재 선택된 노드들로부터 면 곡률(Face Curvature)을 기반으로 선택 영역을 확장합니다 (Flood Fill).
        
        Args:
            angle_limit_deg: 법선 벡터 변화량 허용치.
            z_min: 확장 시 고려할 최소 높이 제한 (필요시).
        """
        if not self.selected_nids: return self
        
        adj = self._get_adj()
        elem_normals = self._get_elem_normals()
        node_to_elems = self._get_node_to_elems()
        
        queue = list(self.selected_nids)
        final_selection = set(self.selected_nids)
        
        while queue:
            curr_nid = queue.pop(0)
            # 기준 법선: 현재 노드의 평균 법선
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
    # 4. 내부 헬퍼 (Internal Helpers)
    # ------------------------------------------------------------------

    def _get_adj(self):
        if self._adj is None:
            self._adj = self.model.get_adjacency()
        return self._adj

    def _get_node_to_elems(self) -> Dict[int, List[int]]:
        """노드 ID를 공유하는 요소 ID 리스트 맵을 생성합니다."""
        mapping = {nid: [] for nid in self.model.nodes.keys()}
        for eid, elem in self.model.elements.items():
            for nid in elem.node_ids:
                if nid in mapping:
                    mapping[nid].append(eid)
        return mapping

    def _get_elem_normals(self) -> Dict[int, np.ndarray]:
        """모든 쉘 요소의 법선 벡터를 계산합니다."""
        if self._elem_normals is not None:
            return self._elem_normals
            
        normals = {}
        for eid, elem in self.model.elements.items():
            nids = elem.node_ids
            if len(nids) < 3:
                normals[eid] = np.array([0.0, 0.0, 1.0])
                continue
            
            # 최소 3개의 노드를 사용하여 법선 계산 (벡터 외적)
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
