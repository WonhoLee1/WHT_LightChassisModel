"""
wht_readers.py
==============
WHT Universal FEM Framework — Reader Layer

Parses solver-specific files into WHTMeshModel.
Supported formats:
  - LS-DYNA (.k / .key)
"""

import re
import numpy as np
from typing import Optional
from .wht_mesh_model import WHTMeshModel


class LSDYNAReader:
    """
    Parser for LS-DYNA keyword files.
    """
    def __init__(self):
        self.model = WHTMeshModel()

    def read(self, file_path: str) -> WHTMeshModel:
        self.model = WHTMeshModel(name=file_path)
        
        with open(file_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('*NODE'):
                i = self._parse_nodes(lines, i + 1)
            elif line.startswith('*ELEMENT_SHELL'):
                i = self._parse_elements(lines, i + 1, "QUAD")
            elif line.startswith('*SET_NODE_LIST'):
                i = self._parse_set_node_list(lines, i + 1)
            elif line.startswith('*SET_NODE_GENERAL'):
                i = self._parse_set_node_general(lines, i + 1)
            elif line.startswith('*DEFINE_BOX'):
                i = self._parse_define_box(lines, i + 1)
            else:
                i += 1
        
        return self.model

    def _parse_nodes(self, lines, start_idx):
        idx = start_idx
        while idx < len(lines) and not lines[idx].startswith('*'):
            line = lines[idx]
            if line.startswith('$') or not line.strip():
                idx += 1
                continue
            # LS-DYNA standard format: 8 columns (id, x, y, z, ...)
            # We use a flexible split to handle both fixed and free format
            parts = line.split()
            if len(parts) >= 4:
                nid = int(parts[0])
                coords = [float(p) for p in parts[1:4]]
                self.model.add_node(nid, *coords)
            idx += 1
        return idx

    def _parse_elements(self, lines, start_idx, etype):
        idx = start_idx
        while idx < len(lines) and not lines[idx].startswith('*'):
            line = lines[idx]
            if line.startswith('$') or not line.strip():
                idx += 1
                continue
            parts = line.split()
            if len(parts) >= 6:
                eid = int(parts[0])
                pid = int(parts[1])
                node_ids = [int(p) for p in parts[2:6]]
                self.model.add_element(eid, node_ids, etype)
                
                # Store PID for SET_GENERAL partitioning
                if pid not in self.model.parts:
                    self.model.parts[pid] = []
                self.model.parts[pid].append(eid)
            idx += 1
        return idx

    def _parse_set_node_list(self, lines, start_idx):
        idx = start_idx
        if idx >= len(lines): return idx
        
        # First line is SID (and other attributes)
        sid_line = lines[idx].strip()
        if not sid_line or sid_line.startswith('$'):
            idx += 1
            sid_line = lines[idx].strip()
            
        sid = int(sid_line.split()[0])
        node_ids = []
        idx += 1
        
        while idx < len(lines) and not lines[idx].startswith('*'):
            line = lines[idx].strip()
            if line and not line.startswith('$'):
                node_ids.extend([int(p) for p in line.split()])
            idx += 1
        
        self.model.create_node_set(sid, f"SET_{sid}", node_ids)
        return idx

    def _parse_set_node_general(self, lines, start_idx):
        idx = start_idx
        sid_line = lines[idx].strip()
        if not sid_line or sid_line.startswith('$'):
            idx += 1
            sid_line = lines[idx].strip()
        sid = int(sid_line.split()[0])
        idx += 1
        
        node_ids = set()
        while idx < len(lines) and not lines[idx].startswith('*'):
            line = lines[idx].strip()
            if not line or line.startswith('$'):
                idx += 1
                continue
            parts = line.split()
            cmd = parts[0]
            if cmd == 'BOX':
                bid = int(parts[1])
                if bid in self.model.boxes:
                    box = self.model.boxes[bid]
                    for nid, coords in self.model.nodes.items():
                        if (box[0] <= coords[0] <= box[1] and
                            box[2] <= coords[1] <= box[3] and
                            box[4] <= coords[2] <= box[5]):
                            node_ids.add(nid)
            elif cmd == 'PART':
                pid = int(parts[1])
                # Find all nodes belonging to elements in this part
                if pid in self.model.parts:
                    for eid in self.model.parts[pid]:
                        node_ids.update(self.model.elements[eid])
            idx += 1
            
        self.model.create_node_set(sid, f"SET_GEN_{sid}", list(node_ids))
        return idx

    def _parse_define_box(self, lines, start_idx):
        idx = start_idx
        while idx < len(lines) and not lines[idx].startswith('*'):
            line = lines[idx].strip()
            if not line or line.startswith('$'):
                idx += 1
                continue
            parts = line.split()
            if len(parts) >= 7:
                bid = int(parts[0])
                coords = [float(p) for p in parts[1:7]]
                self.model.boxes[bid] = np.array(coords)
            idx += 1
        return idx
