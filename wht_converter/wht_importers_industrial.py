# -*- coding: utf-8 -*-
"""
wht_importers_industrial.py
===========================
WHT Industrial Solver Importers
Converts industrial solver formats (LS-DYNA, Radioss, OptiStruct) into WHTMeshModel.
Specifically focuses on reading Node Sets and Element Sets.
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from wht_modeler.wht_mesh_model import WHTMeshModel


class IndustrialImporterBase:
    """Base class for industrial solver importers."""
    
    def _clean_line(self, line: str) -> str:
        """Removes comments and whitespace."""
        # Nastran/Optistruct use $ for comments
        # LS-DYNA uses $ for comments
        # Radioss uses # for comments (sometimes)
        if '$' in line:
            line = line.split('$')[0]
        if '#' in line:
            line = line.split('#')[0]
        return line.strip()


class LSDYNAImporter(IndustrialImporterBase):
    """Importer for LS-DYNA Keyword format (*.k)"""
    
    def read(self, path: str) -> WHTMeshModel:
        model = WHTMeshModel(name=os.path.basename(path))
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('*NODE'):
                i = self._parse_nodes(model, lines, i + 1)
            elif line.startswith('*ELEMENT_SHELL'):
                i = self._parse_elements(model, lines, i + 1, "QUAD4")
            elif line.startswith('*ELEMENT_SOLID'):
                i = self._parse_elements(model, lines, i + 1, "HEXA8")
            elif line.startswith('*SET_NODE_LIST'):
                i = self._parse_set_node(model, lines, i + 1)
            elif line.startswith('*SET_PART_LIST') or line.startswith('*SET_SHELL_LIST'):
                i = self._parse_set_elem(model, lines, i + 1)
            else:
                i += 1
        return model

    def _parse_nodes(self, model, lines, start_idx):
        i = start_idx
        while i < len(lines):
            line = lines[i]
            if line.startswith('*'): break
            if line.startswith('$') or not line.strip(): 
                i += 1; continue
            
            # nid (10), x (16), y (16), z (16)
            try:
                nid = int(line[0:10])
                x = float(line[10:26])
                y = float(line[26:42])
                z = float(line[42:58])
                model.add_node(nid, x, y, z)
            except: pass
            i += 1
        return i

    def _parse_elements(self, model, lines, start_idx, etype):
        i = start_idx
        while i < len(lines):
            line = lines[i]
            if line.startswith('*'): break
            if line.startswith('$') or not line.strip(): 
                i += 1; continue
            
            # eid (10), pid (10), n1..n4 (10 each)
            try:
                eid = int(line[0:10])
                nids = []
                for j in range(20, len(line), 10):
                    val = line[j:j+10].strip()
                    if val: nids.append(int(val))
                model.add_element(eid, nids, etype)
            except: pass
            i += 1
        return i

    def _parse_set_node(self, model, lines, start_idx):
        i = start_idx
        sid = None
        nids = []
        while i < len(lines):
            line = lines[i]
            if line.startswith('*'): break
            if line.startswith('$') or not line.strip(): 
                i += 1; continue
            
            if sid is None:
                sid = int(line[0:10])
            else:
                # Multiple nids in a line (8 x 10)
                for j in range(0, len(line), 10):
                    val = line[j:j+10].strip()
                    if val: nids.append(int(val))
            i += 1
        if sid is not None:
            model.add_node_set(sid, nids)
        return i

    def _parse_set_elem(self, model, lines, start_idx):
        i = start_idx
        sid = None
        eids = []
        while i < len(lines):
            line = lines[i]
            if line.startswith('*'): break
            if line.startswith('$') or not line.strip(): 
                i += 1; continue
            
            if sid is None:
                sid = int(line[0:10])
            else:
                for j in range(0, len(line), 10):
                    val = line[j:j+10].strip()
                    if val: eids.append(int(val))
            i += 1
        if sid is not None:
            model.add_elem_set(sid, eids)
        return i


class RadiossImporter(IndustrialImporterBase):
    """Importer for OpenRadioss format (*.rad)"""
    
    def read(self, path: str) -> WHTMeshModel:
        model = WHTMeshModel(name=os.path.basename(path))
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('/NODE'):
                i = self._parse_nodes(model, lines, i + 1)
            elif line.startswith('/QUAD'):
                i = self._parse_elements(model, lines, i + 1, "QUAD4")
            elif line.startswith('/HEXA8'):
                i = self._parse_elements(model, lines, i + 1, "HEXA8")
            elif line.startswith('/GRNOD/NODE'):
                i = self._parse_grnod(model, lines, i + 1)
            else:
                i += 1
        return model

    def _parse_nodes(self, model, lines, start_idx):
        i = start_idx
        while i < len(lines):
            line = lines[i]
            if line.startswith('/'): break
            if line.startswith('#') or not line.strip(): 
                i += 1; continue
            try:
                nid = int(line[0:10])
                x = float(line[10:30])
                y = float(line[30:50])
                z = float(line[50:70])
                model.add_node(nid, x, y, z)
            except: pass
            i += 1
        return i

    def _parse_elements(self, model, lines, start_idx, etype):
        i = start_idx
        while i < len(lines):
            line = lines[i]
            if line.startswith('/'): break
            if line.startswith('#') or not line.strip(): 
                i += 1; continue
            try:
                eid = int(line[0:10])
                nids = []
                for j in range(20, len(line), 10):
                    val = line[j:j+10].strip()
                    if val: nids.append(int(val))
                model.add_element(eid, nids, etype)
            except: pass
            i += 1
        return i

    def _parse_grnod(self, model, lines, start_idx):
        i = start_idx
        name = lines[i].strip()
        i += 1
        nids = []
        while i < len(lines):
            line = lines[i]
            if line.startswith('/'): break
            for j in range(0, len(line), 10):
                val = line[j:j+10].strip()
                if val: nids.append(int(val))
            i += 1
        model.add_node_set_by_name(name, nids)
        return i


class OptistructImporter(IndustrialImporterBase):
    """Importer for OptiStruct/Nastran format (*.fem)"""
    
    def read(self, path: str) -> WHTMeshModel:
        model = WHTMeshModel(name=os.path.basename(path))
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = self._clean_line(lines[i])
            if line.startswith('GRID'):
                # Fixed format GRID
                try:
                    nid = int(line[8:16])
                    x = float(line[24:32])
                    y = float(line[32:40])
                    z = float(line[40:48])
                    model.add_node(nid, x, y, z)
                except: pass
            elif line.startswith('CQUAD4'):
                try:
                    eid = int(line[8:16])
                    nids = [int(line[j:j+8]) for j in range(24, len(line), 8) if line[j:j+8].strip()]
                    model.add_element(eid, nids, "QUAD4")
                except: pass
            elif line.startswith('SET'):
                i = self._parse_set(model, lines, i)
                continue
            i += 1
        return model

    def _parse_set(self, model, lines, idx):
        line = lines[idx]
        # SET, ID, TYPE, LIST
        parts = line.split(',')
        if len(parts) < 3: return idx + 1
        sid = int(parts[1].strip())
        stype = parts[2].strip()
        
        ids = []
        i = idx
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('+') or line.startswith(','):
                # Continuation
                curr_line = line[1:].split(',')
                for val in curr_line:
                    if val.strip(): ids.append(int(val.strip()))
            elif i != idx:
                break
            i += 1
            
        if stype == "GRID":
            model.add_node_set(sid, ids)
        elif stype in ["ELEM", "PART"]:
            model.add_elem_set(sid, ids)
        return i
