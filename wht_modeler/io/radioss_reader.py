"""
radioss_reader.py
=================
OpenRadioss format (.rad) reader for WHTMeshModel.
Reads /NODE, /QUAD, /TRIA, /HEXA and node sets.
"""

from __future__ import annotations
from pathlib import Path
from .base_reader import BaseFEMReader
from ..wht_mesh_model import WHTMeshModel

class RadiossReader(BaseFEMReader):
    """Parser for OpenRadioss format."""

    def read(self, file_path: str) -> WHTMeshModel:
        path = Path(file_path)
        model = WHTMeshModel(name=path.name)
        
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
                
            if line.startswith('/NODE'):
                i += 1
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
                    except ValueError:
                        pass
                    i += 1
                continue
            elif line.startswith('/QUAD') or line.startswith('/SH3N') or line.startswith('/TRIA'):
                etype = "QUAD4" if line.startswith('/QUAD') else "TRIA3"
                i += 1
                while i < len(lines):
                    line = lines[i]
                    if line.startswith('/'): break
                    if line.startswith('#') or not line.strip(): 
                        i += 1; continue
                    try:
                        eid = int(line[0:10])
                        pid = int(line[10:20]) if len(line) >= 20 and line[10:20].strip() else 0
                        nids = [int(line[j:j+10]) for j in range(20, len(line), 10) if line[j:j+10].strip()]
                        model.add_element(eid, nids, etype, pid)
                    except ValueError:
                        pass
                    i += 1
                continue
            elif line.startswith('/GRNOD/NODE'):
                i += 1
                if i < len(lines):
                    name = lines[i].strip()
                    i += 1
                    nids = []
                    while i < len(lines):
                        line = lines[i]
                        if line.startswith('/'): break
                        if not line.startswith('#') and line.strip():
                            for j in range(0, len(line), 10):
                                val = line[j:j+10].strip()
                                if val: nids.append(int(val))
                        i += 1
                    model.add_node_set_by_name(name, nids)
                continue
            
            i += 1
            
        return model
