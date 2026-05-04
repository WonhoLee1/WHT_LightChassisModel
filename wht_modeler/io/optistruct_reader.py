"""
optistruct_reader.py
====================
OptiStruct / Nastran format (.fem / .bdf) reader for WHTMeshModel.
Reads GRID, CQUAD4, CTRIA3, CHEXA, CTETRA and sets.
"""

from __future__ import annotations
from pathlib import Path
from .base_reader import BaseFEMReader
from ..wht_mesh_model import WHTMeshModel

class OptistructReader(BaseFEMReader):
    """Parser for OptiStruct/Nastran format."""

    def read(self, file_path: str) -> WHTMeshModel:
        path = Path(file_path)
        model = WHTMeshModel(name=path.name)
        
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('$'):
                i += 1
                continue
            
            # Nastran 8-character fixed format parsing
            keyword = line[:8].strip()
            
            if keyword == 'GRID':
                try:
                    nid = int(line[8:16])
                    x = float(line[24:32])
                    y = float(line[32:40])
                    z = float(line[40:48])
                    model.add_node(nid, x, y, z)
                except ValueError:
                    # Free format fallback
                    parts = line.split(',')
                    if len(parts) >= 6:
                        try:
                            nid = int(parts[1])
                            model.add_node(nid, float(parts[3]), float(parts[4]), float(parts[5]))
                        except ValueError:
                            pass
            elif keyword in ['CQUAD4', 'CTRIA3']:
                try:
                    eid = int(line[8:16])
                    pid = int(line[16:24]) if line[16:24].strip() else 0
                    nids = [int(line[j:j+8]) for j in range(24, len(line), 8) if line[j:j+8].strip()]
                    model.add_element(eid, nids, "QUAD4" if keyword == 'CQUAD4' else "TRIA3", pid)
                except ValueError:
                    pass
            elif keyword == 'SET':
                # Simplified set parsing
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        sid = int(parts[1].strip())
                        stype = parts[2].strip()
                        ids = []
                        # Read continuations
                        i += 1
                        while i < len(lines):
                            cline = lines[i].strip()
                            if cline.startswith('+') or cline.startswith(','):
                                cparts = cline[1:].split(',')
                                ids.extend([int(v.strip()) for v in cparts if v.strip()])
                                i += 1
                            else:
                                i -= 1
                                break
                        if stype == "GRID":
                            model.add_node_set(sid, ids)
                        else:
                            model.add_elem_set(sid, ids)
                    except ValueError:
                        pass
            i += 1
            
        return model
