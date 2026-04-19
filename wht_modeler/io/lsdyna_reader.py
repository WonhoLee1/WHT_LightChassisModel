"""
lsdyna_reader.py
================
LS-DYNA keyword file reader (.k / .key)

Supported keywords
------------------
*NODE
*ELEMENT_SHELL, *ELEMENT_BEAM
*PART
*MAT_ELASTIC (and common MAT variants)
*SECTION_SHELL
*SET_NODE_LIST
*SET_NODE_GENERAL  (BOX, PART sub-options)
*SET_ELEMENT_LIST
*DEFINE_BOX
*CONSTRAINED_NODAL_RIGID_BODY  (RBE2 equivalent)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from .base_reader import BaseFEMReader
from ..wht_mesh_model import WHTMeshModel


class LSDYNAReader(BaseFEMReader):
    """Parser for LS-DYNA keyword files."""

    def read(self, file_path: str) -> WHTMeshModel:
        path = Path(file_path)
        model = WHTMeshModel(name=path.name)

        with open(path, "r") as fh:
            lines = fh.readlines()

        # Property / material temporary storage (resolved after full parse)
        _parts:    dict[int, dict] = {}   # {pid: {mid, title}}
        _sections: dict[int, float] = {}  # {pid: t}   (SECTION_SHELL → pid)
        _mats:     dict[int, dict] = {}   # {mid: {E, nu, rho}}

        i = 0
        while i < len(lines):
            raw = lines[i]
            kw  = raw.strip().upper()

            if kw.startswith("*NODE"):
                i = self._parse_nodes(lines, i + 1, model)
            elif kw.startswith("*ELEMENT_SHELL"):
                i = self._parse_elements_shell(lines, i + 1, model)
            elif kw.startswith("*ELEMENT_BEAM"):
                i = self._parse_elements_beam(lines, i + 1, model)
            elif kw.startswith("*PART"):
                i = self._parse_part(lines, i + 1, _parts)
            elif kw.startswith("*SECTION_SHELL"):
                i = self._parse_section_shell(lines, i + 1, _sections)
            elif kw.startswith("*MAT_ELASTIC") or kw.startswith("*MAT_001"):
                i = self._parse_mat_elastic(lines, i + 1, _mats)
            elif kw.startswith("*MAT_"):
                i = self._parse_mat_generic(lines, i + 1, _mats)
            elif kw.startswith("*SET_NODE_LIST"):
                i = self._parse_set_node_list(lines, i + 1, model)
            elif kw.startswith("*SET_NODE_GENERAL"):
                i = self._parse_set_node_general(lines, i + 1, model)
            elif kw.startswith("*SET_ELEMENT_LIST"):
                i = self._parse_set_element_list(lines, i + 1, model)
            elif kw.startswith("*DEFINE_BOX"):
                i = self._parse_define_box(lines, i + 1, model)
            elif kw.startswith("*CONSTRAINED_NODAL_RIGID_BODY"):
                i = self._parse_cnrb(lines, i + 1, model)
            else:
                i += 1

        # Resolve properties and materials
        self._resolve_props(model, _parts, _sections, _mats)
        return model

    # ------------------------------------------------------------------
    # Section parsers
    # ------------------------------------------------------------------

    def _skip(self, lines, idx):
        """Skip comment / blank lines; return first data line index."""
        while idx < len(lines):
            s = lines[idx].strip()
            if s and not s.startswith("$"):
                return idx
            idx += 1
        return idx

    def _is_end(self, line: str) -> bool:
        s = line.strip()
        return s.startswith("*") or s.upper().startswith("*END")

    def _parse_nodes(self, lines, start, model: WHTMeshModel) -> int:
        idx = start
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            line = lines[idx]
            if line.startswith("$") or not line.strip():
                idx += 1
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    nid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    model.add_node(nid, x, y, z)
                except ValueError:
                    pass
            idx += 1
        return idx

    def _parse_elements_shell(self, lines, start, model: WHTMeshModel) -> int:
        idx = start
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            line = lines[idx]
            if line.startswith("$") or not line.strip():
                idx += 1
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    eid = int(parts[0])
                    pid = int(parts[1])
                    nids = [int(p) for p in parts[2:6]]
                    # Skip degenerate nodes (LS-DYNA uses 0 for TRIA in QUAD card)
                    nids = [n for n in nids if n != 0]
                    etype = "QUAD4" if len(nids) == 4 else "TRIA3"
                    model.add_element(eid, nids, etype, pid)
                except ValueError:
                    pass
            idx += 1
        return idx

    def _parse_elements_beam(self, lines, start, model: WHTMeshModel) -> int:
        idx = start
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            line = lines[idx]
            if line.startswith("$") or not line.strip():
                idx += 1
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    eid = int(parts[0])
                    pid = int(parts[1])
                    nids = [int(parts[2]), int(parts[3])]
                    model.add_element(eid, nids, "BEAM2", pid)
                except ValueError:
                    pass
            idx += 1
        return idx

    def _parse_part(self, lines, start, _parts: dict) -> int:
        """*PART format: optional title line, then pid/secid/mid line."""
        idx = start
        # Collect up to 2 non-comment, non-blank lines
        data_lines = []
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            line = lines[idx].strip()
            if line and not line.startswith("$"):
                data_lines.append(line)
                if len(data_lines) == 2:
                    idx += 1
                    break
            idx += 1

        # Try last collected line as "pid secid mid"
        for candidate in reversed(data_lines):
            parts = candidate.split()
            if len(parts) >= 3:
                try:
                    pid   = int(parts[0])
                    secid = int(parts[1])
                    mid   = int(parts[2])
                    _parts[pid] = {"mid": mid, "secid": secid}
                    break
                except ValueError:
                    continue

        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            idx += 1
        return idx

    def _parse_section_shell(self, lines, start, _sections: dict) -> int:
        """*SECTION_SHELL: secid / elform / shrf / nip / propt / qr/irid / icomp / setyp"""
        idx = self._skip(lines, start)
        if idx >= len(lines) or lines[idx].strip().startswith("*"):
            return idx
        header = lines[idx].split()
        idx += 1
        # Next line has t1~t4
        idx = self._skip(lines, idx)
        if idx < len(lines) and not lines[idx].strip().startswith("*"):
            try:
                secid = int(header[0])
                t = float(lines[idx].split()[0])
                _sections[secid] = t
            except (ValueError, IndexError):
                pass
            idx += 1
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            idx += 1
        return idx

    def _parse_mat_elastic(self, lines, start, _mats: dict) -> int:
        """*MAT_ELASTIC: mid / rho / E / nu ..."""
        idx = self._skip(lines, start)
        if idx >= len(lines) or lines[idx].strip().startswith("*"):
            return idx
        parts = lines[idx].split()
        if len(parts) >= 4:
            try:
                mid = int(parts[0])
                rho = float(parts[1])
                E   = float(parts[2])
                nu  = float(parts[3])
                _mats[mid] = {"E": E, "nu": nu, "rho": rho}
            except ValueError:
                pass
        idx += 1
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            idx += 1
        return idx

    def _parse_mat_generic(self, lines, start, _mats: dict) -> int:
        """Try to extract mid/rho/E/nu from first data line of any *MAT_ card."""
        idx = self._skip(lines, start)
        if idx >= len(lines) or lines[idx].strip().startswith("*"):
            return idx
        parts = lines[idx].split()
        if len(parts) >= 4:
            try:
                mid = int(parts[0])
                rho = float(parts[1])
                E   = float(parts[2])
                nu  = float(parts[3])
                _mats.setdefault(mid, {"E": E, "nu": nu, "rho": rho})
            except ValueError:
                pass
        idx += 1
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            idx += 1
        return idx

    def _parse_set_node_list(self, lines, start, model: WHTMeshModel) -> int:
        idx = self._skip(lines, start)
        if idx >= len(lines):
            return idx
        try:
            sid = int(lines[idx].split()[0])
        except (ValueError, IndexError):
            return idx + 1
        idx += 1
        node_ids = []
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            line = lines[idx].strip()
            if line and not line.startswith("$"):
                for tok in line.split():
                    try:
                        node_ids.append(int(tok))
                    except ValueError:
                        pass
            idx += 1
        model.add_node_set(sid, node_ids, name=f"nset_{sid}")
        return idx

    def _parse_set_node_general(self, lines, start, model: WHTMeshModel) -> int:
        idx = self._skip(lines, start)
        if idx >= len(lines):
            return idx
        try:
            sid = int(lines[idx].split()[0])
        except (ValueError, IndexError):
            return idx + 1
        idx += 1
        node_ids: set[int] = set()
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            line = lines[idx].strip()
            if not line or line.startswith("$"):
                idx += 1
                continue
            parts = line.split()
            cmd = parts[0].upper()
            if cmd == "BOX" and len(parts) >= 2:
                try:
                    bid = int(parts[1])
                    box = model._boxes.get(bid)
                    if box is not None:
                        for nid, node in model.nodes.items():
                            if (box[0] <= node.x <= box[1] and
                                    box[2] <= node.y <= box[3] and
                                    box[4] <= node.z <= box[5]):
                                node_ids.add(nid)
                except ValueError:
                    pass
            elif cmd == "PART" and len(parts) >= 2:
                try:
                    pid = int(parts[1])
                    for eid in model._parts.get(pid, []):
                        node_ids.update(model.elements[eid].node_ids)
                except ValueError:
                    pass
            idx += 1
        model.add_node_set(sid, sorted(node_ids), name=f"nset_gen_{sid}")
        return idx

    def _parse_set_element_list(self, lines, start, model: WHTMeshModel) -> int:
        idx = self._skip(lines, start)
        if idx >= len(lines):
            return idx
        try:
            sid = int(lines[idx].split()[0])
        except (ValueError, IndexError):
            return idx + 1
        idx += 1
        elem_ids = []
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            line = lines[idx].strip()
            if line and not line.startswith("$"):
                for tok in line.split():
                    try:
                        elem_ids.append(int(tok))
                    except ValueError:
                        pass
            idx += 1
        model.add_elem_set(sid, elem_ids, name=f"eset_{sid}")
        return idx

    def _parse_define_box(self, lines, start, model: WHTMeshModel) -> int:
        idx = start
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            line = lines[idx].strip()
            if not line or line.startswith("$"):
                idx += 1
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    bid = int(parts[0])
                    coords = [float(p) for p in parts[1:7]]
                    model._boxes[bid] = np.array(coords, dtype=np.float64)
                except ValueError:
                    pass
            idx += 1
        return idx

    def _parse_cnrb(self, lines, start, model: WHTMeshModel) -> int:
        """*CONSTRAINED_NODAL_RIGID_BODY: sid / nsid / pnode"""
        idx = self._skip(lines, start)
        if idx >= len(lines) or lines[idx].strip().startswith("*"):
            return idx
        parts = lines[idx].split()
        idx += 1
        if len(parts) < 3:
            while idx < len(lines) and not lines[idx].strip().startswith("*"):
                idx += 1
            return idx
        try:
            rbe2_id  = int(parts[0])
            set_id   = int(parts[1])   # node set ID for slaves
            master   = int(parts[2])   # master node ID
            # Slave nodes from the set (may not be parsed yet; store as-is)
            slaves = model.node_sets.get(set_id, None)
            slave_ids = slaves.node_ids if slaves else []
            model.add_rbe2(rbe2_id, master, slave_ids)
        except (ValueError, IndexError):
            pass
        while idx < len(lines) and not lines[idx].strip().startswith("*"):
            idx += 1
        return idx

    # ------------------------------------------------------------------
    # Post-parse property resolution
    # ------------------------------------------------------------------

    def _resolve_props(
        self,
        model: WHTMeshModel,
        _parts: dict,
        _sections: dict,
        _mats: dict,
    ) -> None:
        """Build WHTProperty and WHTMaterial from parsed *PART, *SECTION, *MAT data."""
        for pid, pdata in _parts.items():
            mid   = pdata.get("mid", 0)
            secid = pdata.get("secid", pid)
            t     = _sections.get(secid, _sections.get(pid, 1.0))
            model.add_property(pid, "PSHELL", t, mid)

        for mid, mdata in _mats.items():
            model.add_material(
                mid,
                E=mdata.get("E", 210000.0),
                nu=mdata.get("nu", 0.3),
                rho=mdata.get("rho", 7.85e-9),
            )
