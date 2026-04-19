"""
lsdyna_writer.py
================
LS-DYNA keyword file writer (.k)

Writes WHTMeshModel to LS-DYNA format.
Outputs: *NODE, *ELEMENT_SHELL, *ELEMENT_BEAM, *PART, *SECTION_SHELL,
         *MAT_ELASTIC, *SET_NODE_LIST, *END
"""

from __future__ import annotations

from pathlib import Path
from typing import TextIO

from .base_writer import BaseFEMWriter
from ..wht_mesh_model import WHTMeshModel


class LSDYNAWriter(BaseFEMWriter):
    """Writer for LS-DYNA keyword files (.k)."""

    def write(self, model: WHTMeshModel, file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as fh:
            self._write_header(fh, model)
            self._write_nodes(fh, model)
            self._write_elements(fh, model)
            self._write_sets(fh, model)
            self._write_parts(fh, model)
            self._write_sections(fh, model)
            self._write_materials(fh, model)
            fh.write("*END\n")

    # ------------------------------------------------------------------

    def _write_header(self, fh: TextIO, model: WHTMeshModel) -> None:
        fh.write("*KEYWORD\n")
        fh.write(f"$ WHT FEM Framework — wht_modeler LSDYNAWriter\n")
        fh.write(f"$ Model: {model.name}\n")
        fh.write(f"$ Nodes: {model.n_nodes}  Elements: {model.n_elements}\n")

    def _write_nodes(self, fh: TextIO, model: WHTMeshModel) -> None:
        fh.write("*NODE\n")
        fh.write("$#   nid               x               y               z\n")
        for nid in sorted(model.nodes.keys()):
            n = model.nodes[nid]
            fh.write(f"{nid:>8}{n.x:>16.8e}{n.y:>16.8e}{n.z:>16.8e}\n")

    def _write_elements(self, fh: TextIO, model: WHTMeshModel) -> None:
        # Split by type
        shells = {eid: e for eid, e in model.elements.items()
                  if e.type in ("QUAD4", "TRIA3")}
        beams  = {eid: e for eid, e in model.elements.items()
                  if e.type == "BEAM2"}

        if shells:
            fh.write("*ELEMENT_SHELL\n")
            fh.write("$#   eid      pid       n1       n2       n3       n4\n")
            for eid in sorted(shells.keys()):
                e   = shells[eid]
                nids = e.node_ids
                # Pad to 4 nodes for TRIA3 (repeat last node)
                while len(nids) < 4:
                    nids = nids + [nids[-1]]
                fh.write(
                    f"{eid:>8}{e.pid:>8}"
                    f"{nids[0]:>8}{nids[1]:>8}{nids[2]:>8}{nids[3]:>8}\n"
                )

        if beams:
            fh.write("*ELEMENT_BEAM\n")
            fh.write("$#   eid      pid       n1       n2\n")
            for eid in sorted(beams.keys()):
                e = beams[eid]
                fh.write(f"{eid:>8}{e.pid:>8}{e.node_ids[0]:>8}{e.node_ids[1]:>8}\n")

    def _write_sets(self, fh: TextIO, model: WHTMeshModel) -> None:
        for sid in sorted(model.node_sets.keys()):
            ns = model.node_sets[sid]
            fh.write("*SET_NODE_LIST\n")
            fh.write(f"$#    sid\n")
            fh.write(f"{sid:>10}\n")
            fh.write("$#   nid1     nid2     nid3     nid4     nid5     nid6     nid7     nid8\n")
            nids = ns.node_ids
            for i in range(0, len(nids), 8):
                chunk = nids[i:i+8]
                fh.write("".join(f"{n:>10}" for n in chunk) + "\n")

    def _write_parts(self, fh: TextIO, model: WHTMeshModel) -> None:
        for pid in sorted(model.properties.keys()):
            prop = model.properties[pid]
            fh.write("*PART\n")
            fh.write(f"$# title\n")
            fh.write(f"Part_{pid}\n")
            fh.write(f"$#   pid    secid      mid\n")
            fh.write(f"{pid:>8}{pid:>8}{prop.mid:>8}\n")

    def _write_sections(self, fh: TextIO, model: WHTMeshModel) -> None:
        written = set()
        for pid in sorted(model.properties.keys()):
            prop = model.properties[pid]
            if prop.type == "PSHELL" and pid not in written:
                fh.write("*SECTION_SHELL\n")
                fh.write("$#  secid    elform      shrf\n")
                fh.write(f"{pid:>8}{2:>8}{1.0:>8.3f}\n")
                fh.write("$#      t1        t2        t3        t4\n")
                fh.write(f"{prop.t:>10.4f}{prop.t:>10.4f}{prop.t:>10.4f}{prop.t:>10.4f}\n")
                written.add(pid)

    def _write_materials(self, fh: TextIO, model: WHTMeshModel) -> None:
        for mid in sorted(model.materials.keys()):
            mat = model.materials[mid]
            fh.write("*MAT_ELASTIC\n")
            fh.write("$#   mid       ro         e        pr\n")
            fh.write(
                f"{mid:>8}  {mat.rho:>12.6e}  {mat.E:>12.6e}  {mat.nu:>10.4f}\n"
            )
