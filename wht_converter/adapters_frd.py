"""
adapters_frd.py
===============
FrdAdapter: Parse CalculiX .frd result files into WHTResultData IR.

Usage::
    from wht_converter.adapters_frd import FrdAdapter
    from wht_converter.wht_exporters import VTKHDFExporter

    adapter = FrdAdapter()
    data    = adapter.convert("result.frd")
    VTKHDFExporter().export(data, "result.hdf")

FRD Format Reference
--------------------
CalculiX FRD is a line-based format derived from IDEAS.

Key sections:
    2C   -- Nodal coordinates
    3C   -- Element connectivity
    100C -- Nodal results (one block per timestep)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from wht_converter.wht_models import (
    WHTMetadata,
    WHTResultData,
    WHTValidationError,
)
from wht_converter.wht_utils import (
    VTKCellType,
    to_vtk_csr,
    merge_csr,
)

_FRD_TO_VTK_TYPE = {
    1:  VTKCellType.HEXAHEDRON,
    2:  25,   # C3D20R
    3:  25,   # C3D20
    4:  12,   # C3D8R
    5:  VTKCellType.TETRA,
    6:  28,   # C3D10
    7:  24,   # C3D15
    8:  VTKCellType.WEDGE,
    9:  VTKCellType.QUAD,
    10: VTKCellType.TRIANGLE,
    11: VTKCellType.LINE,
}


class FrdAdapter:
    """
    Adapter: CalculiX .frd -> WHTResultData

    Parses nodal coordinates, element connectivity (HEX8, etc.), and
    transient displacement results from a single .frd file.
    """

    def __init__(self, f64_precision: bool = False) -> None:
        self.dtype = np.float64 if f64_precision else np.float32

    def convert(
        self,
        frd_path: str | Path,
        metadata: Optional[WHTMetadata] = None,
    ) -> WHTResultData:
        path = Path(frd_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"FRD file not found: {path}")

        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()

        nodes, id_map = self._parse_nodes(lines)
        connectivity, offsets, cell_types = self._parse_elements(lines, id_map)
        point_data, time_values = self._parse_results(lines, len(nodes))

        if metadata is None:
            metadata = WHTMetadata(
                solver_name="CalculiX",
                solver_version="2.23",
                analysis_type="transient",
                coordinate_system="cartesian",
                unit_length="mm",
                unit_force="N",
            )

        return WHTResultData(
            nodes=nodes,
            connectivity=connectivity,
            offsets=offsets,
            cell_types=cell_types,
            point_data=point_data,
            cell_data={},
            field_data={},
            time_values=time_values,
            metadata=metadata,
        )

    def _parse_nodes(self, lines: list[str]) -> tuple[np.ndarray, dict[int, int]]:
        nodes_raw: dict[int, list[float]] = {}
        i = 0
        n = len(lines)
        while i < n:
            if "2C" in lines[i][4:8]:
                i += 1
                while i < n:
                    line = lines[i]
                    if " -1" in line or line.startswith(" -1"):
                        nid = int(line[3:13])
                        x = float(line[13:25])
                        y = float(line[25:37])
                        z = float(line[37:49])
                        nodes_raw[nid] = [x, y, z]
                    elif " -3" in line or line.startswith(" -3"):
                        break
                    i += 1
                break
            i += 1

        if not nodes_raw:
            raise WHTValidationError("No nodes found (missing 2C section).")

        sorted_ids = sorted(nodes_raw.keys())
        id_map = {nid: idx for idx, nid in enumerate(sorted_ids)}
        coords = np.array([nodes_raw[nid] for nid in sorted_ids], dtype=np.float64)
        return coords, id_map

    def _parse_elements(
        self, lines: list[str], id_map: dict[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        elem_list: list[tuple[int, list[int]]] = []

        i = 0
        n = len(lines)
        while i < n:
            if "3C" in lines[i][4:8]:
                i += 1
                while i < n:
                    line = lines[i]
                    if " -1" in line or line.startswith(" -1"):
                        typ_raw = line[13:18].strip()
                        if not typ_raw:
                            i += 1
                            continue
                        etype = int(typ_raw)
                        node_ids_1b: list[int] = []
                        i += 1
                        while i < n and lines[i].startswith(" -2"):
                            seg = lines[i]
                            for off in range(3, min(len(seg), 103), 10):
                                val = seg[off:off+10].strip()
                                if val:
                                    node_ids_1b.append(int(val))
                            i += 1
                        mapped = [id_map[nid] for nid in node_ids_1b if nid in id_map]
                        vtk_type = _FRD_TO_VTK_TYPE.get(etype, VTKCellType.HEXAHEDRON)
                        elem_list.append((vtk_type, mapped))
                        continue
                    elif " -3" in line or line.startswith(" -3"):
                        break
                    i += 1
                break
            i += 1

        if not elem_list:
            raise WHTValidationError("No elements found (missing 3C section).")

        groups: dict[int, list[list[int]]] = defaultdict(list)
        for vtk_type, node_list in elem_list:
            groups[vtk_type].append(node_list)

        csr_groups = []
        for vtk_type, conn_list in groups.items():
            conn_array = np.array(conn_list, dtype=np.int64)
            csr_groups.append(to_vtk_csr(conn_array, vtk_type))

        return csr_groups[0] if len(csr_groups) == 1 else merge_csr(csr_groups)

    def _parse_results(
        self, lines: list[str], n_nodes: int
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Parse 100C result blocks into per-field time series."""
        hundred_c_indices = [
            idx for idx, line in enumerate(lines)
            if line.strip().startswith("100CL")
        ]
        if not hundred_c_indices:
            raise WHTValidationError("No 100C result blocks found in FRD file.")

        field_blocks: dict[str, list[tuple[int, float]]] = {}
        for hidx in hundred_c_indices:
            parts = lines[hidx].strip().split()
            time_val = float(parts[2]) if len(parts) > 2 else 0.0
            j = hidx + 1
            field_name = None
            while j < len(lines):
                lj = lines[j]
                if lj.strip().startswith("100CL"):
                    break
                if "  -4" in lj or lj.strip().startswith("-4"):
                    field_name = lj[5:17].strip()
                    break
                j += 1
            if field_name:
                field_blocks.setdefault(field_name, []).append((hidx, time_val))

        target_fields = ["DISP", "Displacement", "STRESS", "TOSTRAIN", "STRAIN", "ERROR"]
        available = [f for f in target_fields if f in field_blocks]
        if not available:
            available = list(field_blocks.keys())

        field_ncomp: dict[str, int] = {}
        for fname in available:
            hidx = field_blocks[fname][0][0]
            j = hidx + 1
            while j < len(lines):
                lj = lines[j]
                if lj.strip().startswith("100CL"):
                    break
                if " -1" in lj:
                    # Detect actual component count from data, not -4 declaration
                    # (DISP declares 4 but stores 3; ALL magnitude is computed)
                    vals = []
                    for off in range(13, min(len(lj), 85), 12):
                        chunk = lj[off:off+12].strip()
                        if chunk:
                            try:
                                float(chunk)
                                vals.append(chunk)
                            except ValueError:
                                break
                    field_ncomp[fname] = len(vals)
                    break
                j += 1
            if fname not in field_ncomp:
                field_ncomp[fname] = 3

        point_data = {}
        all_times = []

        for fname in available:
            blocks = field_blocks[fname]
            ncomp = field_ncomp.get(fname, 3)
            field_data_list = []
            field_times = []
            for hidx, time_val in blocks:
                field_times.append(time_val)
                arr = np.zeros((n_nodes, ncomp), dtype=np.float64)
                j = hidx + 1
                while j < len(lines):
                    lj = lines[j]
                    if lj.strip().startswith("100CL"):
                        break
                    if "  -4" in lj and j > hidx + 1:
                        break
                    if " -1" in lj or lj.strip().startswith("-1"):
                        nid = int(lj[3:13])
                        vals = []
                        for offset in range(13, min(len(lj), 85), 12):
                            chunk = lj[offset:offset+12].strip()
                            if chunk:
                                try:
                                    vals.append(float(chunk))
                                except ValueError:
                                    break
                        if vals:
                            idx = nid - 1
                            if 0 <= idx < n_nodes:
                                arr[idx] = vals[:ncomp]
                    elif " -3" in lj or lj.strip().startswith("-3"):
                        break
                    j += 1
                field_data_list.append(arr)

            if field_data_list:
                stack = np.stack(field_data_list, axis=0).astype(self.dtype)
                point_data[_FRD_RESULT_MAP.get(fname, fname)] = stack
                if fname == "DISP":
                    all_times = field_times

        if not all_times and available:
            all_times = [t for _, t in field_blocks[available[0]]]

        tv = np.array(all_times, dtype=np.float64) if all_times else np.array([0.0])
        return point_data, tv


_FRD_RESULT_MAP = {
    "DISP": "Displacement",
    "STRESS": "Stress",
    "TOSTRAIN": "Strain",
    "ERROR": "Error",
}


if __name__ == "__main__":
    import sys
    from wht_converter.wht_exporters import VTKHDFExporter

    if len(sys.argv) < 2:
        print("Usage: python -m wht_converter.adapters_frd <input.frd> [output.hdf]")
        sys.exit(1)

    frd_path = sys.argv[1]
    hdf_path = sys.argv[2] if len(sys.argv) > 2 else frd_path.replace(".frd", ".hdf")

    print(f"\nFrdAdapter v0.1")
    print(f"  Input  : {frd_path}")
    print(f"  Output : {hdf_path}")

    adapter = FrdAdapter(f64_precision=False)
    data = adapter.convert(frd_path)

    print(f"  Nodes    : {data.n_nodes}")
    print(f"  Elements : {data.n_cells}")
    print(f"  Steps    : {data.n_timesteps}")
    print(f"  PointData: {list(data.point_data.keys())}")
    for k, v in data.point_data.items():
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
    print(f"  Time     : {data.time_values[0]:.6f} .. {data.time_values[-1]:.6f}")
    print()

    exporter = VTKHDFExporter(compression="gzip", compression_opts=4)
    exporter.export(data, hdf_path)
    print(f"Done: {hdf_path}")
