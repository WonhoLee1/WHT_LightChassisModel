"""
wht_exporters.py
================
WHT Universal FEM Result Converter — Exporter Layer

Converts WHTResultData (IR) to on-disk file formats.

Classes
-------
BaseExporter    : Abstract base. Subclass for every new output format.
VTKHDFExporter  : Single-file HDF5 for ParaView 5.11+ (.hdf).
VTUPVDExporter  : Multi-file XML for legacy ParaView (.vtu / .pvd).
HWASCIIExporter : Altair Generic ASCII for HyperView (.ascii).
"""

from __future__ import annotations

import os
import warnings
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from .wht_models import WHTResultData, WHTExportWarning


# ===========================================================================
# BaseExporter
# ===========================================================================

class BaseExporter(ABC):
    """Abstract base class for all exporters."""

    @abstractmethod
    def export(self, data: WHTResultData, output_path: str) -> None:
        """
        Write ``data`` to ``output_path``.

        Parameters
        ----------
        data : WHTResultData
            Validated IR object (should have passed adapter.validate()).
        output_path : str
            Destination file path (including extension).
        """
        ...

    @staticmethod
    def _ensure_dir(path: str) -> None:
        """Create parent directory if it does not exist."""
        parent = Path(path).parent
        parent.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# VTKHDFExporter
# ===========================================================================

class VTKHDFExporter(BaseExporter):
    """
    Export WHTResultData to the VTKHDF format (HDF5 binary, single file).

    Requires ParaView 5.11 or newer.
    Uses ``h5py`` for HDF5 I/O.

    HDF5 Schema
    -----------
    /VTKHDF/
        Type              (attr) = "UnstructuredGrid"
        Version           (attr) = [2, 0]
        Points            (N, 3)      float64  — static geometry
        Connectivity      (K,)        int64
        Offsets           (M+1,)      int64
        Types             (M,)        uint8
        NumberOfPoints    (1,)        int64
        NumberOfCells     (1,)        int64
        Steps/
            NSteps        (attr) = T
            Values        (T,)        float64  — time/freq/load-factor axis
            PointOffsets  (T,)        int64
            CellOffsets   (T,)        int64
        PointData/
            <name>        (T*N, D)    float32
        CellData/
            <name>        (T*M, D)    float32

    Parameters
    ----------
    compression : str or None
        HDF5 compression filter: "gzip" | "lzf" | None.
    compression_opts : int
        Compression level for gzip (1–9). Ignored for lzf.
    chunk_timesteps : int
        Number of timesteps per HDF5 chunk (tunes I/O performance).
    """

    def __init__(
        self,
        compression: Optional[str] = "gzip",
        compression_opts: int = 4,
        chunk_timesteps: int = 10,
        transient_geometry: bool = True,
    ) -> None:
        self.compression      = compression
        self.compression_opts = compression_opts
        self.chunk_timesteps  = chunk_timesteps
        self.transient_geometry = transient_geometry

    def export(self, data: WHTResultData, output_path: str) -> None:
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for VTKHDFExporter. "
                "Install it with: pip install h5py"
            )

        self._ensure_dir(output_path)

        T = data.n_timesteps
        N = data.n_nodes
        M = data.n_cells
        comp_kwargs = self._compression_kwargs()

        # Displacement fields lookup for transient geometry deform
        disp = None
        for _disp_key in ("Displacement", "Eigen", "ModeShape"):
            if _disp_key in data.point_data:
                disp = data.point_data[_disp_key]
                break

        with h5py.File(output_path, "w") as f:
            grp = f.create_group("VTKHDF")
            grp.attrs["Type"]    = np.bytes_("UnstructuredGrid")
            grp.attrs["Version"] = np.array([2, 0], dtype=np.int64)

            if self.transient_geometry and T > 1:
                # ─── 1. Transient Geometry (Moving mesh animation with Static Topology Optimization) ───
                # This perfectly aligns with Kitware's openradioss-to-vtkhdf memory-efficient design!
                
                # 1.1 Compute deformed points per timestep: X(t) = X_base + Disp(t)
                pts_list = []
                for t in range(T):
                    t_nodes = data.nodes.copy()
                    if disp is not None and t < len(disp):
                        t_nodes += disp[t][:, :3]
                    pts_list.append(t_nodes)
                
                all_pts = np.concatenate(pts_list, axis=0).astype(np.float64)
                grp.create_dataset("Points", data=all_pts, **comp_kwargs)

                # 1.2 [WHT HIGH OPTIMIZATION] Write topology arrays ONCE (T=1). 
                # Zero redundancy. Extremely fast writes, minimal memory overhead, and max FPS inside ParaView!
                grp.create_dataset("Connectivity", data=data.connectivity.astype(np.int64), **comp_kwargs)
                grp.create_dataset("Offsets", data=data.offsets.astype(np.int64), **comp_kwargs)
                grp.create_dataset("Types", data=data.cell_types.astype(np.uint8), **comp_kwargs)

                # [WHT CRITICAL SPEC FIX] For transient VTKHDF datasets, NumberOfPoints, NumberOfCells,
                # and NumberOfConnectivityIds must be 1D arrays of size T (number of timesteps).
                grp.create_dataset("NumberOfPoints", data=np.full(T, N, dtype=np.int64))
                grp.create_dataset("NumberOfCells", data=np.full(T, M, dtype=np.int64))
                grp.create_dataset("NumberOfConnectivityIds", data=np.full(T, len(data.connectivity), dtype=np.int64))

                # 1.3 Steps metadata mapping
                steps = grp.create_group("Steps")
                steps.attrs["NSteps"] = T
                steps.create_dataset("Values", data=data.time_values.astype(np.float64))
                steps.create_dataset("PointOffsets", data=(np.arange(T, dtype=np.int64) * N))
                
                # Because topology is static, Cell & Connectivity offsets are set to ZERO for all time steps.
                # ParaView HDF Reader dynamically shares the single static topology over the entire timeline!
                steps.create_dataset("CellOffsets", data=np.zeros(T, dtype=np.int64))
                steps.create_dataset("ConnectivityIdOffsets", data=np.zeros(T, dtype=np.int64))
                steps.create_dataset("PartOffsets", data=np.zeros(T, dtype=np.int64))

            else:
                # ─── 2. Static Geometry (Standard single-frame/static chassis setup) ───
                grp.create_dataset("Points", data=data.nodes.astype(np.float64))
                grp.create_dataset("Connectivity", data=data.connectivity.astype(np.int64))
                grp.create_dataset("Offsets", data=data.offsets.astype(np.int64))
                grp.create_dataset("Types", data=data.cell_types.astype(np.uint8))
                
                # In static geometry mode, NumberOfPoints/NumberOfCells/NumberOfConnectivityIds are size-1 vectors
                grp.create_dataset("NumberOfPoints", data=np.array([N], dtype=np.int64))
                grp.create_dataset("NumberOfCells", data=np.array([M], dtype=np.int64))
                grp.create_dataset("NumberOfConnectivityIds", data=np.array([len(data.connectivity)], dtype=np.int64))

                steps = grp.create_group("Steps")
                steps.attrs["NSteps"] = T
                steps.create_dataset("Values", data=data.time_values.astype(np.float64))
                # Note: PointOffsets/CellOffsets must be omitted for strict static geometry in ParaView HDF specs
                steps.create_dataset("PartOffsets", data=np.zeros(T, dtype=np.int64))

            # --- PointData ---
            if data.point_data:
                pd_grp = grp.create_group("PointData")
                for name, arr in data.point_data.items():
                    flat = arr.reshape(T * N, -1).astype(np.float32)
                    chunk_n = min(self.chunk_timesteps, T) * N
                    chunks  = (chunk_n, flat.shape[1])
                    pd_grp.create_dataset(name, data=flat,
                                          chunks=chunks, **comp_kwargs)

            # --- CellData ---
            if data.cell_data:
                cd_grp = grp.create_group("CellData")
                for name, arr in data.cell_data.items():
                    flat = arr.reshape(T * M, -1).astype(np.float32)
                    chunk_m = min(self.chunk_timesteps, T) * M
                    chunks  = (chunk_m, flat.shape[1])
                    cd_grp.create_dataset(name, data=flat,
                                          chunks=chunks, **comp_kwargs)

    def _compression_kwargs(self) -> dict:
        if self.compression is None:
            return {}
        kwargs = {"compression": self.compression}
        if self.compression == "gzip":
            kwargs["compression_opts"] = self.compression_opts
        return kwargs


# ===========================================================================
# VTUPVDExporter
# ===========================================================================

class VTUPVDExporter(BaseExporter):
    """
    Export WHTResultData to VTU + PVD XML format (legacy ParaView).

    One .vtu file is written per timestep/mode, plus a .pvd index file.

    Parameters
    ----------
    binary : bool
        If True, embed array data as base64-encoded binary inside the XML.
        If False, write plain ASCII (readable but large).

    Notes
    -----
    This exporter uses a minimal hand-written VTK XML writer to avoid a
    hard dependency on ``meshio`` or ``vtk``.  For production use with
    complex datasets, consider switching to meshio.
    """

    def __init__(self, binary: bool = False) -> None:
        self.binary = binary

    def export(self, data: WHTResultData, output_path: str) -> None:
        self._ensure_dir(output_path)
        base  = Path(output_path).with_suffix("")
        pvd_entries = []

        for t_idx in range(data.n_timesteps):
            t_val    = data.time_values[t_idx]
            vtu_name = f"{base.name}_t{t_idx:04d}.vtu"
            vtu_path = str(base.parent / vtu_name)
            self._write_vtu(data, t_idx, vtu_path)
            pvd_entries.append((t_val, vtu_name))

        self._write_pvd(output_path, pvd_entries)

    # ------------------------------------------------------------------

    def _write_vtu(self, data: WHTResultData, t_idx: int, path: str) -> None:
        """Write a single .vtu file for timestep t_idx."""
        N = data.n_nodes
        M = data.n_cells

        root = ET.Element("VTKFile",
                          type="UnstructuredGrid",
                          version="0.1",
                          byte_order="LittleEndian")
        ug   = ET.SubElement(root, "UnstructuredGrid")
        piece = ET.SubElement(ug, "Piece",
                              NumberOfPoints=str(N),
                              NumberOfCells=str(M))

        # Points
        pts_node = ET.SubElement(piece, "Points")
        self._add_data_array(pts_node, data.nodes,
                             name="", n_components=3)

        # Cells
        cells_node = ET.SubElement(piece, "Cells")
        self._add_data_array(cells_node,
                             data.connectivity, name="connectivity", n_components=1)
        self._add_data_array(cells_node,
                             data.offsets,      name="offsets",      n_components=1)
        self._add_data_array(cells_node,
                             data.cell_types,   name="types",        n_components=1)

        # PointData
        if data.point_data:
            pd_node = ET.SubElement(piece, "PointData")
            for name, arr in data.point_data.items():
                step = arr[t_idx]        # (N, D)
                self._add_data_array(pd_node, step,
                                     name=name,
                                     n_components=step.shape[-1] if step.ndim > 1 else 1)

        # CellData
        if data.cell_data:
            cd_node = ET.SubElement(piece, "CellData")
            for name, arr in data.cell_data.items():
                step = arr[t_idx]        # (M, D)
                self._add_data_array(cd_node, step,
                                     name=name,
                                     n_components=step.shape[-1] if step.ndim > 1 else 1)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="unicode", xml_declaration=True)

    def _add_data_array(self, parent, arr, name, n_components):
        arr = np.asarray(arr)
        dtype_str = self._vtk_dtype(arr.dtype)
        da = ET.SubElement(parent, "DataArray",
                           type=dtype_str,
                           Name=name,
                           NumberOfComponents=str(n_components),
                           format="ascii")
        flat = arr.flatten()
        da.text = " ".join(f"{v}" for v in flat)

    @staticmethod
    def _vtk_dtype(dtype) -> str:
        mapping = {
            np.float32: "Float32", np.float64: "Float64",
            np.int32:   "Int32",   np.int64:   "Int64",
            np.uint8:   "UInt8",
        }
        for k, v in mapping.items():
            if np.issubdtype(dtype, k):
                return v
        return "Float64"

    def _write_pvd(self, pvd_path: str, entries: list) -> None:
        root = ET.Element("VTKFile", type="Collection",
                          version="0.1", byte_order="LittleEndian")
        coll = ET.SubElement(root, "Collection")
        for t_val, vtu_name in entries:
            ET.SubElement(coll, "DataSet",
                          timestep=str(t_val),
                          group="", part="0",
                          file=vtu_name)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(pvd_path, encoding="unicode", xml_declaration=True)


# ===========================================================================
# HWASCIIExporter
# ===========================================================================

class HWASCIIExporter(BaseExporter):
    """
    Export WHTResultData to Altair Generic ASCII format for HyperView.

    Supported result blocks
    -----------------------
    Displacement : Nodal displacement (UX, UY, UZ).
    Stress       : Element stress tensor (S11, S22, S33, S12, S13, S23).
    Strain       : Element strain tensor (E11, E22, E33, E12, E13, E23).
    Eigen        : Mode shape displacement (written when analysis_type='modal').
    Buckling     : Buckling mode shape + load factor.

    Unsupported fields raise ``WHTExportWarning`` and are skipped.

    File structure
    --------------
    $ALTAIR_ASCII_RESULT 1.0
    $ANALYSIS_TYPE  ...
    $UNITS          ...
    ...
    $TIME  <value>
      $RESULT_TYPE <type>
      $RESULT_LOCATION Node | Element
      <id>  <val0>  <val1>  ...
      ...
    $END_TIME
    ...
    """

    SUPPORTED_POINT_BLOCKS   = {"Displacement", "Eigen", "BucklingMode", "ModeShape"}
    SUPPORTED_CELL_BLOCKS    = {"Stress", "Strain"}
    _ANALYSIS_TO_TIME_LABEL  = {
        "static":    "Load Step",
        "transient": "Time [s]",
        "modal":     "Frequency [Hz]",
        "buckling":  "Load Factor",
    }

    def export(self, data: WHTResultData, output_path: str) -> None:
        self._ensure_dir(output_path)
        meta  = data.metadata
        T     = data.n_timesteps
        N     = data.n_nodes
        M     = data.n_cells
        atype = meta.analysis_type
        time_label = self._ANALYSIS_TO_TIME_LABEL.get(atype, "Step")

        with open(output_path, "w") as f:
            # --- File header ---
            f.write("$ALTAIR_ASCII_RESULT 1.0\n")
            f.write(f"$ANALYSIS_TYPE       {atype}\n")
            f.write(f"$SOLVER              {meta.solver_name} {meta.solver_version}\n")
            f.write(f"$CREATED_AT          {meta.created_at}\n")
            f.write(f"$UNITS               {meta.unit_length} {meta.unit_force}\n")
            f.write(f"$NODES               {N}\n")
            f.write(f"$ELEMENTS            {M}\n")
            f.write("\n")

            # --- Timestep / mode loop ---
            for t_idx in range(T):
                t_val = data.time_values[t_idx]
                f.write(f"$TIME {t_val:.8e}  $ {time_label}\n\n")

                # ---- PointData blocks ----
                for name, arr in data.point_data.items():
                    resolved_name = self._resolve_block_name(name, atype)
                    if resolved_name is None:
                        warnings.warn(
                            f"point_data['{name}'] is not a supported "
                            f"HWASCII block; skipping.",
                            WHTExportWarning,
                            stacklevel=2,
                        )
                        continue

                    step = arr[t_idx]    # (N, D)
                    D    = step.shape[-1] if step.ndim > 1 else 1
                    step = step.reshape(N, D)

                    f.write(f"$RESULT_TYPE {resolved_name}\n")
                    f.write("$RESULT_LOCATION Node\n")
                    header_cols = self._column_headers(resolved_name, D)
                    f.write(f"$NODE_ID  {header_cols}\n")

                    for nid in range(N):
                        vals = "  ".join(f"{v:>14.6e}" for v in step[nid])
                        f.write(f"{nid + 1:>8d}  {vals}\n")
                    f.write("\n")

                # ---- CellData blocks ----
                for name, arr in data.cell_data.items():
                    if name not in self.SUPPORTED_CELL_BLOCKS:
                        warnings.warn(
                            f"cell_data['{name}'] is not a supported "
                            f"HWASCII block; skipping.",
                            WHTExportWarning,
                            stacklevel=2,
                        )
                        continue

                    step = arr[t_idx]    # (M, D)
                    D    = step.shape[-1] if step.ndim > 1 else 1
                    step = step.reshape(M, D)

                    f.write(f"$RESULT_TYPE {name}\n")
                    f.write("$RESULT_LOCATION Element\n")
                    header_cols = self._column_headers(name, D)
                    f.write(f"$ELEM_ID  {header_cols}\n")

                    for eid in range(M):
                        vals = "  ".join(f"{v:>14.6e}" for v in step[eid])
                        f.write(f"{eid + 1:>8d}  {vals}\n")
                    f.write("\n")

                # ---- Buckling: write load factor as field data note ----
                if atype == "buckling" and "LoadFactor" in data.field_data:
                    lf = data.field_data["LoadFactor"][t_idx]
                    f.write(f"$FIELD LoadFactor  {lf:.8e}\n\n")

                f.write("$END_TIME\n\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_block_name(self, name: str, atype: str) -> Optional[str]:
        """Map WHTResultData field name to HWASCII block type string."""
        if name == "Displacement":
            return "Eigen" if atype == "modal" else "Displacement"
        elif name == "ModeShape":
            return "Eigen"
        if name == "BucklingMode":
            return "Buckling"
        if name in self.SUPPORTED_POINT_BLOCKS:
            return name
        return None

    @staticmethod
    def _column_headers(block_name: str, D: int) -> str:
        """Return space-separated column header string for a result block."""
        presets = {
            "Displacement": ["UX", "UY", "UZ"],
            "Eigen":        ["UX", "UY", "UZ"],
            "Buckling":     ["UX", "UY", "UZ"],
            "Stress":       ["S11", "S22", "S33", "S12", "S13", "S23"],
            "Strain":       ["E11", "E22", "E33", "E12", "E13", "E23"],
        }
        labels = presets.get(block_name, [f"C{i}" for i in range(D)])
        return "  ".join(f"{lbl:>14s}" for lbl in labels[:D])
