"""
tests/test_exporters.py
=======================
Unit tests for VTKHDFExporter, VTUPVDExporter, HWASCIIExporter.

h5py is an optional dependency; VTKHDF tests are skipped if not installed.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wht_exporters import HWASCIIExporter, VTKHDFExporter, VTUPVDExporter
from wht_models import WHTExportWarning, WHTMetadata, WHTResultData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_meta(analysis_type="static"):
    return WHTMetadata(
        solver_name="TestSolver", solver_version="0.0",
        analysis_type=analysis_type,
        coordinate_system="cartesian",
        unit_length="m", unit_force="N",
    )


def make_data(T=2, N=4, M=2, analysis_type="static"):
    """
    Minimal WHTResultData with:
     - 4 nodes forming 2 triangles
     - Displacement in point_data
     - Stress in cell_data (6 components)
    """
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    conn  = np.array([0, 1, 2, 1, 2, 3], dtype=np.int64)
    offs  = np.array([0, 3, 6],           dtype=np.int64)
    types = np.array([5, 5],              dtype=np.uint8)   # VTK_TRIANGLE

    # time axis
    if analysis_type == "modal":
        t_vals = np.linspace(1.0, T * 1.0, T)
    else:
        t_vals = np.linspace(0.0, (T - 1) * 0.1, T)

    disp   = np.random.randn(T, N, 3).astype(np.float32)
    stress = np.random.randn(T, M, 6).astype(np.float32)

    return WHTResultData(
        nodes=nodes,
        connectivity=conn, offsets=offs, cell_types=types,
        point_data={"Displacement": disp},
        cell_data={"Stress": stress},
        time_values=t_vals,
        metadata=make_meta(analysis_type),
    )


# ===========================================================================
# VTKHDFExporter
# ===========================================================================

h5py_available = pytest.importorskip("h5py", reason="h5py not installed")


class TestVTKHDFExporter:

    def test_file_created(self):
        import h5py
        data = make_data(T=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.hdf")
            VTKHDFExporter(transient_geometry=False).export(data, path)
            assert os.path.exists(path)

    def test_hdf5_schema_static(self):
        import h5py
        data = make_data(T=3, N=4, M=2)
        T, N, M = data.n_timesteps, data.n_nodes, data.n_cells
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.hdf")
            VTKHDFExporter(transient_geometry=False).export(data, path)
            with h5py.File(path, "r") as f:
                grp = f["VTKHDF"]
                # Static geometry
                assert grp["Points"].shape       == (N, 3)
                assert grp["Connectivity"].shape == (len(data.connectivity),)
                assert grp["Offsets"].shape      == (M + 1,)
                assert grp["Types"].shape        == (M,)
                assert grp["NumberOfPoints"].shape          == (1,)
                assert grp["NumberOfCells"].shape           == (1,)
                assert grp["NumberOfConnectivityIds"].shape == (1,)
                # Steps
                assert grp["Steps/Values"].shape == (T,)
                assert grp["Steps"].attrs["NSteps"] == T
                assert "PointOffsets" not in grp["Steps"] # Omitted for static geometry
                # PointData
                assert "Displacement" in grp["PointData"]
                assert grp["PointData/Displacement"].shape == (T * N, 3)
                # CellData
                assert "Stress" in grp["CellData"]
                assert grp["CellData/Stress"].shape == (T * M, 6)

    def test_hdf5_schema_transient(self):
        import h5py
        data = make_data(T=3, N=4, M=2)
        T, N, M = data.n_timesteps, data.n_nodes, data.n_cells
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out_transient.hdf")
            VTKHDFExporter(transient_geometry=True).export(data, path)
            with h5py.File(path, "r") as f:
                grp = f["VTKHDF"]
                # Transient geometry with Static Topology Optimization
                assert grp["Points"].shape       == (T * N, 3)
                assert grp["Connectivity"].shape == (len(data.connectivity),)
                assert grp["Offsets"].shape      == (M + 1,)
                assert grp["Types"].shape        == (M,)
                assert grp["NumberOfPoints"].shape          == (T,)
                assert grp["NumberOfCells"].shape           == (T,)
                assert grp["NumberOfConnectivityIds"].shape == (T,)
                # Steps
                assert grp["Steps/Values"].shape == (T,)
                assert grp["Steps"].attrs["NSteps"] == T
                assert grp["Steps/PointOffsets"].shape == (T,)
                assert grp["Steps/CellOffsets"].shape  == (T,)
                assert grp["Steps/ConnectivityIdOffsets"].shape == (T,)
                # PointData
                assert "Displacement" in grp["PointData"]
                assert grp["PointData/Displacement"].shape == (T * N, 3)

    def test_values_roundtrip(self):
        import h5py
        data = make_data(T=2, N=4, M=2)
        disp_original = data.point_data["Displacement"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.hdf")
            VTKHDFExporter(transient_geometry=False).export(data, path)
            with h5py.File(path, "r") as f:
                disp_read = f["VTKHDF/PointData/Displacement"][:]
                # (T*N, 3) → (T, N, 3)
                T, N = data.n_timesteps, data.n_nodes
                disp_restored = disp_read.reshape(T, N, 3)
                np.testing.assert_allclose(
                    disp_restored, disp_original, atol=1e-5
                )

    def test_no_compression(self):
        import h5py
        data = make_data(T=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out_nocomp.hdf")
            VTKHDFExporter(compression=None, transient_geometry=False).export(data, path)
            assert os.path.exists(path)

    def test_output_dir_created(self):
        import h5py
        data = make_data(T=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "out.hdf")
            VTKHDFExporter(transient_geometry=False).export(data, path)
            assert os.path.exists(path)


# ===========================================================================
# VTUPVDExporter
# ===========================================================================

class TestVTUPVDExporter:

    def test_pvd_created(self):
        data = make_data(T=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            pvd_path = os.path.join(tmpdir, "out.pvd")
            VTUPVDExporter().export(data, pvd_path)
            assert os.path.exists(pvd_path)

    def test_vtu_files_created(self):
        data = make_data(T=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            pvd_path = os.path.join(tmpdir, "out.pvd")
            VTUPVDExporter().export(data, pvd_path)
            vtu_files = list(Path(tmpdir).glob("*.vtu"))
            assert len(vtu_files) == data.n_timesteps

    def test_pvd_xml_structure(self):
        import xml.etree.ElementTree as ET
        data = make_data(T=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            pvd_path = os.path.join(tmpdir, "out.pvd")
            VTUPVDExporter().export(data, pvd_path)
            tree = ET.parse(pvd_path)
            root = tree.getroot()
            assert root.tag == "VTKFile"
            datasets = root.findall("./Collection/DataSet")
            assert len(datasets) == data.n_timesteps

    def test_vtu_xml_structure(self):
        import xml.etree.ElementTree as ET
        data = make_data(T=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            pvd_path = os.path.join(tmpdir, "out.pvd")
            VTUPVDExporter().export(data, pvd_path)
            vtu_path = str(list(Path(tmpdir).glob("*.vtu"))[0])
            tree     = ET.parse(vtu_path)
            root     = tree.getroot()
            assert root.attrib["type"] == "UnstructuredGrid"


# ===========================================================================
# HWASCIIExporter
# ===========================================================================

class TestHWASCIIExporter:

    def test_file_created(self):
        data = make_data(T=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.ascii")
            HWASCIIExporter().export(data, path)
            assert os.path.exists(path)

    def test_header_present(self):
        data = make_data(T=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.ascii")
            HWASCIIExporter().export(data, path)
            content = Path(path).read_text()
            assert "$ALTAIR_ASCII_RESULT" in content
            assert "$ANALYSIS_TYPE" in content
            assert "$NODES" in content

    def test_time_blocks(self):
        T = 3
        data = make_data(T=T)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.ascii")
            HWASCIIExporter().export(data, path)
            content = Path(path).read_text()
            assert content.count("$TIME") == T
            assert content.count("$END_TIME") == T

    def test_displacement_block(self):
        data = make_data(T=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.ascii")
            HWASCIIExporter().export(data, path)
            content = Path(path).read_text()
            assert "$RESULT_TYPE Displacement" in content

    def test_stress_block(self):
        data = make_data(T=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.ascii")
            HWASCIIExporter().export(data, path)
            content = Path(path).read_text()
            assert "$RESULT_TYPE Stress" in content

    def test_modal_writes_eigen_block(self):
        data = make_data(T=2, analysis_type="modal")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.ascii")
            HWASCIIExporter().export(data, path)
            content = Path(path).read_text()
            assert "$RESULT_TYPE Eigen" in content

    def test_unsupported_field_warns(self):
        data = make_data(T=1)
        # Inject an unsupported field
        data.point_data["UnknownResult"] = np.zeros((1, data.n_nodes, 3))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.ascii")
            with pytest.warns(WHTExportWarning, match="not a supported"):
                HWASCIIExporter().export(data, path)

    def test_correct_node_count_in_block(self):
        N = 4
        data = make_data(T=1, N=N)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.ascii")
            HWASCIIExporter().export(data, path)
            content = Path(path).read_text()
            # Count lines that start with a node ID (8-char right-justified int)
            disp_lines = [
                ln for ln in content.splitlines()
                if ln.strip() and ln.strip()[0].isdigit()
                   and not ln.strip().startswith("$")
            ]
            # T=1 → N lines for Displacement + M lines for Stress
            assert len(disp_lines) == N + data.n_cells
