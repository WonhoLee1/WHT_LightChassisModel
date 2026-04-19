"""
tests/test_models_and_utils.py
==============================
Unit tests for wht_models.py and wht_utils.py
"""

import warnings

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wht_models import (
    WHTMetadata,
    WHTResultData,
    WHTValidationError,
    WHTExportWarning,
)
from wht_utils import (
    VTKCellType,
    merge_csr,
    node_dict_to_array,
    remap_connectivity,
    to_vtk_csr,
)


# ===========================================================================
# WHTMetadata
# ===========================================================================

class TestWHTMetadata:

    def test_valid_creation(self):
        meta = WHTMetadata(
            solver_name="JaxSSO", solver_version="0.1",
            analysis_type="modal", coordinate_system="cartesian",
            unit_length="m", unit_force="N",
        )
        assert meta.solver_name == "JaxSSO"
        assert meta.created_at != ""   # auto-filled

    def test_invalid_analysis_type(self):
        with pytest.raises(WHTValidationError, match="analysis_type"):
            WHTMetadata(
                solver_name="JaxSSO", solver_version="0.1",
                analysis_type="dynamic",        # invalid
                coordinate_system="cartesian",
                unit_length="m", unit_force="N",
            )

    def test_invalid_unit_length(self):
        with pytest.raises(WHTValidationError, match="unit_length"):
            WHTMetadata(
                solver_name="JaxSSO", solver_version="0.1",
                analysis_type="static",
                coordinate_system="cartesian",
                unit_length="cm",               # invalid
                unit_force="N",
            )


# ===========================================================================
# to_vtk_csr
# ===========================================================================

class TestToVtkCsr:

    def test_quad_mesh(self):
        quads = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        conn, offs, types = to_vtk_csr(quads, VTKCellType.QUAD)
        assert list(conn)  == [0, 1, 2, 3, 4, 5, 6, 7]
        assert list(offs)  == [0, 4, 8]
        assert list(types) == [VTKCellType.QUAD, VTKCellType.QUAD]

    def test_line_mesh(self):
        lines = np.array([[0, 1], [1, 2]])
        conn, offs, types = to_vtk_csr(lines, VTKCellType.LINE)
        assert list(offs) == [0, 2, 4]
        assert all(t == VTKCellType.LINE for t in types)

    def test_1d_input_raises(self):
        with pytest.raises(ValueError):
            to_vtk_csr(np.array([0, 1, 2, 3]), VTKCellType.QUAD)


# ===========================================================================
# merge_csr
# ===========================================================================

class TestMergeCsr:

    def test_merge_line_and_quad(self):
        lines = to_vtk_csr(np.array([[0, 1]]),        VTKCellType.LINE)
        quads = to_vtk_csr(np.array([[0, 1, 2, 3]]),  VTKCellType.QUAD)
        conn, offs, types = merge_csr([lines, quads])
        # 1 line (2 nodes) + 1 quad (4 nodes) = 6 total
        assert len(conn)  == 6
        assert len(offs)  == 3    # M+1 = 2+1
        assert list(types) == [VTKCellType.LINE, VTKCellType.QUAD]
        assert list(offs) == [0, 2, 6]

    def test_empty_groups_raises(self):
        with pytest.raises(ValueError):
            merge_csr([])


# ===========================================================================
# node_dict_to_array / remap_connectivity
# ===========================================================================

class TestNodeUtils:

    def test_node_dict_to_array(self):
        nd = {10: [1.0, 0.0, 0.0], 20: [0.0, 0.0, 0.0]}
        arr, mp = node_dict_to_array(nd)
        assert arr.shape == (2, 3)
        assert mp[10] == 0
        assert mp[20] == 1

    def test_remap_connectivity(self):
        nd = {10: [0, 0, 0], 20: [1, 0, 0], 30: [1, 1, 0], 40: [0, 1, 0]}
        _, id_map = node_dict_to_array(nd)
        elems = {1: [10, 20, 30, 40]}
        remapped = remap_connectivity(elems, id_map)
        assert remapped.shape == (1, 4)
        assert list(remapped[0]) == [0, 1, 2, 3]


# ===========================================================================
# WHTResultData
# ===========================================================================

def make_minimal_data(T=1, N=4, M=2):
    """Helper: build a minimal valid WHTResultData."""
    meta = WHTMetadata(
        solver_name="Test", solver_version="0.0",
        analysis_type="static", coordinate_system="cartesian",
        unit_length="m", unit_force="N",
    )
    nodes = np.zeros((N, 3))
    # 2 triangle elements
    conn  = np.array([0, 1, 2, 1, 2, 3], dtype=np.int64)
    offs  = np.array([0, 3, 6],           dtype=np.int64)
    types = np.array([5, 5],              dtype=np.uint8)   # VTK_TRIANGLE
    u     = np.zeros((T, N, 3))
    t_vals = np.array([0.0] * T)

    return WHTResultData(
        nodes=nodes, connectivity=conn, offsets=offs, cell_types=types,
        point_data={"Displacement": u},
        time_values=t_vals,
        metadata=meta,
    )


class TestWHTResultData:

    def test_valid_creation(self):
        data = make_minimal_data()
        assert data.n_nodes == 4
        assert data.n_cells == 2
        assert data.n_timesteps == 1

    def test_wrong_nodes_shape(self):
        with pytest.raises(WHTValidationError, match="nodes must be"):
            meta = WHTMetadata("T", "0", "static", "cartesian", "m", "N")
            WHTResultData(
                nodes=np.zeros((4, 2)),             # wrong: must be (N,3)
                connectivity=np.array([0, 1, 2]),
                offsets=np.array([0, 3]),
                cell_types=np.array([5]),
                time_values=np.array([0.0]),
                metadata=meta,
            )

    def test_point_data_wrong_N(self):
        with pytest.raises(WHTValidationError, match="shape"):
            meta = WHTMetadata("T", "0", "static", "cartesian", "m", "N")
            WHTResultData(
                nodes=np.zeros((4, 3)),
                connectivity=np.array([0, 1, 2]),
                offsets=np.array([0, 3]),
                cell_types=np.array([5]),
                point_data={"U": np.zeros((1, 99, 3))},  # N=99 != 4
                time_values=np.array([0.0]),
                metadata=meta,
            )

    def test_repr(self):
        data = make_minimal_data()
        assert "WHTResultData" in repr(data)
        assert "N=4" in repr(data)
