"""
wht_utils.py
============
WHT Universal FEM Result Converter — Utility Functions

Provides:
    - to_vtk_csr()   : Convert a uniform (M, V) element array to VTK CSR format.
    - merge_csr()    : Merge multiple CSR meshes (different element types) into one.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# VTK Cell-Type Constants (most common)
# ---------------------------------------------------------------------------

class VTKCellType:
    """VTK cell type integer constants."""
    LINE         = 3    # 2-node line (beam / truss)
    TRIANGLE     = 5    # 3-node triangle
    QUAD         = 9    # 4-node quadrilateral (shell)
    TETRA        = 10   # 4-node tetrahedron
    HEXAHEDRON   = 12   # 8-node hexahedron
    WEDGE        = 13   # 6-node wedge / prism
    PYRAMID      = 14   # 5-node pyramid
    BIQUADRATIC_HEXAHEDRON = 29 # 27-node hexahedron (High-order)


# ---------------------------------------------------------------------------
# to_vtk_csr
# ---------------------------------------------------------------------------

def to_vtk_csr(
    elements: np.ndarray,
    vtk_type: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a uniform element connectivity array to VTK CSR flat format.

    Use this when *all* elements in the array have the same number of nodes
    (e.g. a pure-quad shell mesh, or a pure-hex solid mesh).
    For mixed meshes, build each group separately and call ``merge_csr()``.

    Parameters
    ----------
    elements : np.ndarray, shape (M, V)
        Element connectivity. Each row lists the V node indices of one element.
    vtk_type : int
        VTK cell-type constant (see ``VTKCellType``).

    Returns
    -------
    connectivity : np.ndarray, shape (M*V,)
    offsets      : np.ndarray, shape (M+1,)
    cell_types   : np.ndarray, shape (M,)

    Examples
    --------
    >>> quads = np.array([[0,1,2,3],[4,5,6,7]])
    >>> conn, offs, types = to_vtk_csr(quads, VTKCellType.QUAD)
    >>> conn
    array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> offs
    array([0, 4, 8])
    >>> types
    array([9, 9], dtype=uint8)
    """
    elements = np.asarray(elements, dtype=np.int64)
    if elements.ndim != 2:
        raise ValueError(f"elements must be 2-D (M, V), got shape {elements.shape}")

    M, V = elements.shape
    connectivity = elements.flatten()
    offsets      = np.arange(0, (M + 1) * V, V, dtype=np.int64)
    cell_types   = np.full(M, vtk_type, dtype=np.uint8)
    return connectivity, offsets, cell_types


# ---------------------------------------------------------------------------
# merge_csr
# ---------------------------------------------------------------------------

def merge_csr(
    groups: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge multiple (connectivity, offsets, cell_types) triples into one
    combined CSR representation.

    Typically used when a model contains several element groups with
    different topologies (beams + shells + solids).

    Parameters
    ----------
    groups : list of (connectivity, offsets, cell_types)
        Each tuple is the output of ``to_vtk_csr()`` for one element group.
        ``offsets`` for each group must start at 0 (as returned by
        ``to_vtk_csr``); this function handles the global offset shift.

    Returns
    -------
    connectivity : np.ndarray, shape (K_total,)
    offsets      : np.ndarray, shape (M_total+1,)
    cell_types   : np.ndarray, shape (M_total,)

    Examples
    --------
    >>> beams = to_vtk_csr(beam_conn, VTKCellType.LINE)
    >>> quads = to_vtk_csr(quad_conn, VTKCellType.QUAD)
    >>> conn, offs, types = merge_csr([beams, quads])
    """
    if not groups:
        raise ValueError("groups must contain at least one (conn, offsets, types) triple.")

    all_conn   = []
    all_offs   = [np.array([0], dtype=np.int64)]
    all_types  = []
    global_offset = 0

    for conn, offs, types in groups:
        conn  = np.asarray(conn,  dtype=np.int64)
        offs  = np.asarray(offs,  dtype=np.int64)
        types = np.asarray(types, dtype=np.uint8)

        all_conn.append(conn)
        # Drop the leading 0 of each group's offsets (already accounted for)
        # and shift by global_offset
        shifted = offs[1:] + global_offset
        all_offs.append(shifted)
        all_types.append(types)
        global_offset += int(offs[-1])

    connectivity = np.concatenate(all_conn)
    offsets      = np.concatenate(all_offs)
    cell_types   = np.concatenate(all_types)
    return connectivity, offsets, cell_types


# ---------------------------------------------------------------------------
# node_dict_to_array
# ---------------------------------------------------------------------------

def node_dict_to_array(
    node_dict: dict,
) -> Tuple[np.ndarray, dict]:
    """
    Convert a JaxSSO-style ``{node_id: [x, y, z]}`` dict to a sorted NumPy
    array, and return the id→row-index mapping for element re-indexing.

    Parameters
    ----------
    node_dict : dict
        Keys are node IDs (int), values are coordinate lists/arrays of length 3.

    Returns
    -------
    nodes : np.ndarray, shape (N, 3)
    id_map : dict
        Maps original node ID → 0-based row index in ``nodes``.

    Examples
    --------
    >>> nd = {10: [1.0, 0.0, 0.0], 20: [0.0, 0.0, 0.0]}
    >>> arr, mp = node_dict_to_array(nd)
    >>> mp
    {10: 0, 20: 1}
    """
    sorted_ids = sorted(node_dict.keys())
    id_map = {nid: i for i, nid in enumerate(sorted_ids)}
    nodes  = np.array([node_dict[nid] for nid in sorted_ids], dtype=np.float64)
    if nodes.shape[1] != 3:
        raise ValueError(f"Node coordinates must have 3 components, got {nodes.shape[1]}.")
    return nodes, id_map


def remap_connectivity(
    elem_dict: dict,
    id_map: dict,
) -> np.ndarray:
    """
    Convert an element dict ``{elem_id: [node_id, ...]}`` to a 2-D NumPy
    array with 0-based node indices, using ``id_map`` from
    ``node_dict_to_array``.

    Parameters
    ----------
    elem_dict : dict
        {element_id: [node_id_0, node_id_1, ...]}
    id_map : dict
        Mapping from original node ID to 0-based index.

    Returns
    -------
    np.ndarray, shape (M, V)
    """
    rows = []
    for eid in sorted(elem_dict.keys()):
        rows.append([id_map[nid] for nid in elem_dict[eid]])
    return np.array(rows, dtype=np.int64)
