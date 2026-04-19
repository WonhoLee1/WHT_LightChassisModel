"""
wht_models.py
=============
WHT Universal FEM Result Converter — Core Data Models

Defines:
    - WHTMetadata      : Solver/analysis metadata (units, coordinate system, etc.)
    - WHTResultData    : Central Intermediate Representation (IR) for all FEM results.
    - WHTValidationError : Raised on structural/shape violations (fail-fast).
    - WHTExportWarning   : Raised when unsupported fields are silently skipped.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# Exceptions & Warnings
# ---------------------------------------------------------------------------

class WHTValidationError(Exception):
    """
    Raised when WHTResultData contains structurally invalid data.
    Causes immediate pipeline termination — do NOT catch silently.

    Examples
    --------
    - nodes shape is not (N, 3)
    - point_data array has wrong N dimension
    - time_values is not monotonically increasing for transient analysis
    """
    pass


class WHTExportWarning(UserWarning):
    """
    Raised when an exporter encounters an unsupported field and skips it.
    The export continues; only a warning is issued.

    Examples
    --------
    - A point_data key not in SUPPORTED_BLOCKS for HWASCIIExporter
    """
    pass


# ---------------------------------------------------------------------------
# WHTMetadata
# ---------------------------------------------------------------------------

@dataclass
class WHTMetadata:
    """
    Solver and analysis metadata, kept separate from numerical arrays so
    exporters can write format-specific headers independently.

    Attributes
    ----------
    solver_name : str
        "JaxSSO" | "jax-fem"
    solver_version : str
        e.g. "0.1.0"
    analysis_type : str
        "static" | "modal" | "transient" | "buckling"
    coordinate_system : str
        "cartesian" | "cylindrical"
    unit_length : str
        "m" | "mm" | "in"
    unit_force : str
        "N" | "kN" | "lbf"
    created_at : str
        ISO 8601 timestamp. Auto-filled if empty string is passed.

    Notes
    -----
    Unit fields are required (no defaults) to prevent the classic mm/m
    scale-factor-of-1000 bug in HyperView.
    """

    solver_name: str
    solver_version: str
    analysis_type: str
    coordinate_system: str
    unit_length: str
    unit_force: str
    unit_mass: str = "tonne"
    unit_time: str = "s"
    created_at: str = ""

    # Valid enumerations
    _VALID_ANALYSIS = {"static", "modal", "transient", "buckling"}
    _VALID_COORD    = {"cartesian", "cylindrical"}
    _VALID_LENGTH   = {"m", "mm", "in"}
    _VALID_FORCE    = {"N", "kN", "lbf"}
    _VALID_MASS     = {"kg", "tonne", "lb"}
    _VALID_TIME     = {"s", "ms"}

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if self.analysis_type not in self._VALID_ANALYSIS:
            raise WHTValidationError(
                f"analysis_type='{self.analysis_type}' is not valid. "
                f"Choose from {self._VALID_ANALYSIS}."
            )
        if self.coordinate_system not in self._VALID_COORD:
            raise WHTValidationError(
                f"coordinate_system='{self.coordinate_system}' is not valid. "
                f"Choose from {self._VALID_COORD}."
            )
        if self.unit_length not in self._VALID_LENGTH:
            raise WHTValidationError(
                f"unit_length='{self.unit_length}' is not valid. "
                f"Choose from {self._VALID_LENGTH}."
            )
        if self.unit_force not in self._VALID_FORCE:
            raise WHTValidationError(
                f"unit_force='{self.unit_force}' is not valid. "
                f"Choose from {self._VALID_FORCE}."
            )
        if self.unit_mass not in self._VALID_MASS:
            raise WHTValidationError(
                f"unit_mass='{self.unit_mass}' is not valid. "
                f"Choose from {self._VALID_MASS}."
            )
        if self.unit_time not in self._VALID_TIME:
            raise WHTValidationError(
                f"unit_time='{self.unit_time}' is not valid. "
                f"Choose from {self._VALID_TIME}."
            )


# ---------------------------------------------------------------------------
# WHTResultData
# ---------------------------------------------------------------------------

@dataclass
class WHTResultData:
    """
    Central Intermediate Representation (IR) for all FEM solver outputs.

    Geometry uses the VTK CSR (Compressed Sparse Row) flat format so that
    mixed-dimensional meshes (beams + shells + solids) are represented exactly.

    Parameters
    ----------
    nodes : np.ndarray, shape (N, 3)
        Node coordinates [x, y, z].
    connectivity : np.ndarray, shape (K,)
        Flattened node-index array for all cells.
    offsets : np.ndarray, shape (M+1,)
        Start index of each cell in ``connectivity``.
        offsets[i] ... offsets[i+1] gives the node indices of cell i.
    cell_types : np.ndarray, shape (M,)
        VTK cell-type integer constants (e.g. 3=Line, 9=Quad, 12=Hexa).
    node_sets : dict[str, np.ndarray]
        Named node subsets, e.g. {"fixed_support": np.array([0, 1, 2])}.
    element_sets : dict[str, np.ndarray]
        Named element subsets, e.g. {"beam_group": np.array([10, 11])}.
    point_data : dict[str, np.ndarray]
        Nodal result arrays {name: (T, N, D)}.
        D is the component dimension (e.g. 3 for displacement, 6 for stress).
    cell_data : dict[str, np.ndarray]
        Element result arrays {name: (T, M, D)}.
    field_data : dict[str, np.ndarray]
        Global scalar arrays {name: (T,)}, e.g. {"LoadFactor": ...}.
    time_values : np.ndarray, shape (T,)
        Axis values whose meaning depends on analysis_type:
          static    → [0.0]
          transient → [t0, t1, ..., tT]   (seconds)
          modal     → [f0, f1, ..., fn]   (Hz)
          buckling  → [λ0, λ1, ..., λn]  (load factor)
    metadata : WHTMetadata
        Solver and unit information.

    Notes
    -----
    ``__post_init__`` performs shape-consistency validation immediately on
    construction (fail-fast principle).
    """

    # Geometry
    nodes: np.ndarray
    connectivity: np.ndarray
    offsets: np.ndarray
    cell_types: np.ndarray

    # Named Sets
    node_sets: Dict[str, np.ndarray] = field(default_factory=dict)
    element_sets: Dict[str, np.ndarray] = field(default_factory=dict)

    # Results
    point_data: Dict[str, np.ndarray] = field(default_factory=dict)
    cell_data:  Dict[str, np.ndarray] = field(default_factory=dict)
    field_data: Dict[str, np.ndarray] = field(default_factory=dict)

    # Axis
    time_values: np.ndarray = field(default_factory=lambda: np.array([0.0]))

    # Meta
    metadata: WHTMetadata = field(default=None)

    # ------------------------------------------------------------------
    # Automatic validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Shape-consistency checks. Raises WHTValidationError on failure."""

        # --- nodes ---
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise WHTValidationError(
                f"nodes must be (N, 3), got shape {self.nodes.shape}."
            )
        N = self.nodes.shape[0]

        # --- CSR mesh ---
        if self.offsets.ndim != 1 or len(self.offsets) < 2:
            raise WHTValidationError(
                "offsets must be a 1-D array of length M+1 (at least 2)."
            )
        M = len(self.offsets) - 1

        if len(self.cell_types) != M:
            raise WHTValidationError(
                f"cell_types length must equal M={M}, got {len(self.cell_types)}."
            )
        if self.connectivity.ndim != 1:
            raise WHTValidationError("connectivity must be a 1-D array.")
        if len(self.connectivity) != int(self.offsets[-1]):
            raise WHTValidationError(
                f"connectivity length {len(self.connectivity)} does not match "
                f"offsets[-1]={int(self.offsets[-1])}."
            )

        # --- time axis ---
        T = len(self.time_values)
        if T == 0:
            raise WHTValidationError("time_values must have at least one entry.")

        # --- point_data ---
        for name, arr in self.point_data.items():
            if arr.ndim < 2:
                raise WHTValidationError(
                    f"point_data['{name}'] must be at least 2-D (T, N, ...), "
                    f"got ndim={arr.ndim}."
                )
            if arr.shape[0] != T:
                raise WHTValidationError(
                    f"point_data['{name}'].shape[0]={arr.shape[0]} != T={T}."
                )
            if arr.shape[1] != N:
                raise WHTValidationError(
                    f"point_data['{name}'].shape[1]={arr.shape[1]} != N={N}."
                )

        # --- cell_data ---
        for name, arr in self.cell_data.items():
            if arr.ndim < 2:
                raise WHTValidationError(
                    f"cell_data['{name}'] must be at least 2-D (T, M, ...), "
                    f"got ndim={arr.ndim}."
                )
            if arr.shape[0] != T:
                raise WHTValidationError(
                    f"cell_data['{name}'].shape[0]={arr.shape[0]} != T={T}."
                )
            if arr.shape[1] != M:
                raise WHTValidationError(
                    f"cell_data['{name}'].shape[1]={arr.shape[1]} != M={M}."
                )

        # --- field_data ---
        for name, arr in self.field_data.items():
            if arr.ndim != 1 or len(arr) != T:
                raise WHTValidationError(
                    f"field_data['{name}'] must be shape (T={T},), "
                    f"got {arr.shape}."
                )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_cells(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_timesteps(self) -> int:
        return len(self.time_values)

    def __repr__(self) -> str:
        return (
            f"WHTResultData("
            f"solver={self.metadata.solver_name if self.metadata else '?'}, "
            f"analysis={self.metadata.analysis_type if self.metadata else '?'}, "
            f"N={self.n_nodes}, M={self.n_cells}, T={self.n_timesteps})"
        )
