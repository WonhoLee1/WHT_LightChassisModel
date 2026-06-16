"""
wht_mapper.py
=============
WHT FEM Framework — RBF Mesh Mapping

Maps nodal results (displacements, mode shapes) from a source mesh
(high-fidelity) to a target mesh (low-fidelity / optimization model).

Uses scipy RBFInterpolator (thin-plate spline by default).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from scipy.interpolate import RBFInterpolator


class WHTMapper:
    """
    Radial Basis Function interpolator for cross-mesh mapping.

    Usage
    -----
        mapper = WHTMapper()
        mapper.fit(source_nodes, source_data)   # (N_hi, 3), (N_hi, D)
        mapped = mapper.transform(target_nodes)  # → (N_lo, D)
    """

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        epsilon: Optional[float] = None,
        smoothing: float = 0.0,
    ):
        self.kernel    = kernel
        self.epsilon   = epsilon
        self.smoothing = smoothing
        self._interp:  Optional[RBFInterpolator] = None
        self._source_nodes: Optional[np.ndarray] = None

    def fit(
        self,
        source_nodes: np.ndarray,
        source_data:  np.ndarray,
    ) -> "WHTMapper":
        """
        Build RBF interpolant.

        Parameters
        ----------
        source_nodes : (N_src, 3) source node coordinates
        source_data  : (N_src, D) data to be interpolated
        """
        self._source_nodes = np.asarray(source_nodes, dtype=np.float64)
        source_data        = np.asarray(source_data,  dtype=np.float64)

        kw = {}
        if self.epsilon is not None:
            kw["epsilon"] = self.epsilon

        # thin_plate_spline's default affine polynomial term (degree=1) is
        # rank-deficient when the source points are (near-)coplanar/collinear
        # (e.g. a flat plate with no Z variation) -> scipy raises LinAlgError.
        # Fall back to a lower polynomial degree, which only needs a smaller
        # subset of the coordinates to be independent.
        for degree in (kw.get("degree", None), 0, -1):
            try:
                attempt_kw = dict(kw)
                if degree is not None:
                    attempt_kw["degree"] = degree
                self._interp = RBFInterpolator(
                    self._source_nodes,
                    source_data,
                    kernel=self.kernel,
                    smoothing=self.smoothing,
                    **attempt_kw,
                )
                return self
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError(
            "WHTMapper.fit(): RBF interpolant is singular even with degree=-1 "
            "(no polynomial term) — source points may be degenerate "
            "(duplicated or fully coincident)."
        )

    def transform(self, target_nodes: np.ndarray) -> np.ndarray:
        """
        Interpolate to target node positions.

        Parameters
        ----------
        target_nodes : (N_tgt, 3) target node coordinates

        Returns
        -------
        (N_tgt, D) interpolated values
        """
        if self._interp is None:
            raise RuntimeError("WHTMapper must be fitted before transform().")
        return self._interp(np.asarray(target_nodes, dtype=np.float64))

    def fit_transform(
        self,
        source_nodes: np.ndarray,
        source_data:  np.ndarray,
        target_nodes: np.ndarray,
    ) -> np.ndarray:
        """Convenience: fit then transform in one call."""
        return self.fit(source_nodes, source_data).transform(target_nodes)

    def map_modes(
        self,
        source_nodes: np.ndarray,
        mode_shapes:  np.ndarray,
        target_nodes: np.ndarray,
    ) -> np.ndarray:
        """
        Map all mode shapes from source to target mesh.

        Parameters
        ----------
        source_nodes : (N_src, 3)
        mode_shapes  : (n_modes, N_src, D)
        target_nodes : (N_tgt, 3)

        Returns
        -------
        (n_modes, N_tgt, D)
        """
        n_modes = mode_shapes.shape[0]
        D       = mode_shapes.shape[2]
        N_tgt   = len(target_nodes)
        result  = np.zeros((n_modes, N_tgt, D))

        for m in range(n_modes):
            self.fit(source_nodes, mode_shapes[m])
            result[m] = self.transform(target_nodes)
        return result

    @staticmethod
    def rmse(a: np.ndarray, b: np.ndarray) -> float:
        """Root Mean Square Error between two arrays."""
        return float(np.sqrt(np.mean((a - b) ** 2)))
