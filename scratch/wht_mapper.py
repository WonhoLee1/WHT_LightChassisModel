"""
wht_mapper.py
=============
WHT Universal FEM Framework — Mapping Module

Provides Nodal Mapping functionality using RBF (Radial Basis Function)
interpolation to compare data between meshes of different densities.
"""

import numpy as np
from scipy.interpolate import RBFInterpolator
from typing import Optional


class WHTMapper:
    """
    Handles interpolation of nodal results from a source mesh (e.g. High-Fi)
    to a destination mesh (e.g. Low-Fi).
    """
    
    def __init__(self, kernel: str = "thin_plate_spline", epsilon: Optional[float] = None):
        self.kernel = kernel
        self.epsilon = epsilon
        self.interpolant = None
        self.source_coords = None

    def fit(self, source_coords: np.ndarray):
        """
        Prepare the interpolant using source node coordinates.
        source_coords: (N_source, 3)
        """
        self.source_coords = np.asarray(source_coords, dtype=np.float64)

    def map(self, source_values: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
        """
        Interpolate source_values to target_coords.
        
        source_values: (N_source, D) or (T, N_source, D)
        target_coords: (N_target, 3)
        
        Returns:
            Mapped values at target locations.
        """
        if self.source_coords is None:
            raise ValueError("Mapper must be fitted with source_coords first.")
        
        target_coords = np.asarray(target_coords, dtype=np.float64)
        source_values = np.asarray(source_values, dtype=np.float64)
        
        # Handle time-series or multi-mode data (T, N, D)
        if source_values.ndim == 3:
            T, N, D = source_values.shape
            mapped_results = []
            for t in range(T):
                # We create a new interpolant per timestep for the values
                # Note: Re-using the basis matrix might be faster for large models,
                # but RBFInterpolator manages it internally.
                interp = RBFInterpolator(
                    self.source_coords, source_values[t],
                    kernel=self.kernel, epsilon=self.epsilon
                )
                mapped_results.append(interp(target_coords))
            return np.stack(mapped_results)
        
        # Handle single step data (N, D)
        else:
            interp = RBFInterpolator(
                self.source_coords, source_values,
                kernel=self.kernel, epsilon=self.epsilon
            )
            return interp(target_coords)

    @staticmethod
    def calculate_rmse(val1: np.ndarray, val2: np.ndarray) -> float:
        """Calculates Root Mean Square Error between two arrays."""
        return np.sqrt(np.mean((val1 - val2)**2))
