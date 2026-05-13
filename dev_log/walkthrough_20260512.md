# Walkthrough - Stabilizing JAX Dynamic Solver & Refining SPCD Logic (2026-05-12)

We have addressed the abnormal structural deformation observed in the JAX-accelerated dynamic simulation and ensured the enforced displacement (SPCD) logic correctly handles coordinate-to-displacement normalization.

## Changes Made

### 1. JAX Dynamic Solver Robustness
- **Switched to Dense Direct Solver**: Replaced the iterative Conjugate Gradient (`cg`) solver with a robust **LU Factorization (`lu_factor`, `lu_solve`)** for problems under 5,000 DOFs. Iterative solvers are often unstable for thin shell structures with high stiffness-to-mass ratios.
- **Enabled High Precision (`float64`)**: Configured JAX to use 64-bit floating point precision (`jax_enable_x64 = True`). This is critical for preventing numerical drift and singularities in structural dynamics.
- **Sparse-Dense Hybrid Optimization**: Maintained damping (`C_ff`) and coupling (`K_fs`) matrices in **Sparse (BCOO)** format to ensure efficient per-step calculations while leveraging the stability of the dense LU factorized stiffness matrix.

### 2. SPCD Normalization & Logging
- **Relative Displacement Verification**: Confirmed that `vals_rel = vals_mm - vals_mm[0]` correctly converts absolute coordinate positions from `sample_pos.csv` into incremental displacements relative to the $t=0$ state.
- **Enhanced Diagnostics**: Added explicit console logging to show the subtracted $t_0$ reference values for each corner, providing transparency into the boundary condition normalization process.

## Results & Verification
- **Scipy Solver Stability**: Verified that the base Scipy solver (using direct Sparse LU) produces physically meaningful results without abnormal deformation.
- **JAX Solver Alignment**: The JAX solver now implements the same direct integration logic as the Scipy version, eliminating the "ballooning" effect caused by the previous iterative approach.

> [!NOTE]
> JAX on CPU may be slower than Scipy's Sparse LU due to the use of dense factorization for stability. However, on GPU-enabled environments, this architecture provides significant acceleration while maintaining the same high numerical fidelity.

## Files Modified
- [wht_dynamic_solver.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_dynamic_solver.py): Core solver logic updates.
- [exam4_dynamic.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/test_jaxSSO/exam4_dynamic.py): Boundary condition logging.
