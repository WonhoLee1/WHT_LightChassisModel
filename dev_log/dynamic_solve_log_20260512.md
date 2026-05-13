@ [04:07:42] JAX Dynamic Solver Optimization & RBE3 Damping Fix
- Resolved local deformation issue by excluding RBE3 penalty from Rayleigh damping.
- Optimized JAX SPCD trajectory and result sub-sampling.
- Switched to Direct LU for robustness.

@ [04:15:30] JAX Nested Scan Optimization
- Implemented block-wise time integration to reduce memory footprint from n_steps to n_blocks.
- Resolved Step 5 conversion hang by minimizing device-to-host data transfer.
- Explicitly managed memory by deleting large matrices before result processing.

