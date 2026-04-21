# Implementation Plan - Differentiable & Multi-core Modal Pipeline (2026-04-22)

Plan to transition the modal analysis engine to a JAX-native, differentiable, and multi-core optimized structure.

## Proposed Changes

### 1. Element Library JAX Conversion
- **`wht_quad4_element.py`**: Convert `numpy` to `jax.numpy`. Implement JAX-style `at[].set()` for matrix assembly.
- **`wht_tria3_element.py`**: Same conversion.

### 2. Multi-core Optimized Assembly
- Implement `_assemble_K_vectorized` using `jax.vmap` to compute K matrices for all elements in parallel.
- Leverage XLA multi-threading for CPU.

### 3. Differentiable Eigensolver Adjoint
- Implement `custom_vjp` for eigenvalues using the analytical sensitivity formula.
- $\frac{df}{ds} = \frac{1}{4\pi f} \phi^T (\frac{dK}{ds} - (2\pi f)^2 \frac{dM}{ds}) \phi$.

## Verification Plan
- Unit tests comparing NumPy vs JAX assembly.
- `jax.grad` sensitivity validation.
