# [Design Proposal] WHT Universal FEM Result Converter

**Date**: 2026-04-18
**Status**: Pending Review
**Target Solvers**: JaxSSO, jax-fem
**Target Viewers**: ParaView, Altair HyperView

## 1. Overview
The goal is to provide a clean, decoupled interface between JAX-based FEM solvers and industrial post-processing tools. The converter will abstract away the complexities of binary HDF5 (VTKHDF) and Altair's ASCII specifications.

## 2. Technical Architecture

### 2.1 Standard Data Interface (`WHTResultData`)
A centralized hub for data exchange. It uses JAX/NumPy to handle large numerical arrays efficiently.

- **Geometry**:
  - `nodes`: `(N, 3)` coordinates.
  - `elements`: `(M, V)` connectivity map.
  - `cell_types`: VTK-standard identifiers.
- **Results (Transient/Fixed)**:
  - `point_data`: Dictionary for nodal results `{name: (T, N, D)}`.
  - `cell_data`: Dictionary for elemental results `{name: (T, M, D)}`.
  - `time_values`: Vector of `(T,)` values (Time, Frequency, or Mode ID).

### 2.2 Adapter Layer
Bridge functions that translate native library formats to `WHTResultData`.
- `JaxSSO_to_WHT(model, vecs, freqs)`
- `JaxFEM_to_WHT(problem_mesh, u, steps)`

### 2.3 Exporter Layer
- **VTU/PVD**: Reliable multi-file XML format for legacy/stable viewing.
- **VTKHDF**: Single-file HDF5 binary for ParaView 5.11+ (High Performance).
- **HWASCII**: Single-file Altair ASCII format for HyperView results overlay.

## 3. Implementation Plan
1. **Core API**: Create `wht_result_data.py` (Universal Container).
2. **Exporter Library**: Create `wht_exporter.py` (VTU -> VTKHDF -> HWASCII).
3. **Refactor Pipeline**: Update `exam1_nf.py` to use these new components.
4. **Validation**: Test cross-compatibility between solvers and viewers.

## 4. Key Benefits
- **Zero Redundancy**: Mesh data is saved once in VTKHDF.
- **Single File Cleanliness**: No more cluttered directories with hundreds of VTUs.
- **Library Agnostic**: Easily switch between JaxSSO and jax-fem without changing the visualization logic.

---
*Please review this proposal and provide your feedback or approval.*
