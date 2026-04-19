# [Final Implementation Plan] Universal FEM Result Converter v0.3+

**Date**: 2026-04-18
**Status**: Finalized (Pending Execution)
**Key Updates**: CSR-style Connectivity, Multi-Dimension Support, Solver Adapters (JaxSSO & jax-fem)

## 1. Core Architecture (V0.3 Refined)

The system is designed as a three-layer pipeline:
1. **Source Adapters**: Extract and normalize data from JAX libraries.
2. **Intermediate IR (`WHTResultData`)**: Professional-grade storage using flat connectivity and offsets.
3. **Format Exporters**: Target binary and text outputs for ParaView and HyperView.

## 2. Technical Data Specification

### 2.1 Mesh Storage (Flat CSR)
To handle mixed-mesh models (e.g., shell + beam), we replace simple arrays with:
- `connectivity`: All node indices in a single continuous array.
- `offsets`: `(M+1)` array indicating where each element starts.
- `cell_types`: `(M,)` array using VTK cell type constants.

### 2.2 Metadata & Units
`WHTMetadata` stores solver provenance and physical units to prevent scaling errors during post-processing.

## 3. Implementation Workflow

### Step 1: Data Models (`wht_models.py`)
- Define `WHTMetadata` and `WHTResultData` using Python `dataclasses`.
- Implement `to_vtk_csr()` utility for legacy code compatibility.

### Step 2: Solver Adapters (`wht_adapters.py`)
- **`JaxSSOAdapter`**: Auto-maps Shell Q4/T3 models and modal/buckling results.
- **`JaxFEMAdapter`**: Auto-maps 3D solid models and transient/modal results.

### Step 3: Multi-Export Engine (`wht_exporters.py`)
- **VTKHDF**: Binary HDF5 for high-performance ParaView playback.
- **HWASCII**: Altair-compliant ASCII for HyperView results overlay.
- **VTU/PVD**: Fallback XML for cross-tool validation.

## 4. Execution Schedule
- [ ] Task 1: Building Data Hub (`wht_models.py`)
- [ ] Task 2: Implementing Adapters (`wht_adapters.py`)
- [ ] Task 3: Building Exporters (`wht_exporters.py`)
- [ ] Task 4: Integration Test (`exam1_nf.py`)

---
*Created by Antigravity based on User Proposal v0.3.*
