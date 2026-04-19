# [Ultra-Detailed Plan] WHT Universal FEM Result Converter v0.3+

**Version**: 0.4 (Ultra-Detailed)
**Date**: 2026-04-18
**Target**: Commercial-grade FEM Post-processing Engine

## 1. Internal Architecture

### 1.1 Data Flow Path
`Solver Input` -> `BaseAdapter.convert()` -> `WHTResultData (IR)` -> `BaseExporter.export()` -> `Output File`

### 1.2 Intermediate Representation (IR) Specification
We use a **Flat Connectivity (CSR)** model to represent the mesh. This is the only way to support mixed-dimensional models (e.g., a chassis with beams, shells, and solids).

- **`connectivity` (array)**: `[n0, n1, n2, n3, ...]` (All cell nodes flattened).
- **`offsets` (array)**: `[0, 4, 7, 15, ...]` (Start index of each cell in the connectivity array).
- **`cell_types` (array)**: `[9, 5, 12, ...]` (VTK constants: 9=Quad, 5=Tria, 12=Hexa).

## 2. Solver Specific Adapters

### 2.1 JaxSSOAdapter
- **Geometry**: Maps `model.nodes` (dict) to a sorted NumPy array. Records original IDs if needed.
- **Elements**: Iterates through `model.quads`, `model.beamcols`, and `model.truss` to build a single CSR mesh.
- **Modes**: Maps natural frequencies to `time_values` and eigenvectors to `point_data['Displacement']`.

### 2.2 JaxFEMAdapter
- **Geometry**: Directly consumes `mesh.node_coords`. Converts JAX `DeviceArray` to NumPy using `np.asarray()`.
- **Results**: Handles transient `u` arrays by reshaping `(T, N, D)` to the IR field format.

## 3. High-Performance Exporters

### 3.1 VTKHDF (ParaView 5.11+)
- **Library**: `h5py`.
- **Logic**: 
  - Create `/VTKHDF/Points` (Static).
  - Create `/VTKHDF/Connectivity`, `Offsets`, `Types` (Static).
  - Create `/VTKHDF/Steps/Values` (Time values).
  - Create `/VTKHDF/PointData/Displacement` using transient indices.

### 3.2 HWASCII (HyperView)
- **Format**: Altair Generic ASCII.
- **Feature**: Supports multiple result blocks in one file. 
- **Logic**: Iterates through `point_data` and writes `$RESULT_TYPE` blocks for each time step defined by `$TIME`.

## 4. Phased Task List

### Phase 1: Models & Base Logic
- [ ] `wht_models.py`: Dataclass definitions with internal `__post_init__` validation.
- [ ] `to_vtk_csr()`: Converter from simple `(M, V)` to flat `(K,) + (M+1)`.

### Phase 2: Exporter Development
- [ ] `VTKHDFExporter`: Implementation of HDF5 transient schema.
- [ ] `HWASCIIExporter`: Implementation of Altair result blocks.

### Phase 3: Adapter Development
- [ ] `JaxSSOAdapter`: Tested against `exam1_nf.py` model.
- [ ] `JaxFEMAdapter`: Support for JAX-based arrays.

### Phase 4: Integration & UX
- [ ] Refactor `exam1_nf.py` main loop.
- [ ] CLI flag `--export all` to generate all formats in the `results/` folder.

---
*Created by Antigravity. Approved by User.*
