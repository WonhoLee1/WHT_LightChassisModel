# Implementation Plan — MITC3+ Shell Element Implementation

We will implement the **MITC3+** shell element, as described in the paper by Bathe et al., to replace the current TRIA3 implementation. This represents the "Gold Standard" for low-order triangular shell elements.

## User Review Required

> [!IMPORTANT]
> **Static Condensation**: The MITC3+ formulation uses internal bubble functions. I will implement the static condensation at the element level so that the global system still sees 6 DOFs per node (18 total per element), making it a drop-in replacement for the current solver.

> [!TIP]
> **Shear Locking**: Using the tying scheme from the MITC3+ paper will specifically fix the "high stiffness" issue seen in previous TRIA3 runs by correctly representing the transverse shear strain.

## Proposed Changes

### MITC3+ Element Formulation

#### [MODIFY] [wht_tria3_element.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_tria3_element.py)
- **Implement MITC3+ Logic**:
  - **Metric Tensors**: Calculate covariant and contravariant basis vectors.
  - **Internal Enrichment**: Add the cubic bubble function for rotations as per the Bathe paper.
  - **Shear Strains**: Implement the specific tying scheme of MITC3+ for transverse shear.
  - **Static Condensation**: Perform $[K_{bb} - K_{bi} K_{ii}^{-1} K_{ib}]$ at the element level to remove internal DOFs.
  - **Drilling**: Integrate the stabilization and hourglass control from the previously provided code.

### Verification Plan

### Automated Tests
- `python exam2_shell_jaxSSO.py`
  - High expectation: The `TRIA3` and `MIXED` results should now align closely with `QUAD4` (within 1-5% range) at **~2.8 Hz**.
  - Accuracy check: Compare frequencies across different mesh densities.

### Manual Verification
- Visual inspection of the first 6 modes to confirm they match the tray's physical bending behavior (e.g., floor breathing mode).
