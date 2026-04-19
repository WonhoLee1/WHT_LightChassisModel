# Implementation Plan — Fixing Mixed Mesh Generation

The `mixed` mesh type currently results in an almost all-triangle mesh because it is excluded from the structured mesh (Transfinite) setup block in `mesh_utils.py`.

## Proposed Changes

### Mesh Generation

#### [MODIFY] [mesh_utils.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/test_jaxSSO/mesh_utils.py)
- **Include `mixed` in Transfinite block**: Add `'mixed'` to the condition checking for structured/transfinite mesh eligibility (around line 108).
- **Ensure Recombine for Base**: Verify that `setRecombine` is called for the base surface (`s1`) when `mesh_type` is `mixed`. (The code at line 132 already handles this if the block is entered).

## Verification Plan

### Automated Tests
- `python exam2_shell_jaxSSO.py`
  - Observe the element counts for the `MIXED` case.
  - Verification: The element count for `MIXED` should be significantly lower than `TRIA3` (~4000) and higher than `QUAD4` (~2200), reflecting a mix of quads and triangles.
  - Visually inspect the generated mesh (if possible) or check the `Assembly Ready` message for cell counts.

### Expected Results
- `QUAD4` (all quads): ~2200 elements.
- `TRIA3` (all trias): ~4000 elements.
- `MIXED` (quad floor + tria walls): should be in the range of 3000-3500 elements.
