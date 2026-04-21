# Implementation Plan - Selective Flange Generation (2026-04-21)

This plan outlines the changes needed to allow selective generation of flanges (Bottom, Right, Top, Left) in the shell mesh generation utility and its integration into the modal analysis pipeline.

## User Review Required

> [!IMPORTANT]
> The flange selection is implemented as a 4-boolean tuple `(bottom, right, top, left)`. 
> The user specifically requested setting the **bottom** flange to `False`.

## Proposed Changes

---

### [test_jaxSSO]

#### [MODIFY] [mesh_utils.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/test_jaxSSO/mesh_utils.py)
- Modify `generate_shell_tray` signature to include `flanges: tuple = (True, True, True, True)`.
- Wrap flange geometry creation (`addCurveLoop`, `addPlaneSurface`) in conditional blocks based on `flanges`.
- Wrap flange mesh settings (`setTransfiniteCurve`, `setTransfiniteSurface`, `setRecombine`) in conditional blocks.
- Ensure connectivity and indexing remain consistent even when some flanges are missing.

#### [MODIFY] [exam2_shell_jaxSSO.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/test_jaxSSO/exam2_shell_jaxSSO.py)
- Update `PipelineConfig` to include a `flanges` field (defaulting to all `True`).
- Update `generate_mesh` to pass `cfg.flanges` to `generate_shell_tray`.
- Update the `main` test cases to demonstrate selective flanges (e.g., setting the bottom flange to `False`).

## Verification Plan

### Automated Tests
- Run `python .\test_jaxSSO\exam2_shell_jaxSSO.py` to ensure the mesh generates correctly and modal analysis completes.
- Verify visually in the PyVista viewer that the bottom flange is indeed missing.

### Manual Verification
- Check terminal output for any Gmsh errors regarding transfinite settings on non-existent entities.
