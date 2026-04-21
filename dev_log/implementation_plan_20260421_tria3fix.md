# Implementation Plan - Fixing TRIA3 Rigid Body Modes (2026-04-21)

The user reported that QUAD4 has rigid body modes (0 Hz) while TRIA3 and other meshes do not. This indicates that TRIA3 elements are accidentally "grounded" (attached to an absolute coordinate system).

## User Review Required

> [!IMPORTANT]
> The grounding is caused by a diagonal penalty stiffness added to the drilling degree of freedom in `K_tria3_scipy`. This stabilizes the numerical system but prevents the structure from moving freely.

## Proposed Changes

---

### [wht_solver]

#### [MODIFY] [wht_tria3_element.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_tria3_element.py)
- Remove the absolute diagonal penalty on the drilling DOF (`K_loc[6*i+5, 6*i+5] += k_drilling`).
- Rely on the existing relative drilling penalty (Allman-type) which is correctly formulated as a internal coupling and does not ground the element.

#### [MODIFY] [wht_solver.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_solver/wht_solver.py)
- Review the `AUTOSPC` logic to ensures it doesn't ground nodes that are part of a rigid body mode.

## Verification Plan

### Automated Tests
- Run `python .\test_jaxSSO\exam2_shell_jaxSSO.py`.
- Verify that all meshes show 6 rigid body modes near 0.00 Hz.
