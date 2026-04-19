# Implementation Plan - Enhanced Scalar Bar Range Adjustment

This plan addresses the user's request for a robust, real-time scalar bar range adjustment system in `WHTVisualizer`. It focuses on handling outliers using statistical methods and providing a high-fidelity adjustment UI.

## User Review Required

> [!IMPORTANT]
> **Statistical Range Suggestion**: I propose using **Percentile-based clipping (2th to 98th percentile)** as the primary method for the "Robust Auto" feature. This effectively ignores common FEA singularities at constraint points or sharp edges, ensuring the main structural response is clearly visible without manual trial-and-error.

> [!NOTE]
> **Live Update Mechanism**: The new adjustment dialog will hold a temporary reference to the visualizer to trigger `_apply_colorbar_range` on every slider/spinbox interaction, providing instant visual feedback.

## Proposed Changes

### Visualization Engine Updates
#### [MODIFY] [wht_visualizer.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_visualizer/wht_visualizer.py)
- **UI Component**:
    - Add an "Adjust Range..." button in the "Fields" group.
    - Create a custom `QDialog` named `RangeAdjustDialog` with the following:
        - Min/Max Sliders (0-1000 scale mapped to global data range).
        - Min/Max DoubleSpinBoxes (linked to sliders).
        - "Robust Auto (2nd-98th)" button.
        - "Global Auto (Full)" button.
- **Logic**:
    - Implement `_calculate_robust_range(field_name)` using `np.percentile`.
    - Update `_on_category_changed` and `_on_component_changed` to immediately re-calculate and apply ranges when the result type is switched.
    - Ensure `_apply_colorbar_range` is called whenever values in the Adjustment Dialog change.

## Verification Plan

### Automated/Manual Testing via Script
- Run `python exam2_shell_jaxSSO_load.py`.
- Change result field to "Stress".
- Click "Adjust Range...".
- Move Min/Max sliders and verify that the colorbar and mesh colors updated **instantly**.
- Click "Robust Auto" and verify that the range narrows to exclude extreme peaks (e.g., at supports).
- Type a specific value in the spinbox and verify the slider moves accordingly.

### Success Criteria
- [ ] No more "lag" between field change and range update.
- [ ] Outliers no longer flatten the entire contour plot (resolved via Robust Auto).
- [ ] Smooth real-time interaction between sliders and visualizer.
