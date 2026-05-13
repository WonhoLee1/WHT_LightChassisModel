# Implementation Plan - Discrete Bead Height Optimization (2026-05-08)

The goal is to fix the issue where the `--height-steps` option does not result in discrete bead heights during topography optimization. The current implementation only performs quantization at the very end of the process, and the parameter is not even passed to the core solver.

## User Review Required

> [!IMPORTANT]
> The interpretation of `--height-steps N` will be updated for better intuition:
> - **Old logic**: `N` steps = `N` intervals = `N+1` levels (e.g., `2` steps -> `0, 5, 10mm`).
> - **New logic**: `N` steps = **`N` discrete levels** (e.g., `2` steps -> `0, 10mm`, `3` steps -> `0, 5, 10mm`).
> - I will implement a smooth projection (continuation method) so that the optimization actually converges to these discrete levels.
> - The Monitoring UI will show these "projected" discrete heights during iterations.

## Proposed Changes

### [wht_topo]

#### [MODIFY] [run_topo.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_topo/run_topo.py)
- Pass `bead_steps=args.height_steps` when initializing `WHTopographySolver`.
- Adjust the final quantization logic to match the solver's internal levels.

#### [MODIFY] [solver.py](file:///d:/PythonCodeStudy/WHT_LightChassisModel/wht_topo/solver.py)
- Implement `_project_x(x, beta)`: A smooth staircase projection function.
- Implement `_project_x_grad(x, beta)`: The gradient of the projection for sensitivity chain-rule.
- Update the `solve()` loop:
    - Apply projection to the design variable `x`.
    - Use projected `x` for FEA and filtering.
    - Update sensitivities: `df/dx = df/dx_proj * dx_proj/dx`.
    - Implement a `beta` continuation (start low, increase over iterations) to ensure stable convergence to discrete values.

## Verification Plan

### Automated Tests
- Run the optimization with `--height-steps 2` and `--iters 10`.
- Verify that the Monitoring UI shows increasingly discrete colors.
- Verify that the final `heights` output contains only values near the discrete levels (0, 10mm for steps=2).

### Manual Verification
- Check the final `.k` file to ensure node coordinates are moved by discrete amounts.
- Observe the "Height Distribution" plot in the GUI to see if it becomes discrete as iterations progress.
