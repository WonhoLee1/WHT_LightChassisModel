# Walkthrough - Discrete Bead Height Optimization (2026-05-08)

We have successfully implemented discrete bead height control within the topography optimization pipeline.

## Changes Made

### 1. Discrete Projection in Solver
- **Staircase Function**: Added `_project_x` and `_project_x_grad` in `solver.py`. This uses a sum of `tanh` functions to create smooth steps between 0 and 1.
- **Beta Continuation**: The projection sharpness (`beta`) now increases from 1.0 to 50.0 over the iterations. This allows the solver to find the optimal topology first and then slowly lock into the discrete levels.
- **Sensitivity Chain-rule**: Correctly propagated the gradients through the projection function to ensure MMA optimization remains stable.

### 2. Intuitive Parameter Handling
- **N Levels**: Updated the interpretation of `--height-steps N`. Now, `N=2` means exactly 2 levels (0 and max height), which is much more intuitive than the previous "interval" logic.
- **Argument Passing**: Fixed a bug where `run_topo.py` was not passing the `height-steps` parameter to the solver instance.

### 3. Integration & UI
- **User Improvements**: Preserved and integrated user-made enhancements such as VTKHDF export per iteration and UI tooltip/labeling updates.
- **Real-time Visualization**: The monitoring GUI now displays the "projected" discrete heights, allowing you to see the beads forming into discrete steps during the optimization.

## Verification

### Test Case: `--height-steps 2`
- Run the command: `python wht_topo/run_topo.py --iters 20 --height-steps 2 --gui`
- **Expected**: The height distribution plot should initially be continuous and gradually become binary (Blue for 0, Red for 10mm) as the iterations approach the end.
- **Final Result**: The terminal output should show `[이산화] 2개 레벨 양자화 완료: [0. 10.] mm`.
