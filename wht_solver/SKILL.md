---
name: WHT Result Statistics Calculation
description: Rule and guideline for extracting comprehensive statistics (Stress, Strain, Displacement, Strain Energy) from WHTSolverResult.
---

# ResultStatsCalculator Usage Guide

## Mandatory Rule
**When calculating result statistics (Max, Mean, Variance/StdDev) or Total Strain Energy from a `WHTSolverResult`, you MUST ALWAYS use the `ResultStatsCalculator` class located in `wht_solver/wht_stats.py`.**

Do not implement one-off scripts or local loops to compute Von Mises stress, Equivalent Strain, or Displacement Magnitudes. Using `ResultStatsCalculator` ensures:
1. Consistency in Equivalent Strain equations across all modules.
2. Unified handling of padded element data (ignoring non-shell elements seamlessly).
3. Centralized calculation of Element Strain Energy and Total Strain Energy.
4. Support for the Multi-Objective Topography Optimization framework.

## Usage Example

```python
from wht_solver.wht_stats import ResultStatsCalculator

# Assuming `result` is a WHTSolverResult from static analysis
# and `model` is the corresponding WHTMeshModel
stats = ResultStatsCalculator.compute_stats(result, model)

print("Max Von Mises Stress:", stats["stress"]["max"])
print("Mean Displacement:", stats["displacement"]["mean"])
print("Total Strain Energy:", stats["energy"]["total"])
```

## Structure of the `stats` dictionary
The `compute_stats` method returns a dictionary with the following structure:
```json
{
    "stress": {"max": float, "mean": float, "std": float},
    "strain": {"max": float, "mean": float, "std": float},
    "displacement": {"max": float, "mean": float, "std": float},
    "energy": {"max": float, "mean": float, "std": float, "total": float}
}
```

- **stress**: Von Mises stress.
- **strain**: Equivalent strain (Von Mises strain).
- **displacement**: 3D displacement vector magnitude.
- **energy**: Element-level strain energy ($U_e = 0.5 \int \sigma \cdot \epsilon dV$) and total structure strain energy.
