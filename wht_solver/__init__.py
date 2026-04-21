"""
wht_solver
==========
WHT FEM Framework — Solver, Mapper, Optimizer Package

Provides JaxSSO-based static/modal analysis, RBF cross-mesh mapping,
multi-objective optimization, and real-time visualization.

Quick start
-----------
    from wht_solver import WHTSolver, WHTMapper, LoadCaseLibrary
    from wht_solver.load_cases import WHTLoadCase
    from wht_solver.objectives import mac, multi_objective_loss
    from wht_solver.wht_optimizer import WHTOptimizer, DesignVariables, DesignBounds
    from wht_solver.wht_monitor import OptimizationMonitor
"""

from .wht_solver      import WHTSolver
from .wht_result      import WHTSolverResult
from .wht_mapper      import WHTMapper
from .load_cases      import WHTLoadCase, LoadCaseLibrary
from .wht_monitor     import OptimizationMonitor
from .wht_optimizer   import DesignVariables, DesignBounds, WHTOptimizer
from .wht_sensitivity import WHTSensitivity
from .wht_eigensolver import make_modal_freq_fn

__version__ = "0.1.0"
__all__ = [
    "WHTSolver",
    "WHTSolverResult",
    "WHTMapper",
    "WHTLoadCase",
    "LoadCaseLibrary",
    "OptimizationMonitor",
    "DesignVariables",
    "DesignBounds",
    "WHTOptimizer",
    "WHTSensitivity",
    "make_modal_freq_fn",
]
