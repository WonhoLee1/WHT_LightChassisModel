"""
wht_optimizer.py
================
WHT FEM Framework — JAX + Optax Optimization Engine

WHTOptimizer minimizes multi_objective_loss over DesignVariables
(t_field, z_offsets, E, rho) using Optax Adam gradient descent.

Strategy B: K_func called as pure JAX function (no Python Model rebuild).
Bounds enforced via jnp.clip projection after each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from wht_modeler.wht_mesh_model import WHTMeshModel
    from .wht_result import WHTSolverResult
    from .wht_mapper import WHTMapper
    from .wht_monitor import OptimizationMonitor
    from .load_cases import WHTLoadCase


# ---------------------------------------------------------------------------
# Design variables and bounds
# ---------------------------------------------------------------------------

@dataclass
class DesignVariables:
    """
    JAX pytree-registered design variable set.

    All fields are JAX arrays so jax.grad works end-to-end.
    """
    t_field:   jnp.ndarray   # (M,) element thicknesses [mm]
    z_offsets: jnp.ndarray   # (N,) nodal Z-offsets [mm]
    E:         float          # global Young's modulus [MPa]
    rho:       float          # global density [t/mm³]


@dataclass
class DesignBounds:
    t_min: float;   t_max: float
    z_min: float;   z_max: float
    E_min: float;   E_max: float
    rho_min: float; rho_max: float
    free_node_mask: Optional[np.ndarray] = None  # (N,) bool, sorted_nids order;
                                                  # False = node frozen at z=0


# Register DesignVariables as a JAX pytree.
# NOTE: unflatten must NOT call float() on the E/rho leaves — under
# jax.grad/value_and_grad those leaves are abstract tracers during trace
# construction, and float(tracer) raises ConcretizationTypeError. Keep them
# as 0-d jnp arrays; they still behave like floats in eager (non-traced) use.
jax.tree_util.register_pytree_node(
    DesignVariables,
    lambda dv: ([dv.t_field, dv.z_offsets,
                 jnp.array(dv.E), jnp.array(dv.rho)], None),
    lambda _, leaves: DesignVariables(
        leaves[0], leaves[1], leaves[2], leaves[3]
    ),
)


def clip_to_bounds(dv: DesignVariables, bounds: DesignBounds) -> DesignVariables:
    """Project design variables to feasible region.

    Nodes outside ``bounds.free_node_mask`` are frozen at z_offset=0
    (e.g. nodes outside the pre-formed bead region).
    """
    z_clipped = jnp.clip(dv.z_offsets, bounds.z_min, bounds.z_max)
    if bounds.free_node_mask is not None:
        z_clipped = jnp.where(jnp.asarray(bounds.free_node_mask), z_clipped, 0.0)
    return DesignVariables(
        t_field   = jnp.clip(dv.t_field,   bounds.t_min,   bounds.t_max),
        z_offsets = z_clipped,
        E         = float(jnp.clip(jnp.array(dv.E),
                                   bounds.E_min,   bounds.E_max)),
        rho       = float(jnp.clip(jnp.array(dv.rho),
                                   bounds.rho_min, bounds.rho_max)),
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class WHTOptimizer:
    """
    Optax Adam optimizer for structural shape / material optimization.

    Uses K_func (pure JAX) for gradient computation.
    Supports multi-objective loss: freq + MAC + static RMSE + smooth.

    Requires optax to be installed:  pip install optax
    """

    def __init__(
        self,
        base_model:      "WHTMeshModel",
        target_results:  Dict[str, "WHTSolverResult"],
        mapper:          "WHTMapper",
        bounds:          DesignBounds,
        load_cases:      List["WHTLoadCase"],
        num_modes:       int = 10,
        lr:              float = 1e-3,
        weights:         Dict[str, float] = None,
        monitor:         Optional["OptimizationMonitor"] = None,
        adjacency:       Optional[np.ndarray] = None,
        target_node_coords: Optional[np.ndarray] = None,
    ):
        try:
            import optax
            self._optax = optax
        except ImportError:
            raise ImportError("optax is required: pip install optax")

        self.base_model     = base_model
        self.target_results = target_results
        self.mapper         = mapper
        self.bounds         = bounds
        self.load_cases     = load_cases
        self.num_modes      = num_modes
        self.lr             = lr
        self.weights        = weights or {
            "freq": 1.0, "mac": 1.0, "static": 1.0, "smooth": 0.01
        }
        self.monitor    = monitor
        self.adjacency  = (jnp.array(adjacency) if adjacency is not None
                           else None)
        self._free_mask = (jnp.asarray(bounds.free_node_mask)
                           if bounds.free_node_mask is not None else None)
        self.target_node_coords = (np.asarray(target_node_coords)
                                   if target_node_coords is not None else None)

        # Pre-extract K_func static args (once)
        from .wht_solver import WHTSolver
        self._k_args = WHTSolver(base_model).get_k_func_args()

    def run(
        self,
        init_vars:  DesignVariables,
        n_steps:    int = 500,
        log_every:  int = 10,
        solver_method: str = 'auto',
    ) -> Tuple[DesignVariables, List[float]]:
        """
        Run optimization loop using analytical eigensensitivity.

        Returns
        -------
        (best_vars, loss_history)
        """
        import optax
        from .objectives import multi_objective_loss
        from .wht_eigensolver import make_modal_freq_fn

        optimizer    = optax.adam(self.lr)
        opt_state    = optimizer.init(init_vars)
        current      = init_vars
        loss_history: List[float] = []

        # Build JAX-differentiable frequency function once
        freq_fn = make_modal_freq_fn(
            self.base_model,
            num_modes=self.num_modes,
            solver_method=solver_method,
        )

        # Pre-compute target data
        target_modal = self.target_results.get("modal")
        freqs_target = (jnp.array(target_modal.frequencies[:self.num_modes])
                        if target_modal else None)

        phis_target = None
        if target_modal is not None:
            raw_target_shapes = target_modal.mode_shapes[:self.num_modes, :, :3]  # (n_modes, N_tgt, 3)
            if self.target_node_coords is not None:
                # Target result lives on a different mesh (different node count /
                # ordering) than base_model — RBF-map each mode's displacement
                # field onto base_model's node coordinates so MAC compares
                # vectors of matching length/order. One-time cost (not per-step).
                base_crds = self._k_args["base_crds"]
                mapped = self.mapper.map_modes(
                    self.target_node_coords, raw_target_shapes, base_crds
                )
                phis_target = jnp.array(mapped.reshape(self.num_modes, -1))
            else:
                # Same mesh / matching node ordering assumed.
                phis_target = jnp.array(raw_target_shapes.reshape(self.num_modes, -1))

        def loss_fn(dv: DesignVariables) -> jnp.ndarray:
            freqs_opt = freq_fn(
                dv.t_field,
                dv.z_offsets,
                jnp.array(dv.E),
                jnp.array(dv.rho),
            )
            # Real mode shapes from the forward solve, used only for MAC-based
            # mode soft-assignment (stop_gradient: avoids costly eigenvector
            # sensitivity — gradient still flows through freqs_opt).
            if phis_target is not None:
                shapes_np = freq_fn.get_last_mode_shapes()
                phis_opt = jax.lax.stop_gradient(jnp.array(shapes_np))
            else:
                phis_opt = None

            return multi_objective_loss(
                freqs_opt    = freqs_opt,
                freqs_target = freqs_target,
                phis_opt     = phis_opt,
                phis_target  = phis_target,
                z_offsets    = dv.z_offsets,
                adjacency    = self.adjacency,
                weights      = self.weights,
            )

        loss_and_grad = jax.value_and_grad(loss_fn)

        print(f"WHT Optimizer: {n_steps} steps, lr={self.lr}")
        for step in range(1, n_steps + 1):
            loss_val, grads = loss_and_grad(current)
            if self._free_mask is not None:
                grads = DesignVariables(
                    t_field   = grads.t_field,
                    z_offsets = grads.z_offsets * self._free_mask,
                    E         = grads.E,
                    rho       = grads.rho,
                )
            updates, opt_state = optimizer.update(grads, opt_state, current)
            current = optax.apply_updates(current, updates)
            current = clip_to_bounds(current, self.bounds)

            loss_f = float(loss_val)
            loss_history.append(loss_f)

            if step % log_every == 0 or step == 1:
                print(f"  Step {step:4d}/{n_steps}  loss={loss_f:.6e}")

                if self.monitor is not None:
                    base_crds = self._k_args["base_crds"].copy()
                    nodes_now = base_crds.copy()
                    nodes_now[:, 2] += np.array(current.z_offsets)
                    self.monitor.update(
                        step      = step,
                        nodes     = nodes_now,
                        z_offsets = np.array(current.z_offsets),
                        loss      = loss_f,
                    )

        if self.monitor is not None:
            self.monitor.close()

        return current, loss_history
