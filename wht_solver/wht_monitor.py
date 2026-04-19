"""
wht_monitor.py
==============
WHT FEM Framework — Real-time Optimization Monitor (PyVistaQt)

OptimizationMonitor displays mesh shape and Z-offset scalar field
in a PyVistaQt BackgroundPlotter during the optimization loop.

Architecture
------------
- BackgroundPlotter runs Qt event loop in a background thread.
- Main optimization loop calls monitor.update() directly.
- Thread safety is handled by pyvistaqt's internal Qt signals.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from wht_modeler.wht_mesh_model import WHTMeshModel


class OptimizationMonitor:
    """
    Real-time visualization of optimization progress via PyVistaQt.

    Usage
    -----
        monitor = OptimizationMonitor(update_every=10)
        monitor.init_mesh(base_model)         # opens Qt window
        # inside optimizer.run():
        monitor.update(step, nodes, z_offsets, loss)
        monitor.close()                       # keep window open for review
    """

    def __init__(
        self,
        update_every: int = 10,
        window_title: str = "WHT Optimization Monitor",
        show_loss: bool = True,
    ):
        self.update_every = update_every
        self.window_title = window_title
        self.show_loss    = show_loss

        self._plotter     = None
        self._mesh        = None
        self._loss_history: list[float] = []
        self._step_history: list[int]   = []
        self._initialized  = False

    def init_mesh(self, model: "WHTMeshModel") -> None:
        """
        Build initial PyVista mesh from WHTMeshModel and open Qt window.

        Must be called before the optimization loop starts.
        """
        try:
            import pyvista as pv
            from pyvistaqt import BackgroundPlotter
        except ImportError as e:
            print(f"[WHTMonitor] PyVistaQt not available: {e}")
            return

        nodes_arr = model.nodes_array()   # (N, 3)
        sorted_nids = model.sorted_node_ids()
        nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}

        cells = []
        cell_types = []
        for eid in sorted(model.elements.keys()):
            elem = model.elements[eid]
            remapped = [nid_to_idx[n] for n in elem.node_ids]
            cells.append(len(remapped))
            cells.extend(remapped)
            cell_types.append(9 if len(remapped) == 4 else 5)

        self._mesh = pv.UnstructuredGrid(cells, cell_types, nodes_arr)
        self._mesh["z_offset"] = np.zeros(len(nodes_arr))

        pv.set_plot_theme("dark")
        self._plotter = BackgroundPlotter(title=self.window_title, show=True)
        self._plotter.add_mesh(
            self._mesh,
            scalars="z_offset",
            cmap="coolwarm",
            show_edges=True,
            clim=[-10.0, 10.0],
        )
        self._plotter.add_axes()
        self._plotter.show_grid(color="white")
        self._plotter.add_text("Step: 0  Loss: —", name="status",
                               position="upper_edge", font_size=10)
        self._plotter.reset_camera()
        self._initialized = True
        print(f"[WHTMonitor] Window opened: '{self.window_title}'")

    def update(
        self,
        step:      int,
        nodes:     np.ndarray,     # (N, 3) current node coordinates
        z_offsets: np.ndarray,     # (N,) Z-offset scalar field
        loss:      float,
    ) -> None:
        """
        Update mesh coordinates and scalar field.

        Called from the main optimization loop.
        Safe to call from any thread via pyvistaqt Qt signals.
        """
        if not self._initialized or self._plotter is None:
            return

        self._mesh.points            = np.asarray(nodes,     dtype=np.float64)
        self._mesh["z_offset"]       = np.asarray(z_offsets, dtype=np.float64)
        self._loss_history.append(float(loss))
        self._step_history.append(step)

        try:
            self._plotter.update_scalars(
                np.asarray(z_offsets, dtype=np.float64),
                mesh=self._mesh,
            )
            self._plotter.update_text(
                f"Step: {step}  Loss: {loss:.4f}",
                name="status",
            )
            self._plotter.update()
        except Exception:
            # Silently ignore render errors mid-loop
            pass

    def close(self) -> None:
        """Keep window open after optimization (user closes manually)."""
        # Don't call plotter.close() — leave window for final inspection
        if self._initialized:
            print("[WHTMonitor] Optimization complete. Window remains open.")

    @property
    def loss_history(self) -> list[float]:
        return list(self._loss_history)
