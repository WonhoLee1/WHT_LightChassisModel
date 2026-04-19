# -*- coding: utf-8 -*-
"""
exam2_shell_jaxSSO.py
Shell modal analysis pipeline for QUAD4 and TRIA3 meshes.
Shared pipeline; only mesh generation differs.
"""
import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_jaxSSO.mesh_utils import generate_shell_tray
from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_modeler.wht_entities import WHTSPCEntry
from wht_solver.wht_solver import WHTSolver
from wht_solver.wht_tria3_element import M_tria3_lumped
from wht_visualizer.wht_visualizer import WHTVisualizer
from wht_converter.wht_adapters import JaxSSOAdapter
from wht_converter.wht_models import WHTMetadata


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    mesh_type: str = 'quad4'          # 'quad4' or 'tria3'
    width: float = 1800.0
    length: float = 1200.0
    height: float = 10.0
    thickness: float = 0.6
    mesh_size_xy: float = 40.0
    draft_angle: float = 15.0
    flange_width: float = 10.0
    E: float = 210.0e3                # MPa
    nu: float = 0.3
    rho: float = 7.85e-9              # ton/mm^3
    num_modes: int = 16


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def generate_mesh(cfg: PipelineConfig):
    print(f" -> Generating {cfg.mesh_type.upper()} mesh "
          f"(W={cfg.width}, L={cfg.length}, H={cfg.height})...")
    return generate_shell_tray(
        width=cfg.width,
        length=cfg.length,
        height=cfg.height,
        mesh_size_xy=cfg.mesh_size_xy,
        draft_angle=cfg.draft_angle,
        flange_width=cfg.flange_width,
        origin='center',
        mesh_type=cfg.mesh_type,
    )


def build_model(node_db, elem_db, cfg: PipelineConfig) -> WHTMeshModel:
    model = WHTMeshModel(name=f"Shell_Tray_{cfg.mesh_type.upper()}")
    model.add_material(1, cfg.E, cfg.nu, cfg.rho)
    model.add_property(1, "PSHELL", cfg.thickness, 1)

    for nid, coords in node_db.items():
        model.add_node(nid, *coords)

    for eid, nids in elem_db.items():
        etype = "TRIA3" if len(nids) == 3 else "QUAD4"
        model.add_element(eid, nids, etype, pid=1)

    # Fixed boundary: top flange (z == height)
    fixed_count = 0
    for nid, node in model.nodes.items():
        if abs(node.z - cfg.height) < 0.1:
            model.spc_conditions.append(WHTSPCEntry(nid, (0, 1, 2, 3, 4, 5)))
            fixed_count += 1
    print(f"    Nodes constrained: {fixed_count}")
    return model


def solve_modal(model: WHTMeshModel, cfg: PipelineConfig):
    print(" -> Solving modal...")
    solver = WHTSolver(model)

    # Sanity: total mass
    jm_tmp, _, _ = solver._build_jaxsso_model()
    jm_tmp.model_ready()
    sorted_nids = model.sorted_node_ids()
    nid_to_idx  = {nid: i for i, nid in enumerate(sorted_nids)}
    M_diag = solver._assemble_lumped_mass(jm_tmp, jm_tmp.ndof, sorted_nids, nid_to_idx)
    M_diag += M_tria3_lumped(model, jm_tmp.ndof, sorted_nids, nid_to_idx)
    total_m = np.sum(M_diag[::6])
    print(f"    Total shell mass: {total_m:.6f} tons")

    return solver.solve_modal(num_modes=cfg.num_modes)


def print_results(results, cfg: PipelineConfig):
    print(f"\n[{cfg.mesh_type.upper()}] Natural Frequencies (Hz):")
    for i, f in enumerate(results.frequencies):
        print(f"  Mode {i+1:2d}: {f:8.2f} Hz")


def export_results(model: WHTMeshModel, results, _cfg: PipelineConfig):
    meta = WHTMetadata(
        solver_name="JaxSSO",
        solver_version="1.0.0",
        analysis_type="modal",
        coordinate_system="cartesian",
        unit_length="mm",
        unit_force="N",
    )
    return results.to_wht_result_data(meta, model)


def visualize(wht_data, cfg: PipelineConfig):
    print(" -> Launching visualizer...")
    viz = WHTVisualizer(
        title=f"Shell Modal Analysis ({cfg.mesh_type.upper()})", show=True
    )
    viz.load_results(wht_data, color="black",
                     label=f"Shell Modal Shape ({cfg.mesh_type.upper()})")
    viz.plotter.view_isometric()
    viz.plotter.reset_camera()
    if hasattr(viz.plotter, 'app'):
        viz.plotter.app.exec_()


def run_pipeline(cfg: PipelineConfig):
    import traceback
    print("\n" + "=" * 60)
    print(f" [Exam 2] Shell Modal Analysis - {cfg.mesh_type.upper()}")
    print("=" * 60)

    try:
        node_db, elem_db = generate_mesh(cfg)
        model = build_model(node_db, elem_db, cfg)
        results = solve_modal(model, cfg)
        print_results(results, cfg)
        wht_data = export_results(model, results, cfg)
        visualize(wht_data, cfg)
        return results
    except BaseException:
        print(f"\n[ERROR] {cfg.mesh_type.upper()} pipeline failed:")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Entry point: run both mesh types and compare
# ---------------------------------------------------------------------------

def main():
    cfgs = [        
        PipelineConfig(mesh_type='quad4'),
        PipelineConfig(mesh_type='tria3'),
        PipelineConfig(mesh_type='mixed'),
        PipelineConfig(mesh_type='tria3_free'),
    ]
    results = [run_pipeline(cfg) for cfg in cfgs]

    valid = [(cfg, r) for cfg, r in zip(cfgs, results) if r is not None]
    if not valid:
        print("\n[ERROR] All pipelines failed.")
        return

    labels = [cfg.mesh_type.upper() for cfg, _ in valid]
    ref = valid[0][1].frequencies  # first successful result as reference

    print("\n" + "=" * 72)
    print(" Frequency Comparison (Hz)  [Diff% relative to first successful]")
    header = f"  {'Mode':>4}" + "".join(f"  {lb:>10}" for lb in labels)
    header += "".join(f"  {'Diff%_'+lb:>10}" for lb in labels[1:])
    print(header)
    print("-" * 72)
    for i, freqs in enumerate(zip(*[r.frequencies for _, r in valid])):
        row = f"  {i+1:4d}" + "".join(f"  {f:10.2f}" for f in freqs)
        for f in freqs[1:]:
            diff = abs(f - ref[i]) / max(abs(ref[i]), 1e-12) * 100.0
            row += f"  {diff:10.2f}%"
        print(row)


if __name__ == "__main__":
    main()
