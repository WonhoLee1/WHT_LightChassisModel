# -*- coding: utf-8 -*-
"""
compare_fold_hinge_beaded.py
============================
ccx_iter016 실제 비드 패널 모델에 대해
fold_alpha 적용 전/후를 CalculiX 기준값과 비교한다.

CalculiX reference (ccx_iter016_Modal_Analysis.dat):
  Mode 7 = 4.934 Hz ... Mode 20 = 77.908 Hz
"""

from __future__ import annotations

import sys
import time
import numpy as np
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_solver import WHTSolver

# CalculiX elastic mode reference (Hz, modes 7-20)
CCX_FREQS = {
    7:  4.934396,
    8:  15.802510,
    9:  21.932870,
    10: 22.288870,
    11: 34.417100,
    12: 36.455130,
    13: 42.458890,
    14: 52.031450,
    15: 57.561600,
    16: 58.882800,
    17: 71.415390,
    18: 74.992540,
    19: 75.145990,
    20: 77.907610,
}


def _load_mesh(mesh_path: str, model: WHTMeshModel, pid: int = 1):
    with open(mesh_path, encoding='utf-8') as f:
        lines = f.readlines()
    mode = None
    etype = 'QUAD4'
    for line in lines:
        line = line.strip()
        if not line or line.startswith('**'):
            continue
        if line.startswith('*'):
            up = line.upper()
            if up.startswith('*NODE'):
                mode = 'NODE'
            elif up.startswith('*ELEMENT'):
                mode = 'ELEM'
                etype = 'QUAD4' if 'S4' in up else 'TRIA3'
            else:
                mode = None
            continue
        if mode == 'NODE':
            p = [s.strip() for s in line.split(',')]
            if len(p) >= 4:
                model.add_node(int(p[0]), float(p[1]), float(p[2]), float(p[3]))
        elif mode == 'ELEM':
            p = [s.strip() for s in line.split(',')]
            if len(p) >= 4:
                model.add_element(int(p[0]), [int(x) for x in p[1:]], etype, pid=pid)


def _build_model():
    curr = Path(__file__).resolve().parent
    mesh_inp = curr / 'ccx_iter016_Modal_Analysis_mesh.inp'
    model = WHTMeshModel(name='ccx_iter016')
    model.add_material(1, 210000.0, 0.3, 7.85e-9)
    model.add_property(1, 'PSHELL', 0.6, 1)
    _load_mesh(str(mesh_inp), model)
    print(f"  Model: {len(model.nodes)} nodes / {len(model.elements)} elements")
    return model


def _run_modal(model, fold_alpha=0.0, num_modes=22):
    solver = WHTSolver(
        model,
        k_backend='numba',
        fold_alpha=fold_alpha,
        fold_phi_min_deg=3.0,
    )
    t0 = time.perf_counter()
    result = solver.solve_modal(num_modes=num_modes)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return result.frequencies, elapsed


def _pick_elastic(freqs, n_rigid=6):
    """Skip first n_rigid modes (rigid body / near-zero), return elastic modes."""
    elastic = [f for f in freqs if f > 0.5]
    # If fewer than expected, try lower threshold
    if len(elastic) < len(CCX_FREQS):
        elastic = [f for f in freqs if f > 0.05]
    return np.array(elastic)


def main():
    print("=" * 70)
    print("  ccx_iter016 Beaded Panel: fold_hinge Before/After vs CalculiX")
    print("=" * 70)

    model = _build_model()

    # --- Run 1: no fold spring ---
    print("\n[1/2] WHT without fold spring ...")
    freqs_base, t_base = _run_modal(model, fold_alpha=0.0)
    print(f"  Done: {t_base:.0f} ms")

    # --- Run 2: with fold spring alpha=0.5 ---
    print("\n[2/2] WHT with fold_alpha=0.5 ...")
    freqs_hinge, t_hinge = _run_modal(model, fold_alpha=0.5)
    print(f"  Done: {t_hinge:.0f} ms")

    # --- Extract elastic modes ---
    e_base  = _pick_elastic(freqs_base)
    e_hinge = _pick_elastic(freqs_hinge)

    ccx_modes  = sorted(CCX_FREQS.keys())
    ccx_values = [CCX_FREQS[m] for m in ccx_modes]

    n_compare = min(len(ccx_modes), len(e_base), len(e_hinge))

    # --- Print comparison table ---
    print()
    print("=" * 90)
    hdr = (f"  {'Mode':>5}  {'CCX [Hz]':>10}  "
           f"{'WHT-base [Hz]':>13}  {'err_base':>9}  "
           f"{'WHT+hinge [Hz]':>14}  {'err_hinge':>9}  {'delta_err':>9}")
    print(hdr)
    print("  " + "-" * 86)

    errs_base, errs_hinge = [], []
    for i in range(n_compare):
        ccx_f   = ccx_values[i]
        wht_b   = e_base[i]
        wht_h   = e_hinge[i]
        err_b   = (wht_b - ccx_f) / ccx_f * 100.0
        err_h   = (wht_h - ccx_f) / ccx_f * 100.0
        delta   = abs(err_b) - abs(err_h)       # positive = improved
        marker  = " <" if abs(err_h) < abs(err_b) else "  "
        print(f"  {ccx_modes[i]:>5}  {ccx_f:>10.3f}  "
              f"{wht_b:>13.3f}  {err_b:>+8.2f}%  "
              f"{wht_h:>14.3f}  {err_h:>+8.2f}%  {delta:>+8.2f}pp{marker}")
        errs_base.append(abs(err_b))
        errs_hinge.append(abs(err_h))

    print("=" * 90)

    # --- Summary ---
    mae_base  = np.mean(errs_base[:n_compare])
    mae_hinge = np.mean(errs_hinge[:n_compare])
    max_base  = np.max(errs_base[:n_compare])
    max_hinge = np.max(errs_hinge[:n_compare])
    improved  = sum(1 for b, h in zip(errs_base, errs_hinge) if h < b)

    print(f"\n  {'':30s}  {'WHT-base':>12}  {'WHT+hinge(a=0.5)':>16}")
    print(f"  {'Mean Abs Error (modes 7-20)':30s}  {mae_base:>11.2f}%  {mae_hinge:>15.2f}%")
    print(f"  {'Max Abs Error':30s}  {max_base:>11.2f}%  {max_hinge:>15.2f}%")
    print(f"  {'Modes improved':30s}  {'':>12}  {improved}/{n_compare}")
    print(f"  {'Solve time [ms]':30s}  {t_base:>11.0f}   {t_hinge:>14.0f}")

    improvement = mae_base - mae_hinge
    label = "IMPROVED" if improvement > 0 else "WORSENED"
    print(f"\n  [{label}] MAE change: {improvement:+.2f} pp "
          f"({mae_base:.2f}% -> {mae_hinge:.2f}%)")

    return {
        'mae_base': mae_base,
        'mae_hinge': mae_hinge,
        'max_base': max_base,
        'max_hinge': max_hinge,
        'improved_count': improved,
        'n_modes': n_compare,
    }


def benchmark_backends(num_modes=22):
    """
    JAX / NumPy / Numba 세 백엔드에 대해 K 조립 + 고유해 시간을 측정한다.
    fold_alpha=0 (spring 없음) 조건으로 고정.
    """
    from wht_solver.wht_quad4_element import _NUMBA_OK
    try:
        from wht_solver.wht_quad4_element_jax import K_quad4_jax  # noqa
        _jax_ok = True
    except Exception:
        _jax_ok = False

    model = _build_model()
    sorted_nids = sorted(model.nodes.keys())
    nid_to_idx  = {nid: i for i, nid in enumerate(sorted_nids)}

    print()
    print("=" * 70)
    print("  Backend Speed Benchmark  (ccx_iter016, %d nodes / %d elems)" %
          (len(model.nodes), len(model.elements)))
    print("=" * 70)
    print(f"  {'Backend':>8}  {'K assemble [ms]':>16}  "
          f"{'Modal solve [ms]':>16}  {'Total [ms]':>10}  "
          f"{'vs Numba (total)':>16}")
    print(f"  {'-'*8}  {'-'*16}  {'-'*16}  {'-'*10}  {'-'*16}")

    records = {}

    for backend in ('jax', 'numpy', 'numba'):
        if backend == 'jax' and not _jax_ok:
            print(f"  {backend:>8}  [SKIP - JAX unavailable]")
            continue
        if backend == 'numba' and not _NUMBA_OK:
            print(f"  {backend:>8}  [SKIP - Numba unavailable]")
            continue

        # Numba warm-up: first call compiles; use tiny model
        if backend == 'numba':
            _tiny = WHTMeshModel(name='warmup')
            _tiny.add_material(1, 210000.0, 0.3, 7.85e-9)
            _tiny.add_property(1, 'PSHELL', 0.6, 1)
            _tiny.add_node(1, 0, 0, 0); _tiny.add_node(2, 1, 0, 0)
            _tiny.add_node(3, 1, 1, 0); _tiny.add_node(4, 0, 1, 0)
            _tiny.add_element(1, [1, 2, 3, 4], 'QUAD4', pid=1)
            _snids = sorted(_tiny.nodes.keys())
            _sidx  = {n: i for i, n in enumerate(_snids)}
            from wht_solver.wht_quad4_element import K_quad4_scipy
            K_quad4_scipy(_tiny, _snids, _sidx, None, backend='numba')

        # --- K assembly only ---
        t0 = time.perf_counter()
        if backend == 'jax':
            from wht_solver.wht_quad4_element_jax import K_quad4_jax as _kjax
            _kjax(model, sorted_nids, nid_to_idx, None)
        elif backend == 'numba':
            from wht_solver.wht_quad4_element import K_quad4_scipy
            K_quad4_scipy(model, sorted_nids, nid_to_idx, None, backend='numba')
        else:
            from wht_solver.wht_quad4_element import K_quad4_scipy
            K_quad4_scipy(model, sorted_nids, nid_to_idx, None, backend='numpy')
        t_k = (time.perf_counter() - t0) * 1000.0

        # --- full modal solve ---
        solver = WHTSolver(model, k_backend=backend, fold_alpha=0.0)
        t0 = time.perf_counter()
        solver.solve_modal(num_modes=num_modes)
        t_m = (time.perf_counter() - t0) * 1000.0

        records[backend] = {'t_k': t_k, 't_m': t_m, 't_total': t_k + t_m}

    # Print with relative speed vs Numba
    ref_total = records.get('numba', {}).get('t_total', None)
    for backend, r in records.items():
        ratio_str = ''
        if ref_total and ref_total > 0:
            ratio = r['t_total'] / ref_total
            ratio_str = f"{ratio:.2f}x"
        print(f"  {backend:>8}  {r['t_k']:>16.1f}  {r['t_m']:>16.1f}  "
              f"{r['t_total']:>10.1f}  {ratio_str:>16}")

    if 'numba' in records and 'numpy' in records:
        speedup_k = records['numpy']['t_k'] / records['numba']['t_k']
        print(f"\n  Numba K-assembly speedup vs NumPy: {speedup_k:.1f}x")
    if 'numba' in records and 'jax' in records:
        speedup_jax = records['jax']['t_total'] / records['numba']['t_total']
        print(f"  JAX total / Numba total: {speedup_jax:.2f}x  "
              f"({'Numba faster' if speedup_jax > 1 else 'JAX faster'})")

    return records


if __name__ == '__main__':
    results = main()
    print()
    bench = benchmark_backends()
