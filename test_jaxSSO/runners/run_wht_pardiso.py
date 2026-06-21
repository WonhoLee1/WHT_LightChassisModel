import os
import sys
import json
import time
import numpy as np

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_PROC_BIND"] = "FALSE"

from test_jaxSSO.runners.common_setup import setup_model_and_lcs
from wht_solver.wht_solver import WHTSolver

def run():
    print("[WHT_Pardiso] 시작...")
    model, lcs = setup_model_and_lcs()
    
    times = {"Modal": "N/A", "Static": {lc.name: "N/A" for lc, _ in lcs}}
    results = {"Modal": {}, "Static": {lc.name: {} for lc, _ in lcs}}
    
    # Modal
    try:
        t0 = time.time()
        solver = WHTSolver(model)
        res_modal = solver.solve_modal(num_modes=20, method="pardiso",
                                       exclude_rigid_body="cutoff:0.1")
        t_m = time.time() - t0

        freqs = res_modal.frequencies[:20]
        shapes = res_modal.mode_shapes[:20, :, :3]
        results["Modal"]["freqs"] = freqs.tolist()
        results["Modal"]["shapes"] = [s.tolist() for s in shapes]
        times["Modal"] = t_m
    except Exception as e:
        print(f"Modal 실패: {e}")

    # Static
    for lc, _ in lcs:
        try:
            t0 = time.time()
            solver = WHTSolver(model)
            res_static = solver.solve_static(lc, solver_method="pardiso")
            t_s = time.time() - t0

            times["Static"][lc.name] = t_s

            max_disp = float(np.max(np.linalg.norm(res_static.displacement[:, :3], axis=1)))
            results["Static"][lc.name]["max_disp"] = max_disp

            s_upper = res_static.cell_data.get("Stress", [None])[0]
            if s_upper is not None:
                vm = np.sqrt(0.5 * (
                    (s_upper[:, 0] - s_upper[:, 1]) ** 2 +
                    (s_upper[:, 1] - s_upper[:, 2]) ** 2 +
                    (s_upper[:, 2] - s_upper[:, 0]) ** 2 +
                    6.0 * (s_upper[:, 3] ** 2 + s_upper[:, 4] ** 2 + s_upper[:, 5] ** 2)
                ))
                results["Static"][lc.name]["max_stress"]    = float(np.max(vm))
                results["Static"][lc.name]["p95_stress"]    = float(np.percentile(vm, 95))
                results["Static"][lc.name]["median_stress"] = float(np.median(vm))
                results["Static"][lc.name]["mean_stress"]   = float(np.mean(vm))
                results["Static"][lc.name]["std_stress"]    = float(np.std(vm))
            else:
                results["Static"][lc.name]["max_stress"] = 0.0

        except Exception as e:
            print(f"Static({lc.name}) 실패: {e}")
            
    out_dict = {"times": times, "results": results}
    out_path = os.path.join(os.path.dirname(__file__), "WHT_Pardiso_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"[WHT_Pardiso] 완료. {out_path}")

if __name__ == "__main__":
    run()
