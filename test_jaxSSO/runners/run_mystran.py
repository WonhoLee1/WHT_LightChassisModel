import os
import sys
import json
import time

from test_jaxSSO.runners.common_setup import setup_model_and_lcs
try:
    from wht_solver.mystran_api import MystranAPI
except ImportError:
    MystranAPI = None

def run():
    print("[MYSTRAN] 시작...")
    model, lcs = setup_model_and_lcs()
    
    times = {"Modal": "N/A", "Static": {lc.name: "N/A" for lc, _ in lcs}}
    results = {"Modal": {}, "Static": {lc.name: {} for lc, _ in lcs}}
    
    if MystranAPI is None:
        print("MYSTRAN API를 불러오지 못했습니다.")
        return
        
    mystran_api = MystranAPI(model)
    
    # Modal
    try:
        t0 = time.time()
        res_modal = mystran_api.run_analysis("modal", num_modes=26)
        t_m = time.time() - t0
        
        all_freqs = res_modal['frequencies'].tolist()
        all_shapes = res_modal['mode_shapes']
        # 0.1 Hz 이하 강체 모드 제거 후 최대 20개
        elastic_pairs = [(f, s) for f, s in zip(all_freqs, all_shapes) if f > 0.1][:20]
        results["Modal"]["freqs"] = [p[0] for p in elastic_pairs]
        results["Modal"]["shapes"] = [p[1].tolist() for p in elastic_pairs]
        times["Modal"] = t_m
    except Exception as e:
        print(f"Modal 실패: {e}")
        
    # Static
    for lc, _ in lcs:
        try:
            t0 = time.time()
            res_static = mystran_api.run_analysis("static", load_case=lc)
            t_s = time.time() - t0
            
            times["Static"][lc.name] = t_s
            results["Static"][lc.name]["max_disp"]      = float(res_static.get("max_disp", 0.0))
            results["Static"][lc.name]["max_stress"]    = float(res_static.get("max_stress", 0.0))
            results["Static"][lc.name]["p95_stress"]    = float(res_static.get("p95_stress", 0.0))
            results["Static"][lc.name]["median_stress"] = float(res_static.get("median_stress", 0.0))
            results["Static"][lc.name]["mean_stress"]   = float(res_static.get("mean_stress", 0.0))
            results["Static"][lc.name]["std_stress"]    = float(res_static.get("std_stress", 0.0))
        except Exception as e:
            print(f"Static({lc.name}) 실패: {e}")
            
    out_dict = {"times": times, "results": results}
    out_path = os.path.join(os.path.dirname(__file__), "MYSTRAN_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"[MYSTRAN] 완료. {out_path}")

if __name__ == "__main__":
    run()
