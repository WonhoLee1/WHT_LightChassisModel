import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# CalculiX 윈도우 버전의 멀티스레딩(libgomp) 충돌 방지를 위해 철저히 차단
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_PROC_BIND"] = "FALSE"

from test_jaxSSO.runners.common_setup import setup_model_and_lcs

_AUTOCALCULIX_SRC = str(Path(__file__).resolve().parent.parent.parent.parent / "AutoCalculix" / "src")
if _AUTOCALCULIX_SRC not in sys.path:
    sys.path.insert(0, _AUTOCALCULIX_SRC)

try:
    from autocalculix_api import run_calculix_analysis
except ImportError:
    run_calculix_analysis = None

def _parse_ccx_static_dat(dat_path: str) -> float:
    if not os.path.exists(dat_path):
        print(f"    [WARN] CCX .dat not found: {dat_path}")
        return 0.0
    with open(dat_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    max_disp = 0.0
    in_disp_table = False
    for line in content.split('\n'):
        lowered = line.lower()
        if 'displacements' in lowered:
            in_disp_table = True
            continue
        if not in_disp_table:
            continue
        stripped = line.strip()
        if not stripped or ('eigenvalue' in lowered and 'node' not in lowered):
            if 'eigenvalue' in lowered:
                break
            continue
        parts = stripped.split()
        if len(parts) >= 4 and parts[0].isdigit():
            try:
                mag = np.sqrt(float(parts[1])**2 + float(parts[2])**2 + float(parts[3])**2)
                if mag > max_disp:
                    max_disp = mag
            except (ValueError, IndexError):
                continue
    return max_disp

def _parse_ccx_stress_frd(frd_path: str) -> dict:
    """FRD에서 Von Mises 통계 파싱. STRESS 블록: sxx,syy,szz,sxy,syz,szx 순.
    반환: {'max_stress': float, 'p95_stress': float, 'median_stress': float,
            'mean_stress': float, 'std_stress': float}
    """
    empty = {'max_stress': 0.0, 'p95_stress': 0.0, 'median_stress': 0.0,
             'mean_stress': 0.0, 'std_stress': 0.0}
    if not os.path.exists(frd_path):
        return empty
    with open(frd_path, 'rb') as f:
        content = f.read().decode('utf-8', errors='replace')
    lines = content.split('\n')
    in_stress = False
    vm_vals = []
    for line in lines:
        if line.startswith(' -4') and 'STRESS' in line:
            in_stress = True
            continue
        if in_stress:
            if line.startswith(' -3') or line.startswith(' -4'):
                in_stress = False
                continue
            if line.startswith(' -5') or not line.startswith(' -1'):
                continue
            # FRD 고정 폭: [-1][  nid(10)][val1(12)][val2(12)][val3(12)][val4(12)][val5(12)][val6(12)]
            if len(line) >= 85:
                try:
                    sxx = float(line[13:25])
                    syy = float(line[25:37])
                    szz = float(line[37:49])
                    sxy = float(line[49:61])
                    syz = float(line[61:73])
                    szx = float(line[73:85])
                    vm = np.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2
                                        + 6*(sxy**2 + syz**2 + szx**2)))
                    vm_vals.append(vm)
                except (ValueError, IndexError):
                    continue
    if not vm_vals:
        return empty
    vm_arr = np.array(vm_vals, dtype=float)
    vm_arr = vm_arr[np.isfinite(vm_arr) & (vm_arr >= 0)]
    if len(vm_arr) == 0:
        return empty
    return {
        'max_stress':    float(np.max(vm_arr)),
        'p95_stress':    float(np.percentile(vm_arr, 95)),
        'median_stress': float(np.median(vm_arr)),
        'mean_stress':   float(np.mean(vm_arr)),
        'std_stress':    float(np.std(vm_arr)),
    }


def run():
    print("[CalculiX] 시작...")
    model, lcs = setup_model_and_lcs()
    
    times = {"Modal": "N/A", "Static": {lc.name: "N/A" for lc, _ in lcs}}
    results = {"Modal": {}, "Static": {lc.name: {} for lc, _ in lcs}}
    
    if run_calculix_analysis is None:
        print("CalculiX API를 불러오지 못했습니다.")
        return
        
    nodes_dict = {nid: (nd.x, nd.y, nd.z) for nid, nd in model.nodes.items()}
    # API 형식: (eid, etype, [node_ids], pid)
    elements_list = []
    for eid, el in model.elements.items():
        if el.type in ('QUAD4', 'QUAD', 'CQUAD4', 'TRIA3', 'TRIA', 'CTRIA3'):
            elements_list.append((eid, el.type, list(el.node_ids), el.pid))
    # API 형식: {pid: (thickness, E, nu, rho)}
    properties_dict = {}
    for pid, prop in model.properties.items():
        mat = model.materials[prop.mid]
        properties_dict[pid] = (prop.t, mat.E, mat.nu, mat.rho)
    ccx_ws_dir = os.path.join(os.path.dirname(__file__), "export_calculix")
    
    # Modal
    try:
        t0 = time.time()
        ccx_res = run_calculix_analysis(
            nodes=nodes_dict, elements=elements_list, properties=properties_dict,
            analysis_type="modal", analysis_config={"num_modes": 26, "job_name": "ccx_modal"},
            bcs=[], forces=[], workspace_dir=ccx_ws_dir
        )
        t_m = time.time() - t0
        ccx_ws = ccx_res.get("workspace", "export_test")
        ccx_job = ccx_res.get("job_name", "ccx_modal")
        ccx_dat = os.path.join(ccx_ws, f"{ccx_job}.dat")
        freqs_raw = ccx_res.get("frequencies", [])[6:]
        freqs_flat = [float(f["hz"] if isinstance(f, dict) else f) for f in freqs_raw]
        
        results["Modal"]["freqs"] = freqs_flat[:20]
        results["Modal"]["shapes"] = [] # 모달 형상은 벤치마크 시간 상 생략
        times["Modal"] = t_m
    except Exception as e:
        print(f"Modal 실패: {e}")
        
    # Static
    for lc, _ in lcs:
        try:
            t0 = time.time()
            bcs_ccx = []
            forces_ccx = []
            for bc in lc.bcs:
                bcs_ccx.append((bc.node_id, list(bc.dofs), getattr(bc, "value", 0.0)))
            for fc in lc.forces:
                forces_ccx.append((fc.node_id, list(fc.load_vector)))
                
            ccx_res = run_calculix_analysis(
                nodes=nodes_dict, elements=elements_list, properties=properties_dict,
                analysis_type="static", analysis_config={"job_name": f"ccx_{lc.name}"},
                bcs=bcs_ccx, forces=forces_ccx, workspace_dir=ccx_ws_dir
            )
            t_s = time.time() - t0
            ccx_ws = ccx_res.get("workspace", "export_test")
            ccx_job = ccx_res.get("job_name", f"ccx_{lc.name}")
            ccx_dat = os.path.join(ccx_ws, f"{ccx_job}.dat")
            max_disp = float(_parse_ccx_static_dat(ccx_dat))
            ccx_frd = os.path.join(ccx_ws, f"{ccx_job}.frd")
            stress_stats = _parse_ccx_stress_frd(ccx_frd)

            results["Static"][lc.name]["max_disp"]      = max_disp
            results["Static"][lc.name]["max_stress"]    = stress_stats["max_stress"]
            results["Static"][lc.name]["p95_stress"]    = stress_stats["p95_stress"]
            results["Static"][lc.name]["median_stress"] = stress_stats["median_stress"]
            results["Static"][lc.name]["mean_stress"]   = stress_stats["mean_stress"]
            results["Static"][lc.name]["std_stress"]    = stress_stats["std_stress"]
            times["Static"][lc.name] = t_s
        except Exception as e:
            print(f"Static({lc.name}) 실패: {e}")
            
    out_dict = {"times": times, "results": results}
    out_path = os.path.join(os.path.dirname(__file__), "CalculiX_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"[CalculiX] 완료. {out_path}")

if __name__ == "__main__":
    run()
