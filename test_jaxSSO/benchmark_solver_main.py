import os
import sys
import time
import json
import subprocess
import argparse
import numpy as np
from pathlib import Path

# 공통 설정을 위해 import
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from test_jaxSSO.runners.common_setup import setup_model_and_lcs

def compute_mac(mode_shapes1, mode_shapes2):
    n1 = len(mode_shapes1)
    n2 = len(mode_shapes2)
    mac_matrix = np.zeros((n1, n2))
    
    for i in range(n1):
        v1 = np.array(mode_shapes1[i]).flatten()
        norm1 = np.linalg.norm(v1)
        if norm1 < 1e-12: continue
        
        for j in range(n2):
            v2 = np.array(mode_shapes2[j]).flatten()
            norm2 = np.linalg.norm(v2)
            if norm2 < 1e-12: continue
            
            mac = (np.dot(v1, v2)**2) / (norm1**2 * norm2**2)
            mac_matrix[i, j] = mac
    return mac_matrix

def main():
    parser = argparse.ArgumentParser(description="JAX SSO Solver Benchmark Orchestrator")
    parser.add_argument("--parallel", action="store_true", help="Run all solver scripts in parallel (Warning: may skew timing results)")
    args = parser.parse_args()

    # 결과 취합을 위해 공통 하중 케이스 목록만 가져오기
    model, lcs = setup_model_and_lcs()
    
    solvers = ["WHT_Scipy", "WHT_Pardiso", "CalculiX", "MYSTRAN"]
    script_map = {
        "WHT_Scipy": "run_wht_scipy.py",
        "WHT_Pardiso": "run_wht_pardiso.py",
        "CalculiX": "run_calculix.py",
        "MYSTRAN": "run_mystran.py",
    }
    
    runners_dir = os.path.join(os.path.dirname(__file__), "runners")
    python_exe = sys.executable
    
    # 하위 프로세스들이 jaxSSO 등을 찾을 수 있도록 PYTHONPATH에 최상위 폴더 추가
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{project_root};{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = project_root
    
    processes = []
    
    print(f"=== JAX SSO Benchmark Orchestrator ===")
    print(f"Execution Mode: {'Parallel' if args.parallel else 'Sequential'}")
    
    if args.parallel:
        print("Starting all solvers simultaneously...")
        for s in solvers:
            script_path = os.path.join(runners_dir, script_map[s])
            p = subprocess.Popen([python_exe, script_path], env=env)
            processes.append((s, p))
            
        for s, p in processes:
            p.wait()
            print(f"[{s}] Process completed with exit code {p.returncode}.")
    else:
        for s in solvers:
            print(f"Starting {s}...")
            script_path = os.path.join(runners_dir, script_map[s])
            p = subprocess.Popen([python_exe, script_path], env=env)
            p.wait()
            print(f"[{s}] Process completed with exit code {p.returncode}.\n")

    # 결과 취합
    print("Aggregating results...")
    times = {"Modal": {}, "Static": {lc.name: {} for lc, _ in lcs}}
    results = {"Modal": {}, "Static": {lc.name: {} for lc, _ in lcs}}
    
    for s in solvers:
        res_file = os.path.join(runners_dir, f"{s}_results.json")
        if os.path.exists(res_file):
            with open(res_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                times["Modal"][s] = data["times"]["Modal"]
                results["Modal"][s] = data["results"]["Modal"]
                for lc, _ in lcs:
                    if lc.name in data["times"]["Static"]:
                        times["Static"][lc.name][s] = data["times"]["Static"][lc.name]
                        results["Static"][lc.name][s] = data["results"]["Static"][lc.name]
        else:
            print(f"Warning: {res_file} not found. {s} results will be N/A.")
            
    # 마크다운 리포트 작성
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "dev_log"))
    os.makedirs(log_dir, exist_ok=True)
    date_str = time.strftime("%Y%m%d")
    report_path = os.path.join(log_dir, f"mystran_benchmark_{date_str}.md")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# JAX SSO Multi-Agent Benchmark Report\n\n")
        
        # 1. 시간 표
        f.write("## 1. 수행 시간 비교 (단위: 초)\n\n")
        f.write("| Solver | Modal |")
        for lc, _ in lcs:
            f.write(f" Static({lc.name}) |")
        f.write("\n|---|---|")
        for lc, _ in lcs:
            f.write("---|")
        f.write("\n")
        
        for s in solvers:
            t_m = times["Modal"].get(s, "N/A")
            if isinstance(t_m, float): t_m = f"{t_m:.3f}"
            line = f"| {s} | {t_m} |"
            for lc, _ in lcs:
                t_s = times["Static"][lc.name].get(s, "N/A")
                if isinstance(t_s, float): t_s = f"{t_s:.3f}"
                line += f" {t_s} |"
            f.write(line + "\n")
            
        # 2. 정해석 결과 표
        for metric, label, fmt in [
            ("max_disp",      "Max Displacement [mm]",    ".4e"),
            ("max_stress",    "Max Von-Mises [MPa]",      ".2f"),
            ("p95_stress",    "P95 Von-Mises [MPa]",      ".2f"),
            ("median_stress", "Median Von-Mises [MPa]",   ".2f"),
            ("mean_stress",   "Mean Von-Mises [MPa]",     ".2f"),
            ("std_stress",    "Std Von-Mises [MPa]",      ".2f"),
        ]:
            f.write(f"\n## 2. 정해석 결과 — {label}\n\n")
            f.write("| Solver |")
            for lc, _ in lcs:
                f.write(f" {lc.name} |")
            f.write("\n|---|")
            for lc, _ in lcs:
                f.write("---|")
            f.write("\n")
            for s in solvers:
                line = f"| {s} |"
                for lc, _ in lcs:
                    v = results["Static"][lc.name].get(s, {}).get(metric, "N/A")
                    if isinstance(v, float):
                        v = f"{v:{fmt}}"
                    line += f" {v} |"
                f.write(line + "\n")
            
        # 3. 모달 표
        f.write("\n## 3. 모달 결과 (Frequencies [Hz])\n\n")
        f.write("| Mode | " + " | ".join(solvers) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(solvers)) + "|\n")
        
        line_time = "| **Time (s)** |"
        for s in solvers:
            t_m = times["Modal"].get(s, "N/A")
            if isinstance(t_m, float): t_m = f"{t_m:.3f}"
            line_time += f" {t_m} |"
        f.write(line_time + "\n")
        
        for m in range(20):
            line = f"| {m+1} |"
            for s in solvers:
                freqs = results["Modal"].get(s, {}).get("freqs", [])
                if len(freqs) > m:
                    line += f" {freqs[m]:.3f} |"
                else:
                    line += " N/A |"
            f.write(line + "\n")
            
        # 4. MAC Calculation between WHT_Scipy and MYSTRAN
        f.write("\n## 4. MAC (Modal Assurance Criterion) - WHT_Scipy vs MYSTRAN\n\n")
        shapes_wht = results["Modal"].get("WHT_Scipy", {}).get("shapes", [])
        shapes_mys = results["Modal"].get("MYSTRAN", {}).get("shapes", [])

        if shapes_wht and shapes_mys and len(shapes_wht) == len(shapes_mys):
            mac = compute_mac(shapes_wht, shapes_mys)
            n_modes = len(shapes_wht)

            # MAC 히트맵 PNG 생성
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(mac, cmap='hot', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel('MYSTRAN Mode')
            ax.set_ylabel('WHT Mode')
            ax.set_title('MAC Matrix: WHT_Scipy vs MYSTRAN')
            for i in range(n_modes):
                for j in range(n_modes):
                    ax.text(j, i, f'{mac[i, j]:.2f}', ha='center', va='center',
                            color='white' if mac[i, j] < 0.5 else 'black', fontsize=6)
            plt.tight_layout()
            png_path = os.path.join(log_dir, "mac_wht_vs_mystran.png")
            plt.savefig(png_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"MAC PNG 저장: {png_path}")

            diag_mac = np.array([mac[m, m] for m in range(n_modes)])
            mean_diag = float(np.mean(diag_mac))
            min_diag  = float(np.min(diag_mac))

            f.write("![MAC Matrix](mac_wht_vs_mystran.png)\n\n")
            f.write("| | 대각선 평균 MAC | 최소 대각 MAC |\n")
            f.write("|--|---|---|\n")
            f.write(f"| WHT_Scipy vs MYSTRAN | {mean_diag:.4f} | {min_diag:.4f} |\n\n")
            f.write("| Mode | MAC |\n")
            f.write("|---|---|\n")
            for m in range(n_modes):
                f.write(f"| {m+1} | {mac[m, m]:.4f} |\n")
        else:
            f.write("모드형상 데이터 없음 (shapes가 비어있거나 모드 수 불일치).\n")
            
    print(f"\nReport successfully saved to {report_path}")

if __name__ == "__main__":
    main()
