# -*- coding: utf-8 -*-
"""
benchmark_whtsolver_modal.py
============================
기본 샤시 모델(ccx_iter016)에 대한 WHT Solver와 CalculiX 솔버의 모달 해석 결과를 비교 및 검증합니다.
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add workspace to path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_solver import WHTSolver

# Add AutoCalculix to path
if "D:/PythonCodeStudy/AutoCalculix" not in sys.path:
    sys.path.append("D:/PythonCodeStudy/AutoCalculix")
from src.core.solver import CalculixSolver
from src.core.dat_parser import CalculixDatParser
from src.core.config import CALCULIX_EXE


def load_ccx_mesh_into_wht(mesh_inp_path: str, model: WHTMeshModel, default_pid: int = 1):
    """
    CalculiX mesh.inp 파일을 읽어 WHTMeshModel에 노드와 요소를 추가합니다.
    """
    with open(mesh_inp_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_mode = None  # "NODE", "ELEMENT"
    element_type = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith('**'):
            continue

        if line.startswith('*'):
            upper_line = line.upper()
            if upper_line.startswith('*NODE'):
                current_mode = "NODE"
            elif upper_line.startswith('*ELEMENT'):
                current_mode = "ELEMENT"
                element_type = "QUAD4" if "S4" in upper_line else "TRIA3"
            else:
                current_mode = None
            continue

        if current_mode == "NODE":
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                nid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                model.add_node(nid, x, y, z)

        elif current_mode == "ELEMENT":
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                eid = int(parts[0])
                conn = [int(p) for p in parts[1:]]
                model.add_element(eid, conn, element_type, pid=default_pid)


def main():
    print("="*60)
    print("   WHT Solver vs CalculiX Modal Analysis Benchmark")
    print("="*60)

    # 1. 파일 경로 정의
    curr_dir = Path(__file__).resolve().parent
    mesh_inp = curr_dir / "ccx_iter016_Modal_Analysis_mesh.inp"
    master_inp = curr_dir / "ccx_iter016_Modal_Analysis.inp"

    if not mesh_inp.exists() or not master_inp.exists():
        print(f"[Error] 필수 파일이 존재하지 않습니다. 복사 여부를 확인해 주세요.\n- Mesh: {mesh_inp}\n- Master: {master_inp}")
        return

    # 2. WHT Solver 모델 빌드 및 해석
    print("\n[WHT Solver] 1. WHTMeshModel 빌드 중...")
    model = WHTMeshModel(name="Chassis_Benchmark")
    model.add_material(1, 210000.0, 0.3, 7.85e-9)
    model.add_property(1, "PSHELL", 0.6, 1) # 두께 0.6mm, 재질 ID 1

    load_ccx_mesh_into_wht(str(mesh_inp), model, default_pid=1)
    print(f"   -> 노드 개수: {len(model.nodes)}개")
    print(f"   -> 요소 개수: {len(model.elements)}개")

    print("\n[WHT Solver] 2. 모달 해석 수행 중...")
    solver = WHTSolver(model)
    t0 = time.perf_counter()
    wht_res = solver.solve_modal(num_modes=20, method='auto', exclude_rigid_body=False)
    wht_time = (time.perf_counter() - t0) * 1000.0
    print(f"   -> 해석 소요 시간: {wht_time:.2f} ms")

    # WHT Solver 결과 리포트 저장 (CalculiX 형식)
    wht_report_path = curr_dir / "wht_iter016_Modal_Analysis.dat"
    wht_res.save_modal_report(str(wht_report_path))

    # 3. CalculiX 모달 해석 수행
    print("\n[CalculiX] 1. 솔버 구동 중...")
    ccx_solver = CalculixSolver(CALCULIX_EXE)
    t0 = time.perf_counter()
    ccx_solver.run("ccx_iter016_Modal_Analysis", str(curr_dir))
    ccx_time = (time.perf_counter() - t0) * 1000.0
    print(f"   -> 해석 소요 시간: {ccx_time:.2f} ms")

    # CalculiX 결과 .dat 파싱
    ccx_dat_path = curr_dir / "ccx_iter016_Modal_Analysis.dat"
    ccx_freqs = []
    if ccx_dat_path.exists():
        parser = CalculixDatParser()
        ccx_freqs = parser.extract_frequencies(ccx_dat_path)
    else:
        print("[Warning] CalculiX .dat 결과 파일을 찾을 수 없습니다.")

    # 4. 결과 비교 및 대조
    print("\n" + "="*80)
    print(f" {'MODE':^6} | {'WHT Solver (Hz)':^18} | {'CalculiX (Hz)':^18} | {'Error (%)':^15}")
    print("-"*80)

    n_compare = min(len(wht_res.frequencies), len(ccx_freqs))
    max_elastic_err = 0.0

    for idx in range(n_compare):
        wht_f = wht_res.frequencies[idx]
        ccx_f = ccx_freqs[idx]["hz"]

        if abs(ccx_f) < 1e-3:
            err = 0.0 if abs(wht_f) < 1e-3 else 100.0
        else:
            err = abs(wht_f - ccx_f) / ccx_f * 100.0

        if idx >= 6 and abs(ccx_f) > 0.1:  # 탄성 모드 영역에 대해서만 에러 추적
            max_elastic_err = max(max_elastic_err, err)

        print(f" {idx+1:^6d} | {wht_f:18.4f} | {ccx_f:18.4f} | {err:14.2f}%")

    print("="*80)
    print(f"   -> 탄성 모드(Mode 7~20) 최대 오차율: {max_elastic_err:.2f}%")
    print(f"   -> WHT Solver .dat 리포트 저장 위치: {wht_report_path}")
    print("="*80)


if __name__ == "__main__":
    main()
