# -*- coding: utf-8 -*-
"""
test_ccx_integration.py
=======================
CalculiX API 및 연동 규격 검증용 스크립트.
임의의 4절점 셸 플레이트 모델을 정의하여, AutoCalculix 공용 API인 
`run_calculix_analysis`가 외부에서 정상 임포트되고 해석을 성공적으로 수행하는지 테스트합니다.
"""

import sys
import os
from pathlib import Path

# D:\PythonCodeStudy\AutoCalculix 경로 주입 및 임포트 테스트
AUTOCALCULIX_PATH = "D:/PythonCodeStudy/AutoCalculix"
if AUTOCALCULIX_PATH not in sys.path:
    sys.path.append(AUTOCALCULIX_PATH)

try:
    from src.autocalculix_api import run_calculix_analysis
    print("[Test] AutoCalculix API imported successfully!")
except ImportError as e:
    print(f"[Test] [Error] Failed to import API: {e}")
    sys.exit(1)


def run_test():
    # 1. 테스트용 플레이트 절점 및 요소 데이터 정의 (2x2 mesh, 9 nodes, 4 elements)
    # 노드 좌표 {nid: (x, y, z)} (mm 단위)
    nodes = {
        1: (0.0,   0.0,   0.0),
        2: (50.0,  0.0,   0.0),
        3: (100.0, 0.0,   0.0),
        4: (0.0,   25.0,  0.0),
        5: (50.0,  25.0,  0.0),
        6: (100.0, 25.0,  0.0),
        7: (0.0,   50.0,  0.0),
        8: (50.0,  50.0,  0.0),
        9: (100.0, 50.0,  0.0),
    }

    # 요소 목록 [(eid, etype, [node_ids], pid)]
    elements = [
        (1, "QUAD4", [1, 2, 5, 4], 1),
        (2, "QUAD4", [2, 3, 6, 5], 1),
        (3, "QUAD4", [4, 5, 8, 7], 1),
        (4, "QUAD4", [5, 6, 9, 8], 1),
    ]

    # 두께 및 재료 {pid: (thickness, E, nu, rho)}
    properties = {
        1: (2.0, 210000.0, 0.3, 7.85e-9) # Steel
    }

    # 작업 공간 경로
    scratch_dir = Path(__file__).resolve().parent / "test_workspace"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- [Test 1] Running Modal Analysis ---")
    analysis_config_modal = {
        "job_name": "test_modal",
        "num_modes": 5
    }
    
    try:
        modal_res = run_calculix_analysis(
            nodes=nodes,
            elements=elements,
            properties=properties,
            analysis_type="modal",
            analysis_config=analysis_config_modal,
            workspace_dir=str(scratch_dir)
        )
        print("\n[Test 1 Results] Success!")
        print(f"  Job Name : {modal_res.get('job_name')}")
        print(f"  Workspace: {modal_res.get('workspace')}")
        print(f"  VTU Base : {modal_res.get('vtu_base')}")
        print("  Frequencies:")
        for f in modal_res.get("frequencies", []):
            print(f"    Mode {f['mode']:d}: {f['hz']:.3f} Hz")
            
    except Exception as e:
        print(f"[Test 1] [Error] Modal analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- [Test 2] Running Static Analysis ---")
    analysis_config_static = {
        "job_name": "test_static"
    }
    
    # 구속조건: 1, 4, 7번 노드를 X, Y, Z 방향 완전 구속 (dofs [0, 1, 2])
    bcs = [
        (1, [0, 1, 2], 0.0),
        (4, [0, 1, 2], 0.0),
        (7, [0, 1, 2], 0.0),
    ]
    
    # 하중조건: 3, 6, 9번 노드에 -100N Z방향 하중 인가 (idx 2에 -100.0)
    forces = [
        (3, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0]),
        (6, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0]),
        (9, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0]),
    ]

    try:
        static_res = run_calculix_analysis(
            nodes=nodes,
            elements=elements,
            properties=properties,
            analysis_type="static",
            analysis_config=analysis_config_static,
            bcs=bcs,
            forces=forces,
            workspace_dir=str(scratch_dir)
        )
        print("\n[Test 2 Results] Success!")
        print(f"  Job Name : {static_res.get('job_name')}")
        print(f"  Workspace: {static_res.get('workspace')}")
        print(f"  VTU Base : {static_res.get('vtu_base')}")
        
        # 3. [추가 검증] VTU 파일 로드 및 WHTResultData 빌드 테스트 (AttributeError/celltypes 방지)
        print("\n--- [Test 3] Verifying VTU Parsing & WHTResultData Packaging ---")
        import numpy as np
        import pyvista as pv
        
        # WHT_LightChassisModel 경로를 sys.path에 수동으로 넣어 wht_converter 임포트 확보
        parent_dir = str(Path(__file__).resolve().parent.parent)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
            
        from wht_converter.wht_models import WHTResultData, WHTMetadata
        
        vtu_base_str = static_res.get("vtu_base", "")
        vtu_path = Path(f"{vtu_base_str}.vtu")
        if not vtu_path.exists():
            vtu_path = Path(f"{vtu_base_str}.01.vtu")
        if not vtu_path.exists():
            vtu_path = Path(f"{vtu_base_str}.1.vtu")
            
        print(f"[Test 3] Reading VTU file from: {vtu_path}")
        mesh_s = pv.read(str(vtu_path))
        vtu_nodes = np.array(mesh_s.points, dtype=np.float32)
        
        raw_cells = np.array(mesh_s.cells)
        offsets = []
        connectivity = []
        idx = 0
        while idx < len(raw_cells):
            n_nodes = raw_cells[idx]
            offsets.append(len(connectivity))
            for o_idx in range(1, n_nodes + 1):
                connectivity.append(raw_cells[idx + o_idx])
            idx += (n_nodes + 1)
        offsets.append(len(connectivity))
        offsets = np.array(offsets, dtype=np.int32)
        connectivity = np.array(connectivity, dtype=np.int32)
        
        # cell_types 호환성 접근 검증
        cell_types = np.array(mesh_s.celltypes if hasattr(mesh_s, "celltypes") else mesh_s.cell_types, dtype=np.uint8)
        print(f"[Test 3] Successfully parsed celltypes! Shape: {cell_types.shape}")
        
        disp_key = next((k for k in mesh_s.point_data.keys() if k.upper() in ('U', 'DISP', 'DISPLACEMENTS')), None)
        if disp_key is None:
            raise ValueError("VTU 정적 결과에서 변위 데이터('U')를 찾지 못했습니다.")
            
        disp_arr = np.array(mesh_s.point_data[disp_key])
        point_data = {
            "Displacement": disp_arr[np.newaxis, :, :3]
        }
        
        meta = WHTMetadata(
            solver_name="CalculiX", solver_version="2.23",
            analysis_type="static", coordinate_system="cartesian",
            unit_length="mm", unit_force="N",
        )
        
        rd = WHTResultData(
            nodes=vtu_nodes,
            connectivity=connectivity,
            offsets=offsets,
            cell_types=cell_types,
            point_data=point_data,
            time_values=np.array([0.0]),
            metadata=meta
        )
        print("[Test 3 Results] Success! WHTResultData constructed seamlessly.")
        print(f"  Nodes count : {len(rd.nodes)}")
        print(f"  Cells count : {len(rd.cell_types)}")
        
    except Exception as e:
        print(f"[Test 2/3] [Error] Static analysis or VTU validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_test()

