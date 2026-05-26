# -*- coding: utf-8 -*-
import pyvista as pv
from pathlib import Path

def debug_vtu():
    # scratch/test_workspace/test_modal.01.vtu 파일 분석
    vtu_path = Path("scratch/test_workspace/test_modal.1.vtu")
    if not vtu_path.exists():
        print(f"[Error] VTU file not found at {vtu_path}")
        return
        
    mesh = pv.read(str(vtu_path))
    print(f"=== VTU Mesh Info ===")
    print(f"  n_points: {mesh.n_points}")
    print(f"  n_cells : {mesh.n_cells}")
    print(f"  Point Data Keys: {list(mesh.point_data.keys())}")
    print(f"  Cell Data Keys : {list(mesh.cell_data.keys())}")
    
    # 원래 노드 ID 태그가 존재하는지 확인
    # ccx2paraview는 종종 원래 노드 정보를 보존하기 위해 'node_num' 또는 'id' 어레이를 추가합니다.
    for k in mesh.point_data.keys():
        arr = mesh.point_data[k]
        print(f"  Array '{k}' shape: {arr.shape}, dtype: {arr.dtype}")
        if arr.ndim == 1:
            print(f"    Min: {arr.min()}, Max: {arr.max()}")
            print(f"    First 10 values: {arr[:10]}")

if __name__ == "__main__":
    debug_vtu()
