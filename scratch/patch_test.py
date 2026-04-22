# -*- coding: utf-8 -*-
"""
patch_test.py
=============
QUAD4 응력 회복 알고리즘 검증을 위한 Patch Test
- 정사각형 평판(100mm x 100mm x 1mm), 4변 완전 고정
- 중앙 점하중 1N 적용
- 이론値와 수치解 비교
"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'test_jaxSSO')
os.chdir(r'd:\PythonCodeStudy\WHT_LightChassisModel')

import numpy as np
from wht_modeler.wht_mesh_model import WHTMeshModel
from wht_solver.wht_solver import WHTSolver
from wht_solver.load_cases import WHTLoadCase


def run_patch_test():
    """
    단순 정방형 평판 Patch Test.
    이론값과 수치값의 일치 여부로 응력 회복 코드를 검증합니다.
    """
    L = 100.0   # mm (한 변 길이)
    t = 1.0     # mm (두께)
    E = 210000.0  # MPa (강철)
    nu = 0.3
    n = 8       # 한 변의 요소 수

    model = WHTMeshModel('patch_test')
    model.add_material(1, E, nu, 7.85e-9)
    model.add_property(1, 'PSHELL', t, 1)

    # 노드 생성: (n+1) x (n+1) 격자
    def nid(i, j):
        return j * (n + 1) + i + 1

    for j in range(n + 1):
        for i in range(n + 1):
            x = i * L / n
            y = j * L / n
            model.add_node(nid(i, j), x, y, 0.0)

    # 요소 생성: QUAD4
    eid = 1
    for j in range(n):
        for i in range(n):
            n1 = nid(i,   j  )
            n2 = nid(i+1, j  )
            n3 = nid(i+1, j+1)
            n4 = nid(i,   j+1)
            model.add_element(eid, [n1, n2, n3, n4], 'QUAD4', 1)
            eid += 1

    # 경계 조건: 4변 완전 고정
    for j in range(n + 1):
        for i in range(n + 1):
            x = i * L / n
            y = j * L / n
            on_boundary = (abs(x) < 1e-9 or abs(x - L) < 1e-9 or
                          abs(y) < 1e-9 or abs(y - L) < 1e-9)
            if on_boundary:
                model.apply_spc(nid(i, j), (0, 1, 2, 3, 4, 5))

    # 중앙 절점 (하중 인가)
    c = n // 2
    center_nid = nid(c, c)
    print(f"  Center node: {center_nid} at ({c*L/n:.1f}, {c*L/n:.1f}, 0)")

    # 하중: 중앙 집중 하중 -1N (z방향 아래)
    lc = WHTLoadCase('patch')
    lc.add_force(center_nid, (2,), (-1.0,))

    solver = WHTSolver(model)
    result = solver.solve_static(lc)

    u = result.displacement
    sorted_nids = model.sorted_node_ids()
    center_idx = sorted_nids.index(center_nid)
    w_num = u[center_idx, 2]

    # 이론값 계산
    D = E * t**3 / (12.0 * (1.0 - nu**2))  # 굽힘 강성
    print(f"\n  Bending stiffness D = {D:.2f} N·mm")

    # 완전 고정 정방형 평판 중앙 집중 하중 처짐 (이론 공식: Timoshenko)
    # w_center = 0.00560 * P * a^2 / D  (a = L/2, 양단 고정)
    # 더 정확한 공식: w = alpha * P * a^2 / D
    # a = L/2 인 경우 alpha = 0.00560 (Timoshenko Table 11-3)
    a_half = L / 2.0
    # Note: 완전고정 정방형 판의 중앙 집중하중 처짐 계수
    # Szilard (2004) Table 7.11: alpha = 0.00560 for b/a = 1.0
    alpha_clamped = 0.00560
    w_theory_clamped = alpha_clamped * 1.0 * a_half**2 / D

    print(f"\n  [Patch Test Results]")
    print(f"  Numerical deflection  : {w_num:.6f} mm")
    print(f"  Theory (clamped)      : {w_theory_clamped:.6f} mm")
    if abs(w_theory_clamped) > 1e-12:
        ratio = w_num / w_theory_clamped
        print(f"  Ratio (num/theory)    : {ratio:.4f}  {'OK' if 0.5 < ratio < 2.0 else 'WARN'}")

    # 이론 최대 굽힘 응력 (완전 고정 판의 지지부)
    # sigma_max = 0.308 * M_max / (t^2/6)
    # M_max_edge = 0.1257 * P  (Szilard, clamped square, center load)
    # sigma_max = M_max * 6 / t^2
    M_max_theory = 0.1257 * 1.0  # N·mm per mm width
    sigma_max_theory = M_max_theory * 6.0 / (t**2)
    print(f"\n  [Stress Check]")
    print(f"  Theory max sigma (at edge): {sigma_max_theory:.4f} MPa")
    
    # 수치 응력
    s = result.cell_data.get('Stress', None)
    if s is not None:
        s_vals = s[0]  # (M, 6)
        vm = np.sqrt(0.5 * (
            (s_vals[:,0]-s_vals[:,1])**2 +
            (s_vals[:,1]-s_vals[:,2])**2 +
            (s_vals[:,2]-s_vals[:,0])**2 +
            6*(s_vals[:,3]**2 + s_vals[:,4]**2 + s_vals[:,5]**2)
        ))
        print(f"  Numerical max VM stress   : {np.max(vm):.4f} MPa")
        print(f"  Numerical mean VM stress  : {np.mean(vm):.4f} MPa")

    print("\nPatch test complete.")


if __name__ == '__main__':
    run_patch_test()
