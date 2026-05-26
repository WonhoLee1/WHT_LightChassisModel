# -*- coding: utf-8 -*-
"""
compare_solvers.py
==================
scipy vs JAX Newmark-beta 솔버 결과 비교.
동일 모델·하중에 두 솔버를 순서대로 실행하고 변위·속도·가속도 오차를 출력한다.
"""
import sys, time
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

from wht_modeler.io import LSDYNAReader
from wht_solver.wht_dynamic_solver import WHTDynamicSolver, DampingSpec
from wht_modeler.wht_dynamic_utils import InterpLoadGroup

# ── 1. 모델 로드 ──────────────────────────────────────────────────────────────
K_FILE = str(__import__('pathlib').Path(__file__).parent.parent / "wht_topo" / "sample_pos.csv")

# sample_pos.csv 대신 chassis k파일 사용
import glob as _glob
k_files = _glob.glob(str(__import__('pathlib').Path(__file__).parent.parent / "*.k"))
if not k_files:
    k_files = _glob.glob(str(__import__('pathlib').Path(__file__).parent.parent / "wht_topo" / "*.k"))
if not k_files:
    print("[ERROR] .k 파일을 찾을 수 없습니다.")
    sys.exit(1)

k_file = k_files[0]
print(f"모델 파일: {k_file}")
model = LSDYNAReader().read(k_file)

# ── 2. 간단한 하중 설정 (Z방향 정현파 강제 변위) ────────────────────────────
sorted_nids = sorted(model.nodes.keys())
n_nodes = len(sorted_nids)

dt   = 1e-3
T    = 0.5
t_arr = np.arange(0, T + dt, dt)
freq  = 5.0  # Hz
amp   = 1.0  # mm

# 코너 노드 4개 선택 (x,y 극값)
coords = np.array([[model.nodes[n].x, model.nodes[n].y] for n in sorted_nids])
corners = [
    sorted_nids[np.argmin(coords[:,0] + coords[:,1])],
    sorted_nids[np.argmax(coords[:,0] - coords[:,1])],
    sorted_nids[np.argmax(coords[:,0] + coords[:,1])],
    sorted_nids[np.argmin(coords[:,0] - coords[:,1])],
]

load_groups = []
for nid in corners:
    model.apply_spc([nid], dofs=(0, 1, 3, 4, 5))
    disp = amp * np.sin(2 * np.pi * freq * t_arr)
    load_groups.append(InterpLoadGroup([nid], 2, t_arr, disp, load_type="SPCD"))

damping = DampingSpec(mode="zeta", zeta=0.02)
N_SAVE  = 100

# ── 3. Scipy 솔버 ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" [1/2] Scipy Solver")
print("="*60)
solver_sp = WHTDynamicSolver(model)
t0 = time.time()
res_sp = solver_sp.solve_direct_dynamic(
    load_groups, dt=dt, T=T, damping=damping, n_save=N_SAVE, method='scipy'
)
t_sp = time.time() - t0
print(f"  완료: {t_sp:.2f}s  |  저장 프레임: {res_sp.n_save}")

# ── 4. JAX 솔버 ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" [2/2] JAX Solver")
print("="*60)
# 모델 재로드 (상태 초기화)
model2 = LSDYNAReader().read(k_file)
load_groups2 = []
for nid in corners:
    model2.apply_spc([nid], dofs=(0, 1, 3, 4, 5))
    disp = amp * np.sin(2 * np.pi * freq * t_arr)
    load_groups2.append(InterpLoadGroup([nid], 2, t_arr, disp, load_type="SPCD"))

solver_jx = WHTDynamicSolver(model2)
t0 = time.time()
res_jx = solver_jx.solve_direct_dynamic(
    load_groups2, dt=dt, T=T, damping=damping, n_save=N_SAVE, method='jax'
)
t_jx = time.time() - t0
print(f"  완료: {t_jx:.2f}s  |  저장 프레임: {res_jx.n_save}")

# ── 5. 비교 ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(" [Result] 솔버 비교")
print("="*60)

n_common = min(res_sp.n_save, res_jx.n_save)
u_sp = res_sp.u[:n_common]   # (frames, nodes, 6)
u_jx = res_jx.u[:n_common]

diff = np.abs(u_sp - u_jx)
rel_denom = np.abs(u_sp).max()

print(f"  {'항목':<30} {'scipy':>12} {'jax':>12} {'max|diff|':>12} {'rel err':>10}")
print(f"  {'-'*76}")

for dof, name in enumerate(['ux','uy','uz','rx','ry','rz']):
    sp_max  = float(np.abs(u_sp[:,:,dof]).max())
    jx_max  = float(np.abs(u_jx[:,:,dof]).max())
    md      = float(diff[:,:,dof].max())
    rel     = md / (sp_max + 1e-12)
    print(f"  {name:<30} {sp_max:>12.4e} {jx_max:>12.4e} {md:>12.4e} {rel:>9.2%}")

print()
u_rms_diff = float(np.sqrt(np.mean(diff**2)))
u_rms_ref  = float(np.sqrt(np.mean(u_sp**2)))
print(f"  전체 RMS 오차: {u_rms_diff:.4e}  (상대: {u_rms_diff/(u_rms_ref+1e-12):.2%})")
print(f"  속도계산 시간: scipy={t_sp:.2f}s  jax={t_jx:.2f}s  "
      f"({'JAX 빠름' if t_jx < t_sp else 'scipy 빠름'} {abs(t_sp/max(t_jx,1e-6)-1)*100:.0f}%)")

# ── 6. 시간 이력 샘플 출력 (uz 최대 노드) ─────────────────────────────────
peak_node = int(np.unravel_index(np.abs(u_sp[:,:,2]).argmax(), u_sp[:,:,2].shape)[1])
print(f"\n  uz 피크 노드 idx={peak_node} 시간 이력 (10 프레임 샘플):")
print(f"  {'frame':>6} {'t(s)':>8} {'scipy uz':>12} {'jax uz':>12} {'diff':>12}")
idx_sample = np.linspace(0, n_common-1, 10, dtype=int)
t_sp_arr = res_sp.t_saved[:n_common]
for fi in idx_sample:
    ts  = t_sp_arr[fi]
    vsp = u_sp[fi, peak_node, 2]
    vjx = u_jx[fi, peak_node, 2]
    print(f"  {fi:>6} {ts:>8.4f} {vsp:>12.4e} {vjx:>12.4e} {abs(vsp-vjx):>12.4e}")
