# -*- coding: utf-8 -*-
"""
RADIOSS 외팔보(Cantilever Beam) 진동 및 감쇠(/DAMP/VREL, /DAMP/FREQUENCY_RANGE) 테스트 생성기
단위계: Mg, mm, s (2000 MPa, 1E-9 tonne/mm^3, 0.3)
"""

from __future__ import annotations
import math
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib
except ImportError:
    pass

# ── 평판(Cantilever Beam) 기하 및 재료 파라미터 (Mg, mm, s) ──
PLATE_LEN_X = 1800.0       # 평판 전체 가로 길이 [mm]
PLATE_LEN_Y = 1200.0       # 평판 전체 세로 길이 [mm]
PLATE_THICK = 4.0          # 평판 두께 [mm]
FREE_LEN_X = 500.0         # 자유단 영역 길이 (x 방향으로 500mm만 남기고 나머지 1300mm 고정) [mm]
FIXED_LEN_X = PLATE_LEN_X - FREE_LEN_X  # 고정 영역 길이 (1300 mm) [mm]
MESH_SIZE = 50.0          # 요소 크기 (타임스텝 충분히 확보를 위해 50 mm로 설정) [mm]

# 재료 속성 (2000 MPa 탄성계수, 1E-9 tonne/mm^3 밀도, 0.3 포아송수)
E_PLATE = 2000.0           # 탄성계수 [MPa = Mg/(mm*s^2)]
RHO_PLATE = 1.0e-9         # 밀도 [tonne/mm^3 = Mg/mm^3]
NU_PLATE = 0.3             # 포아송비 [-]
SIG_Y_PLATE = 100.0        # 항복 응력 [MPa] (탄성 범위 내 거동 유도)
SIG_UTS_PLATE = 200.0      # 인장 강도 [MPa]
EPS_UTS_PLATE = 0.2        # 인장 강도 변형률 [-]
EPS_MAX_PLATE = 0.0        # 파단 변형률 (0=파단 없음)

# 하중 및 해석 파라미터
GRAVITY = 9810.0           # 중력 가속도 [mm/s^2] (-Z 방향 적용)
TSTOP = 10.0               # 해석 종료 시간 [s]
DT_MIN_CST = 2.5e-5        # /DT/NODA/CST 최소 타임스텝 [s] (과도한 감쇠나 변형 시 타임스텝 하한 강제)
DT_ANIM = 0.05             # /ANIM/DT 애니메이션 출력 간격 [s]
DT_HIS = 0.005             # /TFILE T01 시간 이력 출력 간격 [s] (10초간 2000 포인트)

# 감쇠(Damping) 테스트 주석용 파라미터
DAMP_ALPHA_VREL = 1.0      # /DAMP/VREL 감쇠 계수 alpha
DAMP_RATIO_FREQ = 0.05     # /DAMP/FREQUENCY_RANGE 감쇠비
DAMP_FREQ_LOW = 1.0        # 저주파수 [Hz]
DAMP_FREQ_HIGH = 20.0      # 고주파수 [Hz]


def ci(v: int, w: int = 10) -> str:
    """10자리 정수 오른쪽 정렬 문자열 변환."""
    try:
        return f"{v:>{w}d}"
    except ValueError:
        return cf(v, w)


def cf(v: float, w: int = 20) -> str:
    """20자리 실수 오른쪽 정렬 문자열 변환 (지수 표기법 등 호환성 보장)."""
    if w <= 10:
        s = f"{v:>{w}.3E}"
        if len(s) > w:
            s = f"{v:>{w}.2E}"
        if len(s) > w:
            s = f"{v:>{w}.1E}"
        if len(s) > w:
            d = min(w - 2, 5)
            s = f"{v:>{w}.{d}f}"[:w]
        return s[:w]
    s = f"{v:>{w}.6E}"
    if len(s) > w:
        s = f"{v:>{w}.4E}"
    if len(s) > w:
        s = f"{v:>{w}.2E}"
    return s[:w]


def create_plate_mesh():
    """
    1800 x 1200 mm 사각 평판 직교 메쉬 생성.
    
    Returns:
        nd: {nid: (x, y, z)} 노드 좌표 딕셔너리
        qd: {eid: (n1, n2, n3, n4, pid)} 4각 쉘 요소 딕셔너리
        fix_nids: x <= FIXED_LEN_X 인 고정 영역 노드 ID 리스트
        free_nids: x > FIXED_LEN_X 인 자유단 영역 노드 ID 리스트
        tip_nid: 자유단 끝단 중앙(x=1800, y=600) 노드 ID (진동 변위 기록용)
    """
    nd = {}
    qd = {}
    nx = int(round(PLATE_LEN_X / MESH_SIZE)) + 1
    ny = int(round(PLATE_LEN_Y / MESH_SIZE)) + 1
    
    dx = PLATE_LEN_X / (nx - 1)
    dy = PLATE_LEN_Y / (ny - 1)
    
    nid = 1
    grid_map = {}
    fix_nids = []
    free_nids = []
    
    for ix in range(nx):
        x = ix * dx
        for iy in range(ny):
            y = iy * dy
            z = 0.0
            nd[nid] = (x, y, z)
            grid_map[(ix, iy)] = nid
            
            if x <= FIXED_LEN_X + 1e-5:
                fix_nids.append(nid)
            else:
                free_nids.append(nid)
            nid += 1
            
    eid = 1
    pid = 1
    for ix in range(nx - 1):
        for iy in range(ny - 1):
            n1 = grid_map[(ix, iy)]
            n2 = grid_map[(ix + 1, iy)]
            n3 = grid_map[(ix + 1, iy + 1)]
            n4 = grid_map[(ix, iy + 1)]
            qd[eid] = (n1, n2, n3, n4, pid)
            eid += 1
            
    # 끝단 중앙 노드 (X=1800, Y=600 근처) 탐색
    target_pt = np.array([PLATE_LEN_X, PLATE_LEN_Y / 2.0, 0.0])
    min_dist = float('inf')
    tip_nid = -1
    for n_id, coords in nd.items():
        dist = np.linalg.norm(np.array(coords) - target_pt)
        if dist < min_dist:
            min_dist = dist
            tip_nid = n_id
            
    return nd, qd, fix_nids, free_nids, tip_nid


def write_starter(fp: str, nd: dict, qd: dict, fix_nids: list, free_nids: list, tip_nid: int, damp_ratio: float = None, damp_f_low: float = 3.0, damp_f_high: float = 20.0):
    """OpenRadioss Starter 파일 (_0000.rad) 작성. 주석 처리된 감쇠 카드 포함."""
    all_nids = sorted(nd.keys())
    
    with open(fp, "w", encoding="utf-8") as f:
        w = f.write
        w("#RADIOSS STARTER\n")
        w("/BEGIN\n")
        w("cantilever_damp\n")
        w(f"{2026:>10}{0:>10}\n")
        w(f"{'Mg':<20}{'mm':<20}{'s':<20}\n")
        w(f"{'Mg':<20}{'mm':<20}{'s':<20}\n")
        
        # 재료 정의 (/MAT/LAW2)
        w("/MAT/LAW2/1\nplate_mat\n")
        w(f"# {'Rho_I':>18}\n")
        w(cf(RHO_PLATE) + "\n")
        w(f"# {'E':>18}{'nu':>20}{'Iflag':>10}\n")
        w(cf(E_PLATE) + cf(NU_PLATE) + ci(1) + "\n")
        w(f"# {'a(yield)':>18}{'b(UTS)':>20}{'n(EPS_UTS)':>20}{'eps_max':>20}\n")
        w(cf(SIG_Y_PLATE) + cf(SIG_UTS_PLATE) + cf(EPS_UTS_PLATE) + cf(EPS_MAX_PLATE) + "\n")
        w(f"# {'c':>18}{'EPS_DOT_0':>20}{'ICC':>10}{'Fsmooth':>10}{'F_cut':>10}{'Chard':>10}\n")
        w(cf(0.0) + cf(0.0) + ci(0) + ci(0) + ci(0) + ci(0) + "\n")
        w(f"# {'m':>18}{'T_melt':>20}{'rhoC_p':>20}{'T_r':>20}\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + cf(0.0) + "\n")
        
        # 쉘 속성 정의 (/PROP/SHELL)
        w("/PROP/SHELL/1\nplate_prop\n")
        w(f"# {'Ishell':>8}{'Ismstr':>10}{'Ish3n':>10}{'Idrill':>10}{'Ipinch':>10}{'':>10}{'P_Thick_Fail':>20}\n")
        w(ci(24) + ci(2) + ci(2) + ci(2) + ci(0) + " " * 10 + cf(0.0) + "\n")
        w(f"# {'hm':>18}{'hf':>20}{'hr':>20}{'dm':>20}{'dn':>20}\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + cf(0.0) + cf(0.015) + "\n")
        w(f"# {'N':>8}{'':>10}{'Thick':>20}{'Ashear':>20}{'':>10}{'Ithick':>10}{'Iplas':>10}{'Ipos':>10}\n")
        w(ci(5) + " " * 10 + cf(PLATE_THICK) + cf(0.0) + " " * 10 + ci(-1) + ci(-1) + ci(0) + "\n")
        
        # 파트 정의 (/PART)
        w("/PART/1\ncantilever_plate\n")
        w(f"# {'prop_ID':>8}{'mat_ID':>10}\n")
        w(ci(1) + ci(1) + "\n")
        
        # 노드 정의 (/NODE)
        w("/NODE\n")
        w(f"# {'nid':>8}{'X':>20}{'Y':>20}{'Z':>20}\n")
        for nid in all_nids:
            x, y, z = nd[nid]
            w(ci(nid) + cf(x) + cf(y) + cf(z) + "\n")
            
        # 요소 정의 (/SHELL)
        w("/SHELL/1\n")
        w(f"# {'eid':>8}{'n1':>10}{'n2':>10}{'n3':>10}{'n4':>10}\n")
        for eid, (n1, n2, n3, n4, pid) in sorted(qd.items()):
            w(ci(eid) + ci(n1) + ci(n2) + ci(n3) + ci(n4) + "\n")
            
        # 노드 그룹 정의 (/GRNOD/NODE)
        w("/GRNOD/NODE/1\nfree_vibrating_nodes\n")
        w("# node_ID list (8 per line)\n")
        for i in range(0, len(free_nids), 8):
            w("".join(ci(n) for n in free_nids[i : i + 8]) + "\n")
            
        w("/GRNOD/NODE/2\nfixed_boundary_nodes\n")
        w("# node_ID list (8 per line)\n")
        for i in range(0, len(fix_nids), 8):
            w("".join(ci(n) for n in fix_nids[i : i + 8]) + "\n")
            
        w("/GRNOD/NODE/3\nall_plate_nodes\n")
        w("# node_ID list (8 per line)\n")
        for i in range(0, len(all_nids), 8):
            w("".join(ci(n) for n in all_nids[i : i + 8]) + "\n")
            
        # 파트 그룹 정의 (/GRPART/PART)
        w("/GRPART/PART/1\nplate_part_group\n")
        w(f"# {'part_ID':>8}\n")
        w(ci(1) + "\n")
        
        # 경계 조건 정의 (/BCS) - 500mm 남기고 나머지 고정 (X <= FIXED_LEN_X 영역)
        w("/BCS/1\nfix_clamped_region\n")
        w(f"# {'Trarot':>8}{'skew_ID':>10}{'grnod_ID':>10}\n")
        w(f"{'111 111':>10}" + ci(0) + ci(2) + "\n")
        
        # 중력 정의 (/GRAV) - Z축 음의 방향
        w("/GRAV/1\ngravity_load\n")
        w(f"# {'funct_IDT':>8}{'DIR':>10}{'skew_ID':>10}{'sensor_ID':>10}{'grnod_ID':>10}{'':>10}{'Ascale_x':>20}{'Fscale_Y':>20}\n")
        w(ci(0) + f"{'Z':>10}" + ci(0) + ci(0) + ci(3) + " " * 10 + cf(1.0) + cf(-GRAVITY) + "\n")
        
        # 끝단 변위 기록용 시간 이력 노드 등록 (/TH/NODE)
        w("/TH/NODE/1\ncantilever_tip_th\n")
        w(f"# {'var1':>8}{'var2':>10}{'var3':>10}{'var4':>10}{'var5':>10}{'var6':>10}{'var7':>10}{'var8':>10}{'var9':>10}{'var10':>10}\n")
        w(f"{'DEF':<10}\n")
        w(f"# {'NODid':>8}{'Iskew':>10}\n")
        w(ci(tip_nid) + ci(0) + "\n")
        
        # ── 감쇠 카드 주석 처리 (#만 제거하면 사용되게 정확한 열 폭 설정) ──
        w("#/DAMP/VREL/1\n")
        w("#damp_vrel_test\n")
        w(f"## {'Alpha_x':>18}{'Alpha2_x':>20}{'grnod_id':>10}{'skew_id':>10}{'Tstart':>20}{'Tstop':>20}\n")
        w("#" + cf(DAMP_ALPHA_VREL) + cf(0.0) + ci(1) + ci(0) + cf(0.0) + cf(TSTOP) + "\n")
        w(f"## {'Freq':>18}{'RbodyID':>10}{'FuncID':>10}{'Xscale':>20}\n")
        w("#" + cf(0.0) + ci(0) + ci(0) + cf(1.0) + "\n")
        w(f"## {'Alpha_y':>18}{'Alpha2_y':>20}\n")
        w("#" + cf(DAMP_ALPHA_VREL) + cf(0.0) + "\n")
        w(f"## {'Alpha_z':>18}{'Alpha2_z':>20}\n")
        w("#" + cf(DAMP_ALPHA_VREL) + cf(0.0) + "\n")
        
        if damp_ratio is not None:
            w("/DAMP/FREQUENCY_RANGE/1\n")
            w("damp_freq_test\n")
            w(f"# {'CDAMP':>18}{'grpart_ID':>10}{'Tstart':>20}{'Tstop':>20}\n")
            w(cf(damp_ratio) + ci(1) + cf(0.001) + cf(TSTOP) + "\n")
            w(f"# {'Freq_Low':>18}{'Freq_High':>20}\n")
            w(cf(damp_f_low) + cf(damp_f_high) + "\n")
        else:
            w("#/DAMP/FREQUENCY_RANGE/1\n")
            w("#damp_freq_test\n")
            w(f"## {'CDAMP':>18}{'grpart_ID':>10}{'Tstart':>20}{'Tstop':>20}\n")
            w("#" + cf(DAMP_RATIO_FREQ) + ci(1) + cf(0.001) + cf(TSTOP) + "\n")
            w(f"## {'Freq_Low':>18}{'Freq_High':>20}\n")
            w("#" + cf(DAMP_FREQ_LOW) + cf(DAMP_FREQ_HIGH) + "\n")
        
        w("/END\n")
    print(f"[OK] Starter 생성 완료: 노드 {len(nd)}개, 요소 {len(qd)}개 -> {fp}")


def write_engine(fp: str, dt_min: float = 5.0e-5):
    """OpenRadioss Engine 파일 (_0001.rad) 작성."""
    with open(fp, "w", encoding="utf-8") as f:
        w = f.write
        w("#RADIOSS ENGINE\n")
        w("/RUN/cantilever/1\n")
        w(cf(TSTOP) + "\n")
        w("/DT/NODA/CST\n")
        w(cf(0.9) + cf(dt_min) + "\n")
        w("/TFILE\n")
        w(cf(DT_HIS) + "\n")
        w("/ANIM/DT\n")
        w(cf(0.0) + cf(DT_ANIM) + "\n")
        w("/ANIM/VECT/DISP\n")
        w("/ANIM/VECT/VEL\n")
        w("/ANIM/SHELL/TENS/STRESS/ALL\n")
        w("/END/ENGINE\n")
    print(f"[OK] Engine 생성 완료: {fp}")


def write_k_file(fp: str, nd: dict, qd: dict):
    """기하 시각화 프리뷰를 위한 LS-DYNA Keyword 파일 (*.k) 생성."""
    with open(fp, "w", encoding="utf-8") as f:
        w = f.write
        w("*KEYWORD\n")
        w("*SECTION_SHELL\n")
        w(f"{1:>10}{2:>10}{0.0:>10}{0.0:>10}{0.0:>10}{0.0:>10}{0.0:>10}\n")
        w(f"{PLATE_THICK:>10.4f}{PLATE_THICK:>10.4f}{PLATE_THICK:>10.4f}{PLATE_THICK:>10.4f}\n")
        w("*MAT_ELASTIC\n")
        w(f"{1:>10}{RHO_PLATE:>10.4E}{E_PLATE:>10.4E}{NU_PLATE:>10.4f}\n")
        w("*PART\ncantilever_plate\n")
        w(f"{1:>10}{1:>10}{1:>10}{0:>10}{0:>10}{0:>10}{0:>10}{0:>10}\n")
        w("*NODE\n")
        for nid in sorted(nd.keys()):
            x, y, z = nd[nid]
            w(f"{nid:>8}{x:>16.6f}{y:>16.6f}{z:>16.6f}{0:>8}{0:>8}\n")
        w("*ELEMENT_SHELL\n")
        for eid, (n1, n2, n3, n4, pid) in sorted(qd.items()):
            w(f"{eid:>8}{pid:>8}{n1:>8}{n2:>8}{n3:>8}{n4:>8}\n")
        w("*END\n")
    print(f"[OK] LS-DYNA Preview 생성 완료 -> {fp}")


def write_vtkhdf_file(fp: str, nd: dict, qd: dict):
    """ParaView 프리뷰를 위한 단일 데이터셋 VTKHDF 파일 생성."""
    import h5py as h5
    
    node_list = sorted(nd.keys())
    node_to_idx = {nid: idx for idx, nid in enumerate(node_list)}
    points = np.array([nd[nid] for nid in node_list], dtype=np.float32)
    
    cells = []
    offsets = [0]
    types = []
    part_ids = []
    
    for eid, (n1, n2, n3, n4, pid) in sorted(qd.items()):
        cells.extend([node_to_idx[n1], node_to_idx[n2], node_to_idx[n3], node_to_idx[n4]])
        offsets.append(len(cells))
        types.append(9)  # VTK_QUAD
        part_ids.append(pid)
        
    cells = np.array(cells, dtype=np.int32)
    offsets = np.array(offsets, dtype=np.int32)
    types = np.array(types, dtype=np.uint8)
    part_ids = np.array(part_ids, dtype=np.int32)
    
    with h5.File(fp, "w") as f:
        root = f.create_group("VTKHDF")
        root.attrs["Version"] = np.array([2, 2], dtype=np.int32)
        root.attrs["Type"] = b"UnstructuredGrid"
        root.create_dataset("Points", data=points)
        root.create_dataset("NumberOfPoints", data=np.array([len(points)], dtype=np.int32))
        root.create_dataset("NumberOfCells", data=np.array([len(types)], dtype=np.int32))
        root.create_dataset("NumberOfConnectivityIds", data=np.array([len(cells)], dtype=np.int32))
        root.create_dataset("Offsets", data=offsets)
        root.create_dataset("Connectivity", data=cells)
        root.create_dataset("Types", data=types)
        
        cell_data = root.create_group("CellData")
        cell_data.create_dataset("vtkPartID", data=part_ids)
        
    print(f"[OK] VTKHDF Preview 생성 완료 -> {fp}")


def get_openradioss_paths():
    """OpenRadioss 실행 경로 자동 탐색 및 환경 변수 반환."""
    for base_str in [r"D:\OpenRadioss", r"D:\OpenRadioss_win64\OpenRadioss"]:
        base = Path(base_str)
        exec_dir = base / "exec"
        if exec_dir.exists():
            env = os.environ.copy()
            env["PATH"] = os.pathsep.join([
                str(exec_dir),
                str(base / "extlib" / "hm_reader" / "win64"),
                str(base / "extlib" / "intelOneAPI_runtime" / "win64"),
                env.get("PATH", "")
            ])
            env["RAD_CFG_PATH"] = str(base / "hm_cfg_files")
            env["RAD_H3D_PATH"] = str(base / "extlib" / "h3d" / "lib" / "win64")
            return base, exec_dir, env
    raise FileNotFoundError("OpenRadioss 설치 디렉토리를 찾을 수 없습니다.")


def run_openradioss(rad_dir: Path, run_name: str):
    """OpenRadioss Starter 및 Engine을 실행하고, T01 파일을 CSV로 변환합니다."""
    base, exec_dir, env = get_openradioss_paths()
    starter_exe = exec_dir / "starter_win64.exe"
    engine_exe = exec_dir / "engine_win64.exe"
    th_to_csv_exe = exec_dir / "th_to_csv_win64.exe"
    
    starter_rad = rad_dir / f"{run_name}_0000.rad"
    engine_rad = rad_dir / f"{run_name}_0001.rad"
    
    print(f"\n[Run] 1/3 OpenRadioss Starter 실행 중: {starter_rad.name}")
    res = subprocess.run([str(starter_exe), "-i", starter_rad.name], cwd=str(rad_dir), capture_output=True, text=True, env=env)
    if res.returncode != 0 or "NORMAL TERMINATION" not in res.stdout.upper() or not (rad_dir / f"{run_name}_0000.out").exists():
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError("OpenRadioss Starter 실행 실패!")
        
    print(f"[Run] 2/3 OpenRadioss Engine 실행 중: {engine_rad.name} (10초 해석)")
    res = subprocess.run([str(engine_exe), "-i", engine_rad.name], cwd=str(rad_dir), capture_output=True, text=True, env=env)
    if res.returncode != 0 or "NORMAL TERMINATION" not in res.stdout.upper() or not (rad_dir / f"{run_name}_0001.out").exists():
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        raise RuntimeError("OpenRadioss Engine 실행 실패!")
        
    t01_file = rad_dir / f"{run_name}T01"
    if not t01_file.exists():
        t01_alt = rad_dir / f"{run_name}t01"
        if t01_alt.exists():
            t01_file = t01_alt
        else:
            print("생성된 T01 파일을 찾을 수 없습니다. (디렉토리 내 파일 목록 확인 필요)")
            return None
            
    print(f"[Run] 3/3 T01 → CSV 변환 실행 중: {t01_file.name}")
    subprocess.run([str(th_to_csv_exe), str(t01_file)], cwd=str(rad_dir), capture_output=True, text=True, env=env)
    
    csv_file = Path(str(t01_file) + ".csv")
    if not csv_file.exists():
        csv_file = rad_dir / f"{run_name}T01.csv"
    return csv_file if csv_file.exists() else None


def analyze_vibration(time_val: np.ndarray, disp_val: np.ndarray):
    """
    시간 이력 변위 데이터로부터 고유 진동 주파수와 감쇠비(Damping Ratio)를 계산합니다.
    - 초기 2.0초 구간에서 FFT를 수행하여 고감쇠 시 모드가 완전히 소멸하기 전 1차 모드 주파수를 검출
    - 진폭이 초기 최대 스윙의 2% 이하로 떨어지는 평탄화 영역을 회귀분석 윈도우에서 제외
    """
    from scipy.signal import find_peaks
    from scipy.stats import linregress

    dt = np.mean(np.diff(time_val))
    
    # 1. 초기 2.0초 구간에서 FFT 주파수 검출 (고감쇠 케이스의 1차 모드 소멸 전 주파수 확보)
    t_2s_mask = time_val <= 2.0
    n_2s = np.sum(t_2s_mask)
    if n_2s > 10:
        disp_centered_2s = disp_val[t_2s_mask] - np.mean(disp_val[t_2s_mask])
        fft_freqs = np.fft.rfft(disp_centered_2s)
        freqs = np.fft.rfftfreq(n_2s, d=dt)
        valid_idx = np.where((freqs >= 0.5) & (freqs <= 50.0))[0]
        if len(valid_idx) > 0:
            best_i = valid_idx[np.argmax(np.abs(fft_freqs[valid_idx]))]
            f_fft = freqs[best_i]
        else:
            f_fft = freqs[np.argmax(np.abs(fft_freqs[1:])) + 1]
    else:
        f_fft = 3.8

    f_num = f_fft if f_fft > 0 else 3.8
    period_mean = 1.0 / f_num

    # 2. Peak & Valley 탐색 (주기 T의 70% 간격을 최소로 설정하여 고주파 리플 제외)
    dist_samples = max(5, int((0.7 * period_mean) / dt))
    valleys, _ = find_peaks(-disp_val, distance=dist_samples)
    peaks, _ = find_peaks(disp_val, distance=dist_samples)

    if len(valleys) > 1:
        valley_times = time_val[valleys]
        periods = np.diff(valley_times)
        period_mean = np.mean(periods)
        if period_mean > 0:
            f_num = 1.0 / period_mean

    # 3. 감쇠비(Damping Ratio, zeta) 계산
    zeta = 0.0
    if len(valleys) >= 2 and len(peaks) >= 2:
        min_len = min(len(valleys), len(peaks))
        amp_times = []
        amps = []
        for i in range(min_len):
            t_v = time_val[valleys[i]]
            p_idx = np.argmin(np.abs(time_val[peaks] - t_v))
            amp = abs(disp_val[peaks[p_idx]] - disp_val[valleys[i]]) / 2.0
            if amp > 1e-6:
                amp_times.append(t_v)
                amps.append(amp)
        
        amp_times = np.array(amp_times)
        amps = np.array(amps)
        
        if len(amps) >= 2:
            # 초기 최대 진폭의 2% 이하로 줄어든 구간(평탄화된 비감쇠 구간)은 회귀분석에서 제외
            threshold = 0.02 * amps[0]
            valid_mask = amps >= threshold
            if np.sum(valid_mask) >= 2:
                amp_times = amp_times[valid_mask]
                amps = amps[valid_mask]
                
            slope, intercept, r_value, p_value, std_err = linregress(amp_times, np.log(amps))
            if slope < 0 and f_num > 0:
                zeta = -slope / (2.0 * math.pi * f_num)
            else:
                zeta = 0.0

    return f_num, period_mean, zeta, f_fft, valleys, peaks


def plot_displacement_history(csv_path: Path, tip_nid: int, out_dir: Path):
    """
    OpenRadioss T01에서 변환된 CSV 파일을 읽어 끝단 노드의 시간에 따른 Z방향 변위 그래프 및 TXT 저장.
    """
    print(f"\n[Plot] CSV 데이터 파싱 및 시각화 생성 중: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 시간 컬럼 탐색
    time_col = df.columns[0]
    for col in df.columns:
        if 'time' in col.lower() or 't' == col.lower().strip():
            time_col = col
            break
            
    # 노드 Z변위 컬럼 탐색
    disp_col = None
    for col in df.columns:
        if str(tip_nid) in col and ('z' in col.lower() or '3' in col):
            disp_col = col
            break
    if disp_col is None:
        for col in df.columns:
            if 'z' in col.lower() and ('disp' in col.lower() or 'trans' in col.lower() or 'def' in col.lower()):
                disp_col = col
                break
    if disp_col is None and len(df.columns) > 1:
        disp_col = df.columns[1]
        print(f"[Warning] Z변위 열을 자동으로 결정하지 못해 2번째 열({disp_col})을 선택합니다.")
        
    print(f" -> Time Column: '{time_col}', Displacement Column: '{disp_col}'")
    
    time_val = df[time_col].values
    disp_val = df[disp_col].values
    
    # 주파수 및 감쇠비 동적 분석 수행
    f_num, period_mean, zeta, f_fft, valleys, peaks = analyze_vibration(time_val, disp_val)
    print(f" -> [분석 결과] 고유 주파수: {f_num:.4f} Hz (주기: {period_mean:.4f} s), FFT 주파수: {f_fft:.4f} Hz")
    print(f" -> [분석 결과] 감쇠비 (Damping Ratio): {zeta:.6f} ({zeta * 100.0:.2f} %)")
    
    # 1. TXT 파일 저장 (상단에 진동 특성 요약 포함)
    txt_path = out_dir / "cantilever_tip_displacement.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# =====================================================================\n")
        f.write("# OpenRadioss 외팔보(Cantilever Beam) 끝단 변위 시간 이력 및 동적 특성 요약\n")
        f.write("# =====================================================================\n")
        f.write(f"# [동적 특성 분석 결과]\n")
        f.write(f"#  - 고유 진동 주파수 (Frequency) : {f_num:.4f} Hz (FFT: {f_fft:.4f} Hz)\n")
        f.write(f"#  - 진동 주기 (Period)          : {period_mean:.4f} s\n")
        f.write(f"#  - 감쇠비 (Damping Ratio, zeta) : {zeta:.6f} ({zeta * 100.0:.2f} %)\n")
        f.write("# =====================================================================\n")
        f.write("# Time [s]\tTip_Z_Displacement [mm]\n")
        for t, d in zip(time_val, disp_val):
            f.write(f"{t:.6E}\t{d:.6E}\n")
    print(f"[OK] TXT 출력 완료 -> {txt_path}")
    
    # 2. PNG 그래프 저장 (9pt 글로벌 폰트, 범례 자동, tight_layout, 분석 결과 박스 표시)
    plt.rcParams.update({'font.size': 9})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_val, disp_val, label=f"Tip Node (ID={tip_nid}) Z-Displacement", color="#1f77b4", linewidth=1.5)
    
    if len(valleys) > 0:
        ax.plot(time_val[valleys], disp_val[valleys], 'ro', markersize=4, label="Valleys (Minima)", alpha=0.7)
    
    ax.set_title("외팔보(Cantilever Beam) 자유단 중력 처짐 및 진동 변위", fontsize=11, fontweight='bold')
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("Z-Displacement [mm]", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    
    # 동적 분석 결과 텍스트 박스 표시
    info_text = (
        f" [진동 및 감쇠 분석 결과]\n"
        f" • 고유 주파수 (Freq) : {f_num:.3f} Hz (T = {period_mean:.3f} s)\n"
        f" • 감쇠비 (Damping Ratio) : ζ = {zeta:.4f} ({zeta * 100.0:.2f} %)\n"
        f" • FFT 주파수 (FFT Freq) : {f_fft:.3f} Hz"
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    ax.text(0.02, 0.05, info_text, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='bottom', bbox=props)
            
    plt.tight_layout()
    
    png_path = out_dir / "cantilever_tip_displacement.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[OK] PNG 시각화 완료 -> {png_path}")


def run_freq_range_sweep():
    """
    /DAMP/FREQUENCY_RANGE 카드의 CDAMP 값(0.0(무감쇠), 0.01, 0.05, 0.07, 0.1, 0.5) 스윕 테스트 케이스 자동 생성,
    해석 실행, 결과 분석 및 통합 비교 시각화 수행.
    """
    print("\n=======================================================")
    print("=== OpenRadioss /DAMP/FREQUENCY_RANGE 스윕 해석 시작 ===")
    print("=======================================================")
    
    out_dir = Path(__file__).parent
    sweep_dir = out_dir / "cases_freq_range"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    nd, qd, fix_nids, free_nids, tip_nid = create_plate_mesh()
    
    cdamp_vals = [0.0, 0.01, 0.05, 0.07, 0.1, 0.5]
    results = {}
    
    for cdamp in cdamp_vals:
        case_name = f"c{cdamp}"
        case_dir = sweep_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Case: {case_name}] 입력 파일 생성 중...")
        run_name = "cantilever"
        sp = str(case_dir / f"{run_name}_0000.rad")
        ep = str(case_dir / f"{run_name}_0001.rad")
        
        # 0.0(무감쇠)일 때는 damp_ratio=None을 주어 감쇠 카드를 주석 처리함
        damp_val = cdamp if cdamp > 0.0 else None
        write_starter(sp, nd, qd, fix_nids, free_nids, tip_nid, damp_ratio=damp_val, damp_f_low=3.0, damp_f_high=20.0)
        write_engine(ep, dt_min=DT_MIN_CST)
        
        try:
            print(f"[Case: {case_name}] OpenRadioss 해석 실행 중...")
            csv_file = run_openradioss(case_dir, run_name)
            if csv_file and csv_file.exists():
                df = pd.read_csv(csv_file)
                # 시간 및 변위 추출
                time_col = df.columns[0]
                for col in df.columns:
                    if 'time' in col.lower() or 't' == col.lower().strip():
                        time_col = col
                        break
                disp_col = df.columns[1]
                for col in df.columns:
                    if str(tip_nid) in col and ('z' in col.lower() or '3' in col):
                        disp_col = col
                        break
                
                time_val = df[time_col].values
                disp_val = df[disp_col].values
                
                # 분석 수행
                f_num, period_mean, zeta, f_fft, valleys, peaks = analyze_vibration(time_val, disp_val)
                print(f" -> [결과] 주파수: {f_num:.4f} Hz, 계산된 감쇠비: {zeta*100.0:.2f} % (목표 CDAMP: {cdamp*100.0:.1f} %)")
                
                # 결과 저장
                results[cdamp] = {
                    'time': time_val,
                    'disp': disp_val,
                    'f_num': f_num,
                    'zeta': zeta
                }
                
                # 개별 케이스 그래프 및 TXT 생성
                plot_displacement_history(csv_file, tip_nid, case_dir)
        except Exception as e:
            print(f"[Error] 케이스 {case_name} 해석 중 오류 발생: {e}")
            
    # 통합 비교 그래프 플롯
    if len(results) > 0:
        plt.rcParams.update({'font.size': 9})
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#d62728']
        for idx, (cdamp, data) in enumerate(results.items()):
            if cdamp == 0.0:
                label = f"Undamped (f={data['f_num']:.2f}Hz, ζ={data['zeta']*100.0:.2f}%)"
            else:
                label = f"CDAMP={cdamp} (f={data['f_num']:.2f}Hz, ζ={data['zeta']*100.0:.2f}%)"
            ax.plot(data['time'], data['disp'], label=label, color=colors[idx % len(colors)], linewidth=1.2, alpha=0.9)
            
        ax.set_title("CDAMP 감쇠비에 따른 외팔보 자유 진동 응답 비교 (3~20 Hz 범위)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time [s]", fontsize=10)
        ax.set_ylabel("Z-Displacement [mm]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        
        comp_png = sweep_dir / "displacement_comparison.png"
        fig.savefig(comp_png, dpi=300)
        plt.close(fig)
        print(f"\n[OK] 통합 비교 시각화 완료 -> {comp_png}")


def run_min_freq_sweep():
    """
    CDAMP = 0.05 고정 상태에서 min freq (0.1, 1, 5, 10, 20 Hz) 스윕 테스트 케이스 자동 생성,
    해석 실행, 결과 분석 및 통합 비교 시각화 수행.
    """
    print("\n=======================================================")
    print("=== OpenRadioss /DAMP/FREQUENCY_RANGE R2 스윕 시작 ===")
    print("=======================================================")
    
    out_dir = Path(__file__).parent
    sweep_dir = out_dir / "cases_freq_ranges_R2"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    nd, qd, fix_nids, free_nids, tip_nid = create_plate_mesh()
    
    min_freq_vals = [0.1, 1.0, 5.0, 10.0, 20.0]
    results = {}
    
    for f_min in min_freq_vals:
        case_name = f"f{f_min}"
        case_dir = sweep_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[Case: {case_name}] 입력 파일 생성 중...")
        run_name = "cantilever"
        sp = str(case_dir / f"{run_name}_0000.rad")
        ep = str(case_dir / f"{run_name}_0001.rad")
        
        # CDAMP=0.05 고정, min freq 변동, max freq=50.0 Hz로 설정
        write_starter(sp, nd, qd, fix_nids, free_nids, tip_nid, damp_ratio=0.05, damp_f_low=f_min, damp_f_high=50.0)
        write_engine(ep, dt_min=DT_MIN_CST)
        
        try:
            print(f"[Case: {case_name}] OpenRadioss 해석 실행 중...")
            csv_file = run_openradioss(case_dir, run_name)
            if csv_file and csv_file.exists():
                df = pd.read_csv(csv_file)
                # 시간 및 변위 추출
                time_col = df.columns[0]
                for col in df.columns:
                    if 'time' in col.lower() or 't' == col.lower().strip():
                        time_col = col
                        break
                disp_col = df.columns[1]
                for col in df.columns:
                    if str(tip_nid) in col and ('z' in col.lower() or '3' in col):
                        disp_col = col
                        break
                
                time_val = df[time_col].values
                disp_val = df[disp_col].values
                
                # 분석 수행
                f_num, period_mean, zeta, f_fft, valleys, peaks = analyze_vibration(time_val, disp_val)
                print(f" -> [결과] 주파수: {f_num:.4f} Hz, 계산된 감쇠비: {zeta*100.0:.2f} %")
                
                # 결과 저장
                results[f_min] = {
                    'time': time_val,
                    'disp': disp_val,
                    'f_num': f_num,
                    'zeta': zeta
                }
                
                # 개별 케이스 그래프 및 TXT 생성
                plot_displacement_history(csv_file, tip_nid, case_dir)
        except Exception as e:
            print(f"[Error] 케이스 {case_name} 해석 중 오류 발생: {e}")
            
    # 통합 비교 그래프 플롯
    if len(results) > 0:
        plt.rcParams.update({'font.size': 9})
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for idx, (f_min, data) in enumerate(results.items()):
            label = f"Min Freq={f_min}Hz (f={data['f_num']:.2f}Hz, ζ={data['zeta']*100.0:.2f}%)"
            ax.plot(data['time'], data['disp'], label=label, color=colors[idx % len(colors)], linewidth=1.2, alpha=0.9)
            
        ax.set_title("Min Frequency 대역 변경에 따른 외팔보 자유 진동 응답 비교 (CDAMP=5%)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time [s]", fontsize=10)
        ax.set_ylabel("Z-Displacement [mm]", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        
        comp_png = sweep_dir / "displacement_comparison.png"
        fig.savefig(comp_png, dpi=300)
        plt.close(fig)
        print(f"\n[OK] R2 통합 비교 시각화 완료 -> {comp_png}")


def main():
    print("=== OpenRadioss 외팔보(Cantilever Beam) 해석 모델 생성기 ===")
    out_dir = Path(__file__).parent
    rad_dir = out_dir / "rad"
    rad_dir.mkdir(parents=True, exist_ok=True)
    
    # ── sweep 옵션이 있는 경우 별도 처리 ──
    if len(sys.argv) > 1 and "--sweep" in sys.argv:
        run_freq_range_sweep()
        return
    elif len(sys.argv) > 1 and "--sweep_r2" in sys.argv:
        run_min_freq_sweep()
        return
        
    # 1. 메쉬 및 경계 노드 생성
    nd, qd, fix_nids, free_nids, tip_nid = create_plate_mesh()
    print(f"[1] 메쉬 생성: 전체 노드 {len(nd)}개, 요소 {len(qd)}개")
    print(f"    - 고정단 노드 (X <= {FIXED_LEN_X}mm): {len(fix_nids)}개")
    print(f"    - 자유단 노드 (X > {FIXED_LEN_X}mm): {len(free_nids)}개")
    print(f"    - 끝단 추적 노드 (X={nd[tip_nid][0]}, Y={nd[tip_nid][1]}): ID = {tip_nid}")
    
    # 2. OpenRadioss 입력 파일 및 시각화 파일 생성
    run_name = "cantilever"
    sp = str(rad_dir / f"{run_name}_0000.rad")
    ep = str(rad_dir / f"{run_name}_0001.rad")
    kp = str(rad_dir / f"{run_name}_geom.k")
    vp = str(rad_dir / f"{run_name}_geom.vtkhdf")
    
    write_starter(sp, nd, qd, fix_nids, free_nids, tip_nid)
    write_engine(ep, dt_min=DT_MIN_CST)
    write_k_file(kp, nd, qd)
    write_vtkhdf_file(vp, nd, qd)
    
    # 3. CLI 인자 처리 (--run 또는 --post 시 솔버 자동 실행 및 그래프 그리기)
    if len(sys.argv) > 1 and "--post" in sys.argv:
        csv_file = rad_dir / f"{run_name}T01.csv"
        if not csv_file.exists():
            t01_file = rad_dir / f"{run_name}T01"
            if t01_file.exists():
                _, exec_dir, env = get_openradioss_paths()
                subprocess.run([str(exec_dir / "th_to_csv_win64.exe"), str(t01_file)], cwd=str(rad_dir), capture_output=True, text=True, env=env)
                csv_file = Path(str(t01_file) + ".csv")
        if csv_file and csv_file.exists():
            plot_displacement_history(csv_file, tip_nid, out_dir)
        else:
            print(f"[Error] CSV 파일({csv_file})을 찾을 수 없습니다. --run으로 해석을 먼저 실행하세요.")
    elif len(sys.argv) > 1 and "--run" in sys.argv:
        try:
            csv_file = run_openradioss(rad_dir, run_name)
            if csv_file and csv_file.exists():
                plot_displacement_history(csv_file, tip_nid, out_dir)
        except Exception as e:
            print(f"[Error] 해석 또는 후처리 중 오류 발생: {e}")
    else:
        print("\n[Tip] 해석을 즉시 실행하고 그래프(PNG/TXT)를 생성하려면 다음 명령을 실행하세요:")
        print(f"      python {Path(__file__).name} --run")
        print("      (기존 해석 결과의 그래프/표만 재출력하려면 --post 플래그 사용)")
        print("      (FREQUENCY_RANGE CDAMP 스윕 테스트를 수행하려면 --sweep 플래그 사용)")
        print("      (FREQUENCY_RANGE Min Freq 스윕 테스트를 수행하려면 --sweep_r2 플래그 사용)")


if __name__ == "__main__":
    main()
