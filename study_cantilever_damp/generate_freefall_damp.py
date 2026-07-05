# -*- coding: utf-8 -*-
"""
OpenRadioss 자유낙하 중 진동(Free-fall with Vibration) 및 주파수 대역 감쇠 테스트 생성기
목적: 강체 낙하 거동(0 Hz)은 감쇠되지 않고, 탄성 진동 모드만 선택적으로 감쇠됨을 증명
단위계: Mg, mm, s
"""

import math
import os
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib
except ImportError:
    pass

# 평판 기하 및 재료 파라미터 (Mg, mm, s)
PLATE_LEN_X = 1800.0
PLATE_LEN_Y = 1200.0
PLATE_THICK = 4.0
FREE_LEN_X = 500.0
FIXED_LEN_X = PLATE_LEN_X - FREE_LEN_X  # 1300 mm (강체 베이스 영역)
MESH_SIZE = 50.0  # 50 mm 메쉬 적용

E_PLATE = 2000.0
RHO_PLATE = 1.0e-9
NU_PLATE = 0.3
SIG_Y_PLATE = 100.0
SIG_UTS_PLATE = 200.0
EPS_UTS_PLATE = 0.2
EPS_MAX_PLATE = 0.0

GRAVITY = 9810.0  # 중력 가속도 [mm/s^2]
TSTOP = 8.0  # 자유낙하 해석 시간 8초 (로컬 1.3Hz 모드의 로그감소율 분석에 충분한 사이클 확보)
DT_MIN_CST = 2.5e-5
DT_ANIM = 0.05
DT_HIS = 0.005


def ci(v: int, w: int = 10) -> str:
    return f"{v:>{w}d}"


def cf(v: float, w: int = 20) -> str:
    return f"{v:>{w}.6E}"


def create_plate_mesh():
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
            nd[nid] = (x, y, 0.0)
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

    # 끝단 중앙 노드 (X=1800, Y=600 근처) 탐색 (진동 기록 및 가속 충격 가함)
    target_pt = np.array([PLATE_LEN_X, PLATE_LEN_Y / 2.0, 0.0])
    min_dist = float("inf")
    tip_nid = -1
    for n_id, coords in nd.items():
        dist = np.linalg.norm(np.array(coords) - target_pt)
        if dist < min_dist:
            min_dist = dist
            tip_nid = n_id

    return nd, qd, fix_nids, free_nids, tip_nid


def write_starter(
    fp: str,
    nd: dict,
    qd: dict,
    fix_nids: list,
    free_nids: list,
    tip_nid: int,
    use_damping: bool,
    cdamp: float = 0.05,
    damp_freq: float = 1.3,
    alpha_is_raw: bool = False,
    rbody_id: int = 1,
):
    all_nids = sorted(nd.keys())

    with open(fp, "w", encoding="utf-8") as f:
        w = f.write
        w("#RADIOSS STARTER\n")
        w("/BEGIN\n")
        w("cantilever\n")
        w(f"{2026:>10}{0:>10}\n")
        w(f"{'Mg':<20}{'mm':<20}{'s':<20}\n")
        w(f"{'Mg':<20}{'mm':<20}{'s':<20}\n")

        # 재료 정의 (/MAT/LAW2)
        w("/MAT/LAW2/1\nplate_mat\n")
        w(cf(RHO_PLATE) + "\n")
        w(cf(E_PLATE) + cf(NU_PLATE) + ci(1) + "\n")
        w(
            cf(SIG_Y_PLATE)
            + cf(SIG_UTS_PLATE)
            + cf(EPS_UTS_PLATE)
            + cf(EPS_MAX_PLATE)
            + "\n"
        )
        w(cf(0.0) + cf(0.0) + ci(0) + ci(0) + ci(0) + ci(0) + "\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + cf(0.0) + "\n")

        # 쉘 속성 정의 (/PROP/SHELL)
        w("/PROP/SHELL/1\nplate_prop\n")
        w(ci(24) + ci(2) + ci(2) + ci(2) + ci(0) + " " * 10 + cf(0.0) + "\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + cf(0.0) + cf(0.015) + "\n")
        w(
            ci(5)
            + " " * 10
            + cf(PLATE_THICK)
            + cf(0.0)
            + " " * 10
            + ci(-1)
            + ci(-1)
            + ci(0)
            + "\n"
        )

        # 파트 정의
        w("/PART/1\ncantilever_plate\n")
        w(ci(1) + ci(1) + "\n")

        # 노드 정의
        w("/NODE\n")
        for nid in all_nids:
            x, y, z = nd[nid]
            w(ci(nid) + cf(x) + cf(y) + cf(z) + "\n")
        # 강체 마스터 노드 추가 (ID=9999)
        w(ci(9999) + cf(FIXED_LEN_X / 2.0) + cf(PLATE_LEN_Y / 2.0) + cf(0.0) + "\n")

        # 요소 정의
        w("/SHELL/1\n")
        for eid, (n1, n2, n3, n4, pid) in sorted(qd.items()):
            w(ci(eid) + ci(n1) + ci(n2) + ci(n3) + ci(n4) + "\n")

        # 노드 그룹 1 (전체 노드 + 마스터 노드 9999)
        w("/GRNOD/NODE/1\nall_nodes\n")
        all_nids_with_master = all_nids + [9999]
        for i in range(0, len(all_nids_with_master), 8):
            w("".join(ci(n) for n in all_nids_with_master[i : i + 8]) + "\n")

        # 노드 그룹 2 (강체 Secondary 노드들)
        w("/GRNOD/NODE/2\nrigid_secondary_nodes\n")
        for i in range(0, len(fix_nids), 8):
            w("".join(ci(n) for n in fix_nids[i : i + 8]) + "\n")

        # 노드 그룹 3 (끝단 충격 하중용 단일 노드 그룹, /CLOAD는 GRNOD ID를 요구함)
        w("/GRNOD/NODE/3\ntip_impact_node\n")
        w(ci(tip_nid) + "\n")

        # 노드 그룹 4 (자유단 노드 전체, /DAMP/VREL 감쇠 적용 대상)
        w("/GRNOD/NODE/4\nfree_region_nodes\n")
        for i in range(0, len(free_nids), 8):
            w("".join(ci(n) for n in free_nids[i : i + 8]) + "\n")

        # 파트 그룹
        w("/GRPART/PART/1\nplate_part_group\n")
        w(ci(1) + "\n")

        # ── 이동식 강체 베이스 정의 (/RBODY) ──
        # Gnod_id=2 (rigid_secondary_nodes), ICOG=3
        w("/RBODY/1\nbase_rigid_body\n")
        # Line 2: node_ID(10) sens_ID(10) Skew_ID(10) Ispher(10) Mass(20) Gnod_id(10) Ikrem(10) ICoG(10) Surf_id(10)
        w(
            ci(9999)
            + ci(0)
            + ci(0)
            + ci(0)
            + cf(0.0)
            + ci(2)
            + ci(0)
            + ci(3)
            + ci(0)
            + "\n"
        )
        # Line 3: Jxx Jyy Jzz
        w(cf(0.0) + cf(0.0) + cf(0.0) + "\n")
        # Line 4: Jxy Jyz Jxz
        w(cf(0.0) + cf(0.0) + cf(0.0) + "\n")
        # Line 5: Ioptoff
        w(ci(0) + "\n")

        # 중력 정의 (전체 자유낙하 유도, 강체 마스터 노드 및 탄성 빔 전체에 가해짐)
        w("/GRAV/1\ngravity_load\n")
        w(
            ci(0)
            + f"{'Z':>10}"
            + ci(0)
            + ci(0)
            + ci(1)
            + " " * 10
            + cf(1.0)
            + cf(-GRAVITY)
            + "\n"
        )

        # ── 과도 충격 하중 정의 (외팔보 진동 유발) ──
        # 끝단 중앙 노드(tip_nid)에 0.05초 동안 500 N의 집중 하중을 주어 휘어지게 함
        w("/CLOAD/1\ntip_impact\n")
        w("#funct_IDT       Dir   skew_ID sensor_ID  grnod_ID   Itypfun             Ascalex             Fscaley\n")
        w(
            ci(1)
            + f"{'Z':>10}"
            + ci(0)
            + ci(0)
            + ci(3)
            + ci(1)
            + cf(1.0)
            + cf(1.0)
            + "\n"
        )

        # 하중 시간 함수 (/FUNCT)
        w("/FUNCT/1\nimpact_curve\n")
        w(cf(0.0) + cf(500.0) + "\n")
        w(cf(0.05) + cf(500.0) + "\n")
        w(cf(0.051) + cf(0.0) + "\n")
        w(cf(TSTOP) + cf(0.0) + "\n")

        # 끝단 노드 Z방향 변위 및 강체 마스터 노드 Z방향 변위 기록 (/TH/NODE)
        w("/TH/NODE/1\ncantilever_tip_th\n")
        w(f"{'DEF':<10}\n")
        w(ci(tip_nid) + ci(0) + "\n")

        w("/TH/NODE/2\nbase_master_th\n")
        w(f"{'DEF':<10}\n")
        w(ci(9999) + ci(0) + "\n")

        # 감쇠 카드 (/DAMP/VREL: 자유단 노드의 "RBODY(강체 베이스) 대비 상대 속도"에
        # 비례하는 점성 감쇠력을 가함. 강체 베이스 자체의 낙하 속도는 항상 자기 자신과의
        # 상대속도=0 이므로 절대 감쇠되지 않고, 로컬 탄성 진동(상대 속도)만 감쇠됨.
        # Alpha_z는 질량비례 감쇠계수(1/s 단위)로, 목표 감쇠비 zeta와 목표 진동수 Freq에
        # 대해 zeta = Alpha_z / (2 * 2*pi*Freq) 관계를 가짐 -> Alpha_z = 2*zeta*2*pi*Freq
        # 카드1: Alpha_x(20) Alpha2_x(20) grnod_id(10) skew_id(10) Tstart(20) Tstop(20)
        # 카드2: Freq(20) RbodyID(10) FuncID(10) Xscale(20)
        # 카드3: Alpha_y(20) Alpha2_y(20)
        # 카드4: Alpha_z(20) Alpha2_z(20)
        alpha_z = cdamp if alpha_is_raw else 2.0 * cdamp * 2.0 * math.pi * damp_freq
        if use_damping:
            w("/DAMP/VREL/1\n")
            w("damp_vrel_test\n")
            w(cf(0.0) + cf(0.0) + ci(4) + ci(0) + cf(0.0) + cf(TSTOP) + "\n")
            w(cf(damp_freq) + ci(rbody_id) + ci(0) + cf(1.0) + "\n")
            w(cf(0.0) + cf(0.0) + "\n")
            w(cf(alpha_z) + cf(0.0) + "\n")
        else:
            w("#/DAMP/VREL/1\n")
            w("#damp_vrel_test\n")
            w("#" + cf(0.0) + cf(0.0) + ci(4) + ci(0) + cf(0.0) + cf(TSTOP) + "\n")
            w("#" + cf(damp_freq) + ci(1) + ci(0) + cf(1.0) + "\n")
            w("#" + cf(0.0) + cf(0.0) + "\n")
            w("#" + cf(alpha_z) + cf(0.0) + "\n")

        w("/END\n")
    print(f"[OK] Starter 생성 완료 -> {fp}")


def write_engine(fp: str, dt_min: float = 2.5e-5):
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
        w("/END/ENGINE\n")
    print(f"[OK] Engine 생성 완료 -> {fp}")


def get_openradioss_paths():
    for base_str in [r"D:\OpenRadioss", r"D:\OpenRadioss_win64\OpenRadioss"]:
        base = Path(base_str)
        exec_dir = base / "exec"
        if exec_dir.exists():
            env = os.environ.copy()
            env["PATH"] = os.pathsep.join(
                [
                    str(exec_dir),
                    str(base / "extlib" / "hm_reader" / "win64"),
                    str(base / "extlib" / "intelOneAPI_runtime" / "win64"),
                    env.get("PATH", ""),
                ]
            )
            env["RAD_CFG_PATH"] = str(base / "hm_cfg_files")
            env["RAD_H3D_PATH"] = str(base / "extlib" / "h3d" / "lib" / "win64")
            return base, exec_dir, env
    raise RuntimeError("OpenRadioss 설치 경로를 찾을 수 없습니다.")


def run_openradioss(rad_dir: Path, run_name: str):
    base, exec_dir, env = get_openradioss_paths()
    starter_exe = exec_dir / "starter_win64.exe"
    engine_exe = exec_dir / "engine_win64.exe"
    th_to_csv_exe = exec_dir / "th_to_csv_win64.exe"

    starter_rad = rad_dir / f"{run_name}_0000.rad"
    engine_rad = rad_dir / f"{run_name}_0001.rad"

    subprocess.run(
        [str(starter_exe), "-i", starter_rad.name],
        cwd=str(rad_dir),
        capture_output=True,
        env=env,
    )
    subprocess.run(
        [str(engine_exe), "-i", engine_rad.name],
        cwd=str(rad_dir),
        capture_output=True,
        env=env,
    )

    t01_file = rad_dir / f"{run_name}T01"
    if not t01_file.exists():
        t01_file = rad_dir / f"{run_name}t01"

    subprocess.run(
        [str(th_to_csv_exe), str(t01_file)],
        cwd=str(rad_dir),
        capture_output=True,
        env=env,
    )
    return Path(str(t01_file) + ".csv")


def analyze_vibration(time_val: np.ndarray, disp_val: np.ndarray):
    """
    시간 이력 변위 데이터로부터 고유 진동 주파수와 감쇠비(Damping Ratio)를 계산.
    generate_cantilever_damp.py의 동일 함수 로직 재사용 (로그 감소율법).
    """
    from scipy.signal import find_peaks
    from scipy.stats import linregress

    dt = np.mean(np.diff(time_val))

    t_2s_mask = time_val <= 2.0
    n_2s = np.sum(t_2s_mask)
    if n_2s > 10:
        disp_centered_2s = disp_val[t_2s_mask] - np.mean(disp_val[t_2s_mask])
        fft_freqs = np.fft.rfft(disp_centered_2s)
        freqs = np.fft.rfftfreq(n_2s, d=dt)
        valid_idx = np.where((freqs >= 0.2) & (freqs <= 50.0))[0]
        if len(valid_idx) > 0:
            best_i = valid_idx[np.argmax(np.abs(fft_freqs[valid_idx]))]
            f_fft = freqs[best_i]
        else:
            f_fft = freqs[np.argmax(np.abs(fft_freqs[1:])) + 1]
    else:
        f_fft = 1.0

    f_num = f_fft if f_fft > 0 else 1.0
    period_mean = 1.0 / f_num

    dist_samples = max(5, int((0.7 * period_mean) / dt))
    valleys, _ = find_peaks(-disp_val, distance=dist_samples)
    peaks, _ = find_peaks(disp_val, distance=dist_samples)

    if len(valleys) > 1:
        valley_times = time_val[valleys]
        periods = np.diff(valley_times)
        period_mean = np.mean(periods)
        if period_mean > 0:
            f_num = 1.0 / period_mean

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
            threshold = 0.02 * amps[0]
            valid_mask = amps >= threshold
            if np.sum(valid_mask) >= 2:
                amp_times = amp_times[valid_mask]
                amps = amps[valid_mask]

            slope, intercept, r_value, p_value, std_err = linregress(
                amp_times, np.log(amps)
            )
            if slope < 0 and f_num > 0:
                zeta = -slope / (2.0 * math.pi * f_num)
            else:
                zeta = 0.0

    return f_num, period_mean, zeta, f_fft, valleys, peaks


def main():
    print("=== OpenRadioss 자유낙하 진동 해석 스터디 ===")
    out_dir = Path(__file__).parent
    freefall_dir = out_dir / "cases_freefall"
    freefall_dir.mkdir(parents=True, exist_ok=True)

    nd, qd, fix_nids, free_nids, tip_nid = create_plate_mesh()
    print(
        f"Mesh: {len(nd)} nodes, {len(qd)} elements. Clamped base nodes: {len(fix_nids)}"
    )

    cases = {"undamped": False, "damped_5pct": True}

    results = {}

    for case_name, use_damp in cases.items():
        case_path = freefall_dir / case_name
        case_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[Case: {case_name}] Starter/Engine 작성 중...")
        sp = str(case_path / "cantilever_0000.rad")
        ep = str(case_path / "cantilever_0001.rad")
        write_starter(sp, nd, qd, fix_nids, free_nids, tip_nid, use_damp, cdamp=0.05)
        write_engine(ep)

        print(f"[Case: {case_name}] OpenRadioss 해석 실행 중...")
        csv_file = run_openradioss(case_path, "cantilever")

        if csv_file.exists():
            df = pd.read_csv(csv_file)
            time_col = df.columns[0]
            for col in df.columns:
                if "time" in col.lower() or "t" == col.lower().strip():
                    time_col = col
                    break

            # /TH/NODE "DEF" 변수는 노드당 6개 컬럼(DX,DY,DZ,VX,VY,VZ) 순으로 출력됨
            # -> 3번째(DZ)가 변위, 6번째(VZ)가 속도
            tip_cols = [c for c in df.columns if str(tip_nid) in c]
            master_cols = [c for c in df.columns if "9999" in c]
            tip_disp_col = tip_cols[2]
            master_disp_col, master_vel_col = master_cols[2], master_cols[5]

            time_val = df[time_col].values
            disp_tip = df[tip_disp_col].values
            disp_master = df[master_disp_col].values
            vel_master = df[master_vel_col].values

            # 끝단 노드의 상대 변위 (외팔보 자체의 탄성 진동 변위)
            relative_disp = disp_tip - disp_master

            f_num, period_mean, zeta, f_fft, valleys, peaks = analyze_vibration(
                time_val, relative_disp
            )
            print(
                f" -> [{case_name}] 로컬 진동 주파수: {f_num:.4f} Hz (FFT: {f_fft:.4f} Hz), "
                f"감쇠비(zeta): {zeta:.6f} ({zeta * 100.0:.2f} %)"
            )

            results[case_name] = {
                "time": time_val,
                "disp_tip": disp_tip,
                "disp_master": disp_master,
                "vel_master": vel_master,
                "relative_disp": relative_disp,
                "f_num": f_num,
                "zeta": zeta,
            }
            print(f"[Case: {case_name}] 완료.")

    # 시각화 그래프 생성
    if len(results) == 2:
        plt.rcParams.update({"font.size": 9})
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # 1. 강체 마스터 노드의 Z방향 낙하 속도 (강체 모션의 감쇠 여부 검증)
        ax0 = axes[0]
        theory_vel = -GRAVITY * results["undamped"]["time"]
        ax0.plot(
            results["undamped"]["time"],
            results["undamped"]["vel_master"],
            label="Undamped Base (Rigid Body)",
            color="#d62728",
            alpha=0.8,
        )
        ax0.plot(
            results["damped_5pct"]["time"],
            results["damped_5pct"]["vel_master"],
            label="Damped 5% Base (Rigid Body)",
            color="#1f77b4",
            alpha=0.8,
        )
        ax0.plot(
            results["undamped"]["time"],
            theory_vel,
            "--",
            label="Theoretical Pure Freefall (v = -gt)",
            color="black",
            linewidth=1.5,
        )

        ax0.set_title(
            "강체 베이스(Rigid Base)의 Z방향 낙하 속도 응답 비교 (강체 모션 감쇠 여부)",
            fontsize=11,
            fontweight="bold",
        )
        ax0.set_xlabel("Time [s]", fontsize=10)
        ax0.set_ylabel("Z-Velocity [mm/s]", fontsize=10)
        ax0.grid(True, linestyle="--", alpha=0.6)
        ax0.legend(loc="best", frameon=True)

        # 2. 강체 베이스 대비 끝단의 상대 변위 (로컬 탄성 외팔보 진동의 감쇠 비교)
        ax1 = axes[1]
        ax1.plot(
            results["undamped"]["time"],
            results["undamped"]["relative_disp"],
            label="Undamped (Relative Elastic Vibration)",
            color="#d62728",
            alpha=0.8,
        )
        ax1.plot(
            results["damped_5pct"]["time"],
            results["damped_5pct"]["relative_disp"],
            label="Damped 5% (Relative Elastic Vibration)",
            color="#1f77b4",
            alpha=0.8,
        )

        ax1.set_title(
            "강체 베이스 대비 끝단의 상대 변위 이력 (순수 로컬 외팔보 진동 감쇠)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_xlabel("Time [s]", fontsize=10)
        ax1.set_ylabel("Relative Z-Displacement [mm]", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="best", frameon=True)

        info_text = (
            f"Undamped  : f={results['undamped']['f_num']:.3f} Hz, "
            f"zeta={results['undamped']['zeta'] * 100.0:.2f} %\n"
            f"Damped 5% : f={results['damped_5pct']['f_num']:.3f} Hz, "
            f"zeta={results['damped_5pct']['zeta'] * 100.0:.2f} %"
        )
        ax1.text(
            0.02,
            0.02,
            info_text,
            transform=ax1.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plot_path = freefall_dir / "freefall_damping_comparison.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"\n[OK] 자유낙하 강체-로컬 감쇠 분리 검증 완료 -> {plot_path}")

        txt_path = freefall_dir / "freefall_damping_summary.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("# ================================================================\n")
            f.write("# 자유낙하 강체-로컬 감쇠 분리 검증 요약\n")
            f.write("# ================================================================\n")
            for case_name in ("undamped", "damped_5pct"):
                r = results[case_name]
                f.write(f"# [{case_name}]\n")
                f.write(f"#  - 로컬 진동 주파수: {r['f_num']:.4f} Hz\n")
                f.write(
                    f"#  - 감쇠비(zeta)    : {r['zeta']:.6f} ({r['zeta'] * 100.0:.2f} %)\n"
                )
            f.write("# ================================================================\n")
        print(f"[OK] 감쇠비 요약 TXT 저장 완료 -> {txt_path}")


if __name__ == "__main__":
    main()
