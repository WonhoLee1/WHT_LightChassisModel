# -*- coding: utf-8 -*-
"""
R2 스터디: 강체 회전 + 국소 진동 공존 상태에서의 2배 두께(8mm) 효과, VREL Freq 영향 및 Rayleigh 감쇠 비교 검증
"""
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from generate_freefall_damp import (
    create_plate_mesh,
    ci,
    cf,
    get_openradioss_paths,
    analyze_vibration,
    write_engine,
)

TSTOP = 4.0
OMEGA0 = 1.0  # Y축 회전 각속도 [rad/s]
V_LOCAL_TIP = 1500.0  # 로컬 Z방향 추가 속도 [mm/s]
DT_MIN_CST = 2.5e-5
DT_HIS = 0.005
PLATE_THICK_R2 = 8.0  # R2 스터디 두께: 8mm (기존 4mm의 2배)


def write_starter(fp, nd, qd, fix_nids, free_nids, cog_x, cog_y, tip_nid, use_damping, cdamp, rbody_id, freq=1.3, use_rayleigh=False, cdamp_rayleigh=0.0, rayleigh_fmin=0.5):
    all_nids = sorted(nd.keys())
    with open(fp, "w", encoding="utf-8") as f:
        w = f.write
        w("#RADIOSS STARTER\n/BEGIN\ncantilever\n")
        w(f"{2026:>10}{0:>10}\n")
        w(f"{'Mg':<20}{'mm':<20}{'s':<20}\n")
        w(f"{'Mg':<20}{'mm':<20}{'s':<20}\n")

        # 재료 정의 (탄성 물성 E=2000)
        w("/MAT/LAW2/1\nplate_mat\n")
        w(cf(1.0e-9) + "\n")
        w(cf(2000.0) + cf(0.3) + ci(1) + "\n")
        w(cf(100.0) + cf(200.0) + cf(0.2) + cf(0.0) + "\n")
        w(cf(0.0) + cf(0.0) + ci(0) + ci(0) + ci(0) + ci(0) + "\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + cf(0.0) + "\n")

        # 쉘 속성 정의 - 두께 PLATE_THICK_R2 = 8mm 적용
        w("/PROP/SHELL/1\nplate_prop\n")
        w(ci(24) + ci(2) + ci(2) + ci(2) + ci(0) + " " * 10 + cf(0.0) + "\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + cf(0.0) + cf(0.015) + "\n")
        w(ci(5) + " " * 10 + cf(PLATE_THICK_R2) + cf(0.0) + " " * 10 + ci(-1) + ci(-1) + ci(0) + "\n")

        w("/PART/1\ncantilever_plate\n")
        w(ci(1) + ci(1) + "\n")

        w("/NODE\n")
        for nid in all_nids:
            x, y, z = nd[nid]
            w(ci(nid) + cf(x) + cf(y) + cf(z) + "\n")
        w(ci(9999) + cf(cog_x) + cf(cog_y) + cf(0.0) + "\n")

        w("/SHELL/1\n")
        for eid, (n1, n2, n3, n4, pid) in sorted(qd.items()):
            w(ci(eid) + ci(n1) + ci(n2) + ci(n3) + ci(n4) + "\n")

        w("/GRNOD/NODE/1\nall_nodes\n")
        all_with_master = all_nids + [9999]
        for i in range(0, len(all_with_master), 8):
            w("".join(ci(n) for n in all_with_master[i : i + 8]) + "\n")

        w("/GRNOD/NODE/2\nrigid_secondary_nodes\n")
        for i in range(0, len(fix_nids), 8):
            w("".join(ci(n) for n in fix_nids[i : i + 8]) + "\n")

        w("/GRNOD/NODE/4\nfree_region_nodes\n")
        for i in range(0, len(free_nids), 8):
            w("".join(ci(n) for n in free_nids[i : i + 8]) + "\n")

        w("/GRPART/PART/1\nplate_part_group\n")
        w(ci(1) + "\n")

        w("/RBODY/1\nbase_rigid_body\n")
        w(ci(9999) + ci(0) + ci(0) + ci(0) + cf(0.0) + ci(2) + ci(0) + ci(3) + ci(0) + "\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + "\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + "\n")
        w(ci(0) + "\n")

        # ── 초기 각속도(Y회전) + 로컬 Z 진동 속도 프로파일 중첩 ──
        w(f"/INIVEL/NODE/1\ninit_motion\n")
        w(ci(9999) + ci(0) + cf(0.0) + cf(0.0) + cf(0.0) + "\n")
        w(" " * 20 + cf(0.0) + cf(OMEGA0) + cf(0.0) + "\n")
        
        x_root = 1300.0
        L_free = 500.0
        for nid in free_nids:
            x = nd[nid][0]
            v_rigid = -OMEGA0 * (x - cog_x)
            v_local = V_LOCAL_TIP * ((x - x_root) / L_free) ** 2
            vz = v_rigid + v_local
            w(ci(nid) + ci(0) + cf(0.0) + cf(0.0) + cf(vz) + "\n")
            w(" " * 20 + cf(0.0) + cf(0.0) + cf(0.0) + "\n")

        # ── 시간 이력 카드 (/TH/NODE) ──
        w("/TH/NODE/1\ntip_th\n")
        w(f"{'DEF':<10}\n")
        w(ci(tip_nid) + ci(0) + "\n")

        w("/TH/NODE/2\nmaster_th\n")
        w(f"{'DEF':<10}\n")
        w(ci(9999) + ci(0) + "\n")

        w("/TH/NODE/3\nmaster_rot_th\n")
        w(f"{'VRY':<10}\n")
        w(ci(9999) + ci(0) + "\n")

        # VREL 감쇠 정의
        if use_damping:
            w("/DAMP/VREL/1\ndamp_vrel_rot_local\n")
            w(cf(0.0) + cf(0.0) + ci(4) + ci(0) + cf(0.0) + cf(TSTOP) + "\n")
            w(cf(freq) + ci(rbody_id) + ci(0) + cf(1.0) + "\n")
            w(cf(0.0) + cf(0.0) + "\n")
            w(cf(cdamp) + cf(0.0) + "\n")

        # Rayleigh 감쇠 정의 (/DAMP/FREQUENCY_RANGE)
        # 카드1 포맷: Cdamp(20) + blank(10) + blank(10) + grpart_id(10) + blank(10) + Tstart(20) + Tstop(20)
        if use_rayleigh:
            w("/DAMP/FREQUENCY_RANGE/1\n")
            w("damp_rayleigh_rot_local\n")
            w(cf(cdamp_rayleigh) + " " * 10 + " " * 10 + ci(1) + " " * 10 + cf(0.0) + cf(TSTOP) + "\n")
            # rayleigh_fmin ~ 50 Hz 대역 설정
            w(cf(rayleigh_fmin) + cf(50.0) + "\n")

        w("/END\n")


def run(rad_dir: Path, run_name: str):
    base, exec_dir, env = get_openradioss_paths()
    starter_exe = exec_dir / "starter_win64.exe"
    engine_exe = exec_dir / "engine_win64.exe"
    th_to_csv_exe = exec_dir / "th_to_csv_win64.exe"

    subprocess.run([str(starter_exe), "-i", f"{run_name}_0000.rad"], cwd=str(rad_dir), capture_output=True, env=env)
    subprocess.run([str(engine_exe), "-i", f"{run_name}_0001.rad"], cwd=str(rad_dir), capture_output=True, env=env)

    t01 = rad_dir / f"{run_name}T01"
    subprocess.run([str(th_to_csv_exe), t01.name], cwd=str(rad_dir), capture_output=True, env=env)
    return Path(str(t01) + ".csv")


def run_case_sweep(cases, test_dir, nd, qd, fix_nids, free_nids, cog_x, cog_y, tip_nid):
    results = {}
    for name, params in cases.items():
        case_path = test_dir / name
        case_path.mkdir(parents=True, exist_ok=True)
        csv_file = case_path / "cantileverT01.csv"
        if not csv_file.exists():
            sp = str(case_path / "cantilever_0000.rad")
            ep = str(case_path / "cantilever_0001.rad")
            starter_params = params.copy()
            dt_min = starter_params.pop("dt_min", 2.5e-5)
            write_starter(sp, nd, qd, fix_nids, free_nids, cog_x, cog_y, tip_nid, **starter_params)
            write_engine(ep, dt_min=dt_min)
            print(f" -> [{name}] 실행 중...")
            csv_file = run(case_path, "cantilever")
        else:
            print(f" -> [{name}] 기존 결과 로드 (해석 스킵)")
        if not csv_file.exists():
            print(f" -> [{name}] CSV 변환 실패")
            continue

        df = pd.read_csv(csv_file)
        time_val = df["time"].values

        # 컬럼 추출
        tip_cols = [c for c in df.columns if str(tip_nid) in c]
        master_cols = [c for c in df.columns if "9999" in c]
        rot_col_name = [c for c in df.columns if "master_rot_th" in c][0]
        
        dx_tip = df[tip_cols[0]].values
        dz_tip = df[tip_cols[2]].values
        dx_master = df[master_cols[0]].values
        dz_master = df[master_cols[2]].values
        ry_master = df[master_cols[4]].values
        vry_master = df[rot_col_name].values

        # u_bending 투영 (integrated theta 이용)
        dt = np.diff(time_val, prepend=0)
        theta = np.cumsum(vry_master * dt)
        delta_x = (1800.0 + dx_tip) - (cog_x + dx_master)
        delta_z = dz_tip - dz_master
        u_bending = delta_x * np.sin(theta) + delta_z * np.cos(theta)

        # 진동 주파수 및 감쇠비 자동 분석
        try:
            f_num, _, zeta, _, _, _ = analyze_vibration(time_val, u_bending)
        except Exception:
            f_num, zeta = 0.0, 0.0

        results[name] = {
            "time": time_val,
            "vry": vry_master,
            "u_bending": u_bending,
            "f_num": f_num,
            "zeta": zeta,
        }
        print(f"    - f={f_num:.3f} Hz, zeta={zeta*100:.2f} %")
    return results


def main():
    print("=== OpenRadioss R2 스터디 (두께 8mm 효과 검증) ===")
    out_dir = Path(__file__).parent
    test_dir = out_dir / "cases_vrel_rotation_test_R2"
    test_dir.mkdir(parents=True, exist_ok=True)

    nd, qd, fix_nids, free_nids, tip_nid = create_plate_mesh()
    xs = np.array([nd[n][0] for n in nd])
    ys = np.array([nd[n][1] for n in nd])
    cog_x, cog_y = float(xs.mean()), float(ys.mean())

    # -------------------------------------------------------------
    # Case Set 1: 두께 8mm 효과 스윕 (VREL cdamp 스윕)
    # -------------------------------------------------------------
    print("\n[Case Set 1] 두께 8mm VREL cdamp 스윕 실행...")
    cases_set1 = {
        "baseline_nodamp": dict(use_damping=False, cdamp=0.0, rbody_id=1, use_rayleigh=False, cdamp_rayleigh=0.0),
        "vrel_rbody1_cdamp0.05": dict(use_damping=True, cdamp=0.05, rbody_id=1, use_rayleigh=False, cdamp_rayleigh=0.0),
        "vrel_rbody1_cdamp0.1": dict(use_damping=True, cdamp=0.1, rbody_id=1, use_rayleigh=False, cdamp_rayleigh=0.0),
        "vrel_rbody1_cdamp0.5": dict(use_damping=True, cdamp=0.5, rbody_id=1, use_rayleigh=False, cdamp_rayleigh=0.0),
        "vrel_rbody1_cdamp1.0": dict(use_damping=True, cdamp=1.0, rbody_id=1, use_rayleigh=False, cdamp_rayleigh=0.0),
        "vrel_rbody1_cdamp2.0": dict(use_damping=True, cdamp=2.0, rbody_id=1, use_rayleigh=False, cdamp_rayleigh=0.0),
        "vrel_rbody1_cdamp20.0": dict(use_damping=True, cdamp=20.0, rbody_id=1, use_rayleigh=False, cdamp_rayleigh=0.0),
        "vrel_rbody0_cdamp2.0": dict(use_damping=True, cdamp=2.0, rbody_id=0, use_rayleigh=False, cdamp_rayleigh=0.0),
    }
    res_set1 = run_case_sweep(cases_set1, test_dir, nd, qd, fix_nids, free_nids, cog_x, cog_y, tip_nid)

    # -------------------------------------------------------------
    # Case Set 2: VREL Freq 영향 분석 (cdamp=0.5 고정, Freq 스윕)
    # -------------------------------------------------------------
    print("\n[Case Set 2] VREL Freq 영향 분석 스윕 실행...")
    cases_set2 = {
        "vrel_freq_f1.3": dict(use_damping=True, cdamp=0.5, rbody_id=1, freq=1.3, use_rayleigh=False, cdamp_rayleigh=0.0),
        "vrel_freq_f2.6": dict(use_damping=True, cdamp=0.5, rbody_id=1, freq=2.6, use_rayleigh=False, cdamp_rayleigh=0.0),
        "vrel_freq_f5.2": dict(use_damping=True, cdamp=0.5, rbody_id=1, freq=5.2, use_rayleigh=False, cdamp_rayleigh=0.0),
    }
    res_set2 = run_case_sweep(cases_set2, test_dir, nd, qd, fix_nids, free_nids, cog_x, cog_y, tip_nid)

    # -------------------------------------------------------------
    # Case Set 3: Rayleigh 감쇠 비교 검증 (/DAMP/FREQUENCY_RANGE)
    # -------------------------------------------------------------
    print("\n[Case Set 3] Rayleigh 감쇠 비교 실행...")
    cases_set3 = {
        "rayleigh_damp_5pct": dict(use_damping=False, cdamp=0.0, rbody_id=1, use_rayleigh=True, cdamp_rayleigh=0.05, rayleigh_fmin=0.5),
        "rayleigh_damp_5pct_fmin0": dict(use_damping=False, cdamp=0.0, rbody_id=1, use_rayleigh=True, cdamp_rayleigh=0.05, rayleigh_fmin=0.05, dt_min=3.0e-6),
    }
    res_set3 = run_case_sweep(cases_set3, test_dir, nd, qd, fix_nids, free_nids, cog_x, cog_y, tip_nid)

    # 모든 결과 병합
    all_res = {**res_set1, **res_set2, **res_set3}

    # -------------------------------------------------------------
    # 시각화 그래프 1: 두께 2배 효과 (Case Set 1 비교)
    # -------------------------------------------------------------
    import matplotlib.pyplot as plt
    try:
        import koreanize_matplotlib  # noqa: F401
    except ImportError:
        pass

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    set1_labels = {
        "baseline_nodamp": "Baseline (무감쇠)",
        "vrel_rbody1_cdamp0.05": "VREL RbodyID=1, cdamp=0.05",
        "vrel_rbody1_cdamp0.1": "VREL RbodyID=1, cdamp=0.1",
        "vrel_rbody1_cdamp0.5": "VREL RbodyID=1, cdamp=0.5",
        "vrel_rbody1_cdamp1.0": "VREL RbodyID=1, cdamp=1.0",
        "vrel_rbody1_cdamp2.0": "VREL RbodyID=1, cdamp=2.0",
        "vrel_rbody1_cdamp20.0": "VREL RbodyID=1, cdamp=20.0",
        "vrel_rbody0_cdamp2.0": "VREL RbodyID=0 (그룹평균), cdamp=2.0",
    }
    colors_set1 = {
        "baseline_nodamp": "black",
        "vrel_rbody1_cdamp0.05": "#aec7e8",
        "vrel_rbody1_cdamp0.1": "#ffbb78",
        "vrel_rbody1_cdamp0.5": "#2ca02c",
        "vrel_rbody1_cdamp1.0": "#98df8a",
        "vrel_rbody1_cdamp2.0": "#d62728",
        "vrel_rbody1_cdamp20.0": "#ff7f0e",
        "vrel_rbody0_cdamp2.0": "#1f77b4",
    }
    for name in cases_set1:
        if name in all_res:
            r = all_res[name]
            lbl = f"{set1_labels.get(name, name)} (f={r['f_num']:.2f}Hz, $\\zeta$={r['zeta']*100:.2f}%)"
            axes[0].plot(r["time"], r["vry"], label=lbl, color=colors_set1.get(name), linestyle="--" if name == "baseline_nodamp" else "-", alpha=0.8)
            axes[1].plot(r["time"], r["u_bending"], label=lbl, color=colors_set1.get(name), linestyle="--" if name == "baseline_nodamp" else "-", alpha=0.8)
            
    axes[0].set_title("R2 스터디(8mm 두께) - VREL cdamp별 강체 회전 속도(VRY) 이력", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Angular Velocity VRY [rad/s]")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("R2 스터디(8mm 두께) - VREL cdamp별 로컬 굽힘 변위(u_bending) 이력", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Local Bending Displacement [mm]")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot1_path = test_dir / "vrel_rotation_r2_thickness_comparison.png"
    fig.savefig(plot1_path, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 시각화 그래프 2: VREL Freq 영향 분석 (Case Set 2 비교)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    set2_labels = {
        "vrel_freq_f1.3": "Freq = 1.3 Hz (Target)",
        "vrel_freq_f2.6": "Freq = 2.6 Hz (2x Damping)",
        "vrel_freq_f5.2": "Freq = 5.2 Hz (4x Damping)",
    }
    colors_set2 = {
        "vrel_freq_f1.3": "#1f77b4",
        "vrel_freq_f2.6": "#2ca02c",
        "vrel_freq_f5.2": "#d62728",
    }
    for name in cases_set2:
        if name in all_res:
            r = all_res[name]
            lbl = f"{set2_labels.get(name, name)} (f={r['f_num']:.2f}Hz, $\\zeta$={r['zeta']*100:.2f}%)"
            axes[0].plot(r["time"], r["vry"], label=lbl, color=colors_set2.get(name), alpha=0.8)
            axes[1].plot(r["time"], r["u_bending"], label=lbl, color=colors_set2.get(name), alpha=0.8)

    axes[0].set_title("R2 스터디(8mm 두께) - VREL Freq 파라미터별 강체 회전 속도(VRY)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Angular Velocity VRY [rad/s]")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("R2 스터디(8mm 두께) - VREL Freq 파라미터별 로컬 굽힘 변위(u_bending)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Local Bending Displacement [mm]")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot2_path = test_dir / "vrel_rotation_r2_freq_comparison.png"
    fig.savefig(plot2_path, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------
    # 시각화 그래프 3: Rayleigh 감쇠 비교 (VREL vs FREQUENCY_RANGE)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    # VREL cdamp=0.05 케이스와 Rayleigh cdamp=0.05 케이스 직접 비교
    compare_cases = ["baseline_nodamp", "vrel_rbody1_cdamp0.5", "rayleigh_damp_5pct", "rayleigh_damp_5pct_fmin0"]
    labels_set3 = {
        "baseline_nodamp": "Baseline (무감쇠)",
        "vrel_rbody1_cdamp0.5": "VREL RbodyID=1, cdamp=0.5 (회전 보존)",
        "rayleigh_damp_5pct": "Rayleigh fmin=0.5 Hz, cdamp=0.05 (회전 보존)",
        "rayleigh_damp_5pct_fmin0": "Rayleigh fmin=0.0 Hz, cdamp=0.05 (회전 감쇠)",
    }
    colors_set3 = {
        "baseline_nodamp": "black",
        "vrel_rbody1_cdamp0.5": "#2ca02c",
        "rayleigh_damp_5pct": "#d62728",
        "rayleigh_damp_5pct_fmin0": "#1f77b4",
    }
    for name in compare_cases:
        if name in all_res:
            r = all_res[name]
            lbl = f"{labels_set3.get(name, name)} (f={r['f_num']:.2f}Hz, $\\zeta$={r['zeta']*100:.2f}%)"
            axes[0].plot(r["time"], r["vry"], label=lbl, color=colors_set3.get(name), linestyle="--" if name == "baseline_nodamp" else "-", alpha=0.8)
            axes[1].plot(r["time"], r["u_bending"], label=lbl, color=colors_set3.get(name), linestyle="--" if name == "baseline_nodamp" else "-", alpha=0.8)

    axes[0].set_title("VREL vs Rayleigh 감쇠 비교 - 강체 회전 속도(VRY)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Angular Velocity VRY [rad/s]")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].set_title("VREL vs Rayleigh 감쇠 비교 - 로컬 굽힘 변위(u_bending)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Local Bending Displacement [mm]")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot3_path = test_dir / "vrel_rotation_r2_rayleigh_comparison.png"
    fig.savefig(plot3_path, dpi=300)
    plt.close(fig)

    print("\n=== R2 스터디 요약 (두께 8mm 평판) ===")
    for name, r in all_res.items():
        print(f"{name:28s}: f={r['f_num']:.3f} Hz  zeta={r['zeta']*100:.3f} %")


if __name__ == "__main__":
    main()
