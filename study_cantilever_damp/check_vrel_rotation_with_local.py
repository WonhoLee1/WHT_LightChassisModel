# -*- coding: utf-8 -*-
"""
강체 회전(Spin)과 로컬 탄성 진동이 동시에 존재하는 상태에서,
/DAMP/VREL의 RbodyID 및 cdamp 크기가 강체 회전과 로컬 진동에 각각 어떻게 영향을 주는지 검증하는 스크립트.
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
)

TSTOP = 4.0
OMEGA0 = 1.0  # Y축 회전 각속도 [rad/s]
V_LOCAL_TIP = 1500.0  # 로컬 Z방향 추가 속도 [mm/s] (진동 여기용)
DT_MIN_CST = 2.5e-5
DT_HIS = 0.005


def write_starter(fp, nd, qd, fix_nids, free_nids, cog_x, cog_y, tip_nid, use_damping, cdamp, rbody_id, skew_id=0):
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

        w("/PROP/SHELL/1\nplate_prop\n")
        w(ci(24) + ci(2) + ci(2) + ci(2) + ci(0) + " " * 10 + cf(0.0) + "\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + cf(0.0) + cf(0.015) + "\n")
        w(ci(5) + " " * 10 + cf(4.0) + cf(0.0) + " " * 10 + ci(-1) + ci(-1) + ci(0) + "\n")

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
        # /INIVEL/NODE 카드로 마스터 노드 및 자유단 노드에 각각 중첩된 속도 부여
        w(f"/INIVEL/NODE/1\ninit_motion\n")
        
        # 1. 마스터 노드 (Y축 회전 각속도 부여)
        w(ci(9999) + ci(0) + cf(0.0) + cf(0.0) + cf(0.0) + "\n")
        w(" " * 20 + cf(0.0) + cf(OMEGA0) + cf(0.0) + "\n")
        
        # 2. 자유단 노드들 (회전 선속도 -omega*(x-cog_x) + 로컬 굽힘 여기 속도 v_local)
        x_root = 1300.0
        L_free = 500.0
        for nid in free_nids:
            x = nd[nid][0]
            v_rigid = -OMEGA0 * (x - cog_x)
            # 끝단으로 갈수록 2차식 형태로 커지는 진동 속도 성분 부여
            v_local = V_LOCAL_TIP * ((x - x_root) / L_free) ** 2
            vz = v_rigid + v_local
            w(ci(nid) + ci(0) + cf(0.0) + cf(0.0) + cf(vz) + "\n")
            w(" " * 20 + cf(0.0) + cf(0.0) + cf(0.0) + "\n")

        # ── 시간 이력 카드 (/TH/NODE) ──
        # 끝단 노드 변위 기록
        w("/TH/NODE/1\ntip_th\n")
        w(f"{'DEF':<10}\n")
        w(ci(tip_nid) + ci(0) + "\n")

        # 마스터 노드 변위 및 회전 기록
        w("/TH/NODE/2\nmaster_th\n")
        w(f"{'DEF':<10}\n")
        w(ci(9999) + ci(0) + "\n")

        # 마스터 노드 Y축 회전속도(VRY) 직접 기록
        w("/TH/NODE/3\nmaster_rot_th\n")
        w(f"{'VRY':<10}\n")
        w(ci(9999) + ci(0) + "\n")

        if use_damping:
            w("/DAMP/VREL/1\ndamp_vrel_rot_local\n")
            w(cf(0.0) + cf(0.0) + ci(4) + ci(skew_id) + cf(0.0) + cf(TSTOP) + "\n")
            w(cf(1.3) + ci(rbody_id) + ci(0) + cf(1.0) + "\n")
            w(cf(0.0) + cf(0.0) + "\n")
            w(cf(cdamp) + cf(0.0) + "\n")

        w("/END\n")


def write_engine(fp):
    with open(fp, "w", encoding="utf-8") as f:
        w = f.write
        w("#RADIOSS ENGINE\n/RUN/cantilever/1\n")
        w(cf(TSTOP) + "\n")
        w("/DT/NODA/CST\n")
        w(cf(0.9) + cf(DT_MIN_CST) + "\n")
        w("/TFILE\n")
        w(cf(DT_HIS) + "\n")
        w("/END/ENGINE\n")


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


def main():
    out_dir = Path(__file__).parent
    test_dir = out_dir / "cases_vrel_rotation_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    nd, qd, fix_nids, free_nids, tip_nid = create_plate_mesh()
    xs = np.array([nd[n][0] for n in nd])
    ys = np.array([nd[n][1] for n in nd])
    cog_x, cog_y = float(xs.mean()), float(ys.mean())
    print(f"System COG: X={cog_x:.1f}, Y={cog_y:.1f}, tip_nid={tip_nid}")

    cases = {
        "baseline_nodamp": dict(use_damping=False, cdamp=0.0, rbody_id=1),
        "vrel_rbody1_cdamp0.05": dict(use_damping=True, cdamp=0.05, rbody_id=1),
        "vrel_rbody1_cdamp0.1": dict(use_damping=True, cdamp=0.1, rbody_id=1),
        "vrel_rbody1_cdamp0.5": dict(use_damping=True, cdamp=0.5, rbody_id=1),
        "vrel_rbody1_cdamp1.0": dict(use_damping=True, cdamp=1.0, rbody_id=1),
        "vrel_rbody1_cdamp2": dict(use_damping=True, cdamp=2.0, rbody_id=1),
        "vrel_rbody1_cdamp20": dict(use_damping=True, cdamp=20.0, rbody_id=1),
        "vrel_rbody0_cdamp2": dict(use_damping=True, cdamp=2.0, rbody_id=0),
    }

    results = {}
    for name, params in cases.items():
        case_path = test_dir / (name + "_with_local")
        case_path.mkdir(parents=True, exist_ok=True)
        csv_file = case_path / "cantileverT01.csv"
        if not csv_file.exists():
            sp = str(case_path / "cantilever_0000.rad")
            ep = str(case_path / "cantilever_0001.rad")
            write_starter(sp, nd, qd, fix_nids, free_nids, cog_x, cog_y, tip_nid, **params)
            write_engine(ep)
            print(f"\n[{name}] 실행 중...")
            csv_file = run(case_path, "cantilever")
        else:
            print(f"\n[{name}] 기존 결과 로드 (해석 스킵)")
        if not csv_file.exists():
            print(f"[{name}] CSV 없음")
            continue

        df = pd.read_csv(csv_file)
        time_val = df["time"].values

        # 컬럼 구분
        tip_cols = [c for c in df.columns if str(tip_nid) in c]
        master_cols = [c for c in df.columns if "9999" in c]
        rot_cols = [c for c in df.columns if "9999" in c and "var 3" in c] # VRY 찾기용
        
        # tip 변위 (DX, DZ)
        dx_tip = df[tip_cols[0]].values
        dz_tip = df[tip_cols[2]].values
        # master 변위 (DX, DZ, RY)
        dx_master = df[master_cols[0]].values
        dz_master = df[master_cols[2]].values
        ry_master = df[master_cols[4]].values # 5번째 컬럼 = RY (RotY)
        
        # master Y축 회전속도 (VRY)
        # TH/NODE/3은 VRY 1개 컬럼만 출력하므로 첫번째 컬럼 참조
        rot_col_name = [c for c in df.columns if "master_rot_th" in c][0]
        vry_master = df[rot_col_name].values

        # ── 궤도 회전 행렬 투영을 통한 순수 로컬 굽힘 변위 추출 ──
        dt = np.diff(time_val, prepend=0)
        theta = np.cumsum(vry_master * dt)
        delta_x = (1800.0 + dx_tip) - (cog_x + dx_master)
        delta_z = dz_tip - dz_master
        
        # 로컬 z'축 투영: u_bending = delta_x * sin(theta) + delta_z * cos(theta)
        u_bending = delta_x * np.sin(theta) + delta_z * np.cos(theta)

        results[name] = {
            "time": time_val,
            "vry": vry_master,
            "u_bending": u_bending,
        }

    # 그래프 시각화
    if len(results) >= 2:
        import matplotlib.pyplot as plt
        try:
            import koreanize_matplotlib  # noqa: F401
        except ImportError:
            pass

        plt.rcParams.update({"font.size": 9})
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        labels = {
            "baseline_nodamp": "Baseline (무감쇠)",
            "vrel_rbody1_cdamp0.05": "VREL RbodyID=1, cdamp=0.05",
            "vrel_rbody1_cdamp0.1": "VREL RbodyID=1, cdamp=0.1",
            "vrel_rbody1_cdamp0.5": "VREL RbodyID=1, cdamp=0.5",
            "vrel_rbody1_cdamp1.0": "VREL RbodyID=1, cdamp=1.0",
            "vrel_rbody1_cdamp2": "VREL RbodyID=1, cdamp=2.0",
            "vrel_rbody1_cdamp20": "VREL RbodyID=1, cdamp=20.0",
            "vrel_rbody0_cdamp2": "VREL RbodyID=0 (그룹평균), cdamp=2.0",
        }
        colors = {
            "baseline_nodamp": "black",
            "vrel_rbody1_cdamp0.05": "#aec7e8",
            "vrel_rbody1_cdamp0.1": "#ffbb78",
            "vrel_rbody1_cdamp0.5": "#2ca02c",
            "vrel_rbody1_cdamp1.0": "#98df8a",
            "vrel_rbody1_cdamp2": "#d62728",
            "vrel_rbody1_cdamp20": "#ff7f0e",
            "vrel_rbody0_cdamp2": "#1f77b4",
        }
        styles = {
            "baseline_nodamp": "--",
        }

        # 1. 강체 회전 각속도 (VRY)
        ax0 = axes[0]
        for name, r in results.items():
            ax0.plot(
                r["time"], r["vry"],
                linestyle=styles.get(name, "-"),
                label=labels.get(name, name),
                color=colors.get(name),
                linewidth=1.5,
            )
        ax0.set_title("1. 마스터 노드 Z-회전 각속도 (VRY) 이력 (강체 회전 보존 여부)", fontsize=11, fontweight="bold")
        ax0.set_xlabel("Time [s]")
        ax0.set_ylabel("Angular Velocity VRY [rad/s]")
        ax0.grid(True, linestyle="--", alpha=0.6)
        ax0.legend(loc="best", frameon=True)

        # 2. 로컬 굽힘 탄성 진동
        ax1 = axes[1]
        for name, r in results.items():
            ax1.plot(
                r["time"], r["u_bending"],
                linestyle=styles.get(name, "-"),
                label=labels.get(name, name),
                color=colors.get(name),
                linewidth=1.3,
                alpha=0.8,
            )
        ax1.set_title("2. 회전 좌표계로 투영한 끝단 로컬 굽힘 변위 이력 (로컬 진동 감쇠 효과)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("Local Bending Displacement u_bending [mm]")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="best", frameon=True)

        plt.tight_layout()
        plot_path = test_dir / "vrel_rotation_with_local_comparison.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"\n[OK] 강체회전 + 로컬진동 시각화 완료 -> {plot_path}")


if __name__ == "__main__":
    main()
