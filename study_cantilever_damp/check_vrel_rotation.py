# -*- coding: utf-8 -*-
"""
/DAMP/VREL이 강체 회전(각속도)까지 감쇠시키는지 확인하는 진단 스크립트.
중력/충격 없이 전체 조립체(고정영역 RBODY + 자유단 탄성영역)에 초기 각속도만
부여하고, RbodyID=0(그룹 평균속도 기준)/1(RBODY 기준), cdamp 크기(2.0/20.0)를
바꿔가며 마스터 노드의 회전각속도(VRY)가 시간에 따라 감쇠되는지 관찰한다.
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
)

TSTOP = 4.0
OMEGA0 = 1.0  # rad/s, Y축 회전
DT_MIN_CST = 2.5e-5
DT_HIS = 0.005


def write_rotation_starter(fp, nd, qd, fix_nids, free_nids, cog_x, cog_y, use_damping, cdamp, rbody_id, skew_id=0):
    all_nids = sorted(nd.keys())
    with open(fp, "w", encoding="utf-8") as f:
        w = f.write
        w("#RADIOSS STARTER\n/BEGIN\ncantilever\n")
        w(f"{2026:>10}{0:>10}\n")
        w(f"{'Mg':<20}{'mm':<20}{'s':<20}\n")
        w(f"{'Mg':<20}{'mm':<20}{'s':<20}\n")

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

        # 초기 각속도 부여: RBODY의 secondary(고정) 노드에 개별 INIVEL을 줘도 무시되고
        # 오직 마스터 노드(9999)의 상태만으로 강체 운동이 결정된다. 따라서 마스터에는
        # 회전각속도(Vyr=OMEGA0)를 직접 부여하고, RBODY가 아닌 자유단 노드들만
        # Vz = -omega*(x-cog_x) 개별 병진속도로 동일한 회전을 부여한다 (SKEW 불필요).
        # /INIVEL/NODE: node_ID(10) skew_ID(10) Vxt(20) Vyt(20) Vzt(20) / Vxr(20) Vyr(20) Vzr(20)
        w(f"/INIVEL/NODE/1\ninit_rotation\n")
        w(ci(9999) + ci(0) + cf(0.0) + cf(0.0) + cf(0.0) + "\n")
        w(" " * 20 + cf(0.0) + cf(OMEGA0) + cf(0.0) + "\n")
        for nid in free_nids:
            x = nd[nid][0]
            vz = -OMEGA0 * (x - cog_x)
            w(ci(nid) + ci(0) + cf(0.0) + cf(0.0) + cf(vz) + "\n")
            w(" " * 20 + cf(0.0) + cf(0.0) + cf(0.0) + "\n")

        w("/TH/NODE/2\nmaster_rot_th\n")
        w(f"{'VRY':<10}\n")
        w(ci(9999) + ci(0) + "\n")

        # 회전축(cog_x)에서 먼 고정영역 코너 노드(node 1, x=0)의 실제 Z속도 이력 추적
        w("/TH/NODE/3\ncorner_th\n")
        w(f"{'DEF':<10}\n")
        w(ci(1) + ci(0) + "\n")

        # skew_id!=0 인 경우, 회전축 위 무게중심(cog_x,cog_y,0)에 원점을 둔 global-aligned
        # skew를 정의해서 /DAMP/VREL의 skew_id로 참조 (VREL 자체의 skew_id 영향 테스트용)
        if skew_id != 0:
            w(f"/SKEW/FIX/{skew_id}\ncog_skew\n")
            w(cf(cog_x) + cf(cog_y) + cf(0.0) + "\n")
            w(cf(0.0) + cf(1.0) + cf(0.0) + "\n")
            w(cf(0.0) + cf(0.0) + cf(1.0) + "\n")

        if use_damping:
            w("/DAMP/VREL/1\ndamp_vrel_rot\n")
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
    subprocess.run([str(th_to_csv_exe), str(t01)], cwd=str(rad_dir), capture_output=True, env=env)
    return Path(str(t01) + ".csv")


def main():
    out_dir = Path(__file__).parent
    test_dir = out_dir / "cases_vrel_rotation_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    nd, qd, fix_nids, free_nids, tip_nid = create_plate_mesh()

    # 조립체 전체(고정+자유) 무게중심 X좌표 계산
    xs = np.array([nd[n][0] for n in nd])
    ys = np.array([nd[n][1] for n in nd])
    cog_x, cog_y = float(xs.mean()), float(ys.mean())
    print(f"System COG estimate: X={cog_x:.1f}, Y={cog_y:.1f}")

    cases = {
        "baseline_nodamp": dict(use_damping=False, cdamp=0.0, rbody_id=1, skew_id=0),
        "vrel_rbody1_cdamp2": dict(use_damping=True, cdamp=2.0, rbody_id=1, skew_id=0),
        "vrel_rbody0_cdamp2": dict(use_damping=True, cdamp=2.0, rbody_id=0, skew_id=0),
        "vrel_rbody1_cdamp20": dict(use_damping=True, cdamp=20.0, rbody_id=1, skew_id=0),
        "vrel_rbody1_skew1_cdamp2": dict(use_damping=True, cdamp=2.0, rbody_id=1, skew_id=1),
    }

    results = {}
    for name, params in cases.items():
        case_path = test_dir / name
        case_path.mkdir(parents=True, exist_ok=True)
        sp = str(case_path / "cantilever_0000.rad")
        ep = str(case_path / "cantilever_0001.rad")
        write_rotation_starter(sp, nd, qd, fix_nids, free_nids, cog_x, cog_y, **params)
        write_engine(ep)
        print(f"\n[{name}] 실행 중...")
        csv_file = run(case_path, "cantilever")
        if not csv_file.exists():
            print(f"[{name}] CSV 없음 - 수동 변환 시도")
            t01 = case_path / "cantileverT01"
            if t01.exists():
                _, exec_dir, env = get_openradioss_paths()
                subprocess.run(
                    [str(exec_dir / "th_to_csv_win64.exe"), str(t01)],
                    cwd=str(case_path), capture_output=True, env=env,
                )
                csv_file = Path(str(t01) + ".csv")
        if not csv_file.exists():
            print(f"[{name}] CSV 변환 실패, 스킵")
            continue
        df = pd.read_csv(csv_file)
        corner_cols = [c for c in df.columns if "corner_th" in c]
        t = df["time"].values
        vz = df[corner_cols[5]].values  # DEF: DX,DY,DZ,VX,VY,VZ -> index5=VZ
        results[name] = (t, vz)
        print(f"[{name}] corner VZ: t=0 -> {vz[0]:.2f}, t={t[-1]:.2f}s -> {vz[-1]:.2f} mm/s")

    print("\n=== 요약 (코너 노드 Z속도, 이론 초기값 900mm/s) ===")
    for name, (t, vz) in results.items():
        print(f"{name:28s}: VZ(0)={vz[0]:.2f}  VZ(T={t[-1]:.2f}s)={vz[-1]:.2f}")

    # 비교 그래프 생성
    if results:
        import matplotlib.pyplot as plt
        try:
            import koreanize_matplotlib  # noqa: F401
        except ImportError:
            pass

        plt.rcParams.update({"font.size": 9})
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = {
            "baseline_nodamp": "Undamped baseline",
            "vrel_rbody1_cdamp2": "VREL RbodyID=1, cdamp=2 (skew=0)",
            "vrel_rbody0_cdamp2": "VREL RbodyID=0 (group avg), cdamp=2",
            "vrel_rbody1_cdamp20": "VREL RbodyID=1, cdamp=20 (10x)",
            "vrel_rbody1_skew1_cdamp2": "VREL RbodyID=1, skew=COG-skew, cdamp=2",
        }
        colors = {
            "baseline_nodamp": "black",
            "vrel_rbody1_cdamp2": "#d62728",
            "vrel_rbody0_cdamp2": "#1f77b4",
            "vrel_rbody1_cdamp20": "#ff7f0e",
            "vrel_rbody1_skew1_cdamp2": "#2ca02c",
        }
        styles = {
            "baseline_nodamp": "--",
        }
        for name, (t, vz) in results.items():
            ax.plot(
                t, vz,
                linestyle=styles.get(name, "-"),
                label=labels.get(name, name),
                color=colors.get(name, None),
                linewidth=1.6 if name == "baseline_nodamp" else 1.3,
                alpha=0.9,
            )
        ax.set_title(
            "순수 강체 회전(omega0=1 rad/s) 시 /DAMP/VREL 설정별 코너노드 Z속도 비교\n"
            "(RbodyID/skew_id에 따라 회전이 감쇠되는지 확인)",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Corner Node (X=0) Z-Velocity [mm/s]")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=8, frameon=True)
        plt.tight_layout()
        plot_path = test_dir / "vrel_rotation_comparison.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"\n[OK] 비교 그래프 저장 완료 -> {plot_path}")


if __name__ == "__main__":
    main()
