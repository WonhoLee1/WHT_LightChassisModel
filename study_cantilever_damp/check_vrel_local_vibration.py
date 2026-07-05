# -*- coding: utf-8 -*-
"""
자유낙하+RBODY 모델에서, 시스템 전체에 순net 힘=0 / 순net 모멘트=0이 되도록 자유단에
분산 하중을 가해 "회전은 전혀 여기하지 않고 순수 국소 굽힘 진동만" 일으킨 뒤,
/DAMP/VREL(RbodyID=1)이 이 국소 진동을 실제로 감쇠시키는지 확인한다.
"""
import math
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
    GRAVITY,
)

TSTOP = 8.0
DT_MIN_CST = 2.5e-5
DT_HIS = 0.005
FORCE_SCALE = 500.0  # 최대 노드 힘 [N]


def compute_balanced_shape(nd, free_nids, cog_x):
    x_root = min(nd[n][0] for n in free_nids)
    xs = sorted(set(nd[n][0] for n in free_nids))
    n_col = len(xs)
    base = np.array([(x - x_root) ** 2 for x in xs])
    lin = np.array([x - cog_x for x in xs])
    A = np.array([[n_col, lin.sum()], [lin.sum(), (lin ** 2).sum()]])
    rhs = -np.array([base.sum(), (base * lin).sum()])
    a, b = np.linalg.solve(A, rhs)
    g = base + a + b * lin
    g = g / np.max(np.abs(g)) * FORCE_SCALE
    return dict(zip(xs, g)), x_root


def write_starter(fp, nd, qd, fix_nids, free_nids, tip_nid, cog_x, use_damping, cdamp, rbody_id):
    all_nids = sorted(nd.keys())
    shape, x_root = compute_balanced_shape(nd, free_nids, cog_x)
    xs = sorted(shape.keys())
    col_nodes = {x: [n for n in free_nids if nd[n][0] == x] for x in xs}

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
        w(ci(9999) + cf(650.0) + cf(600.0) + cf(0.0) + "\n")

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

        # 노드 그룹 9 (강체 마스터 노드 9999 고정용)
        w("/GRNOD/NODE/9\nmaster_node_group\n")
        w(ci(9999) + "\n")

        # 자유단 x-column별 GRNOD (10, 20, 30, ...)
        col_ids = {}
        gid = 10
        for x in xs:
            col_ids[x] = gid
            w(f"/GRNOD/NODE/{gid}\ncol_x{int(x)}\n")
            nodes = col_nodes[x]
            for i in range(0, len(nodes), 8):
                w("".join(ci(n) for n in nodes[i : i + 8]) + "\n")
            gid += 1

        w("/GRPART/PART/1\nplate_part_group\n")
        w(ci(1) + "\n")

        w("/RBODY/1\nbase_rigid_body\n")
        w(ci(9999) + ci(0) + ci(0) + ci(0) + cf(0.0) + ci(2) + ci(0) + ci(3) + ci(0) + "\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + "\n")
        w(cf(0.0) + cf(0.0) + cf(0.0) + "\n")
        w(ci(0) + "\n")

        # 강체 마스터 노드 경계 조건 (/BCS) - Z방향 병진 낙하만 허용
        w("/BCS/1\nmaster_constraints\n")
        w(f"# {'grnod_ID':>8}{'skew_ID':>10}{'TRA':>10}{'ROT':>10}\n")
        w(f"{'110111':>10}" + ci(0) + ci(9) + "\n")

        w("/GRAV/1\ngravity_load\n")
        w(ci(0) + f"{'Z':>10}" + ci(0) + ci(0) + ci(1) + " " * 10 + cf(1.0) + cf(-GRAVITY) + "\n")

        # 순net힘=0, 순net모멘트=0 이 되도록 계산된 컬럼별 임펄스 하중 (0~0.05s)
        w("/FUNCT/1\nimpact_curve\n")
        w(cf(0.0) + cf(1.0) + "\n")
        w(cf(0.05) + cf(1.0) + "\n")
        w(cf(0.051) + cf(0.0) + "\n")
        w(cf(TSTOP) + cf(0.0) + "\n")

        cload_id = 1
        for x in xs:
            w(f"/CLOAD/{cload_id}\ncol_load_x{int(x)}\n")
            w(
                ci(1)
                + f"{'Z':>10}"
                + ci(0)
                + ci(0)
                + ci(col_ids[x])
                + ci(1)
                + cf(1.0)
                + cf(shape[x])
                + "\n"
            )
            cload_id += 1

        w("/TH/NODE/1\ntip_th\n")
        w(f"{'DEF':<10}\n")
        w(ci(tip_nid) + ci(0) + "\n")

        w("/TH/NODE/2\nmaster_th\n")
        w(f"{'DEF':<10}\n")
        w(ci(9999) + ci(0) + "\n")

        if use_damping:
            w("/DAMP/VREL/1\ndamp_vrel_local\n")
            w(cf(0.0) + cf(0.0) + ci(4) + ci(0) + cf(0.0) + cf(TSTOP) + "\n")
            w(cf(1.3) + ci(rbody_id) + ci(0) + cf(1.0) + "\n")
            w(cf(0.0) + cf(0.0) + "\n")
            w(cf(cdamp) + cf(0.0) + "\n")

        w("/END\n")

    return shape


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
    test_dir = out_dir / "cases_vrel_local_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    nd, qd, fix_nids, free_nids, tip_nid = create_plate_mesh()
    xs_all = np.array([nd[n][0] for n in nd])
    cog_x = float(xs_all.mean())
    print(f"System COG X = {cog_x:.1f}, tip_nid={tip_nid}")

    cases = {
        "undamped": dict(use_damping=False, cdamp=0.0, rbody_id=1),
        "vrel_cdamp100": dict(use_damping=True, cdamp=100.0, rbody_id=1),
        "vrel_cdamp1000": dict(use_damping=True, cdamp=1000.0, rbody_id=1),
        "vrel_cdamp10000": dict(use_damping=True, cdamp=10000.0, rbody_id=1),
        "vrel_cdamp100000": dict(use_damping=True, cdamp=100000.0, rbody_id=1),
    }

    results = {}
    for name, params in cases.items():
        case_path = test_dir / name
        case_path.mkdir(parents=True, exist_ok=True)
        sp = str(case_path / "cantilever_0000.rad")
        ep = str(case_path / "cantilever_0001.rad")
        shape = write_starter(sp, nd, qd, fix_nids, free_nids, tip_nid, cog_x, **params)
        write_engine(ep)
        print(f"\n[{name}] 실행 중... (net force={sum(shape.values()):.4f}, "
              f"net moment={sum(v*(x-cog_x) for x, v in shape.items()):.4f})")
        csv_file = run(case_path, "cantilever")
        if not csv_file.exists():
            t01 = case_path / "cantileverT01"
            if t01.exists():
                _, exec_dir, env = get_openradioss_paths()
                subprocess.run(
                    [str(exec_dir / "th_to_csv_win64.exe"), t01.name],
                    cwd=str(case_path), capture_output=True, env=env,
                )
                csv_file = Path(str(t01) + ".csv")
        if not csv_file.exists():
            print(f"[{name}] CSV 없음, 스킵")
            continue

        df = pd.read_csv(csv_file)
        tip_cols = [c for c in df.columns if str(tip_nid) in c]
        master_cols = [c for c in df.columns if "9999" in c]
        t = df["time"].values
        disp_tip = df[tip_cols[2]].values
        disp_master = df[master_cols[2]].values
        relative_disp = disp_tip - disp_master

        f_num, period_mean, zeta, f_fft, valleys, peaks = analyze_vibration(t, relative_disp)
        results[name] = (t, relative_disp, f_num, zeta)
        print(f"[{name}] f={f_num:.4f} Hz, zeta={zeta*100:.3f} %, "
              f"peak amps(first6): {[round(relative_disp[p],1) for p in peaks[:6]]}")

    print("\n=== 요약 ===")
    for name, (t, rel, f_num, zeta) in results.items():
        print(f"{name:15s}: f={f_num:.3f} Hz  zeta={zeta*100:.3f} %")

    if results:
        import matplotlib.pyplot as plt
        try:
            import koreanize_matplotlib  # noqa: F401
        except ImportError:
            pass
        plt.rcParams.update({"font.size": 9})
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {
            "undamped": "#d62728",
            "vrel_cdamp2": "#1f77b4",
            "vrel_cdamp20": "#2ca02c",
            "vrel_cdamp100": "#ff7f0e",
        }
        for name, (t, rel, f_num, zeta) in results.items():
            ax.plot(t, rel, label=f"{name} (zeta={zeta*100:.2f}%)", color=colors.get(name), linewidth=1.3, alpha=0.9)
        ax.set_title("순net힘=0/순net모멘트=0 하중으로 여기한 순수 국소 굽힘 진동\n(/DAMP/VREL RbodyID=1 감쇠 효과)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Relative Z-Displacement (tip - master) [mm]")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plot_path = test_dir / "vrel_local_vibration_comparison.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"\n[OK] 그래프 저장 -> {plot_path}")


if __name__ == "__main__":
    main()
