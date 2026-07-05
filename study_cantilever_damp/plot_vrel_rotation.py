# -*- coding: utf-8 -*-
"""이미 실행된 cases_vrel_rotation_test 결과들을 읽어 비교 그래프만 생성."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pass

test_dir = Path(__file__).parent / "cases_vrel_rotation_test"

cases = {
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

results = {}
for name in cases:
    csv_path = test_dir / name / "cantileverT01.csv"
    if not csv_path.exists():
        print(f"[skip] {name}: csv 없음")
        continue
    df = pd.read_csv(csv_path)
    corner_cols = [c for c in df.columns if "corner_th" in c]
    t = df["time"].values
    vz = df[corner_cols[5]].values
    results[name] = (t, vz)
    print(f"{name:28s}: VZ(0)={vz[0]:.2f}  VZ(T={t[-1]:.2f}s)={vz[-1]:.2f}")

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(10, 6))
for name, (t, vz) in results.items():
    ax.plot(
        t, vz,
        linestyle="--" if name == "baseline_nodamp" else "-",
        label=cases.get(name, name),
        color=colors.get(name),
        linewidth=1.8 if name == "baseline_nodamp" else 1.3,
        alpha=0.9,
    )
ax.set_title(
    "순수 강체 회전(omega0=1 rad/s) 시 /DAMP/VREL 설정별 코너노드 Z속도 비교\n"
    "(RbodyID / skew_id 설정에 따라 회전이 감쇠되는지 확인)",
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
print(f"\n[OK] 저장 완료 -> {plot_path}")
