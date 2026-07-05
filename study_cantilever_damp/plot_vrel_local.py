# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_freefall_damp import analyze_vibration

try:
    import koreanize_matplotlib  # noqa: F401
except ImportError:
    pass

test_dir = Path(__file__).parent / "cases_vrel_local_test"
tip_nid = 913

cases = {
    "undamped": "Undamped",
    "vrel_cdamp2": "VREL RbodyID=1, cdamp=2",
    "vrel_cdamp20": "VREL RbodyID=1, cdamp=20 (10x)",
}
colors = {"undamped": "#d62728", "vrel_cdamp2": "#1f77b4", "vrel_cdamp20": "#2ca02c"}

results = {}
for name in cases:
    csv_path = test_dir / name / "cantileverT01.csv"
    if not csv_path.exists():
        print(f"[skip] {name}")
        continue
    df = pd.read_csv(csv_path)
    tip_cols = [c for c in df.columns if str(tip_nid) in c]
    master_cols = [c for c in df.columns if "9999" in c]
    t = df["time"].values
    rel = df[tip_cols[2]].values - df[master_cols[2]].values
    f_num, period_mean, zeta, f_fft, valleys, peaks = analyze_vibration(t, rel)
    results[name] = (t, rel, f_num, zeta)
    print(f"{name:15s}: f={f_num:.3f} Hz  zeta={zeta*100:.3f} %  peak(first5)={[round(rel[p],1) for p in peaks[:5]]}")

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(10, 6))
for name, (t, rel, f_num, zeta) in results.items():
    ax.plot(t, rel, label=f"{cases[name]} (zeta={zeta*100:.2f}%)", color=colors.get(name), linewidth=1.3, alpha=0.9)
ax.set_title(
    "순net힘=0 / 순net모멘트=0 하중으로 여기한 순수 국소 굽힘 진동\n(/DAMP/VREL RbodyID=1 감쇠 효과 확인)",
    fontsize=11, fontweight="bold",
)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Relative Z-Displacement (tip - master) [mm]")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="best", fontsize=9)
plt.tight_layout()
plot_path = test_dir / "vrel_local_vibration_comparison.png"
fig.savefig(plot_path, dpi=300)
plt.close(fig)
print(f"\n[OK] 저장 -> {plot_path}")
