# -*- coding: utf-8 -*-
"""
/DAMP/VREL의 "Freq" 필드가 실제 감쇠력 계산에 쓰이는지(같은 Alpha_z에서 Freq만
바꿔서 결과가 달라지는지) 확인하기 위한 진단 스크립트.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_freefall_damp import (
    create_plate_mesh,
    write_starter,
    write_engine,
    run_openradioss,
    analyze_vibration,
)
import pandas as pd

out_dir = Path(__file__).parent
test_dir = out_dir / "cases_vrel_freq_test"
test_dir.mkdir(parents=True, exist_ok=True)

nd, qd, fix_nids, free_nids, tip_nid = create_plate_mesh()
print(f"Mesh: {len(nd)} nodes, {len(qd)} elements.")

# RbodyID=0(노드그룹 평균속도 기준) vs RbodyID=1(RBODY 기준) 비교, cdamp=2.0으로 강하게
cases = {
    "rbody0_cdamp2": 0,
    "rbody1_cdamp2": 1,
}

for case_name, rbody_val in cases.items():
    case_path = test_dir / case_name
    case_path.mkdir(parents=True, exist_ok=True)
    sp = str(case_path / "cantilever_0000.rad")
    ep = str(case_path / "cantilever_0001.rad")
    write_starter(
        sp, nd, qd, fix_nids, free_nids, tip_nid,
        use_damping=True, cdamp=2.0, damp_freq=1.3, alpha_is_raw=True,
        rbody_id=rbody_val,
    )
    write_engine(ep)
    print(f"\n[Case: {case_name}] (Freq=1.3, cdamp=2.0, RbodyID={rbody_val}) 실행 중...")
    csv_file = run_openradioss(case_path, "cantilever")

    if not csv_file.exists():
        print(f"[Case: {case_name}] CSV 없음 - 실행 실패")
        continue

    df = pd.read_csv(csv_file)
    tip_cols = [c for c in df.columns if str(tip_nid) in c]
    master_cols = [c for c in df.columns if "9999" in c]
    time_val = df["time"].values
    disp_tip = df[tip_cols[2]].values
    disp_master = df[master_cols[2]].values
    relative_disp = disp_tip - disp_master

    f_num, period_mean, zeta, f_fft, valleys, peaks = analyze_vibration(time_val, relative_disp)
    print(
        f" -> [{case_name}] f={f_num:.4f} Hz, zeta={zeta*100:.3f} %, "
        f"peak amplitudes: {[round(relative_disp[p], 1) for p in peaks[:6]]}"
    )
