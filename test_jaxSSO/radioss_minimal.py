# -*- coding: utf-8 -*-
import os
import subprocess
from pathlib import Path

def run_simple_test():
    job_name = "minimal"
    rad_content = [
        "#RADIOSS STARTER",
        "/BEGIN",
        f"{job_name:70}",
        f"{2025:10}{0:10}",
        f"{'Mg':20}{'mm':20}{'s':20}",
        f"{'Mg':20}{'mm':20}{'s':20}",
        "/MAT/LAW1/1",
        "Steel",
        " 7.800E-09 2.100E+05     0.300",
        "         0",
        "/PROP/SHELL/1",
        "TrayProp",
        "       2.0",
        "       0.0         0         5",
        "/PART/1",
        "TrayPart",
        "         1         1",
        "/NODE",
        "         1                 0.0                 0.0                 0.0",
        "         2               100.0                 0.0                 0.0",
        "         3               100.0               100.0                 0.0",
        "         4                 0.0               100.0                 0.0",
        "/SHELL/1",
        "         1         1         2         3         4",
        "/RUN/minimal/1",
        "0.01",
        "/END"
    ]
    
    with open("minimal_0000.rad", "w", encoding='utf-8') as f:
        f.write("\n".join(rad_content) + "\n")
    
    exec_root = r"D:\OpenRadioss_win64\OpenRadioss\exec"
    base_path = Path(exec_root).parent
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([
        str(exec_root),
        str(base_path / "extlib" / "hm_reader" / "win64"),
        str(base_path / "extlib" / "intelOneAPI_runtime" / "win64"),
        env.get("PATH", "")
    ])
    env["RAD_CFG_PATH"] = str(base_path / "hm_cfg_files")
    env["RAD_H3D_PATH"] = str(base_path / "extlib" / "h3d" / "lib" / "win64")
    
    print(" -> Testing Minimal Deck...")
    res = subprocess.run([str(Path(exec_root)/"starter_win64.exe"), "-i", "minimal_0000.rad"], capture_output=True, text=True, env=env)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)

if __name__ == "__main__":
    run_simple_test()
