# -*- coding: utf-8 -*-
import os
import subprocess
import time
from pathlib import Path
from mesh_utils import generate_shell_tray, get_nodes_in_box

def main():
    jobname = "challenge_dyna"
    base_path = r"D:\OpenRadioss_win64\OpenRadioss"
    exec_root = Path(base_path) / "exec"
    env = os.environ.copy()
    env["RAD_CFG_PATH"] = str(Path(base_path) / "hm_cfg_files")
    env["RAD_H3D_PATH"] = str(Path(base_path) / "extlib" / "h3d" / "lib" / "win64")
    lib_path = str(Path(base_path) / "extlib" / "intelOneAPI_runtime" / "win64")
    env["PATH"] = os.pathsep.join([str(exec_root), lib_path, env.get("PATH", "")])

    # 6. Automatic VTK Conversion (Python Redirection)
    anim_to_vtk = str(exec_root / "anim_to_vtk_win64.exe")
    anim_files = sorted(list(Path(".").glob(f"{jobname}A*")))
    anim_files = [f for f in anim_files if f.suffix == ""]

    if anim_files:
        print(f" -> Converting {len(anim_files)} Animation files to VTK (STDOUT Redirection)...")
        for anim in anim_files:
            vtk_file = f"{anim}.vtk"
            # Force redirection from STDOUT to file
            with open(vtk_file, 'w', encoding='utf-8') as f:
                subprocess.run([anim_to_vtk, str(anim)], env=env, stdout=f)
        print(" [SUCCESS] All frames successfully redirected and saved as .vtk files.")
    else:
        print(" [!] No animation files found. Run the full solver first.")

if __name__ == "__main__":
    main()
