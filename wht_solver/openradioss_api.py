import os
import re
import subprocess
import tempfile
import numpy as np
from typing import Dict, Any

class OpenRadiossAPI:
    """
    API for interacting with OpenRadioss solver for Implicit Linear Modal Analysis.
    Generates _0000.rad and _0001.rad files, and parses job_0001.out.
    """
    STARTER_EXEC = r"D:\OpenRadioss_win64\OpenRadioss\exec\starter_win64.exe"
    ENGINE_EXEC  = r"D:\OpenRadioss_win64\OpenRadioss\exec\engine_win64.exe"

    def __init__(self, model, starter_path: str = None, engine_path: str = None):
        self.model = model
        self.starter_path = starter_path or self.STARTER_EXEC
        self.engine_path = engine_path or self.ENGINE_EXEC

    def run_modal(self, num_modes: int = 26) -> Dict[str, Any]:
        """
        Runs OpenRadioss modal analysis.
        Returns a dictionary with parsed frequencies.
        """
        openradioss_path = os.path.dirname(os.path.dirname(self.starter_path))
        custom_env = os.environ.copy()
        
        additional_paths_win = [
            os.path.join(openradioss_path, "extlib", "hm_reader", "win64"),
            os.path.join(openradioss_path, "extlib", "intelOneAPI_runtime", "win64")
        ]
        
        custom_env["OPENRADIOSS_PATH"] = openradioss_path
        custom_env["RAD_CFG_PATH"] = os.path.join(openradioss_path, "hm_cfg_files")
        custom_env["RAD_H3D_PATH"] = os.path.join(openradioss_path, "extlib", "h3d", "lib", "win64")
        custom_env["PATH"] = os.pathsep.join(additional_paths_win + [custom_env.get("PATH", "")])

        with tempfile.TemporaryDirectory() as tmpdir:
            starter_file = os.path.join(tmpdir, "job_0000.rad")
            engine_file  = os.path.join(tmpdir, "job_0001.rad")
            
            self._write_starter(starter_file, num_modes)
            self._write_engine(engine_file)
            
            # Run Starter
            try:
                subprocess.run([self.starter_path, "-i", "job_0000.rad"], env=custom_env, cwd=tmpdir, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"OpenRadioss Starter failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
                raise RuntimeError("OpenRadioss Starter failed.") from e
                
            # Run Engine
            try:
                subprocess.run([self.engine_path, "-i", "job_0001.rad"], env=custom_env, cwd=tmpdir, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"OpenRadioss Engine failed:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
                raise RuntimeError("OpenRadioss Engine failed.") from e
            
            out_file = os.path.join(tmpdir, "job_0001.out")
            if not os.path.exists(out_file):
                raise FileNotFoundError(f"OpenRadioss output file not found: {out_file}")
            
            return self._parse_out(out_file, num_modes)

    def _write_starter(self, filepath: str, num_modes: int):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("#RADIOSS STARTER\n")
            f.write("/BEGIN\n")
            f.write("job\n")
            f.write("      2026         0\n")
            # Unit scale factors (numerical): mass [kg] length [mm] time [ms]
            # kg=1.0, mm=1e-3 (m), ms=1e-3 (s) relative to SI -> use 1.0 if input data already in kg/mm/ms
            f.write("                  1.0                 1.0                 1.0\n")
            f.write("                  1.0                 1.0                 1.0\n")
            
            # Write MAT
            mat_written = set()
            for pid, prop in self.model.properties.items():
                if prop.type in ('PSHELL', 'SHELL'):
                    if prop.mid not in mat_written:
                        mat = self.model.materials[prop.mid]
                        f.write(f"/MAT/ELAST/{mat.mid}\n")
                        f.write(f"{mat.name if hasattr(mat, 'name') else 'MAT_'+str(mat.mid)}\n")
                        # Density (kg/mm^3 in kg-mm-ms system): e.g. 7.85e-9
                        f.write(f"{mat.rho:20.6E}\n")
                        # E [MPa] and nu
                        f.write(f"{mat.E:20.4f}{mat.nu:20.6f}\n")
                        mat_written.add(prop.mid)
            
            # Write PROP/TYPE1 (OpenRadioss shell property format)
            for pid, prop in self.model.properties.items():
                if prop.type in ('PSHELL', 'SHELL'):
                    f.write(f"/PROP/TYPE1/{pid}\n")
                    f.write(f"PROP_{pid}\n")
                    # Line 3: Ishell Ismstr Ish3n Idrill P_thick_fail
                    f.write(f"{1:10d}{0:10d}{0:10d}{0:10d}{0:10d}\n")
                    # Line 4: hm hf hr dm dn (hourglass damping)
                    f.write(f"{0.01:10.4f}{0.01:10.4f}{0.01:10.4f}{0.01:10.4f}{0.01:10.4f}\n")
                    # Line 5: Npt Istrain Thick Ashear Ithick
                    f.write(f"{5:10d}{0:10d}{prop.t:20.6f}{0.8333333:20.4f}{0:10d}\n")
                    
            # Write PART (each part references prop + mat via /PART/part_id)
            for pid, prop in self.model.properties.items():
                if prop.type in ('PSHELL', 'SHELL'):
                    f.write(f"/PART/{pid}\n")
                    f.write(f"PART_{pid}\n")
                    # prop_id   mat_id
                    f.write(f"{pid:10d}{prop.mid:10d}\n")
                    
            # Write NODE
            f.write("/NODE\n")
            for nid, node in self.model.nodes.items():
                f.write(f"{nid:10d}{node.x:20.6f}{node.y:20.6f}{node.z:20.6f}\n")
                
            # Write SHELL: /SHELL/part_id 형식으로 파트별 그룹화
            # 포맷: el_id  n1  n2  n3  n4
            from collections import defaultdict
            shells_by_part = defaultdict(list)
            for eid, elem in self.model.elements.items():
                etype = getattr(elem, 'type', '')
                if etype in ('QUAD4', 'QUAD', 'TRIA3', 'TRIA'):
                    shells_by_part[elem.pid].append((eid, elem))
            
            for pid, elems in shells_by_part.items():
                f.write(f"/SHELL/{pid}\n")
                for eid, elem in elems:
                    n = elem.node_ids
                    etype = getattr(elem, 'type', '')
                    if etype in ('QUAD4', 'QUAD'):
                        f.write(f"{eid:10d}{n[0]:10d}{n[1]:10d}{n[2]:10d}{n[3]:10d}\n")
                    elif etype in ('TRIA3', 'TRIA'):
                        # Tri: n3==n4로 복제
                        f.write(f"{eid:10d}{n[0]:10d}{n[1]:10d}{n[2]:10d}{n[2]:10d}\n")
            
            # node_id가 int 또는 list일 수 있으므로 flatten 처리
            spc_node_set = set()
            for spc in self.model.spc_conditions:
                nid = spc.node_id
                if isinstance(nid, (list, tuple)):
                    spc_node_set.update(nid)
                else:
                    spc_node_set.add(nid)
            spc_nodes = list(spc_node_set)
            
            if spc_nodes:
                f.write("/GRNOD/NODE/1\n")
                f.write("BC_NODES\n")
                for i in range(0, len(spc_nodes), 10):
                    chunk = spc_nodes[i:i+10]
                    f.write(" ".join(f"{nid:10d}" for nid in chunk) + "\n")
                
                # /BCS 포맷: title 라인, 그 다음 "Tra Rot  skew_id  grnd_id" 한 줄
                # 예제: "   111 111         0         2"
                f.write("/BCS/1\n")
                f.write("FIXED_BC\n")
                f.write("   111 111         0         1\n")
            
            f.write("/EIG/1\n")
            f.write("MODAL ANALYSIS\n")
            f.write("         0         0         0         0\n")
            f.write(f"{num_modes:10d}         0       0.0       0.0\n")
            f.write("         0         0       300         0       0.0\n")
            
            f.write("/END\n")

    def _write_engine(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("#RADIOSS ENGINE\n")
            f.write("/RUN/job/1/\n")
            f.write("1.0\n")
            f.write("/IMPL/LINEAR\n")
            f.write("/IMPL/SOLVER/2\n")
            f.write("5 0 0 0 0.0\n")
            f.write("/END\n")

    def _parse_out(self, filepath: str, num_modes: int) -> Dict[str, Any]:
        freqs = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if "FREQUENCY OF THE MODE" in line:
                # E.g. "  FREQUENCY OF THE MODE  1 =   123.456  Hz"
                match = re.search(r'=\s+([\d.E+-]+)\s+Hz', line)
                if match:
                    freqs.append(float(match.group(1)))
                    
        return {
            'frequencies': np.array(freqs[:num_modes])
        }
