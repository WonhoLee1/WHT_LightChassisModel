import os
import re
import subprocess
import numpy as np
from typing import Dict, Any


class MystranAPI:
    """
    MYSTRAN solver wrapper.

    결과 파싱 전략 (순수 f06 텍스트 파싱):
      - 모달 주파수  : R E A L   E I G E N V A L U E S 섹션
      - 정해석 변위  : D I S P L A C E M E N T S 섹션
      - 정해석 응력  : CENTER / GRD 행의 von Mises 열 (index 9)
    """
    MYSTRAN_EXEC = r"D:\SOFTWARE\MYSTRAN\mystran-18.0.0-windows-x86_64.exe"

    def __init__(self, model, exec_path: str = None):
        self.model = model
        self.exec_path = exec_path or self.MYSTRAN_EXEC

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_analysis(self, analysis_type: str = 'modal',
                     num_modes: int = 10, load_case=None) -> Dict[str, Any]:
        lc_suffix = f"_{load_case.name}" if load_case else ""
        tmpdir = os.path.join(
            r"d:\PythonCodeStudy\WHT_LightChassisModel\test_jaxSSO",
            f"tmp_mystran_{analysis_type}{lc_suffix}"
        )
        os.makedirs(tmpdir, exist_ok=True)
        input_file = os.path.join(tmpdir, "job.dat")
        self._write_dat(input_file, analysis_type, num_modes, load_case)

        try:
            subprocess.run(
                [self.exec_path, input_file],
                cwd=tmpdir, check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"MYSTRAN 실행 실패:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            raise RuntimeError("MYSTRAN solve failed.") from e

        f06_file = os.path.join(tmpdir, "job.f06")
        if not os.path.exists(f06_file):
            # MYSTRAN은 대문자 확장자로 생성하는 경우도 있음
            f06_file = os.path.join(tmpdir, "job.F06")
        if not os.path.exists(f06_file):
            raise FileNotFoundError(f"F06 not found in: {tmpdir}")

        if analysis_type == 'modal':
            result = self._parse_modal(f06_file, num_modes)
        else:
            result = self._parse_static(f06_file)

        err_file = os.path.join(tmpdir, "job.ERR")
        if os.path.exists(err_file):
            try:
                os.remove(err_file)
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # DAT 파일 작성
    # ------------------------------------------------------------------

    def _write_dat(self, filepath: str, analysis_type: str,
                   num_modes: int, load_case):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("SOL 103\n" if analysis_type == 'modal' else "SOL 101\n")
            f.write("CEND\n")
            f.write("TITLE = MYSTRAN ANALYSIS\n")
            f.write("ECHO = NONE\n")
            has_spc = (
                (load_case and len(load_case.bcs) > 0) or
                (self.model.spc_conditions and len(self.model.spc_conditions) > 0)
            )
            if has_spc:
                f.write("SPC = 1\n")
            if analysis_type == 'modal':
                f.write("METHOD = 1\n")
                f.write("DISPLACEMENT(PRINT) = ALL\n")
            else:
                f.write("LOAD = 1\n")
                f.write("DISPLACEMENT(PRINT) = ALL\n")
                f.write("STRESS(PRINT) = ALL\n")
            f.write("BEGIN BULK\n")
            f.write("PARAM,AUTOSPC,YES\n")
            f.write("PARAM,QUAD4TYP,MITC4+\n")

            for nid, node in self.model.nodes.items():
                f.write(f"GRID,{nid},,{node.x:.6f},{node.y:.6f},{node.z:.6f}\n")

            for eid, elem in self.model.elements.items():
                etype = getattr(elem, 'type', '')
                if etype in ('QUAD4', 'QUAD'):
                    n = elem.node_ids
                    f.write(f"CQUAD4,{eid},{elem.pid},{n[0]},{n[1]},{n[2]},{n[3]}\n")
                elif etype in ('TRIA3', 'TRIA'):
                    n = elem.node_ids
                    f.write(f"CTRIA3,{eid},{elem.pid},{n[0]},{n[1]},{n[2]}\n")

            mat_written = set()
            for pid, prop in self.model.properties.items():
                if prop.type in ('PSHELL', 'SHELL'):
                    f.write(f"PSHELL,{pid},{prop.mid},{prop.t:.6f},{prop.mid},,{prop.mid}\n")
                    if prop.mid not in mat_written:
                        mat = self.model.materials[prop.mid]
                        G = mat.E / (2 * (1 + mat.nu))
                        f.write(f"MAT1,{mat.mid},{mat.E:.6f},{G:.6f},{mat.nu:.6f},{mat.rho:.3e}\n")
                        mat_written.add(prop.mid)

            spc_nodes = {}
            if load_case:
                for bc in load_case.bcs:
                    dofs = "".join(str(d + 1) for d in bc.dofs)
                    spc_nodes.setdefault(dofs, []).append(bc.node_id)
            else:
                for spc in self.model.spc_conditions:
                    dofs = "".join(str(d + 1) for d in spc.dofs)
                    spc_nodes.setdefault(dofs, []).append(spc.node_id)
            for dofs, nids in spc_nodes.items():
                for i in range(0, len(nids), 6):
                    chunk = nids[i:i + 6]
                    f.write("SPC1,1," + dofs + "," + ",".join(map(str, chunk)) + "\n")

            if analysis_type == 'static' and load_case:
                for force in load_case.forces:
                    fx, fy, fz = force.load_vector[:3]
                    f.write(f"FORCE,1,{force.node_id},0,1.0,{fx:.6f},{fy:.6f},{fz:.6f}\n")

            if analysis_type == 'modal':
                f.write(f"EIGRL,1,,,{num_modes}\n")

            f.write("ENDDATA\n")

    # ------------------------------------------------------------------
    # 모달 파싱
    # ------------------------------------------------------------------

    def _parse_modal(self, f06_file: str, num_modes: int) -> Dict[str, Any]:
        freqs = self._parse_f06_frequencies(f06_file)
        shapes = self._parse_f06_mode_shapes(f06_file, num_modes)
        return {'frequencies': np.array(freqs[:num_modes]), 'mode_shapes': shapes}

    def _parse_f06_mode_shapes(self, filepath: str, num_modes: int) -> list:
        """f06 모달 DISPLACEMENT 섹션에서 모드형상(T1,T2,T3) 파싱. 모드별 리스트 반환."""
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        shapes = []
        i = 0
        while i < len(lines) and len(shapes) < num_modes:
            if 'GRID' in lines[i] and 'COORD' in lines[i] and 'T1' in lines[i]:
                i += 2  # 헤더 + SYS 행 건너뜀
                mode_disp = {}
                while i < len(lines):
                    row = lines[i].strip()
                    if not row:
                        i += 1
                        continue
                    parts = row.split()
                    if parts and parts[0].isdigit():
                        try:
                            nid = int(parts[0])
                            t1 = float(parts[2])
                            t2 = float(parts[3])
                            t3 = float(parts[4])
                            mode_disp[nid] = [t1, t2, t3]
                        except (ValueError, IndexError):
                            pass
                        i += 1
                    else:
                        break
                if mode_disp:
                    nids_sorted = sorted(mode_disp.keys())
                    vec = np.array([mode_disp[n] for n in nids_sorted])  # (N,3)
                    shapes.append(vec)
            else:
                i += 1
        return shapes

    # ------------------------------------------------------------------
    # 정해석 파싱
    # ------------------------------------------------------------------

    def _parse_static(self, f06_file: str) -> Dict[str, Any]:
        return self._parse_f06_static(f06_file)

    # ------------------------------------------------------------------
    # f06 파싱 헬퍼
    # ------------------------------------------------------------------

    def _parse_f06_frequencies(self, filepath: str):
        """R E A L   E I G E N V A L U E S 섹션에서 Hz(CYCLES) 추출."""
        freqs = []
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if "R E A L   E I G E N V A L U E S" in lines[i]:
                i += 1
                while i < len(lines):
                    stripped = lines[i].strip()
                    if not stripped:
                        i += 1
                        continue
                    parts = stripped.split()
                    if len(parts) >= 5 and parts[0].isdigit() and parts[1].isdigit():
                        try:
                            freqs.append(float(parts[4].replace('*', '')))
                        except (ValueError, IndexError):
                            pass
                    elif any(kw in stripped for kw in [
                        "MODE", "NUMBER", "EXTRACTION", "RADIANS", "CYCLES",
                        "GENERALIZED", "STIFFNESS", "MASS", "ORDER"
                    ]):
                        pass
                    else:
                        break
                    i += 1
            i += 1
        return freqs

    def _parse_f06_static(self, filepath: str) -> Dict[str, Any]:
        """f06에서 정해석 변위 및 응력 파싱.

        변위 형식:
            GRID     COORD      T1    T2    T3    R1    R2    R3
                      SYS
               1        0  val  val  val  ...

        응력 형식 (CENTER/GRD 행 모두 von Mises = parts[9]):
            409  CENTER  fibre  Nxx  Nyy  Nxy  Angle  Major  Minor  vonMises ...
                 GRD  1  fibre  Nxx  Nyy  Nxy  Angle  Major  Minor  vonMises ...
        """
        disps = {}
        max_disp = 0.0
        vm_vals = []

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]

            # ---- 변위 섹션 ----
            if "D I S P L A C E M E N T S" in line:
                # "GRID ... COORD" 헤더까지 전진
                while i < len(lines) and "GRID" not in lines[i]:
                    i += 1
                i += 2  # GRID 헤더 행 + SYS 행 건너뜀
                while i < len(lines):
                    row = lines[i]
                    stripped = row.strip()
                    if not stripped:
                        # 빈 행: 페이지 나눔일 수 있으므로 다음 비어있지 않은 행 확인
                        j = i + 1
                        while j < len(lines) and not lines[j].strip():
                            j += 1
                        if j < len(lines):
                            nxt = lines[j].split()
                            if nxt and nxt[0].isdigit():
                                i = j
                                continue  # 여전히 변위 테이블 내부
                        break  # 테이블 종료
                    parts = stripped.split()
                    if parts[0].isdigit():
                        try:
                            t1 = float(parts[2])
                            t2 = float(parts[3])
                            t3 = float(parts[4])
                            nid = int(parts[0])
                            disps[nid] = [t1, t2, t3]
                            mag = np.sqrt(t1 * t1 + t2 * t2 + t3 * t3)
                            if mag > max_disp:
                                max_disp = mag
                        except (ValueError, IndexError):
                            pass
                        i += 1
                    else:
                        break  # 비정수 → 다른 섹션 시작

            # ---- 응력 섹션 ----
            # CENTER 행: EID CENTER fibre Nxx Nyy Nxy Angle Major Minor vonMises ...
            # GRD 행:    GRD nid fibre Nxx Nyy Nxy Angle Major Minor vonMises ...
            # 두 경우 모두 von Mises = parts[9]
            parts = line.split()
            if len(parts) >= 10:
                try:
                    if parts[1] == 'CENTER' and parts[0].isdigit():
                        vm_vals.append(abs(float(parts[9])))
                    elif parts[0] == 'GRD' and len(parts) >= 10:
                        vm_vals.append(abs(float(parts[9])))
                except (ValueError, IndexError):
                    pass

            i += 1

        result: Dict[str, Any] = {
            'displacements': disps,
            'max_disp': max_disp,
            'max_stress': 0.0,
            'p95_stress': 0.0,
            'median_stress': 0.0,
            'mean_stress': 0.0,
            'std_stress': 0.0,
        }
        if vm_vals:
            vm_arr = np.array(vm_vals, dtype=float)
            vm_arr = vm_arr[np.isfinite(vm_arr) & (vm_arr > 0)]
            if len(vm_arr) > 0:
                result['max_stress']    = float(np.max(vm_arr))
                result['p95_stress']    = float(np.percentile(vm_arr, 95))
                result['median_stress'] = float(np.median(vm_arr))
                result['mean_stress']   = float(np.mean(vm_arr))
                result['std_stress']    = float(np.std(vm_arr))
        return result
