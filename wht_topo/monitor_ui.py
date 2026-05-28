# -*- coding: utf-8 -*-
"""
monitor_ui.py
=============
WHT Topography Optimization — 실시간 모니터링 + Post-Processing UI (V3.0).

탭 구성:
  0. Summary Table     — 이터레이션별 수치 요약
  1. Convergence Curve — 수렴 곡선
  2. Iteration Results — 비드 높이 분포 + Mesh View / Run Analysis / Export
  3. Modal Analysis    — 고유진동수 변화

아키텍처:
  - solver.py 가 results_dir/snapshots/ 에 init.pkl + iter_NNN.pkl 저장
  - monitor 프로세스가 이 파일을 QThread 에서 로드 → WHTSolver 재해석
  - 재해석 결과는 WHTVisualizer(PyVista BackgroundPlotter)로 별도 창 표시
  - OptiStruct .fem 파일 생성 (GRID/CQUAD4/PSHELL/MAT1/SPC1/FORCE/SUBCASE)
"""

import os
import sys
import pickle
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QHeaderView,
    QComboBox, QLabel, QPushButton, QSlider, QFileDialog,
    QMessageBox, QSizePolicy,
    QMenu, QDialog, QFormLayout, QDialogButtonBox,
    QSplitter, QTextEdit,
)
from PySide6.QtCore import Signal, QObject, QThread, Qt
from PySide6.QtGui import QPixmap, QAction
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
# AutoCalculix API Dynamic Path & Import
# ────────────────────────────────────────────────────────────────────────────
import sys
AUTOCALCULIX_PATH = "D:/PythonCodeStudy/AutoCalculix"
if AUTOCALCULIX_PATH not in sys.path:
    sys.path.append(AUTOCALCULIX_PATH)

try:
    from src.autocalculix_api import run_calculix_analysis
except ImportError:
    # 예외 처리: 경로가 아직 유효하지 않거나 오타가 난 경우에 대비
    run_calculix_analysis = None
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'Segoe UI'


# ────────────────────────────────────────────────────────────────────────────
# Canvas helper
# ────────────────────────────────────────────────────────────────────────────

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


# ────────────────────────────────────────────────────────────────────────────
# Re-analysis worker (QThread)
# ────────────────────────────────────────────────────────────────────────────

class _ReAnalysisWorker(QThread):
    """
    별도 QThread 에서 FEA 재해석을 실행합니다.
    solver 프로세스와 분리된 monitor 프로세스 내에서 직접 WHTSolver 를 임포트합니다.
    """
    finished = Signal(dict)
    error    = Signal(str)
    progress = Signal(str)

    def __init__(self, snap_dir: str, iter_num: int, case_name: str,
                 num_modal_modes: int = 20, parent=None):
        super().__init__(parent)
        self.snap_dir         = snap_dir
        self.iter_num         = iter_num
        self.case_name        = case_name
        self.num_modal_modes  = num_modal_modes

    def run(self):
        try:
            import numpy as np
            from pathlib import Path
            snap_dir = Path(self.snap_dir)

            self.progress.emit("init.pkl 로드 중...")
            with open(snap_dir / "init.pkl", "rb") as f:
                init = pickle.load(f)

            model       = init["model"]
            bead_dir    = init["bead_dir"]
            design_nids = init["design_nids"]
            aggr_src    = init["aggr_src"]
            aggr_dst    = init["aggr_dst"]
            orig_coords = init["orig_coords"]

            # 원본 좌표 복원
            for nid, (x, y, z) in orig_coords.items():
                nd = model.nodes[nid]
                nd.x, nd.y, nd.z = x, y, z

            # 이터레이션 높이 적용 (iter 0 = 초기 상태 → h_elem = 0)
            if self.iter_num == 0:
                h_elem = np.zeros(len(init["design_elems"]))
            else:
                iter_file = snap_dir / f"iter_{self.iter_num:03d}.pkl"
                self.progress.emit(f"iter_{self.iter_num:03d}.pkl 로드 중...")
                with open(iter_file, "rb") as f:
                    snap = pickle.load(f)
                h_elem = snap["h_elem"]

            # h_elem → h_node (scatter-mean)
            n_int      = len(design_nids)
            h_node_sum = np.zeros(n_int)
            np.add.at(h_node_sum, aggr_src, h_elem[aggr_dst])
            node_adj   = np.bincount(aggr_src, minlength=n_int)
            h_node     = h_node_sum / (node_adj + 1e-12)

            for i, nid in enumerate(design_nids):
                ox, oy, oz = orig_coords[nid]
                nd = model.nodes[nid]
                dh = float(h_node[i])
                nd.x = ox + dh * bead_dir[0]
                nd.y = oy + dh * bead_dir[1]
                nd.z = oz + dh * bead_dir[2]

            from wht_solver.wht_solver import WHTSolver
            fea = WHTSolver(model)

            # ── 모달 해석 ──────────────────────────────────────────────────
            if self.case_name == "Modal Analysis":
                self.progress.emit("고유진동수 해석 실행 중...")
                result = fea.solve_modal(num_modes=self.num_modal_modes)
                self.finished.emit({
                    "type":   "modal",
                    "result": result,
                    "model":  model,
                })
                return

            # ── 동해석 (과도응답) ─────────────────────────────────────────
            if self.case_name.startswith("Dynamic: "):
                scen_name = self.case_name[len("Dynamic: "):]
                scenarios = init.get("dynamic_scenarios", [])
                scen = next((s for s in scenarios if s["name"] == scen_name), None)
                if scen is None:
                    self.error.emit(f"동적 시나리오 '{scen_name}' 를 init.pkl 에서 찾을 수 없습니다.")
                    return
                self.progress.emit(f"동해석 준비 중: {scen_name} ...")
                self._run_dynamic_analysis(model, scen)
                return

            # ── 정적 해석 ──────────────────────────────────────────────────
            # 하중 케이스 탐색: iter 스냅샷 우선, 없으면 init 의 static_load_cases
            lc_obj = None
            if self.iter_num > 0:
                all_lcs = snap.get("load_cases", [])
                for entry in all_lcs:
                    # entry = (name, weight, WHTLoadCase)
                    if entry[0] == self.case_name:
                        lc_obj = entry[2]
                        break
            if lc_obj is None:
                for lc, _w in init.get("static_load_cases", []):
                    if lc.name == self.case_name:
                        lc_obj = lc
                        break
            if lc_obj is None:
                self.error.emit(f"하중 케이스 '{self.case_name}' 를 찾을 수 없습니다.")
                return

            self.progress.emit(f"정적 해석 실행 중: {self.case_name} ...")
            result = fea.solve_static(lc_obj)

            self.finished.emit({
                "type":    "static",
                "lc_name": self.case_name,
                "result":  result,
                "model":   model,
            })

        except Exception:
            self.error.emit(traceback.format_exc())

    def _run_dynamic_analysis(self, model, scen: dict) -> None:
        """동적 과도응답 해석 실행 후 finished 시그널 발신."""
        try:
            import sys, os
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from wht_topo.run_topo import _load_csv, ZETA
            from wht_solver.wht_dynamic_solver import WHTDynamicSolver
            from wht_solver.wht_dynamic_common import DampingSpec

            csv_path = scen["csv_path"]
            t_start  = scen.get("t_start")
            t_end    = scen.get("t_end")
            n_modes  = int(scen.get("n_modes", 0))
            zeta     = float(scen.get("zeta", 0.02))

            self.progress.emit(f"CSV 로드 중: {Path(csv_path).name} ...")
            df, time_arr, header = _load_csv(csv_path, t_start, t_end)
            dt = float(time_arr[1] - time_arr[0]) if len(time_arr) > 1 else 1e-4
            T  = float(time_arr[-1])

            # 코너 노드 탐색 + 하중 그룹 생성 (ESLExtractor.prepare() 로직 재사용)
            self.progress.emit("코너 노드 탐색 및 하중 그룹 생성 중...")
            node_db = {nid: (nd.x, nd.y, nd.z)
                       for nid, nd in model.nodes.items()}

            from wht_topo.run_topo import (
                ESLExtractor, _build_tray,
                find_corner_nodes, find_nodes_for_corners,
                CORNER_NAMES, TRAY_W, TRAY_L,
                _apply_inertia_loads, parse_csv_header,
            )
            ext = ESLExtractor(
                model        = model,
                node_db      = node_db,
                csv_path     = csv_path,
                t_start      = t_start,
                t_end        = t_end,
                add_inertia  = bool(scen.get("add_inertia", True)),
                use_global_z = bool(scen.get("use_global_z", False)),
                n_modes      = n_modes,
                modal_modes  = self.num_modal_modes,
            )
            ext.prepare()  # 코너 노드 탐색 + 하중 그룹 빌드

            dyn_solver = WHTDynamicSolver(model)
            damping    = DampingSpec(mode="zeta", zeta=zeta)

            if n_modes > 0:
                self.progress.emit(
                    f"모달 중첩법 동해석 실행 중 (modes={n_modes}, T={T:.3f}s)...")
                dyn_result = dyn_solver.solve_modal_dynamic(
                    ext._load_groups, dt=dt, T=T,
                    n_modes=n_modes, damping=damping,
                )
            else:
                self.progress.emit(
                    f"직접 Newmark-β 동해석 실행 중 (T={T:.3f}s, dt={dt:.2e}s)...")
                dyn_result = dyn_solver.solve_direct_dynamic(
                    ext._load_groups, dt=dt, T=T,
                    damping=damping,
                    label=scen["name"],
                    modal_modes=self.num_modal_modes,
                )

            self.finished.emit({
                "type":      "dynamic",
                "lc_name":   self.case_name,
                "result":    dyn_result,
                "model":     model,
                "scen_name": scen["name"],
            })

        except Exception:
            self.error.emit(traceback.format_exc())


# ────────────────────────────────────────────────────────────────────────────
# CalculiX Re-analysis worker (QThread)
# ────────────────────────────────────────────────────────────────────────────

class _CalculixReAnalysisWorker(QThread):
    """
    별도 QThread 에서 CalculiX를 사용한 모달/정적 재해석을 구동하고
    CalculiX가 뱉어낸 3D 가시화 mesh(VTU) 자체로부터 WHTResultData를 직접 빌드하여
    원래 2D 모델 노드 개수와의 불일치(ValueError: broadcast)를 우회하고 3D 볼륨 가시화를 실현합니다.
    """
    finished = Signal(dict)
    error    = Signal(str)
    progress = Signal(str)

    def __init__(self, snap_dir: str, iter_num: int, case_name: str,
                 num_modal_modes: int = 20, parent=None):
        super().__init__(parent)
        self.snap_dir         = snap_dir
        self.iter_num         = iter_num
        self.case_name        = case_name
        self.num_modal_modes  = num_modal_modes

    def run(self):
        try:
            import numpy as np
            import pickle
            from pathlib import Path
            snap_dir = Path(self.snap_dir)

            self.progress.emit("Snapshots 정보 로드 및 변형 메시 생성 중...")
            with open(snap_dir / "init.pkl", "rb") as f:
                init = pickle.load(f)

            model       = init["model"]
            bead_dir    = init["bead_dir"]
            design_nids = init["design_nids"]
            aggr_src    = init["aggr_src"]
            aggr_dst    = init["aggr_dst"]
            orig_coords = init["orig_coords"]

            # 원본 좌표 복원
            for nid, (x, y, z) in orig_coords.items():
                nd = model.nodes[nid]
                nd.x, nd.y, nd.z = x, y, z

            # 이터레이션 비드 높이 적용
            if self.iter_num == 0:
                h_elem = np.zeros(len(init["design_elems"]))
                load_cases_raw = init.get("static_load_cases", [])
                load_cases     = [(lc.name, w, lc) for lc, w in load_cases_raw]
            else:
                iter_file = snap_dir / f"iter_{self.iter_num:03d}.pkl"
                self.progress.emit(f"iter_{self.iter_num:03d}.pkl 로드 중...")
                with open(iter_file, "rb") as f:
                    snap = pickle.load(f)
                h_elem = snap["h_elem"]
                load_cases = snap.get("load_cases", [])

            # h_elem -> h_node
            n_int      = len(design_nids)
            h_node_sum = np.zeros(n_int)
            np.add.at(h_node_sum, aggr_src, h_elem[aggr_dst])
            node_adj   = np.bincount(aggr_src, minlength=n_int)
            h_node     = h_node_sum / (node_adj + 1e-12)

            for i, nid in enumerate(design_nids):
                ox, oy, oz = orig_coords[nid]
                nd = model.nodes[nid]
                dh = float(h_node[i])
                nd.x = ox + dh * bead_dir[0]
                nd.y = oy + dh * bead_dir[1]
                nd.z = oz + dh * bead_dir[2]

            # ── CalculiX 용 입력 매핑 ─────────────────────────────────────────
            nodes_dict = {nid: (nd.x, nd.y, nd.z) for nid, nd in model.nodes.items()}
            
            elements_list = []
            for eid, elem in model.elements.items():
                etype = getattr(elem, "element_type", "QUAD4")
                pid   = getattr(elem, "pid", 1)
                elements_list.append((eid, etype, list(elem.node_ids), pid))
                
            properties_dict = {}
            for pid, prop in getattr(model, "properties", {}).items():
                t = getattr(prop, "t", 1.2)
                mid = getattr(prop, "mid", 1)
                mat = getattr(model, "materials", {}).get(mid)
                E = getattr(mat, "E", 210000.0)
                nu = getattr(mat, "nu", 0.3)
                rho = getattr(mat, "rho", 7.85e-9)
                properties_dict[pid] = (t, E, nu, rho)
                
            if not properties_dict:
                properties_dict[1] = (1.2, 210000.0, 0.3, 7.85e-9)

            job_name = f"ccx_iter{self.iter_num:03d}_{self.case_name.replace(' ', '_')}"
            analysis_config = {
                "job_name": job_name,
                "num_modes": self.num_modal_modes
            }
            
            bcs = []
            forces = []
            analysis_type = "modal" if self.case_name == "Modal Analysis" else "static"
            
            if analysis_type == "static":
                lc_obj = None
                for entry in load_cases:
                    if entry[0] == self.case_name:
                        lc_obj = entry[2]
                        break
                if lc_obj is None:
                    raise ValueError(f"하중 케이스 '{self.case_name}'를 찾을 수 없습니다.")
                
                # SPC 구속조건 매핑
                for bc in lc_obj.bcs:
                    bcs.append((bc.node_id, list(bc.dofs), getattr(bc, "value", 0.0)))
                # FORCE 하중조건 매핑
                for fc in lc_obj.forces:
                    forces.append((fc.node_id, list(fc.load_vector)))

            # ── AutoCalculix API 호출 ─────────────────────────────────────────
            self.progress.emit("CalculiX 솔버 및 변환 스레드 구동 중...")
            
            if run_calculix_analysis is None:
                raise ImportError("AutoCalculix API 모듈을 임포트하지 못했습니다. D:/PythonCodeStudy/AutoCalculix 경로를 점검하십시오.")
                
            ccx_result = run_calculix_analysis(
                nodes=nodes_dict,
                elements=elements_list,
                properties=properties_dict,
                analysis_type=analysis_type,
                analysis_config=analysis_config,
                bcs=bcs,
                forces=forces,
                workspace_dir=str(snap_dir) # 스냅샷 디렉토리를 작업공간으로 활용
            )
            
            if analysis_type == "modal" and "workspace" in ccx_result and "job_name" in ccx_result:
                dat_path = Path(ccx_result["workspace"]) / f"{ccx_result['job_name']}.dat"
                self._print_calculix_dat_summary(dat_path)
            
            vtu_base_str = ccx_result.get("vtu_base", "")
            if not vtu_base_str:
                raise RuntimeError(f"CalculiX 해석이 비정상 종료되었거나 .frd 가시화 파일 생성에 실패했습니다.")

            # ── CalculiX 결과 3D Volume Mesh -> WHTResultData 직접 가공 ───────
            self.progress.emit("CalculiX 3D 볼륨 결과 및 가시화 포맷 변환 중...")
            
            import pyvista as pv
            from wht_converter.wht_models import WHTResultData, WHTMetadata
            
            if analysis_type == "modal":
                freq_list = ccx_result.get("frequencies", [])
                n_modes = len(freq_list)
                if n_modes == 0:
                    raise RuntimeError("파싱된 고유진동수가 존재하지 않습니다.")
                    
                frequencies = np.array([f["hz"] for f in freq_list])
                
                # 1번 모드 파일 로드하여 공통 3D Volume 기하 정보 추출
                vtu_path_1 = Path(f"{vtu_base_str}.1.vtu")
                if not vtu_path_1.exists():
                    vtu_path_1 = Path(f"{vtu_base_str}.01.vtu")
                if not vtu_path_1.exists():
                    vtu_path_1 = Path(f"{vtu_base_str}.001.vtu")
                    
                if not vtu_path_1.exists():
                    raise FileNotFoundError(f"CalculiX 모달 해석 1번 결과를 찾을 수 없습니다: {vtu_path_1} (.1.vtu 또는 .01.vtu 검색 실패)")
                    
                mesh_1 = pv.read(str(vtu_path_1))
                vtu_nodes = np.array(mesh_1.points, dtype=np.float32)
                
                # CSR 메쉬 토폴로지 추출
                raw_cells = np.array(mesh_1.cells)
                offsets = []
                connectivity = []
                idx = 0
                while idx < len(raw_cells):
                    n_nodes = raw_cells[idx]
                    offsets.append(len(connectivity))
                    for o_idx in range(1, n_nodes + 1):
                        connectivity.append(raw_cells[idx + o_idx])
                    idx += (n_nodes + 1)
                offsets.append(len(connectivity))
                offsets = np.array(offsets, dtype=np.int32)
                connectivity = np.array(connectivity, dtype=np.int32)
                cell_types = np.array(mesh_1.celltypes if hasattr(mesh_1, "celltypes") else mesh_1.cell_types, dtype=np.uint8)
                
                # 모든 모드 파일 순회하며 'U' 변형 데이터(ModeShape) 수집
                mode_shapes = []
                for i in range(1, n_modes + 1):
                    vtu_path = Path(f"{vtu_base_str}.{i}.vtu")
                    if not vtu_path.exists():
                        vtu_path = Path(f"{vtu_base_str}.{i:02d}.vtu")
                    if not vtu_path.exists():
                        vtu_path = Path(f"{vtu_base_str}.{i:03d}.vtu")
                        
                    if not vtu_path.exists():
                        continue
                    mesh_m = pv.read(str(vtu_path))
                    disp_key = next((k for k in mesh_m.point_data.keys() if k.upper() in ('U', 'DISP', 'DISPLACEMENTS')), None)
                    if disp_key is None:
                        raise ValueError("VTU 모드 결과에서 변위 데이터('U')를 찾지 못했습니다.")
                        
                    disp_arr = np.array(mesh_m.point_data[disp_key])
                    mode_shapes.append(disp_arr[:, :3]) # (N, 3)
                    
                mode_shapes = np.array(mode_shapes) # (n_modes, N, 3)
                
                point_data = {
                    "ModeShape": mode_shapes
                }
                field_data = {
                    "Frequency_Hz": frequencies
                }
                
                meta = WHTMetadata(
                    solver_name="CalculiX", solver_version="2.23",
                    analysis_type="modal", coordinate_system="cartesian",
                    unit_length="mm", unit_force="N",
                )
                
                rd = WHTResultData(
                    nodes=vtu_nodes,
                    connectivity=connectivity,
                    offsets=offsets,
                    cell_types=cell_types,
                    point_data=point_data,
                    field_data=field_data,
                    time_values=frequencies,
                    metadata=meta
                )
                
            else:
                # 정적 해석 단일 변위 파싱
                vtu_path = Path(f"{vtu_base_str}.vtu")
                if not vtu_path.exists():
                    vtu_path = Path(f"{vtu_base_str}.01.vtu")
                if not vtu_path.exists():
                    vtu_path = Path(f"{vtu_base_str}.1.vtu")
                if not vtu_path.exists():
                    vtu_paths = sorted(Path(vtu_base_str).parent.glob(f"{Path(vtu_base_str).name}*.vtu"))
                    if vtu_paths:
                        vtu_path = vtu_paths[0]
                    else:
                        raise FileNotFoundError(f"CalculiX VTU 정적 해석 결과 파일을 찾을 수 없습니다.")
                        
                mesh_s = pv.read(str(vtu_path))
                vtu_nodes = np.array(mesh_s.points, dtype=np.float32)
                
                raw_cells = np.array(mesh_s.cells)
                offsets = []
                connectivity = []
                idx = 0
                while idx < len(raw_cells):
                    n_nodes = raw_cells[idx]
                    offsets.append(len(connectivity))
                    for o_idx in range(1, n_nodes + 1):
                        connectivity.append(raw_cells[idx + o_idx])
                    idx += (n_nodes + 1)
                offsets.append(len(connectivity))
                offsets = np.array(offsets, dtype=np.int32)
                connectivity = np.array(connectivity, dtype=np.int32)
                cell_types = np.array(mesh_s.celltypes if hasattr(mesh_s, "celltypes") else mesh_s.cell_types, dtype=np.uint8)
                
                disp_key = next((k for k in mesh_s.point_data.keys() if k.upper() in ('U', 'DISP', 'DISPLACEMENTS')), None)
                if disp_key is None:
                    raise ValueError("VTU 정적 결과에서 변위 데이터('U')를 찾지 못했습니다.")
                    
                disp_arr = np.array(mesh_s.point_data[disp_key]) # (N, 3)
                
                point_data = {
                    "Displacement": disp_arr[np.newaxis, :, :3] # (1, N, 3)
                }
                
                meta = WHTMetadata(
                    solver_name="CalculiX", solver_version="2.23",
                    analysis_type="static", coordinate_system="cartesian",
                    unit_length="mm", unit_force="N",
                )
                
                rd = WHTResultData(
                    nodes=vtu_nodes,
                    connectivity=connectivity,
                    offsets=offsets,
                    cell_types=cell_types,
                    point_data=point_data,
                    time_values=np.array([0.0]),
                    metadata=meta
                )

            self.finished.emit({
                "type":   analysis_type,
                "wht_result_data": rd,
                "lc_name": self.case_name,
                "is_ccx": True
            })

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())

    def _print_calculix_dat_summary(self, dat_path):
        try:
            if not dat_path.exists():
                return
            print(f"\n--- CalculiX Modal Summary ({dat_path.name}) ---", flush=True)
            with open(dat_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            in_eigen = False
            empty_count = 0
            for line in lines:
                if "E I G E N V A L U E" in line:
                    in_eigen = True
                
                if in_eigen:
                    print(line.rstrip(), flush=True)
                    if not line.strip():
                        empty_count += 1
                    else:
                        empty_count = 0
                    
                    if empty_count > 2 or "TOTAL TIME" in line:
                        break
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# OptiStruct .fem 내보내기
# ────────────────────────────────────────────────────────────────────────────

def _write_optistruct_fem(out_path: str, snap_dir: str, iter_num: int) -> str:
    """
    선택 이터레이션의 변형 메시 + 모든 하중 케이스를 OptiStruct .fem 파일로 저장합니다.

    GRID / CQUAD4/CTRIA3 / PSHELL / MAT1 / SPC1 / FORCE / MOMENT / SUBCASE 포함.
    """
    snap_dir = Path(snap_dir)

    with open(snap_dir / "init.pkl", "rb") as f:
        init = pickle.load(f)

    model       = init["model"]
    bead_dir    = init["bead_dir"]
    design_nids = init["design_nids"]
    aggr_src    = init["aggr_src"]
    aggr_dst    = init["aggr_dst"]
    orig_coords = init["orig_coords"]

    # 원본 좌표 복원 후 높이 적용
    for nid, (x, y, z) in orig_coords.items():
        nd = model.nodes[nid]
        nd.x, nd.y, nd.z = x, y, z

    if iter_num == 0:
        h_elem = np.zeros(len(init["design_elems"]))
        load_cases_raw = init.get("static_load_cases", [])
        load_cases     = [(lc.name, w, lc) for lc, w in load_cases_raw]
    else:
        with open(snap_dir / f"iter_{iter_num:03d}.pkl", "rb") as f:
            snap = pickle.load(f)
        h_elem     = snap["h_elem"]
        load_cases = snap.get("load_cases", [])
        if not load_cases:
            load_cases_raw = init.get("static_load_cases", [])
            load_cases = [(lc.name, w, lc) for lc, w in load_cases_raw]

    # h_elem → h_node → 노드 좌표 업데이트
    n_int      = len(design_nids)
    h_node_sum = np.zeros(n_int)
    np.add.at(h_node_sum, aggr_src, h_elem[aggr_dst])
    node_adj   = np.bincount(aggr_src, minlength=n_int)
    h_node     = h_node_sum / (node_adj + 1e-12)
    for i, nid in enumerate(design_nids):
        ox, oy, oz = orig_coords[nid]
        nd = model.nodes[nid]
        nd.x = ox + float(h_node[i]) * bead_dir[0]
        nd.y = oy + float(h_node[i]) * bead_dir[1]
        nd.z = oz + float(h_node[i]) * bead_dir[2]

    lines = []
    lines.append(f"$ WHT Topography Optimization — OptiStruct Export")
    lines.append(f"$ Iteration: {iter_num}   Load Cases: {len(load_cases)}")
    lines.append("$")

    # SUBCASE 블록 (BEGIN BULK 앞)
    for sc_id, (lc_name, _w, _lc) in enumerate(load_cases, start=1):
        lines.append(f"SUBCASE   {sc_id}")
        lines.append(f"  LABEL = {lc_name}")
        lines.append(f"  SPC = 1")
        lines.append(f"  LOAD = {sc_id}")

    lines.append("BEGIN BULK")
    lines.append("PARAM,POST,-1")
    lines.append("PARAM,AUTOSPC,YES")
    lines.append("$")

    # GRID
    lines.append("$ === NODES ===")
    for nid in sorted(model.nodes.keys()):
        nd = model.nodes[nid]
        lines.append(f"GRID,{nid:>8d},    ,{nd.x:>12.4f},{nd.y:>12.4f},{nd.z:>12.4f}")

    lines.append("$")
    lines.append("$ === ELEMENTS ===")

    # 속성 수집 (pid → (t, E, nu, rho, mid))
    pid_props: dict = {}
    mid_counter = [1]

    def _get_pid_props(eid):
        elem = model.elements[eid]
        pid  = getattr(elem, "pid", 1)
        if pid not in pid_props:
            prop = model.properties.get(pid) if hasattr(model, "properties") else None
            t    = getattr(prop, "t",   1.2)  if prop else 1.2
            mat  = None
            if prop and hasattr(prop, "mid"):
                mat = model.materials.get(prop.mid) if hasattr(model, "materials") else None
            E   = getattr(mat, "E",   70000.0) if mat else 70000.0
            nu  = getattr(mat, "nu",  0.3)     if mat else 0.3
            rho = getattr(mat, "rho", 2.7e-9)  if mat else 2.7e-9
            mid = mid_counter[0]; mid_counter[0] += 1
            pid_props[pid] = (t, E, nu, rho, mid)
        return pid

    for eid in sorted(model.elements.keys()):
        elem  = model.elements[eid]
        pid   = _get_pid_props(eid)
        nids  = list(elem.node_ids)
        etype = getattr(elem, "element_type", "")
        if len(nids) == 4 or etype in ("QUAD4", "QUAD"):
            lines.append(f"CQUAD4,{eid:>8d},{pid:>8d},"
                         + ",".join(f"{n:>8d}" for n in nids))
        elif len(nids) == 3 or etype in ("TRIA3", "TRIA"):
            lines.append(f"CTRIA3,{eid:>8d},{pid:>8d},"
                         + ",".join(f"{n:>8d}" for n in nids))

    lines.append("$")
    lines.append("$ === PROPERTIES ===")
    for pid, (t, E, nu, rho, mid) in pid_props.items():
        lines.append(f"PSHELL,{pid:>8d},{mid:>8d},{t:>12.4f}")

    lines.append("$")
    lines.append("$ === MATERIALS ===")
    for pid, (t, E, nu, rho, mid) in pid_props.items():
        G = E / (2 * (1 + nu))
        lines.append(f"MAT1,{mid:>8d},{E:>12.2f},{G:>12.2f},{nu:>8.4f},{rho:>12.4e}")

    lines.append("$")
    lines.append("$ === BOUNDARY CONDITIONS (SPC = 1, 모델 레벨) ===")
    # 모델 레벨 SPC
    if hasattr(model, "spcs") and model.spcs:
        for spc in model.spcs:
            nid_list = getattr(spc, "node_ids", [getattr(spc, "node_id", None)])
            dofs_raw = getattr(spc, "dofs", (0, 1, 2, 3, 4, 5))
            dof_str  = "".join(str(d + 1) for d in sorted(dofs_raw))
            for nid in nid_list:
                if nid is not None:
                    lines.append(f"SPC1,       1,{dof_str:>8s},{nid:>8d}")
    else:
        lines.append("$ (모델 레벨 SPC 없음 — 하중 케이스 BC 참조)")

    # 하중 케이스별 SPC + FORCE
    for sc_id, (lc_name, _w, lc) in enumerate(load_cases, start=1):
        lines.append(f"$")
        lines.append(f"$ === SUBCASE {sc_id}: {lc_name} ===")

        # 하중 케이스 BC (SID = sc_id + 100 으로 구분, SUBCASE 에서는 SPC=1 공유)
        bc_sid = sc_id + 100
        for bc in lc.bcs:
            dof_str = "".join(str(d + 1) for d in sorted(bc.dofs))
            val     = getattr(bc, "value", 0.0)
            if abs(val) < 1e-12:
                lines.append(f"SPC1,{bc_sid:>8d},{dof_str:>8s},{bc.node_id:>8d}")
            else:
                lines.append(f"SPC,{bc_sid:>8d},{bc.node_id:>8d},"
                             + "".join(str(d + 1) for d in sorted(bc.dofs))
                             + f",{val:>12.6f}")

        # 하중 (FORCE / MOMENT)
        for force in lc.forces:
            vec = force.load_vector
            fx, fy, fz = float(vec[0]), float(vec[1]), float(vec[2])
            mx, my, mz = (float(vec[3]), float(vec[4]), float(vec[5])) if len(vec) > 3 else (0, 0, 0)
            if abs(fx) + abs(fy) + abs(fz) > 1e-12:
                lines.append(f"FORCE,{sc_id:>8d},{force.node_id:>8d},       0,"
                             f"       1,{fx:>12.4f},{fy:>12.4f},{fz:>12.4f}")
            if abs(mx) + abs(my) + abs(mz) > 1e-12:
                lines.append(f"MOMENT,{sc_id:>7d},{force.node_id:>8d},       0,"
                             f"       1,{mx:>12.4f},{my:>12.4f},{mz:>12.4f}")

    lines.append("$")
    lines.append("ENDDATA")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


# ────────────────────────────────────────────────────────────────────────────
# Main Monitor Window
# ────────────────────────────────────────────────────────────────────────────

class WHTMonitorWindow(QMainWindow):
    def __init__(self, stop_event=None, results_dir: str = "", num_modal_modes: int = 20):
        super().__init__()
        self.setWindowTitle("WHT Topography Optimization Monitor (V3.0)")
        self.resize(1300, 900)
        self.stop_event       = stop_event
        self.results_dir      = results_dir
        self.snap_dir         = ""
        self.num_modal_modes  = num_modal_modes

        # ── 데이터 저장소 ──────────────────────────────────────────────────
        self.history = {
            "iter": [], "compliance": [], "avg_h": [], "max_h": [],
            "dx": [], "area_ratio": [], "min_width": [], "frequencies": [], "cases": {},
        }
        self.height_snapshots: list = []  # {"iter", "coords", "heights"}
        self.mesh_edge_segs = None  # Optional[np.ndarray]
        self.ref_freqs = None
        self.case_names: list = []

        # ── 위젯 참조 ──────────────────────────────────────────────────────
        self.height_canvas = self.curve_canvas = None
        self.table = self.modal_table = None
        self.metric_combo = self.height_iter_combo = None
        self._height_colorbar = None
        self.status_label = self.stop_btn = self.top_btn = self.reset_btn = None
        self.height_slider = self.btn_prev_h = self.btn_next_h = None
        self.iter_case_combo = None   # Load Case 콤보 (Iteration Results 탭)
        self.iter_run_btn = None      # Run Analysis 버튼
        self.iter_ccx_btn = None      # Run CalculiX 버튼
        self.iter_export_btn = None   # Export OptiStruct 버튼
        self.iter_mesh_btn = None     # Mesh View 버튼
        self.iter_status_label = None # 해석 상태 텍스트
        self._re_worker = None        # Optional[_ReAnalysisWorker]
        self._vis_window = None       # Optional[WHTVisualizer] — 재사용 창

        self._init_ui()
        self._setup_shortcuts()
        self._restore_history()

    def _restore_history(self):
        """재시작 시 기존 이터레이션 pkl 파일들을 읽어들여 그래프 히스토리를 자동 복원합니다."""
        import glob
        import pickle
        import os

        if not self.results_dir: return
        snap_dir = Path(self.results_dir) / "snapshots"
        if not snap_dir.exists(): return

        self.snap_dir = str(snap_dir)
        try:
            self._register_dynamic_scenarios(self.snap_dir)
        except Exception:
            pass

        # init.pkl 읽기 (mesh_edge_segs 복원)
        init_file = snap_dir / "init.pkl"
        if init_file.exists():
            try:
                with open(init_file, "rb") as f:
                    init_data = pickle.load(f)
                if "mesh_edge_segs" in init_data:
                    self.mesh_edge_segs = init_data["mesh_edge_segs"]
            except Exception as e:
                print(f"[Monitor] init.pkl 로드 실패: {e}")

        # iter_XXX.pkl 읽어서 히스토리 재구성 (주요 지표 추출)
        iter_files = sorted(glob.glob(os.path.join(str(snap_dir), "iter_*.pkl")))
        for pkl_path in iter_files:
            try:
                iter_str = Path(pkl_path).stem.split('_')[-1]
                if not iter_str.isdigit(): continue
                iter_num = int(iter_str)
                
                with open(pkl_path, "rb") as f:
                    snap = pickle.load(f)

                if iter_num not in self.history["iter"]:
                    self.history["iter"].append(iter_num)
                    # 파일에 저장된 지표들이 있으면 불러오고 없으면 0.0으로 초기화
                    self.history["compliance"].append(snap.get("compliance", 0.0))
                    self.history["avg_h"].append(snap.get("avg_h", 0.0))
                    self.history["max_h"].append(snap.get("max_h", 0.0))
                    self.history["dx"].append(snap.get("dx", 0.0))
                    self.history["area_ratio"].append(snap.get("area_ratio", 0.0))
                    self.history["min_width"].append(snap.get("min_width", 0.0))
                    
                    if "frequencies" in snap:
                        self.history["frequencies"].append(snap["frequencies"])
                    else:
                        self.history["frequencies"].append([])
                        
                    if "load_cases" in snap:
                        cases_dict = {lc.name: {"compliance": 0.0} for lc, w in snap["load_cases"]}
                        self.history["cases"][iter_num] = cases_dict
                    
                    # 3D 뷰용 높이/좌표 스냅샷 복원
                    self.height_snapshots.append({
                        "iter": iter_num,
                        "coords": snap.get("coords", []),
                        "heights": snap.get("h_elem", [])
                    })
            except Exception as e:
                print(f"[Monitor] {pkl_path} 로드 실패: {e}")


    # ────────────────────────────────────────────────────────────────────────
    # UI 구성
    # ────────────────────────────────────────────────────────────────────────

    def _init_ui(self):
        central    = QWidget()
        self.setCentralWidget(central)
        root_lay   = QVBoxLayout(central)
        root_lay.setContentsMargins(4, 4, 4, 4)
        root_lay.setSpacing(4)

        # ── 로고 + 상단 컨트롤 바 ────────────────────────────────────────
        top_widget = QWidget()
        top_widget.setFixedHeight(64)
        top_bar = QHBoxLayout(top_widget)
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(4)

        logo_label = QLabel()
        logo_label.setFixedSize(64, 64)
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "logo_icon_64x64.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(os.path.dirname(__file__), "resources", "logo_icon_64x64.png")
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path).scaled(
                64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            logo_label.setPixmap(pix)
        else:
            logo_label.setText("WHT")
            logo_label.setAlignment(Qt.AlignCenter)
            logo_label.setStyleSheet(
                "background:#1a3a5c;color:white;font-weight:bold;font-size:18px;"
                "border-radius:6px;"
            )
        top_bar.addWidget(logo_label)

        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet(
            "font-weight:bold;color:#ffffff;font-size:13px;"
        )
        self.status_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()

        for text, slot, style, w in [
            ("Stay on Top: OFF", self._toggle_always_on_top, "", 130),
            ("Reset UI",         self._on_reset_clicked,     "", 90),
            ("Request STOP",     self._on_stop_clicked,
             "background:#fdd;color:#c00;font-weight:bold;", 120),
        ]:
            btn = QPushButton(text)
            btn.setFixedWidth(w)
            if style:
                btn.setStyleSheet(style)
            btn.clicked.connect(slot)
            top_bar.addWidget(btn)
            if text == "Stay on Top: OFF":
                self.top_btn = btn
            elif text == "Request STOP":
                self.stop_btn = btn

        root_lay.addWidget(top_widget)

        # ── 탭 ────────────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        root_lay.addWidget(self.tabs)

        self._build_tab_summary()
        self._build_tab_curve()
        self._build_tab_height()
        self._build_tab_modal()

    # ── 탭 빌더 ─────────────────────────────────────────────────────────────

    def _build_tab_summary(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self.table = QTableWidget(0, 0)
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(self._on_table_header_context_menu)
        lay.addWidget(self.table)
        self.tabs.addTab(tab, "Summary Table")

    def _build_tab_curve(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        ctrl = QHBoxLayout()
        self.metric_combo = QComboBox()
        self.metric_combo.setMaxVisibleItems(25)
        self.metric_combo.addItems([
            "ALL (Normalized)", "Compliance", "Avg_h", "Max_h",
            "dx", "Area_Ratio", "Min_Width", "Natural Frequencies",
        ])
        self.metric_combo.currentTextChanged.connect(self._update_curves)
        ctrl.addWidget(QLabel("Select Metric:"))
        ctrl.addWidget(self.metric_combo)
        ctrl.addStretch()
        lay.addLayout(ctrl)
        self.curve_canvas = PlotCanvas(tab)
        lay.addWidget(self.curve_canvas)
        self.tabs.addTab(tab, "Convergence Curve")

    def _build_tab_height(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # ── 이터레이션 선택 행 ────────────────────────────────────────────
        ctrl_iter_w = QWidget()
        ctrl_iter_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        ctrl_iter = QHBoxLayout(ctrl_iter_w)
        ctrl_iter.setContentsMargins(0, 0, 0, 0)
        self.height_iter_combo = QComboBox()
        self.height_iter_combo.setMaxVisibleItems(25)
        self.height_iter_combo.addItem("Latest")
        self.height_iter_combo.currentIndexChanged.connect(self._on_height_combo_changed)
        self.btn_prev_h = QPushButton("<"); self.btn_prev_h.setFixedWidth(30)
        self.btn_prev_h.clicked.connect(self._on_prev_height)
        self.btn_next_h = QPushButton(">"); self.btn_next_h.setFixedWidth(30)
        self.btn_next_h.clicked.connect(self._on_next_height)
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setMinimum(0); self.height_slider.setMaximum(0)
        self.height_slider.valueChanged.connect(self._on_height_slider_changed)
        for w in [QLabel("Iteration:"), self.btn_prev_h,
                  self.height_iter_combo, self.btn_next_h,
                  QLabel("  Slider:"), self.height_slider]:
            ctrl_iter.addWidget(w)
        ctrl_iter.addStretch()
        lay.addWidget(ctrl_iter_w)

        # ── 캔버스 + 하단 패널을 QSplitter로 분리 (드래그로 크기 조절) ──
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        lay.addWidget(splitter, stretch=1)

        # 상단: 높이 분포 캔버스
        self.height_canvas = PlotCanvas(tab)
        splitter.addWidget(self.height_canvas)

        # 하단: 버튼 + 상태창
        bottom_widget = QWidget()
        bottom_lay = QVBoxLayout(bottom_widget)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(2)
        splitter.addWidget(bottom_widget)
        splitter.setSizes([400, 120])  # 초기 비율 (캔버스:하단)

        # ── 액션 행: Mesh View / Run Analysis / Export ───────────────────
        ctrl_widget = QWidget()
        ctrl_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        ctrl_action = QHBoxLayout(ctrl_widget)
        ctrl_action.setContentsMargins(0, 4, 0, 4)
        ctrl_action.setSpacing(6)

        self.iter_mesh_btn = QPushButton("Mesh View")
        self.iter_mesh_btn.setStyleSheet("font-weight:bold;")
        self.iter_mesh_btn.setToolTip("선택 이터레이션 메시를 wht_visualizer 창에 표시")
        self.iter_mesh_btn.clicked.connect(self._on_mesh_view_clicked)
        ctrl_action.addWidget(self.iter_mesh_btn)

        ctrl_action.addWidget(QLabel("  Load Case:"))
        self.iter_case_combo = QComboBox(); self.iter_case_combo.setMinimumWidth(160)
        self.iter_case_combo.setMaxVisibleItems(25)
        self.iter_case_combo.addItem("Modal Analysis")
        ctrl_action.addWidget(self.iter_case_combo)

        self.iter_run_btn = QPushButton("Run Analysis")
        self.iter_run_btn.setStyleSheet("font-weight:bold;")
        self.iter_run_btn.setToolTip("선택 하중 케이스로 wht_solver 해석 후 결과를 visualizer에 표시")
        self.iter_run_btn.clicked.connect(self._on_run_analysis)
        ctrl_action.addWidget(self.iter_run_btn)

        self.iter_ccx_btn = QPushButton("Run CalculiX")
        self.iter_ccx_btn.setStyleSheet("font-weight:bold; background-color:#1e3d59; color:white;")
        self.iter_ccx_btn.setToolTip("선택 하중 케이스로 CalculiX 해석 후 결과를 visualizer에 표시")
        self.iter_ccx_btn.clicked.connect(self._on_run_calculix)
        ctrl_action.addWidget(self.iter_ccx_btn)

        self.iter_export_btn = QPushButton("Export OptiStruct .fem")
        self.iter_export_btn.setStyleSheet("font-weight:bold;")
        self.iter_export_btn.setToolTip("선택 이터레이션의 변형 메시 + 하중 케이스를 .fem 파일로 저장")
        self.iter_export_btn.clicked.connect(self._on_export_optistruct)
        ctrl_action.addWidget(self.iter_export_btn)

        ctrl_action.addWidget(QLabel("  뷰어:"))
        self.viewer_combo = QComboBox()
        self.viewer_combo.addItems(["WHTVisualizer", "ParaView"])
        self.viewer_combo.setToolTip("해석 결과 표시 방식 선택\nWHTVisualizer: PyVista 내장 뷰어\nParaView: VTKHDF 내보내기 후 ParaView 실행")
        ctrl_action.addWidget(self.viewer_combo)

        ctrl_action.addStretch()
        bottom_lay.addWidget(ctrl_widget)

        # ── 상태 표시창 (스크롤 가능, 기본 2-3줄) ───────────────────────
        self.iter_status_label = QTextEdit()
        self.iter_status_label.setReadOnly(True)
        self.iter_status_label.setStyleSheet(
            "color:#555; font-size:11px; background:#1e1e1e; border:none;"
        )
        fm = self.iter_status_label.fontMetrics()
        line_h = fm.lineSpacing()
        self.iter_status_label.setMinimumHeight(line_h * 2 + 8)
        self.iter_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bottom_lay.addWidget(self.iter_status_label, stretch=1)

        self.tabs.addTab(tab, "Iteration Results")

    def _build_tab_modal(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        n = self.num_modal_modes
        # Mode N은 vertical header(행 제목), 데이터 컬럼은 "Ref. (Hz)" + Iter 별 추가
        self.modal_table = QTableWidget(n, 1)
        self.modal_table.setHorizontalHeaderLabels(["Ref. (Hz)"])
        self.modal_table.setVerticalHeaderLabels([f"Mode {i+1}" for i in range(n)])
        # 가로 스크롤 허용: Stretch 대신 고정 폭 사용 (이터레이션마다 컬럼 증가)
        self.modal_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.modal_table.horizontalHeader().setDefaultSectionSize(80)
        self.modal_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.modal_table.setColumnWidth(0, 80)
        lay.addWidget(self.modal_table)
        self.tabs.addTab(tab, "Modal Analysis")


    def _on_table_header_context_menu(self, pos):
        menu = QMenu(self)
        
        col_menu = menu.addMenu("Toggle Columns")
        for i in range(self.table.columnCount()):
            col_name = self.table.horizontalHeaderItem(i).text()
            action = QAction(col_name, self)
            action.setCheckable(True)
            action.setChecked(not self.table.isColumnHidden(i))
            action.triggered.connect(lambda checked, i=i: self.table.setColumnHidden(i, not checked))
            col_menu.addAction(action)
            
        menu.addSeparator()
        
        plot_action = QAction("Plot Custom Graph", self)
        plot_action.triggered.connect(self._show_custom_plot_dialog)
        menu.addAction(plot_action)
        
        header = self.table.horizontalHeader()
        global_pos = header.mapToGlobal(pos)
        menu.exec(global_pos)

    def _show_custom_plot_dialog(self):
        if self.table.rowCount() == 0:
            QMessageBox.warning(self, "No Data", "플롯할 데이터가 없습니다.")
            return
            
        dlg = QDialog(self)
        dlg.setWindowTitle("Custom Plot")
        dlg.setMinimumWidth(300)
        lay = QFormLayout(dlg)
        
        combo_x = QComboBox()
        combo_y = QComboBox()
        combo_x.setMaxVisibleItems(25)
        combo_y.setMaxVisibleItems(25)
        
        headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
        combo_x.addItems(headers)
        combo_y.addItems(headers)
        
        lay.addRow("X-Axis:", combo_x)
        lay.addRow("Y-Axis:", combo_y)
        
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        lay.addRow(btn_box)
        
        if dlg.exec() == QDialog.Accepted:
            idx_x = combo_x.currentIndex()
            idx_y = combo_y.currentIndex()
            
            x_data = []
            y_data = []
            for r in range(self.table.rowCount()):
                try:
                    x_val = float(self.table.item(r, idx_x).text())
                    y_val = float(self.table.item(r, idx_y).text())
                    x_data.append(x_val)
                    y_data.append(y_val)
                except Exception as e:
                    pass
                    
            if not x_data: return
            
            plot_win = QDialog(self)
            plot_win.setWindowTitle(f"{headers[idx_y]} vs {headers[idx_x]}")
            plot_win.resize(600, 450)
            plot_lay = QVBoxLayout(plot_win)
            canvas = PlotCanvas(plot_win)
            plot_lay.addWidget(canvas)
            
            canvas.ax.plot(x_data, y_data, marker='o', linestyle='-', color='b')
            canvas.ax.set_xlabel(headers[idx_x])
            canvas.ax.set_ylabel(headers[idx_y])
            canvas.ax.grid(True, linestyle='--', alpha=0.7)
            canvas.fig.tight_layout()
            canvas.draw()
            
            plot_win.show()

    # ────────────────────────────────────────────────────────────────────────
    # 단축키 / 버튼 핸들러
    # ────────────────────────────────────────────────────────────────────────

    def _setup_shortcuts(self):
        from PySide6.QtGui import QShortcut, QKeySequence
        sc = QShortcut(QKeySequence("Ctrl+T"), self)
        sc.activated.connect(self._toggle_always_on_top)

    def _toggle_always_on_top(self):
        on = bool(self.windowFlags() & Qt.WindowStaysOnTopHint)
        if on:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
            if self.top_btn:
                self.top_btn.setText("Stay on Top: OFF")
                self.top_btn.setStyleSheet("")
        else:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            if self.top_btn:
                self.top_btn.setText("Stay on Top: ON")
                self.top_btn.setStyleSheet(
                    "background-color:#ffcccc;font-weight:bold;"
                )
        self.show()

    def _on_stop_clicked(self):
        if self.stop_event:
            self.stop_event.set()
            if self.status_label:
                self.status_label.setText("Status: Stop Requested...")
                self.status_label.setStyleSheet(
                    "color:red;font-weight:bold;font-size:14px;"
                )
            if self.stop_btn:
                self.stop_btn.setEnabled(False)
                self.stop_btn.setText("Stopping...")

    def _on_reset_clicked(self):
        self._clear_history()
        if self.curve_canvas:  self.curve_canvas.draw()
        if self.height_canvas: self.height_canvas.draw()

    # ────────────────────────────────────────────────────────────────────────
    # 데이터 수신
    # ────────────────────────────────────────────────────────────────────────

    def update_data(self, data: dict):
        if data.get("status") == "STOP":
            if self.status_label:
                self.status_label.setText("Status: Optimization Completed / Stopped")
                self.status_label.setStyleSheet(
                    "color:green;font-weight:bold;font-size:14px;"
                )
            if self.stop_btn:
                self.stop_btn.setEnabled(False)
            if "iter" not in data:
                return

        try:
            it = data["iter"]

            # snap_dir 수신 (최초 1회만 동적 시나리오 콤보 등록)
            if data.get("snap_dir") and not self.snap_dir:
                self.snap_dir = data["snap_dir"]
                self._register_dynamic_scenarios(self.snap_dir)

            # ── 메시 엣지 수신 (최초 1회) ────────────────────────────────
            if "mesh_edge_segs" in data and self.mesh_edge_segs is None:
                self.mesh_edge_segs = data["mesh_edge_segs"]

            # ── 리셋 감지 (iter 0이 다시 오면 새 최적화 시작) ────────────────
            if it == 0 and len(self.history["iter"]) > 0:
                self._clear_history()

            # ── 히스토리 축적 ────────────────────────────────────────────
            self.history["iter"].append(it)
            self.history["compliance"].append(data.get("compliance", 0.0))
            self.history["avg_h"].append(data.get("avg_h", 0.0))
            self.history["max_h"].append(data.get("max_h", 0.0))
            self.history["dx"].append(data.get("dx", 0.0))
            self.history["area_ratio"].append(data.get("area_ratio", 0.0))
            self.history["min_width"].append(data.get("min_width", 0.0))
            freqs = data.get("frequencies", [])
            self.history["frequencies"].append(freqs)

            # Modal Ref. 초기화 (최초 수신 시) — column 0 = "Ref. (Hz)"
            if self.ref_freqs is None and freqs:
                self.ref_freqs = freqs
                if self.modal_table:
                    n = self.modal_table.rowCount()
                    for i, f in enumerate(freqs):
                        if i < n:
                            self.modal_table.setItem(
                                i, 0, QTableWidgetItem(f"{f:.2f}")
                            )

            # ── 하중 케이스 등록 ─────────────────────────────────────────
            cases_data = data.get("cases", {})
            n_iters = len(self.history["iter"])
            if not self.case_names and cases_data:
                self.case_names = sorted(cases_data.keys())
                for name in self.case_names:
                    if self.metric_combo:
                        for pfx in ("U_", "Disp_", "Stress_"):
                            self.metric_combo.addItem(f"{pfx}{name}")
                    self.history["cases"][name] = {
                        "U": [], "max_disp": [], "max_stress": []
                    }
                    if self.iter_case_combo:
                        self.iter_case_combo.addItem(name)
                # Summary Table 헤더
                if self.table:
                    headers = ["Iter", "C_total", "Avg_h", "Max_h", "dx", "Area_Ratio", "Min_Width"]
                    for name in self.case_names:
                        headers += [f"U_{name}", f"D_{name}", f"S_{name}"]
                    self.table.setColumnCount(len(headers))
                    self.table.setHorizontalHeaderLabels(headers)

            for name, res in cases_data.items():
                if name not in self.history["cases"]:
                    self.history["cases"][name] = {
                        "U":          [0.0] * (n_iters - 1),
                        "max_disp":   [0.0] * (n_iters - 1),
                        "max_stress": [0.0] * (n_iters - 1),
                    }
                    if self.iter_case_combo:
                        # 중복 추가 방지
                        existing = [self.iter_case_combo.itemText(i)
                                    for i in range(self.iter_case_combo.count())]
                        if name not in existing:
                            self.iter_case_combo.addItem(name)
                self.history["cases"][name]["U"].append(res.get("U", 0.0))
                self.history["cases"][name]["max_disp"].append(res.get("max_disp", 0.0))
                self.history["cases"][name]["max_stress"].append(res.get("max_stress", 0.0))

            for name, hist in self.history["cases"].items():
                if len(hist["U"]) < n_iters:
                    hist["U"].append(0.0)
                    hist["max_disp"].append(0.0)
                    hist["max_stress"].append(0.0)

            # ── 스냅샷 저장 ──────────────────────────────────────────────
            coords  = data.get("coords")
            heights = data.get("heights")
            if coords is not None and heights is not None:
                self.height_snapshots.append({
                    "iter":    it,
                    "coords":  coords.copy(),
                    "heights": heights.copy(),
                })
                label_str = f"Iter {it:03d}"
                if self.height_iter_combo:
                    self.height_iter_combo.blockSignals(True)
                    self.height_iter_combo.insertItem(1, label_str)
                    self.height_iter_combo.blockSignals(False)

                if self.height_slider:
                    self.height_slider.blockSignals(True)
                    self.height_slider.setMaximum(it)
                    if self.height_iter_combo.currentIndex() == 0:
                        self.height_slider.setValue(it)
                    self.height_slider.blockSignals(False)

            # ── 테이블 행 추가 ────────────────────────────────────────────
            if self.table and self.table.columnCount() > 0:
                row = self.table.rowCount()
                self.table.insertRow(row)
                ar   = data.get("area_ratio", 0.0)
                vals = [str(it),
                        f"{data.get('compliance', 0):.3e}",
                        f"{data.get('avg_h', 0):.2f}",
                        f"{data.get('max_h', 0):.2f}",
                        f"{data.get('dx', 0):.4f}",
                        f"{ar:.3f}",
                        f"{data.get('min_width', 0):.1f}"]
                for col, v in enumerate(vals):
                    self.table.setItem(row, col, QTableWidgetItem(v))
                c_off = len(vals)
                for name in self.case_names:
                    res = cases_data.get(name, {"U": 0, "max_disp": 0, "max_stress": 0})
                    self.table.setItem(row, c_off,   QTableWidgetItem(f"{res.get('U',0):.2e}"))
                    self.table.setItem(row, c_off+1, QTableWidgetItem(f"{res.get('max_disp',0):.2f}"))
                    self.table.setItem(row, c_off+2, QTableWidgetItem(f"{res.get('max_stress',0):.1f}"))
                    c_off += 3
                self.table.scrollToBottom()

            # ── Modal 테이블 갱신: 이터레이션마다 열 추가 ────────────────────
            if self.modal_table and freqs:
                n_rows = self.modal_table.rowCount()
                col = self.modal_table.columnCount()
                self.modal_table.insertColumn(col)
                self.modal_table.setColumnWidth(col, 80)
                self.modal_table.setHorizontalHeaderItem(
                    col, QTableWidgetItem(f"Iter {it}")
                )
                for i, f in enumerate(freqs):
                    if i < n_rows:
                        self.modal_table.setItem(
                            i, col, QTableWidgetItem(f"{f:.2f}")
                        )
                # 가로 스크롤: 새로 추가된 컬럼이 보이도록 오른쪽 끝으로 이동
                self.modal_table.scrollToBottom()
                hbar = self.modal_table.horizontalScrollBar()
                if hbar is not None:
                    hbar.setValue(hbar.maximum())

            self._update_curves()
            self._update_height_plot()

            if self.status_label:
                self.status_label.setText(
                    f"Status: Running — Iter {it}  "
                    f"C={data.get('compliance', 0):.3e}"
                )
                self.status_label.setStyleSheet(
                    "font-weight:bold;color:#ffffff;font-size:13px;"
                )

        except Exception as e:
            print(f" -> [Monitor Error] {e}")

    # ────────────────────────────────────────────────────────────────────────
    # _clear_history
    # ────────────────────────────────────────────────────────────────────────

    def _register_dynamic_scenarios(self, snap_dir: str) -> None:
        """init.pkl 의 dynamic_scenarios 를 Load Case 콤보박스에 등록."""
        if not self.iter_case_combo:
            return
        try:
            init_path = Path(snap_dir) / "init.pkl"
            if not init_path.exists():
                return
            with open(init_path, "rb") as f:
                init = pickle.load(f)
            scenarios = init.get("dynamic_scenarios", [])
            existing = [self.iter_case_combo.itemText(i)
                        for i in range(self.iter_case_combo.count())]
            for scen in scenarios:
                label = f"Dynamic: {scen['name']}"
                if label not in existing:
                    self.iter_case_combo.addItem(label)
        except Exception:
            pass  # 로드 실패 시 무시

    def _clear_history(self):
        self.history = {
            "iter": [], "compliance": [], "avg_h": [], "max_h": [],
            "dx": [], "area_ratio": [], "min_width": [], "frequencies": [], "cases": {},
        }
        self.height_snapshots = []
        self.mesh_edge_segs   = None
        self.ref_freqs        = None
        self.case_names       = []

        if self.metric_combo:
            self.metric_combo.blockSignals(True)
            self.metric_combo.clear()
            self.metric_combo.addItems([
                "ALL (Normalized)", "Compliance", "Avg_h", "Max_h",
                "dx", "Area_Ratio", "Min_Width", "Natural Frequencies",
            ])
            self.metric_combo.blockSignals(False)
        if self.table:
            self.table.setRowCount(0); self.table.setColumnCount(0)
        if self.modal_table:
            n_rows = self.modal_table.rowCount()
            n_cols = self.modal_table.columnCount()
            # Iter 컬럼들 모두 초기화 (column 0 = "Ref. (Hz)" 는 유지)
            for i in range(n_rows):
                for c in range(1, n_cols):
                    self.modal_table.setItem(i, c, QTableWidgetItem(""))
        if self.height_iter_combo:
            self.height_iter_combo.blockSignals(True)
            self.height_iter_combo.clear()
            self.height_iter_combo.addItem("Latest")
            self.height_iter_combo.blockSignals(False)
        if self.iter_case_combo:
            self.iter_case_combo.blockSignals(True)
            self.iter_case_combo.clear()
            self.iter_case_combo.addItem("Modal Analysis")
            self.iter_case_combo.blockSignals(False)
        if self.height_slider:
            self.height_slider.blockSignals(True)
            self.height_slider.setMinimum(0)
            self.height_slider.setMaximum(0)
            self.height_slider.setValue(0)
            self.height_slider.blockSignals(False)
        if self.status_label:
            self.status_label.setText("Status: Running (Reset)")
            self.status_label.setStyleSheet("font-weight:bold;color:#ffffff;")
        if self.stop_btn:
            self.stop_btn.setEnabled(True)
            self.stop_btn.setText("Request STOP")
        for canvas in (self.curve_canvas, self.height_canvas):
            if canvas:
                canvas.ax.clear()
                if canvas is self.height_canvas and self._height_colorbar:
                    try: self._height_colorbar.remove()
                    except: pass
                    self._height_colorbar = None

    # ────────────────────────────────────────────────────────────────────────
    # Convergence Curve
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _freq_array(freq_history: list) -> np.ndarray:
        """history["frequencies"] → (n_iters, n_modes) 2-D 배열.
        빈 리스트(iter 0)와 길이가 다른 항목을 0으로 패딩 — 행 수는 iters와 일치."""
        if not freq_history:
            return np.empty((0, 0))
        max_len = max((len(r) for r in freq_history), default=0)
        if max_len == 0:
            return np.zeros((len(freq_history), 0))
        arr = np.zeros((len(freq_history), max_len))
        for i, r in enumerate(freq_history):
            arr[i, :len(r)] = r
        return arr

    def _update_curves(self):
        if not self.curve_canvas or not self.history["iter"]:
            return
        metric = self.metric_combo.currentText()
        ax     = self.curve_canvas.ax
        ax.clear()
        iters  = self.history["iter"]

        if metric == "ALL (Normalized)":
            metrics = [
                ("Compliance", self.history["compliance"], 'o-'),
                ("Avg_h",      self.history["avg_h"],      's-'),
                ("Max_h",      self.history["max_h"],      '^-'),
                ("dx",         self.history["dx"],         'd-'),
                ("Area_Ratio", self.history["area_ratio"], 'x-'),
                ("Min_Width",  self.history["min_width"],  'v-'),
            ]
            fmts = ['--', '-.', ':', '-', '--', '-.']
            for i, name in enumerate(self.case_names):
                u = self.history["cases"][name]["U"]
                if u:
                    metrics.append((f"U_{name}", u, fmts[i % len(fmts)]))
            RIGID = 0.5
            fa = self._freq_array(self.history["frequencies"])
            if fa.ndim == 2 and fa.shape[0] > 0 and fa.shape[1] > 0:
                cols = np.where(fa[-1] >= RIGID)[0]
                if len(cols) > 0:
                    metrics.append(("Freq_1", list(fa[:, cols[0]]), 'P-'))
            for label, lst, fmt in metrics:
                arr = np.array(lst)
                if len(arr) > 0:
                    mx = np.max(np.abs(arr))
                    ax.plot(iters, arr / mx if mx > 1e-12 else arr,
                            fmt, label=label, markersize=3, linewidth=1)
            ax.set_ylabel("Normalized Value")
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      fontsize=7, borderaxespad=0)
        elif metric == "Natural Frequencies":
            RIGID = 0.5
            fa = self._freq_array(self.history["frequencies"])
            if fa.ndim == 2 and fa.shape[1] > 0:
                cols = np.where(fa[-1] >= RIGID)[0]
                for rank, col in enumerate(cols[:10]):
                    ax.plot(iters, fa[:, col],
                            label=f"M{rank+1}({fa[-1, col]:.1f}Hz)")
                ax.legend(fontsize=8, loc='upper left',
                          bbox_to_anchor=(1, 1))
            ax.set_ylabel("Frequency (Hz)")
        elif metric == "Area_Ratio":
            ax.plot(iters, self.history["area_ratio"], 'D-',
                    color='darkorange', label="Area Ratio")
            ax.set_ylabel("Bead Area Ratio")
            ax.set_ylim(0, 1.05)
        elif metric in ("Compliance", "Avg_h", "Max_h", "dx", "Min_Width"):
            ax.plot(iters, self.history[metric.lower()], 'o-', label=metric)
            ax.set_ylabel(metric)
        else:
            for name in self.case_names:
                if metric.endswith(name):
                    pfx = metric.split('_')[0]
                    key = {"U": "U", "Disp": "max_disp", "Stress": "max_stress"}.get(pfx, "U")
                    ax.plot(iters, self.history["cases"][name][key], 's-', label=metric)
                    break

        ax.set_title(f"Optimization Trend: {metric}")
        ax.set_xlabel("Iteration")
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='-', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        self.curve_canvas.fig.tight_layout()
        self.curve_canvas.draw()

    # ────────────────────────────────────────────────────────────────────────
    # Height Distribution
    # ────────────────────────────────────────────────────────────────────────

    def _on_height_combo_changed(self, _idx):
        if self.height_slider and not self.height_slider.signalsBlocked():
            self.height_slider.blockSignals(True)
            mx = self.height_slider.maximum()
            self.height_slider.setValue(mx - _idx)
            self.height_slider.blockSignals(False)
        self._update_height_plot()

    def _on_height_slider_changed(self, val):
        if self.height_iter_combo and not self.height_iter_combo.signalsBlocked():
            self.height_iter_combo.blockSignals(True)
            mx = self.height_slider.maximum()
            self.height_iter_combo.setCurrentIndex(mx - val)
            self.height_iter_combo.blockSignals(False)
        self._update_height_plot()

    def _on_prev_height(self):
        idx = self.height_iter_combo.currentIndex()
        if idx < self.height_iter_combo.count() - 1:
            self.height_iter_combo.setCurrentIndex(idx + 1)

    def _on_next_height(self):
        idx = self.height_iter_combo.currentIndex()
        if idx > 0:
            self.height_iter_combo.setCurrentIndex(idx - 1)

    def _update_height_plot(self):
        if not self.height_canvas or not self.height_snapshots:
            return
        snap = self._get_snap(self.height_iter_combo)
        if snap is None:
            return
        coords  = snap["coords"]   # (M, 3) — 요소 도심
        heights = snap["heights"]  # (M,)
        it_lbl  = snap["iter"]

        ax  = self.height_canvas.ax
        fig = self.height_canvas.fig
        ax.clear()

        x, y = coords[:, 0], coords[:, 1]

        # 고정 컬러 범위 = ±h_max (--bead-height 값 고정, 이터레이션 무관)
        h_max      = float(snap.get("h_max", 15.0))   # 기본 15mm (bead-height 기본값)
        bead_steps = int(snap.get("bead_steps", 0))

        from scipy.interpolate import griddata
        from matplotlib.colors import BoundaryNorm
        from matplotlib import cm
        # 격자 해상도: 요소 수에 비례, 최소 64 — 흐릿함 방지
        n_grid = max(64, int(len(x) ** 0.5) * 3)
        xi = np.linspace(x.min(), x.max(), n_grid)
        yi = np.linspace(y.min(), y.max(), n_grid)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), heights, (Xi, Yi), method='linear')
        Zi_near = griddata((x, y), heights, (Xi, Yi), method='nearest')
        Zi = np.where(np.isnan(Zi), Zi_near, Zi)
        Zi = np.clip(Zi, -h_max, h_max)   # 보간 아티팩트가 범위 밖으로 나가지 않도록

        # 이산 colormap: 항상 ±h_max 대칭 범위, coolwarm 발산형
        # - 단방향(+): [0, h_max] 구간에 값 분포 → 파란쪽이 0(비드 없음)
        # - 단방향(-): [-h_max, 0] 구간에 값 분포
        # - 양방향:    [-h_max, h_max] 전체 범위 활용
        n_steps = bead_steps if bead_steps >= 1 else 5
        pos_levels  = np.linspace(0, h_max, n_steps + 1)
        full_levels = np.concatenate([-pos_levels[::-1], pos_levels[1:]])  # -(n_steps+1)레벨
        vmin, vmax  = -h_max, h_max

        boundaries = (full_levels[:-1] + full_levels[1:]) * 0.5
        boundaries = np.concatenate([[vmin - 1e-6], boundaries, [vmax + 1e-6]])
        cmap_disc = cm.get_cmap('coolwarm', len(full_levels))
        norm = BoundaryNorm(boundaries, ncolors=cmap_disc.N, clip=True)
        sc = ax.imshow(
            Zi, origin='lower', aspect='equal',
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap=cmap_disc, norm=norm,
            interpolation='nearest',
        )

        _cb_label = "Bead Height (mm)  [+: outward / −: inward]"

        if self._height_colorbar is None:
            self._height_colorbar = fig.colorbar(sc, ax=ax)
            self._height_colorbar.set_label(_cb_label)
        else:
            self._height_colorbar.update_normal(sc)
            self._height_colorbar.set_label(_cb_label)
        self._height_colorbar.set_ticks(full_levels)
        self._height_colorbar.set_ticklabels([f"{v:.1f}" for v in full_levels])

        ax.set_aspect('equal')
        ax.set_title(f"Height Distribution — Iter {it_lbl}")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        fig.tight_layout()
        self.height_canvas.draw()
        self.height_canvas.draw_idle()

    # ────────────────────────────────────────────────────────────────────────
    # Iteration Results — Mesh View / Run Analysis / Export
    # ────────────────────────────────────────────────────────────────────────

    def _set_iter_status(self, msg: str, color: str = "#555"):
        if self.iter_status_label:
            self.iter_status_label.setStyleSheet(
                f"color:{color}; font-size:11px; background:#1e1e1e; border:none;"
            )
            self.iter_status_label.setPlainText(msg)
            # 최신 내용이 보이도록 스크롤
            self.iter_status_label.verticalScrollBar().setValue(
                self.iter_status_label.verticalScrollBar().maximum()
            )

    def _get_selected_iter_num(self) -> int:
        """height_iter_combo 현재 선택 이터레이션 번호 반환 (Latest → 최신 snap의 iter)."""
        if not self.height_snapshots:
            return 0
        snap = self._get_snap(self.height_iter_combo)
        if snap is None:
            return 0
        return int(snap["iter"])

    def _open_visualizer(self, wht_result_data, title: str = "WHT Visualizer"):
        """WHTResultData를 선택된 뷰어로 표시합니다."""
        use_paraview = (hasattr(self, 'viewer_combo') and
                        self.viewer_combo.currentText() == "ParaView")
        try:
            if use_paraview:
                self._open_paraview(wht_result_data, title)
            else:
                from wht_visualizer.wht_visualizer import WHTVisualizer
                vis = WHTVisualizer(title=title, show=True)
                vis.show_result(wht_result_data)
                self._vis_window = vis   # GC 방지
        except Exception:
            self._set_iter_status(f"Visualizer 실행 오류: {traceback.format_exc()[:120]}", "red")

    def _open_paraview(self, wht_result_data, title: str = "result"):
        """WHTResultData를 VTKHDF로 내보내고 ParaView를 실행합니다."""
        import tempfile, re
        from wht_converter.wht_exporters import VTKHDFExporter
        from wht_visualizer.wht_visualizer import launch_paraview

        safe = re.sub(r'[^\w\-]', '_', title)[:48]
        tmp_dir = Path(tempfile.mkdtemp(prefix="wht_pv_"))
        hdf_path = str(tmp_dir / f"{safe}.hdf")
        VTKHDFExporter().export(wht_result_data, hdf_path)
        self._set_iter_status(f"ParaView HDF: {hdf_path}", "#2a7a2a")
        launch_paraview(hdf_path)

    def _on_mesh_view_clicked(self):
        """선택 이터레이션 메시(변형 좌표)를 WHTVisualizer로 표시합니다."""
        if not self.snap_dir:
            QMessageBox.warning(self, "경고", "스냅샷 디렉토리가 아직 수신되지 않았습니다.\n최적화가 시작되면 재시도하세요.")
            return
        iter_num = self._get_selected_iter_num()
        self._set_iter_status(f"Iter {iter_num} 메시 로드 중...", "blue")

        class _MeshWorker(QThread):
            finished = Signal(object)
            error    = Signal(str)

            def __init__(self, snap_dir, iter_num):
                super().__init__()
                self.snap_dir = snap_dir
                self.iter_num = iter_num

            def run(self):
                try:
                    from pathlib import Path
                    from wht_converter.wht_models import WHTMetadata
                    snap_dir = Path(self.snap_dir)
                    with open(snap_dir / "init.pkl", "rb") as f:
                        init = pickle.load(f)
                    model       = init["model"]
                    bead_dir    = init["bead_dir"]
                    design_nids = init["design_nids"]
                    aggr_src    = init["aggr_src"]
                    aggr_dst    = init["aggr_dst"]
                    orig_coords = init["orig_coords"]
                    for nid, (x, y, z) in orig_coords.items():
                        nd = model.nodes[nid]
                        nd.x, nd.y, nd.z = x, y, z
                    if self.iter_num > 0:
                        with open(snap_dir / f"iter_{self.iter_num:03d}.pkl", "rb") as f:
                            snap = pickle.load(f)
                        h_elem = snap["h_elem"]
                    else:
                        h_elem = np.zeros(len(init["design_elems"]))
                    n_int      = len(design_nids)
                    h_node_sum = np.zeros(n_int)
                    np.add.at(h_node_sum, aggr_src, h_elem[aggr_dst])
                    node_adj   = np.bincount(aggr_src, minlength=n_int)
                    h_node     = h_node_sum / (node_adj + 1e-12)
                    for i, nid in enumerate(design_nids):
                        ox, oy, oz = orig_coords[nid]
                        nd = model.nodes[nid]
                        nd.x = ox + float(h_node[i]) * bead_dir[0]
                        nd.y = oy + float(h_node[i]) * bead_dir[1]
                        nd.z = oz + float(h_node[i]) * bead_dir[2]
                    meta = WHTMetadata(
                        solver_name="WHT-Topo", solver_version="1.0",
                        analysis_type="mesh", coordinate_system="cartesian",
                        unit_length="mm", unit_force="N",
                    )
                    rd = model.to_wht_result_data(meta)
                    self.finished.emit(rd)
                except Exception:
                    self.error.emit(traceback.format_exc())

        worker = _MeshWorker(self.snap_dir, iter_num)
        worker.finished.connect(lambda rd: (
            self._set_iter_status(f"Iter {iter_num} 메시 표시 중...", "blue"),
            self._open_visualizer(rd, f"Mesh View — Iter {iter_num}"),
            self._set_iter_status(f"Iter {iter_num} 메시 표시 완료", "#2a7a2a"),
        ))
        worker.error.connect(lambda msg: (
            self._set_iter_status(f"오류: {msg[:80]}", "red"),
            QMessageBox.critical(self, "Mesh View 오류", msg[:400]),
        ))
        worker.start()
        self._re_worker = worker   # GC 방지

    def _on_run_calculix(self):
        if not self.snap_dir:
            QMessageBox.warning(self, "경고", "스냅샷 디렉토리가 아직 수신되지 않았습니다.\n최적화가 시작되면 재시도하세요.")
            return
        if self._re_worker and self._re_worker.isRunning():
            QMessageBox.information(self, "실행 중", "이전 해석이 아직 실행 중입니다.")
            return

        iter_num  = self._get_selected_iter_num()
        case_name = self.iter_case_combo.currentText() if self.iter_case_combo else "Modal Analysis"

        self._set_iter_status(f"CalculiX [Iter {iter_num} / {case_name}] 실행 중...", "blue")
        if self.iter_ccx_btn:
            self.iter_ccx_btn.setEnabled(False)

        self._re_worker = _CalculixReAnalysisWorker(self.snap_dir, iter_num, case_name,
                                                   num_modal_modes=self.num_modal_modes)
        self._re_worker.progress.connect(lambda msg: self._set_iter_status(msg, "white"))
        self._re_worker.finished.connect(self._on_calculix_analysis_finished)
        self._re_worker.error.connect(self._on_calculix_analysis_error)
        self._re_worker.start()

    def _on_calculix_analysis_finished(self, result: dict):
        if self.iter_ccx_btn:
            self.iter_ccx_btn.setEnabled(True)

        try:
            rtype = result.get("type")
            rd    = result["wht_result_data"]

            if rtype == "modal":
                freqs = [float(f) for f in rd.time_values]
                freq_str = "  ".join(f"f{i+1}={f:.2f}Hz" for i, f in enumerate(freqs))
                self._set_iter_status(f"CCX 모달: {freq_str}", "#2a7a2a")
                title = f"CalculiX Modal Analysis — Iter {self._get_selected_iter_num()}"
            else:
                lc_name = result.get("lc_name", "Static")
                disp = rd.point_data["Displacement"][0]
                u_max = float(np.max(np.abs(disp[:, :3])))
                self._set_iter_status(f"CCX {lc_name}  Max|U|={u_max:.4f} mm", "#2a7a2a")
                title = f"CalculiX Static: {lc_name} — Iter {self._get_selected_iter_num()}"

            self._open_visualizer(rd, title)

        except Exception:
            tb = traceback.format_exc()
            print(f"\n[Monitor] _on_calculix_analysis_finished 오류:\n{tb}", flush=True)
            self._set_iter_status(f"CCX 결과 변환 오류: {tb.splitlines()[-1][:160]}", "red")

    def _on_calculix_analysis_error(self, msg: str):
        if self.iter_ccx_btn:
            self.iter_ccx_btn.setEnabled(True)
        self._set_iter_status(f"CCX 오류: {msg[:120]}", "red")
        QMessageBox.critical(self, "CalculiX 해석 오류", msg[:400])

    def _on_run_analysis(self):
        if not self.snap_dir:
            QMessageBox.warning(self, "경고", "스냅샷 디렉토리가 아직 수신되지 않았습니다.\n최적화가 시작되면 재시도하세요.")
            return
        if self._re_worker and self._re_worker.isRunning():
            QMessageBox.information(self, "실행 중", "이전 해석이 아직 실행 중입니다.")
            return

        iter_num  = self._get_selected_iter_num()
        case_name = self.iter_case_combo.currentText() if self.iter_case_combo else "Modal Analysis"

        self._set_iter_status(f"Iter {iter_num} / {case_name} 해석 실행 중...", "blue")
        if self.iter_run_btn:
            self.iter_run_btn.setEnabled(False)
        if self.iter_ccx_btn:
            self.iter_ccx_btn.setEnabled(False)

        self._re_worker = _ReAnalysisWorker(self.snap_dir, iter_num, case_name,
                                             num_modal_modes=self.num_modal_modes)
        self._re_worker.progress.connect(lambda msg: self._set_iter_status(msg, "blue"))
        self._re_worker.finished.connect(self._on_analysis_finished)
        self._re_worker.error.connect(self._on_analysis_error)
        self._re_worker.start()

    def _on_analysis_finished(self, result: dict):
        if self.iter_run_btn:
            self.iter_run_btn.setEnabled(True)
        if self.iter_ccx_btn:
            self.iter_ccx_btn.setEnabled(True)

        rtype     = result.get("type")
        model     = result["model"]
        solver_result = result["result"]

        try:
            from wht_converter.wht_models import WHTMetadata

            if rtype == "dynamic":
                dyn       = solver_result   # DynamicResult
                scen_name = result.get("scen_name", "Dynamic")
                peak_idx  = dyn.peak_time_index()
                t_peak    = float(dyn.t_saved[peak_idx])
                u_peak    = dyn.u[peak_idx]   # (N, 6)
                u_max     = float(np.max(np.abs(u_peak[:, :3])))

                meta = WHTMetadata(
                    solver_name="WHT-Topo", solver_version="1.0",
                    analysis_type="transient", coordinate_system="cartesian",
                    unit_length="mm", unit_force="N",
                )
                rd = dyn.to_wht_result_data(meta, model)
                t_str = f"t={t_peak:.4f}s"
                self._set_iter_status(
                    f"{scen_name}  피크({t_str})  Max|U|={u_max:.4f} mm", "#2a7a2a")
                title = f"Dynamic peak: {scen_name} ({t_str})"
                self._open_visualizer(rd, title)
                return

            meta = WHTMetadata(
                solver_name="WHT-Topo", solver_version="1.0",
                analysis_type=rtype, coordinate_system="cartesian",
                unit_length="mm", unit_force="N",
            )
            rd = solver_result.to_wht_result_data(meta, model)

            if rtype == "modal":
                freqs = [float(f) for f in solver_result.frequencies]
                freq_str = "  ".join(f"f{i+1}={f:.2f}Hz" for i, f in enumerate(freqs))
                self._set_iter_status(f"모달: {freq_str}", "#2a7a2a")
                title = "Modal Analysis"
            else:
                lc_name = result.get("lc_name", "Static")
                disp = solver_result.displacement
                u_max = float(np.max(np.abs(disp[:, :3])))
                self._set_iter_status(f"{lc_name}  Max|U|={u_max:.4f} mm", "#2a7a2a")
                title = f"Static: {lc_name}"

            self._open_visualizer(rd, title)

        except Exception:
            tb = traceback.format_exc()
            print(f"\n[Monitor] _on_analysis_finished 오류:\n{tb}", flush=True)
            self._set_iter_status(f"결과 변환 오류: {tb.splitlines()[-1][:160]}", "red")

    def _on_analysis_error(self, msg: str):
        if self.iter_run_btn:
            self.iter_run_btn.setEnabled(True)
        if self.iter_ccx_btn:
            self.iter_ccx_btn.setEnabled(True)
        self._set_iter_status(f"오류: {msg[:120]}", "red")
        QMessageBox.critical(self, "해석 오류", msg[:400])

    def _on_export_optistruct(self):
        if not self.snap_dir:
            QMessageBox.warning(self, "경고", "스냅샷 디렉토리가 아직 수신되지 않았습니다.")
            return

        iter_num = self._get_selected_iter_num()
        default_name = f"topo_iter{iter_num:03d}.fem"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "OptiStruct .fem 저장",
            os.path.join(self.results_dir or ".", default_name),
            "OptiStruct FEM (*.fem);;All Files (*)"
        )
        if not out_path:
            return

        self._set_iter_status(f"Iter {iter_num} → {out_path} 저장 중...", "blue")
        try:
            result = _write_optistruct_fem(out_path, self.snap_dir, iter_num)
            self._set_iter_status(f"저장 완료: {result}", "#2a7a2a")
            QMessageBox.information(self, "내보내기 완료", f"저장 완료:\n{result}")
        except Exception as e:
            err = traceback.format_exc()
            self._set_iter_status(f"저장 오류: {str(e)[:80]}", "red")
            QMessageBox.critical(self, "내보내기 오류", err[:400])

    # ────────────────────────────────────────────────────────────────────────
    # 헬퍼
    # ────────────────────────────────────────────────────────────────────────

    def _get_snap(self, combo):
        """콤보박스 선택에 따라 스냅샷을 반환합니다."""
        if not self.height_snapshots:
            return None
        idx = combo.currentIndex() if combo else 0
        if idx == 0:
            # "Latest" → 가장 최근 스냅샷
            return self.height_snapshots[-1]
        # insertItem(1,…) 로 최신이 항상 idx=1 에 삽입됨
        # → idx=1: snapshots[n-1](최신), idx=n: snapshots[0](Initial)
        snap_idx = len(self.height_snapshots) - idx
        snap_idx = max(0, min(snap_idx, len(self.height_snapshots) - 1))
        return self.height_snapshots[snap_idx]


# ────────────────────────────────────────────────────────────────────────────
# IPC: Queue → Signal 브리지
# ────────────────────────────────────────────────────────────────────────────

class MonitorDataHandler(QObject):
    data_received = Signal(dict)


class _TeeStream:
    """stdout/stderr를 콘솔과 파일에 동시에 출력하는 스트림 래퍼."""
    def __init__(self, original, file_handle):
        self._orig = original
        self._file = file_handle

    def write(self, data):
        self._orig.write(data)
        try:
            self._file.write(data)
            self._file.flush()
        except Exception:
            pass

    def flush(self):
        self._orig.flush()
        try:
            self._file.flush()
        except Exception:
            pass

    def __getattr__(self, attr):
        return getattr(self._orig, attr)


def start_monitor_ui(queue, stop_event=None, results_dir: str = "",
                     num_modal_modes: int = 20):
    """
    별도 프로세스에서 PySide6 UI를 실행합니다.

    Parameters
    ----------
    queue            : multiprocessing.Queue  — 솔버 → 모니터 데이터 큐
    stop_event       : multiprocessing.Event  — GUI 종료 시 솔버에 중단 신호
    results_dir      : str                    — out_dir 경로 (OptiStruct 저장 기본 경로)
    num_modal_modes  : int                    — 모달 해석 모드 수 (기본: 10)
    """
    # ── 로그 파일 설정 (콘솔 + 파일 동시 출력) ─────────────────────────────
    _log_handle = None
    if results_dir:
        import datetime
        log_path = Path(results_dir) / "topopt_log.txt"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
            _log_handle.write(f"\n{'='*60}\n[LOG START] {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n{'='*60}\n")
            sys.stdout = _TeeStream(sys.stdout, _log_handle)
            sys.stderr = _TeeStream(sys.stderr, _log_handle)
            print(f"[Monitor] 로그 파일: {log_path}", flush=True)
        except Exception as _e:
            print(f"[Monitor] 로그 파일 열기 실패: {_e}", flush=True)

    app    = QApplication.instance() or QApplication(sys.argv)
    window = WHTMonitorWindow(stop_event=stop_event, results_dir=results_dir,
                              num_modal_modes=num_modal_modes)
    window.show()

    class Receiver(QThread):
        def __init__(self, q):
            super().__init__()
            self.q       = q
            self.handler = MonitorDataHandler()

        def run(self):
            import queue as _queue
            while True:
                try:
                    data = self.q.get(timeout=2.0)
                    if isinstance(data, str) and data == "STOP":
                        self.handler.data_received.emit({"status": "STOP"})
                        return
                    self.handler.data_received.emit(data)
                except _queue.Empty:
                    continue
                except Exception:
                    return

    receiver = Receiver(queue)
    receiver.handler.data_received.connect(window.update_data)
    receiver.start()
    _exit_code = app.exec()
    if _log_handle:
        try:
            import datetime
            sys.stdout = sys.stdout._orig if isinstance(sys.stdout, _TeeStream) else sys.stdout
            sys.stderr = sys.stderr._orig if isinstance(sys.stderr, _TeeStream) else sys.stderr
            _log_handle.write(f"[LOG END] {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
            _log_handle.close()
        except Exception:
            pass
    sys.exit(_exit_code)
