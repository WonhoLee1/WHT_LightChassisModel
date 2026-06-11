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
    QCheckBox, QListWidget, QDoubleSpinBox,
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
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'Segoe UI'


# ────────────────────────────────────────────────────────────────────────────
# Canvas helper
# ────────────────────────────────────────────────────────────────────────────

def _grid_res_from_elements(x, y):
    """요소 중심 좌표로부터 1 cell ≈ 1 element 가 되는 격자 해상도(nx, ny) 산출.

    요소 간 중앙 간격(median nearest spacing)을 셀 크기로 사용해, 30mm 요소를
    30mm 격자로 표시 → 가짜 sub-element 보간 제거.
    """
    xs = np.unique(np.round(x, 3))
    ys = np.unique(np.round(y, 3))
    # 고유 좌표 간 중앙 간격 (요소 피치 추정)
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 1.0
    dx = max(dx, 1e-6); dy = max(dy, 1e-6)
    nx = int(round((x.max() - x.min()) / dx)) + 1
    ny = int(round((y.max() - y.min()) / dy)) + 1
    # 안전 클램프 (과도/과소 방지)
    nx = max(16, min(nx, 400))
    ny = max(16, min(ny, 400))
    return nx, ny


def _app_icon():
    """resources/logo_icon_*.png 에서 가장 적합한 크기의 QIcon 반환."""
    from PySide6.QtGui import QIcon
    _base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
    # .ico는 모든 크기를 포함하므로 우선 사용
    ico = os.path.join(_base, "logo_icon.ico")
    if os.path.exists(ico):
        return QIcon(ico)
    # 없으면 크기별 PNG로 QIcon 구성
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        png = os.path.join(_base, f"logo_icon_{size}x{size}.png")
        if os.path.exists(png):
            icon.addFile(png)
    return icon


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

            # iter_num == -1: 이미 좌표가 외부에서 적용된 상태 (컨셉 평가 등)
            if self.iter_num == -1:
                h_elem = np.zeros(len(init["design_elems"]))
                load_cases_raw = init.get("static_load_cases", [])
                load_cases = [(lc.name, w, lc) for lc, w in load_cases_raw]
            else:
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

            # h_elem -> h_node → 좌표 적용
            # iter_num==-1: 외부에서 이미 좌표 적용 완료 → 덮어쓰기 금지
            if self.iter_num != -1:
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

def _write_optistruct_fem(out_path: str, snap_dir: str, iter_num: int,
                          h_elem_override=None) -> str:
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

    if h_elem_override is not None:
        h_elem = np.asarray(h_elem_override, dtype=float)
        load_cases_raw = init.get("static_load_cases", [])
        load_cases     = [(lc.name, w, lc) for lc, w in load_cases_raw]
    elif iter_num == 0:
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
        self.setWindowIcon(_app_icon())
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
        self._settings_table: "QTableWidget | None" = None
        self.table = self.modal_table = None
        self.metric_combo = self.height_iter_combo = None
        self._height_colorbar = None
        self.status_label = self.stop_btn = self.top_btn = self.reset_btn = None
        self.height_slider = self.btn_prev_h = self.btn_next_h = None
        self.height_interp_combo: "QComboBox | None" = None
        self.height_contour_check: "QCheckBox | None" = None
        self.height_toolbar = None
        self.concept_tool_btn = None
        self.iter_case_combo = None   # Load Case 콤보 (Iteration Results 탭)
        self.iter_solver_combo = None  # Solver 선택 콤보
        self.iter_run_btn = None      # Run Analysis 버튼
        self.iter_ccx_btn = None      # Run CalculiX (iter_run_btn alias)
        self.iter_export_btn     = None   # Export OptiStruct 버튼
        self.iter_export_ccx_btn = None   # Export CalculiX 버튼
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
        self._design_centroids = None
        init_data = {}
        if init_file.exists():
            try:
                with open(init_file, "rb") as f:
                    init_data = pickle.load(f)
                if "mesh_edge_segs" in init_data:
                    self.mesh_edge_segs = init_data["mesh_edge_segs"]
                if "run_settings" in init_data:
                    self._update_settings_tab(init_data["run_settings"])
                
                # 설계 요소 도심 좌표 계산하여 저장
                model = init_data.get("model")
                design_elems = init_data.get("design_elems", [])
                orig_coords = init_data.get("orig_coords", {})
                if model and design_elems and orig_coords:
                    self._design_centroids = np.array([
                        np.mean([orig_coords[nid] for nid in model.elements[eid].node_ids], axis=0)
                        for eid in design_elems
                    ], dtype=np.float32)
            except Exception as e:
                print(f"[Monitor] init.pkl 로드 실패: {e}")

        def get_hist_name(n, h_dict):
            if n in h_dict: return n
            if '_t' in n:
                base = n.split('_t')[0]
                for old_n in h_dict:
                    if old_n.startswith(base + '_t'):
                        return old_n
            return n

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
                    
                    freqs = snap.get("frequencies", [])
                    self.history["frequencies"].append(freqs)

                    if self.ref_freqs is None and freqs:
                        self.ref_freqs = freqs

                    # 케이스 명칭 및 개별 데이터 복원
                    cases_data = snap.get("cases", {})
                    if not cases_data and "load_cases" in snap:
                        # 구버전 스냅샷 파일 폴백
                        cases_data = {lc_name: {"U": 0.0, "max_disp": 0.0, "max_stress": 0.0}
                                      for lc_name, w, lc in snap["load_cases"]}

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

                    # 각 케이스별 데이터 누적
                    n_iters = len(self.history["iter"])
                    for raw_name, res in cases_data.items():
                        name = get_hist_name(raw_name, self.history["cases"])
                        if name not in self.history["cases"]:
                            self.history["cases"][name] = {
                                "U": [0.0] * (n_iters - 1),
                                "max_disp": [0.0] * (n_iters - 1),
                                "max_stress": [0.0] * (n_iters - 1)
                            }
                            if self.iter_case_combo:
                                existing = [self.iter_case_combo.itemText(idx_c)
                                            for idx_c in range(self.iter_case_combo.count())]
                                if name not in existing:
                                    self.iter_case_combo.addItem(name)
                        self.history["cases"][name]["U"].append(res.get("U", 0.0))
                        self.history["cases"][name]["max_disp"].append(res.get("max_disp", 0.0))
                        self.history["cases"][name]["max_stress"].append(res.get("max_stress", 0.0))

                    # 누락된 케이스 패딩
                    for name, hist in self.history["cases"].items():
                        if len(hist["U"]) < n_iters:
                            hist["U"].append(0.0)
                            hist["max_disp"].append(0.0)
                            hist["max_stress"].append(0.0)
                    
                    # 3D 뷰용 높이/좌표 스냅샷 복원
                    coords_data = snap.get("coords")
                    if coords_data is None or len(coords_data) == 0:
                        coords_data = self._design_centroids
                    
                    self.height_snapshots.append({
                        "iter": iter_num,
                        "coords": coords_data,
                        "heights": snap.get("h_elem", []),
                        "h_max": float(init_data.get("h_max", 15.0)) if init_file.exists() else 15.0,
                        "bead_steps": int(init_data.get("run_settings", {}).get("height_steps", 0)) if init_file.exists() else 0
                    })
            except Exception as e:
                print(f"[Monitor] {pkl_path} 로드 실패: {e}")

        # ── Summary Table 전체 복원 빌드 ──────────────────────────────────────────
        if self.table and self.history["iter"]:
            self.table.setRowCount(0)
            if self.case_names:
                headers = ["Iter", "C_total", "Avg_h", "Max_h", "dx", "Area_Ratio", "Min_Width"]
                for name in self.case_names:
                    headers += [f"U_{name}", f"D_{name}", f"S_{name}"]
                self.table.setColumnCount(len(headers))
                self.table.setHorizontalHeaderLabels(headers)
                
                for idx_i, it in enumerate(self.history["iter"]):
                    row = self.table.rowCount()
                    self.table.insertRow(row)
                    vals = [str(it),
                            f"{self.history['compliance'][idx_i]:.3e}",
                            f"{self.history['avg_h'][idx_i]:.2f}",
                            f"{self.history['max_h'][idx_i]:.2f}",
                            f"{self.history['dx'][idx_i]:.4f}",
                            f"{self.history['area_ratio'][idx_i]:.3f}",
                            f"{self.history['min_width'][idx_i]:.1f}"]
                    for col_idx, v in enumerate(vals):
                        self.table.setItem(row, col_idx, QTableWidgetItem(v))
                        
                    c_off = len(vals)
                    for name in self.case_names:
                        u_val    = self.history["cases"][name]["U"][idx_i]
                        d_val    = self.history["cases"][name]["max_disp"][idx_i]
                        s_val    = self.history["cases"][name]["max_stress"][idx_i]
                        self.table.setItem(row, c_off,   QTableWidgetItem(f"{u_val:.2e}"))
                        self.table.setItem(row, c_off+1, QTableWidgetItem(f"{d_val:.2f}"))
                        self.table.setItem(row, c_off+2, QTableWidgetItem(f"{s_val:.1f}"))
                        c_off += 3
                self.table.scrollToBottom()

        # ── Modal Table 전체 복원 빌드 ────────────────────────────────────────────
        if self.modal_table and self.history["iter"]:
            self.modal_table.setColumnCount(1)
            self.modal_table.setHorizontalHeaderLabels(["Ref. (Hz)"])
            if self.ref_freqs:
                for idx_f, f in enumerate(self.ref_freqs):
                    if idx_f < self.modal_table.rowCount():
                        self.modal_table.setItem(idx_f, 0, QTableWidgetItem(f"{f:.2f}"))
                        
            for idx_i, it in enumerate(self.history["iter"]):
                freqs = self.history["frequencies"][idx_i]
                if freqs:
                    col = self.modal_table.columnCount()
                    self.modal_table.insertColumn(col)
                    self.modal_table.setColumnWidth(col, 80)
                    self.modal_table.setHorizontalHeaderItem(col, QTableWidgetItem(f"Iter {it}"))
                    for idx_f, f in enumerate(freqs):
                        if idx_f < self.modal_table.rowCount():
                            self.modal_table.setItem(idx_f, col, QTableWidgetItem(f"{f:.2f}"))
            self.modal_table.scrollToBottom()

        # ── 콤보박스 및 슬라이더 복원 ─────────────────────────────────────────────
        if self.height_iter_combo and self.history["iter"]:
            self.height_iter_combo.blockSignals(True)
            self.height_iter_combo.clear()
            self.height_iter_combo.addItem("Latest")
            for it in reversed(self.history["iter"]):
                self.height_iter_combo.addItem(f"Iter {it:03d}")
            self.height_iter_combo.blockSignals(False)
            
        if self.height_slider and self.history["iter"]:
            self.height_slider.blockSignals(True)
            max_it = max(self.history["iter"])
            self.height_slider.setMinimum(0)
            self.height_slider.setMaximum(max_it)
            self.height_slider.setValue(max_it)
            self.height_slider.blockSignals(False)

        # ── 그래프 및 시각화 갱신 ────────────────────────────────────────────────
        self._update_curves()
        self._update_height_plot()


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

        self._build_tab_settings()
        self._build_tab_summary()
        self._build_tab_curve()
        self._build_tab_height()
        self._build_tab_modal()

    # ── 탭 빌더 ─────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # Settings 탭
    # ────────────────────────────────────────────────────────────────────────

    # 파라미터 메타 정보: (표시 이름, 단위, 설명)
    _SETTINGS_META = {
        "max_iter":               ("최대 이터레이션",    "iter", "최적화 반복 횟수 상한"),
        "h_max":                  ("최대 비드 높이",      "mm",   "비드가 돌출 또는 함입될 수 있는 최대 높이"),
        "bead_area":              ("비드 면적 비율",      "",     "전체 설계 영역 대비 비드 점유 목표 비율 (0~1)"),
        "bead_area_ramp":         ("면적 램프 이터",      "iter", "[타임라인 1단계] 면적 비율을 1.0→목표까지 줄이는 기간"),
        "min_width":              ("최소 비드 폭 (목표)", "mm",   "공간 필터 반경. 이 값 미만의 피처는 평활화됨"),
        "min_width_init":         ("최소 비드 폭 (초기)", "mm",   "필터 연속화 시작값. 크게 시작 후 min_width까지 감소"),
        "min_width_ramp":         ("폭 램프 이터",        "iter", "[타임라인 2단계] min_width_init→min_width 감소 기간"),
        "max_width":              ("최대 비드 폭",        "mm",   "넓은 비드 내부를 비우는 패널티 반경. -1=비활성"),
        "max_width_weight":       ("max_width 강도",      "×C₀",  "최대 폭 패널티 강도. 클수록 hollow 효과 강함"),
        "height_steps":           ("이산화 단계",         "단계", "비드 높이 양자화 수준. 2→{0,h/2,h} 3단계"),
        "beta_init":              ("β 시작값",            "",     "이산화 투사 β 초기값. 작을수록 초기 전이 부드러움"),
        "beta_max":               ("β 최대값",            "",     "이산화 투사 β 상한. 클수록 최종 경계 선명"),
        "beta_start_iter":        ("β 시작 이터",         "iter", "[타임라인 3단계] -1=bead_area_ramp 완료 후 자동"),
        "projection":             ("Heaviside β",         "",     "0=비활성. 활성 시 0.5 기준 이분화 강화"),
        "filter_type":            ("필터 유형",           "",     "linear=hat 커널 / gaussian=부드러운 덩어리"),
        "bidirectional":          ("양방향 비드",         "",     "True=±방향 동시 허용, False=단방향"),
        "sym_x":                  ("좌우 대칭",           "",     "True=X축 대칭 강제 적용"),
        "bead_connect":           ("비드 연결 갭",        "mm",   "이 거리 이하의 단절 비드를 자동 연결. 0=비활성"),
        "bead_connect_alg":       ("연결 알고리즘",       "",     "closing / mst / geodesic / hybrid"),
        "bead_connect_start_iter":("연결 시작 이터",      "iter", "[타임라인 4단계] -1=β 중간점 자동"),
        "obj_type":               ("목적함수 유형",       "",     "sum / max(softmax 최악) / sum+max"),
        "freq_weight":            ("진동수 패널티 강도",  "",     "0=비활성. 목표 진동수 이하일 때 패널티"),
        "freq_target":            ("진동수 목표",         "Hz",   "이 진동수 이상이 되도록 패널티 적용"),
        "n_nodes":                ("총 노드 수",          "개",   "FEM 모델 전체 노드 수"),
        "n_design_elems":         ("설계 요소 수",        "개",   "최적화 변수로 사용되는 설계 요소 수"),
    }

    def _build_tab_settings(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(6, 6, 6, 6)

        self._settings_table = QTableWidget(0, 4)
        self._settings_table.setHorizontalHeaderLabels(["파라미터", "값", "단위", "설명"])
        self._settings_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._settings_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._settings_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._settings_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._settings_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._settings_table.setAlternatingRowColors(True)
        self._settings_table.setStyleSheet(
            "QTableWidget { font-size: 11px; }"
            "QTableWidget::item { padding: 3px 6px; }"
        )
        lay.addWidget(self._settings_table)
        self.tabs.addTab(tab, "⚙ Settings")

    def _update_settings_tab(self, settings: dict):
        """run_settings dict를 받아 Settings 탭 테이블 갱신."""
        if not self._settings_table:
            return
        t = self._settings_table
        t.setRowCount(0)

        # 그룹 헤더 색상
        from PySide6.QtGui import QColor, QFont
        _GRP_COLOR = QColor("#2a2a3e")
        _GRP_FONT  = QFont(); _GRP_FONT.setBold(True)

        def _add_group(label: str):
            r = t.rowCount(); t.insertRow(r)
            item = QTableWidgetItem(f"  {label}")
            item.setBackground(_GRP_COLOR)
            item.setFont(_GRP_FONT)
            item.setForeground(QColor("#aabbff"))
            t.setItem(r, 0, item)
            for c in range(1, 4):
                g = QTableWidgetItem("")
                g.setBackground(_GRP_COLOR)
                t.setItem(r, c, g)
            t.setSpan(r, 0, 1, 4)

        def _add_row(key: str, value):
            meta = self._SETTINGS_META.get(key, (key, "", ""))
            name, unit, desc = meta
            r = t.rowCount(); t.insertRow(r)
            t.setItem(r, 0, QTableWidgetItem(f"  {name}"))
            # 값 포맷
            if isinstance(value, float):
                val_str = f"{value:.4g}"
            elif isinstance(value, bool):
                val_str = "✓ ON" if value else "OFF"
            else:
                val_str = str(value)
            val_item = QTableWidgetItem(val_str)
            val_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            t.setItem(r, 1, val_item)
            t.setItem(r, 2, QTableWidgetItem(unit))
            t.setItem(r, 3, QTableWidgetItem(desc))

        _add_group("■ 기본 설정")
        for k in ("max_iter", "h_max", "n_nodes", "n_design_elems"):
            if k in settings: _add_row(k, settings[k])

        _add_group("■ Continuation 전략  [1] 면적")
        for k in ("bead_area", "bead_area_ramp"):
            if k in settings: _add_row(k, settings[k])

        _add_group("■ Continuation 전략  [2] 폭")
        for k in ("min_width", "min_width_init", "min_width_ramp", "max_width", "max_width_weight"):
            if k in settings: _add_row(k, settings[k])

        _add_group("■ Continuation 전략  [3] 이산화 (β)")
        for k in ("height_steps", "beta_init", "beta_max", "beta_start_iter", "projection", "filter_type"):
            if k in settings: _add_row(k, settings[k])

        _add_group("■ Continuation 전략  [4] 연결")
        for k in ("bead_connect", "bead_connect_alg", "bead_connect_start_iter"):
            if k in settings: _add_row(k, settings[k])

        _add_group("■ 최적화 설정")
        for k in ("obj_type", "freq_weight", "freq_target", "bidirectional", "sym_x"):
            if k in settings: _add_row(k, settings[k])

    def _build_tab_summary(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self.table = QTableWidget(0, 0)
        self.table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.horizontalHeader().customContextMenuRequested.connect(self._on_table_header_context_menu)
        lay.addWidget(self.table)
        self.tabs.addTab(tab, "Summary Table")

    def _build_tab_curve(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        
        self.curve_canvas = PlotCanvas(tab)
        self.curve_toolbar = NavigationToolbar(self.curve_canvas, tab)
        from PySide6.QtCore import QSize
        _sz = self.curve_toolbar.iconSize()
        self.curve_toolbar.setIconSize(QSize(int(_sz.width() * 0.6), int(_sz.height() * 0.6)))
        lay.addWidget(self.curve_toolbar)
        
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
        lay.addWidget(self.curve_canvas)
        self.tabs.addTab(tab, "Convergence Curve")

    def _build_tab_height(self):
        tab = QWidget()
        lay = QVBoxLayout(tab)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        self.height_canvas = PlotCanvas(tab)
        self.height_toolbar = NavigationToolbar(self.height_canvas, tab)
        from PySide6.QtCore import QSize
        _sz = self.height_toolbar.iconSize()
        self.height_toolbar.setIconSize(QSize(int(_sz.width() * 0.6), int(_sz.height() * 0.6)))
        lay.addWidget(self.height_toolbar)

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

        # 보간 방식 선택
        ctrl_iter.addWidget(QLabel("  Interp:"))
        self.height_interp_combo = QComboBox()
        self.height_interp_combo.addItems(["linear", "nearest"])
        self.height_interp_combo.setFixedWidth(80)
        self.height_interp_combo.currentIndexChanged.connect(self._update_height_plot)
        ctrl_iter.addWidget(self.height_interp_combo)

        # 등고선 표시 여부
        self.height_contour_check = QCheckBox("Contour")
        self.height_contour_check.setChecked(False)
        self.height_contour_check.stateChanged.connect(self._update_height_plot)
        ctrl_iter.addWidget(self.height_contour_check)

        ctrl_iter.addStretch()
        lay.addWidget(ctrl_iter_w)

        # ── 캔버스 + 하단 패널을 QSplitter로 분리 (드래그로 크기 조절) ──
        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        lay.addWidget(splitter, stretch=1)

        # 상단: 높이 분포 캔버스
        canvas_container = QWidget()
        canvas_vlay = QVBoxLayout(canvas_container)
        canvas_vlay.setContentsMargins(0, 0, 0, 0)
        canvas_vlay.setSpacing(0)
        canvas_vlay.addWidget(self.height_canvas)
        splitter.addWidget(canvas_container)

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

        self.iter_stats_btn = QPushButton("View Analysis Results")
        self.iter_stats_btn.setStyleSheet("font-weight:bold;")
        self.iter_stats_btn.setToolTip("통계 결과(응력, 변형률, 변위, 에너지 등)를 텍스트로 확인")
        self.iter_stats_btn.clicked.connect(self._on_view_analysis_results)
        ctrl_action.addWidget(self.iter_stats_btn)

        ctrl_action.addWidget(QLabel("  Load Case:"))
        self.iter_case_combo = QComboBox(); self.iter_case_combo.setMinimumWidth(160)
        self.iter_case_combo.setMaxVisibleItems(25)
        self.iter_case_combo.addItem("Modal Analysis")
        ctrl_action.addWidget(self.iter_case_combo)

        ctrl_action.addWidget(QLabel("  Solver:"))
        self.iter_solver_combo = QComboBox()
        self.iter_solver_combo.addItems(["WHT Solver", "CalculiX"])
        self.iter_solver_combo.setFixedWidth(100)
        ctrl_action.addWidget(self.iter_solver_combo)

        self.iter_run_btn = QPushButton("Run Analysis")
        self.iter_run_btn.setStyleSheet("font-weight:bold;")
        self.iter_run_btn.setToolTip("선택 솔버로 해석 후 결과를 visualizer에 표시")
        self.iter_run_btn.clicked.connect(self._on_run_analysis_dispatch)
        ctrl_action.addWidget(self.iter_run_btn)

        # 하위 호환용 — 직접 접근하는 코드가 있을 수 있으므로 참조 유지
        self.iter_ccx_btn = self.iter_run_btn

        self.iter_export_btn = QPushButton("Export OptiStruct")
        self.iter_export_btn.setStyleSheet("font-weight:bold;")
        self.iter_export_btn.setToolTip("선택 이터레이션의 변형 메시 + 하중 케이스를 .fem 파일로 저장")
        self.iter_export_btn.clicked.connect(self._on_export_optistruct)
        ctrl_action.addWidget(self.iter_export_btn)

        self.iter_export_ccx_btn = QPushButton("Export CalculiX")
        self.iter_export_ccx_btn.setStyleSheet("font-weight:bold;")
        self.iter_export_ccx_btn.setToolTip("선택 이터레이션의 변형 메시를 CalculiX .inp 파일로 저장")
        self.iter_export_ccx_btn.clicked.connect(self._on_export_calculix)
        ctrl_action.addWidget(self.iter_export_ccx_btn)

        ctrl_action.addWidget(QLabel("  Visualizer:"))
        self.viewer_combo = QComboBox()
        self.viewer_combo.addItems(["WHTVisualizer", "ParaView"])
        self.viewer_combo.setToolTip("해석 결과 표시 방식 선택\nWHTVisualizer: PyVista 내장 뷰어\nParaView: VTKHDF 내보내기 후 ParaView 실행")
        ctrl_action.addWidget(self.viewer_combo)

        self.concept_tool_btn = QPushButton("Concept Tool")
        self.concept_tool_btn.setStyleSheet(
            "font-weight:bold; background:#4a235a; color:white;")
        self.concept_tool_btn.setToolTip("현재 비드 상태 위에 직접 비드를 그려 FEA 평가")
        self.concept_tool_btn.clicked.connect(self._on_concept_tool)
        ctrl_action.addWidget(self.concept_tool_btn)

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
        if "run_settings" in data:
            self._update_settings_tab(data["run_settings"])
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

            def get_hist_name(n, h_dict):
                if n in h_dict: return n
                if '_t' in n:
                    base = n.split('_t')[0]
                    for old_n in h_dict:
                        if old_n.startswith(base + '_t'):
                            return old_n
                return n

            for raw_name, res in cases_data.items():
                name = get_hist_name(raw_name, self.history["cases"])
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
                def get_case_res(n, c_dict):
                    if n in c_dict: return c_dict[n]
                    if '_t' in n:
                        base = n.split('_t')[0]
                        for new_n, val in c_dict.items():
                            if new_n.startswith(base + '_t'):
                                return val
                    return {"U": 0, "max_disp": 0, "max_stress": 0}

                c_off = len(vals)
                for name in self.case_names:
                    res = get_case_res(name, cases_data)
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
        interp_method = (self.height_interp_combo.currentText()
                         if self.height_interp_combo else "linear")
        show_contour  = (self.height_contour_check.isChecked()
                         if self.height_contour_check else False)
        # 격자 해상도: 요소 크기에 맞춤 (1 cell ≈ 1 element) → 가짜 보간 제거
        nx_g, ny_g = _grid_res_from_elements(x, y)
        xi = np.linspace(x.min(), x.max(), nx_g)
        yi = np.linspace(y.min(), y.max(), ny_g)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), heights, (Xi, Yi), method=interp_method)
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

        if show_contour:
            ax.contour(Xi, Yi, Zi, levels=full_levels,
                       colors='k', linewidths=0.5, alpha=0.6)

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

    def _update_iter_status_white(self, msg: str):
        self._set_iter_status(msg, "white")
        
    def _update_iter_status_blue(self, msg: str):
        self._set_iter_status(msg, "blue")

    def _on_mesh_worker_finished(self, rd):
        iter_num = self._get_selected_iter_num()
        self._set_iter_status(f"Iter {iter_num} 메시 표시 중...", "blue")
        self._open_visualizer(rd, f"Mesh View — Iter {iter_num}")
        self._set_iter_status(f"Iter {iter_num} 메시 표시 완료", "#2a7a2a")

    def _on_mesh_worker_error(self, msg: str):
        self._set_iter_status(f"오류: {msg[:80]}", "red")
        QMessageBox.critical(self, "Mesh View 오류", msg[:400])

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
        from wht_visualizer.wht_visualizer import launch_paraview, paraview_warp_vector

        safe = re.sub(r'[^\w\-]', '_', title)[:48]
        tmp_dir = Path(tempfile.mkdtemp(prefix="wht_pv_"))
        hdf_path = str(tmp_dir / f"{safe}.hdf")
        VTKHDFExporter().export(wht_result_data, hdf_path)
        self._set_iter_status(f"ParaView HDF: {hdf_path}", "#2a7a2a")
        # 정적(미변형) 결과는 변위로 자동 Warp, 모달/동적(이미 변형됨)은 생략
        warp_vec = paraview_warp_vector(wht_result_data, transient_geometry=True)
        launch_paraview(hdf_path, warp_vector=warp_vec)

    def _on_view_analysis_results(self):
        """해당 이터레이션의 해석 통계 결과를 텍스트 박스로 보여주는 다이얼로그 표시"""
        stats_path = Path(self.results_dir) / "iter_stats.json"
        if not stats_path.exists():
            QMessageBox.warning(self, "경고", "해석 통계 결과 파일(iter_stats.json)이 없습니다.\n최적화가 1회 이상 진행되어야 합니다.")
            return
        
        try:
            import json
            with open(stats_path, "r", encoding="utf-8") as f:
                iter_stats_data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "오류", f"통계 결과 로드 실패: {e}")
            return
            
        current_iter = self._get_selected_iter_num()
        if current_iter == -1 and self.history:
            current_iter = self.history[-1]["iter"]
            
        dlg = _AnalysisResultsDialog(self, iter_stats_data, current_iter)
        dlg.exec()

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
        worker.finished.connect(self._on_mesh_worker_finished)
        worker.error.connect(self._on_mesh_worker_error)
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
        self._re_worker.progress.connect(self._update_iter_status_white)
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

    def _on_run_analysis_dispatch(self):
        """Solver 콤보 선택에 따라 WHT Solver 또는 CalculiX로 분기."""
        solver = self.iter_solver_combo.currentText() if self.iter_solver_combo else "WHT Solver"
        if solver == "CalculiX":
            self._on_run_calculix()
        else:
            self._on_run_analysis()

    def _on_concept_tool(self):
        if not self.height_snapshots:
            QMessageBox.information(self, "Concept Tool",
                                    "먼저 최적화를 실행하여 비드 결과가 생성된 후 사용 가능합니다.")
            return
        snap = self._get_snap(self.height_iter_combo)
        if snap is None:
            return
        cases = [self.iter_case_combo.itemText(i)
                 for i in range(self.iter_case_combo.count())]
        dlg = BeadConceptDialog(snap, self.snap_dir, cases, parent=self)
        dlg.show()

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
        self._re_worker.progress.connect(self._update_iter_status_blue)
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

    def _on_export_calculix(self):
        """선택 이터레이션 → CalculiX .inp 내보내기 (Run CalculiX 재활용)."""
        if not self.snap_dir:
            QMessageBox.warning(self, "경고", "스냅샷 디렉토리가 아직 수신되지 않았습니다.")
            return
        if self._re_worker and self._re_worker.isRunning():
            QMessageBox.information(self, "실행 중", "이전 해석이 아직 실행 중입니다.")
            return
        iter_num  = self._get_selected_iter_num()
        case_name = self.iter_case_combo.currentText() if self.iter_case_combo else "Modal Analysis"
        self._set_iter_status(
            f"CalculiX Export [Iter {iter_num} / {case_name}] 실행 중...", "blue")
        if self.iter_export_ccx_btn:
            self.iter_export_ccx_btn.setEnabled(False)
        self._re_worker = _CalculixReAnalysisWorker(
            self.snap_dir, iter_num, case_name,
            num_modal_modes=self.num_modal_modes)
        self._re_worker.progress.connect(self._update_iter_status_white)
        self._re_worker.finished.connect(self._on_calculix_analysis_finished)
        self._re_worker.error.connect(self._on_calculix_analysis_error)
        self._re_worker.finished.connect(
            lambda _: (self.iter_export_ccx_btn.setEnabled(True)
                       if self.iter_export_ccx_btn else None))
        self._re_worker.start()

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


# ─────────────────────────────────────────────────────────────────────────────
# Bead Concept Evaluation Tool — 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _seg_dist_arr(xs, ys, ax, ay, bx, by):
    """(N,) 점 배열과 선분 (ax,ay)-(bx,by) 사이 최소 거리."""
    dx, dy = bx - ax, by - ay
    l2 = dx*dx + dy*dy
    if l2 < 1e-12:
        return np.hypot(xs - ax, ys - ay)
    t = np.clip(((xs - ax)*dx + (ys - ay)*dy) / l2, 0.0, 1.0)
    return np.hypot(xs - ax - t*dx, ys - ay - t*dy)


def _shape_mask(coords2d, shape):
    """도형 안에 드는 요소 도심 Boolean 마스크 반환."""
    xs, ys = coords2d[:, 0], coords2d[:, 1]
    pts = shape["pts"]; w = shape["width"]; t = shape["type"]
    x1, y1 = pts[0]; x2, y2 = pts[1]

    if t == "Line":
        return _seg_dist_arr(xs, ys, x1, y1, x2, y2) < w / 2.0

    elif t in ("Filled Rect", "Outline Rect"):
        xlo, xhi = min(x1, x2), max(x1, x2)
        ylo, yhi = min(y1, y2), max(y1, y2)
        outer = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
        if t == "Filled Rect":
            return outer
        xlo_i, xhi_i, ylo_i, yhi_i = xlo+w, xhi-w, ylo+w, yhi-w
        if xlo_i >= xhi_i or ylo_i >= yhi_i:
            return outer
        inner = (xs >= xlo_i) & (xs <= xhi_i) & (ys >= ylo_i) & (ys <= yhi_i)
        return outer & ~inner

    elif t in ("Filled Circle", "Outline Circle"):
        r = np.hypot(x2 - x1, y2 - y1)
        d = np.hypot(xs - x1, ys - y1)
        if t == "Filled Circle":
            return d < r
        return (d >= r - w/2) & (d < r + w/2)

    return np.zeros(len(xs), dtype=bool)


def _add_shape_artist(ax, shape_type, p1, p2, width, color, alpha=0.6):
    """ax에 도형 아티스트를 추가하고 반환."""
    import matplotlib.patches as mp
    x1, y1 = p1; x2, y2 = p2
    if shape_type == "Line":
        lw = max(1.0, width / 8.0)
        art, = ax.plot([x1, x2], [y1, y2], '-', color=color, alpha=alpha, linewidth=lw)
        return art
    elif shape_type == "Filled Rect":
        xlo, xhi = min(x1,x2), max(x1,x2); ylo, yhi = min(y1,y2), max(y1,y2)
        art = mp.Rectangle((xlo,ylo), xhi-xlo, yhi-ylo,
                            linewidth=0, edgecolor='none', facecolor=color, alpha=alpha)
        ax.add_patch(art); return art
    elif shape_type == "Outline Rect":
        xlo, xhi = min(x1,x2), max(x1,x2); ylo, yhi = min(y1,y2), max(y1,y2)
        lw = max(1.5, width / 5.0)
        art = mp.Rectangle((xlo,ylo), xhi-xlo, yhi-ylo,
                            linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha)
        ax.add_patch(art); return art
    elif shape_type == "Filled Circle":
        r = np.hypot(x2-x1, y2-y1)
        art = mp.Circle((x1,y1), r, linewidth=0, edgecolor='none', facecolor=color, alpha=alpha)
        ax.add_patch(art); return art
    elif shape_type == "Outline Circle":
        r = np.hypot(x2-x1, y2-y1); lw = max(1.5, width/5.0)
        art = mp.Circle((x1,y1), r, linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha)
        ax.add_patch(art); return art
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BeadConceptEvalWorker
# ─────────────────────────────────────────────────────────────────────────────

class BeadConceptEvalWorker(QThread):
    """개념 비드 높이 배열을 받아 FEA 해석 후 iter 0 대비 % 결과를 반환."""
    finished = Signal(list)
    error    = Signal(str)
    progress = Signal(str)

    def __init__(self, snap_dir: str, h_concept, case_names: list, parent=None):
        super().__init__(parent)
        self.snap_dir  = snap_dir
        self.h_concept = h_concept
        self.cases     = case_names

    def run(self):
        try:
            import numpy as np
            from pathlib import Path
            snap_dir = Path(self.snap_dir)

            self.progress.emit("init.pkl 로드 중...")
            with open(snap_dir / "init.pkl", "rb") as f:
                init = pickle.load(f)

            model       = init["model"]
            bead_dir    = np.array(init["bead_dir"])
            design_nids = init["design_nids"]
            aggr_src    = init["aggr_src"]
            aggr_dst    = init["aggr_dst"]
            orig_coords = init["orig_coords"]
            static_lcs  = init.get("static_load_cases", [])

            def _apply(h_elem):
                for nid, (x, y, z) in orig_coords.items():
                    nd = model.nodes[nid]; nd.x, nd.y, nd.z = x, y, z
                n_int = len(design_nids)
                hs = np.zeros(n_int)
                np.add.at(hs, aggr_src, h_elem[aggr_dst])
                adj = np.bincount(aggr_src, minlength=n_int)
                hn = hs / (adj + 1e-12)
                for i, nid in enumerate(design_nids):
                    ox, oy, oz = orig_coords[nid]; nd = model.nodes[nid]
                    nd.x = ox + float(hn[i])*bead_dir[0]
                    nd.y = oy + float(hn[i])*bead_dir[1]
                    nd.z = oz + float(hn[i])*bead_dir[2]

            from wht_solver.wht_solver import WHTSolver

            # iter 0 기준 해석
            self.progress.emit("기준(iter 0) 해석 중...")
            _apply(np.zeros(len(self.h_concept)))
            fea0 = WHTSolver(model)
            baselines = {}
            for case in self.cases:
                if case == "Modal Analysis":
                    r0 = fea0.solve_modal(num_modes=20)
                    baselines[case] = ("modal", r0.frequencies[0] if len(r0.frequencies) > 0 else 1e-6)
                else:
                    lc0 = next((lc for lc, _ in static_lcs if lc.name == case), None)
                    if lc0:
                        r0 = fea0.solve_static(lc0)
                        u0 = np.max(np.abs(np.array(r0.displacement)[:, :3]))
                        baselines[case] = ("static", float(u0))

            # 개념 비드 해석
            self.progress.emit("개념 비드 해석 중...")
            _apply(self.h_concept)
            fea = WHTSolver(model)
            results = []
            for case in self.cases:
                self.progress.emit(f"해석: {case}")
                try:
                    if case == "Modal Analysis":
                        r  = fea.solve_modal(num_modes=20)
                        freqs = list(r.frequencies) if len(r.frequencies) > 0 else []
                        f1 = freqs[0] if freqs else 0.0
                        b_entry = baselines.get(case)
                        if b_entry is None:
                            results.append({"type":"error","case":case,"msg":"iter0 기준값 계산 실패"}); continue
                        b = b_entry[1]
                        # effective mass
                        eff_mass_data = None
                        try:
                            eff_mass, total_mass = r.calculate_effective_mass()
                            if eff_mass is not None:
                                eff_mass_data = {"eff_mass": eff_mass, "total_mass": total_mass}
                        except Exception:
                            pass
                        # WHTResultData for visualizer
                        rd = None
                        try:
                            from wht_converter.wht_adapters import JaxSSOAdapter
                            from wht_converter.wht_exporters import WHTMetadata
                            meta = WHTMetadata(solver_name="WHT-Concept", solver_version="1.0",
                                              analysis_type="modal", coordinate_system="cartesian",
                                              unit_length="mm", unit_force="N")
                            rd = r.to_wht_result_data(meta, model)
                        except Exception:
                            pass
                        results.append({"type":"modal","case":case,"f1":f1,"freqs":freqs,
                                        "base":b,"delta_pct":(f1/(b+1e-12)-1)*100,
                                        "eff_mass_data":eff_mass_data,"wht_result_data":rd})
                    else:
                        lc = next((l for l, _ in static_lcs if l.name == case), None)
                        if lc is None:
                            results.append({"type":"error","case":case,"msg":"하중케이스 없음"}); continue
                        r    = fea.solve_static(lc)
                        umax = float(np.max(np.abs(np.array(r.displacement)[:, :3])))
                        b_entry = baselines.get(case)
                        if b_entry is None:
                            results.append({"type":"error","case":case,"msg":"iter0 기준값 계산 실패"}); continue
                        b = b_entry[1]
                        rd = None
                        try:
                            from wht_converter.wht_exporters import WHTMetadata
                            meta = WHTMetadata(solver_name="WHT-Concept", solver_version="1.0",
                                              analysis_type="static", coordinate_system="cartesian",
                                              unit_length="mm", unit_force="N")
                            rd = r.to_wht_result_data(meta, model)
                        except Exception:
                            pass
                        results.append({"type":"static","case":case,"u_max":umax,"base":b,
                                        "delta_pct":(umax/(b+1e-12)-1)*100,
                                        "wht_result_data":rd})
                except Exception as e:
                    results.append({"type":"error","case":case,"msg":str(e)})

            self.finished.emit(results)
        except Exception:
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# _ConceptCcxWorker — CalculiX 해석 (컨셉 비드 높이 적용)
# ─────────────────────────────────────────────────────────────────────────────

class _ConceptCcxWorker(QThread):
    """컨셉 비드 높이를 적용한 후 CalculiX 해석을 실행합니다."""
    finished = Signal(dict)
    error    = Signal(str)
    progress = Signal(str)

    def __init__(self, snap_dir: str, h_concept, case_name: str, parent=None):
        super().__init__(parent)
        self.snap_dir  = snap_dir
        self.h_concept = h_concept
        self.case_name = case_name

    def run(self):
        try:
            import numpy as np
            from pathlib import Path

            # ── 사전 점검 ────────────────────────────────────────────────────
            if run_calculix_analysis is None:
                self.error.emit(
                    "AutoCalculix API를 임포트하지 못했습니다.\n"
                    "D:/PythonCodeStudy/AutoCalculix 경로를 점검하세요.\n"
                    "WHT Solver 선택 후 재시도하거나 CalculiX를 설치하세요.")
                return

            snap_dir = Path(self.snap_dir)

            self.progress.emit("init.pkl 로드 중...")
            with open(snap_dir / "init.pkl", "rb") as f:
                init = pickle.load(f)

            model       = init["model"]
            bead_dir    = np.array(init["bead_dir"])
            design_nids = init["design_nids"]
            aggr_src    = init["aggr_src"]
            aggr_dst    = init["aggr_dst"]
            orig_coords = init["orig_coords"]

            # 로드케이스 존재 여부 사전 확인
            static_lcs = init.get("static_load_cases", [])
            if self.case_name != "Modal Analysis":
                lc_names = [lc.name for lc, _ in static_lcs]
                if self.case_name not in lc_names:
                    self.error.emit(
                        f"하중케이스 '{self.case_name}'를 init.pkl에서 찾을 수 없습니다.\n"
                        f"사용 가능한 케이스: {lc_names}")
                    return

            # 원본 좌표 복원 + 컨셉 비드 높이 적용
            for nid, (x, y, z) in orig_coords.items():
                nd = model.nodes[nid]; nd.x, nd.y, nd.z = x, y, z
            n_int = len(design_nids)
            hs = np.zeros(n_int)
            np.add.at(hs, aggr_src, self.h_concept[aggr_dst])
            adj = np.bincount(aggr_src, minlength=n_int)
            hn = hs / (adj + 1e-12)
            for i, nid in enumerate(design_nids):
                ox, oy, oz = orig_coords[nid]; nd = model.nodes[nid]
                nd.x = ox + float(hn[i]) * bead_dir[0]
                nd.y = oy + float(hn[i]) * bead_dir[1]
                nd.z = oz + float(hn[i]) * bead_dir[2]

            # _CalculixReAnalysisWorker 의 내부 run() 로직을 직접 호출
            ccx_worker = _CalculixReAnalysisWorker(
                str(snap_dir), 0, self.case_name)
            ccx_worker._model_override = model   # 이미 좌표 적용된 모델 주입

            # _CalculixReAnalysisWorker 의 run 을 직접 실행 (모델은 이미 변형됨)
            # iter_num=0 → h_elem=0 경로로 진입하지만 model은 이미 mutated 상태
            # → 좌표 복원/재적용 대신 현재 좌표를 그대로 사용
            ccx_worker.snap_dir  = str(snap_dir)
            ccx_worker.iter_num  = -1           # 특수값: 좌표 복원 스킵

            # 시그널 연결 후 동기 실행
            _result_holder = []
            _error_holder  = []
            ccx_worker.finished.connect(lambda r: _result_holder.append(r))
            ccx_worker.error.connect(lambda e: _error_holder.append(e))
            ccx_worker.progress.connect(self.progress.emit)
            ccx_worker.run()   # 동기 실행 (이미 별도 QThread 내)

            if _error_holder:
                self.error.emit(_error_holder[0]); return
            if _result_holder:
                self.finished.emit(_result_holder[0])
        except Exception:
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Bead Concept Dialog
# ─────────────────────────────────────────────────────────────────────────────

class BeadConceptDialog(QDialog):
    """비드 컨셉 평가 도구 — 비드를 직접 그려 FEA 평가."""

    _SHAPE_TYPES = ["Line", "Filled Rect", "Outline Rect", "Filled Circle", "Outline Circle"]

    def __init__(self, snap: dict, snap_dir: str, load_cases: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bead Concept Evaluation Tool")
        self.setWindowIcon(_app_icon())
        self.resize(1100, 720)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        self.snap       = snap
        self.snap_dir   = snap_dir
        self.load_cases = load_cases if load_cases else ["Modal Analysis"]

        self._shapes: list   = []       # 그려진 도형 목록
        self._pts_buf: list  = []       # 현재 진행 중 점 버퍼
        self._preview: list  = []       # preview matplotlib 아티스트
        self._cid_press = self._cid_move = None
        self._worker: "BeadConceptEvalWorker | None" = None
        self._shape_id_counter = 0

        self._build_ui()
        self._refresh_canvas()

    # ── UI 구성 ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        from PySide6.QtCore import QSize as _QS
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(4)

        # ── 상단: 도형 컨트롤 행 ──────────────────────────────────────────
        top = QHBoxLayout()
        top.addWidget(QLabel("Shape:"))
        self._shape_combo = QComboBox()
        self._shape_combo.addItems(self._SHAPE_TYPES)
        self._shape_combo.setFixedWidth(130)
        top.addWidget(self._shape_combo)

        top.addWidget(QLabel("  Width(mm):"))
        self._width_spin = QDoubleSpinBox()
        self._width_spin.setRange(1.0, 1000.0); self._width_spin.setValue(30.0)
        self._width_spin.setDecimals(1); self._width_spin.setFixedWidth(72)
        top.addWidget(self._width_spin)

        top.addWidget(QLabel("  Bead H(mm):"))
        self._bead_h_spin = QDoubleSpinBox()
        self._bead_h_spin.setRange(-200.0, 200.0); self._bead_h_spin.setValue(-10.0)
        self._bead_h_spin.setDecimals(1); self._bead_h_spin.setFixedWidth(72)
        top.addWidget(self._bead_h_spin)

        self._draw_btn = QPushButton("Draw Mode: OFF")
        self._draw_btn.setCheckable(True)
        self._draw_btn.setStyleSheet("font-weight:bold; padding:3px 10px;")
        self._draw_btn.toggled.connect(self._on_draw_toggled)
        top.addWidget(self._draw_btn)

        top.addWidget(QLabel("  [1st click → start  /  2nd click → commit]"))
        top.addStretch()
        root.addLayout(top)

        # ── 메인 영역 + 메시지 패널 (QSplitter) ─────────────────────────
        self._outer_splitter = QSplitter(Qt.Vertical)
        self._outer_splitter.setChildrenCollapsible(False)
        root.addWidget(self._outer_splitter, stretch=1)

        # 메인 컨텐츠 위젯
        main_w = QWidget()
        main_lay = QVBoxLayout(main_w)
        main_lay.setContentsMargins(0, 0, 0, 0); main_lay.setSpacing(4)

        # 중앙: 도형 리스트(좌) + 캔버스(우)
        center = QHBoxLayout(); center.setSpacing(6)

        left = QVBoxLayout(); left.setSpacing(4)
        left.addWidget(QLabel("Shapes:"))
        self._shape_list = QListWidget()
        self._shape_list.setFixedWidth(210)
        self._shape_list.setToolTip("클릭으로 선택 → 캔버스에 하이라이트 / Delete 버튼으로 제거")
        self._shape_list.currentRowChanged.connect(lambda _: self._refresh_canvas())
        left.addWidget(self._shape_list, stretch=1)
        btn_row = QHBoxLayout()
        del_btn = QPushButton("Delete"); del_btn.clicked.connect(self._on_delete)
        clr_btn = QPushButton("Clear All"); clr_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(del_btn); btn_row.addWidget(clr_btn)
        left.addLayout(btn_row)
        center.addLayout(left)

        cw = QWidget(); cv = QVBoxLayout(cw); cv.setContentsMargins(0,0,0,0); cv.setSpacing(0)
        self._canvas = PlotCanvas(cw, width=8, height=5)
        tb = NavigationToolbar(self._canvas, cw)
        _s = tb.iconSize()
        tb.setIconSize(_QS(int(_s.width()*0.6), int(_s.height()*0.6)))
        cv.addWidget(tb); cv.addWidget(self._canvas)
        center.addWidget(cw, stretch=1)
        main_lay.addLayout(center, stretch=1)

        # 하단 버튼 행
        bot = QHBoxLayout(); bot.setSpacing(6)

        bot.addWidget(QLabel("Case:"))
        self._case_combo = QComboBox()
        self._case_combo.addItems(self.load_cases)
        self._case_combo.setMinimumWidth(150)
        bot.addWidget(self._case_combo)

        bot.addWidget(QLabel("  Solver:"))
        self._solver_combo = QComboBox()
        self._solver_combo.addItems(["WHT Solver", "CalculiX"])
        self._solver_combo.setFixedWidth(100)
        bot.addWidget(self._solver_combo)

        ev_btn = QPushButton("Evaluate")
        ev_btn.setStyleSheet("font-weight:bold; background:#1e5f2f; color:white;")
        ev_btn.setToolTip("선택 솔버로 현재 하중 케이스 해석")
        ev_btn.clicked.connect(self._on_eval_one)
        bot.addWidget(ev_btn)

        ev_all_btn = QPushButton("Evaluate All")
        ev_all_btn.setStyleSheet("font-weight:bold; background:#1e3d59; color:white;")
        ev_all_btn.setToolTip("선택 솔버로 전체 하중 케이스 해석")
        ev_all_btn.clicked.connect(self._on_eval_all)
        bot.addWidget(ev_all_btn)

        bot.addWidget(QLabel("  Visualizer:"))
        self._viewer_combo = QComboBox()
        self._viewer_combo.addItems(["WHTVisualizer", "ParaView"])
        self._viewer_combo.setFixedWidth(110)
        bot.addWidget(self._viewer_combo)

        exp_os_btn = QPushButton("Export OptiStruct")
        exp_os_btn.setStyleSheet("font-weight:bold;")
        exp_os_btn.setToolTip("현재 컨셉 비드 형상을 OptiStruct .fem으로 내보내기")
        exp_os_btn.clicked.connect(self._on_export_optistruct)
        bot.addWidget(exp_os_btn)

        self._status_lbl = QLabel("준비")
        self._status_lbl.setStyleSheet("color:#aaa; font-size:10px;")
        bot.addWidget(self._status_lbl)
        bot.addStretch()

        # 패널 방향 전환 버튼
        dock_btn = QPushButton("⇔")
        dock_btn.setFixedWidth(28)
        dock_btn.setToolTip("메시지 패널을 우측/하단으로 전환")
        dock_btn.clicked.connect(self._toggle_panel_orientation)
        bot.addWidget(dock_btn)

        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.close)
        bot.addWidget(close_btn)

        bot_w = QWidget(); bot_w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        bot_w.setLayout(bot)
        main_lay.addWidget(bot_w)
        self._outer_splitter.addWidget(main_w)

        # 메시지 패널
        self._msg_panel = QTextEdit()
        self._msg_panel.setReadOnly(True)
        self._msg_panel.setStyleSheet(
            "font-family:'Consolas',monospace; font-size:10px;"
            "background:#111; color:#ccc; border:none;")
        self._msg_panel.setMinimumHeight(60)
        self._outer_splitter.addWidget(self._msg_panel)
        self._outer_splitter.setSizes([580, 120])

    # ── 메시지 패널 ──────────────────────────────────────────────────────────

    def _log(self, msg: str, color: str = "#ccc"):
        html = f'<span style="color:{color};">{msg}</span>'
        self._msg_panel.append(html)
        self._msg_panel.verticalScrollBar().setValue(
            self._msg_panel.verticalScrollBar().maximum())
        self._status_lbl.setText(msg[:80])

    def _toggle_panel_orientation(self):
        cur = self._outer_splitter.orientation()
        new_o = Qt.Horizontal if cur == Qt.Vertical else Qt.Vertical
        self._outer_splitter.setOrientation(new_o)

    # ── Visualizer 호출 ──────────────────────────────────────────────────────

    def _open_result_in_viewer(self, rd, title: str):
        use_pv = self._viewer_combo.currentText() == "ParaView"
        try:
            if use_pv:
                import tempfile, re as _re
                from wht_converter.wht_exporters import VTKHDFExporter
                from wht_visualizer.wht_visualizer import launch_paraview
                safe = _re.sub(r'[^\w\-]', '_', title)[:48]
                tmp = Path(tempfile.mkdtemp(prefix="wht_pv_")) / f"{safe}.hdf"
                VTKHDFExporter().export(rd, str(tmp))
                launch_paraview(str(tmp))
                self._log(f"ParaView 실행: {tmp}", "#2a7a2a")
            else:
                from wht_visualizer.wht_visualizer import WHTVisualizer
                vis = WHTVisualizer(title=title, show=True)
                vis.show_result(rd)
                self._vis_ref = vis
                self._log(f"WHTVisualizer: {title}", "#2a7a2a")
        except Exception:
            self._log(f"Visualizer 오류: {traceback.format_exc().splitlines()[-1]}", "#f55")

    # ── Export / CCX ─────────────────────────────────────────────────────────

    def _on_export_optistruct(self):
        if not self.snap_dir:
            QMessageBox.warning(self, "오류", "snap_dir가 설정되지 않았습니다."); return
        h_concept = self._compute_heights()
        out_path, _ = QFileDialog.getSaveFileName(
            self, "OptiStruct .fem 저장", "concept_bead.fem",
            "OptiStruct FEM (*.fem);;All Files (*)")
        if not out_path:
            return
        try:
            result = _write_optistruct_fem(out_path, self.snap_dir, 0,
                                           h_elem_override=h_concept)
            self._log(f"OptiStruct 저장: {result}", "#2a7a2a")
        except Exception:
            self._log(f"저장 오류: {traceback.format_exc().splitlines()[-1]}", "#f55")

    def _on_run_ccx(self):
        if not self.snap_dir:
            QMessageBox.warning(self, "오류", "snap_dir가 설정되지 않았습니다."); return
        if hasattr(self, '_ccx_worker') and self._ccx_worker and self._ccx_worker.isRunning():
            QMessageBox.information(self, "실행 중", "CalculiX 해석이 진행 중입니다."); return
        case = self._case_combo.currentText()
        h_concept = self._compute_heights()
        self._log(f"CalculiX 해석 시작: {case}", "#6af")
        self._ccx_worker = _ConceptCcxWorker(
            self.snap_dir, h_concept, case, parent=self)
        self._ccx_worker.progress.connect(lambda m: self._log(m, "#aaa"))
        self._ccx_worker.finished.connect(self._on_ccx_done)
        self._ccx_worker.error.connect(lambda e: self._log(f"CCX 오류: {e[:200]}", "#f55"))
        self._ccx_worker.start()

    def _on_ccx_done(self, result: dict):
        rd  = result.get("wht_result_data")
        rtype = result.get("type", "")
        if rtype == "modal":
            freqs = [float(f) for f in rd.time_values]
            self._log("CCX 모달: " + "  ".join(f"f{i+1}={f:.2f}Hz"
                                               for i, f in enumerate(freqs[:10])), "#2a7a2a")
        else:
            disp = rd.point_data.get("Displacement", [None])[0]
            if disp is not None:
                umax = float(np.max(np.abs(np.array(disp)[:, :3])))
                self._log(f"CCX {result.get('lc_name','')}: Max|U|={umax:.4f} mm", "#2a7a2a")
        if rd:
            self._open_result_in_viewer(rd, f"CCX Concept — {result.get('lc_name','Modal')}")

    # ── 도형 그리기 ─────────────────────────────────────────────────────────


    def _on_draw_toggled(self, on):
        self._draw_btn.setText(f"Draw Mode: {'ON' if on else 'OFF'}")
        self._pts_buf.clear(); self._clear_preview()
        if on:
            self._cid_press = self._canvas.mpl_connect('button_press_event', self._on_press)
            self._cid_move  = self._canvas.mpl_connect('motion_notify_event', self._on_move)
        else:
            if self._cid_press: self._canvas.mpl_disconnect(self._cid_press); self._cid_press = None
            if self._cid_move:  self._canvas.mpl_disconnect(self._cid_move);  self._cid_move  = None
        self._canvas.draw()

    def _on_press(self, ev):
        if ev.button != 1 or not ev.inaxes or ev.xdata is None: return
        self._pts_buf.append((ev.xdata, ev.ydata))
        if len(self._pts_buf) == 2:
            self._commit()

    def _on_move(self, ev):
        if not self._pts_buf or not ev.inaxes or ev.xdata is None: return
        self._clear_preview()
        ax = self._canvas.ax
        p1 = self._pts_buf[0]; p2 = (ev.xdata, ev.ydata)
        art = _add_shape_artist(ax, self._shape_combo.currentText(),
                                p1, p2, self._width_spin.value(), 'yellow', alpha=0.45)
        m, = ax.plot(*p1, 'yo', markersize=6)
        if art: self._preview.append(art)
        self._preview.append(m)
        self._canvas.draw_idle()

    def _commit(self):
        pts = self._pts_buf.copy(); self._pts_buf.clear(); self._clear_preview()
        self._shape_id_counter += 1
        shape = {"id":    self._shape_id_counter,
                 "type":  self._shape_combo.currentText(),
                 "pts":   pts,
                 "width": self._width_spin.value(),
                 "bead_h": self._bead_h_spin.value()}
        self._shapes.append(shape)
        h = shape["bead_h"]
        self._shape_list.addItem(
            f"#{shape['id']}  {shape['type']}  w={shape['width']:.0f}  h={h:+.1f}mm")
        self._shape_list.setCurrentRow(self._shape_list.count() - 1)
        self._refresh_canvas()

    def _clear_preview(self):
        for a in self._preview:
            try: a.remove()
            except Exception: pass
        self._preview.clear()

    # ── 리스트 조작 ─────────────────────────────────────────────────────────

    def _on_delete(self):
        idx = self._shape_list.currentRow()
        if 0 <= idx < len(self._shapes):
            del self._shapes[idx]
            self._shape_list.takeItem(idx)
            self._refresh_canvas()

    def _on_clear(self):
        self._shapes.clear(); self._shape_list.clear(); self._refresh_canvas()

    # ── 캔버스 렌더링 ───────────────────────────────────────────────────────

    def _compute_heights(self):
        """기본 높이 + 그려진 도형 (절대값 큰 것 우선)."""
        coords  = np.array(self.snap["coords"])
        result  = np.array(self.snap["heights"], dtype=float)
        for shape in self._shapes:
            mask = _shape_mask(coords[:, :2], shape)
            bh = shape["bead_h"]
            # bh=0: 기존 비드 제거 (무조건 덮어쓰기), 아니면 절대값 큰 것 우선
            if bh == 0.0:
                result[mask] = 0.0
            else:
                improve = mask & (np.abs(bh) > np.abs(result))
                result[improve] = bh
        return result

    def _refresh_canvas(self):
        from scipy.interpolate import griddata
        from matplotlib.colors import BoundaryNorm
        from matplotlib import cm

        ax = self._canvas.ax; fig = self._canvas.fig
        ax.clear()

        coords  = np.array(self.snap["coords"])
        heights = self._compute_heights()
        x, y   = coords[:, 0], coords[:, 1]
        h_max   = float(self.snap.get("h_max", 15.0))
        bsteps  = int(self.snap.get("bead_steps", 0))

        # 격자 해상도: 요소 크기에 맞춤 (1 cell ≈ 1 element)
        nx_g, ny_g = _grid_res_from_elements(x, y)
        Xi, Yi = np.meshgrid(np.linspace(x.min(), x.max(), nx_g),
                             np.linspace(y.min(), y.max(), ny_g))
        Zi = griddata((x,y), heights, (Xi,Yi), method='nearest')
        Zi = np.clip(Zi, -h_max, h_max)

        n_steps    = max(bsteps, 5)
        pos_levels = np.linspace(0, h_max, n_steps + 1)
        full_lev   = np.concatenate([-pos_levels[::-1], pos_levels[1:]])
        bounds     = np.concatenate([[-h_max - 1e-6],
                                     (full_lev[:-1]+full_lev[1:])*0.5,
                                     [h_max + 1e-6]])
        cmap  = cm.get_cmap('coolwarm', len(full_lev))
        norm  = BoundaryNorm(bounds, ncolors=cmap.N, clip=True)
        ax.imshow(Zi, origin='lower', aspect='equal',
                  extent=[x.min(), x.max(), y.min(), y.max()],
                  cmap=cmap, norm=norm, interpolation='nearest')

        # 그려진 도형 오버레이
        sel_idx = self._shape_list.currentRow()
        for i, shape in enumerate(self._shapes):
            selected = (i == sel_idx)
            color = '#00e5ff' if selected else 'white'
            alpha = 0.85 if selected else 0.65
            _add_shape_artist(ax, shape["type"],
                              shape["pts"][0], shape["pts"][1],
                              shape["width"], color, alpha=alpha)
            cx = (shape["pts"][0][0] + shape["pts"][1][0]) / 2
            cy = (shape["pts"][0][1] + shape["pts"][1][1]) / 2
            label = f"#{shape['id']} {shape['bead_h']:+.1f}"
            ax.text(cx, cy, label,
                    color='white' if not selected else '#00e5ff',
                    fontsize=7, ha='center', va='center', fontweight='bold' if selected else 'normal',
                    bbox=dict(facecolor='black', alpha=0.6 if selected else 0.5,
                              pad=1, edgecolor='#00e5ff' if selected else 'none'))

        # 레전드
        if self._shapes:
            import matplotlib.patches as _mp
            handles = []
            for shape in self._shapes:
                patch = _mp.Patch(color='white', alpha=0.65,
                                  label=f"#{shape['id']} {shape['type']} h={shape['bead_h']:+.1f}")
                handles.append(patch)
            ax.legend(handles=handles, loc='upper right', fontsize=7,
                      facecolor='#1e1e1e', edgecolor='#555', labelcolor='white',
                      framealpha=0.85)

        ax.set_title(f"Concept Bead Preview  (iter {self.snap.get('iter','?')})")
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        fig.tight_layout(); self._canvas.draw()

    # ── 평가 ────────────────────────────────────────────────────────────────

    def _on_eval_one(self):
        self._run_eval([self._case_combo.currentText()])

    def _on_eval_all(self):
        self._run_eval(list(self.load_cases))

    def _is_ccx(self) -> bool:
        return (hasattr(self, '_solver_combo') and
                self._solver_combo.currentText() == "CalculiX")

    def _run_eval(self, cases):
        if not self.snap_dir:
            QMessageBox.warning(self, "오류", "snap_dir가 설정되지 않았습니다."); return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "실행 중", "이전 해석이 진행 중입니다."); return
        h_concept = self._compute_heights()
        if self._is_ccx():
            # CalculiX: 케이스 하나씩 순차 실행
            case = cases[0] if cases else self._case_combo.currentText()
            self._log(f"CalculiX 해석: {case}", "#6af")
            self._status_lbl.setText("CalculiX 해석 중...")
            self._worker = _ConceptCcxWorker(self.snap_dir, h_concept, case, parent=self)
            self._worker.progress.connect(lambda m: self._log(m, "#aaa"))
            self._worker.finished.connect(self._on_ccx_done)
            self._worker.error.connect(lambda e: (
                self._status_lbl.setText("오류"),
                self._log(f"CCX 오류: {e[:200]}", "#f55")))
            self._worker.start()
        else:
            # WHT Solver
            self._log(f"WHT Solver 해석: {cases}", "#6af")
            self._status_lbl.setText("WHT 해석 중...")
            self._worker = BeadConceptEvalWorker(self.snap_dir, h_concept, cases, parent=self)
            self._worker.progress.connect(lambda m: self._log(m, "#aaa"))
            self._worker.finished.connect(self._on_eval_done)
            self._worker.error.connect(lambda e: (
                self._status_lbl.setText("오류"),
                QMessageBox.critical(self, "해석 오류", e[:500])))
            self._worker.start()

    def _on_eval_done(self, results: list):
        self._log(f"── Concept Evaluation 완료 [iter {self.snap.get('iter','?')}] ──", "#ff0")
        result_lines = [f"== Bead Concept Evaluation  [iter {self.snap.get('iter','?')}] ==", ""]
        for r in results:
            case = r["case"]
            if r["type"] == "modal":
                freqs = r.get("freqs", [r["f1"]])
                b, d  = r["base"], r["delta_pct"]
                sign  = "▲" if d >= 0 else "▼"
                # 메시지 패널 출력
                self._log(f"[{case}]", "#6cf")
                self._log(f"  f1={freqs[0]:.2f} Hz  (iter0: {b:.2f} Hz)  {sign} {d:+.1f}%", "#2a7a2a")
                for i, f in enumerate(freqs[:10], 1):
                    self._log(f"    Mode {i}: {f:.2f} Hz")
                # effective mass
                em = r.get("eff_mass_data")
                if em is not None:
                    try:
                        eff = em["eff_mass"]; tot = em["total_mass"]
                        for i, row in enumerate(eff[:6], 1):
                            ratio = float(np.sum(np.abs(row))) / (float(tot) + 1e-12) * 100
                            self._log(f"    Mode {i} eff.mass ratio: {ratio:.1f}%")
                    except Exception:
                        pass
                # result dialog line
                result_lines += [f"[{case}]",
                                  f"  f1={freqs[0]:.2f} Hz  (iter0: {b:.2f} Hz)  {sign} {d:+.1f}%"]
                for i, f in enumerate(freqs[:10], 1):
                    result_lines.append(f"    Mode {i}: {f:.2f} Hz")
                # visualizer
                rd = r.get("wht_result_data")
                if rd:
                    self._open_result_in_viewer(rd, f"Concept Modal — iter {self.snap.get('iter','?')}")
            elif r["type"] == "static":
                u, b, d = r["u_max"], r["base"], r["delta_pct"]
                sign = "▲" if d >= 0 else "▼"
                self._log(f"[{case}]", "#6cf")
                self._log(f"  Max|U|={u:.4f} mm  (iter0: {b:.4f} mm)  {sign} {d:+.1f}%", "#2a7a2a")
                result_lines += [f"[{case}]",
                                  f"  Max|U|={u:.4f} mm  (iter0: {b:.4f} mm)  {sign} {d:+.1f}%"]
                rd = r.get("wht_result_data")
                if rd:
                    self._open_result_in_viewer(rd, f"Concept {case} — iter {self.snap.get('iter','?')}")
            else:
                msg = r.get("msg", "")
                self._log(f"[{case}]  ERROR: {msg}", "#f55")
                result_lines.append(f"[{case}]  ERROR: {msg}")
        dlg = _ConceptResultDialog("\n".join(result_lines), self)
        dlg.exec()


class _ConceptResultDialog(QDialog):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Concept Evaluation Results")
        self.resize(560, 320)
        lay = QVBoxLayout(self)
        te = QTextEdit(); te.setReadOnly(True); te.setPlainText(text)
        te.setStyleSheet("font-family:'Consolas',monospace; font-size:11px;"
                         "background:#1e1e1e; color:#d4d4d4; border:none;")
        lay.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept); lay.addWidget(bb)


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

class _AnalysisResultsDialog(QDialog):
    def __init__(self, parent=None, iter_stats_data=None, initial_iter=0):
        super().__init__(parent)
        self.setWindowTitle("Analysis Results")
        self.resize(600, 500)
        self.iter_stats_data = iter_stats_data or []
        self._build_ui(initial_iter)

    def _build_ui(self, initial_iter):
        lay = QVBoxLayout(self)
        
        top_lay = QHBoxLayout()
        top_lay.addWidget(QLabel("Select Iteration:"))
        self.combo = QComboBox()
        for item in self.iter_stats_data:
            self.combo.addItem(f"Iteration {item['iter']}", userData=item)
            
        top_lay.addWidget(self.combo)
        top_lay.addStretch()
        lay.addLayout(top_lay)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        font = self.text_edit.font()
        font.setFamily("Consolas")
        self.text_edit.setFont(font)
        lay.addWidget(self.text_edit)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        lay.addWidget(btn_box)

        self.combo.currentIndexChanged.connect(self._on_combo_changed)
        
        idx = 0
        for i, item in enumerate(self.iter_stats_data):
            if item["iter"] == initial_iter:
                idx = i
                break
        self.combo.setCurrentIndex(idx)
        self._on_combo_changed(idx)

    def _on_combo_changed(self, idx):
        if idx < 0 or idx >= len(self.iter_stats_data):
            return
        item = self.iter_stats_data[idx]
        text_lines = []
        text_lines.append(f"Iteration: {item['iter']}")
        text_lines.append("-" * 50)
        for case_name, stats in item.get("cases", {}).items():
            text_lines.append(f"\n[Load Case: {case_name}]")
            for k, v in stats.items():
                if isinstance(v, float):
                    text_lines.append(f"  {k:<30}: {v:.6e}")
                else:
                    text_lines.append(f"  {k:<30}: {v}")
        self.text_edit.setPlainText("\\n".join(text_lines))

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
    app.setWindowIcon(_app_icon())
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
