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
)
from PySide6.QtCore import Signal, QObject, QThread, Qt
from PySide6.QtGui import QPixmap
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
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

    def __init__(self, snap_dir: str, iter_num: int, case_name: str, parent=None):
        super().__init__(parent)
        self.snap_dir  = snap_dir
        self.iter_num  = iter_num
        self.case_name = case_name

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
                result = fea.solve_modal(num_modes=10)
                self.finished.emit({
                    "type":   "modal",
                    "result": result,
                    "model":  model,
                })
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
    def __init__(self, stop_event=None, results_dir: str = ""):
        super().__init__()
        self.setWindowTitle("WHT Topography Optimization Monitor (V3.0)")
        self.resize(1300, 900)
        self.stop_event  = stop_event
        self.results_dir = results_dir   # out_dir 전달받음 (snap_dir 동적 갱신 가능)
        self.snap_dir    = ""            # solver 콜백에서 수신

        # ── 데이터 저장소 ──────────────────────────────────────────────────
        self.history = {
            "iter": [], "compliance": [], "avg_h": [], "max_h": [],
            "dx": [], "area_ratio": [], "frequencies": [], "cases": {},
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
        self.iter_export_btn = None   # Export OptiStruct 버튼
        self.iter_mesh_btn = None     # Mesh View 버튼
        self.iter_status_label = None # 해석 상태 텍스트
        self._re_worker = None        # Optional[_ReAnalysisWorker]
        self._vis_window = None       # Optional[WHTVisualizer] — 재사용 창

        self._init_ui()
        self._setup_shortcuts()

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
        top_widget.setFixedHeight(100)
        top_bar = QHBoxLayout(top_widget)
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(4)

        logo_label = QLabel()
        logo_label.setFixedSize(100, 100)
        logo_path = os.path.join(os.path.dirname(__file__), "resources", "sidebar_logo.png")
        if os.path.exists(logo_path):
            pix = QPixmap(logo_path).scaled(
                100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
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
        lay.addWidget(self.table)
        self.tabs.addTab(tab, "Summary Table")

    def _build_tab_curve(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        ctrl = QHBoxLayout()
        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            "ALL (Normalized)", "Compliance", "Avg_h", "Max_h",
            "dx", "Area_Ratio", "Natural Frequencies",
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

        # ── 높이 분포 캔버스 ─────────────────────────────────────────────
        self.height_canvas = PlotCanvas(tab)
        lay.addWidget(self.height_canvas, stretch=1)

        # ── 액션 행: Mesh View / Run Analysis / Export ───────────────────
        ctrl_widget = QWidget()
        ctrl_widget.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Fixed
        )
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
        self.iter_case_combo.addItem("Modal Analysis")
        ctrl_action.addWidget(self.iter_case_combo)

        self.iter_run_btn = QPushButton("Run Analysis")
        self.iter_run_btn.setStyleSheet("font-weight:bold;")
        self.iter_run_btn.setToolTip("선택 하중 케이스로 wht_solver 해석 후 결과를 visualizer에 표시")
        self.iter_run_btn.clicked.connect(self._on_run_analysis)
        ctrl_action.addWidget(self.iter_run_btn)

        self.iter_export_btn = QPushButton("Export OptiStruct .fem")
        self.iter_export_btn.setStyleSheet("font-weight:bold;")
        self.iter_export_btn.setToolTip("선택 이터레이션의 변형 메시 + 하중 케이스를 .fem 파일로 저장")
        self.iter_export_btn.clicked.connect(self._on_export_optistruct)
        ctrl_action.addWidget(self.iter_export_btn)

        ctrl_action.addStretch()
        lay.addWidget(ctrl_widget)

        # ── 상태 표시줄 ──────────────────────────────────────────────────
        self.iter_status_label = QLabel("")
        self.iter_status_label.setStyleSheet("color:#555;font-size:11px;")
        lay.addWidget(self.iter_status_label)

        self.tabs.addTab(tab, "Iteration Results")

    def _build_tab_modal(self):
        tab = QWidget(); lay = QVBoxLayout(tab)
        self.modal_table = QTableWidget(10, 3)
        self.modal_table.setHorizontalHeaderLabels(
            ["Mode", "Ref. (Hz)", "Current (Hz)"]
        )
        self.modal_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(10):
            self.modal_table.setItem(i, 0, QTableWidgetItem(f"Mode {i+1}"))
        lay.addWidget(self.modal_table)
        self.tabs.addTab(tab, "Modal Analysis")

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

            # snap_dir 수신
            if data.get("snap_dir"):
                self.snap_dir = data["snap_dir"]

            # ── 메시 엣지 수신 (최초 1회) ────────────────────────────────
            if "mesh_edge_segs" in data and self.mesh_edge_segs is None:
                self.mesh_edge_segs = data["mesh_edge_segs"]

            # ── 리셋 감지 (iter 1이 다시 오면 새 최적화 시작) ────────────────
            if it == 1 and len(self.history["iter"]) > 0:
                self._clear_history()

            # ── 히스토리 축적 ────────────────────────────────────────────
            self.history["iter"].append(it)
            self.history["compliance"].append(data.get("compliance", 0.0))
            self.history["avg_h"].append(data.get("avg_h", 0.0))
            self.history["max_h"].append(data.get("max_h", 0.0))
            self.history["dx"].append(data.get("dx", 0.0))
            self.history["area_ratio"].append(data.get("area_ratio", 0.0))
            freqs = data.get("frequencies", [])
            self.history["frequencies"].append(freqs)

            # Modal Ref. 초기화 (최초 수신 시)
            if self.ref_freqs is None and freqs:
                self.ref_freqs = freqs
                if self.modal_table:
                    for i, f in enumerate(freqs):
                        if i < 10:
                            self.modal_table.setItem(
                                i, 1, QTableWidgetItem(f"{f:.2f}")
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
                    headers = ["Iter", "C_total", "Avg_h", "Max_h", "dx", "Area_Ratio"]
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
                        f"{ar:.3f}"]
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

            # ── Modal 테이블 갱신 ─────────────────────────────────────────
            if self.modal_table and freqs:
                for i, f in enumerate(freqs):
                    if i < 10:
                        self.modal_table.setItem(i, 2, QTableWidgetItem(f"{f:.2f}"))

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

    def _clear_history(self):
        self.history = {
            "iter": [], "compliance": [], "avg_h": [], "max_h": [],
            "dx": [], "area_ratio": [], "frequencies": [], "cases": {},
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
                "dx", "Area_Ratio", "Natural Frequencies",
            ])
            self.metric_combo.blockSignals(False)
        if self.table:
            self.table.setRowCount(0); self.table.setColumnCount(0)
        if self.modal_table:
            for i in range(10):
                self.modal_table.setItem(i, 1, QTableWidgetItem(""))
                self.modal_table.setItem(i, 2, QTableWidgetItem(""))
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
        elif metric in ("Compliance", "Avg_h", "Max_h", "dx"):
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
        max_abs = max(1e-5, float(np.max(np.abs(heights))))

        from scipy.interpolate import griddata
        n_grid = max(16, int(len(x) ** 0.5))
        xi = np.linspace(x.min(), x.max(), n_grid)
        yi = np.linspace(y.min(), y.max(), n_grid)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), heights, (Xi, Yi), method='linear')
        sc = ax.imshow(
            Zi, origin='lower', aspect='equal',
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap='coolwarm', vmin=-max_abs, vmax=max_abs,
            interpolation='bilinear',
        )

        if self._height_colorbar is None:
            self._height_colorbar = fig.colorbar(sc, ax=ax)
            self._height_colorbar.set_label("Bead Height (mm)  [+: outward / −: inward]")
        else:
            self._height_colorbar.update_normal(sc)

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
            self.iter_status_label.setText(msg)
            self.iter_status_label.setStyleSheet(f"color:{color};font-size:11px;")

    def _get_selected_iter_num(self) -> int:
        """height_iter_combo 현재 선택 이터레이션 번호 반환 (Latest → 최신 snap의 iter)."""
        if not self.height_snapshots:
            return 0
        snap = self._get_snap(self.height_iter_combo)
        if snap is None:
            return 0
        return int(snap["iter"])

    def _open_visualizer(self, wht_result_data, title: str = "WHT Visualizer"):
        """WHTResultData를 WHTVisualizer 창으로 표시합니다 (BackgroundPlotter — 비차단)."""
        try:
            from wht_visualizer.wht_visualizer import WHTVisualizer
            vis = WHTVisualizer(title=title, show=True)
            vis.show_result(wht_result_data)
            self._vis_window = vis   # GC 방지
        except Exception:
            self._set_iter_status(f"Visualizer 실행 오류: {traceback.format_exc()[:120]}", "red")

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

        self._re_worker = _ReAnalysisWorker(self.snap_dir, iter_num, case_name)
        self._re_worker.progress.connect(lambda msg: self._set_iter_status(msg, "blue"))
        self._re_worker.finished.connect(self._on_analysis_finished)
        self._re_worker.error.connect(self._on_analysis_error)
        self._re_worker.start()

    def _on_analysis_finished(self, result: dict):
        if self.iter_run_btn:
            self.iter_run_btn.setEnabled(True)

        rtype     = result.get("type")
        model     = result["model"]
        solver_result = result["result"]

        try:
            from wht_converter.wht_models import WHTMetadata
            meta = WHTMetadata(
                solver_name="WHT-Topo", solver_version="1.0",
                analysis_type=rtype, coordinate_system="cartesian",
                unit_length="mm", unit_force="N",
            )
            rd = solver_result.to_wht_result_data(meta, model)

            if rtype == "modal":
                freqs = [float(f) for f in solver_result.frequencies]
                freq_str = "  ".join(f"f{i+1}={f:.2f}Hz" for i, f in enumerate(freqs[:6]))
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
            self._set_iter_status(f"결과 변환 오류: {traceback.format_exc()[:120]}", "red")

    def _on_analysis_error(self, msg: str):
        if self.iter_run_btn:
            self.iter_run_btn.setEnabled(True)
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


def start_monitor_ui(queue, stop_event=None, results_dir: str = ""):
    """
    별도 프로세스에서 PySide6 UI를 실행합니다.

    Parameters
    ----------
    queue       : multiprocessing.Queue  — 솔버 → 모니터 데이터 큐
    stop_event  : multiprocessing.Event  — GUI 종료 시 솔버에 중단 신호
    results_dir : str                    — out_dir 경로 (OptiStruct 저장 기본 경로)
    """
    app    = QApplication.instance() or QApplication(sys.argv)
    window = WHTMonitorWindow(stop_event=stop_event, results_dir=results_dir)
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
    sys.exit(app.exec())
