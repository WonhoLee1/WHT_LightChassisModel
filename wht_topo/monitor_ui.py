# -*- coding: utf-8 -*-
"""
monitor_ui.py
=============
최적화 프로세스 실시간 모니터링을 위한 PySide6 기반 UI 모듈 (V2.2).

변경사항:
- Area Ratio 지표 추가 (테이블, 수렴 커브)
- Height Distribution 모든 이터레이션 스냅샷 저장
- 이터레이션 선택 드롭다운으로 과거 Height Distribution 열람 가능
- 노드 간격 기반 사각형 마커(marker='s') 크기 자동 계산
"""

import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QHeaderView,
    QComboBox, QLabel, QPushButton, QSlider
)
from PySide6.QtCore import Signal, QObject, QThread, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'Segoe UI'


# ────────────────────────────────────────────────────────────────────────────
# Matplotlib Canvas Helper
# ────────────────────────────────────────────────────────────────────────────

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


def _estimate_node_spacing(coords: np.ndarray) -> float:
    """
    노드 좌표 배열로부터 평균 이웃 노드 간격을 추정합니다.

    Parameters
    ----------
    coords : (N, 3) ndarray
        노드 좌표 배열 (XYZ)

    Returns
    -------
    float : 추정 평균 노드 간격 [mm]
    """
    if len(coords) < 2:
        return 40.0
    # X, Y 방향 최솟값을 간격의 근사치로 사용 (정렬된 메시 가정)
    xs = np.sort(np.unique(np.round(coords[:, 0], 1)))
    ys = np.sort(np.unique(np.round(coords[:, 1], 1)))
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 40.0
    dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 40.0
    return min(dx, dy)


# ────────────────────────────────────────────────────────────────────────────
# Main Monitor Window
# ────────────────────────────────────────────────────────────────────────────

class WHTMonitorWindow(QMainWindow):
    def __init__(self, stop_event=None):
        super().__init__()
        self.setWindowTitle("WHT Topography Optimization Monitor (V2.2)")
        self.resize(1200, 860)
        self.stop_event = stop_event

        # ── 1. 속성 초기 선언 (AttributeError 방지) ──
        self.history = {
            "iter": [],
            "compliance": [],
            "avg_h": [],
            "max_h": [],
            "dx": [],
            "area_ratio": [],       # 비드 점유 면적 비율
            "frequencies": [],
            "cases": {},
        }
        # Height Distribution 히스토리: 이터레이션별 스냅샷 저장
        self.height_snapshots = []  # list of {"iter": int, "coords": ndarray, "heights": ndarray}

        self.ref_freqs = None
        self.coords = None
        self.current_h = None
        self.case_names = []

        # UI 위젯 참조 (초기화 전 None)
        self.height_canvas = None
        self.curve_canvas = None
        self.table = None
        self.modal_table = None
        self.metric_combo = None
        self.height_iter_combo = None
        self._height_colorbar = None
        self.status_label = None
        self.stop_btn = None
        self.height_slider = None
        self.btn_prev_h = None
        self.btn_next_h = None

        # ── 2. UI 초기화 ──
        self._init_ui()

        # ── 3. 단축키 설정 ──
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        """단축키 설정 (예: Ctrl+T -> 항상 위 토글)"""
        from PySide6.QtGui import QShortcut, QKeySequence
        self.top_shortcut = QShortcut(QKeySequence("Ctrl+T"), self)
        self.top_shortcut.activated.connect(self._toggle_always_on_top)

    def _toggle_always_on_top(self):
        """창의 '항상 위' 속성을 토글합니다."""
        is_on_top = bool(self.windowFlags() & Qt.WindowStaysOnTopHint)
        if is_on_top:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
            if self.top_btn:
                self.top_btn.setText("Stay on Top: OFF")
                self.top_btn.setStyleSheet("")
        else:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            if self.top_btn:
                self.top_btn.setText("Stay on Top: ON")
                self.top_btn.setStyleSheet("background-color: #ffcccc; font-weight: bold;")
        
        # WindowFlags 변경 후 반드시 show()를 다시 호출해야 반영됨 (Windows)
        self.show()
        print(f" -> [Monitor] Stay on Top: {'ON' if not is_on_top else 'OFF'}")

    def _on_stop_clicked(self):
        """솔버에 중단 신호를 보냅니다."""
        if self.stop_event:
            self.stop_event.set()
            if self.status_label:
                self.status_label.setText("Status: STOP Requested...")
                self.status_label.setStyleSheet("color: #cc6600; font-weight: bold; font-size: 13px;")
            self.stop_btn.setEnabled(False)
            print(" -> [Monitor] STOP event set.")
        else:
            print(" -> [Monitor] STOP event not available.")

    def _on_reset_clicked(self):
        """UI와 히스토리 데이터를 강제로 초기화합니다."""
        self._clear_history()
        # 캔버스 강제 갱신
        if self.curve_canvas: self.curve_canvas.draw()
        if self.height_canvas: self.height_canvas.draw()
        print(" -> [Monitor] UI and History Reset by user.")

    # ────────────────────────────────────────────────────────────────────────
    # UI 구성
    # ────────────────────────────────────────────────────────────────────────

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(5, 5, 5, 5)
        root_layout.setSpacing(5)

        # ── Top Control Bar ───────────────────────────────────────────────
        top_bar = QHBoxLayout()
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-weight: bold; color: #444; font-size: 13px;")
        
        self.stop_btn = QPushButton("Request STOP")
        self.stop_btn.setFixedWidth(120)
        self.stop_btn.setStyleSheet("background-color: #fdd; color: #c00; font-weight: bold;")
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        
        self.top_btn = QPushButton("Stay on Top: OFF")
        self.top_btn.setFixedWidth(120)
        self.top_btn.clicked.connect(self._toggle_always_on_top)
        
        self.reset_btn = QPushButton("Reset UI")
        self.reset_btn.setFixedWidth(100)
        self.reset_btn.clicked.connect(self._on_reset_clicked)
        
        top_bar.addWidget(self.status_label)
        top_bar.addStretch()
        top_bar.addWidget(self.top_btn)
        top_bar.addWidget(self.reset_btn)
        top_bar.addWidget(self.stop_btn)
        root_layout.addLayout(top_bar)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        # Tab 1: Summary Table ──────────────────────────────────────────────
        tab1 = QWidget()
        lay1 = QVBoxLayout(tab1)
        self.table = QTableWidget(0, 0)
        lay1.addWidget(self.table)
        self.tabs.addTab(tab1, "Summary Table")

        # Tab 2: Convergence Curve ──────────────────────────────────────────
        tab2 = QWidget()
        lay2 = QVBoxLayout(tab2)
        ctrl2 = QHBoxLayout()
        self.metric_combo = QComboBox()
        self.metric_combo.addItems([
            "ALL (Normalized)",
            "Compliance", "Avg_h", "Max_h", "dx",
            "Area_Ratio",               # ← 신규
            "Natural Frequencies",
        ])
        self.metric_combo.currentTextChanged.connect(self._update_curves)
        ctrl2.addWidget(QLabel("Select Metric:"))
        ctrl2.addWidget(self.metric_combo)
        ctrl2.addStretch()
        lay2.addLayout(ctrl2)
        self.curve_canvas = PlotCanvas(tab2)
        lay2.addWidget(self.curve_canvas)
        self.tabs.addTab(tab2, "Convergence Curve")

        # Tab 3: Height Distribution ────────────────────────────────────────
        tab3 = QWidget()
        lay3 = QVBoxLayout(tab3)
        ctrl3 = QHBoxLayout()
        self.height_iter_combo = QComboBox()
        self.height_iter_combo.addItem("Latest")
        self.height_iter_combo.currentIndexChanged.connect(self._on_height_combo_changed)
        
        self.btn_prev_h = QPushButton("<")
        self.btn_prev_h.setFixedWidth(30)
        self.btn_prev_h.clicked.connect(self._on_prev_height)
        
        self.btn_next_h = QPushButton(">")
        self.btn_next_h.setFixedWidth(30)
        self.btn_next_h.clicked.connect(self._on_next_height)

        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setMinimum(0)
        self.height_slider.setMaximum(0)
        self.height_slider.valueChanged.connect(self._on_height_slider_changed)

        ctrl3.addWidget(QLabel("Iteration:"))
        ctrl3.addWidget(self.btn_prev_h)
        ctrl3.addWidget(self.height_iter_combo)
        ctrl3.addWidget(self.btn_next_h)
        ctrl3.addWidget(QLabel("  Slider:"))
        ctrl3.addWidget(self.height_slider)
        ctrl3.addStretch()
        lay3.addLayout(ctrl3)
        self.height_canvas = PlotCanvas(tab3)
        lay3.addWidget(self.height_canvas)
        self.tabs.addTab(tab3, "Height Distribution")

        # Tab 4: Modal Analysis ─────────────────────────────────────────────
        tab4 = QWidget()
        lay4 = QVBoxLayout(tab4)
        self.modal_table = QTableWidget(10, 3)
        self.modal_table.setHorizontalHeaderLabels(["Mode", "Ref. (Hz)", "Current (Hz)"])
        self.modal_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(10):
            self.modal_table.setItem(i, 0, QTableWidgetItem(f"Mode {i + 1}"))
        lay4.addWidget(self.modal_table)
        self.tabs.addTab(tab4, "Modal Analysis")

    def _on_stop_clicked(self):
        if self.stop_event is not None:
            self.stop_event.set()
            if self.status_label:
                self.status_label.setText("Status: Stop Requested...")
                self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            if self.stop_btn:
                self.stop_btn.setEnabled(False)
                self.stop_btn.setText("Stopping...")

    # ────────────────────────────────────────────────────────────────────────
    # 데이터 수신 및 UI 갱신
    # ────────────────────────────────────────────────────────────────────────

    def update_data(self, data: dict):
        if data.get("status") == "STOP":
            if self.status_label:
                self.status_label.setText("Status: Optimization Completed / Stopped")
                self.status_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
            if self.stop_btn:
                self.stop_btn.setEnabled(False)
            # 마지막 상태 반영을 위해 조기 리턴하지 않고 갱신 수행 (데이터가 함께 왔을 경우)
            if "iter" not in data:
                return

        try:
            it = data["iter"]

            # ── 리셋(Reset) 감지 ──
            # iter=0인데 이미 히스토리가 있다면 새로운 해석 시작으로 간주하고 초기화
            if it == 0 and len(self.history["iter"]) > 0:
                print(" -> [Monitor] Reset detected (Iter 0). Clearing history.")
                self._clear_history()

            # ── 히스토리 축적 ──
            self.history["iter"].append(it)
            self.history["compliance"].append(data["compliance"])
            self.history["avg_h"].append(data["avg_h"])
            self.history["max_h"].append(data["max_h"])
            self.history["dx"].append(data["dx"])
            self.history["area_ratio"].append(data.get("area_ratio", 0.0))

            freqs = data.get("frequencies", [])
            self.history["frequencies"].append(freqs)

            # Modal Ref. 초기화 (첫 이터레이션)
            if it == 0 and self.ref_freqs is None:
                self.ref_freqs = freqs
                if self.modal_table:
                    for i, f in enumerate(freqs):
                        if i < 10:
                            self.modal_table.setItem(i, 1, QTableWidgetItem(f"{f:.2f}"))

            # 하중 케이스 초기 설정 (첫 수신 시)
            cases_data = data.get("cases", {})
            if not self.case_names and cases_data:
                self.case_names = sorted(cases_data.keys())
                for name in self.case_names:
                    if self.metric_combo:
                        self.metric_combo.addItem(f"U_{name}")
                        self.metric_combo.addItem(f"Disp_{name}")
                        self.metric_combo.addItem(f"Stress_{name}")
                    self.history["cases"][name] = {"U": [], "max_disp": [], "max_stress": []}

                if self.table:
                    headers = ["Iter", "C_total", "Avg_h", "Max_h", "dx", "Area_Ratio"]
                    for name in self.case_names:
                        headers += [f"U_{name}", f"D_{name}", f"S_{name}"]
                    self.table.setColumnCount(len(headers))
                    self.table.setHorizontalHeaderLabels(headers)

                    # ── 열 제목 툴팁(Tooltip) 추가 ──
                    tooltips = {
                        "Iter": "반복 횟수 (Iteration Number)",
                        "C_total": "전체 하중 케이스 가중 컴플라이언스 합 (Total Compliance)",
                        "Avg_h": "전체 모델 평균 비드 높이 (Average Bead Height) [mm]",
                        "Max_h": "전체 모델 최대 비드 높이 (Maximum Bead Height) [mm]",
                        "dx": "이전 스텝 대비 최대 밀도 변화량 (Max Density Change)",
                        "Area_Ratio": "현재 비드가 점유한 면적 비율 (Bead Area Ratio) [0~1]"
                    }
                    for col, h_text in enumerate(headers):
                        h_item = self.table.horizontalHeaderItem(col)
                        if h_item:
                            if h_text in tooltips:
                                h_item.setToolTip(tooltips[h_text])
                            elif h_text.startswith("U_"):
                                h_item.setToolTip(f"[{h_text[2:]}] 하중 케이스의 컴플라이언스 (변형 에너지)")
                            elif h_text.startswith("D_"):
                                h_item.setToolTip(f"[{h_text[2:]}] 하중 케이스의 최대 변위 [mm]")
                            elif h_text.startswith("S_"):
                                h_item.setToolTip(f"[{h_text[2:]}] 하중 케이스의 최대 폰-미세스 응력 [MPa]")

            n_iters_so_far = len(self.history["iter"])  # 이미 append된 후의 길이
            for name, res in cases_data.items():
                if name not in self.history["cases"]:
                    # 반복 ESL 모드: 이터레이션마다 ESL 케이스 이름이 바뀜 → 동적 등록
                    # 이전 이터레이션분을 0으로 패딩
                    self.history["cases"][name] = {
                        "U":          [0.0] * (n_iters_so_far - 1),
                        "max_disp":   [0.0] * (n_iters_so_far - 1),
                        "max_stress": [0.0] * (n_iters_so_far - 1),
                    }
                self.history["cases"][name]["U"].append(res["U"])
                self.history["cases"][name]["max_disp"].append(res["max_disp"])
                self.history["cases"][name]["max_stress"].append(res["max_stress"])

            # 이번 이터레이션에 없는 케이스는 0으로 패딩 (길이 불일치 방지)
            for name, hist in self.history["cases"].items():
                if len(hist["U"]) < n_iters_so_far:
                    hist["U"].append(0.0)
                    hist["max_disp"].append(0.0)
                    hist["max_stress"].append(0.0)

            # ── Height Distribution 스냅샷 저장 ──
            coords = data.get("coords")
            heights = data.get("heights")
            if coords is not None and heights is not None:
                self.coords = coords
                self.current_h = heights
                self.height_snapshots.append({
                    "iter": it,
                    "coords": coords.copy(),
                    "heights": heights.copy(),
                })
                # 드롭다운에 이터레이션 항목 추가 (최신이 맨 위)
                if self.height_iter_combo:
                    self.height_iter_combo.blockSignals(True)
                    self.height_iter_combo.insertItem(1, f"Iter {it:03d}")
                    self.height_iter_combo.blockSignals(False)
                    
                if self.height_slider:
                    self.height_slider.blockSignals(True)
                    self.height_slider.setMaximum(it + 1) # 0:Latest, 1:Iter0, ..., N+1:IterN
                    if self.height_iter_combo.currentIndex() == 0:
                        self.height_slider.setValue(it + 1)
                    self.height_slider.blockSignals(False)

            # ── 테이블 행 추가 ──
            if self.table and self.table.columnCount() > 0:
                row = self.table.rowCount()
                self.table.insertRow(row)
                ar = data.get("area_ratio", 0.0)
                vals = [str(it), f"{data['compliance']:.3e}", f"{data['avg_h']:.2f}",
                        f"{data['max_h']:.2f}", f"{data['dx']:.4f}", f"{ar:.3f}"]
                for col, v in enumerate(vals):
                    self.table.setItem(row, col, QTableWidgetItem(v))
                c_off = len(vals)
                for name in self.case_names:
                    res = cases_data.get(name, {"U": 0, "max_disp": 0, "max_stress": 0})
                    self.table.setItem(row, c_off,   QTableWidgetItem(f"{res['U']:.2e}"))
                    self.table.setItem(row, c_off+1, QTableWidgetItem(f"{res['max_disp']:.2f}"))
                    self.table.setItem(row, c_off+2, QTableWidgetItem(f"{res['max_stress']:.1f}"))
                    c_off += 3
                self.table.scrollToBottom()

            # ── Modal 테이블 갱신 ──
            if self.modal_table:
                for i, f in enumerate(freqs):
                    if i < 10:
                        self.modal_table.setItem(i, 2, QTableWidgetItem(f"{f:.2f}"))

            # ── 그래프 갱신 ──
            self._update_curves()
            self._update_height_plot()

        except Exception as e:
            print(f" -> [Monitor Error] {e}")

    def _clear_history(self):
        """모든 데이터와 UI 테이블/그래프를 초기화합니다."""
        self.history = {
            "iter": [], "compliance": [], "avg_h": [], "max_h": [], "dx": [],
            "area_ratio": [], "frequencies": [], "cases": {},
        }
        self.height_snapshots = []
        self.ref_freqs = None
        self.case_names = []
        
        if self.table:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
        
        if self.modal_table:
            for i in range(10):
                self.modal_table.setItem(i, 1, QTableWidgetItem(""))
                self.modal_table.setItem(i, 2, QTableWidgetItem(""))
        
        if self.height_iter_combo:
            self.height_iter_combo.blockSignals(True)
            self.height_iter_combo.clear()
            self.height_iter_combo.addItem("Latest")
            self.height_iter_combo.blockSignals(False)
        
        if self.height_slider:
            self.height_slider.blockSignals(True)
            self.height_slider.setMinimum(0)
            self.height_slider.setMaximum(0)
            self.height_slider.setValue(0)
            self.height_slider.blockSignals(False)
        
        if self.status_label:
            self.status_label.setText("Status: Running (Reset)")
            self.status_label.setStyleSheet("font-weight: bold; color: blue;")
        
        if self.stop_btn:
            self.stop_btn.setEnabled(True)
            self.stop_btn.setText("Request STOP")

        # 그래프 축 초기화
        if self.curve_canvas: self.curve_canvas.ax.clear()
        if self.height_canvas: 
            self.height_canvas.ax.clear()
            if self._height_colorbar:
                try: self._height_colorbar.remove()
                except: pass
                self._height_colorbar = None

    # ────────────────────────────────────────────────────────────────────────
    # 수렴 커브 탭
    # ────────────────────────────────────────────────────────────────────────

    def _update_curves(self):
        if not self.curve_canvas or not self.history["iter"]:
            return
        metric = self.metric_combo.currentText()
        ax = self.curve_canvas.ax
        ax.clear()
        iters = self.history["iter"]

        if metric == "ALL (Normalized)":
            metrics_to_plot = [
                ("Compliance", self.history["compliance"], 'o-'),
                ("Avg_h",      self.history["avg_h"],      's-'),
                ("Max_h",      self.history["max_h"],      '^-'),
                ("dx",         self.history["dx"],         'd-'),
                ("Area_Ratio", self.history["area_ratio"], 'x-'),
            ]
            # 하중 케이스별 컴플라이언스
            fmts_case = ['--', '-.', ':', '-', '--', '-.', ':', '-', '--']
            for i, name in enumerate(self.case_names):
                u_list = self.history["cases"][name]["U"]
                if u_list:
                    metrics_to_plot.append((f"U_{name}", u_list, fmts_case[i % len(fmts_case)]))
            # 1차 고유진동수
            RIGID_BODY_THRESHOLD = 0.5  # Hz — 강체 운동 모드 제외 기준
            freq_arr = np.array(self.history["frequencies"])
            if freq_arr.ndim == 2 and freq_arr.shape[0] > 0 and freq_arr.shape[1] > 0:
                # 마지막 이터레이션 기준으로 구조 모드인 열 인덱스 선택
                last_freqs = freq_arr[-1]
                struct_cols = np.where(last_freqs >= RIGID_BODY_THRESHOLD)[0]
                if len(struct_cols) > 0:
                    metrics_to_plot.append(("Freq_1", list(freq_arr[:, struct_cols[0]]), 'P-'))

            for label, data_list, fmt in metrics_to_plot:
                arr = np.array(data_list)
                if len(arr) > 0:
                    max_val = np.max(np.abs(arr))
                    norm_arr = arr / max_val if max_val > 1e-12 else arr
                    ax.plot(iters, norm_arr, fmt, label=label, markersize=3, linewidth=1)
            ax.set_ylabel("Normalized Value")
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
                      fontsize=7, borderaxespad=0)
        elif metric == "Natural Frequencies":
            RIGID_BODY_THRESHOLD = 0.5  # Hz
            freq_arr = np.array(self.history["frequencies"])
            if freq_arr.ndim == 2 and freq_arr.shape[1] > 0:
                last_freqs = freq_arr[-1]
                struct_cols = np.where(last_freqs >= RIGID_BODY_THRESHOLD)[0]
                for rank, col in enumerate(struct_cols[:10]):
                    ax.plot(iters, freq_arr[:, col], label=f"M{rank+1}({freq_arr[-1, col]:.1f}Hz)")
                ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_ylabel("Frequency (Hz)")
        elif metric == "Area_Ratio":
            ax.plot(iters, self.history["area_ratio"], 'D-', color='darkorange', label="Area Ratio")
            ax.set_ylabel("Bead Area Ratio")
            ax.set_ylim(0, 1.05)
            ax.axhline(y=self.history["area_ratio"][0] if self.history["area_ratio"] else 0.3,
                       color='gray', linestyle='--', alpha=0.5, label="Target")
            ax.legend()
        elif metric in ["Compliance", "Avg_h", "Max_h", "dx"]:
            ax.plot(iters, self.history[metric.lower()], 'o-', label=metric)
            if metric == "Compliance": ax.set_ylabel("Compliance (N·mm)")
            elif metric == "Avg_h": ax.set_ylabel("Average Height (mm)")
            elif metric == "Max_h": ax.set_ylabel("Maximum Height (mm)")
            elif metric == "dx": ax.set_ylabel("Density Change (dx)")
        else:
            for name in self.case_names:
                if metric.endswith(name):
                    m_type = metric.split('_')[0]
                    key = "U" if m_type == "U" else ("max_disp" if m_type == "Disp" else "max_stress")
                    ax.plot(iters, self.history["cases"][name][key], 's-', label=metric)
                    if key == "U":
                        ax.set_ylabel("Compliance (N·mm)")
                    elif key == "max_disp":
                        ax.set_ylabel("Maximum Displacement (mm)")
                    elif key == "max_stress":
                        ax.set_ylabel("Max Von-Mises Stress (MPa)")
                    break

        ax.set_title(f"Optimization Trend: {metric}")
        ax.set_xlabel("Iteration")
        ax.minorticks_on()
        ax.grid(True, which='major', linestyle='-', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        self.curve_canvas.fig.tight_layout()
        self.curve_canvas.draw()

    # ────────────────────────────────────────────────────────────────────────
    # Height Distribution 탭
    # ────────────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────────────
    # Height Distribution 제어 핸들러
    # ────────────────────────────────────────────────────────────────────────

    def _on_height_combo_changed(self, idx):
        """콤보박스 변경 시 슬라이더 동기화 및 그래프 갱신."""
        if self.height_slider and not self.height_slider.signalsBlocked():
            self.height_slider.blockSignals(True)
            # 콤보박스 인덱스 0: Latest -> 슬라이더 최대값
            # 콤보박스 인덱스 1: Iter N -> 슬라이더 N+1
            # 콤보박스 인덱스 M: Iter N-M+1 -> 슬라이더 N-M+2
            # 슬라이더 값 = (최대값) - idx + (일부 오프셋)? 
            # 규칙: 슬라이더 0 = 최신(Latest), 1 = Iter 0, 2 = Iter 1 ... 이면 직관적임.
            # 하지만 현재 콤보박스는 0:Latest, 1:Iter N, 2:Iter N-1 ... 임.
            # 직관적으로 슬라이더 오른쪽이 최신이게 하려면:
            # 슬라이더 값 0: Iter 0, 1: Iter 1 ... N: Iter N, N+1: Latest
            max_val = self.height_slider.maximum()
            self.height_slider.setValue(max_val - idx)
            self.height_slider.blockSignals(False)
        self._update_height_plot()

    def _on_height_slider_changed(self, val):
        """슬라이더 변경 시 콤보박스 동기화 및 그래프 갱신."""
        if self.height_iter_combo and not self.height_iter_combo.signalsBlocked():
            self.height_iter_combo.blockSignals(True)
            max_val = self.height_slider.maximum()
            self.height_iter_combo.setCurrentIndex(max_val - val)
            self.height_iter_combo.blockSignals(False)
        self._update_height_plot()

    def _on_prev_height(self):
        """이전 이터레이션 (슬라이더 감소/콤보박스 인덱스 증가)"""
        idx = self.height_iter_combo.currentIndex()
        if idx < self.height_iter_combo.count() - 1:
            self.height_iter_combo.setCurrentIndex(idx + 1)

    def _on_next_height(self):
        """다음 이터레이션 (슬라이더 증가/콤보박스 인덱스 감소)"""
        idx = self.height_iter_combo.currentIndex()
        if idx > 0:
            self.height_iter_combo.setCurrentIndex(idx - 1)

    def _update_height_plot(self):
        if not self.height_canvas or not self.height_snapshots:
            return

        # 드롭다운 선택에 따른 스냅샷 결정
        combo_idx = self.height_iter_combo.currentIndex() if self.height_iter_combo else 0
        if combo_idx == 0:
            # "Latest" 선택
            snap = self.height_snapshots[-1]
        else:
            # "Iter NNN" 선택: combo_idx=1 이 가장 최신, 높은 인덱스일수록 과거
            snap_idx = len(self.height_snapshots) - combo_idx
            snap_idx = max(0, min(snap_idx, len(self.height_snapshots) - 1))
            snap = self.height_snapshots[snap_idx]

        coords  = snap["coords"]
        heights = snap["heights"]
        it_label = snap["iter"]

        # 노드 간격 기반 마커 크기 계산 (사각형, data 좌표 기준)
        spacing = _estimate_node_spacing(coords)

        ax = self.height_canvas.ax
        ax.clear()

        # scatter 마커 크기(s)는 포인트^2 단위 → 데이터 좌표 spacing을 포인트로 변환
        # 피규어 DPI와 좌표 범위를 이용한 변환
        fig = self.height_canvas.fig
        fig_w_inch = fig.get_figwidth()
        data_range_x = coords[:, 0].max() - coords[:, 0].min() + spacing
        pts_per_data = (fig_w_inch * fig.dpi) / data_range_x if data_range_x > 0 else 1.0
        marker_pts = spacing * pts_per_data * 0.85  # 85%로 약간 여백
        marker_pts = max(1.0, marker_pts - 1.0)     # 계산값에서 1픽셀(pt) 차감
        marker_size_sq = marker_pts ** 2

        # 0을 중앙(보통 흰색)으로 두기 위한 대칭 범위(vmin, vmax) 설정
        max_abs_h = max(1e-5, float(np.max(np.abs(heights))))

        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=heights, cmap='coolwarm',
            marker='s',
            s=marker_size_sq,
            linewidths=0,
            vmin=-max_abs_h,
            vmax=max_abs_h,
        )

        # 등고선(Contour) 추가
        if np.max(np.abs(heights)) > 1e-3:
            try:
                ax.tricontour(
                    coords[:, 0], coords[:, 1], heights,
                    levels=10, colors='black', linewidths=0.5, alpha=0.5
                )
            except Exception:
                pass

        # 컬러바 관리 (중복 생성 방지)
        if self._height_colorbar is None:
            self._height_colorbar = fig.colorbar(sc, ax=ax)
            self._height_colorbar.set_label("Bead Height (mm)  [+: outward / −: inward]")
        else:
            self._height_colorbar.update_normal(sc)

        ax.set_aspect('equal')
        ax.set_title(f"Height Distribution — Iter {it_label}")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        fig.tight_layout()
        self.height_canvas.draw()
        self.height_canvas.draw_idle()  # 강제 갱신 보장


# ────────────────────────────────────────────────────────────────────────────
# IPC: Queue → Signal 브리지
# ────────────────────────────────────────────────────────────────────────────

class MonitorDataHandler(QObject):
    data_received = Signal(dict)


def start_monitor_ui(queue, stop_event=None):
    """
    별도 프로세스에서 PySide6 UI를 실행합니다.

    Parameters
    ----------
    queue : multiprocessing.Queue
        솔버로부터 이터레이션 데이터를 수신하는 큐.
        데이터는 dict 형식이며, "STOP" 문자열 수신 시 종료합니다.
    stop_event : multiprocessing.Event, optional
        사용자가 GUI를 닫을 때 솔버에 중단 신호를 보내는 이벤트.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    window = WHTMonitorWindow(stop_event=stop_event)
    window.show()

    class Receiver(QThread):
        def __init__(self, q):
            super().__init__()
            self.q = q
            self.handler = MonitorDataHandler()

        def run(self):
            while True:
                try:
                    data = self.q.get()
                    if isinstance(data, str) and data == "STOP":
                        self.handler.data_received.emit({"status": "STOP"})
                        # 루프 종료하지 않음 — 창은 사용자가 직접 닫을 때까지 유지
                        return
                    self.handler.data_received.emit(data)
                except Exception:
                    return

    receiver = Receiver(queue)
    receiver.handler.data_received.connect(window.update_data)
    receiver.start()
    sys.exit(app.exec())
