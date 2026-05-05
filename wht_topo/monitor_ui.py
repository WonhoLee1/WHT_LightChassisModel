# -*- coding: utf-8 -*-
"""
monitor_ui.py
=============
최적화 프로세스 실시간 모니터링을 위한 PySide6 기반 UI 모듈 (V2.1 - 안정성 강화 버전).
초기화 순서 보강 및 데이터 수신 타이밍 문제를 해결했습니다.
"""

import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QTabWidget, QHeaderView, QComboBox, QLabel
)
from PySide6.QtCore import Qt, Signal, QObject, QThread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import koreanize_matplotlib

plt.rcParams['font.size'] = 9

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()

class WHTMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WHT Topography Optimization Monitor (V2.1)")
        self.resize(1100, 800)

        # 1. 속성 초기 선언 (AttributeError 방지)
        self.history = {
            "iter": [], "compliance": [], "avg_h": [], "max_h": [], "dx": [],
            "frequencies": [], "cases": {}
        }
        self.ref_freqs = None
        self.coords = None
        self.current_h = None
        self.case_names = []
        self.height_canvas = None
        self.curve_canvas = None
        self.table = None
        self.modal_table = None
        self.metric_combo = None

        # 2. UI 초기화
        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Summary Table
        self.table_tab = QWidget()
        self.table_layout = QVBoxLayout(self.table_tab)
        self.table = QTableWidget(0, 0)
        self.table_layout.addWidget(self.table)
        self.tabs.addTab(self.table_tab, "Summary Table")

        # Tab 2: Convergence Curve
        self.curve_tab = QWidget()
        self.curve_layout = QVBoxLayout(self.curve_tab)
        ctrl_layout = QHBoxLayout()
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Compliance", "Avg_h", "Max_h", "dx", "Natural Frequencies"])
        self.metric_combo.currentTextChanged.connect(self._update_curves)
        ctrl_layout.addWidget(QLabel("Select Metric:"))
        ctrl_layout.addWidget(self.metric_combo)
        ctrl_layout.addStretch()
        self.curve_layout.addLayout(ctrl_layout)
        self.curve_canvas = PlotCanvas(self.curve_tab)
        self.curve_layout.addWidget(self.curve_canvas)
        self.tabs.addTab(self.curve_tab, "Convergence Curve")

        # Tab 3: Height Distribution
        self.height_tab = QWidget()
        self.height_layout = QVBoxLayout(self.height_tab)
        self.height_canvas = PlotCanvas(self.height_tab)
        self.height_layout.addWidget(self.height_canvas)
        self.tabs.addTab(self.height_tab, "Height Distribution")

        # Tab 4: Modal Analysis
        self.modal_tab = QWidget()
        self.modal_layout = QVBoxLayout(self.modal_tab)
        self.modal_table = QTableWidget(10, 3)
        self.modal_table.setHorizontalHeaderLabels(["Mode", "Ref. (Hz)", "Current (Hz)"])
        self.modal_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for i in range(10):
            self.modal_table.setItem(i, 0, QTableWidgetItem(f"Mode {i+1}"))
        self.modal_layout.addWidget(self.modal_table)
        self.tabs.addTab(self.modal_tab, "Modal Analysis")

    def update_data(self, data: dict):
        try:
            it = data["iter"]
            self.history["iter"].append(it)
            self.history["compliance"].append(data["compliance"])
            self.history["avg_h"].append(data["avg_h"])
            self.history["max_h"].append(data["max_h"])
            self.history["dx"].append(data["dx"])
            
            freqs = data.get("frequencies", [])
            self.history["frequencies"].append(freqs)
            
            if it == 0 and self.ref_freqs is None:
                self.ref_freqs = freqs
                if self.modal_table:
                    for i, f in enumerate(freqs):
                        if i < 10: self.modal_table.setItem(i, 1, QTableWidgetItem(f"{f:.2f}"))

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
                    headers = ["Iter", "C_total", "Avg_h", "Max_h", "dx"]
                    for name in self.case_names:
                        headers += [f"U_{name}", f"D_{name}", f"S_{name}"]
                    self.table.setColumnCount(len(headers))
                    self.table.setHorizontalHeaderLabels(headers)

            for name, res in cases_data.items():
                self.history["cases"][name]["U"].append(res["U"])
                self.history["cases"][name]["max_disp"].append(res["max_disp"])
                self.history["cases"][name]["max_stress"].append(res["max_stress"])

            self.coords = data.get("coords")
            self.current_h = data.get("heights")

            # ── UI 업데이트 (객체 존재 여부 체크) ──
            if self.table:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(str(it)))
                self.table.setItem(row, 1, QTableWidgetItem(f"{data['compliance']:.3e}"))
                self.table.setItem(row, 2, QTableWidgetItem(f"{data['avg_h']:.2f}"))
                self.table.setItem(row, 3, QTableWidgetItem(f"{data['max_h']:.2f}"))
                self.table.setItem(row, 4, QTableWidgetItem(f"{data['dx']:.4f}"))
                c_off = 5
                for name in self.case_names:
                    res = cases_data.get(name, {"U": 0, "max_disp": 0, "max_stress": 0})
                    self.table.setItem(row, c_off, QTableWidgetItem(f"{res['U']:.2e}"))
                    self.table.setItem(row, c_off+1, QTableWidgetItem(f"{res['max_disp']:.2f}"))
                    self.table.setItem(row, c_off+2, QTableWidgetItem(f"{res['max_stress']:.1f}"))
                    c_off += 3
                self.table.scrollToBottom()

            if self.modal_table:
                for i, f in enumerate(freqs):
                    if i < 10: self.modal_table.setItem(i, 2, QTableWidgetItem(f"{f:.2f}"))

            self._update_curves()
            self._update_height_plot()
        except Exception as e:
            print(f" -> [Monitor Error] {e}")

    def _update_curves(self):
        if not self.curve_canvas or not self.history["iter"]: return
        metric = self.metric_combo.currentText()
        ax = self.curve_canvas.ax
        ax.clear()
        iters = self.history["iter"]
        
        if metric == "Natural Frequencies":
            freq_history = np.array(self.history["frequencies"])
            if freq_history.ndim == 2:
                for i in range(min(10, freq_history.shape[1])):
                    ax.plot(iters, freq_history[:, i], label=f"M{i+1}")
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        elif metric in ["Compliance", "Avg_h", "Max_h", "dx"]:
            ax.plot(iters, self.history[metric.lower()], 'o-', label=metric)
        else:
            for name in self.case_names:
                if metric.endswith(name):
                    m_type = metric.split('_')[0]
                    key = "U" if m_type == "U" else ("max_disp" if m_type == "Disp" else "max_stress")
                    ax.plot(iters, self.history["cases"][name][key], 's-', label=metric)
                    break
        ax.set_title(f"Optimization Trend: {metric}")
        ax.grid(True, alpha=0.3)
        self.curve_canvas.fig.tight_layout()
        self.curve_canvas.draw()

    def _update_height_plot(self):
        if not self.height_canvas or self.coords is None or self.current_h is None: return
        ax = self.height_canvas.ax
        ax.clear()
        sc = ax.scatter(self.coords[:, 0], self.coords[:, 1], c=self.current_h, cmap='viridis', s=10)
        if not hasattr(self, 'colorbar'):
            self.colorbar = self.height_canvas.fig.colorbar(sc, ax=ax)
            self.colorbar.set_label("Bead Height (mm)")
        else:
            self.colorbar.update_normal(sc)
        ax.set_aspect('equal')
        self.height_canvas.fig.tight_layout()
        self.height_canvas.draw()

class MonitorDataHandler(QObject):
    data_received = Signal(dict)

def start_monitor_ui(queue):
    app = QApplication(sys.argv)
    window = WHTMonitorWindow()
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
                    if data == "STOP": break
                    self.handler.data_received.emit(data)
                except: break
    receiver = Receiver(queue)
    receiver.handler.data_received.connect(window.update_data)
    receiver.start()
    sys.exit(app.exec())
