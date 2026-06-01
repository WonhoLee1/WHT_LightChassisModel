"""
wht_visualizer.py
=================
WHT Universal FEM Framework — Visualization Module (PyVistaQt)

Provides a Qt-based professional visualization main frame using pyvistaqt.
Supports background plotting, standard menus, and real-time updates.
"""

import os
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import matplotlib.pyplot as plt
import koreanize_matplotlib
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Tuple

# Force qtpy to use PySide6 backend
os.environ['QT_API'] = 'pyside6'

try:
    from qtpy import QtWidgets, QtCore, QtGui
except ImportError:
    QtWidgets = None

# TEMPORARY BYPASS: Disable qt-material to diagnose hanging issues.
HAS_QT_MATERIAL = False 

if TYPE_CHECKING:
    from wht_converter.wht_models import WHTResultData

class WHTRangeDialog(QtWidgets.QDialog):
    """
    [WHT Premium UI] Scalar range adjustment dialog.

    - Static 모드: 사용자가 입력한 min/max 고정 범위를 Apply 시 적용.
                   Field Min/Max 는 프레임 이동으로 누적된 전체-프레임 범위 표시.
    - Dynamic 모드: min/max 입력 비활성, colorbar 범위는 자동(Robust/Auto) 유지.
    - Apply: 설정 적용 후 닫기.  Cancel: 이전 상태 복원 후 닫기.
    """
    def __init__(self, parent_vis, field_name: str, dummy_group, get_limits_fn, get_robust_fn):
        super().__init__(parent_vis.plotter.app_window)
        self.vis = parent_vis
        self.field = field_name
        self.dummy_group = dummy_group
        self.get_limits_fn = get_limits_fn
        self.get_robust_fn = get_robust_fn

        self.setWindowTitle(f"Adjust Range: {field_name}")
        self.setMinimumWidth(560)

        # 열릴 때 현재 상태 저장 (Cancel 용)
        self._saved_mode = self.vis.current_range_mode
        self._saved_min  = self.vis.range_min
        self._saved_max  = self.vis.range_max

        # 전체-프레임 누적 범위 (Field Min/Max 라벨용)
        self.af_min, self.af_max = self.vis.get_allframe_range(field_name)

        # 슬라이더 매핑 범위 (af 기준으로 여유 50%)
        span = self.af_max - self.af_min
        if span <= 0: span = 1e-6
        self.s_min = self.af_min - 0.5 * span
        self.s_max = self.af_max + 0.5 * span

        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # ── 1. Static / Dynamic 라디오 버튼 ─────────────────────────────
        grp_mode = QtWidgets.QGroupBox("Range Mode")
        h_mode = QtWidgets.QHBoxLayout(grp_mode)
        self.radio_static  = QtWidgets.QRadioButton("Static  (Fixed Min/Max)")
        self.radio_dynamic = QtWidgets.QRadioButton("Dynamic  (Auto)")
        h_mode.addWidget(self.radio_static)
        h_mode.addWidget(self.radio_dynamic)
        is_static = (self.vis.current_range_mode == "Static (Fixed)")
        self.radio_static.setChecked(is_static)
        self.radio_dynamic.setChecked(not is_static)
        self.radio_static.toggled.connect(self._on_mode_toggled)
        layout.addWidget(grp_mode)

        # ── 2. Min / Max 입력 그룹 ────────────────────────────────────────
        def create_entry(label, current_val, allframe_val):
            group = QtWidgets.QGroupBox(label)
            vbox = QtWidgets.QVBoxLayout(group)

            lbl_limit = QtWidgets.QLabel(
                f"Field {label.split()[0]} (All Frames): {allframe_val:.4e}")
            lbl_limit.setStyleSheet(
                "color: #888888; font-size: 8pt; font-family: 'Consolas', monospace;")
            vbox.addWidget(lbl_limit)

            hbox = QtWidgets.QHBoxLayout()
            edit = QtWidgets.QLineEdit(f"{current_val:.4e}")
            val_validator = QtGui.QDoubleValidator()
            val_validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
            edit.setValidator(val_validator)
            edit.setMinimumWidth(120)

            btn_snap = QtWidgets.QPushButton("⏮️" if "Minimum" in label else "⏭️")
            btn_snap.setFixedWidth(40)
            btn_snap.setToolTip(f"Snap to all-frame field {label.lower()}")

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 1000)

            def update_slider_from_edit():
                try:
                    v = float(edit.text())
                    pct = (v - self.s_min) / (self.s_max - self.s_min)
                    slider.blockSignals(True)
                    slider.setValue(int(np.clip(pct, 0, 1) * 1000))
                    slider.blockSignals(False)
                except ValueError:
                    pass

            def update_edit_from_slider(v):
                val = self.s_min + (v / 1000.0) * (self.s_max - self.s_min)
                edit.blockSignals(True)
                edit.setText(f"{val:.4e}")
                edit.blockSignals(False)

            def snap_to_limit():
                edit.setText(f"{allframe_val:.4e}")
                update_slider_from_edit()

            edit.textChanged.connect(update_slider_from_edit)
            slider.valueChanged.connect(update_edit_from_slider)
            btn_snap.clicked.connect(snap_to_limit)

            pct_init = (current_val - self.s_min) / (self.s_max - self.s_min)
            slider.blockSignals(True)
            slider.setValue(int(np.clip(pct_init, 0, 1) * 1000))
            slider.blockSignals(False)

            hbox.addWidget(slider, 3)
            hbox.addWidget(edit, 1)
            hbox.addWidget(btn_snap)
            vbox.addLayout(hbox)
            return group, edit, slider

        self.grp_min, self.edit_min, self.slider_min = create_entry(
            "Minimum Threshold", self.vis.range_min, self.af_min)
        self.grp_max, self.edit_max, self.slider_max = create_entry(
            "Maximum Threshold", self.vis.range_max, self.af_max)
        layout.addWidget(self.grp_min)
        layout.addWidget(self.grp_max)

        # ── 3. Robust 도구 그룹 ───────────────────────────────────────────
        group_robust = QtWidgets.QGroupBox("Statistical Robustness  (Static 모드 전용)")
        h_robust = QtWidgets.QHBoxLayout(group_robust)
        self.spin_robust = QtWidgets.QDoubleSpinBox()
        self.spin_robust.setRange(50.0, 100.0)
        self.spin_robust.setValue(self.dummy_group.get_robust_pct())
        self.spin_robust.setSuffix(" %")
        self.spin_robust.setToolTip("데이터 분포의 중심 백분율 (예: 98% → 상/하위 1% 특이점 제외)")
        btn_robust = QtWidgets.QPushButton("Apply Robust Auto")
        btn_robust.clicked.connect(self._apply_robust)
        btn_global = QtWidgets.QPushButton("Full Global Auto")
        btn_global.setToolTip("전체-프레임 절대 최소/최대값으로 범위 설정")
        btn_global.clicked.connect(self._apply_global)
        btn_find = QtWidgets.QPushButton("🔍 Find Outliers")
        btn_find.clicked.connect(self._find_outliers)
        h_robust.addWidget(QtWidgets.QLabel("Threshold:"))
        h_robust.addWidget(self.spin_robust)
        h_robust.addWidget(btn_robust)
        h_robust.addWidget(btn_global)
        h_robust.addWidget(btn_find)
        self.group_robust = group_robust
        layout.addWidget(group_robust)

        # ── 4. Apply / Cancel 버튼 ────────────────────────────────────────
        h_btns = QtWidgets.QHBoxLayout()
        btn_apply  = QtWidgets.QPushButton("Apply")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_apply.setDefault(True)
        btn_apply.clicked.connect(self._on_apply)
        btn_cancel.clicked.connect(self._on_cancel)
        h_btns.addStretch()
        h_btns.addWidget(btn_apply)
        h_btns.addWidget(btn_cancel)
        layout.addLayout(h_btns)

        self.finished.connect(self._cleanup)

        # 초기 활성화 상태 동기화
        self._on_mode_toggled(self.radio_static.isChecked())

    def _on_mode_toggled(self, static_checked: bool):
        """Static/Dynamic 전환에 따라 min/max 입력 활성화 여부 제어."""
        self.grp_min.setEnabled(static_checked)
        self.grp_max.setEnabled(static_checked)
        self.group_robust.setEnabled(static_checked)

    def _on_apply(self):
        if self.radio_static.isChecked():
            try:
                self.vis.range_min = float(self.edit_min.text())
                self.vis.range_max = float(self.edit_max.text())
            except ValueError:
                pass
            self.vis.current_range_mode = "Static (Fixed)"
        else:
            self.vis.current_range_mode = "Dynamic (Auto)"
        self.vis._apply_colorbar_range(show_stats=True)
        self.accept()

    def _on_cancel(self):
        # 열리기 전 상태로 복원
        self.vis.current_range_mode = self._saved_mode
        self.vis.range_min = self._saved_min
        self.vis.range_max = self._saved_max
        self.vis._apply_colorbar_range()
        self.reject()

    def _cleanup(self):
        self.vis.clear_outliers()

    def _apply_robust(self):
        pct = self.spin_robust.value()
        p_low = (100.0 - pct) / 2.0
        rng = self.vis._calculate_robust_range(self.field, p_low=p_low, p_high=100.0 - p_low)
        self.edit_min.setText(f"{rng[0]:.4e}")
        self.edit_max.setText(f"{rng[1]:.4e}")

    def _apply_global(self):
        self.edit_min.setText(f"{self.af_min:.4e}")
        self.edit_max.setText(f"{self.af_max:.4e}")

    def _find_outliers(self):
        try:
            threshold = float(self.edit_max.text())
            self.vis._highlight_outliers(self.field, threshold)
        except ValueError:
            pass


class WHTVisualizer:
    """
    Professional Visualization Hub using PyVistaQt.
    Provides a main window with menus and non-blocking background plotting.
    
    [Field Naming Conventions (필드 이름 명명 규칙 및 자동 생성)]
    결과 데이터(WHTResultData)의 point_data 및 cell_data 필드 이름은 다음 규칙을 따를 때 
    UI에서 카테고리 및 컴포넌트로 깔끔하게 자동 분류/생성됩니다:
    1. Vector 데이터 (예: "Displacement", "ModeShape"): 
       - 데이터 형상이 (T, N, 3) 이상일 경우, 시각화 모듈이 자동으로 
         "_X", "_Y", "_Z", "_Magnitude" 접미사가 붙은 하위 필드를 계산하여 생성합니다.
    2. Tensor 데이터 (예: "Stress", "Strain"):
       - 데이터 형상이 (T, M, 6)일 경우, 시각화 모듈이 자동으로 
         "_XX", "_YY", "_ZZ", "_VonMises", "_Max_Principal", "_Min_Principal"을 생성합니다.
    3. 중복 방지: 
       - 원본 필드 이름(Base name) 자체에 위 접미사들을 직접 붙여서 전달하는 것은 권장하지 않습니다. 
         (UI의 Category 목록이 중복해서 나타나는 것을 방지하기 위함)
    """
    
    def __init__(self, title: str = "WHT FEM Professional Visualizer", show=True):
        # 1. Setup Global Theme (Premium Aesthetics)
        pv.set_plot_theme("dark")
        pv.global_theme.background = 'black'
        pv.global_theme.font.color = 'white'
        pv.global_theme.font.size = 12
        pv.global_theme.show_edges = True
        pv.global_theme.edge_color = 'darkgray'
        
        # 2. Initialize BackgroundPlotter First
        # BackgroundPlotter automatically creates a QApplication if one doesn't exist.
        # Force toolbar=False to hide redundant default PyVistaQt buttons.
        self.plotter = BackgroundPlotter(title=title, show=show, toolbar=False)
        self.plotter.set_background('black') # Rule-aligned: Force black as requested
        
        # Setup Main App Icon
        if QtWidgets:
            app_icon = QtGui.QIcon()
            sizes = [16, 24, 32, 48, 64, 128]
            for size in sizes:
                icon_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "resources", f"logo_icon_{size}x{size}.png"))
                if os.path.exists(icon_path):
                    app_icon.addFile(icon_path, QtCore.QSize(size, size))
            self.plotter.app_window.setWindowIcon(app_icon)
        
        # Ensure styling is applied to the active app instance
        app = QtWidgets.QApplication.instance()
        if app:
            # Force high-quality typography via Global StyleSheet
            app.setStyleSheet("""
                QWidget { 
                    font-family: 'Noto Sans CJK KR', 'Noto Sans KR', 'Malgun Gothic', sans-serif; 
                    font-size: 9pt; 
                }
            """)
            
        # 2. Appearance & Aesthetics (Strict Initial State)
        pv.global_theme.background = 'black'
        pv.global_theme.font.color = 'white'
        self.plotter.set_background('black')
        for renderer in self.plotter.renderers:
            renderer.set_background('black')
        self.plotter.add_axes(color='white')
        self.plotter.render()
        # self._setup_ui_enhancements()
        self.actor = None
        
        # Data State Management
        self.result_data: Optional["WHTResultData"] = None
        self.current_timestep: int = 0
        self.base_coords: Optional[np.ndarray] = None
        self._is_updating = False
        self._is_ready = False
        self._kabsch = None  # KabschPreprocessor instance (강체 변환 시각화용)
        
        # State: Multi-Part Management
        self.parts: Dict[str, dict] = {} # {name: {"mesh": Grid, "actor": Actor, "style": dict}}
        self._bg_is_dark = True
        self.actors_misc = {} # {"BC": Actor, "Load": Actor}
        
        # Colorbar Settings
        self.cb_title_size = 12
        self.cb_label_size = 10
        self.cb_mode = "Continuous"
        self.cb_levels = 10
        self.cb_decimals = 1

        # Bead discrete levels (0 = continuous)
        self._bead_steps = 0

        # 3. Setup UI Pipeline (Order Matters!)
        self._setup_tabbed_dock()     # Creates self.list_parts
        self._setup_playback_ui()     # Creates playback controls
        self._setup_view_controls()
        self._setup_toolbar()         # Connects to list_parts
        self._setup_menubar()         # Creates pull-down menus
        
        # Setup Mouse Query Interaction (Surgical hook)
        self._query_label_names = []
        if hasattr(self.plotter, "interactor") and self.plotter.interactor is not None:
            self._orig_double_click = self.plotter.interactor.mouseDoubleClickEvent
            self.plotter.interactor.mouseDoubleClickEvent = self._on_mouse_double_click
            
            self._orig_mouse_move = self.plotter.interactor.mouseMoveEvent
            self.plotter.interactor.mouseMoveEvent = self._on_qt_mouse_move
        
        self._is_ready = True

    def set_bead_discrete_levels(self, bead_steps: int) -> None:
        """
        Bead_Height scalar bar를 이산 레벨 표시로 설정합니다.

        bead_steps=0 → 연속 colormap (기본)
        bead_steps=N → N+1 레벨 discrete colormap
            bead_steps=1 → {0, h_max}  2색
            bead_steps=2 → {0, 0.5, 1} 3색
        """
        self._bead_steps = bead_steps

    def _apply_ui_theme(self, is_dark: bool = True):
        """
        [WHT Exclusive] Dynamic Styling Engine - Native Palette Implementation.
        """
        if not QtWidgets: return
        app = QtWidgets.QApplication.instance()
        if not app: return

        # 1. Define Unique Design Tokens
        if is_dark:
            colors = {
                "bg_dark": "#1e1e1e", "bg_med": "#2b2b2b", "bg_light": "#3d3d3d",
                "accent": "#2a82da", "text_main": "#ffffff", "text_dim": "#aaaaaa",
                "border": "#4d4d4d"
            }
        else:
            colors = {
                "bg_dark": "#f0f0f0", "bg_med": "#e4e4e4", "bg_light": "#ffffff",
                "accent": "#005bb5", "text_main": "#000000", "text_dim": "#555555",
                "border": "#b0b0b0"
            }
        
        # 2. Comprehensive Palette Construction
        app.setStyle("Fusion")
        p = QtGui.QPalette()
        p.setColor(QtGui.QPalette.Window, QtGui.QColor(colors["bg_med"]))
        p.setColor(QtGui.QPalette.WindowText, QtGui.QColor(colors["text_main"]))
        p.setColor(QtGui.QPalette.Base, QtGui.QColor(colors["bg_dark"]))
        p.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(colors["bg_med"]))
        p.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
        p.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
        p.setColor(QtGui.QPalette.Text, QtGui.QColor(colors["text_main"]))
        p.setColor(QtGui.QPalette.Button, QtGui.QColor(colors["bg_light"]))
        p.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(colors["text_main"]))
        p.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
        p.setColor(QtGui.QPalette.Link, QtGui.QColor(colors["accent"]))
        p.setColor(QtGui.QPalette.Highlight, QtGui.QColor(colors["accent"]))
        p.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black if is_dark else QtCore.Qt.white)
        app.setPalette(p)
        
        # 3. Dynamic Style Assembly with Global Visibility
        ui_styles = [f"* {{ color: {colors['text_main']}; }}"]  # Force all text to main color
        
        # Specific Widget Polishing
        ui_styles.append(f"QToolTip {{ background-color: {colors['accent']}; border: 1px solid white; }}")
        ui_styles.append(f"QDockWidget::title {{ background: {colors['bg_dark']}; padding: 9px; font-weight: bold; }}")
        
        ui_styles.append(f"""
            QGroupBox {{ 
                border: 2px solid {colors['border']}; border-radius: 5px; 
                margin-top: 1.3em; padding: 10px 5px; font-weight: bold; 
            }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 15px; padding: 0 5px; color: {colors['accent']}; }}
        """)
        
        ui_styles.append(f"""
            QPushButton, QComboBox, QSpinBox, QDoubleSpinBox {{ 
                background-color: {colors['bg_light']}; border: 1px solid {colors['border']}; 
                border-radius: 4px; padding: 5px 10px; color: {colors['text_main']};
            }}
            QPushButton:hover {{ background-color: {colors['border']}; border-color: {colors['accent']}; }}
            QToolBar {{ background: {colors['bg_med']}; border-bottom: 1px solid {colors['border']}; spacing: 8px; padding: 4px; }}
            QToolButton {{ 
                background-color: transparent; border: 1px solid transparent; 
                border-radius: 4px; padding: 4px 8px; font-weight: bold;
            }}
            QToolButton:hover {{ background-color: {colors['bg_light']}; border: 1px solid {colors['accent']}; }}
            QToolButton:checked {{ background-color: {colors['accent']}; color: {'black' if is_dark else 'white'}; }}
        """)
        
        ui_styles.append(f"""
            QSlider::groove:horizontal {{ background: {colors['bg_dark']}; height: 5px; border-radius: 2px; }}
            QSlider::handle:horizontal {{ 
                background: {colors['accent']}; border: 1px solid {colors['accent']}; 
                width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; 
            }}
        """)
        
        app.setStyleSheet("\n".join(ui_styles))


    def _setup_ui_enhancements(self):
        """Standard visual enhancements for a premium feel."""
        self.plotter.add_axes()
        self.plotter.show_grid(color='white', font_size=10)
        self.plotter.add_text("WHT LightChassisModel Engine", position='upper_right', font_size=8, color='grey')

    def _setup_tabbed_dock(self):
        """Setup a tabbed interface for Properties and Part Management."""
        if not QtWidgets: return
        
        self.dock = QtWidgets.QDockWidget("WHT Inspector", self.plotter.app_window)
        self.plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock)
        
        self.tabs = QtWidgets.QTabWidget()
        self.dock.setWidget(self.tabs)
        
        # --- Tab 1: Properties ---
        self.prop_tab = QtWidgets.QWidget()
        prop_layout = QtWidgets.QVBoxLayout(self.prop_tab)
        prop_layout.setSpacing(2) # User requested ~2px gap
        prop_layout.setContentsMargins(5, 5, 5, 5)
        
        # Deform Group (Renamed from Basics/Deformation)
        group_deform = QtWidgets.QGroupBox("Deform")
        vbox_deform = QtWidgets.QVBoxLayout()
        vbox_deform.setSpacing(2)
        
        # Row 1: Deform Control
        hbox_warp = QtWidgets.QHBoxLayout()
        hbox_warp.setSpacing(5)
        self.chk_warp = QtWidgets.QCheckBox("Use Deform")
        self.chk_warp.setChecked(True)
        self.chk_warp.setToolTip("Enable/Disable mesh warping based on a vector field.")
        self.chk_warp.stateChanged.connect(self._on_warp_toggled)
        
        self.spin_scale = QtWidgets.QDoubleSpinBox()
        self.spin_scale.setRange(-1000.0, 1000.0)
        self.spin_scale.setValue(1.0)
        self.spin_scale.setFixedWidth(80)
        self.spin_scale.setToolTip("Warping scale factor.")
        self.spin_scale.valueChanged.connect(self._on_warp_scale_changed)
        
        hbox_warp.addWidget(self.chk_warp)

        self.chk_rigid_body = QtWidgets.QCheckBox("Rigid Body")
        self.chk_rigid_body.setChecked(False)
        self.chk_rigid_body.setEnabled(False)
        self.chk_rigid_body.setToolTip("Kabsch 강체 변환(R, T)을 시각화에 적용합니다.")
        self.chk_rigid_body.stateChanged.connect(self._apply_warping)
        hbox_warp.addWidget(self.chk_rigid_body)

        # Deform Vector Selection
        self.combo_warp_vec = QtWidgets.QComboBox()
        self.combo_warp_vec.setMinimumWidth(120)
        self.combo_warp_vec.setToolTip("Select the vector field [X,Y,Z] to use for deformation.")
        self.combo_warp_vec.currentTextChanged.connect(self._on_warp_field_changed)
        hbox_warp.addWidget(self.combo_warp_vec)
        
        hbox_warp.addStretch()
        hbox_warp.addWidget(self.spin_scale)
        
        # Row 2: BC & Load
        hbox_bcload = QtWidgets.QHBoxLayout()
        self.chk_bc = QtWidgets.QCheckBox("Boundary Condition")
        self.chk_load = QtWidgets.QCheckBox("Load")
        self.chk_bc.setChecked(True)
        self.chk_load.setChecked(True)
        self.chk_bc.stateChanged.connect(self._update_symbolic_viz)
        self.chk_load.stateChanged.connect(self._update_symbolic_viz)
        
        hbox_bcload.addWidget(self.chk_bc)
        hbox_bcload.addWidget(self.chk_load)
        
        vbox_deform.addLayout(hbox_warp)
        vbox_deform.addLayout(hbox_bcload)
        group_deform.setLayout(vbox_deform)

        
        # Contour/Colorbar Group
        group_contour = QtWidgets.QGroupBox("Fields")
        vbox_contour = QtWidgets.QVBoxLayout()
        vbox_contour.setSpacing(2)
        
        # Dual-Combo System: Category and Component (Split into separate rows for readability)
        hbox_cat = QtWidgets.QHBoxLayout()
        hbox_cat.setSpacing(5)
        hbox_cat.addWidget(QtWidgets.QLabel("Category:"))
        self.combo_category = QtWidgets.QComboBox()
        self.combo_category.currentTextChanged.connect(self._on_category_changed)
        hbox_cat.addWidget(self.combo_category, 1)

        hbox_comp = QtWidgets.QHBoxLayout()
        hbox_comp.setSpacing(5)
        hbox_comp.addWidget(QtWidgets.QLabel("Component:"))
        self.combo_component = QtWidgets.QComboBox()
        self.combo_component.currentTextChanged.connect(self._on_component_changed)
        
        self.btn_adjust_range = QtWidgets.QPushButton("Adjust")
        self.btn_adjust_range.setFixedWidth(55)
        self.btn_adjust_range.clicked.connect(self._open_range_adjust_dialog)
        self.btn_adjust_range.setEnabled(False)
        
        hbox_comp.addWidget(self.combo_component, 1)
        hbox_comp.addWidget(self.btn_adjust_range)
        
        # Internal state for manual ranges (Replacing UI widgets in main panel)
        self.range_min = 0.0
        self.range_max = 1.0
        self.current_range_mode = "Dynamic (Auto)"
        self.robust_pct = 98.0
        # 프레임 이동 누적 전체-프레임 min/max 캐시 {field_name: (min, max)}
        self._field_allframe_range: dict = {}
        
        # --- Colorbar Display Mode Group ---
        hbox_cb_mode = QtWidgets.QHBoxLayout()
        hbox_cb_mode.setSpacing(2)
        
        self.combo_cb_mode = QtWidgets.QComboBox()
        self.combo_cb_mode.addItems(["Continuous", "Discrete"])
        self.combo_cb_mode.currentTextChanged.connect(self._on_cb_style_changed)
        hbox_cb_mode.addWidget(QtWidgets.QLabel("Type:"))
        hbox_cb_mode.addWidget(self.combo_cb_mode)
        
        self.combo_cb_levels = QtWidgets.QComboBox()
        self.combo_cb_levels.setEditable(True)
        self.combo_cb_levels.addItems([str(i) for i in range(3, 16)])
        self.combo_cb_levels.setCurrentText("10")
        self.combo_cb_levels.setEnabled(False)
        self.combo_cb_levels.currentTextChanged.connect(self._on_cb_style_changed)
        hbox_cb_mode.addWidget(QtWidgets.QLabel("Levels:"))
        hbox_cb_mode.addWidget(self.combo_cb_levels)
        
        self.spin_cb_decimals = QtWidgets.QSpinBox()
        self.spin_cb_decimals.setRange(0, 5)
        self.spin_cb_decimals.setValue(1)
        self.spin_cb_decimals.valueChanged.connect(self._on_cb_style_changed)
        hbox_cb_mode.addWidget(QtWidgets.QLabel("Decimals:"))
        hbox_cb_mode.addWidget(self.spin_cb_decimals)
        # -----------------------------------
        
        hbox_cmap = QtWidgets.QHBoxLayout()
        hbox_cmap.setSpacing(2)
        hbox_cmap.addWidget(QtWidgets.QLabel("Colormap:"))
        self.combo_cmap = QtWidgets.QComboBox()
        self.combo_cmap.addItems(['jet', 'coolwarm', 'viridis', 'plasma', 'inferno', 'magma', 'rainbow', 'jet_r', 'coolwarm_r'])
        self.combo_cmap.setCurrentText('jet')
        self.combo_cmap.currentTextChanged.connect(self._on_colormap_changed)
        hbox_cmap.addWidget(self.combo_cmap)
        
        self.btn_cb_font = QtWidgets.QPushButton("CBar Font...")
        self.btn_cb_font.clicked.connect(self._open_cb_font_dialog)
        hbox_cmap.addWidget(self.btn_cb_font)
        
        # Shell Layer Selection (Through-Thickness Integration Point)
        hbox_layer = QtWidgets.QHBoxLayout()
        hbox_layer.setSpacing(2)
        hbox_layer.addWidget(QtWidgets.QLabel("Shell Layer:"))
        self.combo_shell_layer = QtWidgets.QComboBox()
        self.combo_shell_layer.addItems([
            "Upper (+t/2)", "Mid (0)", "Lower (-t/2)",
            "Max Envelope", "Membrane", "Bending"
        ])
        self.combo_shell_layer.setCurrentText("Max Envelope")
        self.combo_shell_layer.setToolTip(
            "두께 방향 적분점 선택.\n"
            "Upper: 상면 (+t/2), Mid: 중립면 (0), Lower: 하면 (-t/2)\n"
            "Max Envelope: Upper/Lower 중 Von Mises 최대값\n"
            "Membrane: 순수 면내, Bending: 순수 굽힘"
        )
        self.combo_shell_layer.currentTextChanged.connect(self._on_shell_layer_changed)
        self.combo_shell_layer.setEnabled(False)  # Stress/Strain 카테고리에서만 활성화
        hbox_layer.addWidget(self.combo_shell_layer)
        
        # Assemble into Fields Main Layout in logical order
        vbox_contour.addLayout(hbox_cat)
        vbox_contour.addLayout(hbox_comp)
        vbox_contour.addLayout(hbox_layer)
        vbox_contour.addLayout(hbox_cb_mode)
        vbox_contour.addLayout(hbox_cmap)
        
        group_contour.setLayout(vbox_contour)
        
        # --- Query Tools Group (New Requirement) ---
        group_query = QtWidgets.QGroupBox("Query Tools")
        vbox_query = QtWidgets.QVBoxLayout()
        vbox_query.setSpacing(2)
        
        self.chk_query = QtWidgets.QCheckBox("Enable Query")
        self.chk_query.setChecked(False)
        self.chk_query.stateChanged.connect(self._on_query_toggled)
        self.chk_query.setToolTip("Enable cursor hover query on node/element scalars.")
        vbox_query.addWidget(self.chk_query)
        
        hbox_query_target = QtWidgets.QHBoxLayout()
        self.rad_query_node = QtWidgets.QRadioButton("Node Value")
        self.rad_query_elem = QtWidgets.QRadioButton("Element Value")
        self.rad_query_node.setChecked(True)
        self.rad_query_node.toggled.connect(self._on_query_target_changed)
        self.rad_query_elem.toggled.connect(self._on_query_target_changed)
        hbox_query_target.addWidget(self.rad_query_node)
        hbox_query_target.addWidget(self.rad_query_elem)
        vbox_query.addLayout(hbox_query_target)
        
        self.btn_clear_labels = QtWidgets.QPushButton("Clear Labels")
        self.btn_clear_labels.clicked.connect(self._clear_query_labels)
        self.btn_clear_labels.setToolTip("Clear all query text labels on screen.")
        vbox_query.addWidget(self.btn_clear_labels)
        
        group_query.setLayout(vbox_query)
        # --------------------------------------------
        
        prop_layout.addWidget(group_deform)
        prop_layout.addWidget(group_contour)
        prop_layout.addWidget(group_query)

        # UI Theme Group (Disabled - Using Native Dark Mode)
        prop_layout.addStretch()
        
        # --- Tab 2: Part Manager ---
        self.part_tab = QtWidgets.QWidget()
        part_layout = QtWidgets.QVBoxLayout(self.part_tab)
        part_layout.setSpacing(2)
        
        self.list_parts = QtWidgets.QListWidget()
        self.list_parts.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.list_parts.itemSelectionChanged.connect(self._on_part_selection_changed)
        part_layout.addWidget(QtWidgets.QLabel("Assembly Parts:"))
        part_layout.addWidget(self.list_parts)
        
        group_part_style = QtWidgets.QGroupBox("Appearance Control")
        v_style = QtWidgets.QVBoxLayout()
        v_style.setSpacing(2)
        
        h_vis = QtWidgets.QHBoxLayout()
        self.btn_show = QtWidgets.QPushButton("Show")
        self.btn_hide = QtWidgets.QPushButton("Hide")
        self.btn_show.clicked.connect(lambda: self._apply_part_attr("visible", True))
        self.btn_hide.clicked.connect(lambda: self._apply_part_attr("visible", False))
        h_vis.addWidget(self.btn_show)
        h_vis.addWidget(self.btn_hide)
        
        h_rep = QtWidgets.QHBoxLayout()
        h_rep.addWidget(QtWidgets.QLabel("Mode:"))
        self.combo_rep = QtWidgets.QComboBox()
        self.combo_rep.addItems(["Surface", "Surface With Edges", "Feature Edges", "Outline", "Wireframe", "Points"])
        self.combo_rep.currentTextChanged.connect(lambda v: self._apply_part_attr("representation", v))
        h_rep.addWidget(self.combo_rep)
        
        h_opac = QtWidgets.QHBoxLayout()
        h_opac.addWidget(QtWidgets.QLabel("Opacity:"))
        self.slider_opac = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_opac.setRange(0, 100)
        self.slider_opac.setValue(100)
        self.slider_opac.valueChanged.connect(lambda v: self._apply_part_attr("opacity", v/100.0))
        h_opac.addWidget(self.slider_opac)
        
        self.btn_reset_parts = QtWidgets.QPushButton("Reset to Default")
        self.btn_reset_parts.clicked.connect(self._reset_part_styles)
        
        v_style.addLayout(h_vis)
        v_style.addLayout(h_rep)
        v_style.addLayout(h_opac)
        v_style.addWidget(self.btn_reset_parts)
        group_part_style.setLayout(v_style)
        
        part_layout.addWidget(group_part_style)
        
        self.tabs.addTab(self.prop_tab, "Properties")
        self.tabs.addTab(self.part_tab, "Part Manager")

    def _setup_playback_ui(self):
        """Standard professional bottom panel that fills the width."""
        if not QtWidgets: return
        
        # Container for full width
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)
        
        # 1. Controls Group
        self.btn_first = QtWidgets.QPushButton("⏮\uFE0E")
        self.btn_prev = QtWidgets.QPushButton("⏪\uFE0E")
        self.btn_play = QtWidgets.QPushButton("▶\uFE0E")
        self.btn_next = QtWidgets.QPushButton("⏩\uFE0E")
        self.btn_last = QtWidgets.QPushButton("⏭\uFE0E")
        
        for btn in [self.btn_first, self.btn_prev, self.btn_play, self.btn_next, self.btn_last]:
            btn.setFixedSize(40, 30)
            layout.addWidget(btn)
        
        # 2. Timeline Slider (Stretches)
        self.slider_time = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_time.setMinimumWidth(200)
        self.slider_time.valueChanged.connect(self._on_time_slider_changed)
        layout.addWidget(self.slider_time, 1) # Give stretch factor 1
        
        # 3. Time Display
        self.lbl_time = QtWidgets.QLabel("Step: 0")
        self.lbl_time.setMinimumWidth(100)
        layout.addWidget(self.lbl_time)
        
        # Signals
        self.btn_first.clicked.connect(lambda: self.slider_time.setValue(0))
        self.btn_prev.clicked.connect(lambda: self.slider_time.setValue(max(0, self.slider_time.value()-1)))
        self.btn_play.clicked.connect(self._toggle_animation)
        self.btn_next.clicked.connect(lambda: self.slider_time.setValue(min(self.slider_time.maximum(), self.slider_time.value()+1)))
        self.btn_last.clicked.connect(lambda: self.slider_time.setValue(self.slider_time.maximum()))
        
        # Integrate as a bottom dock widget for full width
        self.dock_play = QtWidgets.QDockWidget()
        self.dock_play.setWidget(container)
        self.dock_play.setTitleBarWidget(QtWidgets.QWidget()) # Hide title
        self.plotter.app_window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.dock_play)
        
        self.spin_fps = QtWidgets.QSpinBox()
        self.spin_fps.setRange(1, 100)
        self.spin_fps.setValue(10)
        self.spin_fps.setPrefix("FPS: ")
        self.spin_fps.valueChanged.connect(self._on_fps_changed)
        layout.addWidget(self.spin_fps)
        
        # Timer Setup
        self.anim_timer = QtCore.QTimer(self.plotter.app_window)
        self.anim_timer.timeout.connect(self._on_animation_tick)
        self.is_playing = False

    def _setup_view_controls(self):
        """Sets up key events and prepares for context menu."""
        # Key bindings for quick view switching
        self.plotter.add_key_event('1', lambda: self.plotter.view_xy())
        self.plotter.add_key_event('2', lambda: self.plotter.view_xz())
        self.plotter.add_key_event('3', lambda: self.plotter.view_yz())
        self.plotter.add_key_event('p', lambda: self.plotter.enable_parallel_projection())
        self.plotter.add_key_event('o', lambda: self.plotter.disable_parallel_projection())
        
        # Right-click context menu (Requires Qt)
        if QtWidgets:
            self.plotter.interactor.AddObserver("RightButtonPressEvent", self._show_context_menu)

    def _show_context_menu(self, obj, event):
        """Displays a professional context menu at the mouse position."""
        if not QtWidgets:
            return
            
        menu = QtWidgets.QMenu(self.plotter.app_window)
        
        # View Projections
        view_menu = menu.addMenu("View Projection")
        view_menu.addAction("Top (XY)", lambda: self.plotter.view_xy())
        view_menu.addAction("Front (XZ)", lambda: self.plotter.view_xz())
        view_menu.addAction("Side (YZ)", lambda: self.plotter.view_yz())
        view_menu.addAction("Isometric", lambda: self.plotter.view_isometric())
        
        menu.addSeparator()
        
        # Projection Mode
        proj_menu = menu.addMenu("Camera Mode")
        p_act = proj_menu.addAction("Perspective", lambda: self.plotter.disable_parallel_projection())
        o_act = proj_menu.addAction("Orthographic", lambda: self.plotter.enable_parallel_projection())
        
        menu.addSeparator()
        
        # Theme Menu (BG 세부 선택)
        theme_menu = menu.addMenu("Background")
        _BG_ITEMS = [
            ("Black",          "Black"),
            ("White",          "White"),
            ("Dark Grey",      "Dark Grey"),
            ("Light Grey",     "Light Grey"),
            ("Grey Grad.",     "Grey Grad."),
            ("Light Grey Grad.", "Light Grey Grad."),
            ("Light Sky Grad.", "Light Sky Grad."),
        ]
        for label, name in _BG_ITEMS:
            theme_menu.addAction(label, lambda checked=False, n=name: self._on_bg_changed(n))
        
        menu.addSeparator()
        menu.addAction("Reset Camera", lambda: self.plotter.reset_camera())
        menu.addAction("Close Plotter", lambda: self.close())
        
        # Display menu at cursor
        cursor = QtGui.QCursor.pos()
        menu.exec_(cursor)

    def _create_view_icon(self, view_type, label=""):
        """[WHT High-Fidelity] Generates ParaView-style coordinate axes icons."""
        if not QtGui or not QtCore: return QtGui.QIcon()
        pixmap = QtGui.QPixmap(64, 64)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # ParaView Color Palette
        c_red = QtGui.QColor("#d9534f")   # X-Axis
        c_green = QtGui.QColor("#5cb85c") # Y-Axis
        c_gold = QtGui.QColor("#f0ad4e")  # Z-Axis
        c_white = QtGui.QColor("#ffffff")
        
        def draw_arrow(start_pos, end_pos, color):
            painter.setPen(QtGui.QPen(color, 4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            painter.drawLine(start_pos, end_pos)
            # Arrow Head
            vec = end_pos - start_pos
            length = (vec.x()**2 + vec.y()**2)**0.5
            if length > 0:
                # Convert to QPointF for proper scaling and addition in PySide6
                u = QtCore.QPointF(vec.x() / length, vec.y() / length)
                side = QtCore.QPointF(-u.y(), u.x()) * 6
                end_pos_f = QtCore.QPointF(end_pos)
                painter.setBrush(QtGui.QBrush(color))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawPolygon(QtGui.QPolygonF([
                    end_pos_f, 
                    end_pos_f - u*10 + side, 
                    end_pos_f - u*10 - side
                ]))

        center = QtCore.QPoint(18, 46) # Origin for axes
        
        if view_type == "box":
            # Emulate mapping from screenshot
            if "+X" in label:
                draw_arrow(center, center + QtCore.QPoint(0, -32), c_green) # Y up
                draw_arrow(center, center + QtCore.QPoint(32, 0), c_gold)   # Z right
                painter.fillRect(center.x()-6, center.y()-6, 12, 12, c_red) # X origin
            elif "-X" in label:
                draw_arrow(center, center + QtCore.QPoint(0, -32), c_green)
                draw_arrow(center, center + QtCore.QPoint(32, 0), c_gold)
                painter.setPen(QtGui.QPen(c_red, 2, QtCore.Qt.DashLine))
                painter.drawRect(center.x()-6, center.y()-6, 12, 12)
            elif "+Y" in label:
                draw_arrow(center, center + QtCore.QPoint(0, -32), c_gold)  # Z up
                draw_arrow(center, center + QtCore.QPoint(32, 0), c_red)    # X right
                painter.fillRect(center.x()-6, center.y()-6, 12, 12, c_green)
            elif "-Y" in label:
                draw_arrow(center, center + QtCore.QPoint(0, -32), c_gold)
                draw_arrow(center, center + QtCore.QPoint(32, 0), c_red)
                painter.setPen(QtGui.QPen(c_green, 2, QtCore.Qt.DashLine))
                painter.drawRect(center.x()-6, center.y()-6, 12, 12)
            elif "+Z" in label:
                draw_arrow(center, center + QtCore.QPoint(0, -32), c_green) # Y up
                draw_arrow(center, center + QtCore.QPoint(32, 0), c_red)    # X right
                painter.fillRect(center.x()-6, center.y()-6, 12, 12, c_gold)
            elif "-Z" in label:
                draw_arrow(center, center + QtCore.QPoint(0, -32), c_green)
                draw_arrow(center, center + QtCore.QPoint(32, 0), c_red)
                painter.setPen(QtGui.QPen(c_gold, 2, QtCore.Qt.DashLine))
                painter.drawRect(center.x()-6, center.y()-6, 12, 12)
            
            painter.setPen(c_white)
            painter.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
            painter.drawText(0, 0, 64, 20, QtCore.Qt.AlignCenter, label)
            
        elif view_type == "iso":
            # Three axes icon
            draw_arrow(center, center + QtCore.QPoint(0, -30), c_green)
            draw_arrow(center, center + QtCore.QPoint(26, 15), c_red)
            draw_arrow(center, center + QtCore.QPoint(-20, 10), c_gold)
            
        elif view_type == "reset":
            painter.setPen(QtGui.QPen(c_white, 3))
            painter.drawRect(12, 12, 40, 40)
            painter.setPen(QtGui.QPen(c_gold, 2))
            painter.drawEllipse(22, 22, 20, 20)
            painter.drawLine(32, 15, 32, 49)
            painter.drawLine(15, 32, 49, 32)
            
        elif view_type == "rotate":
            painter.setPen(QtGui.QPen(c_green, 3))
            painter.drawArc(15, 15, 34, 34, 45 * 16, 270 * 16)
            painter.setBrush(QtGui.QBrush(c_green))
            painter.drawPolygon(QtGui.QPolygon([QtCore.QPoint(49, 32), QtCore.QPoint(40, 22), QtCore.QPoint(58, 22)]))
        
        elif view_type == "camera":
            painter.setPen(QtGui.QPen(c_white, 2))
            painter.drawRect(15, 20, 34, 25)
            painter.drawEllipse(25, 25, 14, 14)
            painter.fillRect(25, 15, 14, 5, c_gold)
            
        elif view_type == "bg":
            painter.setPen(QtGui.QPen(c_white, 2))
            painter.drawEllipse(15, 15, 34, 34)
            painter.setBrush(QtGui.QBrush(c_gold))
            painter.drawPie(15, 15, 34, 34, 90 * 16, 180 * 16)
            
        painter.end()
        return QtGui.QIcon(pixmap)

    def _setup_toolbar(self):
        """[WHT Professional] Comprehensive ParaView-style Graphical Toolbar."""
        if not QtWidgets: return
        self.toolbar = QtWidgets.QToolBar("WHT Main Toolbar")
        self.plotter.app_window.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)
        self.toolbar.setIconSize(QtCore.QSize(30, 30))
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._add_toolbar_action("Reset", "Reset Camera", self._create_view_icon("reset"), self.plotter.reset_camera)
        self.toolbar.addSeparator()

        v_group = [
            ("+X", "View +X", lambda: self.plotter.view_zy()), ("-X", "View -X", lambda: self.plotter.view_yz()),
            ("+Y", "View +Y", lambda: self.plotter.view_xz()), ("-Y", "View -Y", lambda: self.plotter.view_zx()),
            ("+Z", "View +Z", lambda: self.plotter.view_xy()), ("-Z", "View -Z", lambda: self.plotter.view_yx()),
        ]
        for label, tip, func in v_group:
            self._add_toolbar_action(label, tip, self._create_view_icon("box", label), func)
            
        self._add_toolbar_action("ISO", "Isometric View", self._create_view_icon("iso"), lambda: self.plotter.view_isometric())
        self.toolbar.addSeparator()
        self._add_toolbar_action("↺90", "Rotate 90 CCW", self._create_view_icon("rotate"), lambda: self._rotate_camera(90))
        self._add_toolbar_action("Capture", "Take Screenshot", self._create_view_icon("camera"), self._take_screenshot)

        # [WHT] BG 버튼: 클릭 → 반전 토글 / 화살표 클릭 → 세부 테마 선택 메뉴
        self.btn_bg_tool = QtWidgets.QToolButton(self.plotter.app_window)
        self.btn_bg_tool.setIcon(self._create_view_icon("bg"))
        self.btn_bg_tool.setToolTip("배경색 설정\n  클릭: Dark/Light 반전\n  ▼: 세부 테마 선택")
        self.btn_bg_tool.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.btn_bg_tool.clicked.connect(self._on_bg_toggle)

        bg_menu = QtWidgets.QMenu(self.plotter.app_window)
        _BG_LIST = [
            "Black", "White", "Dark Grey", "Light Grey",
            "Grey Grad.", "Light Grey Grad.", "Light Sky Grad."
        ]
        for bg_name in _BG_LIST:
            act = bg_menu.addAction(bg_name)
            act.triggered.connect(lambda checked=False, n=bg_name: self._on_bg_changed(n))
        self.btn_bg_tool.setMenu(bg_menu)
        self.toolbar.addWidget(self.btn_bg_tool)

        self.list_parts.itemChanged.connect(self._on_part_item_changed)

        # 툴바 우측 끝 고정 로고
        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        logo_lbl = QtWidgets.QLabel()
        _logo_path = os.path.join(os.path.dirname(__file__), "..", "resources", "logo_icon_48x48.png")
        _logo_path = os.path.normpath(_logo_path)
        if os.path.exists(_logo_path):
            _pix = QtGui.QPixmap(_logo_path).scaled(
                36, 36, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            logo_lbl.setPixmap(_pix)
        logo_lbl.setToolTip("WHT FEM Visualizer")
        logo_lbl.setContentsMargins(4, 0, 8, 0)
        self.toolbar.addWidget(logo_lbl)

    def _add_toolbar_action(self, text, tooltip, icon, func, checkable=False, checked=False):
        """Helper to create and add graphical actions to the toolbar."""
        action = QtGui.QAction(icon if isinstance(icon, QtGui.QIcon) else QtGui.QIcon(), text, self.plotter.app_window)
        action.setToolTip(tooltip)
        action.setCheckable(checkable)
        action.setChecked(checked)
        action.triggered.connect(func)
        self.toolbar.addAction(action)
        return action

    def _setup_menubar(self):
        """Standard top-level pull-down menu."""
        if not QtWidgets: return
        menubar = self.plotter.app_window.menuBar()
        
        view_menu = menubar.addMenu("&View")
        
        theme_menu = view_menu.addMenu("&Theme")
        dark_action = QtGui.QAction("Dark Mode", self.plotter.app_window)
        dark_action.triggered.connect(lambda: self._set_theme(True))
        light_action = QtGui.QAction("Light Mode", self.plotter.app_window)
        light_action.triggered.connect(lambda: self._set_theme(False))
        
        theme_menu.addAction(dark_action)
        theme_menu.addAction(light_action)

        # --- 메뉴바 우측 끝 WHT 프리미엄 로고 아이콘 표시 ---
        logo_lbl = QtWidgets.QLabel()
        logo_paths = [
            os.path.join(os.path.dirname(__file__), "..", "resources", "logo_icon_32x32.png"),
            os.path.join(os.path.dirname(__file__), "..", "wht_topo", "resources", "logo_icon_32x32.png"),
            os.path.join(os.path.dirname(__file__), "resources", "logo_icon_32x32.png"),
            os.path.join(os.path.dirname(__file__), "..", "wht_visualizer", "resources", "logo_icon_32x32.png"),
        ]
        logo_path = None
        for path in logo_paths:
            norm_p = os.path.normpath(path)
            if os.path.exists(norm_p):
                logo_path = norm_p
                break
                
        if logo_path:
            pix = QtGui.QPixmap(logo_path)
            if not pix.isNull():
                logo_lbl.setPixmap(pix.scaled(32, 32, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                logo_lbl.setContentsMargins(0, 0, 10, 0) # 우측 10px 마진 부여로 레이아웃 정합성 유지
                menubar.setCornerWidget(logo_lbl, QtCore.Qt.TopRightCorner)

    # --- Toolbar Event Handlers ---
    def _on_projection_toggle(self, checked):
        if checked:
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()
        self.plotter.render()

    def _on_axes_toggle(self, checked):
        if checked:
            self.plotter.add_axes()
        else:
            self.plotter.hide_axes()
        self.plotter.render()

    def _rotate_camera(self, angle):
        """Rotates the camera view by a specified angle."""
        self.plotter.camera.roll += angle
        self.plotter.render()

    def _on_bg_changed(self, color_name: str):
        """
        [WHT] 배경색 변경 및 폰트/축 색상 자동 최적화 (whts_multipostprocessor_ui 동일 스타일).

        Parameters
        ----------
        color_name : str
            선택된 배경 테마 이름.
            지원 목록: "Black", "White", "Dark Grey", "Light Grey",
                      "Grey Grad.", "Light Grey Grad.", "Light Sky Grad."
        """
        # 1. 배경색 및 폰트 포어그라운드 결정
        if color_name == "Black":
            self.plotter.set_background("black")
            fg = "white";  self._bg_is_dark = True
        elif color_name == "White":
            self.plotter.set_background("white")
            fg = "black";  self._bg_is_dark = False
        elif color_name == "Dark Grey":
            self.plotter.set_background("#222222")
            fg = "white";  self._bg_is_dark = True
        elif color_name == "Light Grey":
            self.plotter.set_background("#D3D3D3")
            fg = "black";  self._bg_is_dark = False
        elif color_name == "Grey Grad.":
            # ParaView Style: Dark Grey(하단) → Black(상단)
            self.plotter.set_background("#666666", top="black")
            fg = "white";  self._bg_is_dark = True
        elif color_name == "Light Grey Grad.":
            # 밝은 회색(하단) → 흰색(상단)
            self.plotter.set_background("white", top="#D3D3D3")
            fg = "black";  self._bg_is_dark = False
        elif color_name == "Light Sky Grad.":
            # 하늘색(하단) → 흰색(상단)
            self.plotter.set_background("white", top="#E0F7FA")
            fg = "black";  self._bg_is_dark = False
        else:
            return

        fg_rgb = pv.Color(fg).float_rgb

        # 2. 전역 테마 동기화 (plotter.set_background()가 이미 전체 렌더러에 적용됨)
        pv.global_theme.font.color = fg

        # 3. 좌표축 재생성 (색상 반전 적용)
        try:
            self.plotter.add_axes(color=fg)
        except Exception:
            pass

        # 4. Scalar Bar 색상 동기화
        if hasattr(self, 'scalar_bar_actor') and self.scalar_bar_actor is not None:
            try:
                self.scalar_bar_actor.title_text_property.color = fg_rgb
                self.scalar_bar_actor.label_text_property.color = fg_rgb
            except Exception:
                pass

        # 5. Qt UI 팔레트 동기화 (다크/라이트 전환)
        self._apply_ui_theme(is_dark=self._bg_is_dark)

        self.plotter.render()

    def _on_bg_toggle(self):
        """[WHT] 현재 배경색 반전 (Dark ↔ Light) — BG 버튼 단순 클릭 시 실행."""
        self._on_bg_changed("Black" if not self._bg_is_dark else "White")

    def _set_theme(self, is_dark: bool):
        """[WHT] 하위 호환성 유지용 래퍼. _on_bg_changed 로 위임합니다."""
        self._on_bg_changed("Black" if is_dark else "White")

    def _take_screenshot(self):
        if not QtWidgets: return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.plotter.app_window, "Save Screenshot", "wht_capture.png", "PNG Image (*.png)"
        )
        if file_path:
            self.plotter.screenshot(file_path)
            print(f" -> Screenshot saved to {file_path}")

    def show_result(self, result: "WHTResultData", group_name: Optional[str] = None,
                    clear: bool = True, kabsch=None):
        """
        Main entry point to display WHTResultData IR objects.

        Parameters
        ----------
        result : WHTResultData
            The result object to visualize.
        group_name : str, optional
            Prefix for part names to distinguish between multiple results.
        clear : bool, default True
            If True, clears the plotter before adding new data.
        kabsch : KabschPreprocessor, optional
            Kabsch 전처리 결과. 지정 시 "Rigid Body" 체크박스가 활성화되어
            강체 변환(R, T)을 시각화에 적용할 수 있습니다.
        """
        if not result or result.nodes is None:
            print(" -> [Visualizer Warning] Attempted to load empty result data.")
            return

        self._is_ready = False
        self._kabsch = kabsch
        self.chk_rigid_body.setEnabled(kabsch is not None)
        if kabsch is None:
            self.chk_rigid_body.setChecked(False)

        # Always update result_data to the latest loaded one
        self.result_data = result
        self.current_timestep = 0
        
        if clear:
            # 1. Rigorous Resource Cleanup
            self.anim_timer.stop()
            self.is_playing = False
            self.btn_play.setText("▶")
            
            # Clear all previous actors and cached meshes
            self.plotter.clear()
            for p in self.parts.values():
                if "outline" in p: p["outline"] = None
                if "feature" in p: p["feature"] = None
            self.parts.clear()

            self.list_parts.blockSignals(True)
            self.list_parts.clear()
            self.list_parts.blockSignals(False)
            
            self.result_data = result
            self.current_timestep = 0
            self.base_coords = result.nodes.copy()
            n_steps = max(1, result.n_timesteps)
            self.slider_time.setMaximum(n_steps - 1)
            
            # 2. Populate UI based on results
            self._populate_combo_box()
        
        # 3. Rebuild Assembly
        shared_base = self._make_pv_grid(result.nodes, result.connectivity, result.offsets, result.cell_types)
        self.whole_mesh = shared_base
        
        prefix = f"{group_name}_" if group_name else ""
        
        self.list_parts.blockSignals(True)
        covered_indices = set()
        if result.element_sets:
            for part_name, elem_indices in result.element_sets.items():
                if len(elem_indices) == 0: continue
                part_mesh = shared_base.extract_cells(elem_indices)
                # vtkOriginalPointIds must be preserved for _bind_data_to_mesh submesh mapping
                self._add_part(f"{prefix}{part_name}", part_mesh)
                covered_indices.update(elem_indices)

        # [WHT-FIX] Ensure elements not in any set are still rendered as "Base_Mesh"
        all_indices = set(range(result.n_cells))
        remaining = sorted(list(all_indices - covered_indices))
        if remaining:
            remaining_mesh = shared_base.extract_cells(remaining)
            name = f"{prefix}Base_Mesh" if result.element_sets else f"{prefix}Mesh_Model"
            self._add_part(name, remaining_mesh)
            
        self.list_parts.blockSignals(False)
            
        # UI State Init with Guard
        self.base_coords = result.nodes.copy()
        n_steps = max(1, result.n_timesteps)
        self.slider_time.setMaximum(n_steps - 1)
        self.set_timestep(0)
        
        # Intelligent Default for Deformation Field
        if result.metadata and result.metadata.analysis_type == "modal":
            if "ModeShape" in self.avail_results:
                self.combo_warp_vec.setCurrentText("ModeShape")
        elif "Displacement" in self.avail_results:
            self.combo_warp_vec.setCurrentText("Displacement")
            
        self.plotter.reset_camera()
        
        # --- [WHT] BC & Load Visualization (Symbolic) ---
        self._create_symbolic_actors()
        
        self._is_ready = True
        
        # 3. Dynamic Title Update
        if result.metadata:
            title = f"WHT Visualizer - {result.metadata.solver_name} [{result.metadata.analysis_type.upper()}]"
            self.plotter.app_window.setWindowTitle(title)
            
        print(f" -> [Visualizer] Assembly Ready: {len(self.parts)} parts, {result.n_nodes} nodes, {result.n_cells} cells.")

    @staticmethod
    def _make_pv_grid(nodes, connectivity, offsets, cell_types):
        """Builds a PyVista UnstructuredGrid using the most compatible format."""
        import numpy as np
        # offsets is CSR format with leading 0: [0, n0, n0+n1, ...]  (n_cells+1 entries)
        # np.diff(offsets) -> n_cells counts; offsets[:-1] -> n_cells start positions
        cell_counts = np.diff(offsets)
        cells = np.empty(len(connectivity) + len(cell_counts), dtype=connectivity.dtype)

        insert_pos = np.arange(len(cell_counts)) + offsets[:-1]
        cells[insert_pos] = cell_counts

        mask = np.ones(len(cells), dtype=bool)
        mask[insert_pos] = False
        cells[mask] = connectivity

        return pv.UnstructuredGrid(cells, cell_types, nodes)

    def load_results(self, result: "WHTResultData", group_name: Optional[str] = None, clear: bool = True, **kwargs):
        """
        Comprehensive result loader with comparison support.
        
        Parameters
        ----------
        result : WHTResultData
            The result data to load.
        group_name : str, optional
            A label for this result set (shown in part list).
        clear : bool, default True
            Whether to clear existing results.
        """
        try:
            self.show_result(result, group_name=group_name, clear=clear)
            # Apply visual hints from kwargs if provided
            if 'color' in kwargs:
                # If background color requested
                self.plotter.set_background(kwargs['color'])
            if 'label' in kwargs:
                self.plotter.add_text(kwargs['label'], position='upper_left', font_size=10, name='main_label', color='grey')
        except Exception as e:
            print(f" -> [Visualizer ERROR] Failed to load results: {e}")
            import traceback
            traceback.print_exc()

    def _extract_submesh(self, result, element_indices):
        """Builds a VTK submesh using only specified element indices."""
        full_mesh = self._make_pv_grid(result.nodes, result.connectivity, result.offsets, result.cell_types)
        # We must keep track of original node IDs for warping/scalar binding
        return full_mesh.extract_cells(element_indices)

    def _add_part(self, name: str, mesh: pv.UnstructuredGrid):
        """Internal helper to register a mesh part and its actor."""
        # 1. Synthesize current field name from dual combos
        current_field = self._get_active_field_name()
            
        if current_field != "Body Color" and (current_field in mesh.point_data or current_field in mesh.cell_data):
            mesh.set_active_scalars(current_field)

        # 2. Add mesh to plotter
        # Bead_Height scalar이고 discrete steps가 설정된 경우 이산 colormap 강제 적용
        is_bead_discrete = (
            current_field == "Bead_Height"
            and getattr(self, "_bead_steps", 0) >= 1
        )
        if is_bead_discrete:
            n_discrete = self._bead_steps + 1  # bead_steps=1→2색, =2→3색
            n_col = n_discrete
            n_lbl = n_discrete
            fmt_str = "%.2f"
        else:
            n_col = self.cb_levels if self.cb_mode == "Discrete" else 256
            n_lbl = (self.cb_levels + 1) if self.cb_mode == "Discrete" else 11
            fmt_str = f"%.{self.cb_decimals}e"
        
        actor = self.plotter.add_mesh(
            mesh, 
            name=name, 
            pickable=True,
            show_edges=True,
            edge_color='darkgray',
            smooth_shading=False, # Disabled by default for stability on large meshes
            cmap=self.combo_cmap.currentText(),
            n_colors=n_col,
            show_scalar_bar=False, 
            color='lightgrey' if current_field == "Body Color" else None,
            scalar_bar_args={
                "title_font_size": self.cb_title_size, 
                "label_font_size": self.cb_label_size,
                "shadow": True,
                "n_labels": n_lbl, 
                "fmt": fmt_str,
                "position_x": 0.85, 
                "width": 0.12,
                "vertical": True
            }
        )
        
        # [WHT CRITICAL FIX] Ensure initial lookup table is correctly built and not inverted
        if hasattr(actor, 'mapper') and hasattr(actor.mapper, 'lookup_table'):
            lut = actor.mapper.lookup_table
            cmap_name = self.combo_cmap.currentText()
            if hasattr(lut, 'cmap'):
                try:
                    temp_cmap = 'coolwarm' if cmap_name != 'coolwarm' else 'jet'
                    lut.cmap = temp_cmap
                    lut.cmap = cmap_name
                except Exception:
                    lut.cmap = cmap_name
            if hasattr(lut, 'Build'):
                lut.Build()

        # 3. Apply Auto-Color logic
        qual_cmap = plt.get_cmap('Set3')
        part_idx = len(self.parts)
        actor.prop.color = qual_cmap(part_idx % 12)[:3]
        # 4. Handle Global Scalar Bar
        if len(self.parts) == 0:
            display_title = current_field if current_field != "Value" else ""
            if len(display_title) > 8:
                display_title = display_title.replace(" ", "\n").replace("_", "\n")
            self.plotter.add_scalar_bar(
                title=display_title,
                mapper=actor.mapper,
                n_labels=n_lbl,
                shadow=True,
                fmt=fmt_str,
                position_x=0.85,
                width=0.2, 
                vertical=True, # 수직 방향 고정
                title_font_size=self.cb_title_size,
                label_font_size=self.cb_label_size
            )
            # [WHT] 폭을 60px로 고정 (사용자 요청 사항)
            self.plotter.scalar_bar.SetMaximumWidthInPixels(60)
            # Bead_Height discrete: scalar bar 레벨 경계를 이산값으로 명시
            if is_bead_discrete:
                import numpy as _np
                levels = _np.linspace(0.0, 1.0, self._bead_steps + 1)
                sb = self.plotter.scalar_bar
                sb.SetNumberOfLabels(len(levels))
                lut = actor.mapper.lookup_table
                lut.n_values = self._bead_steps + 1
                lut.scalar_range = (0.0, 1.0)

        # Trigger explicit update if Body Color
        if current_field == "Body Color":
            actor.prop.color = 'lightgrey'
            if self.plotter.scalar_bars:
                self.plotter.remove_scalar_bar()
            
        self.parts[name] = {
            "mesh": mesh, 
            "actor": actor, 
            "orig_pts": mesh.points.copy(),
            "active_mesh": mesh,
            "result_data": self.result_data  # Store result data reference in part
        }
        
        item = QtWidgets.QListWidgetItem(name)
        item.setCheckState(QtCore.Qt.Checked)
        self.list_parts.addItem(item)

    # --- Part Attribute Handlers ---
    def _on_part_selection_changed(self):
        """Syncs the style controls with the first selected part with stability guards."""
        if self._is_updating: return
        selected = self.list_parts.selectedItems()
        if not selected: return
        
        name = selected[0].text()
        part = self.parts.get(name)
        if not part or "actor" not in part: return
        actor = part["actor"]
        
        # Reflect state to UI (block signals to avoid feedback loop)
        try:
            self.combo_rep.blockSignals(True)
            self.slider_opac.blockSignals(True)
            
            # Use safe property access for VTK objects
            prop = actor.GetProperty() if hasattr(actor, 'GetProperty') else actor.prop
            rep = prop.GetRepresentation() if hasattr(prop, 'GetRepresentation') else prop.representation
            show_edges = prop.GetEdgeVisibility() if hasattr(prop, 'GetEdgeVisibility') else getattr(prop, 'show_edges', False)
            opacity = prop.GetOpacity() if hasattr(prop, 'GetOpacity') else prop.opacity
            
            if rep in [2, 3]: # Surface / Surface With Edges
                if show_edges:
                    self.combo_rep.setCurrentText("Surface With Edges")
                else:
                    self.combo_rep.setCurrentText("Surface")
            elif rep == 1:
                self.combo_rep.setCurrentText("Wireframe")
            else:
                self.combo_rep.setCurrentText("Points")
                
            self.slider_opac.setValue(int(opacity * 100))
        except Exception as e:
            print(f" -> [Visualizer Debug] Sync Error: {e}")
        finally:
            self.combo_rep.blockSignals(False)
            self.slider_opac.blockSignals(False)

    def _apply_part_attr(self, attr, value):
        """Applies an attribute (visibility, representation, etc.) to all selected parts."""
        selected = self.list_parts.selectedItems()
        if not selected: return
        
        for item in selected:
            name = item.text()
            if name not in self.parts: continue
            part = self.parts[name]
            actor = part["actor"]
            
            if attr == "visible":
                actor.visibility = value
                # Sync checkbox without triggering signals
                self.list_parts.blockSignals(True)
                item.setCheckState(QtCore.Qt.Checked if value else QtCore.Qt.Unchecked)
                self.list_parts.blockSignals(False)
            elif attr == "opacity":
                actor.prop.opacity = value
            elif attr == "representation":
                try:
                    # Default back to standard mesh
                    target_mesh = part["mesh"]
                    prop = actor.GetProperty() if hasattr(actor, 'GetProperty') else actor.prop
                    
                    # Safe visibility/edge reset
                    if hasattr(prop, 'SetEdgeVisibility'): prop.SetEdgeVisibility(False)
                    else: prop.show_edges = False
                    
                    if value == "Surface":
                        if hasattr(prop, 'SetRepresentationToSurface'): prop.SetRepresentationToSurface()
                        else: prop.representation = 2
                    elif value == "Surface With Edges":
                        if hasattr(prop, 'SetRepresentationToSurface'): prop.SetRepresentationToSurface()
                        if hasattr(prop, 'SetEdgeVisibility'): prop.SetEdgeVisibility(True)
                        else: 
                            prop.representation = 2
                            prop.show_edges = True
                    elif value == "Wireframe":
                        if hasattr(prop, 'SetRepresentationToWireframe'): prop.SetRepresentationToWireframe()
                        else: prop.representation = 1
                    elif value == "Points":
                        if hasattr(prop, 'SetRepresentationToPoints'): prop.SetRepresentationToPoints()
                        else: prop.representation = 0
                    elif value == "Outline":
                        if "outline" not in part:
                            part["outline"] = part["mesh"].outline()
                        target_mesh = part["outline"]
                        if hasattr(prop, 'SetRepresentationToSurface'): prop.SetRepresentationToSurface()
                        if hasattr(prop, 'SetEdgeVisibility'): prop.SetEdgeVisibility(True)
                    elif value == "Feature Edges":
                        if "feature" not in part:
                            part["feature"] = part["mesh"].extract_feature_edges()
                        target_mesh = part["mesh"]
                        if hasattr(prop, 'SetRepresentationToSurface'): prop.SetRepresentationToSurface()
                        if hasattr(prop, 'SetEdgeVisibility'): prop.SetEdgeVisibility(True)
                    
                    actor.mapper.SetInputData(target_mesh)
                    part["active_mesh"] = target_mesh 
                except Exception as e:
                    print(f" -> [Visualizer Debug] Rep Change error: {e}")
        self.plotter.render()

    def _on_part_item_changed(self, item):
        """Toggles visibility when the list item checkbox is clicked."""
        name = item.text()
        if name in self.parts:
            visible = (item.checkState() == QtCore.Qt.Checked)
            self.parts[name]["actor"].visibility = visible
            self.plotter.render()

    def _reset_part_styles(self):
        """Restores all parts to default premium view."""
        for name, part in self.parts.items():
            actor = part["actor"]
            actor.visibility = True
            actor.prop.representation = 2
            actor.prop.show_edges = True
            actor.prop.opacity = 1.0
            
            # CRITICAL: Restore the base mesh if it was Outline/Feature
            actor.mapper.SetInputData(part["mesh"])
            part["active_mesh"] = part["mesh"]
            
            for i in range(self.list_parts.count()):
                item = self.list_parts.item(i)
                if item.text() == name:
                    item.setCheckState(QtCore.Qt.Checked)
        self.plotter.render()

    # --- Colorbar Range Handlers ---
    def _on_range_mode_changed(self, mode):
        """Toggles spinbox availability."""
        is_static = (mode == "Static (Fixed)")
        self.spin_min.setEnabled(is_static)
        self.spin_max.setEnabled(is_static)
        
        # [WHT] Manual Override: Do NOT auto-snap to global range here.
        # This prevents overwriting user manual input when the dialog switches to Static mode.
        # Users can still use the 'Fit' button in the main UI if they want to snap.
        
        self.set_timestep(self.current_timestep)

    def _apply_colorbar_range(self, show_stats=False):
        """Applies global min-max range to the scalar bar across all parts.
        [Optimization] Prevents flickering by only re-creating the scalar bar when the field changes.
        """
        if not self.parts: return
        
        mode = self.current_range_mode
        field = self._get_active_field_name()
        if not field: return
        
        if mode == "Dynamic (Auto)":
            rng = self._get_field_global_range(field)
        elif mode == "Robust (Auto)":
            rng = self._calculate_robust_range(field, show_stats=show_stats)
            self.range_min, self.range_max = rng
        else:
            # Static (Fixed) mode
            r_min, r_max = self.range_min, self.range_max
            if r_min > r_max: r_min, r_max = r_max, r_min
            if r_min == r_max: r_max = r_min + 1e-6
            rng = [r_min, r_max]
            
        # [Numerical Safety] — rng을 항상 mutable list로 유지
        rng = list(rng)
        if rng[0] > rng[1]: rng = [rng[1], rng[0]]
        if rng[0] == rng[1]: rng[1] = rng[0] + 1e-6
            
        # Update all mappers
        for part in self.parts.values():
            part["actor"].mapper.scalar_range = rng
            
        # Update scalar bar according to User Rule 3.1
        if field == "Body Color":
            if self.plotter.scalar_bars:
                self.plotter.remove_scalar_bar()
        else:
            # [WHT Flicker Fix] Ensure scalar bar exists and is updated without jumping
            if not self.plotter.scalar_bars:
                # First time: Add scalar bar with fixed width
                n_lbl = (self.cb_levels + 1) if self.cb_mode == "Discrete" else 11
                fmt_str = f"%.{self.spin_cb_decimals.value()}e"
                
                # Use first actor's mapper as reference
                ref_mapper = list(self.parts.values())[0]["actor"].mapper if self.parts else None
                
                self.plotter.add_scalar_bar(
                    title="", # Will set below
                    mapper=ref_mapper,
                    n_labels=n_lbl,
                    shadow=True,
                    fmt=fmt_str,
                    position_x=0.85,
                    width=0.1, # Percent-based, but we override with pixels
                    vertical=True,
                    title_font_size=self.cb_title_size,
                    label_font_size=self.cb_label_size
                )
                if self.plotter.scalar_bar:
                    self.plotter.scalar_bar.SetMaximumWidthInPixels(60)

            # Update existing bar's range and title
            self.plotter.update_scalar_bar_range(rng)
            if self.plotter.scalar_bar:
                display_title = field.replace(" ", "\n").replace("_", "\n") if len(field) > 8 else field
                self.plotter.scalar_bar.SetTitle(display_title)
                # Re-enforce 60px to prevent jumping during range updates
                self.plotter.scalar_bar.SetMaximumWidthInPixels(60)
            
    def _calculate_robust_range(self, field_name: str, p_low: float = None, p_high: float = None, show_stats: bool = False) -> List[float]:
        """Ignores outliers based on percentiles."""
        if p_low is None or p_high is None:
            pct = self.robust_pct
            p_low = (100.0 - pct) / 2.0
            p_high = 100.0 - p_low
            
        """Calculates robust percentile-based range to handle FEA singularities/outliers."""
        if not self.parts: return [0.0, 1.0]
        
        all_vals = []
        for part in self.parts.values():
            mesh = part["mesh"]
            data = None
            if field_name in mesh.point_data:
                data = mesh.point_data[field_name]
            elif field_name in mesh.cell_data:
                data = mesh.cell_data[field_name]
            
            if data is not None:
                # Merge current timestep data from all parts
                all_vals.append(np.array(data).flatten())
        
        if not all_vals: return [0.0, 1.0]
            
        merged = np.concatenate(all_vals)
        merged = merged[~np.isnan(merged)]
        if len(merged) == 0: return [0.0, 1.0]
            
        r_min = float(np.percentile(merged, p_low))
        r_max = float(np.percentile(merged, p_high))
        
        if merged.size == 0:
            return [0.0, 1.0]

        # [WHT-ZERO-DEFECT] Robust Statistical Calculation
        try:
            v_max = float(np.nanmax(merged))
            p99 = float(np.percentile(merged, 99.0))
            p95 = float(np.percentile(merged, 95.0))
            
            mu = np.nanmean(merged)
            std = np.nanstd(merged)
            def get_z(val): 
                if std is None or np.isnan(std) or std <= 1e-12: return 0.0
                return (val - mu) / std
            
            if show_stats and not getattr(self, 'is_playing', False):
                min_v = float(np.nanmin(merged))
                cv = (std / mu * 100) if mu != 0 else 0.0
                
                print(f"[WHT-STATS] Field: {field_name} | Nodes: {len(merged):,}")
                print(f"  > MAX: {v_max:.4e} (Z:{get_z(v_max):.2f}) | 99%: {p99:.4e} | 95%: {p95:.4e} | MIN: {min_v:.4e}")
                print(f"  > Robust: {r_max:.4e} ({p_high}%) | Mean: {mu:.4e} | Std: {std:.4e} | CV: {cv:.1f}%")
        except Exception as e:
            if not getattr(self, 'is_playing', False):
                print(f"[WHT-ERROR] Stats calculation failed: {e}")

        if r_min == r_max: r_max = r_min + 1e-6 # Safety expand
        return [r_min, r_max]

    def _highlight_outliers(self, field_name: str, threshold: float):
        """Highlights nodes exceeding the threshold with Magenta spheres."""
        found_any = False
        
        # Clear existing highlight actor if it exists
        if hasattr(self, "_outlier_actor"):
            try:
                self.plotter.remove_actor(self._outlier_actor)
            except: pass
            
        all_pts = []
        for part in self.parts.values():
            mesh = part["mesh"]
            if field_name not in mesh.point_data: continue
            
            vals = mesh.point_data[field_name]
            mask = vals > threshold
            if np.any(mask):
                pts = mesh.points[mask]
                all_pts.append(pts)
        
        if all_pts:
            merged_pts = np.concatenate(all_pts)
            cloud = pv.PolyData(merged_pts)
            self._outlier_actor = self.plotter.add_mesh(
                cloud, color="magenta", point_size=15, 
                render_points_as_spheres=True, 
                name="_wht_outliers"
            )
            found_any = True
            print(f" -> [WHT-DEBUG] Highlighted {len(merged_pts)} nodes > {threshold:.4e}")
        else:
            print(f" -> [WHT-DEBUG] No nodes found exceeding {threshold:.4e}")
            
        self.plotter.render()

    def clear_outliers(self):
        """Removes the magenta outlier highlight actor."""
        if hasattr(self, "_outlier_actor"):
            try:
                self.plotter.remove_actor(self._outlier_actor)
                delattr(self, "_outlier_actor")
                self.plotter.render()
            except: pass

    # --- Data Update Core ---
    def _bind_data_to_mesh(self, t_idx: int):
        """Binds scalar/vector data to all part meshes."""
        if not self.result_data: return
        
        current_field = self._get_active_field_name()
        
        try:
            for name, part in self.parts.items():
                mesh = part["mesh"]
                orig_ids = mesh.point_data.get('vtkOriginalPointIds')
                
                for sc_name, array_3d in self.result_data.point_data.items():
                    if t_idx < array_3d.shape[0]:
                        data = np.nan_to_num(array_3d[t_idx], nan=0.0, posinf=0.0, neginf=0.0)
                        if data.ndim == 2 and data.shape[1] == 1:
                            data = data.ravel()
                        if orig_ids is not None:
                            # Submesh mapping: Use >= to avoid skipping boundary case
                            if len(data) > np.max(orig_ids):
                                val = data[orig_ids]
                                mesh.point_data[sc_name] = val
                        else:
                            # Full mesh mapping
                            if len(data) == mesh.n_points:
                                mesh.point_data[sc_name] = data
                                
                        # Recover vector components if available
                        curr_data = mesh.point_data.get(sc_name)
                        if curr_data is not None and curr_data.ndim > 1 and curr_data.shape[1] >= 3:
                            # Numerical Safety: NaN/Inf check
                            curr_data = np.nan_to_num(curr_data)
                            mag = np.linalg.norm(curr_data[:, :3], axis=1)
                            mesh.point_data[f"{sc_name}_Magnitude"] = mag
                            mesh.point_data[f"{sc_name}_X"] = curr_data[:, 0]
                            mesh.point_data[f"{sc_name}_Y"] = curr_data[:, 1]
                            mesh.point_data[f"{sc_name}_Z"] = curr_data[:, 2]

                # Cell Data
                for sc_name, array_3d in self.result_data.cell_data.items():
                    if t_idx < array_3d.shape[0]:
                        data = array_3d[t_idx] 
                        if data.ndim == 2 and data.shape[1] == 1:
                            data = data.ravel()
                        orig_cids = mesh.cell_data.get('vtkOriginalCellIds')
                        if orig_cids is not None:
                            # [WHT] 안전한 인덱스 필터링: data 범위 내 orig_cids만 사용
                            valid_mask = orig_cids < len(data)
                            if not np.any(valid_mask):
                                continue
                            # 기본값 0으로 초기화 후 유효 셀만 채움
                            if data.ndim == 1:
                                c_data = np.zeros(len(orig_cids), dtype=data.dtype)
                                c_data[valid_mask] = data[orig_cids[valid_mask]]
                            else:
                                c_data = np.zeros((len(orig_cids), data.shape[1]), dtype=data.dtype)
                                c_data[valid_mask] = data[orig_cids[valid_mask]]
                        else:
                            if len(data) == mesh.n_cells:
                                c_data = data
                            else:
                                continue
                        
                        c_data = np.nan_to_num(c_data, nan=0.0, posinf=0.0, neginf=0.0)
                        mesh.cell_data[sc_name] = c_data
                        
                        if "Stress" in sc_name or "Strain" in sc_name:
                            if c_data.ndim > 1 and c_data.shape[1] == 6:
                                # Standard FEA tensor decomposition
                                s11, s22, s33, s12, s13, s23 = c_data.T
                                
                                # 1. Natural Components (Cell)
                                mesh.cell_data[f"{sc_name}_XX"] = s11
                                mesh.cell_data[f"{sc_name}_YY"] = s22
                                mesh.cell_data[f"{sc_name}_ZZ"] = s33
                                mesh.cell_data[f"{sc_name}_XY"] = s12
                                mesh.cell_data[f"{sc_name}_XZ"] = s13
                                mesh.cell_data[f"{sc_name}_YZ"] = s23
                                
                                # 2. Equivalent Value (Stress: Von Mises, Strain: Equivalent Strain)
                                is_strain = ("Strain" in sc_name)
                                if is_strain:
                                    # Equivalent Strain (Standard Plasticity formula)
                                    diff_sq = (s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2
                                    shear_sq = 1.5 * (s12**2 + s13**2 + s23**2)
                                    vm = (np.sqrt(2.0)/3.0) * np.sqrt(diff_sq + shear_sq)
                                else:
                                    diff_sq = (s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2
                                    shear_sq = 6.0 * (s12**2 + s13**2 + s23**2)
                                    vm = np.sqrt(0.5 * (diff_sq + shear_sq))
                                mesh.cell_data[f"{sc_name}_VonMises"] = vm
                                
                                # 3. Principal Stresses/Strains (Shell-Aware Filtering)
                                tensors = np.zeros((len(c_data), 3, 3))
                                tensors[:, 0, 0] = s11
                                tensors[:, 1, 1] = s22
                                tensors[:, 2, 2] = s33
                                if is_strain:
                                    tensors[:, 0, 1] = tensors[:, 1, 0] = s12 / 2.0
                                    tensors[:, 0, 2] = tensors[:, 2, 0] = s13 / 2.0
                                    tensors[:, 1, 2] = tensors[:, 2, 1] = s23 / 2.0
                                else:
                                    tensors[:, 0, 1] = tensors[:, 1, 0] = s12
                                    tensors[:, 0, 2] = tensors[:, 2, 0] = s13
                                    tensors[:, 1, 2] = tensors[:, 2, 1] = s23
                                
                                principal_vals = np.linalg.eigvalsh(tensors)
                                N = len(c_data)
                                
                                # Identify In-Plane Components for Shells
                                if is_strain:
                                    # For Shell Strain, one eigenvalue is eps_zz = -nu/(1-nu)*(e1+e2)
                                    # We find the one that most closely matches the Poisson ratio (default 0.3)
                                    nu = 0.3
                                    f = -nu / (1.0 - nu)
                                    e = principal_vals
                                    errs = np.stack([
                                        np.abs(e[:,0] - f*(e[:,1]+e[:,2])),
                                        np.abs(e[:,1] - f*(e[:,0]+e[:,2])),
                                        np.abs(e[:,2] - f*(e[:,0]+e[:,1]))
                                    ], axis=1)
                                    idx_out = np.argmin(errs, axis=1)
                                else:
                                    # For Shell Stress, one eigenvalue is sig_zz = 0
                                    idx_out = np.argmin(np.abs(principal_vals), axis=1)
                                
                                mask_in = np.ones((N, 3), dtype=bool)
                                mask_in[np.arange(N), idx_out] = False
                                in_plane = principal_vals[mask_in].reshape(N, 2)
                                
                                # Store In-Plane Principals
                                mesh.cell_data[f"{sc_name}_Max_Principal"] = in_plane[:, 1]
                                mesh.cell_data[f"{sc_name}_Min_Principal"] = in_plane[:, 0]
                                
                                # 4. Signed Von Mises (Unified Shell Logic)
                                # For shells, Trace(Stress) and Trace(Strain) are proportional and share the same sign.
                                # This ensures perfect distribution consistency between Stress and Strain views.
                                trace = s11 + s22 + s33
                                sign_mask = np.where(np.abs(trace) > 1e-12, np.sign(trace), 1.0)
                                
                                # Fallback for pure shear (Trace=0)
                                shear_mask = (np.abs(trace) <= 1e-12)
                                if np.any(shear_mask):
                                    # For pure shear, use the sign of the larger in-plane principal
                                    ip_max = in_plane[:, 1]
                                    ip_min = in_plane[:, 0]
                                    shear_sign = np.where(np.abs(ip_max) >= np.abs(ip_min), np.sign(ip_max), np.sign(ip_min))
                                    sign_mask[shear_mask] = shear_sign[shear_mask]
                                
                                sign_mask[sign_mask == 0] = 1.0
                                mesh.cell_data[f"{sc_name}_Signed_VonMises"] = vm * sign_mask
                                
                                # [WHT KEY FIX] Cell → Point Data 절점 평균화
                                # cell_data_to_point_data()로 연속 색상 표시 구현
                                try:
                                    converted = mesh.cell_data_to_point_data()
                                    for suffix in ["_XX", "_YY", "_ZZ", "_XY", "_XZ", "_YZ",
                                                   "_VonMises", "_Signed_VonMises", 
                                                   "_Max_Principal", "_Min_Principal",
                                                   "_Max_3D_Principal", "_Mid_3D_Principal", "_Min_3D_Principal"]:
                                        field_key = f"{sc_name}{suffix}"
                                        if field_key in converted.point_data:
                                            mesh.point_data[field_key] = converted.point_data[field_key]
                                except Exception as _e:
                                    pass  # Fallback: cell_data 그대로 사용

                if current_field and (current_field in mesh.point_data or current_field in mesh.cell_data):
                    mesh.set_active_scalars(current_field)
                    part["actor"].mapper.SetInputData(mesh)
                    # Force scalar visibility if a real result is selected
                    if current_field != "Body Color":
                        part["actor"].mapper.scalar_visibility = True
                        
                        # [WHT] Robust Outlier Handling: If mode is Robust, set above-range color to grey
                        # This "removes" them visually by making them neutral.
                        if self.current_range_mode == "Robust (Auto)":
                             part["actor"].mapper.lookup_table.above_range_color = 'grey'
                        else:
                             part["actor"].mapper.lookup_table.above_range_color = 'magenta'
                    else:
                        part["actor"].mapper.scalar_visibility = False

            self._apply_colorbar_range(show_stats=False)
        except Exception as e:
            print(f" -> [Visualizer Error] Failed to bind data at step {t_idx}: {e}")

    def _kabsch_frame(self, fem_t_idx: int) -> int:
        """FEM 저장 프레임 인덱스를 Kabsch 시간 배열 인덱스로 변환합니다."""
        if self._kabsch is None or self._kabsch.time_arr is None:
            return 0
        t_val = float(self.result_data.time_values[fem_t_idx]) if \
            self.result_data and fem_t_idx < len(self.result_data.time_values) else 0.0
        return int(np.searchsorted(self._kabsch.time_arr, t_val, side='left').clip(
            0, len(self._kabsch.time_arr) - 1))

    def _warp_pts(self, orig_pts: np.ndarray, disp: np.ndarray,
                  ids: Optional[np.ndarray], scale: float) -> np.ndarray:
        """orig_pts 에 변위를 적용하고 (선택적으로) 강체 변환을 덧씌웁니다."""
        if ids is not None:
            u = disp[ids, :3]
        elif len(disp) == len(orig_pts):
            u = disp[:, :3]
        else:
            u = np.zeros_like(orig_pts)
        if self._kabsch is not None and self.chk_rigid_body.isChecked():
            k_idx = self._kabsch_frame(self.current_timestep)
            return self._kabsch.apply_rigid_body(orig_pts, u, k_idx, scale)
        return orig_pts + u * scale

    def _apply_warping(self):
        """Applies displacement warping to all parts."""
        if not self.result_data: return
        chk = self.chk_warp.isChecked()
        scale = self.spin_scale.value()

        warp_field = self.combo_warp_vec.currentText()
        if not warp_field or (warp_field not in self.result_data.point_data and warp_field != "Bead_Height"):
            for name, part in self.parts.items():
                part["mesh"].points = part["orig_pts"]
                active_mesh = part.get("active_mesh", part["mesh"])
                if active_mesh.n_points == part["mesh"].n_points:
                    active_mesh.points = part["orig_pts"]
            return

        use_rigid = self._kabsch is not None and self.chk_rigid_body.isChecked()

        for name, part in self.parts.items():
            mesh = part["mesh"]
            orig_pts = part["orig_pts"]
            orig_ids = mesh.point_data.get('vtkOriginalPointIds')

            if chk:
                if warp_field == "Bead_Height":
                    # Bead_Height에 따른 가상 변위 오프셋 계산 (스칼라 -> draw-dir [0, 0, -1] 가상 3D 변위)
                    bh_val = None
                    if "Bead_Height" in mesh.point_data:
                        bh_val = mesh.point_data["Bead_Height"]
                    elif "Bead_Height" in mesh.cell_data:
                        try:
                            converted = mesh.cell_data_to_point_data()
                            if "Bead_Height" in converted.point_data:
                                bh_val = converted.point_data["Bead_Height"]
                        except Exception:
                            pass
                            
                    if bh_val is None:
                        # Fallback: result_data 직접 참조
                        if self.current_timestep < len(self.result_data.point_data.get("Bead_Height", [])):
                            raw_bh = self.result_data.point_data["Bead_Height"][self.current_timestep]
                            if orig_ids is not None:
                                bh_val = raw_bh[orig_ids]
                            else:
                                bh_val = raw_bh
                        elif self.current_timestep < len(self.result_data.cell_data.get("Bead_Height", [])):
                            raw_bh = self.result_data.cell_data["Bead_Height"][self.current_timestep]
                            try:
                                temp_mesh = mesh.copy()
                                temp_mesh.cell_data["Bead_Height"] = raw_bh[orig_ids] if orig_ids is not None else raw_bh
                                converted = temp_mesh.cell_data_to_point_data()
                                bh_val = converted.point_data["Bead_Height"]
                            except:
                                bh_val = np.zeros(mesh.n_points)
                        else:
                            bh_val = np.zeros(mesh.n_points)
                            
                    bh_val = np.nan_to_num(bh_val)
                    # draw-dir [0, 0, -1] 방향으로 가상 변위 벡터 생성: [0, 0, -bh]
                    disp = np.zeros((mesh.n_points, 3))
                    disp[:, 2] = -bh_val
                    mesh.points = self._warp_pts(orig_pts, disp, None, scale)
                else:
                    disp_all = self.result_data.point_data[warp_field]
                    if self.current_timestep < len(disp_all):
                        disp = np.nan_to_num(disp_all[self.current_timestep])
                        mesh.points = self._warp_pts(orig_pts, disp, orig_ids, scale)
                    elif use_rigid:
                        mesh.points = self._warp_pts(orig_pts,
                                                     np.zeros_like(orig_pts), orig_ids, scale)
            elif use_rigid:
                # Use Deform off이지만 Rigid Body on → 변형 없이 강체 변환만 적용
                mesh.points = self._warp_pts(orig_pts,
                                             np.zeros_like(orig_pts), orig_ids, scale)
            else:
                mesh.points = orig_pts

            active_mesh = part.get("active_mesh", mesh)
            if active_mesh is not mesh and active_mesh.n_points == mesh.n_points:
                active_mesh.points = mesh.points

            if part["actor"] and hasattr(part["actor"], "mapper"):
                part["actor"].mapper.SetInputData(mesh)

        self._sync_symbolic_positions()

        if hasattr(self.plotter, 'update'): self.plotter.update()

    def set_timestep(self, t_idx: int):
        """Entry point for slider/playback changes."""
        if self._is_updating: return
        self._is_updating = True
        
        try:
            self.current_timestep = t_idx
            t_val = self.result_data.time_values[t_idx] if self.result_data else 0.0
            self.lbl_time.setText(f"Step: {t_idx} (Val: {t_val:.2f})")
            
            self._bind_data_to_mesh(t_idx)
            self._apply_warping()
            self._update_allframe_range()
            # [WHT Performance] Scalar range is usually held static during playback
            # unless Dynamic mode is on.
            self._apply_colorbar_range(show_stats=False)
            self.plotter.render()
        finally:
            self._is_updating = False
            
    def _on_category_changed(self, category):
        """Update component list based on category with smart filtering."""
        if not self.result_data or not category: return
        
        self.combo_component.blockSignals(True)
        self.combo_component.clear()
        
        if category == "Body Color":
            self.combo_component.setEnabled(False)
            self.combo_shell_layer.setEnabled(False)
            self.btn_adjust_range.setEnabled(False)
            self.combo_component.blockSignals(False)
            self._update_active_result()
            return
            
        self.combo_component.setEnabled(True)
        self.btn_adjust_range.setEnabled(True)
        
        # Shell Layer 활성화: Stress 또는 Strain 카테고리일 때만
        is_tensor = ("Stress" in category or "Strain" in category)
        self.combo_shell_layer.setEnabled(is_tensor)
        
        # Find all fields belonging to this category
        comps = []
        prefix = category + "_"
        all_fields = list(self.result_data.point_data.keys()) + list(self.result_data.cell_data.keys())
        
        for f in self.avail_results:
            if f == category:
                comps.append("Value")
            elif f.startswith(prefix):
                comps.append(f.replace(prefix, ""))
                
        # Smart Filter: Hide redundant "Value" for vector/tensor fields
        filtered_comps = []
        # Normalization: If X/Y/Z exist, we should call them cleanly X/Y/Z, not XYZ_X
        # Since we and result.py standardized the names, this is easier.
        if any(d in comps for d in ["X", "Y", "Z", "Magnitude", "VonMises", "Max_Principal"]):
            filtered_comps = [c for c in comps if c != "Value"]
        else:
            filtered_comps = comps
            
        self.combo_component.addItems(sorted(list(set(filtered_comps))))
        
        # Default Selection Focus
        if "Magnitude" in filtered_comps: self.combo_component.setCurrentText("Magnitude")
        elif "VonMises" in filtered_comps: self.combo_component.setCurrentText("VonMises")
        elif self.combo_component.count() > 0: self.combo_component.setCurrentIndex(0)
            
        self.combo_component.blockSignals(False)
        
        # [User Request] Ensure user feels the change: 
        # Trigger a range reset if switching categories to a meaningful new field
        if category != "Body Color":
            self.current_range_mode = "Dynamic (Auto)"
            print(f" -> [Visualizer] Category changed to '{category}'. Auto-resetting range for visibility.")
            
        self._update_active_result()

    def _on_component_changed(self, component):
        # [WHT FIX] Field change should ensure visibility by fitting range if in Static mode
        if self.current_range_mode == "Static (Fixed)":
             # Stay in Static mode, but 'Fit' the range to the new field's global limits
             cat = self.combo_category.currentText()
             field_key = self._get_full_field_key(cat, component)
             rng = self._get_field_global_range(field_key)
             self.range_min, self.range_max = rng
             print(f" -> [Visualizer] Field component changed to {component}. Fitting Static range to {rng}")
             
        self._update_active_result()

    def _on_shell_layer_changed(self, layer_text):
        """Shell Layer(두께 방향 적분점) 변경 시 결과를 갱신합니다."""
        cat = self.combo_category.currentText()
        if cat and ("Stress" in cat or "Strain" in cat):
            # Dynamic 모드로 자동 전환하여 새 레이어 범위를 즉시 반영
            self.current_range_mode = "Dynamic (Auto)"
            print(f" -> [Visualizer] Shell layer changed to '{layer_text}'.")
            self._update_active_result()

    def _get_field_global_range(self, field_name: str) -> Tuple[float, float]:
        """Calculates absolute min/max for a field across all parts (current frame)."""
        rng = [float('inf'), float('-inf')]
        for part in self.parts.values():
            m = part["mesh"]
            if field_name in m.point_data or field_name in m.cell_data:
                r = m.get_data_range(field_name)
                rng[0] = min(rng[0], r[0])
                rng[1] = max(rng[1], r[1])
        if rng[0] == float('inf'): return 0.0, 1.0
        return float(rng[0]), float(rng[1])

    def _update_allframe_range(self):
        """프레임 이동 시 현재 프레임의 활성 필드 범위를 누적하여 전체-프레임 min/max 캐시를 갱신한다."""
        field = self._get_active_field_name()
        if not field or field == "Body Color":
            return
        cur_min, cur_max = self._get_field_global_range(field)
        prev = self._field_allframe_range.get(field)
        if prev is None:
            self._field_allframe_range[field] = (cur_min, cur_max)
        else:
            self._field_allframe_range[field] = (min(prev[0], cur_min), max(prev[1], cur_max))

    def get_allframe_range(self, field_name: str) -> Tuple[float, float]:
        """field_name 의 전체-프레임 누적 min/max 를 반환한다. 미탐색 프레임은 현재 프레임 범위로 fallback."""
        if field_name in self._field_allframe_range:
            return self._field_allframe_range[field_name]
        return self._get_field_global_range(field_name)

    def _resolve_shell_layer_category(self, base_category: str) -> str:
        """Shell Layer 콤보 선택에 따라 실제 cell_data 키를 결정합니다.
        
        Parameters
        ----------
        base_category : str
            기본 카테고리 (예: "Stress", "Strain").
        
        Returns
        -------
        str
            실제 cell_data 키 (예: "Stress", "Stress (Mid)", "Stress (Max Envelope)").
        """
        if not hasattr(self, 'combo_shell_layer'):
            return base_category
        if not self.combo_shell_layer.isEnabled():
            return base_category
        
        layer = self.combo_shell_layer.currentText()
        # Mapping: UI text -> cell_data key suffix
        layer_map = {
            "Upper (+t/2)": "",              # Default: "Stress", "Strain"
            "Mid (0)": " (Mid)",
            "Lower (-t/2)": " (Lower)",
            "Max Envelope": " (Max Envelope)",
            "Membrane": " (Membrane)",
            "Bending": " (Bending)",
        }
        suffix = layer_map.get(layer, "")
        return base_category + suffix

    def _get_full_field_key(self, category, component):
        """Helper to resolve full field key used in mesh data."""
        if category == "Body Color": return None
        
        # Shell Layer 적용 (Stress/Strain 카테고리에서만)
        resolved_cat = category
        if "Stress" in category or "Strain" in category:
            # 기본 카테고리 추출 (예: "Stress" from "Stress_XX")
            base = category.split("_")[0] if "_" in category else category
            resolved_cat = self._resolve_shell_layer_category(base)
        
        if component == "Magnitude": return f"{resolved_cat}_Magnitude"
        if component in ["X", "Y", "Z"]: return f"{resolved_cat}_{component}"
        # Stress/Strain components like VonMises, XX, etc.
        if component in ["VonMises", "Signed_VonMises", "XX", "YY", "ZZ", "XY", "YZ", "ZX", "Max_Principal", "Min_Principal", "Max_3D_Principal", "Mid_3D_Principal", "Min_3D_Principal"]:
            return f"{resolved_cat}_{component}"
        return resolved_cat

    def _on_range_mode_changed(self, mode):
        """[User Request] Handles transition between Auto, Robust, and Fixed ranges."""
        # Spinboxes removed from main panel as requested; Adjust dialog covers details.
        self._apply_colorbar_range(show_stats=True)

    def _open_range_adjust_dialog(self):
        """[WHT] Range Dialog를 모달리스(Modeless)로 띄워 실시간 조작을 지원합니다."""
        field = self._get_active_field_name()
        if not field or field == "Body Color": return
        
        # 이미 창이 열려 있다면 포커스만 이동
        if hasattr(self, '_range_dialog') and self._range_dialog is not None:
            if self._range_dialog.isVisible():
                self._range_dialog.raise_()
                self._range_dialog.activateWindow()
                return

        def get_limits():
            total_rng = [float('inf'), float('-inf')]
            for part in self.parts.values():
                m = part["mesh"]
                if field in m.point_data or field in m.cell_data:
                    r = m.get_data_range(field)
                    total_rng[0] = min(total_rng[0], r[0])
                    total_rng[1] = max(total_rng[1], r[1])
            if total_rng[0] == float('inf'): return 0.0, 1.0
            return float(total_rng[0]), float(total_rng[1])

        def get_robust(pct):
            return self._calculate_robust_range(field, p_low=(100-pct)/2.0, p_high=100.0-(100-pct)/2.0)

        # Build dummy group to sync with main UI controls
        class DummyGroup:
            def __init__(self, vis):
                self.vis = vis
                # Use internal state since main UI widgets are removed
            def get_range(self): return self.vis.range_min, self.vis.range_max
            def get_mode(self): return self.vis.current_range_mode
            def set_mode(self, mode): self.vis.current_range_mode = mode
            def get_robust_pct(self): return self.vis.robust_pct
            def set_robust_pct(self, pct): self.vis.robust_pct = pct
            def set_range(self, v_min, v_max):
                self.vis.range_min, self.vis.range_max = v_min, v_max
                self.vis._apply_colorbar_range(show_stats=True)

        self._range_dialog = WHTRangeDialog(self, field, DummyGroup(self), get_limits, get_robust)
        self._range_dialog.setWindowFlags(self._range_dialog.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self._range_dialog.show()

    def _update_active_result(self):
        """Synthesizes the full result name and applies it, ensuring data is bound."""
        # [WHT-FIX] Ensure mesh has all derived fields (VonMises, etc.) for the current selection
        self._bind_data_to_mesh(self.current_timestep)
        
        full_name = self._get_active_field_name()
        self._on_result_type_changed(full_name)

    def _get_active_field_name(self) -> str:
        """Synthesizes the full result name from dual combos + shell layer."""
        if not hasattr(self, 'combo_category'): return ""
        cat = self.combo_category.currentText()
        comp = self.combo_component.currentText()
        if cat == "Body Color": return "Body Color"
        
        # Shell Layer 적용
        resolved_cat = cat
        if "Stress" in cat or "Strain" in cat:
            resolved_cat = self._resolve_shell_layer_category(cat)
        
        if comp == "Value": return resolved_cat
        return f"{resolved_cat}_{comp}"

    def _on_result_type_changed(self, name):
        """Switches active scalar across all parts with Auto Body Color support."""
        if not name: return
        
        is_body = (name == "Body Color")
        # Qualitative colormap for auto-coloring parts like ParaView/HyperView
        import matplotlib.pyplot as plt
        qual_cmap = plt.get_cmap('Set3') # Soft, distinguishable colors
        
        for i, (p_name, part) in enumerate(self.parts.items()):
            mesh = part["mesh"]
            actor = part["actor"]
            
            if is_body:
                actor.mapper.scalar_visibility = False
                # Assign distinct color per part index
                rgb = qual_cmap(i % 12)[:3]
                actor.prop.color = rgb
            else:
                actor.mapper.scalar_visibility = True
                if name in mesh.point_data or name in mesh.cell_data:
                    mesh.set_active_scalars(name)
                    actor.mapper.SetInputData(mesh)
                    
                    # [Fix] 스칼라가 처음 활성화될 때 PyVista 기본값 덮어쓰기로 인해 Colormap이 반전되는 현상 방지
                    # [WHT CRITICAL FIX] PyVista의 cmap 캐싱 바이패스 및 룩업테이블 강제 동기화
                    if hasattr(actor.mapper, 'lookup_table'):
                        lut = actor.mapper.lookup_table
                        cmap_name = self.combo_cmap.currentText()
                        if hasattr(lut, 'cmap'):
                            try:
                                temp_cmap = 'coolwarm' if cmap_name != 'coolwarm' else 'jet'
                                lut.cmap = temp_cmap
                                lut.cmap = cmap_name
                            except Exception:
                                lut.cmap = cmap_name
                        else:
                            lut.cmap = cmap_name
                        if hasattr(lut, 'Build'):
                            lut.Build()
                    
        self._apply_colorbar_range(show_stats=True)
        print(f" -> [Visualizer] Switched result to: {name}")
        self.plotter.render()

    def _on_time_slider_changed(self, value):
        self.set_timestep(value)

    def _on_warp_toggled(self, state):
        self._apply_warping()

    def _on_warp_scale_changed(self, value):
        self._apply_warping()

    def _on_warp_field_changed(self, field):
        self._apply_warping()

    def _update_symbolic_viz(self):
        """Updates visibility of BC and Load markers."""
        if "BC" in self.actors_misc:
            self.actors_misc["BC"].SetVisibility(self.chk_bc.isChecked())
        if "Load" in self.actors_misc:
            self.actors_misc["Load"].SetVisibility(self.chk_load.isChecked())
        self.plotter.render()

    def _create_symbolic_actors(self):
        """[WHT High-Fidelity] Automatically identifies and visualizes BCs and Loads."""
        if not self.result_data: return
        
        # Clear existing misc actors
        for act in self.actors_misc.values():
            self.plotter.remove_actor(act)
        self.actors_misc = {}
        
        # 1. Boundary Conditions (node_sets)
        bc_indices = []
        for name, idxs in self.result_data.node_sets.items():
            if any(k in name.lower() for k in ["bc", "fixed", "spc", "constraint", "support"]):
                bc_indices.extend(idxs)
        
        if bc_indices:
            unique_idxs = np.unique(bc_indices)
            bc_pts = self.result_data.nodes[unique_idxs]
            poly = pv.PolyData(bc_pts)
            poly["orig_idxs"] = unique_idxs # Store for dynamic tracking
            # [WHT-FIX] Apply slight Z-offset to markers to avoid Z-fighting with mesh surface
            poly.points[:, 2] += 2.0  # 2mm offset
            
            act_bc = self.plotter.add_mesh(
                poly, color='#ff2222', point_size=12, 
                render_points_as_spheres=True, 
                pickable=False,
                name="_wht_bc_symbols"
            )
            act_bc.SetVisibility(self.chk_bc.isChecked())
            self.actors_misc["BC"] = act_bc

        # [WHT-NEW] Render RBE Rigids as a distinct layer (Bold White Lines)
        if "RIGIDS" in self.result_data.element_sets:
            rigid_indices = self.result_data.element_sets["RIGIDS"]
            # vtkOriginalPointIds must be preserved for _sync_symbolic_positions
            rigid_mesh = self.whole_mesh.extract_cells(rigid_indices)
            
            act_rigids = self.plotter.add_mesh(
                rigid_mesh, 
                color='white', 
                line_width=3, 
                label="Rigid Elements",
                pickable=False,
                name="_wht_rigid_lines"
            )
            self.actors_misc["RIGIDS"] = act_rigids
            
        # 2. Loads (point_data)
        load_indices = []
        for name, data in self.result_data.point_data.items():
            if "load" in name.lower() or "force" in name.lower():
                # For transient/modal, check first timestep visibility
                mag = np.linalg.norm(data[0], axis=-1) if data.ndim > 2 else np.abs(data[0])
                load_indices.extend(np.where(mag > 1e-9)[0])
        
        if load_indices:
            unique_idxs = np.unique(load_indices)
            load_pts = self.result_data.nodes[unique_idxs]
            poly = pv.PolyData(load_pts)
            poly["orig_idxs"] = unique_idxs # Store for dynamic tracking
            act_load = self.plotter.add_mesh(
                poly, color='#22ff22', point_size=5, # [User Request] 10 -> 5
                render_points_as_spheres=True, 
                pickable=False,
                name="_wht_load_symbols"
            )
            act_load.SetVisibility(self.chk_load.isChecked())
            self.actors_misc["Load"] = act_load

    def _sync_symbolic_positions(self):
        """[WHT-FIX] Warps BC/Load symbols based on current global mesh deformation."""
        if not self.result_data: return
        
        # Get warping parameters from UI state
        chk = self.chk_warp.isChecked()
        scale = self.spin_scale.value()
        warp_field = self.combo_warp_vec.currentText()
        
        # Base coordinates for all nodes (N, 3)
        base_pts = self.result_data.nodes
        
        # Displacement for current timestep (N, 3)
        disp = None
        if chk and warp_field in self.result_data.point_data:
            disp_all = self.result_data.point_data[warp_field]
            if self.current_timestep < len(disp_all):
                disp = np.nan_to_num(disp_all[self.current_timestep])
        
        use_rigid = self._kabsch is not None and self.chk_rigid_body.isChecked()
        k_idx = self._kabsch_frame(self.current_timestep) if use_rigid else 0

        for key in ["BC", "Load", "RIGIDS"]:
            if key in self.actors_misc:
                actor = self.actors_misc[key]
                mesh = actor.mapper.dataset

                if key == "RIGIDS":
                    if "vtkOriginalPointIds" in mesh.point_data:
                        orig_nids = mesh.point_data["vtkOriginalPointIds"]
                        orig = base_pts[orig_nids]
                        u = disp[orig_nids, :3] if disp is not None else np.zeros_like(orig)
                        if use_rigid:
                            mesh.points = self._kabsch.apply_rigid_body(orig, u, k_idx, scale)
                        else:
                            mesh.points = orig + u * scale
                    # else: vtkOriginalPointIds 없으면 스킵
                else:
                    poly = mesh
                    if "orig_idxs" in poly.point_data:
                        idxs = poly.point_data["orig_idxs"]
                        orig = base_pts[idxs]
                        u = disp[idxs, :3] if disp is not None else np.zeros_like(orig)
                        if use_rigid:
                            pts = self._kabsch.apply_rigid_body(orig, u, k_idx, scale)
                        else:
                            pts = orig + u * scale

                        if key == "BC":
                            pts[:, 2] += 2.0

                        poly.points = pts

                actor.mapper.SetInputData(mesh)

    def _on_colormap_changed(self, cmap):
        for part in self.parts.values():
            actor = part.get("actor")
            if actor and hasattr(actor, "mapper") and hasattr(actor.mapper, "lookup_table"):
                lut = actor.mapper.lookup_table
                # [WHT CRITICAL FIX] PyVista의 cmap 캐싱 바이패스 및 강제 리빌드
                if hasattr(lut, 'cmap'):
                    try:
                        temp_cmap = 'coolwarm' if cmap != 'coolwarm' else 'jet'
                        lut.cmap = temp_cmap
                        lut.cmap = cmap
                    except Exception:
                        lut.cmap = cmap
                else:
                    lut.cmap = cmap
                if hasattr(lut, 'Build'):
                    lut.Build()
        self.plotter.render()

    def _on_fps_changed(self, value):
        if self.is_playing:
            self.anim_timer.setInterval(int(1000 / value))

    def _toggle_animation(self):
        self.is_playing = not self.is_playing
        self.btn_play.setText("⏸\uFE0E" if self.is_playing else "▶\uFE0E")
        if self.is_playing:
            self.anim_timer.start(int(1000 / self.spin_fps.value()))
        else:
            self.anim_timer.stop()

    def _on_animation_tick(self):
        curr = self.slider_time.value()
        maxx = self.slider_time.maximum()
        if curr >= maxx:
            self.slider_time.setValue(0)
        else:
            self.slider_time.setValue(curr + 1)

    def _open_cb_font_dialog(self):
        """Opens a dialog to adjust colorbar font sizes."""
        if not QtWidgets: return
        dialog = QtWidgets.QDialog(self.plotter.app_window)
        dialog.setWindowTitle("Colorbar Font Settings")
        
        layout = QtWidgets.QFormLayout(dialog)
        
        spin_title = QtWidgets.QSpinBox()
        spin_title.setRange(6, 60)
        spin_title.setValue(self.cb_title_size)
        
        spin_label = QtWidgets.QSpinBox()
        spin_label.setRange(6, 60)
        spin_label.setValue(self.cb_label_size)
        
        layout.addRow("Title Font Size:", spin_title)
        layout.addRow("Label Font Size:", spin_label)
        
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addRow(btn_box)
        
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.set_colorbar_font_sizes(spin_title.value(), spin_label.value())

    def set_colorbar_font_sizes(self, title_size: int, label_size: int):
        """[WHT] External API to update colorbar font sizes dynamically."""
        self.cb_title_size = title_size
        self.cb_label_size = label_size
        
        if hasattr(self, 'plotter') and self.plotter:
            try:
                self.plotter.remove_scalar_bar()
            except Exception:
                pass
            
            if self.parts:
                current_field = self._get_active_field_name()
                if current_field and current_field != "Body Color":
                    n_lbl = (self.cb_levels + 1) if self.cb_mode == "Discrete" else 11
                    fmt_str = f"%.{self.cb_decimals}e"
                    
                    display_title = current_field if current_field != "Value" else ""
                    if len(display_title) > 8:
                        display_title = display_title.replace(" ", "\n").replace("_", "\n")
                        
                    active_mapper = list(self.parts.values())[0]["actor"].mapper if self.parts else None
                    self.plotter.add_scalar_bar(
                        title=display_title,
                        mapper=active_mapper,
                        n_labels=n_lbl,
                        shadow=True,
                        fmt=fmt_str,
                        position_x=0.85,
                        position_y=0.1,
                        height=0.8,
                        width=0.2,
                        vertical=True,
                        font_family='arial',
                        title_font_size=self.cb_title_size,
                        label_font_size=self.cb_label_size
                    )
                    # [WHT] 폭을 60px로 고정 (사용자 요청 사항)
                    self.plotter.scalar_bar.SetMaximumWidthInPixels(60)
            self.plotter.render()

    def _on_cb_style_changed(self, *_):
        """Handles changes in Colorbar Display Mode, Levels, and Decimals."""
        if not QtWidgets: return
        self.cb_mode = self.combo_cb_mode.currentText()
        try:
            self.cb_levels = int(self.combo_cb_levels.currentText())
        except ValueError:
            self.cb_levels = 10
        self.cb_decimals = self.spin_cb_decimals.value()
        
        self.combo_cb_levels.setEnabled(self.cb_mode == "Discrete")
        
        n_colors = self.cb_levels if self.cb_mode == "Discrete" else 256
        for name, part in self.parts.items():
            actor = part.get("actor")
            if actor and hasattr(actor, "mapper"):
                mapper = actor.mapper
                if hasattr(mapper, "lookup_table"):
                    try:
                        mapper.lookup_table.n_values = n_colors
                        mapper.lookup_table.Build()
                    except (AttributeError, pyvista.core.errors.PyVistaAttributeError):
                        pass
        
        # Redraw scalar bar with new formats
        if self._is_ready:
            self.set_colorbar_font_sizes(self.cb_title_size, self.cb_label_size)

    def _populate_combo_box(self):
        """Fills available results into the properties dropdown grouping by category."""
        if self.result_data is None:
            return
            
        self.combo_category.blockSignals(True)
        self.combo_category.clear()
        
        avail = set()
        
        # Safely get point_data keys
        point_data = getattr(self.result_data, 'point_data', {})
        for name in point_data.keys():
            avail.add(name)
            if self.result_data.point_data[name].ndim > 1:
                avail.add(f"{name}_Magnitude")
                avail.add(f"{name}_X")
                avail.add(f"{name}_Y")
                avail.add(f"{name}_Z")
        
        # Safely get cell_data keys
        cell_data = getattr(self.result_data, 'cell_data', {})
        _tensor_suffixes = ["_XX", "_YY", "_ZZ", "_XY", "_XZ", "_YZ",
                            "_VonMises", "_Signed_VonMises", "_Max_Principal", "_Min_Principal",
                            "_Max_3D_Principal", "_Mid_3D_Principal", "_Min_3D_Principal"]
        for name in cell_data.keys():
            avail.add(name)
            if "Stress" in name or "Strain" in name:
                # [WHT] 파생 필드 사전 등록: cell_data_to_point_data() 후 생성될 필드들
                for sfx in _tensor_suffixes:
                    avail.add(f"{name}{sfx}")
        
        self.avail_results = avail
        
        # Categorize: Filter out redundant suffixed fields from the main Category list
        categories = set()
        suffixes = ["_Magnitude", "_X", "_Y", "_Z", "_XX", "_YY", "_ZZ", "_XY", "_YZ", "_XZ", "_VonMises", "_Signed_VonMises", "_Max_Principal", "_Min_Principal", "_Max_3D_Principal", "_Mid_3D_Principal", "_Min_3D_Principal"]
        
        # Shell Layer 접미사: 카테고리 목록에서 제외 (Shell Layer 콤보로 전환)
        _shell_layer_suffixes = [" (Mid)", " (Lower)", " (Max Envelope)", " (Membrane)", " (Bending)"]
        
        for f in avail:
            is_redundant = False
            for s in suffixes:
                if f.endswith(s):
                    is_redundant = True
                    break
            
            if not is_redundant:
                # Shell Layer 변형 카테고리 필터링
                is_layer_variant = False
                for ls in _shell_layer_suffixes:
                    if ls in f:
                        is_layer_variant = True
                        break
                if not is_layer_variant:
                    categories.add(f)
        
        cats = sorted(list(categories))
        cats.insert(0, "Body Color")
        
        self.combo_category.addItems(cats)
        
        # Forced Default to Body Color as per USER_REQUEST
        self.combo_category.setCurrentText("Body Color")
            
        self.combo_category.blockSignals(False)
        
        # Populate Warp Field Selector (Discovery: All 3D point-data vectors + Bead_Height exception)
        self.combo_warp_vec.blockSignals(True)
        self.combo_warp_vec.clear()
        warp_candidates = []
        for name, data in self.result_data.point_data.items():
            # Standard WHT Vector is (Steps, Nodes, 3) or (Steps, Nodes, D>=3)
            # We filter for fields that can provide a coordinate offset.
            if data.ndim == 3 and data.shape[2] >= 3:
                warp_candidates.append(name)
                
        # 만약 Bead_Height 가 point_data 나 cell_data 에 있다면 강제 후보 추가
        has_bead = False
        if "Bead_Height" in self.result_data.point_data:
            has_bead = True
        elif "Bead_Height" in self.result_data.cell_data:
            has_bead = True
            
        if has_bead and "Bead_Height" not in warp_candidates:
            warp_candidates.append("Bead_Height")
                
        self.combo_warp_vec.addItems(sorted(warp_candidates))
        self.combo_warp_vec.blockSignals(False)
        
        self._on_category_changed(self.combo_category.currentText())

    def show(self, block=True):
        """Displays the visualizer and enters the event loop if requested."""
        if not self.plotter or not hasattr(self.plotter, 'app_window'):
            print(" -> [Visualizer Error] Plotter or app_window not initialized.")
            return

        print(" -> [Visualizer] Invoking Window...")
        self.plotter.app_window.show()
        self.plotter.app_window.raise_()
        self.plotter.app_window.activateWindow()
            
        app = QtWidgets.QApplication.instance()
        if not app:
            # BackgroundPlotter usually handles this, but we provide a fallback for robustness.
            import sys
            app = QtWidgets.QApplication(sys.argv)
            
        if block:
            app.setQuitOnLastWindowClosed(True)
            app.processEvents()
            print(" -> [Visualizer] Entering Qt Event Loop. Close window to release terminal.")
            # Use exec_() for compatibility or exec() for modern PySide
            if hasattr(app, 'exec'):
                app.exec()
            elif hasattr(app, 'exec_'):
                app.exec_()
            else:
                app.exec_()

    def visualize(self):
        """Compatibility alias for script execution."""
        self.show()

    def close(self):
        """Safe shutdown."""
        self.anim_timer.stop()
        if self.plotter:
            self.plotter.close()

    def _on_query_toggled(self, state):
        """Query 기능 체크/언체크 이벤트 핸들러."""
        enabled = (state == QtCore.Qt.Checked)
        print(f" -> [Visualizer] Query mode toggled: {enabled}")
        if not enabled:
            if hasattr(self.plotter, "app_window") and self.plotter.app_window:
                sb = self.plotter.app_window.statusBar()
                if sb:
                    sb.clearMessage()

    def _on_query_target_changed(self, toggled):
        """Query 대상(Node / Element) 변경 이벤트 핸들러."""
        if not self._is_ready: return

    def _clear_query_labels(self):
        """화면 상에 생성된 모든 Query 라벨 소거."""
        if not hasattr(self, "_query_label_names") or not self._query_label_names:
            self._query_label_names = []
            return
        for name in list(self._query_label_names):
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        self._query_label_names.clear()
        self.plotter.render()
        print(" -> [Visualizer] Cleared all query labels.")

    def _get_picker_result(self, pos: QtCore.QPoint):
        """마우스 Qt 좌표를 기반으로 vtkCellPicker를 통해 3D 충돌 결과를 가져옵니다."""
        import vtk
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        
        interactor = self.plotter.interactor
        if not interactor:
            return None
            
        size = interactor.GetRenderWindow().GetSize()
        x = pos.x()
        y = size[1] - pos.y()
        
        renderer = self.plotter.renderer
        picker.Pick(x, y, 0.0, renderer)
        
        actor = picker.GetActor()
        if not actor:
            return None
            
        return picker

    def _on_qt_mouse_move(self, event):
        """마우스 hover 시 실시간 값을 상태바에 출력하기 위한 QVTK interactor 몽키패치 핸들러."""
        if hasattr(self, "_orig_mouse_move") and self._orig_mouse_move:
            try:
                self._orig_mouse_move(event)
            except Exception:
                pass
            
        if hasattr(self, "chk_query") and self.chk_query.isChecked():
            try:
                self._process_hover_pick(event.pos())
            except Exception as e:
                print(f" -> [Visualizer Query Move Error] {e}")

    def _process_hover_pick(self, pos: QtCore.QPoint):
        """Hover 쿼리 비즈니스 로직 처리."""
        if not self._is_ready or not self.parts: return
        picker = self._get_picker_result(pos)
        sb = self.plotter.app_window.statusBar()
        if not sb: return
        
        if not picker:
            sb.clearMessage()
            return
            
        actor = picker.GetActor()
        found_name = None
        found_part = None
        for name, p in self.parts.items():
            if p["actor"] == actor:
                found_name = name
                found_part = p
                break
                
        if not found_part:
            sb.clearMessage()
            return
            
        mesh = found_part["mesh"]
        field = self._get_active_field_name()
        if not field or field == "Body Color":
            sb.showMessage(f"[Part: {found_name}] Query active but active scalar is 'Body Color'")
            return
            
        if self.rad_query_node.isChecked():
            p_idx = picker.GetPointId()
            if p_idx < 0 or p_idx >= mesh.n_points:
                sb.clearMessage()
                return
            
            val = 0.0
            if field in mesh.point_data:
                val = mesh.point_data[field][p_idx]
            elif field in mesh.cell_data:
                # Node Value 모드인데 필드가 cell_data에만 있는 경우: 인접 셀들의 평균 계산
                import vtk
                cell_ids = vtk.vtkIdList()
                mesh.GetPointCells(p_idx, cell_ids)
                n_cells = cell_ids.GetNumberOfIds()
                if n_cells > 0:
                    vals = [mesh.cell_data[field][cell_ids.GetId(i)] for i in range(n_cells) if cell_ids.GetId(i) < mesh.n_cells]
                    val = np.mean(vals) if vals else 0.0
            
            orig_nids = mesh.point_data.get("vtkOriginalPointIds")
            global_node_id = orig_nids[p_idx] if orig_nids is not None else p_idx
            coords = mesh.points[p_idx]
            
            sb.showMessage(f"[Part: {found_name}] Node: {global_node_id} | Coord: ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}) | Value: {val:.4e}")
            
        else:
            c_idx = picker.GetCellId()
            if c_idx < 0 or c_idx >= mesh.n_cells:
                sb.clearMessage()
                return
                
            val = 0.0
            if field in mesh.cell_data:
                val = mesh.cell_data[field][c_idx]
            elif field in mesh.point_data:
                cell = mesh.GetCell(c_idx)
                pts_ids = cell.GetPointIds()
                n_ids = pts_ids.GetNumberOfIds()
                val = np.mean([mesh.point_data[field][pts_ids.GetId(i)] for i in range(n_ids)])
                
            orig_cids = mesh.cell_data.get("vtkOriginalCellIds")
            global_cell_id = orig_cids[c_idx] if orig_cids is not None else c_idx
            
            sb.showMessage(f"[Part: {found_name}] Element: {global_cell_id} | Value: {val:.4e}")

    def _on_mouse_double_click(self, event):
        """마우스 더블클릭 시 3D 공간 상에 8pt 값 라벨을 생성하기 위한 QVTK interactor 몽키패치 핸들러."""
        if hasattr(self, "_orig_double_click") and self._orig_double_click:
            try:
                self._orig_double_click(event)
            except Exception:
                pass
            
        if hasattr(self, "chk_query") and self.chk_query.isChecked():
            try:
                self._process_double_click_pick(event.pos())
            except Exception as e:
                print(f" -> [Visualizer Query DoubleClick Error] {e}")

    def _process_double_click_pick(self, pos: QtCore.QPoint):
        """더블클릭 쿼리 및 라벨 생성 비즈니스 로직 처리."""
        if not self._is_ready or not self.parts: return
        picker = self._get_picker_result(pos)
        if not picker: return
        
        actor = picker.GetActor()
        found_name = None
        found_part = None
        for name, p in self.parts.items():
            if p["actor"] == actor:
                found_name = name
                found_part = p
                break
                
        if not found_part: return
        
        mesh = found_part["mesh"]
        field = self._get_active_field_name()
        if not field or field == "Body Color": return
        
        pick_pos = picker.GetPickPosition()
        
        label_text = ""
        unique_id = 0
        if self.rad_query_node.isChecked():
            p_idx = picker.GetPointId()
            if p_idx < 0 or p_idx >= mesh.n_points: return
            
            val = 0.0
            if field in mesh.point_data:
                val = mesh.point_data[field][p_idx]
            elif field in mesh.cell_data:
                # Node Value 모드인데 필드가 cell_data에만 있는 경우: 인접 셀들의 평균 계산
                import vtk
                cell_ids = vtk.vtkIdList()
                mesh.GetPointCells(p_idx, cell_ids)
                n_cells = cell_ids.GetNumberOfIds()
                if n_cells > 0:
                    vals = [mesh.cell_data[field][cell_ids.GetId(i)] for i in range(n_cells) if cell_ids.GetId(i) < mesh.n_cells]
                    val = np.mean(vals) if vals else 0.0
            
            orig_nids = mesh.point_data.get("vtkOriginalPointIds")
            global_node_id = orig_nids[p_idx] if orig_nids is not None else p_idx
            label_text = f"N{global_node_id}: {val:.3e}"
            unique_id = global_node_id
            
        else:
            c_idx = picker.GetCellId()
            if c_idx < 0 or c_idx >= mesh.n_cells: return
            
            val = 0.0
            if field in mesh.cell_data:
                val = mesh.cell_data[field][c_idx]
            elif field in mesh.point_data:
                cell = mesh.GetCell(c_idx)
                pts_ids = cell.GetPointIds()
                n_ids = pts_ids.GetNumberOfIds()
                val = np.mean([mesh.point_data[field][pts_ids.GetId(i)] for i in range(n_ids)])
                
            orig_cids = mesh.cell_data.get("vtkOriginalCellIds")
            global_cell_id = orig_cids[c_idx] if orig_cids is not None else c_idx
            label_text = f"E{global_cell_id}: {val:.3e}"
            unique_id = global_cell_id
            
        if not hasattr(self, "_query_label_names") or self._query_label_names is None:
            self._query_label_names = []
            
        lbl_name = f"_query_lbl_{len(self._query_label_names)}_{unique_id}"
        
        self.plotter.add_point_labels(
            [pick_pos], [label_text],
            font_size=8,
            font_family='courier',
            show_points=True,
            point_color='magenta',
            point_size=5,
            text_color='cyan',
            name=lbl_name
        )
        
        self._query_label_names.append(lbl_name)
        self.plotter.render()


def export_to_wht_result(model, result):
    """
    Convenience function to convert WHTSolverResult -> WHTResultData.
    Internalizes WHTMetadata creation with default N-mm-ton-s units.
    """
    from wht_converter.wht_models import WHTMetadata
    metadata = WHTMetadata(
        solver_name="WHTSolver",
        solver_version="0.1.0",
        analysis_type=result.analysis_type,
        coordinate_system="cartesian",
        unit_length="mm",
        unit_force="N",
        unit_mass="tonne",
        unit_time="s"
    )
    return result.to_wht_result_data(metadata, model)


def visualize(result_data, block=True):
    """
    Convenience function to pop up the WHTVisualizer frame with a given result.
    If block=False, returns the visualizer instance without entering the event loop.
    """
    vis = WHTVisualizer()
    vis.show_result(result_data)
    vis.show(block=block)
    return vis


def launch_paraview(file_path: str) -> bool:
    """
    [WHT Premium UX] Automatically locates and launches ParaView, loading the specified file (.hdf / .pvd / .vtu).
    Runs as a decoupled, non-blocking background subprocess.
    """
    import subprocess
    import shutil
    import glob
    from pathlib import Path

    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        print(f" -> [ParaView Launcher Warning] Result file not found: {abs_path}")
        return False

    # 1. Search in system PATH
    paraview_bin = shutil.which("paraview")
    if paraview_bin:
        print(f" -> [ParaView Launcher] Found paraview command in system PATH: {paraview_bin}")
    else:
        # 2. Search in standard Windows installations (C:\\Program Files\\ParaView*)
        pf = os.environ.get("ProgramFiles", "C:\\Program Files")
        search_pattern = os.path.join(pf, "ParaView*", "bin", "paraview.exe")
        candidates = glob.glob(search_pattern)
        if candidates:
            # Sort to pick the latest version candidate folder
            candidates.sort(reverse=True)
            paraview_bin = candidates[0]
            print(f" -> [ParaView Launcher] Found standard installation: {paraview_bin}")

    if not paraview_bin:
        print("\n [ParaView Launcher Warning] ParaView installation not detected.")
        print("   - Please add ParaView's bin/ directory to your Windows PATH environment variable, or")
        print("   - Install ParaView into the standard 'C:\\Program Files\\ParaView-X.X.X' directory for auto-launching.")
        return False

    print(f" -> [ParaView Launcher] Launching ParaView non-blockingly... Loading: {abs_path}")
    try:
        creationflags = 0
        if os.name == 'nt':
            # CREATE_NEW_PROCESS_GROUP (0x00000200) to safely detach subprocess lifecycle from parent Python process
            creationflags = 0x00000200

        subprocess.Popen(
            [paraview_bin, abs_path],
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True
        )
        print(" -> [ParaView Launcher] ParaView successfully spawned in background!")
        return True
    except Exception as e:
        print(f" -> [ParaView Launcher ERROR] Failed to spawn ParaView: {e}")
        return False


def visualize_in_paraview(result_data: "WHTResultData", temp_dir: Optional[str] = None) -> bool:
    """
    [WHT Premium UX] Exports WHTResultData IR to a temporary VTKHDF file and opens it in ParaView instantly.
    """
    import tempfile
    from wht_converter.wht_exporters import VTKHDFExporter

    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    hdf_path = os.path.join(temp_dir, "wht_transient_temp.hdf")
    print(f" -> [ParaView Visualizer] Exporting temporary high-fidelity transient geometry VTKHDF file...")
    try:
        # Default to transient_geometry=True for smooth moving-mesh animation
        VTKHDFExporter(transient_geometry=True).export(result_data, hdf_path)
    except Exception as e:
        print(f" -> [ParaView Visualizer ERROR] HDF export failed: {e}")
        return False

    return launch_paraview(hdf_path)
