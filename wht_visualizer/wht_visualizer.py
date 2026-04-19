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
from typing import Dict, List, Optional, Any, TYPE_CHECKING

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
        pv.global_theme.edge_color = 'grey'
        
        # 2. Initialize BackgroundPlotter First
        # BackgroundPlotter automatically creates a QApplication if one doesn't exist.
        # Force toolbar=False to hide redundant default PyVistaQt buttons.
        self.plotter = BackgroundPlotter(title=title, show=show, toolbar=False)
        self.plotter.set_background('black') # Rule-aligned: Force black as requested
        
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

        # 3. Setup UI Pipeline (Order Matters!)
        self._setup_tabbed_dock()     # Creates self.list_parts
        self._setup_playback_ui()     # Creates playback controls
        self._setup_view_controls()
        self._setup_toolbar()         # Connects to list_parts
        self._setup_menubar()         # Creates pull-down menus
        
        self._is_ready = True

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
        
        # Basics Group (Renamed from Deformation)
        group_basics = QtWidgets.QGroupBox("Basics")
        vbox_basics = QtWidgets.QVBoxLayout()
        vbox_basics.setSpacing(2)
        
        # Row 1: Deformation
        hbox_warp = QtWidgets.QHBoxLayout()
        hbox_warp.setSpacing(5)
        self.chk_warp = QtWidgets.QCheckBox("Use Deformation")
        self.chk_warp.setChecked(True)
        self.chk_warp.stateChanged.connect(self._on_warp_toggled)
        
        self.spin_scale = QtWidgets.QDoubleSpinBox()
        self.spin_scale.setRange(-1000.0, 1000.0)
        self.spin_scale.setValue(1.0)
        self.spin_scale.setFixedWidth(80)
        self.spin_scale.valueChanged.connect(self._on_warp_scale_changed)
        
        hbox_warp.addWidget(self.chk_warp)
        
        # New: Selectable Deformation Vector Field
        self.combo_warp_vec = QtWidgets.QComboBox()
        self.combo_warp_vec.setMinimumWidth(100)
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
        
        vbox_basics.addLayout(hbox_warp)
        vbox_basics.addLayout(hbox_bcload)
        group_basics.setLayout(vbox_basics)

        
        # Contour/Colorbar Group
        group_contour = QtWidgets.QGroupBox("Fields")
        vbox_contour = QtWidgets.QVBoxLayout()
        vbox_contour.setSpacing(2)
        
        hbox_result = QtWidgets.QHBoxLayout()
        hbox_result.setSpacing(2)
        # Coloring label removed as per USER_REQUEST
        
        # Dual-Combo System: Category and Component
        self.combo_category = QtWidgets.QComboBox()
        self.combo_component = QtWidgets.QComboBox()
        self.combo_category.currentTextChanged.connect(self._on_category_changed)
        self.combo_component.currentTextChanged.connect(self._on_component_changed)
        
        hbox_result.addWidget(self.combo_category, 2)
        hbox_result.addWidget(self.combo_component, 1)
        
        # Colorbar Range Logic (Flattened into Fields)
        self.combo_range_mode = QtWidgets.QComboBox()
        self.combo_range_mode.addItems(["Dynamic (Auto)", "Static (Fixed)"])
        self.combo_range_mode.currentTextChanged.connect(self._on_range_mode_changed)
        
        hbox_spin = QtWidgets.QHBoxLayout()
        hbox_spin.setSpacing(2)
        self.spin_min = QtWidgets.QDoubleSpinBox()
        self.spin_min.setRange(-1e15, 1e15)
        self.spin_min.setEnabled(False)
        self.spin_max = QtWidgets.QDoubleSpinBox()
        self.spin_max.setRange(-1e15, 1e15)
        self.spin_max.setEnabled(False)
        hbox_spin.addWidget(self.spin_min)
        hbox_spin.addWidget(self.spin_max)
        
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
        
        # Assemble into Fields Main Layout in logical order
        vbox_contour.addLayout(hbox_result)
        vbox_contour.addWidget(self.combo_range_mode)
        vbox_contour.addLayout(hbox_spin)
        vbox_contour.addLayout(hbox_cb_mode)
        vbox_contour.addLayout(hbox_cmap)
        
        group_contour.setLayout(vbox_contour)
        
        prop_layout.addWidget(group_basics)
        prop_layout.addWidget(group_contour)

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
        self.btn_first = QtWidgets.QPushButton("|<")
        self.btn_prev = QtWidgets.QPushButton("<")
        self.btn_play = QtWidgets.QPushButton("▶")
        self.btn_next = QtWidgets.QPushButton(">")
        self.btn_last = QtWidgets.QPushButton(">|")
        
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
        
        # Theme Menu
        theme_menu = menu.addMenu("Theme")
        theme_menu.addAction("Dark Mode", lambda: self._set_theme(True))
        theme_menu.addAction("Light Mode", lambda: self._set_theme(False))
        
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
        self._add_toolbar_action("BG", "Toggle Background", self._create_view_icon("bg"), self._on_bg_toggle)
        self.list_parts.itemChanged.connect(self._on_part_item_changed)

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

    def _set_theme(self, is_dark: bool):
        if self._bg_is_dark == is_dark: return
        self._bg_is_dark = is_dark
        
        bg_color = 'black' if self._bg_is_dark else 'white'
        font_color = 'white' if self._bg_is_dark else 'black'
        
        # 1. Apply to plotter and global theme
        self.plotter.set_background(bg_color)
        pv.global_theme.font.color = font_color
        
        # 2. Force apply to all renderers (Critical for BackgroundPlotter)
        for renderer in self.plotter.renderers:
            renderer.set_background(bg_color)
            
        # 3. Update Axes
        self.plotter.add_axes(color=font_color)
        self.plotter.render()

    def _on_bg_toggle(self):
        """[WHT Professional] Toggles between Dark and Light analysis themes."""
        self._set_theme(not self._bg_is_dark)

    def _take_screenshot(self):
        if not QtWidgets: return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.plotter.app_window, "Save Screenshot", "wht_capture.png", "PNG Image (*.png)"
        )
        if file_path:
            self.plotter.screenshot(file_path)
            print(f" -> Screenshot saved to {file_path}")

    def show_result(self, result: "WHTResultData"):
        """Main entry point to display WHTResultData IR objects with integrity guards."""
        if not result or result.nodes is None:
            print(" -> [Visualizer Warning] Attempted to load empty result data.")
            return

        self._is_ready = False
        self.result_data = result
        self.current_timestep = 0
        
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
        
        # 2. Populate UI based on results (Before adding parts!)
        self._populate_combo_box()
        
        # 3. Rebuild Assembly (Shared Pointer Oriented)
        shared_base = self._make_pv_grid(result.nodes, result.connectivity, result.offsets, result.cell_types)
        
        # Determine Parts from element_sets
        if result.element_sets:
            for part_name, elem_indices in result.element_sets.items():
                if len(elem_indices) == 0: continue
                # Use shared_base to extract submesh directly
                part_mesh = shared_base.extract_cells(elem_indices)
                self._add_part(part_name, part_mesh)
        else:
            # Fallback: Single Default Part
            self._add_part("Mesh_Model", shared_base)
            
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
        # PyVista/VTK 9+ preferred 'cells' format: [n1, p1, p2, ..., n2, p_k, ...]
        cell_counts = np.diff(offsets)
        cells = np.empty(len(connectivity) + len(cell_counts), dtype=connectivity.dtype)
        
        insert_pos = np.arange(len(cell_counts)) + offsets[:-1]
        cells[insert_pos] = cell_counts
        
        mask = np.ones(len(cells), dtype=bool)
        mask[insert_pos] = False
        cells[mask] = connectivity
        
        return pv.UnstructuredGrid(cells, cell_types, nodes)

    def load_results(self, result: "WHTResultData", **kwargs):
        """Compatibility alias for show_result."""
        try:
            self.show_result(result)
            # Apply visual hints from kwargs if provided
            if 'color' in kwargs:
                self.plotter.set_background(kwargs['color'])
            if 'label' in kwargs:
                # Overwrite or add a floating label
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
        n_col = self.cb_levels if self.cb_mode == "Discrete" else 256
        n_lbl = (self.cb_levels + 1) if self.cb_mode == "Discrete" else 11
        fmt_str = f"%.{self.cb_decimals}e"
        
        actor = self.plotter.add_mesh(
            mesh, 
            name=name, 
            pickable=True,
            show_edges=True,
            edge_color='grey',
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
                width=0.12,
                title_font_size=self.cb_title_size,
                label_font_size=self.cb_label_size
            )
        
        # Trigger explicit update if Body Color
        if current_field == "Body Color":
            actor.prop.color = 'lightgrey'
            if self.plotter.scalar_bars:
                self.plotter.remove_scalar_bar()
            
        self.parts[name] = {
            "mesh": mesh, 
            "actor": actor, 
            "orig_pts": mesh.points.copy(),
            "active_mesh": mesh
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
                        target_mesh = part["feature"]
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
        """Toggles spinbox availability and snaps to current range if static."""
        is_static = (mode == "Static (Fixed)")
        self.spin_min.setEnabled(is_static)
        self.spin_max.setEnabled(is_static)
        
        if is_static:
            current_field = self._get_active_field_name()
            if current_field:
                # Find global min-max across all parts for this field
                total_rng = [float('inf'), float('-inf')]
                for part in self.parts.values():
                    if current_field in part["mesh"].point_data:
                        r = part["mesh"].get_data_range(current_field)
                        total_rng[0] = min(total_rng[0], r[0])
                        total_rng[1] = max(total_rng[1], r[1])
                
                if total_rng[0] != float('inf'):
                    self.spin_min.setValue(total_rng[0])
                    self.spin_max.setValue(total_rng[1])
        
        self.set_timestep(self.current_timestep)

    def _apply_colorbar_range(self):
        """Applies global min-max range to the scalar bar across all parts."""
        if not self.parts: return # Guard: Cannot add scalar bar without active mappers
        
        mode = self.combo_range_mode.currentText()
        field = self._get_active_field_name()
        if not field: return
        
        if mode == "Dynamic (Auto)":
            total_rng = [float('inf'), float('-inf')]
            for part in self.parts.values():
                if field in part["mesh"].point_data or field in part["mesh"].cell_data:
                    r = part["mesh"].get_data_range(field)
                    total_rng[0] = min(total_rng[0], r[0])
                    total_rng[1] = max(total_rng[1], r[1])
            
            if total_rng[0] == float('inf'): rng = [0, 1]
            else: rng = total_rng
        else:
            rng = [self.spin_min.value(), self.spin_max.value()]
            
        # Update all mappers to use this range
        for part in self.parts.values():
            part["actor"].mapper.scalar_range = rng
            
        # Update scalar bar according to User Rule 3.1
        if field == "Body Color":
            if self.plotter.scalar_bars:
                self.plotter.remove_scalar_bar()
        else:
            if self.plotter.scalar_bars:
                self.plotter.remove_scalar_bar()
                
            n_lbl = (self.cb_levels + 1) if self.cb_mode == "Discrete" else 11
            fmt_str = f"%.{self.cb_decimals}e"
            
            display_title = field.replace(" ", "\n").replace("_", "\n") if len(field) > 8 else field
            active_mapper = list(self.parts.values())[0]["actor"].mapper if self.parts else None
            
            self.plotter.add_scalar_bar(
                title=display_title,
                mapper=active_mapper,
                vertical=True,
                position_x=0.85, # Standard ParaView offset
                position_y=0.1,
                height=0.8,
                width=0.12, # Elegant narrow bar
                title_font_size=self.cb_title_size,
                label_font_size=self.cb_label_size,
                font_family='arial',
                fmt=fmt_str,
                shadow=True,
                n_labels=n_lbl
            )
            self.plotter.update_scalar_bar_range(rng)

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
                        if orig_ids is not None:
                            # Submesh mapping
                            if len(data) >= np.max(orig_ids):
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
                        orig_cids = mesh.cell_data.get('vtkOriginalCellIds')
                        if orig_cids is not None:
                            # Strict Index Safety Guard: Ensure data array covers all original IDs
                            if len(data) > np.max(orig_cids):
                                c_data = data[orig_cids]
                            else:
                                continue
                        else:
                            if len(data) == mesh.n_cells:
                                c_data = data
                            else:
                                continue
                        
                        mesh.cell_data[sc_name] = c_data
                        
                        if "Stress" in sc_name or "Strain" in sc_name:
                            if c_data.ndim > 1 and c_data.shape[1] == 6:
                                # Standard FEA tensor decomposition
                                c_data = np.nan_to_num(c_data)
                                s11, s22, s33, s12, s13, s23 = c_data.T
                                
                                # 1. Natural Components
                                mesh.cell_data[f"{sc_name}_XX"] = s11
                                mesh.cell_data[f"{sc_name}_YY"] = s22
                                mesh.cell_data[f"{sc_name}_ZZ"] = s33
                                
                                # 2. Von Mises (Classic)
                                vm = np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 + 6*(s12**2 + s13**2 + s23**2)))
                                mesh.cell_data[f"{sc_name}_VonMises"] = vm
                                
                                # 3. Principal Stresses (Simplified 2D for shell models)
                                avg = (s11 + s22) / 2.0
                                radius = np.sqrt(((s11 - s22) / 2.0)**2 + s12**2)
                                mesh.cell_data[f"{sc_name}_Max_Principal"] = avg + radius
                                mesh.cell_data[f"{sc_name}_Min_Principal"] = avg - radius

                if current_field and (current_field in mesh.point_data or current_field in mesh.cell_data):
                    mesh.set_active_scalars(current_field)
                    part["actor"].mapper.SetInputData(mesh)
                    # Force scalar visibility if a real result is selected
                    if current_field != "Body Color":
                        part["actor"].mapper.scalar_visibility = True
                    else:
                        part["actor"].mapper.scalar_visibility = False

            self._apply_colorbar_range()
        except Exception as e:
            print(f" -> [Visualizer Error] Failed to bind data at step {t_idx}: {e}")

    def _apply_warping(self):
        """Applies displacement warping to all parts."""
        if not self.result_data: return
        chk = self.chk_warp.isChecked()
        scale = self.spin_scale.value()
        
        
        warp_field = self.combo_warp_vec.currentText()
        if not warp_field or warp_field not in self.result_data.point_data:
            # Fallback to pure original points if no valid field
            for name, part in self.parts.items():
                part["mesh"].points = part["orig_pts"]
                active_mesh = part.get("active_mesh", part["mesh"])
                if active_mesh.n_points == part["mesh"].n_points:
                    active_mesh.points = part["orig_pts"]
            return

        for name, part in self.parts.items():
            mesh = part["mesh"]
            orig_pts = part["orig_pts"]
            orig_ids = mesh.point_data.get('vtkOriginalPointIds')
            
            if chk:
                disp_all = self.result_data.point_data[warp_field]
                if self.current_timestep < len(disp_all):
                    disp = np.nan_to_num(disp_all[self.current_timestep])
                    
                    # Update Base Mesh
                    if orig_ids is not None:
                        mesh.points = orig_pts + disp[orig_ids, :3] * scale
                    else:
                        if len(disp) == len(orig_pts):
                            mesh.points = orig_pts + disp[:, :3] * scale
                    
                    # Proactive Integrity: Also update active filtered mesh points if they share the same point count
                    # Outline often has different points, but we can try to warp it if it has original IDs too
                    active_mesh = part.get("active_mesh", mesh)
                    if active_mesh is not mesh:
                        # For filters like Feature Edges that preserve points, we can warp directly
                        if active_mesh.n_points == mesh.n_points:
                            active_mesh.points = mesh.points
                        else:
                            # If it's an outline or decimated mesh, we might need to re-generate 
                            # but that's slow. For now, we only warp point-preserving filters.
                            pass
            else:
                mesh.points = orig_pts
                active_mesh = part.get("active_mesh", mesh)
                if active_mesh.n_points == mesh.n_points:
                    active_mesh.points = orig_pts
                
            if part["actor"] and hasattr(part["actor"], "mapper"):
                part["actor"].mapper.SetInputData(mesh)
        
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
            self.combo_component.blockSignals(False)
            self._update_active_result()
            return
            
        self.combo_component.setEnabled(True)
        
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
        self._update_active_result()

    def _on_component_changed(self, component):
        self._update_active_result()

    def _update_active_result(self):
        """Synthesizes the full result name and applies it."""
        full_name = self._get_active_field_name()
        self._on_result_type_changed(full_name)

    def _get_active_field_name(self) -> str:
        """Synthesizes the full result name from dual combos."""
        if not hasattr(self, 'combo_category'): return ""
        cat = self.combo_category.currentText()
        comp = self.combo_component.currentText()
        if cat == "Body Color": return "Body Color"
        if comp == "Value": return cat
        return f"{cat}_{comp}"

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
                    if hasattr(actor.mapper, 'lookup_table'):
                        actor.mapper.lookup_table.cmap = self.combo_cmap.currentText()
                        if hasattr(actor.mapper.lookup_table, 'Build'):
                            actor.mapper.lookup_table.Build()
                    
        self._apply_colorbar_range()
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
            act_bc = self.plotter.add_mesh(
                poly, color='#ff2222', point_size=12, 
                render_points_as_spheres=True, 
                pickable=False
            )
            act_bc.SetVisibility(self.chk_bc.isChecked())
            self.actors_misc["BC"] = act_bc
            
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
            act_load = self.plotter.add_mesh(
                poly, color='#22ff22', point_size=10, 
                render_points_as_spheres=True, 
                pickable=False
            )
            act_load.SetVisibility(self.chk_load.isChecked())
            self.actors_misc["Load"] = act_load

    def _on_colormap_changed(self, cmap):
        for part in self.parts.values():
            actor = part.get("actor")
            if actor and hasattr(actor, "mapper") and hasattr(actor.mapper, "lookup_table"):
                actor.mapper.lookup_table.cmap = cmap
                # Force rebuild of the LookupTable to apply the new colormap immediately
                if hasattr(actor.mapper.lookup_table, "Build"):
                    actor.mapper.lookup_table.Build()
        self.plotter.render()

    def _on_fps_changed(self, value):
        if self.is_playing:
            self.anim_timer.setInterval(int(1000 / value))

    def _toggle_animation(self):
        self.is_playing = not self.is_playing
        self.btn_play.setText("⏸" if self.is_playing else "▶")
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
                        width=0.12,
                        vertical=True,
                        font_family='arial',
                        title_font_size=self.cb_title_size,
                        label_font_size=self.cb_label_size
                    )
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
        for name in cell_data.keys():
            avail.add(name)
            if "Stress" in name or "Strain" in name:
                avail.add(f"{name}_VonMises")
        
        self.avail_results = avail
        
        # Categorize: Filter out redundant suffixed fields from the main Category list
        categories = set()
        suffixes = ["_Magnitude", "_X", "_Y", "_Z", "_XX", "_YY", "_ZZ", "_XY", "_YZ", "_XZ", "_VonMises", "_Max_Principal", "_Min_Principal"]
        
        for f in avail:
            is_redundant = False
            for s in suffixes:
                if f.endswith(s):
                    is_redundant = True
                    break
            
            if not is_redundant:
                categories.add(f)
        
        cats = sorted(list(categories))
        cats.insert(0, "Body Color")
        
        self.combo_category.addItems(cats)
        
        # Forced Default to Body Color as per USER_REQUEST
        self.combo_category.setCurrentText("Body Color")
            
        self.combo_category.blockSignals(False)
        
        # Populate Warp Field Selector (Vectors only)
        self.combo_warp_vec.blockSignals(True)
        self.combo_warp_vec.clear()
        warp_candidates = []
        for name, data in self.result_data.point_data.items():
            if data.ndim == 3 and data.shape[2] >= 3: # (steps, nodes, components)
                warp_candidates.append(name)
        self.combo_warp_vec.addItems(sorted(warp_candidates))
        self.combo_warp_vec.blockSignals(False)
        
        self._on_category_changed(self.combo_category.currentText())

    def show(self):
        """Displays the visualizer and enters the event loop if necessary."""
        if not self.plotter or not hasattr(self.plotter, 'app_window'):
            print(" -> [Visualizer Error] Plotter or app_window not initialized.")
            return

        print(" -> [Visualizer] Invoking Window...")
        self.plotter.app_window.show()
        self.plotter.app_window.raise_()
        self.plotter.app_window.activateWindow()
            
        app = QtWidgets.QApplication.instance()
        if app:
            app.setQuitOnLastWindowClosed(True)
            app.processEvents()
            print(" -> [Visualizer] Entering Qt Event Loop. Close window to release terminal.")
            # Use exec_() for compatibility or exec() for modern PySide
            if hasattr(app, 'exec'):
                app.exec()
            else:
                app.exec_()
        else:
            print(" -> [Visualizer Error] No active QApplication instance.")

    def visualize(self):
        """Compatibility alias for script execution."""
        self.show()

    def close(self):
        """Safe shutdown."""
        self.anim_timer.stop()
        if self.plotter:
            self.plotter.close()
