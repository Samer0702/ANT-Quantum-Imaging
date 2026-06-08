import sys
import os
import ctypes
import json
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QToolBar, QSizePolicy, QMessageBox, QCheckBox, QTabWidget, QFileDialog,
    QSlider, QSplitter, QMenu, QWidgetAction, QToolButton, QStyle
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon

from piezo_control_open import PiezoController
from camera_control import CameraController 
from ui_components import ClickableLabel, ScalableImageLabel
from acquisition_workers import SingleAcquisitionWorker, LiveFeedWorker, LiveProcessingWorker


class QIUP_APP(QMainWindow):
    """Main application window for Quantum Imaging with Undetected Photons (QIUP)."""

    _DEFAULT_SCAN_V_START = 0.0
    _DEFAULT_SCAN_V_END = 4.5
    _DEFAULT_SETTLING_MS = 10

    def __init__(self):
        super().__init__()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.logo_path = os.path.join(self.base_dir, "logo", "Logo_ANT.png")

        self.setWindowTitle("NANO: QIUP Dashboard")
        self.setGeometry(50, 50, 1800, 1000)
        
        # Set window icon if file exists
        if os.path.exists(self.logo_path):
            self.setWindowIcon(QIcon(self.logo_path))

        self.piezo: PiezoController | None = None
        self.camera: CameraController | None = None

        self.acq_worker: SingleAcquisitionWorker | None = None
        self.live_worker: LiveFeedWorker | None = None
        self.live_proc_worker: LiveProcessingWorker | None = None

        self.displacements_intensities: dict[float, float] = {}
        self.has_strain_gauge = False
        
        # Hidden variables for ROI center (replaces spinboxes)
        self.roi_x = 0
        self.roi_y = 0
        
        # Prevent rapid button clicking
        self._acquisition_in_progress = False

        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self):
        """Initialize the user interface."""
        # Preload standard icons
        self.play_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        self.stop_icon = self.style().standardIcon(QStyle.SP_MediaStop)
        self.save_icon = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        self.load_icon = self.style().standardIcon(QStyle.SP_ArrowUp)

        # Top toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("QToolBar { spacing: 8px; padding: 5px; }")
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        # Connect/Disconnect buttons
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setToolTip("Connect to Thorlabs camera and piezo controller")
        self.connect_btn.clicked.connect(self._connect_hardware)
        toolbar.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setToolTip("Safely disconnect all hardware")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(self._disconnect_hardware)
        toolbar.addWidget(self.disconnect_btn)

        toolbar.addSeparator()

        # Primary action buttons
        self.start_btn = QPushButton(" Single")
        self.start_btn.setProperty("isAction", True)
        self.start_btn.setIcon(self.play_icon)
        self.start_btn.setIconSize(QSize(16, 16))
        self.start_btn.setToolTip("Run single fixed-frame phase-stepping acquisition")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._run_acquisition)
        toolbar.addWidget(self.start_btn)

        self.live_proc_btn = QPushButton(" Live")
        self.live_proc_btn.setProperty("isAction", True)
        self.live_proc_btn.setIcon(self.play_icon)
        self.live_proc_btn.setIconSize(QSize(16, 16))
        self.live_proc_btn.setToolTip("Start continuous real-time quantum phase-stepping")
        self.live_proc_btn.setEnabled(False)
        self.live_proc_btn.clicked.connect(self._toggle_live_processing)
        toolbar.addWidget(self.live_proc_btn)

        self.live_btn = QPushButton(" Raw Feed")
        self.live_btn.setProperty("isAction", True) 
        self.live_btn.setIcon(self.play_icon)
        self.live_btn.setIconSize(QSize(16, 16))
        self.live_btn.setToolTip("View live raw CMOS camera feed (no piezo scanning)")
        self.live_btn.setEnabled(False)
        self.live_btn.clicked.connect(self._toggle_live_feed)
        toolbar.addWidget(self.live_btn)

        toolbar.addSeparator()

        # Scan parameters dropdown
        scan_menu = QMenu(self)
        scan_widget = QWidget()
        scan_widget.setObjectName("dropdownMenu")
        scan_lay = QFormLayout(scan_widget)
        
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(3, 1000)
        self.frames_spin.setValue(8)
        self.frames_spin.valueChanged.connect(self._validate_scan_params)
        
        self.scan_end_spin = QDoubleSpinBox()
        self.scan_end_spin.setRange(0.01, PiezoController.MAX_VOLTAGE)
        self.scan_end_spin.setDecimals(2)
        self.scan_end_spin.setSingleStep(0.25)
        self.scan_end_spin.setValue(self._DEFAULT_SCAN_V_END)
        self.scan_end_spin.setSuffix(" V")
        self.scan_end_spin.valueChanged.connect(self._validate_scan_params)
        
        self.settling_spin = QSpinBox()
        self.settling_spin.setRange(0, 1000)
        self.settling_spin.setValue(self._DEFAULT_SETTLING_MS)
        self.settling_spin.setSuffix(" ms")
        
        self.reset_btn = QPushButton("Reset System")
        self.reset_btn.setMinimumHeight(30)
        self.reset_btn.clicked.connect(self._reset_system)
        
        scan_lay.addRow("Frames (N):", self.frames_spin)
        scan_lay.addRow("Fringe Period:", self.scan_end_spin)
        scan_lay.addRow("Settling time:", self.settling_spin)
        scan_lay.addRow(self.reset_btn)
        
        scan_action = QWidgetAction(self)
        scan_action.setDefaultWidget(scan_widget)
        scan_menu.addAction(scan_action)

        self.scan_btn = QToolButton()
        self.scan_btn.setText("Scan Params")
        self.scan_btn.setToolTip("Configure frames, voltage period, and settling time")
        self.scan_btn.setMenu(scan_menu)
        self.scan_btn.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(self.scan_btn)

        toolbar.addSeparator()

        # Separate Load / Save Buttons
        self.load_btn = QPushButton("Load")
        self.load_btn.setToolTip("Load previous settings.")
        self.load_btn.clicked.connect(self._load_settings)
        toolbar.addWidget(self.load_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Save currently acquired data.")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_data)
        toolbar.addWidget(self.save_btn)

        # Central layout with splitter
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(5, 5, 5, 5)

        self.main_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.main_splitter)

        # Left column: Preview + Graph
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        preview_group = QGroupBox("Current Frame Preview")
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(10, 20, 10, 10)

        # --- CONSOLIDATED PREVIEW SETTINGS MENU ---
        preview_top_bar = QHBoxLayout()
        preview_top_bar.addStretch()
        
        preview_menu = QMenu(self)
        preview_settings_widget = QWidget()
        preview_settings_widget.setObjectName("dropdownMenu")
        preview_settings_lay = QVBoxLayout(preview_settings_widget)
        preview_settings_lay.setSpacing(15)

        # 1. CMOS Settings Group
        cmos_group = QGroupBox("CMOS Settings")
        cmos_lay = QFormLayout(cmos_group)
        
        self.exposure_spin = QSpinBox()
        self.exposure_spin.setRange(1, 5000)
        self.exposure_spin.setValue(200)
        self.exposure_spin.setSuffix(" ms")
        self.exposure_spin.valueChanged.connect(self._on_exposure_changed)
        
        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 48)
        self.gain_spin.setValue(35)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        
        self.ma_checkbox = QCheckBox("Use Moving Average Filter")
        self.ma_checkbox.toggled.connect(self._on_ma_toggled)
        
        self.ma_size_spin = QSpinBox()
        self.ma_size_spin.setSingleStep(2)
        self.ma_size_spin.setRange(3, 101)
        self.ma_size_spin.setValue(3)
        self.ma_size_spin.setSuffix(" px")
        self.ma_size_spin.setEnabled(False)
        self.ma_size_spin.valueChanged.connect(self._on_ma_size_changed)

        cmos_lay.addRow("Exposure:", self.exposure_spin)
        cmos_lay.addRow("Gain:", self.gain_spin)
        cmos_lay.addRow(self.ma_checkbox)
        cmos_lay.addRow("Window Size:", self.ma_size_spin)
        
        # 2. Raw Intensity Group
        intensity_group = QGroupBox("Raw Display Intensity (0 - 1023)")
        intensity_lay = QVBoxLayout(intensity_group)
        
        self.raw_auto_cb = QCheckBox("Auto Scale to Frame Min/Max")
        self.raw_auto_cb.setChecked(True)
        intensity_lay.addWidget(self.raw_auto_cb)

        lbl_layout = QHBoxLayout()
        self.raw_min_lbl = QLabel("Min: 0")
        self.raw_max_lbl = QLabel("Max: 1023")
        self.raw_max_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_layout.addWidget(self.raw_min_lbl)
        lbl_layout.addWidget(self.raw_max_lbl)
        intensity_lay.addLayout(lbl_layout)

        self.raw_min_sl = QSlider(Qt.Horizontal)
        self.raw_min_sl.setRange(0, 1023)
        self.raw_min_sl.setValue(0)
        self.raw_min_sl.setEnabled(False)
        
        self.raw_max_sl = QSlider(Qt.Horizontal)
        self.raw_max_sl.setRange(0, 1023)
        self.raw_max_sl.setValue(1023)
        self.raw_max_sl.setEnabled(False)

        intensity_lay.addWidget(self.raw_min_sl)
        intensity_lay.addWidget(self.raw_max_sl)

        self.raw_auto_cb.toggled.connect(self._on_raw_auto_toggled)
        self.raw_min_sl.valueChanged.connect(self._on_raw_slider_changed)
        self.raw_max_sl.valueChanged.connect(self._on_raw_slider_changed)

        # 3. ROI Settings Group
        roi_group = QGroupBox("ROI Plot Settings")
        roi_lay = QFormLayout(roi_group)
        
        self.roi_size_spin = QSpinBox()
        self.roi_size_spin.setRange(1, 1000)
        self.roi_size_spin.setValue(50)
        self.roi_size_spin.setSuffix(" px")
        
        roi_lay.addRow("Box Size:", self.roi_size_spin)
        hint_lbl = QLabel("(Click on preview image to center ROI)")
        hint_lbl.setStyleSheet("font-size: 10px; color: #888888;")
        roi_lay.addRow(hint_lbl)

        # Add all to settings widget
        preview_settings_lay.addWidget(cmos_group)
        preview_settings_lay.addWidget(intensity_group)
        preview_settings_lay.addWidget(roi_group)

        preview_action = QWidgetAction(self)
        preview_action.setDefaultWidget(preview_settings_widget)
        preview_menu.addAction(preview_action)

        self.raw_settings_btn = QPushButton(" ⚙ Settings ")
        self.raw_settings_btn.setMenu(preview_menu)
        self.raw_settings_btn.setStyleSheet("""
            QPushButton::menu-indicator { image: none; }
            QPushButton { background-color: #2d2d30; border: 1px solid #3f3f46; padding: 5px 15px; }
        """)
        preview_top_bar.addWidget(self.raw_settings_btn)
        preview_layout.addLayout(preview_top_bar)
        # ------------------------------------------

        self.raw_preview = ClickableLabel("Waiting for trigger…")
        self.raw_preview.setMinimumSize(400, 400)
        self.raw_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.raw_preview.setAlignment(Qt.AlignCenter)
        self.raw_preview.setProperty("is_image", True)
        self.raw_preview.clicked.connect(self._on_preview_clicked)
        self.raw_preview.double_clicked.connect(self._toggle_maximize_preview)   

        preview_layout.addWidget(self.raw_preview, stretch=1)
        preview_group.setLayout(preview_layout)
        left_layout.addWidget(preview_group, stretch=3)

        self.cycle_group = QGroupBox("ROI Intensity vs Piezo Displacement")
        cycle_layout = QVBoxLayout()
        cycle_layout.setContentsMargins(10, 20, 10, 10)

        self.fig, self.ax = plt.subplots(figsize=(6, 3.5))
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(250)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cycle_layout.addWidget(self.canvas)
        self.cycle_group.setLayout(cycle_layout)
        left_layout.addWidget(self.cycle_group, stretch=2)

        self.main_splitter.addWidget(self.left_widget)

        # Right column: Dynamic maps with tabs
        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.map_tabs = QTabWidget()
        self.map_tabs.setDocumentMode(False)
        self.map_tabs.setTabPosition(QTabWidget.North)
        self.map_tabs.setMinimumSize(500, 500)

        self.vis_img = ScalableImageLabel("No Data")
        self.vis_img.setProperty("is_image", True)
        self.contrast_img = ScalableImageLabel("No Data")
        self.contrast_img.setProperty("is_image", True)
        self.phase_img = ScalableImageLabel("No Data")
        self.phase_img.setProperty("is_image", True)

        self.vis_img.double_clicked.connect(self._toggle_maximize_maps)
        self.contrast_img.double_clicked.connect(self._toggle_maximize_maps)
        self.phase_img.double_clicked.connect(self._toggle_maximize_maps)

        vis_page, self.vis_min_sl, self.vis_max_sl, self.vis_min_lbl, self.vis_max_lbl = (
            self._create_map_tab("Visibility", self.vis_img)
        )
        con_page, self.con_min_sl, self.con_max_sl, self.con_min_lbl, self.con_max_lbl = (
            self._create_map_tab("Contrast", self.contrast_img)
        )
        pha_page, self.pha_min_sl, self.pha_max_sl, self.pha_min_lbl, self.pha_max_lbl = (
            self._create_map_tab("Phase", self.phase_img)
        )

        self.map_tabs.addTab(vis_page, "Visibility")
        self.map_tabs.addTab(con_page, "Contrast")
        self.map_tabs.addTab(pha_page, "Phase")

        right_layout.addWidget(self.map_tabs)
        self.main_splitter.addWidget(self.right_widget)

        self.main_splitter.setSizes([600, 1200])

        self.statusBar().showMessage("Ready. Connect devices.")

    def _on_raw_auto_toggled(self, checked: bool):
        """Enable or disable manual slider control for raw image."""
        self.raw_min_sl.setEnabled(not checked)
        self.raw_max_sl.setEnabled(not checked)

    def _on_raw_slider_changed(self):
        """Handle raw preview slider updates and prevent crossing."""
        self.raw_min_sl.blockSignals(True)
        self.raw_max_sl.blockSignals(True)
        
        if self.raw_min_sl.value() >= self.raw_max_sl.value():
            self.raw_min_sl.setValue(max(0, self.raw_max_sl.value() - 1))
            
        self.raw_min_sl.blockSignals(False)
        self.raw_max_sl.blockSignals(False)

        self.raw_min_lbl.setText(f"Min: {self.raw_min_sl.value()}")
        self.raw_max_lbl.setText(f"Max: {self.raw_max_sl.value()}")

    def _validate_scan_params(self):
        """Ensure scan end voltage is greater than scan start."""
        scan_end = self.scan_end_spin.value()
        if scan_end <= self._DEFAULT_SCAN_V_START:
            self.scan_end_spin.setValue(self._DEFAULT_SCAN_V_START + 0.1)
            QMessageBox.warning(
                self, "Invalid Range",
                f"Fringe period must be greater than {self._DEFAULT_SCAN_V_START} V"
            )

    def _toggle_maximize_maps(self):
        """Toggle between maximized map view and split view."""
        if self.left_widget.isVisible():
            self.left_widget.hide()
            self.statusBar().showMessage("Map view maximized. Double-click to restore.")
        else:
            self.left_widget.show()
            self.statusBar().showMessage("Restored default layout.")

    def _toggle_maximize_preview(self):
        """Toggle between maximized raw preview view and split view."""
        if self.right_widget.isVisible():
            self.right_widget.hide()  # Hide the maps side
            self.cycle_group.hide()   # Hide the intensity plot
            self.statusBar().showMessage("Preview maximized. Double-click to restore.")
        else:
            self.right_widget.show()  # Show the maps side
            self.cycle_group.show()   # Show the intensity plot
            self.statusBar().showMessage("Restored default layout.")

    def _create_map_tab(self, map_type, image_label):
        """Create a tab with adjustable colormap scaling."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        top_bar = QHBoxLayout()
        top_bar.addStretch()
        
        settings_btn = QPushButton(f"Adjust {map_type} Scale")
        settings_btn.setCheckable(True)
        settings_btn.setMinimumHeight(30)
        settings_btn.setFixedWidth(200)
        settings_btn.setStyleSheet("background-color: #2d2d30; border: 1px solid #3f3f46;")
        
        top_bar.addWidget(settings_btn)
        layout.addLayout(top_bar)

        sliders_container = QWidget()
        sliders_layout = QVBoxLayout(sliders_container)
        sliders_layout.setContentsMargins(0, 0, 0, 0)
        
        min_sl, max_sl, min_lbl, max_lbl = self._create_dual_slider(
            sliders_layout, f"{map_type} Scale Adjustment"
        )
        sliders_container.setVisible(False)
        
        settings_btn.toggled.connect(sliders_container.setVisible)
        
        layout.addWidget(sliders_container)
        layout.addWidget(image_label, stretch=1)
        
        return page, min_sl, max_sl, min_lbl, max_lbl

    def _create_dual_slider(self, layout, title):
        """Create min/max slider pair for colormap adjustment."""
        group = QGroupBox(title)
        lyt = QVBoxLayout()
        lyt.setContentsMargins(10, 15, 10, 10)
        lyt.setSpacing(2)

        lbl_layout = QHBoxLayout()
        min_lbl = QLabel("Min: --")
        max_lbl = QLabel("Max: --")
        max_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_layout.addWidget(min_lbl)
        lbl_layout.addWidget(max_lbl)
        lyt.addLayout(lbl_layout)

        min_slider = QSlider(Qt.Horizontal)
        min_slider.setRange(0, 1000)
        min_slider.setValue(0)
        
        max_slider = QSlider(Qt.Horizontal)
        max_slider.setRange(0, 1000)
        max_slider.setValue(1000)

        min_slider.valueChanged.connect(self._update_image_scales)
        max_slider.valueChanged.connect(self._update_image_scales)

        lyt.addWidget(min_slider)
        lyt.addWidget(max_slider)
        
        group.setLayout(lyt)
        layout.addWidget(group)
        return min_slider, max_slider, min_lbl, max_lbl

    def get_scale_factors(self):
        """Extract current slider positions as scale factors."""
        return {
            'v_min_pct': self.vis_min_sl.value() / 1000.0,
            'v_max_pct': self.vis_max_sl.value() / 1000.0,
            'c_min_pct': self.con_min_sl.value() / 1000.0,
            'c_max_pct': self.con_max_sl.value() / 1000.0,
            'p_min_pct': self.pha_min_sl.value() / 1000.0,
            'p_max_pct': self.pha_max_sl.value() / 1000.0,
        }

    def _reset_sliders(self):
        """Reset all colormap sliders to full range."""
        for slider in [self.vis_min_sl, self.vis_max_sl, self.con_min_sl, 
                      self.con_max_sl, self.pha_min_sl, self.pha_max_sl]:
            slider.blockSignals(True)

        self.vis_min_sl.setValue(0)
        self.con_min_sl.setValue(0)
        self.pha_min_sl.setValue(0)
        self.vis_max_sl.setValue(1000)
        self.con_max_sl.setValue(1000)
        self.pha_max_sl.setValue(1000)

        for slider in [self.vis_min_sl, self.vis_max_sl, self.con_min_sl, 
                      self.con_max_sl, self.pha_min_sl, self.pha_max_sl]:
            slider.blockSignals(False)

    def _update_labels(self):
        """Update slider labels to reflect current data ranges."""
        if not self.camera or not hasattr(self.camera, 'data_limits'):
            return
            
        sf = self.get_scale_factors()
        dl = self.camera.data_limits
        
        v_min = dl['v_min'] + sf['v_min_pct'] * (dl['v_max'] - dl['v_min'])
        v_max = dl['v_min'] + sf['v_max_pct'] * (dl['v_max'] - dl['v_min'])
        c_min = dl['c_min'] + sf['c_min_pct'] * (dl['c_max'] - dl['c_min'])
        c_max = dl['c_min'] + sf['c_max_pct'] * (dl['c_max'] - dl['c_min'])
        p_min = dl['p_min'] + sf['p_min_pct'] * (dl['p_max'] - dl['p_min'])
        p_max = dl['p_min'] + sf['p_max_pct'] * (dl['p_max'] - dl['p_min'])

        self.vis_min_lbl.setText(f"Min: {v_min:.3f}")
        self.vis_max_lbl.setText(f"Max: {v_max:.3f}")
        self.con_min_lbl.setText(f"Min: {c_min:.2f}")
        self.con_max_lbl.setText(f"Max: {c_max:.2f}")
        self.pha_min_lbl.setText(f"Min: {p_min:.2f}")
        self.pha_max_lbl.setText(f"Max: {p_max:.2f}")

    def _update_image_scales(self):
        """Recalculate and redraw images when sliders change."""
        # Ensure min < max for each pair
        for min_sl, max_sl in [(self.vis_min_sl, self.vis_max_sl),
                               (self.con_min_sl, self.con_max_sl),
                               (self.pha_min_sl, self.pha_max_sl)]:
            min_sl.blockSignals(True)
            max_sl.blockSignals(True)
            if min_sl.value() >= max_sl.value():
                min_sl.setValue(max(0, max_sl.value() - 1))
            min_sl.blockSignals(False)
            max_sl.blockSignals(False)

        sf = self.get_scale_factors()

        # Update worker scale factors if running
        if self.acq_worker is not None:
            self.acq_worker.scale_factors = sf
        if self.live_proc_worker is not None:
            self.live_proc_worker.scale_factors = sf

        # Re-render maps if data exists
        if self.camera and getattr(self.camera, 'last_visibility', None) is not None:
            dl = self.camera.data_limits
            
            v_min = dl['v_min'] + sf['v_min_pct'] * (dl['v_max'] - dl['v_min'])
            v_max = dl['v_min'] + sf['v_max_pct'] * (dl['v_max'] - dl['v_min'])
            c_min = dl['c_min'] + sf['c_min_pct'] * (dl['c_max'] - dl['c_min'])
            c_max = dl['c_min'] + sf['c_max_pct'] * (dl['c_max'] - dl['c_min'])
            p_min = dl['p_min'] + sf['p_min_pct'] * (dl['p_max'] - dl['p_min'])
            p_max = dl['p_min'] + sf['p_max_pct'] * (dl['p_max'] - dl['p_min'])

            vis_img, con_img, pha_img = self.camera.render_colormaps(
                v_min, v_max, c_min, c_max, p_min, p_max
            )
            if vis_img is not None:
                self._display_maps(vis_img, con_img, pha_img)

    def _load_settings(self):
        """Load acquisition parameters from JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", self.base_dir, "JSON Files (*.json);;All Files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                settings = json.load(f)

            if "exposure_ms" in settings:
                self.exposure_spin.setValue(settings["exposure_ms"])
            if "gain_db" in settings:
                self.gain_spin.setValue(settings["gain_db"])
            if "n_frames" in settings:
                self.frames_spin.setValue(settings["n_frames"])
            if "fringe_period_v" in settings:
                self.scan_end_spin.setValue(settings["fringe_period_v"])
            if "settling_ms" in settings:
                self.settling_spin.setValue(settings["settling_ms"])
            if "roi_center_x" in settings:
                self.roi_x = settings["roi_center_x"]
            if "roi_center_y" in settings:
                self.roi_y = settings["roi_center_y"]
            if "roi_box_size" in settings:
                self.roi_size_spin.setValue(settings["roi_box_size"])
            if "moving_average" in settings:
                self.ma_checkbox.setChecked(settings["moving_average"])
            if "ma_kernel_size" in settings:
                self.ma_size_spin.setValue(settings["ma_kernel_size"])

            self.statusBar().showMessage(f"Loaded parameters from {os.path.basename(file_path)}")

        except json.JSONDecodeError:
            QMessageBox.critical(self, "Load Error", "Invalid JSON file.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load settings:\n{str(e)}")

    def _save_data(self):
        """Save acquisition results and settings to a timestamped folder."""
        date_str = datetime.date.today().strftime("%d_%m_%Y")
        file_path, _ = QFileDialog.getSaveFileName(self)
        
        if not file_path:
            return

        base_dir = os.path.dirname(file_path)
        user_name = os.path.basename(file_path)
        
        if user_name.endswith(date_str):
            folder_name = user_name
        else:
            folder_name = f"{user_name}_Acquisition_{date_str}"
            
        full_path = os.path.join(base_dir, folder_name)

        # Auto-increment folder name if exists
        counter = 1
        final_path = full_path
        while os.path.exists(final_path):
            final_path = f"{full_path}_{counter}"
            counter += 1

        try:
            os.makedirs(final_path, exist_ok=True)

            # Save acquisition settings
            settings = {
                "exposure_ms": self.exposure_spin.value(),
                "gain_db": self.gain_spin.value(),
                "n_frames": self.frames_spin.value(),
                "fringe_period_v": self.scan_end_spin.value(),
                "settling_ms": self.settling_spin.value(),
                "roi_center_x": self.roi_x,
                "roi_center_y": self.roi_y,
                "roi_box_size": self.roi_size_spin.value(),
                "processing_time_s": getattr(self.acq_worker, "last_proc_time", 0.0),
                "moving_average": self.ma_checkbox.isChecked(),
                "ma_kernel_size": self.ma_size_spin.value(),
                "scale_factors_pct": self.get_scale_factors()
            }
            with open(os.path.join(final_path, "settings.json"), "w") as f:
                json.dump(settings, f, indent=4)

            # Save processed maps
            save_map = {
                "visibility_map.png": self.vis_img,
                "contrast_map.png": self.contrast_img,
                "phase_map.png": self.phase_img,
            }
            for filename, label in save_map.items():
                pm = label.original_pixmap() if hasattr(label, 'original_pixmap') else label.pixmap()
                if pm and not pm.isNull():
                    pm.save(os.path.join(final_path, filename), "PNG")

            # Save raw preview
            pm = self.raw_preview.pixmap()
            if pm and not pm.isNull():
                pm.save(os.path.join(final_path, "last_raw_frame.png"), "PNG")

            # Save intensity plot
            self.fig.savefig(os.path.join(final_path, "intensity_plot.png"))

            self.statusBar().showMessage(f"Data saved to {os.path.basename(final_path)}")
            QMessageBox.information(self, "Data Saved", f"Saved to:\n{final_path}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")
    
    def _on_exposure_changed(self, val_ms: int):
        """Update camera exposure in real-time."""
        if self.camera and self.camera.camera:
            try:
                self.camera.camera.exposure_time_us = val_ms * 1000
                self.camera.camera.image_poll_timeout_ms = val_ms + 100
            except Exception as e:
                self.statusBar().showMessage(f"Warning: Failed to update exposure ({e})")

    def _on_gain_changed(self, val_db: int):
        """Update camera gain in real-time."""
        if self.camera and self.camera.camera:
            try:
                self.camera.camera.gain = val_db * 10
            except Exception as e:
                self.statusBar().showMessage(f"Warning: Failed to update gain ({e})")

    def _on_preview_clicked(self, label_x: int, label_y: int):
        """Set ROI center by clicking on the preview image."""
        if not self.camera or not self.raw_preview.pixmap():
            return

        pm = self.raw_preview.pixmap()
        pm_w, pm_h = pm.width(), pm.height()
        label_w, label_h = self.raw_preview.width(), self.raw_preview.height()

        # Calculate offset for centered image
        offset_x = (label_w - pm_w) / 2.0
        offset_y = (label_h - pm_h) / 2.0

        pixmap_x = label_x - offset_x
        pixmap_y = label_y - offset_y

        # Check if click is within image bounds
        if pixmap_x < 0 or pixmap_x > pm_w or pixmap_y < 0 or pixmap_y > pm_h:
            return

        # Map pixmap coordinates to camera coordinates
        cam_w = self.camera.image_width
        cam_h = self.camera.image_height

        cam_x = int((pixmap_x / pm_w) * cam_w)
        cam_y = int((pixmap_y / pm_h) * cam_h)

        # Store invisibly instead of spinboxes
        self.roi_x = cam_x
        self.roi_y = cam_y

    def _connect_hardware(self):
        """Initialize and connect to camera and piezo controller."""
        if self._acquisition_in_progress:
            QMessageBox.warning(self, "Busy", "Wait for current operation to complete.")
            return
            
        self.statusBar().showMessage("Connecting to hardware…")

        try:
            self.piezo = PiezoController()
            self.camera = CameraController()
            
            p_ok, p_msg = self.piezo.connect()
            c_ok = self.camera.connect()
        except Exception as exc:
            QMessageBox.critical(self, "Hardware Error", f"Connection error:\n{exc}")
            self.statusBar().showMessage("Hardware connection failed.")
            return

        if p_ok and c_ok:
            w = self.camera.image_width
            h = self.camera.image_height
            
            # Default ROI to center of camera
            self.roi_x = w // 2
            self.roi_y = h // 2
            
            self.has_strain_gauge = self.piezo.has_strain_gauge

            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)

            self.statusBar().showMessage("Hardware connected successfully")
            self._apply_theme()
            QMessageBox.information(self, "Success", "All hardware connected.")
        else:
            errors = []
            if not p_ok: 
                errors.append(f"Piezo: {p_msg}")
            if not c_ok: 
                errors.append("Camera: failed to initialize")
            QMessageBox.critical(self, "Connection Error", "Failed to connect:\n" + "\n".join(errors))
            self.statusBar().showMessage("Hardware connection failed.")

    def _disconnect_hardware(self):
        """Safely disconnect all hardware."""
        reply = QMessageBox.question(
            self, "Confirm Disconnect", "Disconnect all hardware?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Stop all workers before disconnecting
        self._stop_live_worker()
        self._stop_live_proc_worker()
        self._stop_worker()

        if self.camera: 
            self.camera.disconnect()
        if self.piezo: 
            self.piezo.disconnect()

        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.live_proc_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.statusBar().showMessage("Hardware disconnected.")

    def _run_acquisition(self):
        """Start a single-shot acquisition sequence."""
        if self._acquisition_in_progress:
            return
            
        self._acquisition_in_progress = True
        self.start_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.live_proc_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.start_btn.setText(" Acquiring…")

        self.displacements_intensities.clear()
        self.ax.clear()

        # Configure camera
        exp_ms = self.exposure_spin.value()
        self.camera.camera.exposure_time_us = exp_ms * 1000
        self.camera.camera.gain = self.gain_spin.value() * 10
        self.camera.camera.image_poll_timeout_ms = exp_ms + 100

        period_v = self.scan_end_spin.value() - self._DEFAULT_SCAN_V_START
        self._reset_sliders()

        self.acq_worker = SingleAcquisitionWorker(
            camera_ctrl=self.camera,
            piezo_ctrl=self.piezo,
            n_frames=self.frames_spin.value(),
            scan_v_start=self._DEFAULT_SCAN_V_START,
            period_v=period_v,
            settling_time=self.settling_spin.value() / 1000.0,
            scale_factors=self.get_scale_factors()
        )
        self.acq_worker.progress_signal.connect(self.statusBar().showMessage)
        self.acq_worker.error_signal.connect(self._on_error)
        self.acq_worker.frame_acquired_signal.connect(self._update_preview_and_plot)
        self.acq_worker.finished_signal.connect(self._display_maps)
        self.acq_worker.finished.connect(self._on_acquisition_complete)
        self.acq_worker.start()

    def _toggle_live_processing(self):
        """Toggle continuous live processing mode."""
        if self.live_proc_worker and self.live_proc_worker.isRunning():
            self._stop_live_proc_worker()
            self.live_proc_btn.setText(" Live")
            self.live_proc_btn.setIcon(self.play_icon)
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.statusBar().showMessage("Live processing stopped.")
        else:
            self.start_btn.setEnabled(False)
            self.live_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.live_proc_btn.setText(" Stop")
            self.live_proc_btn.setIcon(self.stop_icon)
            self.statusBar().showMessage("Live processing running…")

            self.displacements_intensities.clear()
            self.ax.clear()

            exp_ms = self.exposure_spin.value()
            self.camera.camera.exposure_time_us = exp_ms * 1000
            self.camera.camera.gain = self.gain_spin.value() * 10
            self.camera.camera.image_poll_timeout_ms = exp_ms + 100

            self.camera.set_single_frame_mode()
            period_v = self.scan_end_spin.value() - self._DEFAULT_SCAN_V_START
            self._reset_sliders()

            self.live_proc_worker = LiveProcessingWorker(
                camera_ctrl=self.camera,
                piezo_ctrl=self.piezo,
                n_frames=self.frames_spin.value(),
                scan_v_start=self._DEFAULT_SCAN_V_START,
                period_v=period_v,
                settling_time=self.settling_spin.value() / 1000.0,
                scale_factors=self.get_scale_factors()
            )
            self.live_proc_worker.frame_acquired_signal.connect(self._update_preview_and_plot)
            self.live_proc_worker.maps_ready_signal.connect(self._display_maps)
            self.live_proc_worker.error_signal.connect(self._on_error)
            self.live_proc_worker.start()

    def _toggle_live_feed(self):
        """Toggle raw camera feed mode."""
        if self.live_worker and self.live_worker.isRunning():
            self._stop_live_worker()
            self.live_btn.setText(" Raw Feed")
            self.live_btn.setIcon(self.play_icon)
            self.start_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)
            self.statusBar().showMessage("Raw feed stopped.")
        else:
            self.start_btn.setEnabled(False)
            self.live_proc_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.live_btn.setText(" Stop")
            self.live_btn.setIcon(self.stop_icon)
            self.statusBar().showMessage("Raw feed running…")

            exp_ms = self.exposure_spin.value()
            self.camera.camera.exposure_time_us = exp_ms * 1000
            self.camera.camera.gain = self.gain_spin.value() * 10
            self.camera.camera.image_poll_timeout_ms = exp_ms + 100

            self.camera.set_continuous_mode()

            self.live_worker = LiveFeedWorker(self.camera)
            self.live_worker.frame_ready_signal.connect(self._update_live_preview)
            self.live_worker.error_signal.connect(self._on_error)
            self.live_worker.start()

    def _on_error(self, msg: str):
        """Handle errors from worker threads."""
        self._stop_live_worker()
        self._stop_live_proc_worker()

        self.live_btn.setText(" Raw Feed")
        self.live_btn.setIcon(self.play_icon)
        self.live_proc_btn.setText(" Live")
        self.live_proc_btn.setIcon(self.play_icon)
        self.start_btn.setText(" Single")

        self.start_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.live_proc_btn.setEnabled(True)
        
        self._acquisition_in_progress = False
        
        QMessageBox.warning(self, "Hardware Error", msg)

    def _reset_system(self):
        """Reset all acquisition state and return piezo to zero."""
        self._stop_live_worker()
        self._stop_live_proc_worker()
        self._stop_worker()

        if self.piezo: 
            self.piezo.set_voltage(0.0)

        self.displacements_intensities.clear()
        self.ax.clear()
        self.canvas.draw()

        self.raw_preview.setPixmap(QPixmap())
        self.raw_preview.setText("System reset")

        for widget in (self.vis_img, self.contrast_img, self.phase_img):
            widget.setPixmap(QPixmap())
            widget.setText("No Data")

        self.statusBar().showMessage("System reset.")
        
        self._acquisition_in_progress = False

        if self.piezo and self.camera:
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
            self.start_btn.setText(" Single")
            self.live_btn.setText(" Raw Feed")
            self.live_btn.setIcon(self.play_icon)
            self.live_proc_btn.setText(" Live")
            self.live_proc_btn.setIcon(self.play_icon)

    def _stop_worker(self):
        """Stop the single acquisition worker."""
        if self.acq_worker and self.acq_worker.isRunning():
            self.acq_worker.is_running = False
            self.acq_worker.wait()

    def _stop_live_worker(self):
        """Stop the live feed worker and restore single-frame mode."""
        if self.live_worker and self.live_worker.isRunning():
            self.live_worker.is_running = False
            self.live_worker.wait()
            self.live_worker = None
            
        # Only change camera mode if no other workers are running
        if self.camera and not (self.live_proc_worker and self.live_proc_worker.isRunning()):
            self.camera.set_single_frame_mode()

    def _stop_live_proc_worker(self):
        """Stop the live processing worker."""
        if self.live_proc_worker and self.live_proc_worker.isRunning():
            self.live_proc_worker.is_running = False
            self.live_proc_worker.wait()
            self.live_proc_worker = None

    def _on_ma_toggled(self, checked: bool):
        """Enable/disable moving average filter."""
        self.ma_size_spin.setEnabled(checked)
        if self.camera:
            self.camera.use_moving_average = checked

    def _on_ma_size_changed(self, val: int):
        """Ensure moving average kernel size is odd."""
        if val % 2 == 0:
            val += 1
            self.ma_size_spin.blockSignals(True)
            self.ma_size_spin.setValue(val)
            self.ma_size_spin.blockSignals(False)
        if self.camera:
            self.camera.ma_kernel_size = val

    def _update_live_preview(self, gray_img: np.ndarray):
        """Update the raw frame preview with ROI overlay."""
        
        # Check if we should use raw auto scaling or calculate custom thresholds
        if self.raw_auto_cb.isChecked():
            norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            # Use raw absolute values (0 - 1023)
            v_min = self.raw_min_sl.value()
            v_max = self.raw_max_sl.value()
            
            if v_min >= v_max:
                v_max = v_min + 1e-5

            clipped = np.clip(gray_img, v_min, v_max)
            scaled = (clipped - v_min) / (v_max - v_min) * 255.0
            norm = scaled.astype(np.uint8)

        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

        # Draw ROI rectangle using hidden internal coordinates
        x = self.roi_x
        y = self.roi_y
        s = self.roi_size_spin.value() // 2
        cv2.rectangle(rgb, (x - s, y - s), (x + s, y + s), (255, 0, 0), 4)

        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self.raw_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.raw_preview.setPixmap(scaled)

    def _update_preview_and_plot(self, gray_img: np.ndarray, value: float, idx: int):
        """Update preview and intensity plot during acquisition."""
        self._update_live_preview(gray_img)

        # Extract ROI intensity using hidden internal coordinates
        x = self.roi_x
        y = self.roi_y
        s = self.roi_size_spin.value() // 2
        h_img, w_img = gray_img.shape

        y_min, y_max = max(0, y - s), min(h_img, y + s + 1)
        x_min, x_max = max(0, x - s), min(w_img, x + s + 1)

        roi_data = gray_img[y_min:y_max, x_min:x_max]
        mean_intensity = float(np.mean(roi_data))

        n_frames = max(1, self.frames_spin.value())
        step_idx = idx % n_frames 
        
        self.displacements_intensities[step_idx] = (value, mean_intensity)

        if step_idx == 0 or not hasattr(self, '_reference_value'):
            self._reference_value = value
            
        # Calculate the relative position/voltage
        relative_value = value - self._reference_value

        # Store the relative value instead of the absolute 'value'
        self.displacements_intensities[step_idx] = (relative_value, mean_intensity)

        # Plot sorted by displacement/voltage
        sorted_items = sorted(self.displacements_intensities.values(), key=lambda item: item[0])
        sorted_values = [item[0] for item in sorted_items]
        sorted_intensities = [item[1] for item in sorted_items]

        self.ax.clear()
        self.ax.grid(True, color="#2d2d30", linestyle="--", linewidth=0.5, zorder=0)
        self.ax.plot(
            sorted_values, sorted_intensities,
            color="#0078d4",
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1,
            zorder=2
        )
        
        x_label = "Piezo Displacement (µm)" if self.has_strain_gauge else "Piezo Voltage (V)"
        
        self.ax.set_xlabel(x_label, fontsize=10, color="#d4d4d4", fontweight="bold")
        self.ax.set_ylabel("ROI Mean Intensity", fontsize=10, color="#d4d4d4", fontweight="bold")
        self.ax.tick_params(colors="#d4d4d4", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#3f3f46")
        self.canvas.draw()
        
    def _display_maps(self, vis: np.ndarray, contrast: np.ndarray, phase: np.ndarray):
        """Display the processed visibility, contrast, and phase maps."""
        self.vis_img.setPixmap(self._cv_to_pixmap(vis))
        self.contrast_img.setPixmap(self._cv_to_pixmap(contrast))
        self.phase_img.setPixmap(self._cv_to_pixmap(phase))
        self._update_labels()

    def _on_acquisition_complete(self):
        """Handle completion of single acquisition."""
        proc_time = getattr(self.acq_worker, "last_proc_time", 0.0)
        msg = f"Acquisition complete. Processing time: {proc_time:.4f} s"

        self.statusBar().showMessage(msg)
        
        self.start_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.live_proc_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.start_btn.setText(" Single")
        
        self._acquisition_in_progress = False

    @staticmethod
    def _cv_to_pixmap(cv_img: np.ndarray) -> QPixmap:
        """Convert OpenCV RGB image to Qt QPixmap."""
        h, w, ch = cv_img.shape
        qimg = QImage(cv_img.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _apply_theme(self):
        """Apply dark theme stylesheet."""
        bg, fg, border, img_bg = "#1e1e1e", "#d4d4d4", "#3f3f46", "#121212"
        btn_bg, bar_bg = "#333337", "#2d2d30"
        accent = "#0078d4"

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {bg}; 
                color: {fg};
                font-family: Segoe UI, Arial; 
                font-size: 12px;
            }}
            QToolBar {{
                background-color: {bar_bg};
                border-bottom: 1px solid {border}; 
                padding: 8px;
            }}
            QGroupBox {{
                border: 1px solid {border}; 
                margin-top: 15px;
                font-weight: bold; 
                border-radius: 4px; 
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; 
                left: 12px; 
                padding: 0 5px;
            }}
            QPushButton {{
                background-color: {btn_bg}; 
                color: white; 
                padding: 6px 12px;
                border: 1px solid {border}; 
                border-radius: 4px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #3e3e42; 
            }}
            QPushButton:pressed {{
                background-color: #2d2d30;
            }}
            QPushButton[isAction="true"] {{
                background-color: {accent}; 
                border: 1px solid #005a9e;
            }}
            QPushButton[isAction="true"]:hover {{
                background-color: #005a9e; 
                border: 1px solid #004578;
            }}
            QPushButton[isAction="true"]:disabled {{
                background-color: #2d2d30; 
                color: #888; 
                border: 1px solid {border};
            }}
            QPushButton#iconBtn {{
                padding: 6px;
                border-radius: 4px;
            }}
            QToolButton {{
                background-color: {btn_bg}; 
                color: white; 
                padding: 6px 12px;
                border: 1px solid {border}; 
                border-radius: 4px;
                font-weight: 600;
            }}
            QToolButton:hover {{
                background-color: #3e3e42; 
            }}
            QToolButton::menu-indicator {{ 
                image: none; 
            }}
            QMenu {{
                background-color: {bar_bg};
                color: {fg};
                border: 1px solid {border};
            }}
            #dropdownMenu {{
                background-color: {bg};
                border-radius: 4px;
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {img_bg}; 
                color: {fg};
                border: 1px solid {border}; 
                padding: 5px;
                border-radius: 3px;
            }}
            QLabel[is_image="true"] {{
                background-color: {img_bg};
                border: 2px dashed {border}; 
                color: #666;
            }}
            QStatusBar {{
                background-color: {bar_bg}; 
                color: {fg};
                border-top: 1px solid {border};
            }}
            QCheckBox {{
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px; 
                height: 18px;
                border: 1px solid {border};
                background-color: {img_bg};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {accent};
                border-color: {accent};
            }}
            QTabWidget::pane {{
                border: 1px solid {border};
                background-color: {bg};
                border-radius: 6px;
                top: -1px;
            }}
            QTabBar::tab {{
                background-color: {bar_bg};
                color: {fg};
                padding: 14px 32px;
                margin-right: 3px;
                border: 1px solid {border};
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                font-size: 13px;
                min-width: 100px;
            }}
            QTabBar::tab:selected {{
                background-color: {accent};
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: #3a3a3e;
            }}
            QSlider::groove:horizontal {{
                border: 1px solid {border};
                height: 6px;
                background: {img_bg};
                margin: 2px 0;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {accent};
                border: 1px solid {accent};
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
        """)

        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(img_bg)
        self.ax.tick_params(colors=fg)
        for spine in self.ax.spines.values():
            spine.set_color(border)
        self.canvas.draw()

    def closeEvent(self, event):
        """Handle application close event with cleanup."""
        if self.camera is None and self.piezo is None:
            event.accept()
            return

        reply = QMessageBox.question(
            self, "Confirm Exit",
            "Exit and disconnect all hardware?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            event.ignore()
            return

        self.statusBar().showMessage("Shutting down…")
        
        # Stop all workers
        self._stop_live_worker()
        self._stop_live_proc_worker()
        self._stop_worker()

        # Disconnect hardware
        try:
            if self.camera: 
                self.camera.disconnect()
            if self.piezo: 
                self.piezo.disconnect()
        except Exception as exc:
            print(f"Error during shutdown: {exc}")

        # Clean up matplotlib resources
        try:
            plt.close(self.fig)
        except Exception:
            pass

        event.accept()


if __name__ == "__main__":
    # Set Windows taskbar icon (Windows-only, fails gracefully on other platforms)
    try:
        if sys.platform == 'win32':
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("nano_qiup.app.2_0")
    except (AttributeError, OSError):
        pass

    app = QApplication(sys.argv)
    window = QIUP_APP()
    
    # Set app icon if logo exists
    if os.path.exists(window.logo_path):
        app.setWindowIcon(QIcon(window.logo_path))
        
    window.show()
    sys.exit(app.exec_())