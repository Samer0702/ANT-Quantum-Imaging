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
    QToolBar, QAction, QSizePolicy, QMessageBox, QCheckBox, QTabWidget, QFileDialog,
    QSlider, QSplitter, QMenu, QWidgetAction, QToolButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon

from piezo_control_closed import PiezoController
from camera_control import CameraController 
from ui_components import ClickableLabel, ScalableImageLabel
from acquisition_workers import SingleAcquisitionWorker, LiveFeedWorker, LiveProcessingWorker

class QIUP_APP(QMainWindow):

    _DEFAULT_START_UM = 0.0
    _DEFAULT_TOTAL_UM = 0.5
    _DEFAULT_SETTLING_MS = 10

    def __init__(self):
        super().__init__()
        # Adjusted pathing as discussed previously
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.logo_path = os.path.join(self.base_dir, "logo", "Logo_ANT.png")

        self.setWindowTitle("NANO: QIUP Dashboard")
        self.setGeometry(50, 50, 1800, 1000)
        self.setWindowIcon(QIcon(self.logo_path))

        self.piezo: PiezoController | None = None
        self.camera: CameraController | None = None

        self.acq_worker: SingleAcquisitionWorker | None = None
        self.live_worker: LiveFeedWorker | None = None
        self.live_proc_worker: LiveProcessingWorker | None = None

        self.positions_intensities: dict[int, tuple[float, float]] = {}

        self._setup_ui()
        self._apply_theme()

    def _setup_ui(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("QToolBar { spacing: 8px; padding: 5px; }")
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setToolTip("Connect to the Thorlabs camera and Piezo controller hardware.")
        self.connect_btn.clicked.connect(self._connect_hardware)
        toolbar.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setToolTip("Safely disconnect and turn off all hardware.")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(self._disconnect_hardware)
        toolbar.addWidget(self.disconnect_btn)

        toolbar.addSeparator()
        
        self.start_btn = QPushButton("Single ")
        self.start_btn.setToolTip("Run a single, fixed-frame phase-stepping acquisition sequence.")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._run_single_acquisition)
        toolbar.addWidget(self.start_btn)

        self.live_proc_btn = QPushButton("Live")
        self.live_proc_btn.setToolTip("Start continuous, real-time quantum phase-stepping and map reconstruction.")
        self.live_proc_btn.setEnabled(False)
        self.live_proc_btn.clicked.connect(self._toggle_live_processing)
        toolbar.addWidget(self.live_proc_btn)

        self.live_btn = QPushButton("Raw Feed")
        self.live_btn.setToolTip("View the live, raw CMOS camera feed (no piezo scanning).")
        self.live_btn.setEnabled(False)
        self.live_btn.clicked.connect(self._toggle_live_feed)
        toolbar.addWidget(self.live_btn)

        toolbar.addSeparator()

        # 5. Expandable Scan Parameters
        scan_menu = QMenu(self)
        scan_widget = QWidget()
        scan_widget.setObjectName("dropdownMenu")
        scan_lay = QFormLayout(scan_widget)
        
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(3, 1000)
        self.frames_spin.setValue(8)
        self.frames_spin.valueChanged.connect(self._update_calc_step)
        
        self.total_range_spin = QDoubleSpinBox()
        self.total_range_spin.setRange(0.01, PiezoController.MAX_TRAVEL_UM)
        self.total_range_spin.setDecimals(3)
        self.total_range_spin.setSingleStep(0.1)
        self.total_range_spin.setValue(self._DEFAULT_TOTAL_UM)
        self.total_range_spin.setSuffix(" µm")
        self.total_range_spin.valueChanged.connect(self._update_calc_step)

        self.calc_step_lbl = QLabel()
        self.calc_step_lbl.setStyleSheet("color: #0078d4; font-weight: bold;")
        self._update_calc_step()
        
        self.settling_spin = QSpinBox()
        self.settling_spin.setRange(0, 1000)
        self.settling_spin.setValue(self._DEFAULT_SETTLING_MS)
        self.settling_spin.setSuffix(" ms")
        self.reset_btn = QPushButton("Reset System")
        self.reset_btn.setMinimumHeight(30)
        self.reset_btn.clicked.connect(self._reset_system)
        
        scan_lay.addRow("Frames (N):", self.frames_spin)
        scan_lay.addRow("Total Scan Range:", self.total_range_spin)
        scan_lay.addRow("Auto Step Size:", self.calc_step_lbl)
        scan_lay.addRow("Settling time:", self.settling_spin)
        scan_lay.addRow(self.reset_btn)
        scan_action = QWidgetAction(self)
        scan_action.setDefaultWidget(scan_widget)
        scan_menu.addAction(scan_action)

        self.scan_btn = QToolButton()
        self.scan_btn.setText("Scan Params")
        self.scan_btn.setToolTip("Expand to set the number of Frames, Total Range, and hardware Settling Time.")
        self.scan_btn.setMenu(scan_menu)
        self.scan_btn.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(self.scan_btn)

        # 3. Expandable CMOS Settings
        cmos_menu = QMenu(self)
        cmos_widget = QWidget()
        cmos_widget.setObjectName("dropdownMenu")
        cmos_lay = QFormLayout(cmos_widget)
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
        cmos_action = QWidgetAction(self)
        cmos_action.setDefaultWidget(cmos_widget)
        cmos_menu.addAction(cmos_action)

        self.cmos_btn = QToolButton()
        self.cmos_btn.setText("CMOS Settings")
        self.cmos_btn.setToolTip("Expand to adjust Camera Exposure, Gain, and processing filters.")
        self.cmos_btn.setMenu(cmos_menu)
        self.cmos_btn.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(self.cmos_btn)

        # 4. Expandable ROI Settings
        roi_menu = QMenu(self)
        roi_widget = QWidget()
        roi_widget.setObjectName("dropdownMenu")
        roi_lay = QFormLayout(roi_widget)
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 4000)
        self.roi_x_spin.setValue(0)
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 4000)
        self.roi_y_spin.setValue(0)
        self.roi_size_spin = QSpinBox()
        self.roi_size_spin.setRange(1, 1000)
        self.roi_size_spin.setValue(50)
        self.roi_size_spin.setSuffix(" px")
        
        roi_lay.addRow("Center X:", self.roi_x_spin)
        roi_lay.addRow("Center Y:", self.roi_y_spin)
        roi_lay.addRow("Box Size:", self.roi_size_spin)
        hint_lbl = QLabel("(Click on preview to select ROI)")
        hint_lbl.setStyleSheet("font-size: 10px; color: #888888;")
        roi_lay.addRow(hint_lbl)
        roi_action = QWidgetAction(self)
        roi_action.setDefaultWidget(roi_widget)
        roi_menu.addAction(roi_action)

        self.roi_btn = QToolButton()
        self.roi_btn.setText("Plot ROI")
        self.roi_btn.setToolTip("Expand to configure the Region of Interest used for the intensity plot.")
        self.roi_btn.setMenu(roi_menu)
        self.roi_btn.setPopupMode(QToolButton.InstantPopup)
        toolbar.addWidget(self.roi_btn)

        toolbar.addSeparator()

        # 6. Separate Load / Save Buttons
        self.load_btn = QPushButton("Load")
        self.load_btn.setToolTip("Load previous settings.")
        self.load_btn.clicked.connect(self._load_settings)
        toolbar.addWidget(self.load_btn)

        self.save_btn = QPushButton("Save")
        self.save_btn.setToolTip("Save currently acquired data.")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_data)
        toolbar.addWidget(self.save_btn)

        # ========== CENTRAL LAYOUT ==========
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(5, 5, 5, 5)

        self.main_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.main_splitter)

        # ========== LEFT COLUMN: Preview + Graph ==========
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        preview_group = QGroupBox("Current Frame Preview")
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(10, 20, 10, 10)

        self.raw_preview = ClickableLabel("Waiting for trigger…")
        self.raw_preview.setMinimumSize(400, 400)
        self.raw_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.raw_preview.setAlignment(Qt.AlignCenter)
        self.raw_preview.setProperty("is_image", True)
        self.raw_preview.clicked.connect(self._on_preview_clicked)
        preview_layout.addWidget(self.raw_preview, stretch=1)
        preview_group.setLayout(preview_layout)
        left_layout.addWidget(preview_group, stretch=3)

        cycle_group = QGroupBox("ROI Intensity vs Piezo Position")
        cycle_layout = QVBoxLayout()
        cycle_layout.setContentsMargins(10, 20, 10, 10)

        self.fig, self.ax = plt.subplots(figsize=(6, 3.5))
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(250)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cycle_layout.addWidget(self.canvas)
        cycle_group.setLayout(cycle_layout)
        left_layout.addWidget(cycle_group, stretch=2)

        self.main_splitter.addWidget(self.left_widget)

        # ========== RIGHT COLUMN: Dynamic Maps & Tabs ==========
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

        vis_page, self.vis_min_sl, self.vis_max_sl, self.vis_min_lbl, self.vis_max_lbl = self._create_map_tab("Visibility", self.vis_img)
        con_page, self.con_min_sl, self.con_max_sl, self.con_min_lbl, self.con_max_lbl = self._create_map_tab("Contrast", self.contrast_img)
        pha_page, self.pha_min_sl, self.pha_max_sl, self.pha_min_lbl, self.pha_max_lbl = self._create_map_tab("Phase", self.phase_img)

        self.map_tabs.addTab(vis_page, "Visibility")
        self.map_tabs.addTab(con_page, "Contrast")
        self.map_tabs.addTab(pha_page, "Phase")

        right_layout.addWidget(self.map_tabs)
        self.main_splitter.addWidget(self.right_widget)

        self.main_splitter.setSizes([600, 1200])

        self.statusBar().showMessage("Ready. Connect devices.")

    def _update_calc_step(self):
        """Calculates and updates the step size label dynamically."""
        total = self.total_range_spin.value()
        frames = self.frames_spin.value()
        if frames > 0:
            step = total / frames
            self.calc_step_lbl.setText(f"{step:.4f} µm")

    def _toggle_maximize_maps(self):
        """Hides the left panel (preview and graph) to maximize the map view."""
        if self.left_widget.isVisible():
            self.left_widget.hide()
            self.statusBar().showMessage("Map view maximized. Double-click the image again to restore.")
        else:
            self.left_widget.show()
            self.statusBar().showMessage("Restored default layout.")

    def _create_map_tab(self, map_type, image_label):
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
        
        min_sl, max_sl, min_lbl, max_lbl = self._create_dual_slider(sliders_layout, f"{map_type} Scale Adjustment")
        sliders_container.setVisible(False)
        
        settings_btn.toggled.connect(sliders_container.setVisible)
        
        layout.addWidget(sliders_container)
        layout.addWidget(image_label, stretch=1)
        
        return page, min_sl, max_sl, min_lbl, max_lbl

    def _create_dual_slider(self, layout, title):
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
        return {
            'v_min_pct': self.vis_min_sl.value() / 1000.0,
            'v_max_pct': self.vis_max_sl.value() / 1000.0,
            'c_min_pct': self.con_min_sl.value() / 1000.0,
            'c_max_pct': self.con_max_sl.value() / 1000.0,
            'p_min_pct': self.pha_min_sl.value() / 1000.0,
            'p_max_pct': self.pha_max_sl.value() / 1000.0,
        }

    def _reset_sliders(self):
        self.vis_min_sl.blockSignals(True)
        self.vis_max_sl.blockSignals(True)
        self.con_min_sl.blockSignals(True)
        self.con_max_sl.blockSignals(True)
        self.pha_min_sl.blockSignals(True)
        self.pha_max_sl.blockSignals(True)

        self.vis_min_sl.setValue(0)
        self.con_min_sl.setValue(0)
        self.pha_min_sl.setValue(0)

        self.vis_max_sl.setValue(1000)
        self.con_max_sl.setValue(1000)
        self.pha_max_sl.setValue(1000)

        self.vis_min_sl.blockSignals(False)
        self.vis_max_sl.blockSignals(False)
        self.con_min_sl.blockSignals(False)
        self.con_max_sl.blockSignals(False)
        self.pha_min_sl.blockSignals(False)
        self.pha_max_sl.blockSignals(False)

    def _update_labels(self):
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
        self.vis_min_sl.blockSignals(True)
        self.vis_max_sl.blockSignals(True)
        if self.vis_min_sl.value() >= self.vis_max_sl.value():
            self.vis_min_sl.setValue(max(0, self.vis_max_sl.value() - 1))
        self.vis_min_sl.blockSignals(False)
        self.vis_max_sl.blockSignals(False)

        self.con_min_sl.blockSignals(True)
        self.con_max_sl.blockSignals(True)
        if self.con_min_sl.value() >= self.con_max_sl.value():
            self.con_min_sl.setValue(max(0, self.con_max_sl.value() - 1))
        self.con_min_sl.blockSignals(False)
        self.con_max_sl.blockSignals(False)

        self.pha_min_sl.blockSignals(True)
        self.pha_max_sl.blockSignals(True)
        if self.pha_min_sl.value() >= self.pha_max_sl.value():
            self.pha_min_sl.setValue(max(0, self.pha_max_sl.value() - 1))
        self.pha_min_sl.blockSignals(False)
        self.pha_max_sl.blockSignals(False)

        sf = self.get_scale_factors()

        if self.acq_worker is not None:
            self.acq_worker.scale_factors = sf
        if self.live_proc_worker is not None:
            self.live_proc_worker.scale_factors = sf

        if self.camera is not None and getattr(self.camera, 'last_visibility', None) is not None:
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
            if "total_range_um" in settings:
                self.total_range_spin.setValue(settings["total_range_um"])
            if "settling_ms" in settings:
                self.settling_spin.setValue(settings["settling_ms"])
            if "roi_center_x" in settings:
                self.roi_x_spin.setValue(settings["roi_center_x"])
            if "roi_center_y" in settings:
                self.roi_y_spin.setValue(settings["roi_center_y"])
            if "roi_box_size" in settings:
                self.roi_size_spin.setValue(settings["roi_box_size"])
            if "moving_average" in settings:
                self.ma_checkbox.setChecked(settings["moving_average"])
            if "ma_kernel_size" in settings:
                self.ma_size_spin.setValue(settings["ma_kernel_size"])

            self.statusBar().showMessage(f"Loaded parameters from {file_path}")

        except json.JSONDecodeError:
            QMessageBox.critical(self, "Load Error", "The selected file is not a valid JSON document.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load settings:\n{str(e)}")

    def _save_data(self):
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

        counter = 1
        final_path = full_path
        while os.path.exists(final_path):
            final_path = f"{full_path}_{counter}"
            counter += 1

        try:
            os.makedirs(final_path, exist_ok=True)

            settings = {
                "exposure_ms": self.exposure_spin.value(),
                "gain_db": self.gain_spin.value(),
                "n_frames": self.frames_spin.value(),
                "total_range_um": self.total_range_spin.value(),
                "settling_ms": self.settling_spin.value(),
                "roi_center_x": self.roi_x_spin.value(),
                "roi_center_y": self.roi_y_spin.value(),
                "roi_box_size": self.roi_size_spin.value(),
                "processing_time_s": getattr(self.acq_worker, "last_proc_time", 0.0),
                "moving_average": self.ma_checkbox.isChecked(),
                "ma_kernel_size": self.ma_size_spin.value(),
                "scale_factors_pct": self.get_scale_factors()
            }
            with open(os.path.join(final_path, "settings.json"), "w") as f:
                json.dump(settings, f, indent=4)

            save_map = {
                "visibility_map.png": self.vis_img,
                "contrast_map.png": self.contrast_img,
                "phase_map.png": self.phase_img,
            }
            for filename, label in save_map.items():
                pm = label.original_pixmap() if hasattr(label, 'original_pixmap') else label.pixmap()
                if pm and not pm.isNull():
                    pm.save(os.path.join(final_path, filename), "PNG")

            pm = self.raw_preview.pixmap()
            if pm and not pm.isNull():
                pm.save(os.path.join(final_path, "last_raw_frame.png"), "PNG")

            self.fig.savefig(os.path.join(final_path, "intensity_plot.png"))

            self.statusBar().showMessage(f"Data saved successfully to {final_path}")
            QMessageBox.information(self, "Data Saved", f"Successfully saved to:\n{final_path}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")
    
    def _on_exposure_changed(self, val_ms: int):
        if self.camera is not None and self.camera.camera is not None:
            try:
                self.camera.camera.exposure_time_us = val_ms * 1000
                self.camera.camera.image_poll_timeout_ms = val_ms + 100
            except Exception as e:
                self.statusBar().showMessage(f"Warning: Failed to update exposure live ({e})")

    def _on_gain_changed(self, val_db: int):
        if self.camera is not None and self.camera.camera is not None:
            try:
                self.camera.camera.gain = val_db * 10
            except Exception as e:
                self.statusBar().showMessage(f"Warning: Failed to update gain live ({e})")

    def _on_preview_clicked(self, label_x: int, label_y: int):
        if not self.camera or not self.raw_preview.pixmap():
            return

        pm = self.raw_preview.pixmap()
        pm_w, pm_h = pm.width(), pm.height()
        label_w, label_h = self.raw_preview.width(), self.raw_preview.height()

        offset_x = (label_w - pm_w) / 2.0
        offset_y = (label_h - pm_h) / 2.0

        pixmap_x = label_x - offset_x
        pixmap_y = label_y - offset_y

        if pixmap_x < 0 or pixmap_x > pm_w or pixmap_y < 0 or pixmap_y > pm_h:
            return

        cam_w = self.camera.image_width
        cam_h = self.camera.image_height

        cam_x = int((pixmap_x / pm_w) * cam_w)
        cam_y = int((pixmap_y / pm_h) * cam_h)

        self.roi_x_spin.setValue(cam_x)
        self.roi_y_spin.setValue(cam_y)

    def _connect_hardware(self):
        self.statusBar().showMessage("Connecting to hardware…")

        self.piezo = PiezoController()
        self.camera = CameraController()

        try:
            p_ok, p_msg = self.piezo.connect()
            c_ok = self.camera.connect()
        except Exception as exc:
            QMessageBox.critical(self, "Hardware Error", f"Unexpected error during setup:\n{exc}")
            self.statusBar().showMessage("Hardware connection error.")
            return

        if p_ok and c_ok:
            w = self.camera.image_width
            h = self.camera.image_height
            self.roi_x_spin.setRange(0, w - 1)
            self.roi_y_spin.setRange(0, h - 1)
            self.roi_x_spin.setValue(w // 2)
            self.roi_y_spin.setValue(h // 2)

            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)

            self.statusBar().showMessage("Hardware connected successfully.")
            self._apply_theme()
            QMessageBox.information(self, "Success", "All hardware connected.")
        else:
            errors = []
            if not p_ok: errors.append(f"Piezo: {p_msg}")
            if not c_ok: errors.append("Camera: failed to initialise.")
            QMessageBox.critical(self, "Connection Error", "Failed to connect:\n" + "\n".join(errors))
            self.statusBar().showMessage("Hardware connection failed.")

    def _disconnect_hardware(self):
        reply = QMessageBox.question(
            self, "Confirm Disconnect", "Disconnect all hardware?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._stop_live_worker()
        self._stop_live_proc_worker()
        self._stop_worker()

        if self.camera: self.camera.disconnect()
        if self.piezo: self.piezo.disconnect()

        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.live_proc_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.statusBar().showMessage("Hardware disconnected.")

    def _run_single_acquisition(self):
        self.start_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.live_proc_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.start_btn.setText("Acquiring…")

        self.positions_intensities.clear()
        self.ax.clear()

        exp_ms = self.exposure_spin.value()
        self.camera.camera.exposure_time_us = exp_ms * 1000
        self.camera.camera.gain = self.gain_spin.value() * 10
        self.camera.camera.image_poll_timeout_ms = exp_ms + 100

        self._reset_sliders()

        n_frames = self.frames_spin.value()
        total_range = self.total_range_spin.value()
        calc_step_um = total_range / n_frames

        # Instantiate the newly named worker
        self.acq_worker = SingleAcquisitionWorker(
            camera_ctrl=self.camera,
            piezo_ctrl=self.piezo,
            n_frames=n_frames,
            start_um=self._DEFAULT_START_UM,
            step_um=calc_step_um,
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
        if self.live_proc_worker is not None and self.live_proc_worker.isRunning():
            self._stop_live_proc_worker()
            self.live_proc_btn.setText("Live")
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.statusBar().showMessage("Live processing stopped. Idle.")
        else:
            self.start_btn.setEnabled(False)
            self.live_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.live_proc_btn.setText("Stop")
            self.statusBar().showMessage("Live processing running...")

            self.positions_intensities.clear()
            self.ax.clear()

            exp_ms = self.exposure_spin.value()
            self.camera.camera.exposure_time_us = exp_ms * 1000
            self.camera.camera.gain = self.gain_spin.value() * 10
            self.camera.camera.image_poll_timeout_ms = exp_ms + 100

            self.camera.set_single_frame_mode()
            self._reset_sliders()

            # DYNAMIC STEP CALCULATION
            n_frames = self.frames_spin.value()
            total_range = self.total_range_spin.value()
            calc_step_um = total_range / n_frames

            self.live_proc_worker = LiveProcessingWorker(
                camera_ctrl=self.camera,
                piezo_ctrl=self.piezo,
                n_frames=n_frames,
                start_um=self._DEFAULT_START_UM,
                step_um=calc_step_um,
                settling_time=self.settling_spin.value() / 1000.0,
                scale_factors=self.get_scale_factors()
            )
            self.live_proc_worker.frame_acquired_signal.connect(self._update_preview_and_plot)
            self.live_proc_worker.maps_ready_signal.connect(self._display_maps)
            self.live_proc_worker.error_signal.connect(self._on_error)
            self.live_proc_worker.start()

    def _toggle_live_feed(self):
        if self.live_worker is not None and self.live_worker.isRunning():
            self._stop_live_worker()
            self.live_btn.setText("Raw Feed")
            self.start_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)
            self.statusBar().showMessage("Raw feed stopped. Idle.")
        else:
            self.start_btn.setEnabled(False)
            self.live_proc_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.live_btn.setText("Stop")
            self.statusBar().showMessage("Raw feed running...")

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
        self._stop_live_worker()
        self._stop_live_proc_worker()

        self.live_btn.setText("Raw Feed")
        self.live_proc_btn.setText("Live")
        self.start_btn.setText("Single")

        self.start_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.live_proc_btn.setEnabled(True)
        
        QMessageBox.warning(self, "Hardware Error", msg)

    def _reset_system(self):
        self._stop_live_worker()
        self._stop_live_proc_worker()
        self._stop_worker()

        if self.piezo: self.piezo.move_to_um(0.0)

        self.positions_intensities.clear()
        self.ax.clear()
        self.canvas.draw()

        self.raw_preview.setPixmap(QPixmap())
        self.raw_preview.setText("System reset")

        for widget in (self.vis_img, self.contrast_img, self.phase_img):
            widget.setPixmap(QPixmap())
            widget.setText("No Data")

        self.statusBar().showMessage("System reset.")

        if self.piezo is not None and self.camera is not None:
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
            self.start_btn.setText("Single")
            self.live_btn.setText("Raw Feed")
            self.live_proc_btn.setText("Live")

    def _stop_worker(self):
        if self.acq_worker and self.acq_worker.isRunning():
            self.acq_worker.is_running = False
            self.acq_worker.wait()

    def _stop_live_worker(self):
        if self.live_worker and self.live_worker.isRunning():
            self.live_worker.is_running = False
            self.live_worker.wait()
            self.live_worker = None
            if self.camera: self.camera.set_single_frame_mode()

    def _stop_live_proc_worker(self):
        if self.live_proc_worker and self.live_proc_worker.isRunning():
            self.live_proc_worker.is_running = False
            self.live_proc_worker.wait()
            self.live_proc_worker = None

    def _on_ma_toggled(self, checked: bool):
        self.ma_size_spin.setEnabled(checked)
        if self.camera is not None:
            self.camera.use_moving_average = checked

    def _on_ma_size_changed(self, val: int):
        if val % 2 == 0:
            val += 1
            self.ma_size_spin.blockSignals(True)
            self.ma_size_spin.setValue(val)
            self.ma_size_spin.blockSignals(False)
        if self.camera is not None:
            self.camera.ma_kernel_size = val

    def _update_live_preview(self, gray_img: np.ndarray):
        norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        s = self.roi_size_spin.value() // 2
        cv2.rectangle(rgb, (x - s, y - s), (x + s, y + s), (255, 0, 0), 4)

        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self.raw_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.raw_preview.setPixmap(scaled)

    def _update_preview_and_plot(self, gray_img: np.ndarray, pos_um: float, idx: int):
        self._update_live_preview(gray_img)

        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        s = self.roi_size_spin.value() // 2
        h_img, w_img = gray_img.shape

        y_min, y_max = max(0, y - s), min(h_img, y + s + 1)
        x_min, x_max = max(0, x - s), min(w_img, x + s + 1)

        roi_data = gray_img[y_min:y_max, x_min:x_max]
        mean_intensity = float(np.mean(roi_data))

        # --- THE FIX ---
        # Instead of storing points by readback position (which constantly changes by sub-microns),
        # we store them strictly by their index in the step cycle (e.g. 0 to 49).
        # This guarantees old points are cleanly overwritten.
        
        n_frames = self.frames_spin.value()
        step_index = idx % n_frames
        
        # We save a tuple of (actual_position_um, intensity)
        self.positions_intensities[step_index] = (pos_um, mean_intensity)

        # Sort the dictionary values by actual position so the line draws cleanly left-to-right
        sorted_points = sorted(self.positions_intensities.values(), key=lambda item: item[0])
        
        sorted_positions = [p[0] for p in sorted_points]
        sorted_intensities = [p[1] for p in sorted_points]
        # ---------------

        self.ax.clear()
        self.ax.grid(True, color="#2d2d30", linestyle="--", linewidth=0.5, zorder=0)
        self.ax.plot(
            sorted_positions, sorted_intensities,
            color="#0078d4",
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1,
            zorder=2
        )
        self.ax.set_xlabel("Piezo Position (µm)", fontsize=10, color="#d4d4d4", fontweight="bold")
        self.ax.set_ylabel("ROI Mean Intensity", fontsize=10, color="#d4d4d4", fontweight="bold")
        self.ax.tick_params(colors="#d4d4d4", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#3f3f46")
        self.canvas.draw()

    def _display_maps(self, vis: np.ndarray, contrast: np.ndarray, phase: np.ndarray):
        self.vis_img.setPixmap(self._cv_to_pixmap(vis))
        self.contrast_img.setPixmap(self._cv_to_pixmap(contrast))
        self.phase_img.setPixmap(self._cv_to_pixmap(phase))
        self._update_labels()

    def _on_acquisition_complete(self):
        proc_time = getattr(self.acq_worker, "last_proc_time", 0.0)
        msg = f"Acquisition complete. Processing time: {proc_time:.4f} s"

        self.statusBar().showMessage(msg)
        
        self.start_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.live_proc_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.start_btn.setText("Single")

    @staticmethod
    def _cv_to_pixmap(cv_img: np.ndarray) -> QPixmap:
        h, w, ch = cv_img.shape
        qimg = QImage(cv_img.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _apply_theme(self):
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
                padding: 6px;
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
                background-color: {accent}; 
            }}
            QPushButton:checked {{
                background-color: {accent};
                border: 1px solid #005a9e;
            }}
            QPushButton:disabled {{
                background-color: #444; 
                color: #888; 
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
                background-color: {accent}; 
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
        self._stop_live_worker()
        self._stop_live_proc_worker()
        self._stop_worker()

        try:
            if self.camera: self.camera.disconnect()
            if self.piezo: self.piezo.disconnect()
        except Exception as exc:
            print(f"Error during shutdown: {exc}")

        event.accept()


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "nano_qiup.app.2_0"
        )
    except AttributeError:
        pass

    app = QApplication(sys.argv)
    window = QIUP_APP()
    app.setWindowIcon(QIcon(window.logo_path))
    window.show()
    sys.exit(app.exec_())