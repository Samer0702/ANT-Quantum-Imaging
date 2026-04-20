import sys
import time
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
    QToolBar, QAction, QSizePolicy, QMessageBox, QCheckBox, QTabWidget, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon

from piezo_control import PiezoController
from camera_control import CameraController


# ---------------------------------------------------------------------------
# Custom UI Components
# ---------------------------------------------------------------------------

class ClickableLabel(QLabel):
    """
    A custom QLabel that emits a signal with (x, y) coordinates 
    whenever the user clicks on it with the left mouse button.
    Used for ROI selection on the preview image.
    """
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)


class ScalableImageLabel(QLabel):
    """
    A QLabel that automatically scales its pixmap to fit the available space
    while maintaining aspect ratio. Stores the original pixmap internally
    for lossless saving.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_pixmap = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setPixmap(self, pixmap):
        """Store original pixmap and trigger rescaling."""
        self._original_pixmap = pixmap
        self._rescale()

    def _rescale(self):
        """Scale the pixmap to fit the current widget size."""
        if self._original_pixmap and not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            super().setPixmap(scaled)
        elif self._original_pixmap is None:
            super().setPixmap(QPixmap())

    def resizeEvent(self, event):
        """Rescale image when widget is resized."""
        self._rescale()
        super().resizeEvent(event)

    def original_pixmap(self):
        """Return the unscaled original pixmap."""
        return self._original_pixmap


# ---------------------------------------------------------------------------
# Background Thread 1: Single Acquisition
# ---------------------------------------------------------------------------

class AcquisitionWorker(QThread):
    """
    Worker thread that performs a single N-frame acquisition scan.
    Steps the piezo through the voltage range, captures one frame per step,
    and computes the quantum maps (visibility, contrast, phase) at the end.
    """
    finished_signal = pyqtSignal(object, object, object)  # (vis, contrast, phase)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    frame_acquired_signal = pyqtSignal(np.ndarray, float, int)  # (image, voltage, index)

    def __init__(
        self,
        camera_ctrl: CameraController,
        piezo_ctrl: PiezoController,
        n_frames: int,
        scan_v_start: float,
        period_v: float,
        settling_time: float,
    ):
        super().__init__()
        self.camera_ctrl = camera_ctrl
        self.piezo_ctrl = piezo_ctrl
        self.n_frames = n_frames
        self.scan_v_start = scan_v_start
        self.period_v = period_v
        self.settling_time = settling_time
        self.is_running = True
        self.last_proc_time = 0.0  # Processing time in seconds

    def run(self):
        """Thread entry point."""
        try:
            self._run_scan()
        except Exception as exc:
            self.error_signal.emit(str(exc))

    def _run_scan(self):
        """Execute the full acquisition scan."""
        self.progress_signal.emit("Starting acquisition…")
        
        # Calculate voltage step size
        dv = self.period_v / self.n_frames
        
        # Prepare image stack
        h = self.camera_ctrl.image_height
        w = self.camera_ctrl.image_width
        image_stack = np.zeros((self.n_frames, h, w), dtype=np.float32)
        frames_acquired = 0

        # Acquire frames
        while frames_acquired < self.n_frames and self.is_running:
            # Move piezo to next voltage
            v = self.scan_v_start + (frames_acquired * dv)
            if not self.piezo_ctrl.set_voltage(v):
                self.error_signal.emit(f"Piezo failed to move to {v:.3f} V.")
                return
            
            # Wait for piezo to settle
            time.sleep(self.settling_time)
            
            # Trigger camera and wait for frame
            self.camera_ctrl.camera.issue_software_trigger()
            frame = self.camera_ctrl.camera.get_pending_frame_or_null()
            
            if frame is not None:
                # Process frame
                img = np.copy(frame.image_buffer).reshape(h, w)
                img = cv2.flip(img, 0)
                
                # Apply moving average filter if enabled
                if self.camera_ctrl.use_moving_average:
                    k = self.camera_ctrl.ma_kernel_size
                    k = k if k % 2 != 0 else k + 1
                    img = cv2.blur(img, (k, k))
                
                # Store and emit
                image_stack[frames_acquired] = img
                self.frame_acquired_signal.emit(img, v, frames_acquired)
                frames_acquired += 1
                self.progress_signal.emit(f"Frame {frames_acquired} / {self.n_frames}")
            else:
                # Dropped frame - retry after short delay
                time.sleep(0.01)

        # Check if aborted
        if not self.is_running:
            self.progress_signal.emit("Acquisition aborted.")
            return

        # Return piezo to start position
        self.piezo_ctrl.set_voltage(self.scan_v_start)
        
        # Compute quantum maps with timing
        self.progress_signal.emit("Computing Fourier transform…")
        t_start = time.perf_counter()
        vis, contrast, phase = self.camera_ctrl.process_quantum_image(image_stack)
        t_end = time.perf_counter()
        self.last_proc_time = t_end - t_start
        
        # Emit results
        self.finished_signal.emit(vis, contrast, phase)


# ---------------------------------------------------------------------------
# Background Thread 2: Raw Live Feed
# ---------------------------------------------------------------------------

class LiveFeedWorker(QThread):
    """
    Worker thread that continuously polls the camera for raw frames
    without moving the piezo. Used for live preview mode.
    """
    frame_ready_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, camera_ctrl: CameraController):
        super().__init__()
        self.camera_ctrl = camera_ctrl
        self.is_running = True

    def run(self):
        """Continuously fetch and emit camera frames."""
        cam = self.camera_ctrl.camera
        w = self.camera_ctrl.image_width
        h = self.camera_ctrl.image_height
        
        while self.is_running:
            try:
                frame = cam.get_pending_frame_or_null()
                if frame is not None:
                    # Process frame
                    img = np.copy(frame.image_buffer).reshape(h, w)
                    img = cv2.flip(img, 0)
                    
                    # Apply moving average if enabled
                    if self.camera_ctrl.use_moving_average:
                        k = self.camera_ctrl.ma_kernel_size
                        k = k if k % 2 != 0 else k + 1
                        img = cv2.blur(img, (k, k))
                    
                    self.frame_ready_signal.emit(img)
                else:
                    time.sleep(0.01)
            except Exception as exc:
                if self.is_running:
                    self.error_signal.emit(str(exc))
                break


# ---------------------------------------------------------------------------
# Background Thread 3: Live Quantum Processing (Circular Buffer)
# ---------------------------------------------------------------------------

class LiveProcessingWorker(QThread):
    """
    Worker thread that continuously steps the piezo and captures frames,
    maintaining a circular buffer. Computes and emits quantum maps after
    each full period is acquired.
    """
    maps_ready_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    frame_acquired_signal = pyqtSignal(np.ndarray, float, int)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        camera_ctrl: CameraController,
        piezo_ctrl: PiezoController,
        n_frames: int,
        scan_v_start: float,
        period_v: float,
        settling_time: float
    ):
        super().__init__()
        self.camera_ctrl = camera_ctrl
        self.piezo_ctrl = piezo_ctrl
        self.n_frames = n_frames
        self.scan_v_start = scan_v_start
        self.period_v = period_v
        self.settling_time = settling_time
        self.is_running = True

    def run(self):
        """Continuously acquire frames and compute quantum maps."""
        cam = self.camera_ctrl.camera
        w = self.camera_ctrl.image_width
        h = self.camera_ctrl.image_height
        
        # Circular buffer indexed by voltage step
        voltage_buffer = np.zeros((self.n_frames, h, w), dtype=np.float32)
        dv = self.period_v / self.n_frames
        total_frames_acquired = 0

        while self.is_running:
            try:
                # Determine current voltage step
                step_index = total_frames_acquired % self.n_frames
                current_v = self.scan_v_start + (step_index * dv)
                
                # Move piezo
                if not self.piezo_ctrl.set_voltage(current_v):
                    self.error_signal.emit(f"Piezo failed at {current_v:.3f} V.")
                    break
                
                # Wait for settling
                time.sleep(self.settling_time)
                
                # Capture frame
                cam.issue_software_trigger()
                frame = cam.get_pending_frame_or_null()
                
                if frame is not None:
                    # Process frame
                    img = np.copy(frame.image_buffer).reshape(h, w)
                    img = cv2.flip(img, 0)
                    
                    # Apply moving average if enabled
                    if self.camera_ctrl.use_moving_average:
                        k = self.camera_ctrl.ma_kernel_size
                        k = k if k % 2 != 0 else k + 1
                        img = cv2.blur(img, (k, k))
                    
                    # Emit preview update
                    self.frame_acquired_signal.emit(img, current_v, total_frames_acquired)
                    
                    # Update circular buffer at the correct index
                    voltage_buffer[step_index] = img
                    total_frames_acquired += 1
                    
                    # Compute quantum maps once we have a full period
                    if total_frames_acquired >= self.n_frames:
                        vis, contrast, phase = self.camera_ctrl.process_quantum_image(voltage_buffer)
                        self.maps_ready_signal.emit(vis, contrast, phase)
                        
            except Exception as exc:
                if self.is_running:
                    self.error_signal.emit(str(exc))
                break


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------

class QIUP_APP(QMainWindow):
    """
    Main application window for the QIUP (Quantum Imaging with Undetected Photons) Dashboard.
    Provides UI for controlling camera, piezo, and viewing quantum imaging results.
    """

    # Default scan parameters
    _DEFAULT_SCAN_V_START = 0.0
    _DEFAULT_SCAN_V_END = 3.9
    _DEFAULT_SETTLING_MS = 10

    def __init__(self):
        super().__init__()

        # Application metadata
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logo_path = os.path.join(self.base_dir, "logo", "Logo_ANT.png")

        self.setWindowTitle("NANO: QIUP Dashboard")
        self.setGeometry(50, 50, 1800, 1000)
        self.setWindowIcon(QIcon(self.logo_path))

        # Hardware controllers
        self.piezo: PiezoController | None = None
        self.camera: CameraController | None = None

        # Worker threads
        self.acq_worker: AcquisitionWorker | None = None
        self.live_worker: LiveFeedWorker | None = None
        self.live_proc_worker: LiveProcessingWorker | None = None

        # Data storage for intensity plot
        self.voltages_intensities: dict[float, float] = {}

        # Build UI and apply theme
        self._setup_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Build the complete user interface."""
        # Create toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Add hardware connection actions
        self.connect_action = QAction("Connect Hardware", self)
        self.connect_action.triggered.connect(self._connect_hardware)
        toolbar.addAction(self.connect_action)

        self.disconnect_action = QAction("Disconnect", self)
        self.disconnect_action.triggered.connect(self._disconnect_hardware)
        self.disconnect_action.setEnabled(False)
        toolbar.addAction(self.disconnect_action)

        # Create central widget with main layout
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(5, 5, 5, 5)
        root.setSpacing(10)

        # ========== LEFT COLUMN: Settings Panel (Fixed Width) ==========
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(self._build_left_panel())
        left_panel_widget.setFixedWidth(320)
        root.addWidget(left_panel_widget)

        # ========== MIDDLE COLUMN: Preview + Graph (30% of remaining space) ==========
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(10)

        # --- Preview Section ---
        preview_group = QGroupBox("Current Frame Preview")
        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(10, 20, 10, 10)

        # Create clickable preview label for ROI selection
        self.raw_preview = ClickableLabel("Waiting for trigger…")
        self.raw_preview.setMinimumSize(300, 300)
        self.raw_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.raw_preview.setAlignment(Qt.AlignCenter)
        self.raw_preview.setProperty("is_image", True)
        self.raw_preview.clicked.connect(self._on_preview_clicked)
        preview_layout.addWidget(self.raw_preview, stretch=1)

        # Live feed toggle button
        self.live_btn = QPushButton("Start Raw Feed")
        self.live_btn.setMinimumHeight(40)
        self.live_btn.setEnabled(False)
        self.live_btn.clicked.connect(self._toggle_live_feed)
        preview_layout.addWidget(self.live_btn)

        preview_group.setLayout(preview_layout)
        middle_layout.addWidget(preview_group, stretch=3)

        # --- Graph Section ---
        cycle_group = QGroupBox("ROI Intensity vs Piezo Voltage")
        cycle_layout = QVBoxLayout()
        cycle_layout.setContentsMargins(10, 20, 10, 10)

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 3.5))
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(250)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cycle_layout.addWidget(self.canvas)

        cycle_group.setLayout(cycle_layout)
        middle_layout.addWidget(cycle_group, stretch=2)

        root.addWidget(middle_widget, stretch=3)  # 30% width

        # ========== RIGHT COLUMN: Tab Widget for Maps (70% of remaining space) ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Create tab widget
        self.map_tabs = QTabWidget()
        self.map_tabs.setDocumentMode(False)
        self.map_tabs.setTabPosition(QTabWidget.North)

        # Create scalable image labels for each quantum map
        self.vis_img = ScalableImageLabel("No Data")
        self.vis_img.setProperty("is_image", True)

        self.contrast_img = ScalableImageLabel("No Data")
        self.contrast_img.setProperty("is_image", True)

        self.phase_img = ScalableImageLabel("No Data")
        self.phase_img.setProperty("is_image", True)

        # Helper function to wrap images in pages
        def make_tab_page(image_label):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(15, 15, 15, 15)
            layout.addWidget(image_label)
            return page

        # Add tabs with clear, untruncated labels
        self.map_tabs.addTab(make_tab_page(self.vis_img), "Visibility")
        self.map_tabs.addTab(make_tab_page(self.contrast_img), "Contrast")
        self.map_tabs.addTab(make_tab_page(self.phase_img), "Phase")

        right_layout.addWidget(self.map_tabs)
        root.addWidget(right_widget, stretch=7)  # 70% width

        # Set status bar
        self.statusBar().showMessage("Ready")

    def _build_left_panel(self) -> QVBoxLayout:
        """
        Build the left settings panel containing all control widgets.
        Returns the layout to be added to the main window.
        """
        panel = QVBoxLayout()
        panel.setContentsMargins(5, 5, 5, 5)
        panel.setSpacing(10)

        # --- Camera Settings Group ---
        cam_group = QGroupBox("CMOS Settings")
        cam_layout = QFormLayout()

        # Exposure time control
        self.exposure_spin = QSpinBox()
        self.exposure_spin.setRange(1, 5000)
        self.exposure_spin.setValue(200)
        self.exposure_spin.setSuffix(" ms")
        self.exposure_spin.valueChanged.connect(self._on_exposure_changed)

        # Gain control
        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 48)
        self.gain_spin.setValue(35)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.valueChanged.connect(self._on_gain_changed)

        cam_layout.addRow("Exposure:", self.exposure_spin)
        cam_layout.addRow("Gain:", self.gain_spin)
        cam_group.setLayout(cam_layout)
        panel.addWidget(cam_group)

        # --- Processing Options Group ---
        proc_group = QGroupBox("Processing Options")
        proc_layout = QFormLayout()

        # Moving average filter toggle
        self.ma_checkbox = QCheckBox("Use Moving Average Filter")
        self.ma_checkbox.toggled.connect(self._on_ma_toggled)

        # Moving average kernel size
        self.ma_size_spin = QSpinBox()
        self.ma_size_spin.setSingleStep(2)
        self.ma_size_spin.setRange(3, 101)
        self.ma_size_spin.setValue(3)
        self.ma_size_spin.setSuffix(" px")
        self.ma_size_spin.setEnabled(False)
        self.ma_size_spin.valueChanged.connect(self._on_ma_size_changed)

        proc_layout.addRow(self.ma_checkbox)
        proc_layout.addRow("Window Size:", self.ma_size_spin)
        proc_group.setLayout(proc_layout)
        panel.addWidget(proc_group)

        # --- ROI Settings Group ---
        roi_group = QGroupBox("Plot ROI Settings")
        roi_layout = QFormLayout()

        # ROI center X coordinate
        self.roi_x_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 4000)
        self.roi_x_spin.setValue(0)

        # ROI center Y coordinate
        self.roi_y_spin = QSpinBox()
        self.roi_y_spin.setRange(0, 4000)
        self.roi_y_spin.setValue(0)

        # ROI box size
        self.roi_size_spin = QSpinBox()
        self.roi_size_spin.setRange(1, 1000)
        self.roi_size_spin.setValue(50)
        self.roi_size_spin.setSuffix(" px")

        roi_layout.addRow("Center X:", self.roi_x_spin)
        roi_layout.addRow("Center Y:", self.roi_y_spin)
        roi_layout.addRow("Box Size:", self.roi_size_spin)

        # Hint label
        hint_lbl = QLabel("(Click on preview to select ROI)")
        hint_lbl.setStyleSheet("font-size: 10px; color: #888888;")
        roi_layout.addRow(hint_lbl)

        roi_group.setLayout(roi_layout)
        panel.addWidget(roi_group)

        # --- Global Scan Parameters Group ---
        params_group = QGroupBox("Global Scan Parameters")
        params_layout = QFormLayout()

        # Number of frames per scan
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(3, 1000)
        self.frames_spin.setValue(8)
        self.frames_spin.setToolTip("Applies to both Single Acquisition and Live Processing buffer.")

        # Fringe period voltage
        self.scan_end_spin = QDoubleSpinBox()
        self.scan_end_spin.setRange(0.01, PiezoController.MAX_VOLTAGE)
        self.scan_end_spin.setDecimals(2)
        self.scan_end_spin.setSingleStep(0.25)
        self.scan_end_spin.setValue(self._DEFAULT_SCAN_V_END)
        self.scan_end_spin.setSuffix(" V")

        # Piezo settling time
        self.settling_spin = QSpinBox()
        self.settling_spin.setRange(0, 1000)
        self.settling_spin.setValue(self._DEFAULT_SETTLING_MS)
        self.settling_spin.setSuffix(" ms")

        params_layout.addRow("Frames / Buffer (N):", self.frames_spin)
        params_layout.addRow("Fringe Period (V):", self.scan_end_spin)
        params_layout.addRow("Settling time:", self.settling_spin)

        # Reset button
        self.reset_btn = QPushButton("Reset System")
        self.reset_btn.setMinimumHeight(38)
        self.reset_btn.clicked.connect(self._reset_system)
        params_layout.addRow(self.reset_btn)

        params_group.setLayout(params_layout)
        panel.addWidget(params_group)

        # --- Hardware Operations Group ---
        ops_group = QGroupBox("Hardware Operations")
        ops_layout = QVBoxLayout()
        ops_layout.setSpacing(10)

        # Single acquisition button
        self.start_btn = QPushButton("Run Single Acquisition")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._run_acquisition)

        # Live processing button
        self.live_proc_btn = QPushButton("Start Live Processing")
        self.live_proc_btn.setMinimumHeight(42)
        self.live_proc_btn.setEnabled(False)
        self.live_proc_btn.clicked.connect(self._toggle_live_processing)

        ops_layout.addWidget(self.start_btn)
        ops_layout.addWidget(self.live_proc_btn)

        # Status label
        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888888; font-style: italic; margin-top: 5px;")
        ops_layout.addWidget(self.status_label)

        ops_group.setLayout(ops_layout)
        panel.addWidget(ops_group)

        # --- Data Export Group ---
        export_group = QGroupBox("Save Data")
        export_layout = QVBoxLayout()
        export_layout.setSpacing(10)

        # Save button
        self.save_btn = QPushButton("Save Acquisition Data")
        self.save_btn.setMinimumHeight(42)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_data)

        export_layout.addWidget(self.save_btn)
        export_group.setLayout(export_layout)
        panel.addWidget(export_group)

        # Add stretch to push everything to the top
        panel.addStretch()
        return panel

    # ------------------------------------------------------------------
    # Data Saving Functionality
    # ------------------------------------------------------------------

    def _save_data(self):
        """
        Save all acquisition data to a user-specified timestamped folder:
        - settings.json with all parameters
        - visibility_map.png, contrast_map.png, phase_map.png
        - last_raw_frame.png
        - intensity_plot.png
        """

        # Format the date clearly: DD_MM_YYYY
        date_str = datetime.date.today().strftime("%d_%m_%Y")
        
        # 1. Open a single "Save As" window. 
        # We set a default name so the user knows what to expect.
        default_name = f"Acquisition_{date_str}"
        file_path, _ = QFileDialog.getSaveFileName(
            self
        )
        
        if not file_path:
            return  # User clicked cancel

        # 2. Split the returned path into the Directory and the Typed Name
        base_dir = os.path.dirname(file_path)
        user_name = os.path.basename(file_path)
        
        # 3. Construct the folder name intelligently
        # If they just left the default name alone, don't duplicate the tag.
        if user_name.endswith(date_str):
            folder_name = user_name
        else:
            folder_name = f"{user_name}_Acquisition_{date_str}"
            
        full_path = os.path.join(base_dir, folder_name)

        # 4. Avoid overwriting by appending a counter if the exact folder already exists
        counter = 1
        final_path = full_path
        while os.path.exists(final_path):
            final_path = f"{full_path}_{counter}"
            counter += 1

        try:
            # Create the directory at the chosen location
            os.makedirs(final_path, exist_ok=True)

            # Save settings as JSON
            settings = {
                "exposure_ms": self.exposure_spin.value(),
                "gain_db": self.gain_spin.value(),
                "n_frames": self.frames_spin.value(),
                "fringe_period_v": self.scan_end_spin.value(),
                "settling_ms": self.settling_spin.value(),
                "roi_center_x": self.roi_x_spin.value(),
                "roi_center_y": self.roi_y_spin.value(),
                "roi_box_size": self.roi_size_spin.value(),
                "processing_time_s": getattr(self.acq_worker, "last_proc_time", 0.0),
                "moving_average": self.ma_checkbox.isChecked(),
                "ma_kernel_size": self.ma_size_spin.value()
            }
            with open(os.path.join(final_path, "settings.json"), "w") as f:
                json.dump(settings, f, indent=4)

            # Save quantum maps
            save_map = {
                "visibility_map.png": self.vis_img,
                "contrast_map.png": self.contrast_img,
                "phase_map.png": self.phase_img,
            }
            for filename, label in save_map.items():
                pm = label.original_pixmap() if hasattr(label, 'original_pixmap') else label.pixmap()
                if pm and not pm.isNull():
                    pm.save(os.path.join(final_path, filename), "PNG")

            # Save raw preview frame
            pm = self.raw_preview.pixmap()
            if pm and not pm.isNull():
                pm.save(os.path.join(final_path, "last_raw_frame.png"), "PNG")

            # Save intensity plot
            self.fig.savefig(os.path.join(final_path, "intensity_plot.png"))

            # Update UI
            self.status_label.setText(f"Data saved to {final_path}")
            self.statusBar().showMessage(f"Data saved successfully to {final_path}")
            QMessageBox.information(self, "Data Saved", f"Successfully saved to:\n{final_path}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save data: {str(e)}")
    
    # ------------------------------------------------------------------
    # Real-Time Camera Hardware Updates
    # ------------------------------------------------------------------

    def _on_exposure_changed(self, val_ms: int):
        """Update camera exposure time immediately when spinbox changes."""
        if self.camera is not None and self.camera.camera is not None:
            try:
                self.camera.camera.exposure_time_us = val_ms * 1000
                self.camera.camera.image_poll_timeout_ms = val_ms + 100
            except Exception as e:
                self.status_label.setText(f"Warning: Failed to update exposure live ({e})")

    def _on_gain_changed(self, val_db: int):
        """Update camera gain immediately when spinbox changes."""
        if self.camera is not None and self.camera.camera is not None:
            try:
                self.camera.camera.gain = val_db * 10
            except Exception as e:
                self.status_label.setText(f"Warning: Failed to update gain live ({e})")

    # ------------------------------------------------------------------
    # ROI Click Handling
    # ------------------------------------------------------------------

    def _on_preview_clicked(self, label_x: int, label_y: int):
        """
        Handle mouse click on preview image to set ROI center.
        Maps click coordinates from the displayed pixmap back to 
        the original camera resolution.
        """
        if not self.camera or not self.raw_preview.pixmap():
            return

        pm = self.raw_preview.pixmap()
        pm_w, pm_h = pm.width(), pm.height()
        label_w, label_h = self.raw_preview.width(), self.raw_preview.height()

        # Calculate black bar offsets (due to KeepAspectRatio scaling)
        offset_x = (label_w - pm_w) / 2.0
        offset_y = (label_h - pm_h) / 2.0

        # Get coordinates relative to the actual pixmap
        pixmap_x = label_x - offset_x
        pixmap_y = label_y - offset_y

        # Ignore clicks on black bars
        if pixmap_x < 0 or pixmap_x > pm_w or pixmap_y < 0 or pixmap_y > pm_h:
            return

        # Map to camera resolution
        cam_w = self.camera.image_width
        cam_h = self.camera.image_height

        cam_x = int((pixmap_x / pm_w) * cam_w)
        cam_y = int((pixmap_y / pm_h) * cam_h)

        # Update spinboxes
        self.roi_x_spin.setValue(cam_x)
        self.roi_y_spin.setValue(cam_y)

    # ------------------------------------------------------------------
    # Hardware Lifecycle
    # ------------------------------------------------------------------

    def _connect_hardware(self):
        """Initialize and connect to piezo controller and camera."""
        self.status_label.setText("Connecting…")
        self.statusBar().showMessage("Connecting to hardware…")

        self.piezo = PiezoController()
        self.camera = CameraController()

        try:
            p_ok, p_msg = self.piezo.connect()
            c_ok = self.camera.connect()
        except Exception as exc:
            QMessageBox.critical(self, "Hardware Error", f"Unexpected error during setup:\n{exc}")
            self.status_label.setText("Error.")
            self.statusBar().showMessage("Hardware connection error.")
            return

        if p_ok and c_ok:
            # Initialize ROI to center of image
            w = self.camera.image_width
            h = self.camera.image_height
            self.roi_x_spin.setRange(0, w - 1)
            self.roi_y_spin.setRange(0, h - 1)
            self.roi_x_spin.setValue(w // 2)
            self.roi_y_spin.setValue(h // 2)

            # Enable UI controls
            self.connect_action.setEnabled(False)
            self.disconnect_action.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)

            self.status_label.setText("Hardware connected.")
            self.statusBar().showMessage("Hardware connected successfully.")
            self._apply_theme()
            QMessageBox.information(self, "Success", "All hardware connected.")
        else:
            # Report connection errors
            errors = []
            if not p_ok: errors.append(f"Piezo: {p_msg}")
            if not c_ok: errors.append("Camera: failed to initialise.")
            QMessageBox.critical(self, "Connection Error", "Failed to connect:\n" + "\n".join(errors))
            self.status_label.setText("Connection failed.")
            self.statusBar().showMessage("Hardware connection failed.")

    def _disconnect_hardware(self):
        """Disconnect all hardware safely."""
        reply = QMessageBox.question(
            self, "Confirm Disconnect", "Disconnect all hardware?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Stop all worker threads
        self._stop_live_worker()
        self._stop_live_proc_worker()
        self._stop_worker()

        # Disconnect hardware
        if self.camera: self.camera.disconnect()
        if self.piezo: self.piezo.disconnect()

        # Update UI state
        self.connect_action.setEnabled(True)
        self.disconnect_action.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.live_proc_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.status_label.setText("Idle")
        self.statusBar().showMessage("Hardware disconnected.")

    # ------------------------------------------------------------------
    # Acquisition and Live Feeds
    # ------------------------------------------------------------------

    def _run_acquisition(self):
        """Start a single N-frame acquisition scan."""
        # Disable controls during acquisition
        self.start_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.live_proc_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.start_btn.setText("Acquiring…")

        # Clear previous data
        self.voltages_intensities.clear()
        self.ax.clear()

        # Apply camera settings
        exp_ms = self.exposure_spin.value()
        self.camera.camera.exposure_time_us = exp_ms * 1000
        self.camera.camera.gain = self.gain_spin.value() * 10
        self.camera.camera.image_poll_timeout_ms = exp_ms + 100

        # Calculate voltage period
        period_v = self.scan_end_spin.value() - self._DEFAULT_SCAN_V_START

        # Create and start worker thread
        self.acq_worker = AcquisitionWorker(
            camera_ctrl=self.camera,
            piezo_ctrl=self.piezo,
            n_frames=self.frames_spin.value(),
            scan_v_start=self._DEFAULT_SCAN_V_START,
            period_v=period_v,
            settling_time=self.settling_spin.value() / 1000.0,
        )
        self.acq_worker.progress_signal.connect(self.status_label.setText)
        self.acq_worker.error_signal.connect(self._on_error)
        self.acq_worker.frame_acquired_signal.connect(self._update_preview_and_plot)
        self.acq_worker.finished_signal.connect(self._display_maps)
        self.acq_worker.finished.connect(self._on_acquisition_complete)
        self.acq_worker.start()

    def _toggle_live_processing(self):
        """Start or stop live quantum processing mode."""
        if self.live_proc_worker is not None and self.live_proc_worker.isRunning():
            # Stop live processing
            self._stop_live_proc_worker()
            self.live_proc_btn.setText("Start Live Processing")
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.status_label.setText("Live processing stopped. Idle.")
        else:
            # Start live processing
            self.start_btn.setEnabled(False)
            self.live_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.live_proc_btn.setText("Stop Live Processing")
            self.status_label.setText("Live processing running...")

            # Clear previous data
            self.voltages_intensities.clear()
            self.ax.clear()

            # Apply camera settings
            exp_ms = self.exposure_spin.value()
            self.camera.camera.exposure_time_us = exp_ms * 1000
            self.camera.camera.gain = self.gain_spin.value() * 10
            self.camera.camera.image_poll_timeout_ms = exp_ms + 100

            self.camera.set_single_frame_mode()
            period_v = self.scan_end_spin.value() - self._DEFAULT_SCAN_V_START

            # Create and start worker thread
            self.live_proc_worker = LiveProcessingWorker(
                camera_ctrl=self.camera,
                piezo_ctrl=self.piezo,
                n_frames=self.frames_spin.value(),
                scan_v_start=self._DEFAULT_SCAN_V_START,
                period_v=period_v,
                settling_time=self.settling_spin.value() / 1000.0,
            )
            self.live_proc_worker.frame_acquired_signal.connect(self._update_preview_and_plot)
            self.live_proc_worker.maps_ready_signal.connect(self._display_maps)
            self.live_proc_worker.error_signal.connect(self._on_error)
            self.live_proc_worker.start()

    def _toggle_live_feed(self):
        """Start or stop raw camera feed (no piezo movement)."""
        if self.live_worker is not None and self.live_worker.isRunning():
            # Stop live feed
            self._stop_live_worker()
            self.live_btn.setText("Start Raw Feed")
            self.start_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)
            self.status_label.setText("Raw feed stopped. Idle.")
        else:
            # Start live feed
            self.start_btn.setEnabled(False)
            self.live_proc_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.live_btn.setText("Stop Raw Feed")
            self.status_label.setText("Raw feed running...")

            # Apply camera settings
            exp_ms = self.exposure_spin.value()
            self.camera.camera.exposure_time_us = exp_ms * 1000
            self.camera.camera.gain = self.gain_spin.value() * 10
            self.camera.camera.image_poll_timeout_ms = exp_ms + 100

            self.camera.set_continuous_mode()

            # Create and start worker thread
            self.live_worker = LiveFeedWorker(self.camera)
            self.live_worker.frame_ready_signal.connect(self._update_live_preview)
            self.live_worker.error_signal.connect(self._on_error)
            self.live_worker.start()

    def _on_error(self, msg: str):
        """Handle errors from worker threads."""
        # Stop all workers
        self._stop_live_worker()
        self._stop_live_proc_worker()

        # Reset button states
        self.live_btn.setText("Start Raw Feed")
        self.live_proc_btn.setText("Start Live Processing")
        self.start_btn.setText("Run Acquisition")

        self.start_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.live_proc_btn.setEnabled(True)
        
        QMessageBox.warning(self, "Hardware Error", msg)

    def _reset_system(self):
        """Reset the entire system to idle state."""
        # Stop all threads
        self._stop_live_worker()
        self._stop_live_proc_worker()
        self._stop_worker()

        # Reset piezo to 0V
        if self.piezo: self.piezo.set_voltage(0.0)

        # Clear all data
        self.voltages_intensities.clear()
        self.ax.clear()
        self.canvas.draw()

        # Clear all images
        self.raw_preview.setPixmap(QPixmap())
        self.raw_preview.setText("System reset")

        for widget in (self.vis_img, self.contrast_img, self.phase_img):
            widget.setPixmap(QPixmap())
            widget.setText("No Data")

        # Update status
        self.status_label.setText("System reset. Idle.")
        self.statusBar().showMessage("System reset.")

        # Re-enable controls if hardware is connected
        if self.piezo is not None and self.camera is not None:
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)
            self.save_btn.setEnabled(False)
            self.start_btn.setText("Run Acquisition")
            self.live_btn.setText("Start Raw Feed")
            self.live_proc_btn.setText("Start Live Processing")

    def _stop_worker(self):
        """Stop the single acquisition worker thread."""
        if self.acq_worker and self.acq_worker.isRunning():
            self.acq_worker.is_running = False
            self.acq_worker.wait()

    def _stop_live_worker(self):
        """Stop the live feed worker thread."""
        if self.live_worker and self.live_worker.isRunning():
            self.live_worker.is_running = False
            self.live_worker.wait()
            self.live_worker = None
            if self.camera: self.camera.set_single_frame_mode()

    def _stop_live_proc_worker(self):
        """Stop the live processing worker thread."""
        if self.live_proc_worker and self.live_proc_worker.isRunning():
            self.live_proc_worker.is_running = False
            self.live_proc_worker.wait()
            self.live_proc_worker = None

    # ------------------------------------------------------------------
    # Image Processing UI Slots
    # ------------------------------------------------------------------

    def _on_ma_toggled(self, checked: bool):
        """Enable or disable moving average filter."""
        self.ma_size_spin.setEnabled(checked)
        if self.camera is not None:
            self.camera.use_moving_average = checked

    def _on_ma_size_changed(self, val: int):
        """Update moving average kernel size (force odd numbers)."""
        if val % 2 == 0:
            val += 1
            self.ma_size_spin.blockSignals(True)
            self.ma_size_spin.setValue(val)
            self.ma_size_spin.blockSignals(False)
        if self.camera is not None:
            self.camera.ma_kernel_size = val

    # ------------------------------------------------------------------
    # UI Update Slots
    # ------------------------------------------------------------------

    def _update_live_preview(self, gray_img: np.ndarray):
        """
        Update the preview label with a new grayscale image.
        Draws ROI rectangle on top.
        """
        # Normalize and convert to RGB
        norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

        # Draw ROI rectangle
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        s = self.roi_size_spin.value() // 2
        cv2.rectangle(rgb, (x - s, y - s), (x + s, y + s), (255, 0, 0), 4)

        # Convert to QPixmap and display
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled = pixmap.scaled(
            self.raw_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.raw_preview.setPixmap(scaled)

    def _update_preview_and_plot(self, gray_img: np.ndarray, v: float, idx: int):
        """
        Update both preview image and intensity plot with new frame data.
        Called during acquisition and live processing.
        """
        # Update preview
        self._update_live_preview(gray_img)

        # Extract ROI and calculate mean intensity
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        s = self.roi_size_spin.value() // 2
        h_img, w_img = gray_img.shape

        y_min, y_max = max(0, y - s), min(h_img, y + s + 1)
        x_min, x_max = max(0, x - s), min(w_img, x + s + 1)

        roi_data = gray_img[y_min:y_max, x_min:x_max]
        mean_intensity = float(np.mean(roi_data))

        # Store intensity at this voltage (rounded to avoid floating point issues)
        v_rounded = round(v, 3)
        self.voltages_intensities[v_rounded] = mean_intensity

        # Update plot
        sorted_voltages = sorted(self.voltages_intensities.keys())
        sorted_intensities = [self.voltages_intensities[v] for v in sorted_voltages]

        self.ax.clear()
        self.ax.grid(True, color="#2d2d30", linestyle="--", linewidth=0.5, zorder=0)
        self.ax.plot(
            sorted_voltages, sorted_intensities,
            color="#0078d4",
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1,
            zorder=2
        )
        self.ax.set_xlabel("Piezo Voltage (V)", fontsize=10, color="#d4d4d4", fontweight="bold")
        self.ax.set_ylabel("ROI Mean Intensity", fontsize=10, color="#d4d4d4", fontweight="bold")
        self.ax.tick_params(colors="#d4d4d4", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#3f3f46")
        self.canvas.draw()

    def _display_maps(self, vis: np.ndarray, contrast: np.ndarray, phase: np.ndarray):
        """Display the three quantum maps in their respective tabs."""
        self.vis_img.setPixmap(self._cv_to_pixmap(vis))
        self.contrast_img.setPixmap(self._cv_to_pixmap(contrast))
        self.phase_img.setPixmap(self._cv_to_pixmap(phase))

    def _on_acquisition_complete(self):
        """Handle completion of single acquisition."""
        proc_time = getattr(self.acq_worker, "last_proc_time", 0.0)
        msg = f"Acquisition complete. Processing time: {proc_time:.4f} s"

        self.status_label.setText(msg)
        self.statusBar().showMessage(msg)
        
        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.live_proc_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.start_btn.setText("Run Acquisition")

    @staticmethod
    def _cv_to_pixmap(cv_img: np.ndarray) -> QPixmap:
        """Convert OpenCV image (RGB) to QPixmap."""
        h, w, ch = cv_img.shape
        qimg = QImage(cv_img.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def _apply_theme(self):
        """Apply dark theme styling to all widgets."""
        # Color palette
        bg, fg, border, img_bg = "#1e1e1e", "#d4d4d4", "#3f3f46", "#121212"
        btn_bg, bar_bg = "#333337", "#2d2d30"
        accent = "#0078d4"

        self.setStyleSheet(f"""
            /* Main window and widgets */
            QMainWindow, QWidget {{
                background-color: {bg}; 
                color: {fg};
                font-family: Segoe UI, Arial; 
                font-size: 12px;
            }}
            
            /* Toolbar */
            QToolBar {{
                background-color: {bar_bg};
                border-bottom: 1px solid {border}; 
                padding: 6px;
            }}
            
            /* Group boxes */
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
            
            /* Buttons */
            QPushButton {{
                background-color: {btn_bg}; 
                color: white; 
                padding: 10px;
                border: 1px solid {border}; 
                border-radius: 4px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {accent}; 
            }}
            QPushButton:disabled {{
                background-color: #444; 
                color: #888; 
            }}
            
            /* Spin boxes */
            QSpinBox, QDoubleSpinBox {{
                background-color: {img_bg}; 
                color: {fg};
                border: 1px solid {border}; 
                padding: 5px;
                border-radius: 3px;
            }}
            
            /* Image labels */
            QLabel[is_image="true"] {{
                background-color: {img_bg};
                border: 2px dashed {border}; 
                color: #666;
            }}
            
            /* Status bar */
            QStatusBar {{
                background-color: {bar_bg}; 
                color: {fg};
                border-top: 1px solid {border};
            }}
            
            /* Checkboxes */
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

            /* Tab Widget */
            QTabWidget::pane {{
                border: 1px solid {border};
                background-color: {bg};
                border-radius: 6px;
                top: -1px;
            }}
            
            /* Individual tabs */
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
        """)

        # Apply theme to matplotlib figure
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(img_bg)
        self.ax.tick_params(colors=fg)
        for spine in self.ax.spines.values():
            spine.set_color(border)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Window Close Handler
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Handle window close event - prompt for confirmation if hardware is connected."""
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

        # Shutdown sequence
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
    # Set Windows taskbar icon
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "nano_qiup.app.1_0"
        )
    except AttributeError:
        pass

    # Create and run application
    app = QApplication(sys.argv)
    window = QIUP_APP()
    app.setWindowIcon(QIcon(window.logo_path))
    window.show()
    sys.exit(app.exec_())