import sys
import time
import os
import ctypes
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QToolBar, QAction, QSizePolicy, QMessageBox, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap, QIcon

from piezo_control import PiezoController
from camera_control import CameraController


# ---------------------------------------------------------------------------
# Custom UI Components
# ---------------------------------------------------------------------------

class ClickableLabel(QLabel):
    """
    A custom QLabel that emits a signal with the X, Y coordinates 
    whenever it is clicked with the left mouse button.
    """
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Background Thread 1: Single Acquisition
# ---------------------------------------------------------------------------

class AcquisitionWorker(QThread):
    """
    Drives the piezo scan and camera capture for a single N-frame pass.
    """
    finished_signal = pyqtSignal(object, object, object)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    frame_acquired_signal = pyqtSignal(np.ndarray, float, int)

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
        self.last_proc_time = 0.0  # <--- Added for tracking processing time

    def run(self):
        try:
            self._run_scan()
        except Exception as exc:
            self.error_signal.emit(str(exc))

    def _run_scan(self):
        self.progress_signal.emit("Starting acquisition…")

        # Use precise dt step to cover exactly one period minus one step 
        dv = self.period_v / self.n_frames
        
        h = self.camera_ctrl.image_height
        w = self.camera_ctrl.image_width
        image_stack = np.zeros((self.n_frames, h, w), dtype=np.float32)

        frames_acquired = 0

        # Run until we have successfully captured exactly n_frames
        while frames_acquired < self.n_frames and self.is_running:
            v = self.scan_v_start + (frames_acquired * dv)

            if not self.piezo_ctrl.set_voltage(v):
                self.error_signal.emit(f"Piezo failed to move to {v:.3f} V.")
                return

            time.sleep(self.settling_time)

            self.camera_ctrl.camera.issue_software_trigger()
            frame = self.camera_ctrl.camera.get_pending_frame_or_null()

            if frame is not None:
                # Frame successfully captured, store it and advance the counter
                img = np.copy(frame.image_buffer).reshape(h, w)
                image_stack[frames_acquired] = img
                self.frame_acquired_signal.emit(img, v, frames_acquired)
                
                frames_acquired += 1
                self.progress_signal.emit(f"Frame {frames_acquired} / {self.n_frames}")
            else:
                # Dropped frame: fail silently, wait 10ms, and let the while loop retry
                time.sleep(0.01)

        if not self.is_running:
            self.progress_signal.emit("Acquisition aborted.")
            return

        self.piezo_ctrl.set_voltage(self.scan_v_start)
        self.progress_signal.emit("Computing Fourier transform…")

        # --- Added Timing Section ---
        t_start = time.perf_counter()
        vis, contrast, phase = self.camera_ctrl.process_quantum_image(image_stack)
        t_end = time.perf_counter()
        
        self.last_proc_time = t_end - t_start
        # ----------------------------
        self.finished_signal.emit(vis, contrast, phase)


# ---------------------------------------------------------------------------
# Background Thread 2: Raw Live Feed
# ---------------------------------------------------------------------------

class LiveFeedWorker(QThread):
    """
    Continuously polls the camera buffer for raw frames (no piezo movement).
    """
    frame_ready_signal = pyqtSignal(np.ndarray)
    error_signal = pyqtSignal(str)

    def __init__(self, camera_ctrl: CameraController):
        super().__init__()
        self.camera_ctrl = camera_ctrl
        self.is_running = True

    def run(self):
        cam = self.camera_ctrl.camera
        w = self.camera_ctrl.image_width
        h = self.camera_ctrl.image_height

        while self.is_running:
            try:
                frame = cam.get_pending_frame_or_null()
                if frame is not None:
                    img = np.copy(frame.image_buffer).reshape(h, w)
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
    Continuously steps the piezo upwards, pushing limits.
    Computes the FFT on every new frame to push live Visibility, Contrast, and Phase.
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
        cam = self.camera_ctrl.camera
        w = self.camera_ctrl.image_width
        h = self.camera_ctrl.image_height

        rolling_buffer = np.zeros((self.n_frames, h, w), dtype=np.float32)
        
        dv = self.period_v / self.n_frames
        total_frames_acquired = 0
        valid_frames_in_buffer = 0

        while self.is_running:
            try:
                current_v = self.scan_v_start + (total_frames_acquired * dv)

                if current_v > self.piezo_ctrl.MAX_VOLTAGE:
                    total_frames_acquired = 0
                    valid_frames_in_buffer = 0 
                    current_v = self.scan_v_start

                if not self.piezo_ctrl.set_voltage(current_v):
                    self.error_signal.emit(f"Piezo failed at {current_v:.3f} V.")
                    break
                
                if total_frames_acquired == 0 and valid_frames_in_buffer == 0:
                    time.sleep(self.settling_time + 0.1) 
                else:
                    time.sleep(self.settling_time)

                cam.issue_software_trigger()
                frame = cam.get_pending_frame_or_null()

                if frame is not None:
                    img = np.copy(frame.image_buffer).reshape(h, w)
                    self.frame_acquired_signal.emit(img, current_v, total_frames_acquired)

                    step_index = valid_frames_in_buffer % self.n_frames
                    rolling_buffer[step_index] = img

                    valid_frames_in_buffer += 1
                    total_frames_acquired += 1

                    if valid_frames_in_buffer >= self.n_frames:
                        ordered_stack = np.roll(rolling_buffer, shift=-(step_index + 1), axis=0)
                        vis, contrast, phase = self.camera_ctrl.process_quantum_image(ordered_stack)
                        self.maps_ready_signal.emit(vis, contrast, phase)

            except Exception as exc:
                if self.is_running:
                    self.error_signal.emit(str(exc))
                break


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class QIUP_APP(QMainWindow):

    _DEFAULT_SCAN_V_START = 0.0
    _DEFAULT_SCAN_V_END = 3.9   
    _DEFAULT_SETTLING_MS = 10      

    def __init__(self):
        super().__init__()

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.logo_path = os.path.join(self.base_dir, "logo", "Logo_ANT.png")

        self.setWindowTitle("NANO: QIUP Dashboard")
        self.setGeometry(50, 50, 1600, 950)
        self.setWindowIcon(QIcon(self.logo_path))

        self.piezo: PiezoController | None = None
        self.camera: CameraController | None = None
        
        self.acq_worker: AcquisitionWorker | None = None
        self.live_worker: LiveFeedWorker | None = None
        self.live_proc_worker: LiveProcessingWorker | None = None

        self.voltages_data: list[float] = []
        self.intensities_data: list[float] = []

        self._setup_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.connect_action = QAction("Connect Hardware", self)
        self.connect_action.triggered.connect(self._connect_hardware)
        toolbar.addAction(self.connect_action)

        self.disconnect_action = QAction("Disconnect", self)
        self.disconnect_action.triggered.connect(self._disconnect_hardware)
        self.disconnect_action.setEnabled(False)
        toolbar.addAction(self.disconnect_action)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        root.addLayout(self._build_left_panel(), stretch=1)
        root.addLayout(self._build_right_panel(), stretch=5)

        self.statusBar().showMessage("Ready")

    def _build_left_panel(self) -> QVBoxLayout:
        panel = QVBoxLayout()
        panel.setContentsMargins(10, 10, 20, 10)

        # --- Camera settings ---
        cam_group = QGroupBox("CMOS Settings")
        cam_layout = QFormLayout()

        self.exposure_spin = QSpinBox()
        self.exposure_spin.setRange(1, 5000)
        self.exposure_spin.setValue(200)
        self.exposure_spin.setSuffix(" ms")

        self.gain_spin = QSpinBox()
        self.gain_spin.setRange(0, 48)
        self.gain_spin.setValue(35)
        self.gain_spin.setSuffix(" dB")

        cam_layout.addRow("Exposure:", self.exposure_spin)
        cam_layout.addRow("Gain:", self.gain_spin)
        cam_group.setLayout(cam_layout)
        panel.addWidget(cam_group)

        # --- ROI Settings ---
        roi_group = QGroupBox("Plot ROI Settings")
        roi_layout = QFormLayout()

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

        roi_layout.addRow("Center X:", self.roi_x_spin)
        roi_layout.addRow("Center Y:", self.roi_y_spin)
        roi_layout.addRow("Box Size:", self.roi_size_spin)
        
        hint_lbl = QLabel("(You can also click on the preview image to select ROI)")
        hint_lbl.setStyleSheet("font-size: 10px; color: #888888;")
        roi_layout.addRow(hint_lbl)
        
        roi_group.setLayout(roi_layout)
        panel.addWidget(roi_group)

        # --- Standard Acquisition Settings ---
        acq_group = QGroupBox("Standard Acquisition")
        acq_layout = QFormLayout()

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(3, 1000)
        self.frames_spin.setValue(8)

        self.scan_end_spin = QDoubleSpinBox()
        self.scan_end_spin.setRange(0.01, PiezoController.MAX_VOLTAGE)
        self.scan_end_spin.setDecimals(2)
        self.scan_end_spin.setSingleStep(0.25)
        self.scan_end_spin.setValue(self._DEFAULT_SCAN_V_END)
        self.scan_end_spin.setSuffix(" V")

        self.settling_spin = QSpinBox()
        self.settling_spin.setRange(1, 1000)
        self.settling_spin.setValue(self._DEFAULT_SETTLING_MS)
        self.settling_spin.setSuffix(" ms")

        acq_layout.addRow("Frames (N):", self.frames_spin)
        acq_layout.addRow("Fringe Period (V):", self.scan_end_spin)
        acq_layout.addRow("Settling time:", self.settling_spin)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Run Acquisition")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._run_acquisition)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.clicked.connect(self._reset_system)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.reset_btn)
        acq_layout.addRow(btn_row)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: #888888; font-style: italic;")
        acq_layout.addRow(self.status_label)

        acq_group.setLayout(acq_layout)
        panel.addWidget(acq_group)

        # --- Live Quantum Processing Settings ---
        live_proc_group = QGroupBox("Live Quantum Processing")
        live_proc_layout = QFormLayout()

        self.live_frames_spin = QSpinBox()
        self.live_frames_spin.setRange(3, 200)
        self.live_frames_spin.setValue(8)
        self.live_frames_spin.setToolTip("Dynamic circular buffer size. Determines phase shift precision.")

        self.live_proc_btn = QPushButton("Start Live Processing")
        self.live_proc_btn.setMinimumHeight(40)
        self.live_proc_btn.setEnabled(False)
        self.live_proc_btn.clicked.connect(self._toggle_live_processing)

        live_proc_layout.addRow("Buffer Frames (N):", self.live_frames_spin)
        live_proc_layout.addRow(self.live_proc_btn)
        live_proc_group.setLayout(live_proc_layout)
        panel.addWidget(live_proc_group)

        panel.addStretch()
        return panel

    def _build_right_panel(self) -> QVBoxLayout:
        panel = QVBoxLayout()

        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)

        self.vis_img = self._make_image_label()
        self.contrast_img = self._make_image_label()
        self.phase_img = self._make_image_label()

        def make_group(title, widget):
            grp = QGroupBox(title)
            lay = QVBoxLayout()
            lay.setAlignment(Qt.AlignCenter)
            lay.addWidget(widget)
            grp.setLayout(lay)
            grp.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            return grp

        # Top Row
        grid_layout.addWidget(make_group("1. Visibility Map", self.vis_img), 0, 0)
        grid_layout.addWidget(make_group("2. Contrast Map", self.contrast_img), 0, 1)
        grid_layout.addWidget(make_group("3. Phase Map", self.phase_img), 0, 2)

        # Bottom Left: Preview (Using Custom ClickableLabel)
        preview_group = QGroupBox("Current Frame Preview")
        preview_layout = QVBoxLayout()
        preview_layout.setAlignment(Qt.AlignCenter)
        
        self.raw_preview = ClickableLabel("Waiting for trigger…")
        self.raw_preview.setFixedSize(400, 400)
        self.raw_preview.setAlignment(Qt.AlignCenter)
        self.raw_preview.setProperty("is_image", True)
        
        # Connect the click event to our new mapping function
        self.raw_preview.clicked.connect(self._on_preview_clicked)
        
        preview_layout.addWidget(self.raw_preview)

        self.live_btn = QPushButton("Start Raw Feed")
        self.live_btn.setMinimumHeight(40)
        self.live_btn.setEnabled(False)
        self.live_btn.clicked.connect(self._toggle_live_feed)
        preview_layout.addWidget(self.live_btn)

        preview_group.setLayout(preview_layout)
        preview_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        grid_layout.addWidget(preview_group, 1, 0)

        # Bottom Right: Graph
        cycle_group = QGroupBox("ROI Intensity vs Piezo Voltage")
        cycle_layout = QVBoxLayout()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(400) 
        
        cycle_layout.addWidget(self.canvas)
        cycle_group.setLayout(cycle_layout)
        cycle_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        grid_layout.addWidget(cycle_group, 1, 1, 1, 2)

        h_center = QHBoxLayout()
        h_center.addStretch()
        h_center.addLayout(grid_layout)
        h_center.addStretch()

        panel.addStretch()
        panel.addLayout(h_center)
        panel.addStretch()

        return panel

    @staticmethod
    def _make_image_label() -> QLabel:
        lbl = QLabel("No Data")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedSize(400, 400)
        lbl.setProperty("is_image", True)
        return lbl

    # ------------------------------------------------------------------
    # ROI Click Handling
    # ------------------------------------------------------------------
    
    def _on_preview_clicked(self, label_x: int, label_y: int):
        """
        Triggered when the user clicks on the raw frame preview label.
        Maps the 400x400 coordinate back to the true sensor resolution and
        updates the ROI spinboxes.
        """
        if not self.camera or not self.raw_preview.pixmap():
            return

        pm = self.raw_preview.pixmap()
        pm_w, pm_h = pm.width(), pm.height()

        label_w, label_h = self.raw_preview.width(), self.raw_preview.height()

        # Calculate the black bar offsets (due to Qt.KeepAspectRatio)
        offset_x = (label_w - pm_w) / 2.0
        offset_y = (label_h - pm_h) / 2.0

        # Click coordinate relative strictly to the image pixels
        pixmap_x = label_x - offset_x
        pixmap_y = label_y - offset_y

        # If they clicked on the black bars, ignore it
        if pixmap_x < 0 or pixmap_x > pm_w or pixmap_y < 0 or pixmap_y > pm_h:
            return

        # Map back to camera resolution
        cam_w = self.camera.image_width
        cam_h = self.camera.image_height

        cam_x = int((pixmap_x / pm_w) * cam_w)
        cam_y = int((pixmap_y / pm_h) * cam_h)

        # Automatically update the spinboxes (which will instantly move the red box)
        self.roi_x_spin.setValue(cam_x)
        self.roi_y_spin.setValue(cam_y)


    # ------------------------------------------------------------------
    # Hardware lifecycle
    # ------------------------------------------------------------------

    def _connect_hardware(self):
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
            # Update ROI to center
            w = self.camera.image_width
            h = self.camera.image_height
            self.roi_x_spin.setRange(0, w - 1)
            self.roi_y_spin.setRange(0, h - 1)
            self.roi_x_spin.setValue(w // 2)
            self.roi_y_spin.setValue(h // 2)

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
            errors = []
            if not p_ok: errors.append(f"Piezo: {p_msg}")
            if not c_ok: errors.append("Camera: failed to initialise.")
            QMessageBox.critical(self, "Connection Error", "Failed to connect:\n" + "\n".join(errors))
            self.status_label.setText("Connection failed.")
            self.statusBar().showMessage("Hardware connection failed.")

    def _disconnect_hardware(self):
        reply = QMessageBox.question(
            self, "Confirm Disconnect", "Disconnect all hardware?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes: return

        self._stop_live_worker()
        self._stop_live_proc_worker()
        self._stop_worker()
        
        if self.camera: self.camera.disconnect()
        if self.piezo: self.piezo.disconnect()

        self.connect_action.setEnabled(True)
        self.disconnect_action.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.live_proc_btn.setEnabled(False)
        self.status_label.setText("Idle")
        self.statusBar().showMessage("Hardware disconnected.")

    # ------------------------------------------------------------------
    # Acquisition and Live Feeds
    # ------------------------------------------------------------------

    def _run_acquisition(self):
        self.start_btn.setEnabled(False)
        self.live_btn.setEnabled(False)
        self.live_proc_btn.setEnabled(False)
        self.start_btn.setText("Acquiring…")
        
        self.voltages_data.clear()
        self.intensities_data.clear()
        self.ax.clear()

        exp_ms = self.exposure_spin.value()
        self.camera.camera.exposure_time_us = exp_ms * 1000
        self.camera.camera.gain = self.gain_spin.value() * 10
        self.camera.camera.image_poll_timeout_ms = exp_ms + 500

        period_v = self.scan_end_spin.value() - self._DEFAULT_SCAN_V_START

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
        if self.live_proc_worker is not None and self.live_proc_worker.isRunning():
            self._stop_live_proc_worker()
            self.live_proc_btn.setText("Start Live Processing")
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.status_label.setText("Live processing stopped. Idle.")
        else:
            self.start_btn.setEnabled(False)
            self.live_btn.setEnabled(False)
            self.live_proc_btn.setText("Stop Live Processing")
            self.status_label.setText("Live processing running...")

            self.voltages_data.clear()
            self.intensities_data.clear()
            self.ax.clear()

            exp_ms = self.exposure_spin.value()
            self.camera.camera.exposure_time_us = exp_ms * 1000
            self.camera.camera.gain = self.gain_spin.value() * 10
            self.camera.camera.image_poll_timeout_ms = exp_ms + 500

            self.camera.set_single_frame_mode()
            period_v = self.scan_end_spin.value() - self._DEFAULT_SCAN_V_START

            self.live_proc_worker = LiveProcessingWorker(
                camera_ctrl=self.camera,
                piezo_ctrl=self.piezo,
                n_frames=self.live_frames_spin.value(),
                scan_v_start=self._DEFAULT_SCAN_V_START,
                period_v=period_v,
                settling_time=self.settling_spin.value() / 1000.0,
            )
            self.live_proc_worker.frame_acquired_signal.connect(self._update_preview_and_plot)
            self.live_proc_worker.maps_ready_signal.connect(self._display_maps)
            self.live_proc_worker.error_signal.connect(self._on_error)
            self.live_proc_worker.start()

    def _toggle_live_feed(self):
        if self.live_worker is not None and self.live_worker.isRunning():
            self._stop_live_worker()
            self.live_btn.setText("Start Raw Feed")
            self.start_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)
            self.status_label.setText("Raw feed stopped. Idle.")
        else:
            self.start_btn.setEnabled(False)
            self.live_proc_btn.setEnabled(False)
            self.live_btn.setText("Stop Raw Feed")
            self.status_label.setText("Raw feed running...")

            exp_ms = self.exposure_spin.value()
            self.camera.camera.exposure_time_us = exp_ms * 1000
            self.camera.camera.gain = self.gain_spin.value() * 10
            self.camera.camera.image_poll_timeout_ms = exp_ms + 500

            self.camera.set_continuous_mode()

            self.live_worker = LiveFeedWorker(self.camera)
            self.live_worker.frame_ready_signal.connect(self._update_live_preview)
            self.live_worker.error_signal.connect(self._on_error)
            self.live_worker.start()

    def _on_error(self, msg: str):
        self._stop_live_worker()
        self._stop_live_proc_worker()
        
        self.live_btn.setText("Start Raw Feed")
        self.live_proc_btn.setText("Start Live Processing")
        self.start_btn.setText("Run Acquisition")
        
        self.start_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.live_proc_btn.setEnabled(True)
        QMessageBox.warning(self, "Hardware Error", msg)

    def _reset_system(self):
        self._stop_live_worker()
        self._stop_live_proc_worker()
        self._stop_worker()

        if self.piezo: self.piezo.set_voltage(0.0)

        self.voltages_data.clear()
        self.intensities_data.clear()
        self.ax.clear()
        self.canvas.draw()

        for widget in (self.raw_preview, self.vis_img, self.contrast_img, self.phase_img):
            widget.setPixmap(QPixmap())
            widget.setText("—")

        self.raw_preview.setText("System reset")
        self.status_label.setText("System reset. Idle.")
        self.statusBar().showMessage("System reset.")

        if self.piezo is not None and self.camera is not None:
            self.start_btn.setEnabled(True)
            self.live_btn.setEnabled(True)
            self.live_proc_btn.setEnabled(True)
            self.start_btn.setText("Run Acquisition")
            self.live_btn.setText("Start Raw Feed")
            self.live_proc_btn.setText("Start Live Processing")

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

    # ------------------------------------------------------------------
    # UI update slots
    # ------------------------------------------------------------------

    def _update_live_preview(self, gray_img: np.ndarray):
        norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        s = self.roi_size_spin.value() // 2
        cv2.rectangle(rgb, (x - s, y - s), (x + s, y + s), (255, 0, 0), 4)

        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.raw_preview.setPixmap(
            QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _update_preview_and_plot(self, gray_img: np.ndarray, v: float, idx: int):
        self._update_live_preview(gray_img)

        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        s = self.roi_size_spin.value() // 2
        h_img, w_img = gray_img.shape

        y_min, y_max = max(0, y - s), min(h_img, y + s + 1)
        x_min, x_max = max(0, x - s), min(w_img, x + s + 1)

        roi_data = gray_img[y_min:y_max, x_min:x_max]
        mean_intensity = float(np.mean(roi_data))

        self.voltages_data.append(v)
        self.intensities_data.append(mean_intensity)

        if len(self.voltages_data) > 200:
            self.voltages_data.pop(0)
            self.intensities_data.pop(0)

        self.ax.clear()
        self.ax.grid(True, color="#2d2d30", linestyle="--", linewidth=0.5, zorder=0)
        self.ax.scatter(
            self.voltages_data, self.intensities_data,
            color="#00ff00", s=50, edgecolors="white", zorder=2,
        )
        self.ax.set_xlabel("Piezo Voltage (V)", fontsize=10, color="#d4d4d4", fontweight="bold")
        self.ax.set_ylabel("ROI Mean Intensity", fontsize=10, color="#d4d4d4", fontweight="bold")
        self.ax.tick_params(colors="#d4d4d4", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#3f3f46")
        self.canvas.draw()

    def _display_maps(self, vis: np.ndarray, contrast: np.ndarray, phase: np.ndarray):
        self.vis_img.setPixmap(self._cv_to_pixmap(vis))
        self.contrast_img.setPixmap(self._cv_to_pixmap(contrast))
        self.phase_img.setPixmap(self._cv_to_pixmap(phase))

    # --- Updated to display timing info ---
    def _on_acquisition_complete(self):
        proc_time = getattr(self.acq_worker, 'last_proc_time', 0.0)
        msg = f"Acquisition complete. Processing time: {proc_time:.4f} s"
        
        self.status_label.setText(msg)
        self.statusBar().showMessage(msg)
        self.start_btn.setEnabled(True)
        self.live_btn.setEnabled(True)
        self.live_proc_btn.setEnabled(True)
        self.start_btn.setText("Run Acquisition")
    # --------------------------------------

    @staticmethod
    def _cv_to_pixmap(cv_img: np.ndarray) -> QPixmap:
        h, w, ch = cv_img.shape
        qimg = QImage(cv_img.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # ------------------------------------------------------------------
    # Theming
    # ------------------------------------------------------------------

    def _apply_theme(self):
        bg, fg, border, img_bg = "#1e1e1e", "#d4d4d4", "#3f3f46", "#121212"
        btn_bg, bar_bg = "#333337", "#2d2d30"

        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {bg}; color: {fg};
                font-family: Segoe UI, Arial;
            }}
            QToolBar {{
                background-color: {bar_bg};
                border-bottom: 1px solid {border}; padding: 5px;
            }}
            QGroupBox {{
                border: 1px solid {border}; margin-top: 15px;
                font-weight: bold; border-radius: 4px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 10px; padding: 0 3px;
            }}
            QPushButton {{
                background-color: {btn_bg}; color: white; padding: 8px;
                border: 1px solid {border}; border-radius: 4px;
            }}
            QPushButton:hover  {{ background-color: #007acc; }}
            QPushButton:disabled {{ background-color: #444; color: #888; }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {img_bg}; color: {fg};
                border: 1px solid {border}; padding: 4px;
            }}
            QLabel[is_image="true"] {{
                background-color: {img_bg};
                border: 2px dashed {border}; color: #555;
            }}
            QStatusBar {{
                background-color: {bar_bg}; color: {fg};
                border-top: 1px solid {border};
            }}
        """)

        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(img_bg)
        self.ax.tick_params(colors=fg)
        for spine in self.ax.spines.values():
            spine.set_color(border)
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Window close
    # ------------------------------------------------------------------

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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "nano_qiup.app.1_0"
        )
    except AttributeError:
        pass

    app = QApplication(sys.argv)
    window = QIUP_APP()
    app.setWindowIcon(QIcon(window.logo_path))
    window.show()
    sys.exit(app.exec_())