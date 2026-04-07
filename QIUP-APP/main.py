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
    QToolBar, QAction, QSizePolicy, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap, QIcon

from piezo_control import PiezoController
from camera_control import CameraController


# ---------------------------------------------------------------------------
# Background acquisition thread
# ---------------------------------------------------------------------------

class AcquisitionWorker(QThread):
    """
    Drives the piezo scan and camera capture on a background thread so the
    GUI stays responsive.  Emits signals to push live data to the UI.

    Signals:
        progress_signal(str)              — status text for the status label.
        error_signal(str)                 — fatal error message; stops the scan.
        frame_acquired_signal(ndarray, float, int)
                                          — (raw frame, voltage, frame index)
                                            emitted after each successful capture.
        finished_signal(ndarray, ndarray, ndarray)
                                          — (vis, contrast, phase) RGB images
                                            emitted after the FFT is complete.
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
        scan_v_end: float,
        settling_time: float,
    ):
        super().__init__()
        self.camera_ctrl = camera_ctrl
        self.piezo_ctrl = piezo_ctrl
        self.n_frames = n_frames
        self.scan_v_start = scan_v_start
        self.scan_v_end = scan_v_end
        self.settling_time = settling_time

        # Set to False from the main thread to abort mid-scan cleanly.
        self.is_running = True

    def run(self):
        try:
            self._run_scan()
        except Exception as exc:
            self.error_signal.emit(str(exc))

    def _run_scan(self):
        self.progress_signal.emit("Starting acquisition…")

        voltages = np.linspace(self.scan_v_start, self.scan_v_end, self.n_frames)
        h = self.camera_ctrl.image_height
        w = self.camera_ctrl.image_width
        image_stack = np.zeros((self.n_frames, h, w), dtype=np.float32)

        for i, v in enumerate(voltages):
            if not self.is_running:
                self.progress_signal.emit("Acquisition aborted.")
                return

            if not self.piezo_ctrl.set_voltage(v):
                self.error_signal.emit(f"Piezo failed to move to {v:.3f} V.")
                return

            time.sleep(self.settling_time)

            self.camera_ctrl.camera.issue_software_trigger()
            frame = self.camera_ctrl.camera.get_pending_frame_or_null()

            if frame is None:
                self.error_signal.emit(f"Camera dropped frame {i} at {v:.3f} V.")
                return

            img = np.copy(frame.image_buffer).reshape(h, w)
            image_stack[i] = img
            self.frame_acquired_signal.emit(img, v, i)
            self.progress_signal.emit(f"Frame {i + 1} / {self.n_frames}")

        if not self.is_running:
            return

        # Return mirror to start position before the (blocking) FFT step.
        self.piezo_ctrl.set_voltage(self.scan_v_start)
        self.progress_signal.emit("Computing Fourier transform…")

        vis, contrast, phase = self.camera_ctrl.process_quantum_image(image_stack)
        self.finished_signal.emit(vis, contrast, phase)


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class QIUP_APP(QMainWindow):

    # Default scan parameters — users can edit scan_v_end in the UI.
    _DEFAULT_SCAN_V_START = 0.0
    _DEFAULT_SCAN_V_END = 3.4     # Tune to cover exactly one fringe period.
    _DEFAULT_SETTLING_MS = 10      # Piezo settling time in milliseconds.

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

        self.voltages_data: list[float] = []
        self.intensities_data: list[float] = []

        self._setup_ui()
        self._apply_theme()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        # Toolbar
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

        # Root layout
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
        self.gain_spin.setRange(0, 48)    # dB. Sent to SDK as value * 10 (tenths of dB).
        self.gain_spin.setValue(35)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setToolTip(
            "Gain in dB (0–48). Multiplied by 10 before being sent to the "
            "TSI SDK, which expects tenths of dB. Default 35 dB is intentionally "
            "high for low-light single-photon-level signals."
        )

        cam_layout.addRow("Exposure:", self.exposure_spin)
        cam_layout.addRow("Gain:", self.gain_spin)
        cam_group.setLayout(cam_layout)
        panel.addWidget(cam_group)

        # --- Acquisition / scan settings ---
        acq_group = QGroupBox("Acquisition")
        acq_layout = QFormLayout()

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(3, 1000)
        self.frames_spin.setValue(8)
        self.frames_spin.setToolTip(
            "Number of frames (N). Minimum 3 (Nyquist). "
            "Increase for smoother results but longer acquisition and processing time.")

        self.scan_end_spin = QDoubleSpinBox()
        self.scan_end_spin.setRange(0.01, PiezoController.MAX_VOLTAGE)
        self.scan_end_spin.setDecimals(2)
        self.scan_end_spin.setSingleStep(0.25)
        self.scan_end_spin.setValue(self._DEFAULT_SCAN_V_END)
        self.scan_end_spin.setSuffix(" V")
        self.scan_end_spin.setToolTip(
            "Scan end voltage. Calibrate so N frames span exactly one "
            "fringe period (≈ λ_idler / 2 of mirror displacement)."
        )

        self.settling_spin = QSpinBox()
        self.settling_spin.setRange(1, 1000)
        self.settling_spin.setValue(self._DEFAULT_SETTLING_MS)
        self.settling_spin.setSuffix(" ms")
        self.settling_spin.setToolTip(
            "Time to wait after each piezo step before triggering the camera."
        )

        acq_layout.addRow("Frames (N):", self.frames_spin)
        acq_layout.addRow("Scan end voltage:", self.scan_end_spin)
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

        # --- Live frame preview ---
        preview_group = QGroupBox("Current Frame Preview")
        preview_layout = QVBoxLayout()
        self.raw_preview = QLabel("Waiting for trigger…")
        self.raw_preview.setFixedSize(250, 250)
        self.raw_preview.setAlignment(Qt.AlignCenter)
        self.raw_preview.setProperty("is_image", True)
        preview_layout.addWidget(self.raw_preview)
        preview_group.setLayout(preview_layout)
        panel.addWidget(preview_group)

        panel.addStretch()
        return panel

    def _build_right_panel(self) -> QVBoxLayout:
        panel = QVBoxLayout()

        # Three result image views
        maps_row = QHBoxLayout()
        maps_row.setSpacing(15)

        self.vis_img = self._make_image_label()
        self.contrast_img = self._make_image_label()
        self.phase_img = self._make_image_label()

        for title, widget in (
            ("1. Visibility map", self.vis_img),
            ("2. Contrast map", self.contrast_img),
            ("3. Phase map", self.phase_img),
        ):
            col = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setFont(QFont("Arial", 13, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            col.addWidget(lbl)
            col.addWidget(widget)
            col.addStretch()
            maps_row.addLayout(col)

        panel.addLayout(maps_row)

        # Intensity vs voltage plot
        cycle_group = QGroupBox("Intensity vs Piezo Voltage")
        cycle_layout = QVBoxLayout()
        self.fig, self.ax = plt.subplots(figsize=(12, 3.5))
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(250)
        cycle_layout.addWidget(self.canvas)
        cycle_group.setLayout(cycle_layout)
        panel.addWidget(cycle_group)

        return panel

    @staticmethod
    def _make_image_label() -> QLabel:
        lbl = QLabel("No Data")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedSize(400, 400)
        lbl.setProperty("is_image", True)
        return lbl

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
            QMessageBox.critical(
                self, "Hardware Error",
                f"Unexpected error during setup:\n{exc}",
            )
            self.status_label.setText("Error.")
            self.statusBar().showMessage("Hardware connection error.")
            return

        if p_ok and c_ok:
            self.connect_action.setEnabled(False)
            self.disconnect_action.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.status_label.setText("Hardware connected.")
            self.statusBar().showMessage("Hardware connected successfully.")
            self._apply_theme()
            QMessageBox.information(self, "Success", "All hardware connected.")
        else:
            errors = []
            if not p_ok:
                errors.append(f"Piezo: {p_msg}")
            if not c_ok:
                errors.append("Camera: failed to initialise.")
            QMessageBox.critical(
                self, "Connection Error",
                "Failed to connect:\n" + "\n".join(errors),
            )
            self.status_label.setText("Connection failed.")
            self.statusBar().showMessage("Hardware connection failed.")

    def _disconnect_hardware(self):
        reply = QMessageBox.question(
            self, "Confirm Disconnect",
            "Disconnect all hardware?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._stop_worker()
        if self.camera:
            self.camera.disconnect()
        if self.piezo:
            self.piezo.disconnect()

        self.connect_action.setEnabled(True)
        self.disconnect_action.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.status_label.setText("Idle")
        self.statusBar().showMessage("Hardware disconnected.")

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def _run_acquisition(self):
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Acquiring…")
        self.voltages_data.clear()
        self.intensities_data.clear()
        self.ax.clear()

        # Apply GUI settings to camera — gain is already in TSI SDK units.
        self.camera.camera.exposure_time_us = self.exposure_spin.value() * 1000
        self.camera.camera.gain = self.gain_spin.value() * 10

        self.acq_worker = AcquisitionWorker(
            camera_ctrl=self.camera,
            piezo_ctrl=self.piezo,
            n_frames=self.frames_spin.value(),
            scan_v_start=self._DEFAULT_SCAN_V_START,
            scan_v_end=self.scan_end_spin.value(),
            settling_time=self.settling_spin.value() / 1000.0,
        )
        self.acq_worker.progress_signal.connect(self.status_label.setText)
        self.acq_worker.error_signal.connect(self._on_acquisition_error)
        self.acq_worker.frame_acquired_signal.connect(self._update_preview_and_plot)
        self.acq_worker.finished_signal.connect(self._display_results)
        self.acq_worker.start()

    def _on_acquisition_error(self, msg: str):
        QMessageBox.warning(self, "Acquisition Error", msg)
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Run Acquisition")

    def _reset_system(self):
        self._stop_worker()

        if self.piezo:
            self.piezo.set_voltage(0.0)

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
            self.start_btn.setText("Run Acquisition")

    def _stop_worker(self):
        """Signal the worker to stop and block until it exits."""
        if self.acq_worker and self.acq_worker.isRunning():
            self.acq_worker.is_running = False
            self.acq_worker.wait()

    # ------------------------------------------------------------------
    # UI update slots
    # ------------------------------------------------------------------

    def _update_preview_and_plot(self, gray_img: np.ndarray, v: float, idx: int):
        # Live thumbnail
        norm = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.raw_preview.setPixmap(
            QPixmap.fromImage(qimg).scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        # Intensity vs voltage scatter plot
        self.voltages_data.append(v)
        self.intensities_data.append(float(np.mean(gray_img)))

        self.ax.clear()
        self.ax.grid(True, color="#2d2d30", linestyle="--", linewidth=0.5, zorder=0)
        self.ax.scatter(
            self.voltages_data, self.intensities_data,
            color="#00ff00", s=50, edgecolors="white", zorder=2,
        )
        self.ax.set_xlabel("Piezo Voltage (V)", fontsize=10, color="#d4d4d4", fontweight="bold")
        self.ax.set_ylabel("Mean Intensity", fontsize=10, color="#d4d4d4", fontweight="bold")
        self.ax.tick_params(colors="#d4d4d4", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#3f3f46")
        self.canvas.draw()

    def _display_results(self, vis: np.ndarray, contrast: np.ndarray, phase: np.ndarray):
        self.vis_img.setPixmap(self._cv_to_pixmap(vis))
        self.contrast_img.setPixmap(self._cv_to_pixmap(contrast))
        self.phase_img.setPixmap(self._cv_to_pixmap(phase))
        self.status_label.setText("Acquisition complete.")
        self.statusBar().showMessage("Acquisition complete.")
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Run Acquisition")

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
        self._stop_worker()
        try:
            if self.camera:
                self.camera.disconnect()
            if self.piezo:
                self.piezo.disconnect()  # PiezoController.disconnect() already zeros voltage
        except Exception as exc:
            print(f"Error during shutdown: {exc}")

        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Set a custom AppUserModelID so Windows shows the correct taskbar icon.
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "nano_qiup.app.1_0"
        )
    except AttributeError:
        pass  # Not on Windows — harmless.

    app = QApplication(sys.argv)
    window = QIUP_APP()
    app.setWindowIcon(QIcon(window.logo_path))
    window.show()
    sys.exit(app.exec_())