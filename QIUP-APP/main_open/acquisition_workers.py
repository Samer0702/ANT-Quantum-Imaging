import time
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from piezo_control_open import PiezoController
from camera_control import CameraController


class SingleAcquisitionWorker(QThread):
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
        scale_factors: dict
    ):
        super().__init__()
        self.camera_ctrl = camera_ctrl
        self.piezo_ctrl = piezo_ctrl
        self.n_frames = n_frames
        self.scan_v_start = scan_v_start
        self.period_v = period_v
        self.settling_time = settling_time
        
        self.scale_factors = scale_factors
        self.is_running = True
        self.last_proc_time = 0.0

    def run(self):
        try:
            self._run_scan()
        except Exception as exc:
            self.error_signal.emit(str(exc))

    def _run_scan(self):
        self.progress_signal.emit("Starting acquisition…")
        dv = self.period_v / self.n_frames
        h = self.camera_ctrl.image_height
        w = self.camera_ctrl.image_width
        image_stack = np.zeros((self.n_frames, h, w), dtype=np.float32)
        frames_acquired = 0

        while frames_acquired < self.n_frames and self.is_running:
            v = self.scan_v_start + (frames_acquired * dv)
            if not self.piezo_ctrl.set_voltage(v):
                self.error_signal.emit(f"Piezo failed to move to {v:.3f} V.")
                return
            
            time.sleep(self.settling_time)
            
            # Get displacement reading if available
            displacement = self.piezo_ctrl.get_displacement()
            if displacement is None:
                # If no strain gauge, fall back to voltage
                display_value = v
            else:
                display_value = displacement
            
            self.camera_ctrl.camera.issue_software_trigger()
            frame = self.camera_ctrl.camera.get_pending_frame_or_null()
            
            if frame is not None:
                img = np.copy(frame.image_buffer).reshape(h, w)
                img = cv2.flip(img, 0)
                
                if self.camera_ctrl.use_moving_average:
                    k = self.camera_ctrl.ma_kernel_size
                    k = k if k % 2 != 0 else k + 1
                    img = cv2.blur(img, (k, k))
                
                image_stack[frames_acquired] = img
                self.frame_acquired_signal.emit(img, display_value, frames_acquired)
                frames_acquired += 1
                self.progress_signal.emit(f"Frame {frames_acquired} / {self.n_frames}")
            else:
                time.sleep(0.01)

        if not self.is_running:
            self.progress_signal.emit("Acquisition aborted.")
            return

        self.piezo_ctrl.set_voltage(self.scan_v_start)
        
        self.progress_signal.emit("Computing Fourier transform…")
        t_start = time.perf_counter()
        
        vis, contrast, phase = self.camera_ctrl.process_quantum_image(
            image_stack, scale_factors=self.scale_factors
        )
        
        t_end = time.perf_counter()
        self.last_proc_time = t_end - t_start
        self.finished_signal.emit(vis, contrast, phase)


# ---------------------------------------------------------------------------
# Background Thread 2: Raw Live Feed
# ---------------------------------------------------------------------------

class LiveFeedWorker(QThread):
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
                    img = cv2.flip(img, 0)
                    
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
# Background Thread 3: Live Quantum Processing
# ---------------------------------------------------------------------------

class LiveProcessingWorker(QThread):
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
        settling_time: float,
        scale_factors: dict
    ):
        super().__init__()
        self.camera_ctrl = camera_ctrl
        self.piezo_ctrl = piezo_ctrl
        self.n_frames = n_frames
        self.scan_v_start = scan_v_start
        self.period_v = period_v
        self.settling_time = settling_time
        self.scale_factors = scale_factors
        self.is_running = True

    def run(self):
        cam = self.camera_ctrl.camera
        w = self.camera_ctrl.image_width
        h = self.camera_ctrl.image_height
        
        voltage_buffer = np.zeros((self.n_frames, h, w), dtype=np.float32)
        dv = self.period_v / self.n_frames
        total_frames_acquired = 0

        while self.is_running:
            try:
                step_index = total_frames_acquired % self.n_frames
                current_v = self.scan_v_start + (step_index * dv)
                
                if not self.piezo_ctrl.set_voltage(current_v):
                    self.error_signal.emit(f"Piezo failed at {current_v:.3f} V.")
                    break
                
                time.sleep(self.settling_time)
                
                # Get displacement reading if available
                displacement = self.piezo_ctrl.get_displacement()
                if displacement is None:
                    # If no strain gauge, fall back to voltage
                    display_value = current_v
                else:
                    display_value = displacement
                
                cam.issue_software_trigger()
                frame = cam.get_pending_frame_or_null()
                
                if frame is not None:
                    img = np.copy(frame.image_buffer).reshape(h, w)
                    img = cv2.flip(img, 0)
                    
                    if self.camera_ctrl.use_moving_average:
                        k = self.camera_ctrl.ma_kernel_size
                        k = k if k % 2 != 0 else k + 1
                        img = cv2.blur(img, (k, k))
                    
                    self.frame_acquired_signal.emit(img, display_value, total_frames_acquired)
                    
                    voltage_buffer[step_index] = img
                    total_frames_acquired += 1
                    
                    if total_frames_acquired >= self.n_frames:
                        vis, contrast, phase = self.camera_ctrl.process_quantum_image(
                            voltage_buffer, scale_factors=self.scale_factors
                        )
                        self.maps_ready_signal.emit(vis, contrast, phase)
                        
            except Exception as exc:
                if self.is_running:
                    self.error_signal.emit(str(exc))
                break            