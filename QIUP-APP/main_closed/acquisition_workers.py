import time
import numpy as np
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from piezo_control_closed import PiezoController
from camera_control import CameraController


# ---------------------------------------------------------------------------
# Background Thread 1: Single Acquisition
# ---------------------------------------------------------------------------

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
        start_um: float,
        step_um: float,
        settling_time: float,
        scale_factors: dict
    ):
        super().__init__()
        self.camera_ctrl = camera_ctrl
        self.piezo_ctrl = piezo_ctrl
        self.n_frames = n_frames
        self.start_um = start_um
        self.step_um = step_um
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
        h = self.camera_ctrl.image_height
        w = self.camera_ctrl.image_width
        image_stack = np.zeros((self.n_frames, h, w), dtype=np.float32)
        frames_acquired = 0

        while frames_acquired < self.n_frames and self.is_running:
            target_um = self.start_um + (frames_acquired * self.step_um)
            
            # Send hardware target
            if not self.piezo_ctrl.move_to_um(target_um):
                self.error_signal.emit(f"Piezo failed to move to {target_um:.3f} µm.")
                return
            
            # Wait exactly the user-defined settling time
            time.sleep(self.settling_time)
            
            # ACCURACY: Instantly flush any old frames from the buffer
            while self.camera_ctrl.camera.get_pending_frame_or_null() is not None:
                pass
            
            # Issue ONE trigger
            self.camera_ctrl.camera.issue_software_trigger()
            
            # SPEED & ACCURACY: Tight polling loop (1ms) to grab the frame the instant the exposure finishes
            frame = None
            wait_start = time.time()
            while frame is None and self.is_running and (time.time() - wait_start) < 5.0:
                frame = self.camera_ctrl.camera.get_pending_frame_or_null()
                if frame is None:
                    time.sleep(0.001)  # 1ms tight loop for maximum speed
            
            if frame is not None:
                img = np.copy(frame.image_buffer).reshape(h, w)
                img = cv2.flip(img, 0)
                
                if self.camera_ctrl.use_moving_average:
                    k = self.camera_ctrl.ma_kernel_size
                    k = k if k % 2 != 0 else k + 1
                    img = cv2.blur(img, (k, k))
                
                image_stack[frames_acquired] = img
                
                # Fetch actual displacement for hardware logging (Speed)
                actual_um = self.piezo_ctrl.get_displacement()
                
                # GUARANTEE EQUAL DISTANCES: Always use the exact target step for FFT processing (Accuracy)
                pos_to_record = target_um
                
                self.frame_acquired_signal.emit(img, pos_to_record, frames_acquired)
                frames_acquired += 1
                self.progress_signal.emit(f"Frame {frames_acquired} / {self.n_frames}")
            else:
                if self.is_running:
                    self.error_signal.emit("Camera timeout: Failed to receive frame after trigger.")
                    return

        if not self.is_running:
            self.progress_signal.emit("Acquisition aborted.")
            return

        # Return to start
        self.piezo_ctrl.move_to_um(self.start_um)
        
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
        start_um: float,
        step_um: float,
        settling_time: float,
        scale_factors: dict
    ):
        super().__init__()
        self.camera_ctrl = camera_ctrl
        self.piezo_ctrl = piezo_ctrl
        self.n_frames = n_frames
        self.start_um = start_um
        self.step_um = step_um
        self.settling_time = settling_time
        self.scale_factors = scale_factors
        self.is_running = True

    def run(self):
        cam = self.camera_ctrl.camera
        w = self.camera_ctrl.image_width
        h = self.camera_ctrl.image_height
        
        voltage_buffer = np.zeros((self.n_frames, h, w), dtype=np.float32)
        total_frames_acquired = 0

        while self.is_running:
            try:
                step_index = total_frames_acquired % self.n_frames
                target_um = self.start_um + (step_index * self.step_um)
                
                if not self.piezo_ctrl.move_to_um(target_um):
                    self.error_signal.emit(f"Piezo failed at {target_um:.3f} µm.")
                    break
                
                # Wait exactly the user-defined settling time
                time.sleep(self.settling_time)
                
                # ACCURACY: Instantly flush any old frames from the buffer
                while cam.get_pending_frame_or_null() is not None:
                    pass
                
                # Issue ONE trigger
                cam.issue_software_trigger()
                
                # SPEED & ACCURACY: Tight polling loop (1ms)
                frame = None
                wait_start = time.time()
                while frame is None and self.is_running and (time.time() - wait_start) < 5.0:
                    frame = cam.get_pending_frame_or_null()
                    if frame is None:
                        time.sleep(0.001)
                
                if frame is not None:
                    img = np.copy(frame.image_buffer).reshape(h, w)
                    img = cv2.flip(img, 0)
                    
                    if self.camera_ctrl.use_moving_average:
                        k = self.camera_ctrl.ma_kernel_size
                        k = k if k % 2 != 0 else k + 1
                        img = cv2.blur(img, (k, k))
                    
                    # Fetch actual displacement for potential tracking (Speed)
                    actual_um = self.piezo_ctrl.get_displacement()
                    
                    # GUARANTEE EQUAL DISTANCES: Always use the exact target step for FFT processing (Accuracy)
                    pos_to_record = target_um
                    
                    self.frame_acquired_signal.emit(img, pos_to_record, total_frames_acquired)
                    
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