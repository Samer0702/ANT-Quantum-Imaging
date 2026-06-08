import numpy as np
import cv2
import pyfftw

try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
except ImportError:
    print("Thorlabs TSI SDK not found. Make sure it is installed.")


class CameraController:
    """
    Manages a Thorlabs CMOS camera via the TSI SDK and performs the
    pixel-wise FFT analysis required for QIUP (Quantum Imaging with
    Undetected Photons).
    """

    def __init__(self):
        self.sdk = None
        self.camera = None

        self.image_width = 0
        self.image_height = 0

        self._fft_plan = None
        self._fft_input = None
        self._fft_output = None
        self._planned_n_frames = 0
        
        self.use_moving_average = False
        self.ma_kernel_size = 3

        # --- Cache for dynamic rendering ---
        self.last_visibility = None
        self.last_contrast = None
        self.last_phase = None

        # Store absolute limits using theoretical bounds to prevent slider jitter
        self.data_limits = {
            'v_min': 0.0, 'v_max': 1.0,       # Visibility is strictly [0, 1]
            'c_min': 0.0, 'c_max': 1.0,       # Contrast max will expand dynamically
            'p_min': -np.pi, 'p_max': np.pi   # Phase is strictly [-pi, pi]
        }

    # ------------------------------------------------------------------
    # Hardware lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        try:
            from camera_windows_setup import configure_path
            configure_path()
        except ImportError:
            pass  

        self.sdk = TLCameraSDK()
        available = self.sdk.discover_available_cameras()

        if not available:
            print("No cameras detected. Check USB connection and power.")
            self.sdk.dispose()
            self.sdk = None
            return False

        self.camera = self.sdk.open_camera(available[0])

        self.camera.frames_per_trigger_zero_for_unlimited = 1
        self.camera.image_poll_timeout_ms = 200 
        self.camera.exposure_time_us = 200_000    
        self.camera.gain = 35 * 10                
        self.camera.arm(2)

        self.image_width = self.camera.image_width_pixels
        self.image_height = self.camera.image_height_pixels
        print(f"Camera connected: {self.image_width} x {self.image_height} px")
        return True

    def set_continuous_mode(self):
        if self.camera:
            self.camera.disarm()
            self.camera.frames_per_trigger_zero_for_unlimited = 0
            self.camera.arm(2)
            self.camera.issue_software_trigger() 

    def set_single_frame_mode(self):
        if self.camera:
            self.camera.disarm()
            self.camera.frames_per_trigger_zero_for_unlimited = 1
            self.camera.arm(2)

    def disconnect(self):
        if self.camera:
            self.camera.disarm()
            self.camera.dispose()
            self.camera = None
        if self.sdk:
            self.sdk.dispose()
            self.sdk = None
        print("Camera disconnected.")

    # ------------------------------------------------------------------
    # FFT planning 
    # ------------------------------------------------------------------

    def _prepare_fft_plan(self, n_frames: int):
        print(f"Building pyFFTW plan for {n_frames} frames...")
        self._fft_input = pyfftw.empty_aligned(
            (n_frames, self.image_height, self.image_width), dtype="complex64"
        )
        self._fft_output = pyfftw.empty_aligned(
            (n_frames, self.image_height, self.image_width), dtype="complex64"
        )
        self._fft_plan = pyfftw.FFTW(
            self._fft_input,
            self._fft_output,
            axes=(0,),
            direction="FFTW_FORWARD",
            flags=("FFTW_MEASURE",),
        )
        self._planned_n_frames = n_frames


    def _get_f1_bin(self, n_frames: int) -> int:
        """
        Return the FFT bin index that holds the interference fringe signal.
        """

        limit = n_frames // 2
        if limit <= 1:
            return 1

        mean_magnitudes = np.abs(self._fft_output[1:limit]).mean(axis=(1, 2))
        return int(mean_magnitudes.argmax()) + 1
    
    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_quantum_image(
        self, image_stack: np.ndarray, scale_factors: dict = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run FFT, cache physics parameters, and apply GUI slider scales 
        to theoretically stable boundaries to prevent visual jitter.
        """
        n_frames = image_stack.shape[0]

        if n_frames < 3:
            raise ValueError("At least 3 frames are required.")

        if n_frames != self._planned_n_frames:
            self._prepare_fft_plan(n_frames)

        self._fft_input[:] = image_stack
        self._fft_plan()

        F0 = np.abs(self._fft_output[0])
        F1_bin = self._get_f1_bin(n_frames)
        F1 = self._fft_output[F1_bin]

        # Calculate raw maps
        visibility = (2.0 * np.abs(F1)) / (F0 + 1e-10)
        contrast = 4.0 * np.abs(F1)
        phase = np.angle(F1)

        self.last_visibility = visibility
        self.last_contrast = contrast
        self.last_phase = phase

        # 1. Use Theoretical Limits to anchor the UI sliders stably
        self.data_limits['v_min'] = 0.0
        self.data_limits['v_max'] = 1.0
        self.data_limits['p_min'] = -np.pi
        self.data_limits['p_max'] = np.pi

        # 2. Contrast is open-ended. Use an expanding maximum so the scale
        # doesn't shrink when the signal drops, locking the slider mapping.
        current_c_max = float(np.max(contrast))
        if current_c_max > self.data_limits['c_max']:
            self.data_limits['c_max'] = current_c_max
        self.data_limits['c_min'] = 0.0

        # 3. Apply the user's slider percentages to determine ACTUAL render limits
        sf = scale_factors or {}
        
        v_min_render = self.data_limits['v_min'] + sf.get('v_min_pct', 0.0) * (self.data_limits['v_max'] - self.data_limits['v_min'])
        v_max_render = self.data_limits['v_min'] + sf.get('v_max_pct', 1.0) * (self.data_limits['v_max'] - self.data_limits['v_min'])

        c_min_render = self.data_limits['c_min'] + sf.get('c_min_pct', 0.0) * (self.data_limits['c_max'] - self.data_limits['c_min'])
        c_max_render = self.data_limits['c_min'] + sf.get('c_max_pct', 1.0) * (self.data_limits['c_max'] - self.data_limits['c_min'])

        p_min_render = self.data_limits['p_min'] + sf.get('p_min_pct', 0.0) * (self.data_limits['p_max'] - self.data_limits['p_min'])
        p_max_render = self.data_limits['p_min'] + sf.get('p_max_pct', 1.0) * (self.data_limits['p_max'] - self.data_limits['p_min'])

        # Pass the locked, user-adjusted limits to the renderer
        return self.render_colormaps(
            v_min=v_min_render, v_max=v_max_render, 
            c_min=c_min_render, c_max=c_max_render, 
            p_min=p_min_render, p_max=p_max_render
        )

    # ------------------------------------------------------------------
    # Scale Bar Addition
    # ------------------------------------------------------------------

    def _add_scale_bar(self, image, scale_length_px):
        """Draws a ruler-style white scale bar anchored at the bottom left."""
        h, w = image.shape[:2]
        
        padding_x = 30
        padding_y = 40
        tick_height = 12
        line_thickness = 2
        
        x_start = padding_x
        x_end = x_start + scale_length_px
        y_pos = h - padding_y
        
        cv2.line(image, (x_start, y_pos), (x_end, y_pos), (255, 255, 255), line_thickness)
        cv2.line(image, (x_start, y_pos), (x_start, y_pos - tick_height), (255, 255, 255), line_thickness)
        
        x_mid = x_start + (scale_length_px // 2)
        cv2.line(image, (x_mid, y_pos), (x_mid, y_pos - (tick_height // 2)), (255, 255, 255), line_thickness)
        cv2.line(image, (x_end, y_pos), (x_end, y_pos - tick_height), (255, 255, 255), line_thickness)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "1 mm"
        font_scale = 0.6
        text_thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, text_thickness)
        text_x = x_start + (scale_length_px // 2) - (text_width // 2)
        text_y = y_pos - tick_height - 8  
        
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
        
        return image

    # ------------------------------------------------------------------
    # Colormap rendering (Dynamic)
    # ------------------------------------------------------------------

    def render_colormaps(
        self, v_min, v_max, c_min, c_max, p_min, p_max
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        if self.last_visibility is None:
            return None, None, None

        # Protect against div by zero if user sets min and max to identical values
        v_range = v_max - v_min if v_max > v_min else 1e-5
        c_range = c_max - c_min if c_max > c_min else 1e-5
        p_range = p_max - p_min if p_max > p_min else 1e-5

        # 1. Visibility
        vis_norm = np.clip((self.last_visibility - v_min) / v_range, 0, 1)
        vis_8bit = (vis_norm * 255.0).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis_8bit, cv2.COLORMAP_VIRIDIS)

        # 2. Contrast
        con_norm = np.clip((self.last_contrast - c_min) / c_range, 0, 1)
        con_8bit = (con_norm * 255.0).astype(np.uint8)
        contrast_color = cv2.applyColorMap(con_8bit, cv2.COLORMAP_PLASMA)

        # 3. Phase 
        pha_norm = np.clip((self.last_phase - p_min) / p_range, 0, 1)
        phase_8bit = (pha_norm * 255.0).astype(np.uint8)
        phase_color = cv2.applyColorMap(phase_8bit, cv2.COLORMAP_TWILIGHT)

        # Convert BGR to RGB
        vis_color = cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB)
        contrast_color = cv2.cvtColor(contrast_color, cv2.COLOR_BGR2RGB)
        phase_color = cv2.cvtColor(phase_color, cv2.COLOR_BGR2RGB)

        # This strictly removes background noise below a 10% visibility 
        # threshold, regardless of the GUI slider position.
        vis_threshold = 0.07
        mask_3d = (self.last_visibility > vis_threshold)[..., np.newaxis]
        phase_color = np.where(mask_3d, phase_color, 0).astype(np.uint8)

        pixels_for_1_mm = 110  
        
        vis_color = self._add_scale_bar(vis_color, pixels_for_1_mm)
        contrast_color = self._add_scale_bar(contrast_color, pixels_for_1_mm)
        phase_color = self._add_scale_bar(phase_color, pixels_for_1_mm)
        
        # 4. Add Colorbars 
        vis_with_scale = self._add_colorbar(vis_color, cv2.COLORMAP_VIRIDIS, f"{v_min:.3f}", f"{v_max:.3f}", "Visibility")
        contrast_with_scale = self._add_colorbar(contrast_color, cv2.COLORMAP_PLASMA, f"{c_min:.1f}", f"{c_max:.1f}", "Contrast")
        phase_with_scale = self._add_colorbar(phase_color, cv2.COLORMAP_TWILIGHT, f"{p_min:.2f}", f"{p_max:.2f}", "Phase")

        return vis_with_scale, contrast_with_scale, phase_with_scale

    def _add_colorbar(self, image, colormap_type, label_min, label_max, title):
        h, w, _ = image.shape
        cb_width = 80  
        padding = 50   
        
        gradient = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
        gradient_strip = np.repeat(gradient, cb_width, axis=1)
        
        color_strip = cv2.applyColorMap(gradient_strip, colormap_type)
        color_strip = cv2.cvtColor(color_strip, cv2.COLOR_BGR2RGB)
        
        canvas = np.zeros((h, w + cb_width + padding, 3), dtype=np.uint8)
        canvas[:, :w] = image
        canvas[:, w:w+cb_width] = color_strip
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        outline_thickness = 3
        white = (255, 255, 255)
        black = (0, 0, 0)

        pos_max = (w + cb_width + 5, 20)
        pos_min = (w + cb_width + 5, h - 10)
        pos_title = (w + 5, 15)

        for pos, text in [(pos_max, label_max), (pos_min, label_min), (pos_title, title)]:
            cv2.putText(canvas, text, pos, font, font_scale, black, outline_thickness, cv2.LINE_AA)
            cv2.putText(canvas, text, pos, font, font_scale, white, thickness, cv2.LINE_AA)

        return canvas