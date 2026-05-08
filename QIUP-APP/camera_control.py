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

    Processing pipeline (per Pearce et al. 2023, Eqs. 1-3):
        F0  = DC Fourier component  (mean brightness)
        F1  = 1st-order component   (interference fringe amplitude/phase)

        Visibility = 2 * |F1| / F0          (Eq. 1)
        Contrast   = 4 * |F1|               (Eq. 2)
        Phase      = angle(F1)              (Eq. 3)

    The fringe frequency bin (F1_BIN) must correspond to exactly one full
    oscillation period across the N acquired frames.  If your scan does not
    cover exactly one period, set F1_BIN to the correct bin index, or enable
    AUTO_DETECT_BIN to let the code find the dominant frequency automatically.
    """

    # --- Configuration ---
    F1_BIN = 1              # FFT bin index of the interference fringe.
                            # Valid when scan covers exactly 1 fringe period.
    AUTO_DETECT_BIN = True # Set True to find the dominant bin automatically.
                            # Useful when the scan range is not perfectly
                            # calibrated to one period (slower, but safer).

    def __init__(self):
        self.sdk = None
        self.camera = None

        self.image_width = 0
        self.image_height = 0

        # PyFFTW plan — created lazily when the first image stack arrives.
        self._fft_plan = None
        self._fft_input = None
        self._fft_output = None
        self._planned_n_frames = 0
        
        # --- Moving Average properties ---
        self.use_moving_average = False
        self.ma_kernel_size = 3

    # ------------------------------------------------------------------
    # Hardware lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Initialise the TSI SDK, open the first available camera, and
        configure it for software-triggered single-frame acquisition.

        Returns:
            True on success, False if no camera is found.
        """
        try:
            from CMOS_windows_setup import configure_path
            configure_path()
        except ImportError:
            pass  # Running on Linux / without the path helper is fine.

        self.sdk = TLCameraSDK()
        available = self.sdk.discover_available_cameras()

        if not available:
            print("No cameras detected. Check USB connection and power.")
            self.sdk.dispose()
            self.sdk = None
            return False

        self.camera = self.sdk.open_camera(available[0])

        # Software-triggered, single-frame mode (required for step-scan).
        self.camera.frames_per_trigger_zero_for_unlimited = 1
        self.camera.image_poll_timeout_ms = 200 
        self.camera.exposure_time_us = 200_000    # 200 ms default
        self.camera.gain = 35 * 10                # ~3.5 dB (SDK units = tenths of dB)
        self.camera.arm(2)

        self.image_width = self.camera.image_width_pixels
        self.image_height = self.camera.image_height_pixels
        print(f"Camera connected: {self.image_width} x {self.image_height} px")
        return True

    def set_continuous_mode(self):
        """Switches the camera to continuous acquisition mode."""
        if self.camera:
            self.camera.disarm()
            self.camera.frames_per_trigger_zero_for_unlimited = 0
            self.camera.arm(2)
            self.camera.issue_software_trigger() # Kick off the continuous stream

    def set_single_frame_mode(self):
        """Switches the camera back to single-frame mode for piezo scanning."""
        if self.camera:
            self.camera.disarm()
            self.camera.frames_per_trigger_zero_for_unlimited = 1
            self.camera.arm(2)

    def disconnect(self):
        """Disarm and release the camera and SDK."""
        if self.camera:
            self.camera.disarm()
            self.camera.dispose()
            self.camera = None
        if self.sdk:
            self.sdk.dispose()
            self.sdk = None
        print("Camera disconnected.")

    # ------------------------------------------------------------------
    # FFT planning (internal)
    # ------------------------------------------------------------------

    def _prepare_fft_plan(self, n_frames: int):
        """
        Allocate SIMD-aligned memory buffers and build the pyFFTW plan.

        The plan performs a 1-D forward FFT along axis 0 (the temporal /
        frame axis) for every pixel simultaneously.  This is the fastest
        way to compute the pixel-wise FFT on a full camera frame stack.

        The plan is rebuilt only when the frame count changes, so repeated
        acquisitions with the same N reuse the same allocation.
        """
        print(f"Building pyFFTW plan for {n_frames} frames "
              f"({self.image_height} x {self.image_width} px)...")

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

        If AUTO_DETECT_BIN is True the bin with the highest mean magnitude
        across all pixels (excluding DC bin 0) is returned.  This is slower
        but tolerates a scan range that does not cover exactly one period.

        If AUTO_DETECT_BIN is False the class constant F1_BIN is returned
        directly (fastest path, requires a correctly calibrated scan range).
        """
        
        if not self.AUTO_DETECT_BIN:
            return self.F1_BIN

        limit = n_frames // 2
        if limit <= 1:
            return 1

        mean_magnitudes = np.abs(self._fft_output[1:limit]).mean(axis=(1, 2))
        return int(mean_magnitudes.argmax()) + 1
    
    
    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process_quantum_image(
        self, image_stack: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the pixel-wise FFT on the acquired frame stack and compute
        the three QIUP parameter maps.

        Args:
            image_stack: Float32 array of shape (N, H, W), where N is the
                         number of frames acquired during the piezo scan.

        Returns:
            (visibility_map, contrast_map, phase_map) — each is an 8-bit
            RGB image (H x W x 3) ready for display in the GUI.
        """
        n_frames = image_stack.shape[0]

        if n_frames < 1:
            raise ValueError(
                "At least 3 frames are required."
            )

        # Rebuild the plan only if N has changed.
        if n_frames != self._planned_n_frames:
            self._prepare_fft_plan(n_frames)

        # --- Step 1: Execute FFT ---
        # Copy into the aligned buffer (preserves alignment guarantees).
        self._fft_input[:] = image_stack
        self._fft_plan()

        # --- Step 2: Extract Fourier components ---
        # F0: DC term — proportional to the mean pixel brightness across
        #     the scan.  Used as the normalisation denominator for visibility.
        F0 = np.abs(self._fft_output[0])  # shape (H, W), real

        # F1: First-order term — complex amplitude of the interference fringe.
        #     Its magnitude encodes transmission; its angle encodes phase shift.
        f1_bin = self._get_f1_bin(n_frames)
        F1 = self._fft_output[f1_bin]     # shape (H, W), complex

        # --- Step 3: Compute physics parameters (Pearce et al. Eqs. 1-3) ---

        # Visibility (Eq. 1): V = 2|F1| / F0
        # The factor of 2 accounts for the negative-frequency mirror image in
        # the FFT output.  A small epsilon avoids division by zero in dark pixels.
        visibility = (2.0 * np.abs(F1)) / (F0 + 1e-10)

        # Contrast (Eq. 2): C = 4|F1|
        # Equivalent to Nmax - Nmin.  Useful when there is a significant
        # detector noise floor, since that floor cancels in the subtraction.
        contrast = 4.0 * np.abs(F1)

        # Phase (Eq. 3): phi = arctan(Im(F1) / Re(F1))
        # np.angle returns values in (-pi, pi], matching the paper's definition.
        phase = np.angle(F1)

        return self._apply_colormaps(visibility, contrast, phase)

    # ------------------------------------------------------------------
    # Colormap rendering (internal)
    # ------------------------------------------------------------------

    def _apply_colormaps(
        self,
        visibility: np.ndarray,
        contrast: np.ndarray,
        phase: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # 1. Standard 8-bit conversions
        vis_8bit = np.clip(visibility * 255.0, 0, 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(vis_8bit, cv2.COLORMAP_VIRIDIS)

        contrast_8bit = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        contrast_color = cv2.applyColorMap(contrast_8bit, cv2.COLORMAP_PLASMA)

        phase_8bit = ((phase + np.pi) / (2.0 * np.pi) * 255.0).astype(np.uint8)
        phase_color = cv2.applyColorMap(phase_8bit, cv2.COLORMAP_TWILIGHT)

        # Convert BGR to RGB
        vis_color = cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB)
        contrast_color = cv2.cvtColor(contrast_color, cv2.COLOR_BGR2RGB)
        phase_color = cv2.cvtColor(phase_color, cv2.COLOR_BGR2RGB)

        # Apply the Amplitude Masking for Phase
        vis_threshold = 0.10
        mask_3d = (visibility > vis_threshold)[..., np.newaxis]
        phase_color = np.where(mask_3d, phase_color, 0).astype(np.uint8)

        v_min, v_max = np.min(visibility), np.max(visibility)
        # 2. Add Colorbars to each image
        vis_with_scale = self._add_colorbar(vis_color, cv2.COLORMAP_VIRIDIS, f"{v_min:.1f}", f"{v_max:.1f}", "Visibility")
        
        # For contrast, we use the local min/max text
        c_min, c_max = np.min(contrast), np.max(contrast)
        contrast_with_scale = self._add_colorbar(contrast_color, cv2.COLORMAP_PLASMA, f"{c_min:.1f}", f"{c_max:.1f}", "Contrast")
        
        phase_with_scale = self._add_colorbar(phase_color, cv2.COLORMAP_TWILIGHT, "-pi", "pi", "Phase")

        return vis_with_scale, contrast_with_scale, phase_with_scale

    def _add_colorbar(self, image, colormap_type, label_min, label_max, title):
        """
        Helper to append a vertical colorbar with labels to the right of an image.
        """
        h, w, _ = image.shape
        cb_width = 80  # Width of the color strip
        padding = 40   # Extra space for text labels
        
        # Create a vertical gradient from 255 (top) to 0 (bottom)
        gradient = np.linspace(255, 0, h).astype(np.uint8).reshape(h, 1)
        gradient_strip = np.repeat(gradient, cb_width, axis=1)
        
        # Apply the same colormap and convert to RGB
        color_strip = cv2.applyColorMap(gradient_strip, colormap_type)
        color_strip = cv2.cvtColor(color_strip, cv2.COLOR_BGR2RGB)
        
        # Create a canvas to hold image + colorbar + text padding
        canvas = np.zeros((h, w + cb_width + padding, 3), dtype=np.uint8)
        canvas[:, :w] = image
        canvas[:, w:w+cb_width] = color_strip
        
        # Add Text Labels using OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        outline_thickness = 3
        white = (255, 255, 255)
        black = (0, 0, 0)

        # Positions for labels
        pos_max = (w + cb_width + 5, 20)
        pos_min = (w + cb_width + 5, h - 10)
        pos_title = (w + 5, 15)

        # Draw labels with outlines
        for pos, text in [(pos_max, label_max), (pos_min, label_min), (pos_title, title)]:
            # 1. Draw the Black Outline
            cv2.putText(canvas, text, pos, font, font_scale, black, outline_thickness, cv2.LINE_AA)
            # 2. Draw the White Text on top
            cv2.putText(canvas, text, pos, font, font_scale, white, thickness, cv2.LINE_AA)

        return canvas
