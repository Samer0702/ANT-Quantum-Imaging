import numpy as np
import cv2
import pyfftw
import os
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

# --- Global Filter Parameters ---
low_cut = 1
high_cut = 65

def on_low_change(val): global low_cut; low_cut = val
def on_high_change(val): global high_cut; high_cut = val

# --- Setup Thorlabs Path for CMOS camera (Windows) ---
try:
    from CMOS_windows_setup import configure_path
    configure_path()
except ImportError:
    pass

def main():
    global low_cut, high_cut

    with TLCameraSDK() as sdk:
        available_cameras = sdk.discover_available_cameras()
        if not available_cameras:
            print("No cameras detected. Check USB/Power.")
            return

        with sdk.open_camera(available_cameras[0]) as camera:
            # 1. Camera Configuration
            camera.exposure_time_us = 200000
            camera.frames_per_trigger_zero_for_unlimited = 0
            camera.image_poll_timeout_ms = 1000
            camera.arm(2)
            
            h, w = camera.image_height_pixels, camera.image_width_pixels
            print(f"Camera Initialized: {w}x{h}")

            # 2. Pre-allocate FFTW Aligned Arrays for Speed
            # Using 'complex64' for the Fourier Domain
            input_ptr = pyfftw.empty_aligned((h, w), dtype='complex64')
            output_ptr = pyfftw.empty_aligned((h, w), dtype='complex64')
            
            # Plan the FFT (This optimizes the math for your specific CPU)
            fft_plan = pyfftw.builders.fft2(input_ptr)
            ifft_plan = pyfftw.builders.ifft2(output_ptr)

            # Pre-calculate the frequency coordinate grid
            crow, ccol = h // 2, w // 2
            y, x = np.ogrid[-crow:h-crow, -ccol:w-ccol]
            dist_sq = x**2 + y**2

            # 3. Setup OpenCV Windows & Sliders
            cv2.namedWindow("Quantum Live Feed", cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Low (Noise)", "Quantum Live Feed", low_cut, 200, on_low_change)
            cv2.createTrackbar("High (Detail)", "Quantum Live Feed", high_cut, 500, on_high_change)

            print("Running... Press 's' to save, 'q' to quit.")

            try:
                while True:
                    camera.issue_software_trigger()
                    frame = camera.get_pending_frame_or_null()

                    if frame is not None:
                        # Convert buffer to numpy
                        raw_img = np.copy(frame.image_buffer).reshape(h, w)
                        
                        # --- THE FOURIER PROCESS ---
                        # Load data into aligned memory
                        input_ptr[:] = raw_img.astype(np.complex64)
                        
                        # Forward FFT and shift zero-freq to center
                        dft = fft_plan()
                        dft_shifted = np.fft.fftshift(dft)

                        # Create the Bandpass Mask (The "Donut")
                        mask = (dist_sq >= low_cut**2) & (dist_sq <= high_cut**2)
                        
                        # Apply mask: zero out frequencies outside the band
                        filtered_dft = np.zeros_like(dft_shifted)
                        filtered_dft[mask] = dft_shifted[mask]
                        
                        # Shift back and Inverse FFT
                        output_ptr[:] = np.fft.ifftshift(filtered_dft)
                        result_complex = ifft_plan()
                        result_mag = np.abs(result_complex)

                        # --- VISUALIZATION ---
                        # Normalize for 8-bit display
                        display_img = cv2.normalize(result_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        raw_display = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                        # Show the Mask (Optional visual aid to see what frequencies are kept)
                        mask_viz = (mask.astype(np.uint8) * 255)
                        mask_viz = cv2.resize(mask_viz, (256, 256)) # Resize for small preview
                        
                        cv2.imshow("Quantum Live Feed", display_img)
                        cv2.imshow("Raw Sensor", raw_display)
                        cv2.imshow("Filter Mask Preview", mask_viz)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        cv2.imwrite("quantum_capture.png", display_img)
                        print("Saved image to quantum_capture.png")

            finally:
                camera.disarm()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()