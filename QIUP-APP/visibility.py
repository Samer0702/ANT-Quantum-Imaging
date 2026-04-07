import numpy as np
import cv2
import pyfftw
import time
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from piezo_control import PiezoController

try:
    from CMOS_windows_setup import configure_path
    configure_path()
except ImportError:
    pass

def main():
    # --- Experimental Parameters ---
    N_FRAMES = 32
    SCAN_V_START = 1.25
    SCAN_V_END = 5              # Must equal exactly one full interference cycle (2*pi)
    SETTLING_TIME = 0.05        
    
    voltages = np.linspace(SCAN_V_START, SCAN_V_END, N_FRAMES)

    print("Connecting to Piezo Controller...")
    piezo = PiezoController(piezo_serial="29252595")
    connected, msg = piezo.connect()
    if not connected:
        print(f"Failed to connect to Piezo: {msg}")
        return
    print("Piezo connected.")

    with TLCameraSDK() as sdk:
        available_cameras = sdk.discover_available_cameras()
        if not available_cameras:
            print("No cameras detected.")
            piezo.disconnect()
            return

        with sdk.open_camera(available_cameras[0]) as camera:
            camera.exposure_time_us = 200000 
            camera.gain = 55
            camera.frames_per_trigger_zero_for_unlimited = 1
            camera.image_poll_timeout_ms = 1000
            camera.arm(2)
            
            h, w = camera.image_height_pixels, camera.image_width_pixels
            print(f"Camera Initialized: {w}x{h}")

            # Pre-allocate memory for FFT speed
            image_stack = pyfftw.empty_aligned((N_FRAMES, h, w), dtype='complex64')
            fft_output = pyfftw.empty_aligned((N_FRAMES, h, w), dtype='complex64')
            
            # Plan 1D FFT along the frame axis
            fft_plan = pyfftw.FFTW(image_stack, fft_output, axes=(0,), direction='FFTW_FORWARD', flags=('FFTW_ESTIMATE',))

            cv2.namedWindow("QIUP Visibility Map", cv2.WINDOW_NORMAL)
            print(f"Acquiring {N_FRAMES} frames for a single quantum image...")

            try:
                # --- 1. ACQUISITION (Runs exactly once) ---
                for i, v in enumerate(voltages):
                    piezo.set_voltage(v)
                    time.sleep(SETTLING_TIME) 
                    
                    camera.issue_software_trigger()
                    frame = camera.get_pending_frame_or_null()
                    if frame is not None:
                        image_stack[i] = np.copy(frame.image_buffer).reshape(h, w)
                        print(f"Captured frame {i+1}/{N_FRAMES}")

                piezo.set_voltage(SCAN_V_START) 

                print("Calculating Fourier Transform...")
                # --- 2. FOURIER ANALYSIS ---
                # Executes the FFT on all pixels simultaneously across the frames
                fft_plan()

                # Extract the DC component (F0) and interference frequency (F1)
                F0 = np.abs(fft_output[0])
                F1 = fft_output[1]

                # --- 3. VISIBILITY CALCULATION ---
                # V = 2 * |F1| / F0
                visibility = (2.0 * np.abs(F1)) / (F0 + 1e-10)

                # --- 4. VISUALIZATION ---
                vis_8bit = np.clip(visibility * 255, 0, 255).astype(np.uint8)
                vis_color = cv2.applyColorMap(vis_8bit, cv2.COLORMAP_VIRIDIS)

                cv2.imshow("QIUP Visibility Map", vis_color)
                print("\nImage generated! Press 's' to save or 'q' to quit.")

                # --- 5. WAIT AND SAVE LOOP ---
                # This keeps the window open indefinitely until you make a choice
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        cv2.imwrite("visibility_map_cat_ears.png", vis_color)
                        print("Saved to visibility_map_cat_ears.png")
                        break # Exit the loop after saving

            finally:
                camera.disarm()
                piezo.set_voltage(0)
                piezo.disconnect()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()