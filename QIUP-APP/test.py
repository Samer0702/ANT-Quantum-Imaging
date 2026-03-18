import numpy as np
import cv2
import pyfftw
import time
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from pze_control import PiezoController

try:
    from CMOS_windows_setup import configure_path
    configure_path()
except ImportError:
    pass

def main():
    # --- Experimental Parameters ---
    N_FRAMES = 256
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

            # Create our display windows
            cv2.namedWindow("QIUP Visibility Map", cv2.WINDOW_NORMAL)
            cv2.namedWindow("QIUP Phase Map", cv2.WINDOW_NORMAL)
            cv2.namedWindow("QIUP Contrast Map", cv2.WINDOW_NORMAL)
            
            print(f"Acquiring {N_FRAMES} frames for a single quantum image...")

            try:
                # --- 1. ACQUISITION  ---
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
                fft_plan()

                # Extract the DC component (F0) and interference frequency (F1)
                F0 = np.abs(fft_output[0])
                F1 = fft_output[1]

                # --- 3. PARAMETER CALCULATION ---
                # A. Visibility (V = 2*|F1|/F0)
                visibility = (2.0 * np.abs(F1)) / (F0 + 1e-10)
                
                # B. Phase (Angle of the complex frequency component)
                phase = np.angle(F1)
                
                # C. Contrast (Raw peak-to-peak amplitude = 4*|F1|)
                contrast = 4.0 * np.abs(F1)

                # --- 4. VISUALIZATION & MAPPING ---
                # A. Map Visibility to 8-bit (0-255) and apply Viridis
                vis_8bit = np.clip(visibility * 255, 0, 255).astype(np.uint8)
                vis_color = cv2.applyColorMap(vis_8bit, cv2.COLORMAP_VIRIDIS)
                
                # B. Map Phase (-pi to pi) linearly to (0 to 255) and apply Twilight
                phase_8bit = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
                phase_color = cv2.applyColorMap(phase_8bit, cv2.COLORMAP_TWILIGHT)
                
                # C. Map Contrast. Since contrast is an arbitrary raw intensity value, 
                # we normalize it against its own minimum and maximum to fill the 0-255 range.
                contrast_8bit = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                contrast_color = cv2.applyColorMap(contrast_8bit, cv2.COLORMAP_PLASMA)

                # Show all three images
                cv2.imshow("QIUP Visibility Map", vis_color)
                cv2.imshow("QIUP Contrast Map", contrast_color)
                cv2.imshow("QIUP Phase Map", phase_color)
                
                print("\nImages generated! Press 's' to save or 'q' to quit.")

                # --- 5. WAIT AND SAVE LOOP ---
                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        cv2.imwrite("visibility_map.png", vis_color)
                        cv2.imwrite("phase_map.png", phase_color)
                        cv2.imwrite("contrast_map.png", contrast_color)
                        print("Saved all 3 maps!")
                        break

            finally:
                camera.disarm()
                piezo.set_voltage(0)
                piezo.disconnect()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()