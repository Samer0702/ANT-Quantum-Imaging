import numpy as np
import time
import matplotlib.pyplot as plt
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
from pze_control import PiezoController

try:
    from CMOS_windows_setup import configure_path
    configure_path()
except ImportError:
    pass

def run_calibration():
    # --- Calibration Parameters ---
    START_V = 0.0
    END_V = 10.0         # Sweep up to 10V to guarantee we see at least one full peak-to-peak
    STEPS = 50           # High resolution to get a smooth sine wave
    SETTLING_TIME = 0.05
    
    voltages = np.linspace(START_V, END_V, STEPS)
    intensities = []

    print("Connecting to Piezo...")
    piezo = PiezoController(piezo_serial="29252595")
    
    # EXACTLY ONE connection call. We capture the status and the message.
    connected, msg = piezo.connect()
    
    if not connected:
        print(f"Piezo connection failed. Reason: {msg}")
        return
        
    print("Piezo successfully connected!")

    with TLCameraSDK() as sdk:
        cameras = sdk.discover_available_cameras()
        if not cameras:
            print("No cameras found.")
            piezo.disconnect()
            return

        with sdk.open_camera(cameras[0]) as camera:
            camera.exposure_time_us = 200000 
            camera.gain = 35
            camera.frames_per_trigger_zero_for_unlimited = 0
            camera.image_poll_timeout_ms = 1000
            camera.arm(2)
            
            h, w = camera.image_height_pixels, camera.image_width_pixels
            
            # Define a small region of interest (ROI) in the center of the camera
            # We average a 100x100 box to reduce noise
            cy, cx = h // 2, w // 2
            roi_size = 50

            print(f"Starting Fringe Scan from {START_V}V to {END_V}V...")
            
            try:
                for v in voltages:
                    piezo.set_voltage(v)
                    time.sleep(SETTLING_TIME)
                    
                    camera.issue_software_trigger()
                    frame = camera.get_pending_frame_or_null()
                    
                    if frame is not None:
                        img = np.copy(frame.image_buffer).reshape(h, w)
                        # Get the mean intensity of the center region
                        mean_intensity = np.mean(img[cy-roi_size:cy+roi_size, cx-roi_size:cx+roi_size])
                        intensities.append(mean_intensity)
                        print(f"Voltage: {v:05.2f}V | Intensity: {mean_intensity:.2f}")
                    else:
                        intensities.append(0)
                        print(f"Voltage: {v:05.2f}V | Dropped Frame")
                        
            finally:
                camera.disarm()
                piezo.set_voltage(0)
                piezo.disconnect()

    # --- Plot the Results ---
    plt.figure(figsize=(10, 6))
    plt.plot(voltages, intensities, marker='o', linestyle='-', color='b')
    plt.title("Interference Fringe Calibration Scan")
    plt.xlabel("Piezo Voltage (V)")
    plt.ylabel("Mean Camera Intensity (Center ROI)")
    plt.grid(True)
        
    plt.show()

if __name__ == "__main__":
    run_calibration()