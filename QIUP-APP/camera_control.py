import numpy as np
import cv2
import pyfftw
import os
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

class CameraController:
    def __init__(self):
        self.initialize_camera()

    def initialize_camera(self):
        try:
            from CMOS_windows_setup import configure_path
            configure_path()
        except ImportError:
            pass

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
                camera.gain = 35.0
                camera.arm(2)

                h, w = camera.image_height_pixels, camera.image_width_pixels
                print(f"Camera Initialized: {w}x{h}")
