import time
import numpy as np
import pyfftw

# Import your existing hardware controllers
from camera_control import CameraController
from piezo_control_open import PiezoController

def run_standalone_benchmark():    
    # ---------------------------------------------------------
    # 1. Initialize Hardware (Using your Controller Classes)
    # ---------------------------------------------------------
    print("Connecting to Piezo...")
    piezo = PiezoController()
    p_ok, p_msg = piezo.connect()
    if not p_ok:
        print(f"Failed to connect to Piezo: {p_msg}")
        return

    print("Connecting to Thorlabs Camera...")
    cam_ctrl = CameraController()
    c_ok = cam_ctrl.connect()
    
    if not c_ok:
        print("Camera Error: Check USB connection.")
        piezo.disconnect()
        return

    # Configure Camera for the scan
    cam_ctrl.camera.exposure_time_us = 200_000   # 200 ms
    cam_ctrl.camera.gain = 350                   # 35.0 dB
    cam_ctrl.set_single_frame_mode()

    width = cam_ctrl.image_width
    height = cam_ctrl.image_height
    print(f"Camera configured: {width} x {height} px")

    # ---------------------------------------------------------
    # 2. Configure Scan & Memory Allocation
    # ---------------------------------------------------------
    n_frames = 32
    start_v = 0.0
    end_v = 3.9
    settling_time_s = 0.05
    voltages = np.linspace(start_v, end_v, n_frames, endpoint=False)

    print(f"\nAllocating pyFFTW byte-aligned arrays for {n_frames} frames...")
    # These must be allocated in the main script for the benchmark
    fft_input = pyfftw.empty_aligned((n_frames, height, width), dtype='complex64')
    fft_output = pyfftw.empty_aligned((n_frames, height, width), dtype='complex64')
    
    # Standard NumPy array acting as a safe backup of the acquired frames
    raw_stack = np.zeros((n_frames, height, width), dtype=np.complex64)

    # ---------------------------------------------------------
    # 3. Hardware Acquisition Loop
    # ---------------------------------------------------------
    print("\nStarting piezo scan and image acquisition...")
    acq_start = time.perf_counter()
    
    for i, v in enumerate(voltages):
        # Step Piezo
        piezo.set_voltage(v)
        time.sleep(settling_time_s)
        
        # Trigger Camera via the controller's camera object
        cam_ctrl.camera.issue_software_trigger()
        
        # Poll for frame
        frame = cam_ctrl.camera.get_pending_frame_or_null()
        timeout = time.time() + 2.0
        while frame is None:
            if time.time() > timeout:
                raise TimeoutError("Camera frame timeout.")
            time.sleep(0.01)
            frame = cam_ctrl.camera.get_pending_frame_or_null()

        # Load frame into our backup raw_stack array
        raw_stack[i] = frame.image_buffer
        
        print(f"  Acquired frame {i+1}/{n_frames} at {v:.2f} V")

    acq_time = time.perf_counter() - acq_start
    print(f"Acquisition completed in {acq_time:.2f} seconds.")

    # ---------------------------------------------------------
    # 4. Processing Benchmarks 
    # ---------------------------------------------------------
    print("\n=== Processing Benchmarks ===")

    # -- Method A: Standard NumPy FFT --
    print("1. Running standard numpy.fft...")
    start_np = time.perf_counter()
    
    fft_numpy = np.fft.fft(raw_stack, axis=0)
    F0_np = np.abs(fft_numpy[0])
    F1_np = fft_numpy[1]
    vis_np = (2.0 * np.abs(F1_np)) / (F0_np + 1e-10)
    
    time_np = time.perf_counter() - start_np
    print(f"   -> NumPy executed in:             {time_np:.5f} seconds")


    # -- Method B: pyFFTW (Planning + Execution) --
    print("\n2. Running pyFFTW (Planning + Execution)...")
    # Load the fresh data into the aligned array
    fft_input[:] = raw_stack[:]
    
    start_fftw_plan = time.perf_counter()
    
    # Building the plan (FFTW_MEASURE will optimize the CPU instructions but destroy fft_input)
    fft_plan = pyfftw.FFTW(
        fft_input, 
        fft_output, 
        axes=(0,), 
        direction='FFTW_FORWARD', 
        flags=('FFTW_MEASURE',)
    )
    # Execute immediately after planning
    fft_plan() 
    
    F0_fftw = np.abs(fft_output[0])
    F1_fftw = fft_output[1]
    vis_fftw = (2.0 * np.abs(F1_fftw)) / (F0_fftw + 1e-10)
    
    time_fftw_plan = time.perf_counter() - start_fftw_plan
    print(f"   -> Planning + Execution took:     {time_fftw_plan:.5f} seconds")


    # -- Method C: pyFFTW (Execution ONLY) --
    print("\n3. Running pyFFTW (Execution after using existing plan)...")
    # Reload fresh data because FFTW_MEASURE destroyed the previous contents
    fft_input[:] = raw_stack[:]
    
    start_fftw_exec = time.perf_counter()
    
    # The plan is already built, we just call it
    fft_plan() 
    
    F0_fast = np.abs(fft_output[0])
    F1_fast = fft_output[1]
    vis_fast = (2.0 * np.abs(F1_fast)) / (F0_fast + 1e-10)
    
    time_fftw_exec = time.perf_counter() - start_fftw_exec
    print(f"   -> Execution took:           {time_fftw_exec:.5f} seconds \n")


    # ---------------------------------------------------------
    # 5. Results & Clean up
    # ---------------------------------------------------------
    print(f"NumPy vs FFTW (After planning): FFTW is {time_np / time_fftw_exec:.2f}x faster")
    
    print("\nDisconnecting hardware...")
    piezo.set_voltage(0.0)
    
    # Safely disconnect using your custom methods
    cam_ctrl.disconnect()
    piezo.disconnect()
    print("Done.")

if __name__ == "__main__":
    run_standalone_benchmark()