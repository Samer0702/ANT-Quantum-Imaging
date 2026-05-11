import os
import time
import clr
import System

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.path.join(BASE_DIR, "thorlabs_lib", "Kinesis")

PIEZO_SERIAL = "29252595"
STRAIN_SERIAL = "59500024"

if not os.path.exists(DLL_PATH):
    print(f"ERROR: Directory not found at {DLL_PATH}")
    exit()

# --- LOAD KINESIS DLLs ---
try:
    clr.AddReference(os.path.join(DLL_PATH, "Thorlabs.MotionControl.DeviceManagerCLI.dll"))
    clr.AddReference(os.path.join(DLL_PATH, "Thorlabs.MotionControl.KCube.PiezoCLI.dll"))
    clr.AddReference(os.path.join(DLL_PATH, "Thorlabs.MotionControl.KCube.StrainGaugeCLI.dll"))
except Exception as e:
    print(f"Error loading DLLs: {e}")
    exit()

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.KCube.PiezoCLI import KCubePiezo
from Thorlabs.MotionControl.KCube.StrainGaugeCLI import KCubeStrainGauge
from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import PiezoControlModeTypes

def run_test():
    try:
        DeviceManagerCLI.BuildDeviceList()
        
        piezo = KCubePiezo.CreateKCubePiezo(PIEZO_SERIAL)
        strain = KCubeStrainGauge.CreateKCubeStrainGauge(STRAIN_SERIAL)

        print(f"Connecting to Piezo Controller {PIEZO_SERIAL}...")
        piezo.Connect(PIEZO_SERIAL)
        print(f"Connecting to Strain Gauge Reader {STRAIN_SERIAL}...")
        strain.Connect(STRAIN_SERIAL)

        if not piezo.IsSettingsInitialized(): 
            piezo.WaitForSettingsInitialized(5000)
        if not strain.IsSettingsInitialized(): 
            strain.WaitForSettingsInitialized(5000)

        # --- THE CRITICAL FIX: HARDWARE CONTEXT ---
        # NOTE: For GetPiezoConfiguration to load correct bounds (e.g., 20um, 75V), 
        # you MUST have previously opened the Thorlabs Kinesis PC software, 
        # assigned your specific actuator to this KPZ101, and clicked "Persist Settings to Device".
        piezo_config = piezo.GetPiezoConfiguration(PIEZO_SERIAL)
        strain_config = strain.GetStrainGaugeConfiguration(STRAIN_SERIAL)
        # ------------------------------------------

        piezo.StartPolling(250)
        strain.StartPolling(250)
        time.sleep(0.5)
        piezo.EnableDevice()
        strain.EnableDevice()
        time.sleep(0.5)

        # 2. Zero the Strain Gauge while the Piezo is relaxed
        print("Zeroing strain gauge... please wait 15 seconds.")
        strain.SetZero()
        time.sleep(15.0)

        # 3. NOW it is safe to engage the PID Loop
        # IMPORTANT: Ensure the SMA cable connects KSG101 'SIG OUT' to KPZ101 'EXT IN', 
        # OR ensure the K-Cube hub routing switches are correctly set on the back of the cubes.
        print("Setting mode: Closed Loop...")
        piezo.SetPositionControlMode(PiezoControlModeTypes.CloseLoop)
        time.sleep(1.0)

        # Test Motion Cycle in absolute micrometers
        test_steps_um = [0.0, 5.0, 10.0, 15.0, 20.0, 0.0]
        MAX_TRAVEL_UM = 20.0
        
        print("\nStarting Test Cycle (Calibrated):")
        print(f"{'Target µm':>10} | {'Readback µm':>15} | {'Raw ADC'}")
        print("-" * 42)

        for step_um in test_steps_um:
            # 1. Convert micrometers to a percentage (0.0 to 100.0)
            target_pct = (step_um / MAX_TRAVEL_UM) * 100.0
            
            # Command the piezo using a percentage to bypass device-unit confusion
            piezo.SetPercentageTravel(System.Convert.ToDecimal(float(target_pct)))
            
            # Allow time for the piezo PID loop to settle
            time.sleep(3.0)
            
            # 2. Get the raw ADC reading from the Strain Gauge
            raw_reading = int(str(strain.GetReading()))
            
            # 3. Convert the raw +/- 32767 value back into Micrometers
            # Note: 32767 is the full-scale 16-bit integer for 100% travel
            displacement_um = (raw_reading / 32767.0) * MAX_TRAVEL_UM
            
            print(f"{step_um:10.1f} | {displacement_um:12.3f} µm    | {raw_reading}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

    finally:
        print("\nShutting down and disconnecting...")
        if piezo.IsConnected:
            # Safely return to 0 Volts and Open Loop before shutting down
            piezo.SetPositionControlMode(PiezoControlModeTypes.OpenLoop)
            time.sleep(0.5)
            piezo.SetOutputVoltage(System.Convert.ToDecimal(0.0))
            piezo.StopPolling()
            piezo.Disconnect()
            
        if strain.IsConnected:
            strain.StopPolling()
            strain.Disconnect()
            
        print("Done.")

if __name__ == "__main__":
    run_test()