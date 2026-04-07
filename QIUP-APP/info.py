import os
import clr
import time
import System

base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths to the Kinesis DLLs
device_manager_cli = os.path.join(base_dir, 'thorlabs_lib', 'Kinesis', 'Thorlabs.MotionControl.DeviceManagerCLI.dll')
piezo_cli = os.path.join(base_dir, 'thorlabs_lib', 'Kinesis', 'Thorlabs.MotionControl.KCube.PiezoCLI.dll')
strain_gauge_cli = os.path.join(base_dir, 'thorlabs_lib', 'Kinesis', 'Thorlabs.MotionControl.KCube.StrainGaugeCLI.dll')
generic_piezo_cli = os.path.join(base_dir, 'thorlabs_lib', 'Kinesis', 'Thorlabs.MotionControl.GenericPiezoCLI.dll')

try:
    # Load the libraries
    clr.AddReference(device_manager_cli)
    clr.AddReference(piezo_cli)
    clr.AddReference(strain_gauge_cli)
    clr.AddReference(generic_piezo_cli)

    # Import the classes
    from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
    from Thorlabs.MotionControl.KCube.PiezoCLI import KCubePiezo
    from Thorlabs.MotionControl.KCube.StrainGaugeCLI import KCubeStrainGauge

    DeviceManagerCLI.BuildDeviceList()
    devices = DeviceManagerCLI.GetDeviceList()

    print(f"Found {len(devices)} devices:")
    for device in devices:
        print(f"Serial Number: {device}")

    # Define serials
    piezo_serial = "29252595"
    strain_gauge_serial = "59500024"

    # --- 1. CONNECT TO PIEZO ---
    piezo = KCubePiezo.CreateKCubePiezo(piezo_serial)
    piezo.Connect(piezo_serial)
    if not piezo.IsSettingsInitialized():
        piezo.WaitForSettingsInitialized(5000)

    # --- 2. CONNECT TO STRAIN GAUGE ---
    strain_gauge = KCubeStrainGauge.CreateKCubeStrainGauge(strain_gauge_serial)
    strain_gauge.Connect(strain_gauge_serial)
    if not strain_gauge.IsSettingsInitialized():
        strain_gauge.WaitForSettingsInitialized(5000)

    print('piezo dir:')
    print(dir(piezo))
    print('strain gauge dir:')
    print(dir(strain_gauge))


except Exception as e:
    print(f"Error: {e}")