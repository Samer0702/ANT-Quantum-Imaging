import os
import clr
import time

base_dir = os.path.dirname(os.path.abspath(__file__))

device_manager_cli = os.path.join(base_dir, 'thorlabs_lib', 'Kinesis', 'Thorlabs.MotionControl.DeviceManagerCLI.dll')
piezo_cli = os.path.join(base_dir, 'thorlabs_lib', 'Kinesis', 'Thorlabs.MotionControl.KCube.PiezoCLI.dll')
strain_gauge_cli = os.path.join(base_dir, 'thorlabs_lib', 'Kinesis', 'Thorlabs.MotionControl.KCube.StrainGaugeCLI.dll')

try:
    clr.AddReference(device_manager_cli)
    clr.AddReference(piezo_cli)
    clr.AddReference(strain_gauge_cli)

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

    # Connect to KPZ101 (Piezo)
    piezo = KCubePiezo.CreateKCubePiezo(piezo_serial)
    piezo.Connect(piezo_serial)
    if not piezo.IsSettingsInitialized():
        piezo.WaitForSettingsInitialized(5000)

    # Connect to KSG101 (Strain Gauge)
    strain_gauge = KCubeStrainGauge.CreateKCubeStrainGauge(strain_gauge_serial)
    strain_gauge.Connect(strain_gauge_serial)
    if not strain_gauge.IsSettingsInitialized():
        strain_gauge.WaitForSettingsInitialized(5000)
    print(f"Connected to Strain Gauge (KSG101): {strain_gauge_serial}")


except Exception as e:
    print(f"Error: {e}")