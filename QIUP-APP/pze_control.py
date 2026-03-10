import os
import clr
import time
import System 

base_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths to the Kinesis DLLs
device_manager_cli = os.path.join(base_dir, 'thorlabs_lib', 'Kinesis', 'Thorlabs.MotionControl.DeviceManagerCLI.dll')
piezo_cli = os.path.join(base_dir, 'thorlabs_lib', 'Kinesis', 'Thorlabs.MotionControl.KCube.PiezoCLI.dll')
strain_gauge_cli = os.path.join(base_dir, 'thorlabs_lib', 'Kinesis', 'Thorlabs.MotionControl.KCube.StrainGaugeCLI.dll')

try:
    # Load the DLLs
    clr.AddReference(device_manager_cli)
    clr.AddReference(piezo_cli)
    clr.AddReference(strain_gauge_cli)

    from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
    from Thorlabs.MotionControl.KCube.PiezoCLI import KCubePiezo
    from Thorlabs.MotionControl.KCube.StrainGaugeCLI import KCubeStrainGauge
    from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import PiezoControlModeTypes

    # Initialize Device Manager
    DeviceManagerCLI.BuildDeviceList()
    devices = DeviceManagerCLI.GetDeviceList()
    
    print(f"Found {len(devices)} devices.")

    # Define serials
    piezo_serial = "29252595"
    strain_gauge_serial = "59500024"

    # 1. CONNECT & INITIALIZE PIEZO (KPZ101)
    piezo = KCubePiezo.CreateKCubePiezo(piezo_serial)
    piezo.Connect(piezo_serial)
    if not piezo.IsSettingsInitialized():
        piezo.WaitForSettingsInitialized(5000)

    # This explicitly initializes the Device Unit Converter
    piezo.GetPiezoConfiguration(piezo_serial)

    # Start polling and enable the hardware
    piezo.StartPolling(250)
    piezo.EnableDevice()
    time.sleep(0.5)  

    # Set to Open Loop Mode (Voltage Control)
    piezo.SetPositionControlMode(PiezoControlModeTypes.OpenLoop)
    time.sleep(0.2)

    # 2. DIAGNOSTICS & MOVEMENT    
    print("Zeroing the piezo stage...")
    try:
        piezo.SetZero()
        time.sleep(2) 
    except AttributeError:
        piezo.SetOutputVoltage(System.Convert.ToDecimal(0.0))
        time.sleep(0.5)
        
    # Apply 10 Volts using System.Convert to guarantee correct C# data type
    target_voltage = System.Convert.ToDecimal(10.0)
    print(f"Attempting to set voltage to {target_voltage}V...")
    
    # Send the command
    try:
        piezo.SetOutputVoltage(target_voltage)
    except AttributeError:
        piezo.SetVoltage(target_voltage)
        
    time.sleep(0.5) 
    
    # Read current voltage back from the controller
    try:
        current_volts = piezo.GetOutputVoltage()
    except AttributeError:
        current_volts = piezo.GetVoltage()
        
    print(f"Current Position (Piezo Voltage): {current_volts}V")

    # 3. CLEANUP & DISCONNECT
    piezo.StopPolling()
    piezo.Disconnect()
    print("Devices disconnected safely.")

except Exception as e:
    print(f"Error: {e}")
    try:
        piezo.StopPolling()
        piezo.Disconnect()
    except:
        pass