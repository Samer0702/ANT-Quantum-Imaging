import os
import clr
import time
import System

class PiezoController:
    def __init__(self, piezo_serial="29252595", strain_serial="59500024"):
        self.piezo_serial = piezo_serial
        self.strain_serial = strain_serial
        self.piezo = None
        self.is_connected = False
        
        # Setup paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self._load_dlls()

    def _load_dlls(self):
        """Internal method to load Thorlabs Kinesis DLLs."""
        kinesis_path = os.path.join(self.base_dir, 'thorlabs_lib', 'Kinesis')
        clr.AddReference(os.path.join(kinesis_path, 'Thorlabs.MotionControl.DeviceManagerCLI.dll'))
        clr.AddReference(os.path.join(kinesis_path, 'Thorlabs.MotionControl.KCube.PiezoCLI.dll'))
        clr.AddReference(os.path.join(kinesis_path, 'Thorlabs.MotionControl.KCube.StrainGaugeCLI.dll'))

    def connect(self):
        """Initializes and connects to the KPZ101 Piezo Controller."""
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        from Thorlabs.MotionControl.KCube.PiezoCLI import KCubePiezo
        from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import PiezoControlModeTypes

        try:
            DeviceManagerCLI.BuildDeviceList()
            self.piezo = KCubePiezo.CreateKCubePiezo(self.piezo_serial)
            self.piezo.Connect(self.piezo_serial)
            
            if not self.piezo.IsSettingsInitialized():
                self.piezo.WaitForSettingsInitialized(5000)

            self.piezo.GetPiezoConfiguration(self.piezo_serial)
            self.piezo.StartPolling(250)
            self.piezo.EnableDevice()
            time.sleep(0.5)

            # Set to Open Loop for interferometric fringe scanning
            self.piezo.SetPositionControlMode(PiezoControlModeTypes.OpenLoop)
            self.is_connected = True
            return True, "Connected successfully"
        except Exception as e:
            return False, str(e)

    def set_voltage(self, voltage):
        """Sets the output voltage (0-75V or 0-150V depending on model)."""
        if not self.is_connected:
            return False
        
        target = System.Convert.ToDecimal(float(voltage))
        try:
            # Try both possible API method names for compatibility
            try:
                self.piezo.SetOutputVoltage(target)
            except AttributeError:
                self.piezo.SetVoltage(target)
            return True
        except Exception as e:
            print(f"Movement error: {e}")
            return False

    def get_voltage(self):
        """Reads the current voltage from the controller."""
        if not self.is_connected:
            return 0.0
        try:
            try:
                return float(str(self.piezo.GetOutputVoltage()))
            except AttributeError:
                return float(str(self.piezo.GetVoltage()))
        except:
            return 0.0

    def disconnect(self):
        """Safely shuts down the connection."""
        if self.piezo and self.is_connected:
            self.piezo.StopPolling()
            self.piezo.Disconnect()
            self.is_connected = False
            print("Piezo disconnected.")