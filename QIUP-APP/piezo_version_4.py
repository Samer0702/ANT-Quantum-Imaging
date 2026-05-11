"""
piezo_control.py
--------------
Controls a Thorlabs KPZ101 K-Cube Piezo Controller and optionally reads
displacement from a KSG101 K-Cube Strain Gauge Reader via the Kinesis SDK.

Closed loop: Target positions are sent in micrometers (converted to 0-100%),
             and real displacement is read back from the strain gauge.

The KPZ101 and KSG101 must be connected together via the SMA feedback cable
for the closed-loop servo and strain gauge readings to function properly.
"""

import os
import clr
import time
import System


class PiezoController:
    """
    Controls a Thorlabs KPZ101 piezo driver in Closed Loop mode, 
    with KSG101 strain gauge readback for real displacement measurement.
    """

    MAX_TRAVEL_UM = 20.0  # Assumes standard 20µm actuator

    def __init__(
        self,
        piezo_serial: str = "29252595",
        strain_serial: str | None = "59500024",
    ):
        """
        Args:
            piezo_serial:  Serial number of the KPZ101.
            strain_serial: Serial number of the KSG101.
        """
        self.piezo_serial = piezo_serial
        self.strain_serial = strain_serial

        self.piezo = None
        self.strain = None
        self.is_connected = False
        self.has_strain_gauge = strain_serial is not None

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self._load_dlls()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _load_dlls(self):
        """Load the required Thorlabs Kinesis .NET DLLs via pythonnet."""
        kinesis_path = os.path.join(self.base_dir, "thorlabs_lib", "Kinesis")
        dlls = [
            "Thorlabs.MotionControl.DeviceManagerCLI.dll",
            "Thorlabs.MotionControl.KCube.PiezoCLI.dll",
        ]
        if self.has_strain_gauge:
            dlls.append("Thorlabs.MotionControl.KCube.StrainGaugeCLI.dll")

        for dll in dlls:
            clr.AddReference(os.path.join(kinesis_path, dll))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> tuple[bool, str]:
        """
        Discover and connect to the KPZ101 and KSG101.
        Configures the system for Closed Loop position control.

        Returns:
            (True, "Connected successfully") on success.
            (False, error_message)           on failure.
        """
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
        from Thorlabs.MotionControl.KCube.PiezoCLI import KCubePiezo
        from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import PiezoControlModeTypes

        try:
            DeviceManagerCLI.BuildDeviceList()

            # --- KPZ101 piezo driver ---
            self.piezo = KCubePiezo.CreateKCubePiezo(self.piezo_serial)
            self.piezo.Connect(self.piezo_serial)

            if not self.piezo.IsSettingsInitialized():
                self.piezo.WaitForSettingsInitialized(5000)

            self.piezo.GetPiezoConfiguration(self.piezo_serial)
            self.piezo.StartPolling(250)
            self.piezo.EnableDevice()
            time.sleep(0.5)

            # --- KSG101 strain gauge ---
            if self.has_strain_gauge:
                from Thorlabs.MotionControl.KCube.StrainGaugeCLI import KCubeStrainGauge

                self.strain = KCubeStrainGauge.CreateKCubeStrainGauge(self.strain_serial)
                self.strain.Connect(self.strain_serial)

                if not self.strain.IsSettingsInitialized():
                    self.strain.WaitForSettingsInitialized(5000)

                self.strain.GetStrainGaugeConfiguration(self.strain_serial)
                self.strain.StartPolling(250)
                self.strain.EnableDevice()
                time.sleep(0.5)

                print(f"KSG101 {self.strain_serial} connected.")

            # Closed Loop: Required for precise micrometer stepping
            self.piezo.SetPositionControlMode(PiezoControlModeTypes.CloseLoop)
            time.sleep(0.5)

            self.is_connected = True
            print(f"KPZ101 {self.piezo_serial} connected in Closed Loop mode.")
            return True, "Connected successfully"

        except Exception as exc:
            return False, str(exc)

    def move_to_um(self, position_um: float) -> bool:
        """
        Command the piezo to an absolute micrometer position using percentage.
        Clamped to [0.0, MAX_TRAVEL_UM].

        Returns:
            True on success, False on failure or if not connected.
        """
        if not self.is_connected:
            return False

        position_um = max(0.0, min(float(position_um), self.MAX_TRAVEL_UM))
        
        # Convert micrometers to a percentage (0.0 to 100.0)
        target_pct = (position_um / self.MAX_TRAVEL_UM) * 100.0
        target = System.Convert.ToDecimal(float(target_pct))

        try:
            self.piezo.SetPercentageTravel(target)
            return True
        except Exception as exc:
            print(f"Piezo move error: {exc}")
            return False

    def get_displacement(self) -> float | None:
        """
        Read the true piezo displacement from the KSG101 strain gauge 
        by decoding the raw 16-bit ADC value.

        Returns:
            Displacement in micrometers (µm), or None if read fails.
        """
        if not self.has_strain_gauge or self.strain is None:
            return None
        try:
            # Get the raw ADC reading (0 to 32767)
            raw_reading = int(str(self.strain.GetReading()))
            
            # Convert raw value back into Micrometers
            displacement_um = (raw_reading / 32767.0) * self.MAX_TRAVEL_UM
            return displacement_um
        except Exception as exc:
            print(f"Strain gauge read error: {exc}")
            return None

    def disconnect(self):
        """Safely relax the crystal, stop polling, and cleanly close connections."""
        from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import PiezoControlModeTypes
        
        if self.piezo and self.is_connected:
            try:
                # Return to 0 and switch to Open Loop before shutdown to relax crystal
                self.piezo.SetPositionControlMode(PiezoControlModeTypes.OpenLoop)
                time.sleep(0.5)
                self.piezo.SetOutputVoltage(System.Convert.ToDecimal(0.0))
                self.piezo.StopPolling()
                self.piezo.Disconnect()
            except Exception as exc:
                print(f"Piezo disconnect error: {exc}")

        if self.strain is not None:
            try:
                self.strain.StopPolling()
                self.strain.Disconnect()
            except Exception as exc:
                print(f"Strain gauge disconnect error: {exc}")
            finally:
                self.strain = None

        self.is_connected = False
        print("Hardware disconnected.")