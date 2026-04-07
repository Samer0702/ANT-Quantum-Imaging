"""
piezo_control.py
--------------
Controls a Thorlabs KPZ101 K-Cube Piezo Controller and optionally reads
displacement from a KSG101 K-Cube Strain Gauge Reader via the Kinesis SDK.

Open loop  (strain_serial=None) : voltage commands only — no displacement data.
Closed loop (strain_serial set)  : voltage commands + real displacement readback
                                   from the strain gauge in micrometres (µm).

The KPZ101 and KSG101 must be connected together via the SMA feedback cable
for the strain gauge readings to reflect the actual piezo displacement.
"""

import os
import clr
import time
import System


class PiezoController:
    """
    Controls a Thorlabs KPZ101 piezo driver, with optional KSG101 strain
    gauge readback for real displacement measurement.

    Max voltage is hardware-dependent:
        KPZ101 standard       : 75 V
    Set MAX_VOLTAGE to match your unit.
    """

    MAX_VOLTAGE = 75.0   # Change to 150.0 for the high-voltage variant.
    MIN_VOLTAGE = 0.0

    def __init__(
        self,
        piezo_serial: str = "29252595",
        strain_serial: str | None = "59500024",
    ):
        """
        Args:
            piezo_serial:  Serial number of the KPZ101.
            strain_serial: Serial number of the KSG101.
                           Pass None to run in open loop (no displacement data).
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
        Discover and connect to the KPZ101 (and KSG101 if configured).

        The KPZ101 is always set to Open Loop voltage mode — the strain
        gauge provides *readback* only and does not drive a position servo.
        This is the correct mode for interferometric fringe scanning, where
        you command voltage steps and read the true displacement separately.

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

            # Open Loop: we command voltage; strain gauge gives displacement.
            self.piezo.SetPositionControlMode(PiezoControlModeTypes.OpenLoop)

            # --- KSG101 strain gauge (optional) ---
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

                print(
                    f"KSG101 {self.strain_serial} connected — "
                    "displacement readback active."
                )

            self.is_connected = True
            mode = "with strain gauge readback" if self.has_strain_gauge else "open loop (no strain gauge)"
            print(f"KPZ101 {self.piezo_serial} connected — {mode}.")
            return True, "Connected successfully"

        except Exception as exc:
            return False, str(exc)

    def set_voltage(self, voltage: float) -> bool:
        """
        Command the piezo to the requested voltage.
        Clamped to [MIN_VOLTAGE, MAX_VOLTAGE] for hardware safety.

        Returns:
            True on success, False on failure or if not connected.
        """
        if not self.is_connected:
            return False

        voltage = max(self.MIN_VOLTAGE, min(float(voltage), self.MAX_VOLTAGE))
        target = System.Convert.ToDecimal(voltage)

        try:
            self.piezo.SetOutputVoltage(target)
            return True
        except Exception as exc:
            print(f"Piezo set_voltage error: {exc}")
            return False

    def get_voltage(self) -> float:
        """
        Read the current output voltage from the KPZ101.

        Returns:
            Current voltage in volts, or 0.0 on error / not connected.
        """
        if not self.is_connected:
            return 0.0
        try:
            return float(str(self.piezo.GetOutputVoltage()))
        except Exception:
            return 0.0

    def get_displacement(self) -> float | None:
        """
        Read the true piezo displacement from the KSG101 strain gauge.

        Returns:
            Displacement in micrometres (µm), or None if the strain gauge
            is not connected or the read fails.
        """
        if not self.has_strain_gauge or self.strain is None:
            return None
        try:
            raw = self.strain.GetReading()
            return float(str(raw.Reading))
        except Exception as exc:
            print(f"Strain gauge read error: {exc}")
            return None

    def disconnect(self):
        """Zero the voltage, stop polling, and cleanly close both connections."""
        if self.piezo and self.is_connected:
            try:
                self.set_voltage(0.0)
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
        print("Piezo disconnected.")