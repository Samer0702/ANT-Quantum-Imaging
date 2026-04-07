import sys
import time
import clr
import System

# --- CONFIG ---
kinesis_path = r"C:\Program Files\Thorlabs\Kinesis"
sys.path.append(kinesis_path)

PIEZO_SERIAL = "29252595"
SG_SERIAL = "59500024"

TARGET_UM = 2.0  # test move

# --- LOAD DLLs ---
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.GenericPiezoCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.PiezoCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.StrainGaugeCLI")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.KCube.PiezoCLI import KCubePiezo
from Thorlabs.MotionControl.KCube.StrainGaugeCLI import KCubeStrainGauge
from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import PiezoControlModeTypes


def main():
    print("=== BUILD DEVICE LIST ===")
    DeviceManagerCLI.BuildDeviceList()

    print("=== CREATE DEVICES ===")
    piezo = KCubePiezo.CreateKCubePiezo(PIEZO_SERIAL)
    sg = KCubeStrainGauge.CreateKCubeStrainGauge(SG_SERIAL)

    if piezo is None or sg is None:
        print("ERROR: Could not create devices")
        return

    try:
        # --- CONNECT ---
        print("=== CONNECT ===")
        piezo.Connect(PIEZO_SERIAL)
        sg.Connect(SG_SERIAL)

        piezo.WaitForSettingsInitialized(5000)
        sg.WaitForSettingsInitialized(5000)

        # --- LOAD CONFIGURATION (CRITICAL FIX) ---
        print("=== LOAD CONFIGURATION ===")
        try:
            piezo.GetPiezoConfiguration(piezo.DeviceID)
            sg.GetStrainGaugeConfiguration(sg.DeviceID)
            time.sleep(1)
        except Exception as e:
            print("Config load warning:", e)

        piezo.EnableDevice()
        piezo.StartPolling(5)
    
        sg.EnableDevice()
        sg.StartPolling(5)


        print('=== Set Close-Loop Control Mode ===')
        piezo.SetPositionControlMode(PiezoControlModeTypes.CloseLoop)
        time.sleep(0.5)
        print(piezo.GetPositionControlMode())


        print("\n=== TESTING ===")




        ###################################################
#
#        piezo.SetOutputVoltage(System.Convert.ToDecimal(0.0))
#        time.sleep(5)
#        piezo.RequestVoltage()
#        time.sleep(1)
#
#        sg.RequestReading()
#        time.sleep(1)
#        print("Strain Gauge Reading :",'Voltage:' ,piezo.GetOutputVoltage(), 'reading:', sg.GetReading())
#
#
#        #=======================================================
#
#        piezo.SetOutputVoltage(System.Convert.ToDecimal(45.0))
#        time.sleep(5)
#        piezo.RequestVoltage()
#        time.sleep(1)
#        sg.RequestReading()
#        time.sleep(1)
#        print("Strain Gauge Reading :",'Voltage:' ,piezo.GetOutputVoltage(), 'reading:', sg.GetReading())
#
#
#        #=======================================================
#
#        piezo.SetOutputVoltage(System.Convert.ToDecimal(75.0))
#        time.sleep(5)
#        piezo.RequestVoltage()
#        time.sleep(1)
#        sg.RequestReading()
#        time.sleep(1)
#        print("Strain Gauge Reading :",'Voltage:' ,piezo.GetOutputVoltage(), 'reading:', sg.GetReading())


        ##############################################################
        

#=======================================================

        if piezo.IsSetOutputVoltageActive():
            print("IsSetOutputVoltageActive(): yes.")
        else:
            print("IsSetOutputVoltageActive(): no.")

        if piezo.IsSetPositionActive():
            print("IsSetPositionActive(): yes.")
        else:
            print("IsSetPositionActive(): no.")

        print(piezo.GetMaxTravel())

    except Exception as e:
        print("ERROR:", e)

    finally:
        print("\n=== SHUTDOWN ===")
        try:
            piezo.StopPolling()
            piezo.Disconnect()
        except:
            pass

        try:
            sg.StopPolling()
            sg.Disconnect()
        except:
            pass

        print("Done.")


if __name__ == "__main__":
    main()