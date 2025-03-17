import os
import sys
import time
import pypixet

from ctypes import cdll, c_int, c_short, c_char_p
# Set DEV mode: True will use dummy functions for testing, False will use the real implementations.
DEV = True

# ------------------------------
# Dummy Functions for DEV mode
# ------------------------------
if DEV:
    def init_detector(capture_enabled):
        """Dummy detector initialization."""
        print("DEV mode: Dummy init_detector called.")
        return None, None

    def init_stage(sim_en, serial_num, x_chan, y_chan):
        """Dummy stage initialization."""
        print("DEV mode: Dummy init_stage called.")

        class DummyLib:
            def __getattr__(self, name):
                # Any method called on this dummy lib will simply print its call details.
                return lambda *args, **kwargs: print(f"DEV mode: Called {name} with args {args} and kwargs {kwargs}")

            def BDC_Close(self, serial_num):
                print("DEV mode: Dummy BDC_Close called.")

        return DummyLib()

    def home_stage(lib, serial_num, x_chan, y_chan, home_timeout):
        """Dummy homing routine."""
        print("DEV mode: Dummy home_stage called.")
        return 0, 0

    def move_stage(lib, serial_num, x_chan, y_chan, x_new, y_new, move_timeout):
        """Dummy stage move."""
        print(f"DEV mode: Dummy move_stage called. Pretending to move to ({x_new}, {y_new}).")
        return x_new, y_new

    def capture_point(dev, pixet, Nframes, Nseconds, filename):
        """Dummy capture routine."""
        print(f"DEV mode: Dummy capture_point called. Pretending to capture at {filename}.")

# ------------------------------
# Real Functions for Device Operation
# ------------------------------
else:
    def init_detector(capture_enabled):
        """Initialize the detector using the Pixet API if capture is enabled."""
        if capture_enabled:
            print('Initializing detector...')
            pypixet.start()
            pixet = pypixet.pixet
            devices = pixet.devices()
            if devices[0].fullName() == 'FileDevice 0':
                print('No devices connected')
                pixet.exitPixet()
                pypixet.exit()
                sys.exit()
            dev = devices[0]
            print('Detector initialized.')
            return pixet, dev
        return None, None

    def init_stage(sim_en, serial_num, x_chan, y_chan):
        """Initialize the XY stage by loading the Thorlabs DLL and setting up the device."""
        if sys.version_info < (3, 8):
            os.chdir(r'C:\Program Files\Thorlabs\Kinesis')
        else:
            os.add_dll_directory(r'C:\Program Files\Thorlabs\Kinesis')
        lib = cdll.LoadLibrary('Thorlabs.MotionControl.Benchtop.DCServo.dll')
        if sim_en:
            lib.TLI_InitializeSimulations()
        if lib.TLI_BuildDeviceList() != 0:
            print('Error building device list.')
        else:
            lib.BDC_Open(serial_num)
            lib.BDC_StartPolling(serial_num, x_chan, c_int(250))
            lib.BDC_StartPolling(serial_num, y_chan, c_int(250))
            lib.BDC_EnableChannel(serial_num, x_chan)
            lib.BDC_EnableChannel(serial_num, y_chan)
            time.sleep(0.5)
        return lib

    def home_stage(lib, serial_num, x_chan, y_chan, home_timeout):
        """Home the XY stage and wait until the stage is at (0,0) within tolerance."""
        print('Homing stage...')
        lib.BDC_Home(serial_num, x_chan)
        lib.BDC_Home(serial_num, y_chan)
        time.sleep(20)  # Initial wait for homing to start
        for i in range(home_timeout):
            lib.BDC_RequestPosition(serial_num, x_chan)
            lib.BDC_RequestPosition(serial_num, y_chan)
            time.sleep(0.5)
            x_pos = lib.BDC_GetPosition(serial_num, x_chan)
            y_pos = lib.BDC_GetPosition(serial_num, y_chan)
            if abs(x_pos) + abs(y_pos) <= 3:
                print('Homed successfully.')
                break
            if i == home_timeout - 1:
                print('Home timed out.')
        time.sleep(0.5)
        return lib.BDC_GetPosition(serial_num, x_chan), lib.BDC_GetPosition(serial_num, y_chan)

    def move_stage(lib, serial_num, x_chan, y_chan, x_new, y_new, move_timeout):
        """Move the stage to the absolute position (x_new, y_new) and wait for the move to complete."""
        x_pos_new = c_int(x_new)
        y_pos_new = c_int(y_new)
        lib.BDC_SetMoveAbsolutePosition(serial_num, x_chan, x_pos_new)
        lib.BDC_SetMoveAbsolutePosition(serial_num, y_chan, y_pos_new)
        time.sleep(0.25)
        lib.BDC_MoveAbsolute(serial_num, x_chan)
        lib.BDC_MoveAbsolute(serial_num, y_chan)
        for i in range(move_timeout):
            lib.BDC_RequestPosition(serial_num, x_chan)
            lib.BDC_RequestPosition(serial_num, y_chan)
            time.sleep(0.5)
            x_pos = lib.BDC_GetPosition(serial_num, x_chan)
            y_pos = lib.BDC_GetPosition(serial_num, y_chan)
            if abs(x_new - x_pos) + abs(y_new - y_pos) <= 4:
                break
            if i == move_timeout - 1:
                print('Move timed out.')
        return lib.BDC_GetPosition(serial_num, x_chan), lib.BDC_GetPosition(serial_num, y_chan)

    def capture_point(dev, pixet, Nframes, Nseconds, filename):
        """Capture data at the current point using the detector."""
        print(f'Capturing at {filename} ...')
        rc = dev.doSimpleIntegralAcquisition(Nframes, Nseconds, pixet.PX_FTYPE_AUTODETECT, filename)
        if rc == 0:
            print('Capture successful.')
        else:
            print('Capture error:', rc)

