import os
import sys
import time
import numpy as np

from ctypes import cdll, c_int, c_short, c_char_p
# Set DEV mode: True will use dummy functions for testing, False will use the real implementations.
DEV = False

sys.path.insert(0, 'C:\\Program Files\\PIXet Pro')

# ------------------------------
# Dummy Functions for DEV mode
# ------------------------------
if DEV:
    def init_detector(capture_enabled):
        """Dummy detector initialization."""
        print("DEV mode: Dummy init_detector called.")
        return True, True

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
        time.sleep(home_timeout)
        print("DEV mode: Dummy home_stage called.")
        return 0, 0

    def move_stage(lib, serial_num, x_chan, y_chan, x_new, y_new, move_timeout=1):
        """Dummy stage move."""
        time.sleep(move_timeout)
        print(f"DEV mode: Dummy move_stage called. Pretending to move to ({x_new}, {y_new}).")
        return x_new, y_new

    def capture_point(dev, pixet, Nframes, Nseconds, filename):
        """
        Dummy capture routine that generates two 2D Gaussian distributions on a 100x100 grid.

        One Gaussian will have a peak intensity greater than 1e6 and the other will have a peak
        intensity less than 1e6. Both Gaussians are generated with random centers and spreads,
        then combined into one matrix which is saved to a text file.
        """
        # Create a 100x100 grid
        x = np.arange(100)
        y = np.arange(100)
        X, Y = np.meshgrid(x, y)

        # Generate first Gaussian with intensity > 1e6
        x0_1 = np.random.uniform(0, 100)
        y0_1 = np.random.uniform(0, 100)
        sigma_x1 = np.random.uniform(5, 15)
        sigma_y1 = np.random.uniform(5, 15)
        amplitude1 = np.random.uniform(1e6 + 1, 2e6)  # Ensuring peak > 1e6
        gaussian1 = amplitude1 * np.exp(-(((X - x0_1) ** 2) / (2 * sigma_x1 ** 2) +
                                          ((Y - y0_1) ** 2) / (2 * sigma_y1 ** 2)))

        # Generate second Gaussian with intensity < 1e6
        x0_2 = np.random.uniform(0, 100)
        y0_2 = np.random.uniform(0, 100)
        sigma_x2 = np.random.uniform(5, 15)
        sigma_y2 = np.random.uniform(5, 15)
        amplitude2 = np.random.uniform(1e5, 1e6 - 1)  # Ensuring peak < 1e6
        gaussian2 = amplitude2 * np.exp(-(((X - x0_2) ** 2) / (2 * sigma_x2 ** 2) +
                                          ((Y - y0_2) ** 2) / (2 * sigma_y2 ** 2)))

        # Combine the two Gaussians
        combined_matrix = gaussian1 + gaussian2

        # Save the combined matrix to a text file with formatted output
        np.savetxt(filename, combined_matrix, fmt='%.6f')

        print(f"DEV mode: Dummy capture_point called. Saved a 100x100 combined Gaussian matrix to {filename}.")

# ------------------------------
# Real Functions for Device Operation
# ------------------------------
else:
    import pypixet
    import time
    from pylablib.devices import Thorlabs

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
                return None, None
            dev = devices[0]
            print('Detector initialized.')
            return pixet, dev

    def capture_point(dev, pixet, Nframes, Nseconds, filename):
        """Capture data at the current point using the detector."""
        print(f'Capturing at {filename} ...')
        rc = dev.doSimpleIntegralAcquisition(Nframes, Nseconds, pixet.PX_FTYPE_AUTODETECT, filename)
        if rc == 0:
            print('Capture successful.')
        else:
            print('Capture error:', rc)

    def init_stage(simulation_enabled, serial_num):
        """
        Initialize the XY stage using pylablib.

        Parameters:
          simulation_enabled (bool): If True, enable simulation.
          serial_num (str): Serial number of the stage.

        Returns:
          stage: An instance of Thorlabs.KinesisMotor with an open connection.
        """

        # Optionally, list devices and check if your device is connected:
        devices = Thorlabs.list_kinesis_devices()
        if not devices:
            print("No Thorlabs devices found!")
            return None
        else:
            print("Detected devices:")
            for dev in devices:
                print(dev)

        # Create the stage object (using the provided serial number)
        stage = Thorlabs.KinesisMotor(serial_num)
        stage.open()  # Open the device connection
        time.sleep(2)  # Allow time for initialization
        return stage

    def home_stage(stage, home_timeout=60):
        """
        Home the XY stage on both axes and return the final positions.

        Parameters:
          stage: The stage object returned from init_stage.
          home_timeout (int): Maximum wait time in seconds for homing.

        Returns:
          (x_pos, y_pos): Tuple of the homed positions in mm.
        """
        print("Homing stage on X and Y axes...")
        # Home channel 1 (X axis) and channel 2 (Y axis)
        stage.home(channel=1)
        stage.home(channel=2)
        stage.wait_for_home(channel=1, timeout=home_timeout)
        stage.wait_for_home(channel=2, timeout=home_timeout)

        # Return the homed positions
        x_final = stage.get_position(channel=1, scale=True)
        y_final = stage.get_position(channel=2, scale=True)
        print(f"Final homed positions: X = {x_final} mm, Y = {y_final} mm")
        return x_final, y_final

    def move_stage(stage, x_new, y_new, move_timeout=60):
        """
        Move the stage to an absolute position (x_new, y_new) in mm.

        Parameters:
          stage: The stage object.
          x_new (float): Desired X position in mm.
          y_new (float): Desired Y position in mm.
          move_timeout (int): Maximum wait time in seconds for the move to complete.

        Returns:
          (x_pos, y_pos): Tuple with the final positions in mm.
        """
        print(f"Moving stage: Target X = {x_new} mm, Target Y = {y_new} mm")
        # Command the stage to move; the API automatically converts from mm to internal units
        stage.move_to(x_new * 10000, channel=2, scale=True)
        stage.move_to(y_new * 10000, channel=1, scale=True)

        stage.wait_move(channel=2, timeout=move_timeout)
        stage.wait_move(channel=1, timeout=move_timeout)

        # Read and return the final positions
        x_final = stage.get_position(channel=2, scale=True)
        y_final = stage.get_position(channel=1, scale=True)
        print(f"Final positions: X = {x_final} mm, Y = {y_final} mm")
        return x_final, y_final
