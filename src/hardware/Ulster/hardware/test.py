'''
M30XY Simple Grid Scanning v0.3
Author: Gavin Sandison
Date of Creation: 2025-01-29
Date of Last Modification: 2025-03-18
Based on:
Thorlabs Example M30XY.py
Requires package: pypixet (for minipix detector)
Example Date of Creation(YYYY-MM-DD) 2024-02-01
Example Date of Last Modification on Github 2024-02-01
Version of Python: 3.9
Version of the Thorlabs SDK used: 1.14.44
==================
Example Description
This example use the C DLLs
Example runs the M30XY stage, this uses the Benchtop not integrated stage DLLs
'''

import os
import sys
import time
import math
import PyPixel
from ctypes import *
from ctypes import CDLL


def in_circle(x, y, centre_x, centre_y, radius, step):
    """Check if point (x, y) is within the circle defined by centre (centre_x, centre_y) and radius+step."""
    return math.sqrt((x - centre_x) ** 2 + (y - centre_y) ** 2) < radius  # - abs(step)/2


def in_square(x, y, centre_x, centre_y, sides):
    """Check if point (x, y) is within the square defined by center (centre_x, centre_y) and sides."""
    return (abs(x - centre_x) < math.floor(sides / 2) + 1) and (abs(y - centre_y) <= math.floor(sides / 2) + 1)


def x_central(x, y, centre_x, centre_y, sides):
    """Check if point (x, y) is within a single central line in x-axis and sides length"""
    return (y == centre_y) and (abs(x - centre_x) < math.floor(sides / 2) + 1)


def y_central(x, y, centre_x, centre_y, sides):
    """Check if point (x, y) is within a single central line in y-axis and sides length"""
    return (x == centre_x) and (abs(y - centre_y) < math.floor(sides / 2) + 1)


def main():
    #########################
    # Sample run parameters #
    #########################

    Nrepeat = 10  # No of times to repeat capture process i.e. no of samples to capture

    ###############################
    # Detector capture parameters #
    ###############################

    Nframes = 60  # Detector number of frames to integrate
    Nseconds = 1  # Detector capture time per frame
    # Detector output file path
    filepath = 'D:\OneDrive\OneDrive - Matur\General - Ulster\Temp'
    # Filename pre-string
    filestring0a = '20250218_U2_P1'  # 'AgBH_20250205_'  #'centering_test' ## First part of filename
    # filestring0b = '41' # Currently set to user input '...Enter P number: ' ## Second part of filename
    filestring0c = '_S8_2cm_60s_'  # '20250109_U2_P4_S8TUM_z1_2cm_60s_' #'_2cm_60s_' ## Third part of filename
    # Filename post-string
    filestring2 = '.txt'  ## Filename extension (also sets output format)
    capture_en = 0  # Enable capturing (disable for a faster test-mode without capture time) - USEFUL!

    #######################
    # XY stage parameters #
    #######################

    x_chan = c_short(2)  # Note X&Y channels swapped in current orientation
    y_chan = c_short(1)
    serial_num = c_char_p(b'101370874')  # Must match serial number of M30XY/M device
    x_zero = 29100  # X = 0 grid units position, in stage units (10000 = 1mm) Update 03/02/2025: 26000->28500 better centering
    y_zero = 37800  # Y = 0 grid units position, in stage units (10000 = 1mm)
    x_limit_pos = 50000  # Reduced limit for X travel to prevent crash (CAREFUL!)
    x_limit_neg = -150000
    y_limit_pos = 150000
    y_limit_neg = -150000
    x_min, x_max = 0, -160000  # Range of position grid, in stage units
    y_min, y_max = 0, -160000  # (negative because of stage orientation)
    sim_en = 0  # Enable if using simulation!
    home_timeout = 90  # x0.5s to try to home
    move_timeout = 40  # x0.5s to try to move to next position

    #########################
    # Grid capture settings #
    #########################

    """
    ############
    # Examples #
    ############

    1. Circular mask, 16mm dia (whole sample holder), 1.0mm grid
    step = -10000
    circ_mask_en = 1
    radius = 80000

    sq_mask_en = 0
    diag_en = 0
    xline_mask_en = 0
    yline_mask_en = 0

    2. Square mask, 6mm each side, 0.5mm grid
    step = -5000
    sq_mask_en = 1
    diag_en = 0
    sides = 60000

    circ_mask_en = 0
    xline_mask_en = 0
    yline_mask_en = 0

    3. Diagonal line mask, within 1.5mm square, 0.5mm grid (3 points, √0.5 = 0.7mm spacing)
    step = -5000
    sq_mask_en = 1
    diag_en = 1
    sides = 15000

    circ_mask_en = 0
    xline_mask_en = 0
    yline_mask_en = 0

    4. X-line mask, 1mm length, 0.1mm spacing (11 points? - test this)
    step = -1000
    xline_mask_en = 1
    sides = 10000

    yline_mask_en = 0
    circ_mask_en = 0
    sq_mask_en = 0
    diag_en = 0

    """

    # Grid resolution / position Step size (0.5mm = -5000 negative for correct orientation)
    step = -20000  # -10000 = 1mm
    coordunit = int(step / -1000)  # Convert grid steps to grid units (stage units / -1000)

    # Mask Settings
    # EITHER
    # Enable for circular field of grid points
    circ_mask_en = 1
    radius = 50000  # Define the radius of the circular mask (CAREFUL not to crash stage, must be positive, < 100000)
    # OR
    # Enable for square field of grid points
    sq_mask_en = 0
    diag_en = 0  # Modifier for a diagonal line of points inside square, not whole grid
    sides = 15000  # Define the side length of the square mask (CAREFUL!)
    # OR
    # Enable for line of points in x or y (uses variable: sides)
    xline_mask_en = 0
    # OR
    yline_mask_en = 0

    ####################################
    # Program use only - do not change #
    ####################################

    # For counting successful captures
    n = 0
    # Flag for breaking out of stage movement position checking loop once position tolerance reached
    xbreakflag = 0
    ybreakflag = 0

    # Title for output
    print(f'M30XY Simple Grid Scanning', end='\n\n')

    # Determine grid centre for masking
    centre_x = int((x_max - x_min) / 2)
    centre_y = int((y_max - y_min) / 2)

    # Detector initialisation
    if capture_en == 1:
        # Initialise Advacam detector API
        print('Pixet core init...', end='\n\n')
        pypixet.start()
        pixet = pypixet.pixet

        # Load Advacam detector devices (needs modification for multiple detectors)
        devices = pixet.devices()
        if devices[0].fullName() == 'FileDevice 0':
            print('No devices connected')
            pixet.exitPixet()
            pypixet.exit()
            exit()
        dev = devices[0]  # First of connected devices
        pixet = pypixet.pixet

        # Debug: Show pixet directories
        # print('pixet.appDataDir', pixet.appDataDir())
        # print('pixet.appDir', pixet.appDir())
        # print('pixet.configsDir', pixet.configsDir())
        # print('pixet.factoryDataDir', pixet.factoryDataDir())
        # print('pixet.logsDir()', pixet.logsDir())

        # Show device info
        print('Pixet Device Info:')
        di = pixet.DevInfo()
        rc = dev.deviceInfo(di)
        print(' name', di.name)
        print(' serial', di.serial)
        print(' type', di.type)
        print(' vendor', di.vendor)
        print()

    # XY Stage Initialisation
    try:
        """Import the correct version of the Thorlabs Kinesis library"""
        if sys.version_info < (3, 8):
            os.chdir(r'C:\Program Files\Thorlabs\Kinesis')
        else:
            os.add_dll_directory(r'C:\Program Files\Thorlabs\Kinesis')

        lib: CDLL = cdll.LoadLibrary('Thorlabs.MotionControl.Benchtop.DCServo.dll')
    except Exception as e:
        print(f'Error: {e}')

    # For simulations
    if sim_en == 1:
        lib.TLI_InitializeSimulations()

    # Open XY stage device
    if True:

        print()
        print(f'Homing...')

        for i in range(0, home_timeout, 1):  # check and wait further for move if nesc
            x_pos_dev = 0
            y_pos_dev = 0

            # Check that position is within tolerance of 0,0 (if so, exit loop)
            if (abs(0 - x_pos_dev) + abs(0 - y_pos_dev)) <= 3:
                print(f'Homed successfully', end='\n\n')
                break

            # Warn user if home timeout reached
            if (i >= home_timeout - 1):
                print(f'HOME TIMED OUT', end='\n\n')

        time.sleep(0.5)  # Wait for stage to settle before confirming final home position

        x_pos_dev = 0
        y_pos_dev = 0

        # Display home positions and wait to continue
        print(f'X position: {x_pos_dev} device units')
        print(f'Y position: {y_pos_dev} device units', end='\n\n')
        print('Check X-ray Beam On, Detector in correct position, He on...')
        input('Press Enter to continue...')
        print()
        print('Moving sample to load position')

        # Move to sample load position (-100000, -150000)
        # Set a new absolute move in device units
        x_pos_new = c_int(-100000)  # 100000 device units = 10.0mm of movement
        y_pos_new = c_int(y_limit_neg)  # 100000 device units = 10.0mm of movement


        for r in range(0, Nrepeat, 1):

            filestring0b = input('Load new sample, Confirm X-ray beam is on. Enter P number: ')

            print(f'Scanning...', end='\n\n')

            # Begin grid scan, point by point, X then Y
            for y in range(y_min, y_max + step, step):
                # to skip unnecessary y steps for x-line
                if xline_mask_en:
                    y = centre_y
                    ybreakflag = 1

                for x in range(x_min, x_max + step, step):
                    # to skip unnecessary x steps for y-line
                    if yline_mask_en:
                        x = centre_x
                        xbreakflag = 1

                    # For diagonal modifier
                    if diag_en == 1:
                        x = y
                        xbreakflag = 1

                    # Convert absolute x, y values to simple coords
                    x_s = int(x * coordunit / step)
                    y_s = int(y * coordunit / step)

                    # Continue to move and capture if point is within mask
                    if circ_mask_en and in_circle(x, y, centre_x, centre_y, radius, step) \
                            or sq_mask_en and in_square(x, y, centre_x, centre_y, sides) \
                            or xline_mask_en and x_central(x, y, centre_x, centre_y, sides) \
                            or yline_mask_en and y_central(x, y, centre_x, centre_y, sides) \
                            or circ_mask_en + sq_mask_en + xline_mask_en + yline_mask_en < 1:

                        # Add offsets
                        x_abs = x + x_zero
                        y_abs = y + y_zero

                        # This is broken - be careful not to exceed limits (= crash)
                        # Restrict move within limits
                        # if x_abs > x_limit_pos:
                        #    x_abs = x_limit_pos
                        # if x_abs < x_limit_neg:
                        #    x_abs = x_limit_neg
                        # if y_abs > y_limit_pos:
                        #    y_abs = y_limit_pos
                        # if y_abs < y_limit_neg:
                        #    y_abs = y_limit_neg

                        # Set a new absolute move in device units
                        x_pos_new = c_int(x_abs)  # 100000 device units = 10.0mm of movement
                        y_pos_new = c_int(y_abs)  # 100000 device units = 10.0mm of movement


                        time.sleep(0.5)  # wait for move


                        # Loop to check position and wait longer for move if necessary
                        for i in range(0, move_timeout, 1):


                            # Check if position is within tolerance of desired position
                            if (abs(x_abs - x_pos_dev) + abs(y_abs - y_pos_dev)) <= 4:
                                break

                            # Warn user if move timed out
                            if (i >= move_timeout - 1):
                                print(f'MOVE TIMED OUT', end='\n\n')

                        # Print the New Positions:
                        print(f'New X position: {x_pos_dev} device units')
                        print(f'New Y position: {y_pos_dev} device units', end='\n')

                        # Produce file path and file name string for capture data
                        strings = [str(x_s), '-', str(y_s)]
                        filestring1 = ''.join(strings)
                        filestringfull = ''.join(
                            [filepath, '/', filestring0a, filestring0b, filestring0c, filestring1, filestring2])

                        print(f'Capturing point ({x_s}, {y_s})')

                        # Capture from detector (if enabled)
                        if capture_en == 1:
                            # Pixet API capture for automatic operation
                            print(f'Capturing integration of {Nframes} frames x {Nseconds} seconds...', end=' ')
                            rc = dev.doSimpleIntegralAcquisition(Nframes, Nseconds, pixet.PX_FTYPE_AUTODETECT,
                                                                 filestringfull)
                            # print('dev.doSimpleIntegralAcquisition - end:', rc, '(0 is OK)')
                            if rc == 0:
                                print(f'Captured', end='\n')
                                n = n + 1
                            else:
                                print()
                                print(f'\nCAPTURE ERROR: {rc}', end='\n\n')

                    else:
                        print(f'Skipping point ({x_s}, {y_s}) - outside mask', end='\n')

                    if xbreakflag == 1:
                        break

                if ybreakflag == 1:
                    break

            print(f'{n} points successfully processed.')

            print('Moving sample to load position')

            # Move to sample load position (-100000, -150000)
            # Set a new absolute move in device units
            x_pos_new = c_int(-100000)  # 100000 device units = 10.0mm of movement
            y_pos_new = c_int(y_limit_neg)  # 100000 device units = 10.0mm of movement


            if r >= Nrepeat - 1:
                print('Exiting...')





if __name__ == '__main__':
    main()
