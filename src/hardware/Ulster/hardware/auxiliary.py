"""
Geometry helper functions
"""

import math


from ctypes import cdll, c_int, c_short, c_char_p

def in_circle(x, y, cx, cy, radius):
    """Return True if (x, y) is within a circle centered at (cx, cy) with the given radius."""
    return math.sqrt((x - cx) ** 2 + (y - cy) ** 2) < radius


def in_square(x, y, cx, cy, side):
    """Return True if (x, y) is within a square centered at (cx, cy) with side length 'side'."""
    half = math.floor(side / 2) + 1
    return abs(x - cx) < half and abs(y - cy) <= half


def in_x_line(x, y, cx, cy, side):
    """Return True if (x, y) is on the horizontal central line of a square mask."""
    half = math.floor(side / 2) + 1
    return y == cy and abs(x - cx) < half


def in_y_line(x, y, cx, cy, side):
    """Return True if (x, y) is on the vertical central line of a square mask."""
    half = math.floor(side / 2) + 1
    return x == cx and abs(y - cy) < half



def scan():
    input('Check X-ray beam, detector, and sample position. Press Enter to continue...')

    # Move sample to a load position before scanning
    load_x = -100000
    load_y = y_limit_neg
    move_stage(lib, serial_num, x_chan, y_chan, load_x, load_y, move_timeout)

    # ------------------------------
    # Grid Scanning Loop
    # ------------------------------
    print('Starting grid scan...')
    points_processed = 0

    for r in range(Nrepeat):
        sample_id = input('Load new sample. Enter sample identifier (P number): ')

        for y in range(y_min, y_max + step, step):
            # If scanning only a horizontal line, set y to centre
            if xline_mask_en:
                y = centre_y

            for x in range(x_min, x_max + step, step):
                # If scanning only a vertical line, set x to centre
                if yline_mask_en:
                    x = centre_x
                # Diagonal modifier: set x equal to y if enabled
                if diag_en:
                    x = y

                # Convert grid coordinates for file naming
                x_s = int(x * coordunit / step)
                y_s = int(y * coordunit / step)

                # Check if the point is within the active mask region
                if ((circ_mask_en and in_circle(x, y, centre_x, centre_y, radius)) or
                        (sq_mask_en and in_square(x, y, centre_x, centre_y, sides)) or
                        (xline_mask_en and in_x_line(x, y, centre_x, centre_y, sides)) or
                        (yline_mask_en and in_y_line(x, y, centre_x, centre_y, sides)) or
                        (not (circ_mask_en or sq_mask_en or xline_mask_en or yline_mask_en))):

                    # Calculate absolute stage coordinates
                    x_abs = x + x_zero
                    y_abs = y + y_zero

                    # Move stage to the calculated position
                    move_stage(lib, serial_num, x_chan, y_chan, x_abs, y_abs, move_timeout)

                    # Construct the filename for the capture data
                    filestring1 = f"{x_s}-{y_s}"
                    filename = os.path.join(filepath,
                                            f"{filestring0a}{sample_id}{filestring0c}{filestring1}{filestring2}")

                    # Optionally, copy filename to clipboard
                    # pyperclip.copy(filename)

                    print(f"Capturing point ({x_s}, {y_s})")
                    if capture_en:
                        capture_point(dev, pixet, Nframes, Nseconds, filename)
                    points_processed += 1
                else:
                    print(f"Skipping point ({x_s}, {y_s}) - outside mask")
        print(f"{points_processed} points processed.")

        # Move sample back to load position after scanning
        move_stage(lib, serial_num, x_chan, y_chan, load_x, load_y, move_timeout)
        if r >= Nrepeat - 1:
            print("Exiting scan.")

    # ------------------------------
    # Clean-up and Shutdown
    # ------------------------------
    lib.BDC_Close(serial_num)
    if sim_en:
        lib.TLI_UninitializeSimulations()
    if capture_en:
        pixet.exitPixet()
        pypixet.exit()