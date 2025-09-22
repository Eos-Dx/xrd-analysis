import json
import os

from hardware_control import *


# ------------------------------
# Main Scanning Routine
# ------------------------------
def main():
    with open("file_parameters.json", "r") as f:
        params = json.load(f)

    print("Capture enabled:", params["capture_en"])

    with open("../resources/config/xy_stage.json", "r") as f:
        xy_stage_params = json.load(f)

    x_max, x_min, y_max, y_min = (
        params["x_max"],
        params["x_min"],
        params["y_max"],
        params["y_min"],
    )
    sim_en = params["sim_en"]
    serial_num = params["serial_number"]
    home_timeout = params["home_timeout"]
    move_timeout = params["move_timeout"]
    x_chan = params["x_channel"]
    y_chan = params["y_channel"]

    # Compute grid centre (for masking)
    centre_x = int((x_max - x_min) / 2)
    centre_y = int((y_max - y_min) / 2)

    # ------------------------------
    # Detector Initialization
    # ------------------------------
    pixet, dev = init_detector(params["capture_en"])

    # ------------------------------
    # XY Stage Initialization and Homing
    # ------------------------------
    lib = init_stage(sim_en, serial_num, x_chan, y_chan)
    home_stage(lib, serial_num, x_chan, y_chan, home_timeout)


if __name__ == "__main__":
    main()
