"""
Example with polar grid
"""
import os

import numpy as np
import pandas as pd

from datetime import datetime
from collections import OrderedDict

import argparse
import json
import glob

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


feature_list = [
        "peak_location_beam",
        "peak_location_9A",
        "peak_location_5A",
        "peak_location_4A",
        "peak_decay_beam",
        "peak_width_9A",
        "peak_width_5A",
        "peak_width_4A",
        ]

# Create dictionary of peaks and their theoretical locations in angstroms
peak_location_angstrom_dict = {
        "peak_location_beam": 1,
        "peak_location_9A": 9.8e-10,
        "peak_location_5A": 5.1e-10,
        "peak_location_4A": 4.5e-10,
        }

## Convert to pixel locations
#peak_location_pixel_dict = {}
#for peak_name, peak_location in peak_location_angstrom_dict.items():
#    peak_location_pixel_dict[peak_name] = feature_pixel_location(peak_location)

def gauss_1d(
        r_position,
        peak_position=0,
        peak_amplitude=1,
        peak_std=1,
        ):
    """
    1-D Gaussian
    """
    gau = peak_amplitude*np.exp(
            -1/2*((r_position-peak_position)/peak_std)**2)
    return gau

def exp_1d(
        r_position,
        peak_position=0,
        peak_amplitude=1,
        peak_decay=1,
        ):
    """
    1-D Exponential decay function
    """
    exp_1d_func = peak_amplitude*np.exp(-(r_position-peak_position)/peak_decay)
    return exp_1d_func

def keratin_function_1d(
        r_position,
        peak_amplitude_beam,
        peak_decay_beam,
        peak_position_9A,
        peak_amplitude_9A,
        peak_std_9A,
        peak_position_5A,
        peak_amplitude_5A,
        peak_std_5A,
        peak_position_4A,
        peak_amplitude_4A,
        peak_std_4A,
        ):
    """
    1-D keratin xrd pattern function
    """

    # Beam Exponential function
    peak_position_beam = 0
    exp_beam = gauss_1d(
            r_position,
            peak_position_beam,
            peak_amplitude_beam,
            peak_decay_beam,
            )
    # 9A Gaussian function
    gauss_9A = gauss_1d(
            r_position,
            peak_position_9A,
            peak_amplitude_9A,
            peak_std_9A,
            )
    # 5A Gaussian function
    gauss_5A = gauss_1d(
            r_position,
            peak_position_5A,
            peak_amplitude_5A,
            peak_std_5A,
            )
    # 4A Gaussian function
    gauss_4A = gauss_1d(
            r_position,
            peak_position_4A,
            peak_amplitude_4A,
            peak_std_4A,
            )

    keratin = exp_beam + gauss_9A + gauss_5A + gauss_4A
    return keratin

def best_fit(
        profile_1d, rmin, rmax, p0_dict=None, lower_bounds_dict=None,
        upper_bounds_dict=None):
    """
    Parameters
    ----------

    rmin : int
        Beam cutoff radius

    rmax : int
        Outer area cutoff radius
    """

    # Initial guess
    p0_dict = {
        "peak_amplitude_beam":  1e3,
        "peak_decay_beam":      50,
        "peak_position_9A":     43,
        "peak_amplitude_9A":    20,
        "peak_std_9A":          10,
        "peak_position_5A":     85,
        "peak_amplitude_5A":    20,
        "peak_std_5A":          15,
        "peak_position_4A":     100,
        "peak_amplitude_4A":    20,
        "peak_std_4A":          15, 
        }

    p_lower_bounds_dict = {
        "peak_amplitude_beam":  0,
        "peak_decay_beam":      0,
        "peak_position_9A":     40,
        "peak_amplitude_9A":    0,
        "peak_std_9A":          1,
        "peak_position_5A":     80,
        "peak_amplitude_5A":    0,
        "peak_std_5A":          1,
        "peak_position_4A":     95,
        "peak_amplitude_4A":    0,
        "peak_std_4A":          1,
    }

    p_upper_bounds_dict = {
        "peak_amplitude_beam":  1e20,
        "peak_decay_beam":      1e3,
        "peak_position_9A":     46,
        "peak_amplitude_9A":    500,
        "peak_std_9A":          20,
        "peak_position_5A":     90,
        "peak_amplitude_5A":    500,
        "peak_std_5A":          20,
        "peak_position_4A":     105,
        "peak_amplitude_4A":    500,
        "peak_std_4A":          20,
        }

    # Get parameters and bounds
    p0 = np.fromiter(p0_dict.values(), dtype=np.float64)

    p_bounds = (
            # Lower bounds
            np.fromiter(p_lower_bounds_dict.values(), dtype=np.float64),
            # Upper bounds
            np.fromiter(p_upper_bounds_dict.values(), dtype=np.float64),
        )

    # Apply beam and outer area mask
    # Offset rmin and rmax by 1 to avoid discretization issues
    rmin_used = rmin + 1
    rmax_used = rmax - 1
    profile_masked = profile_1d[rmin_used:rmax_used]
    x_data = np.arange(rmin_used, rmax_used)
    y_data = profile_masked.astype(np.float64)

    popt, pcov = curve_fit(
            keratin_function_1d, x_data, y_data, p0, bounds=p_bounds)

    # Create popt_dict so we can have keys
    popt_dict = OrderedDict()
    idx = 0
    for key, value in p0_dict.items():
        popt_dict[key] = popt[idx]
        idx += 1

    return popt_dict, pcov

def run_peak_fitting(input_path, output_path, rmin, rmax):
    """
    Run peak fitting on 1-D azimuthal integration profiles
    """
    # Get filepath list
    filepath_list = glob.glob(os.path.join(input_path, "*.txt"))
    # Sort files list
    filepath_list.sort()

    # Create dataframe to collect extracted features
    columns = ["Filename"] + feature_list
    df = pd.DataFrame(data={}, columns=columns, dtype=object)

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Store output directory info
    # Create output directory if it does not exist
    output_dir = "best_fit_{}".format(timestamp)
    output_path = os.path.join(output_path, output_dir)

    # Data output path
    data_output_dir = "data"
    data_output_path = os.path.join(output_path, data_output_dir)
    os.makedirs(data_output_path, exist_ok=True)

    # Image output path
    image_output_dir = "images"
    image_output_path = os.path.join(output_path, image_output_dir)
    os.makedirs(image_output_path, exist_ok=True)

    # Loop over files list
    for filepath in filepath_list:
        filename = os.path.basename(filepath)

        # Read 1-D azimuthal integration profile data frome file
        radial_profile = np.loadtxt(filepath, dtype=np.float64)

        # Get best fit curve to 1-D azimuthal integration profile
        popt_dict, pcov = best_fit(radial_profile, rmin, rmax)

        rmin_used = rmin + 1
        rmax_used = rmax - 1

        x_data = np.arange(rmin_used, rmax_used)

        best_fit_cropped = keratin_function_1d(x_data, **popt_dict)

        # Pad with zeros
        best_fit_curve = np.zeros(128)
        best_fit_curve[rmin_used:rmax_used] = best_fit_cropped

        # Save data to file
        data_output_filename = "radial_{}".format(filename)
        data_output_filepath = os.path.join(data_output_path,
                data_output_filename)

        np.savetxt(data_output_filepath, best_fit_curve)

        # Save image preview to file
        # Plot comparison
        plot_title = "Azimuthal Integration Best Fit Curve Comparison {}".format(filename)
        fig = plt.figure(plot_title)
        plt.plot(range(radial_profile.size), radial_profile, label="original")
        plt.plot(range(best_fit_curve.size), best_fit_curve, label="best fit")

        plt.xlabel("Radius [pixel units]")
        plt.ylabel("Mean Intensity [photon count]")
        plt.legend()

        plt.title(plot_title)

        # Set image output file
        image_output_filename = "radial_best_fit_comparison_{}".format(filename) + ".png"
        image_output_filepath = os.path.join(image_output_path,
                image_output_filename)

        # Save image preview to file
        plt.savefig(image_output_filepath)

        # Add extracted features to dataframe
        # df.loc[len(df.index)+1] = [filename] + [radial_profile]

        plt.close(fig)

    # Save dataframe to csv
    # df.to_csv(output_filepath, index=False)



if __name__ == '__main__':
    """
    Run azimuthal integration on an image
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None, required=True,
            help="The path to data to extract features from")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The output path to save radial profiles and peak features")
    parser.add_argument(
            "--rmin", default=None, type=int, required=True,
            help="The rmin value defining the beam cutoff")
    parser.add_argument(
            "--rmax", default=None, type=int, required=True,
            help="The rmax value defining the outer diffraction pattern cutoff")

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    rmin = args.rmin
    rmax = args.rmax

    run_peak_fitting(
        input_path=input_path,
        output_path=output_path,
        rmin=rmin,
        rmax=rmax
        )
