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
        "peak_width_beam",
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

def keratin_function_1d(
        r_position,
        peak_amplitude_beam,
        peak_std_beam,
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

    # Beam Gaussian function
    peak_position_beam = 0
    gauss_beam = gauss_1d(
            r_position,
            peak_position_beam,
            peak_amplitude_beam,
            peak_std_beam,
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

    keratin = gauss_beam + gauss_9A + gauss_5A + gauss_4A
    return keratin

def best_fit(
        # profile_1d, p0_dict, lower_bounds_dict, upper_bounds_dict,
        profile_1d, rmin=0, rmax=0):
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
        "peak_amplitude_beam":  1e2,
        "peak_std_beam":        20,
        "peak_position_9A":     43,
        "peak_amplitude_9A":    227,
        "peak_std_9A":          10,
        "peak_position_5A":     85,
        "peak_amplitude_5A":    260,
        "peak_std_5A":          15,
        "peak_position_4A":     100,
        "peak_amplitude_4A":    240,
        "peak_std_4A":          15, 
        }

    p_lower_bounds_dict = {
        "peak_amplitude_beam":  0,
        "peak_std_beam":        0,
        "peak_position_9A":     35,
        "peak_amplitude_9A":    0,
        "peak_std_9A":          0,
        "peak_position_5A":     75,
        "peak_amplitude_5A":    0,
        "peak_std_5A":          0,
        "peak_position_4A":     95,
        "peak_amplitude_4A":    0,
        "peak_std_4A":          0, 
    }

    p_upper_bounds_dict = {
        "peak_amplitude_beam":  1e20,
        "peak_std_beam":        128,
        "peak_position_9A":     56,
        "peak_amplitude_9A":    300,
        "peak_std_9A":          50,
        "peak_position_5A":     90,
        "peak_amplitude_5A":    300,
        "peak_std_5A":          40,
        "peak_position_4A":     105,
        "peak_amplitude_4A":    300,
        "peak_std_4A":          30, 
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
    profile_masked = profile_1d[rmin:rmax]
    x_data = np.arange(rmin, rmax)
    y_data = profile_masked.astype(np.float64)

    popt, pcov = curve_fit(
            keratin_function_1d, x_data, y_data, p0, bounds=p_bounds)
            # keratin_function_1d, x_data, y_data, p0)

    # Create popt_dict so we can have keys
    popt_dict = OrderedDict()
    idx = 0
    for key, value in p0_dict.items():
        popt_dict[key] = popt[idx]
        idx += 1

    return popt_dict, pcov


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

    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    run_feature_extraction(
        input_path=input_path,
        output_path=output_path,
        )
