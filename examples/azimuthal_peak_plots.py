"""
Code to plot azimuthal integration curve fitting
"""
import os
import glob

import argparse

import matplotlib.pyplot as plt
import numpy as np

from examples.azimuthal_peak_fitting import best_fit
from examples.azimuthal_peak_fitting import keratin_function_1d
from eosdxanalysis.preprocessing.utils import azimuthal_integration



def plot_azimuthal_curve_fitting_dir(input_path):
    rmin = 24
    rmax = 110

    # Shrink due to aliasing
    rmin_use = rmin + 1
    rmax_use = rmax - 1

    filepath_list = glob.glob(os.path.join(input_path, "*.txt"))

    for filepath in filepath_list:
        # Load 1-D profile
        profile_1d = np.loadtxt(filepath, dtype=np.float64)

        # Get best-fit curve parameters
        popt_dict, pcov = best_fit(profile_1d, rmin=rmin, rmax=rmax)

        # Generate best-fit curve data
        y = keratin_function_1d(
            np.arange(rmin_use, rmax_use),
            popt_dict["peak_amplitude_beam"],
            popt_dict["peak_std_beam"],
            popt_dict["peak_position_9A"],
            popt_dict["peak_amplitude_9A"],
            popt_dict["peak_std_9A"],
            popt_dict["peak_position_5A"],
            popt_dict["peak_amplitude_5A"],
            popt_dict["peak_std_5A"],
            popt_dict["peak_position_4A"],
            popt_dict["peak_amplitude_4A"],
            popt_dict["peak_std_4A"],)

        plt.plot(np.arange(profile_1d.size), profile_1d, label="original")
        plt.plot(np.arange(rmin_use, rmax_use), y, label="fit")
        plt.ylim([-10, 300])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    """
    Plot azimuthal integration curve fitting
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None, required=True,
            help="The path to data to extract features from")

    args = parser.parse_args()

    input_path = args.input_path

    plot_azimuthal_curve_fitting_dir(
        input_path=input_path,
        )

