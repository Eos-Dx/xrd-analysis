"""
Code to check sensitivity of azimuthal integration to centering errors
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

from eosdxanalysis.preprocessing.utils import azimuthal_integration


def test_azimuthal_integration_centering_error(
        x_offset=0, y_offset=0, peak_position=0):
    """
    Test for sensitivity of azimuthal integration to centering errors
    off by pixel units
    """
    size = 256
    array_center = np.array([size]*2)/2 - 0.5
    x_label_offset = 20
    y_label_offset = -0.075

    x = np.linspace(-array_center[1], array_center[1], num=size)
    y = np.linspace(-array_center[0], array_center[0], num=size)

    YY, XX = np.meshgrid(y, x)
    RR = np.sqrt(YY**2 + XX**2)

    # Set the test function as exp(-(r^2))
    sigma = 40
    test_image = np.exp(-((RR-peak_position)/sigma)**2)

    # Calculate azimuthal integration profile
    test_center = (size/2-y_offset, size/2+x_offset)
    profile_test = azimuthal_integration(test_image, center=test_center)
    profile_orig = azimuthal_integration(test_image, center=array_center)

    # Find peaks
    profile_test_peak_indices, profile_test_peak_properties = find_peaks(
            profile_test, width=1)
    profile_orig_peak_indices, profile_orig_peak_properties = find_peaks(
            profile_orig, width=1)

    # Plot image and radial profile
    plot_title = "Azimuthal Integration Test Image Peak Radius {} Pixels".format(
            peak_position)
    fig, axs = plt.subplots(num=plot_title)
    plt.imshow(test_image)
    plt.title(plot_title)

    # 1-D profile with correct center
    plot_title = "Azimuthal Integration 1-D Profile"
    fig, axs = plt.subplots(num=plot_title)
    plt.scatter(np.arange(profile_orig.size), profile_orig, s=2)
    plt.scatter(profile_orig_peak_indices, profile_orig[profile_orig_peak_indices])

    # Show peak properties
    for idx in range(profile_orig_peak_indices.size):
        peak_index = profile_orig_peak_indices[idx]
        peak_position_str = "{}".format(peak_index)

        peak_value = profile_orig[peak_index]
        peak_value_str = "{:.4f}".format(peak_value)

        peak_width = profile_orig_peak_properties["widths"][idx]
        peak_width_str = "{:f}".format(peak_width)

        axs.annotate(
                "Position: {}\nAmplitude: {}\nWidth: {}".format(
                    peak_position_str,
                    peak_value_str,
                    peak_width_str,
                    ),
            (peak_index+x_label_offset, peak_value+y_label_offset))

    plt.title(plot_title)
    plt.xlabel("Radius [pixel units]")
    plt.ylabel("Mean intensity [photon count]")
    plt.ylim([-0.1, 1.1])

    # 1-D profile with center error
    plot_title = "Azimuthal Integration 1-D Profile" \
                " ({},{}) offset error".format(x_offset, y_offset)
    fig, axs = plt.subplots(num=plot_title)
    plt.scatter(np.arange(profile_test.size), profile_test, s=2)
    plt.scatter(profile_test_peak_indices, profile_test[profile_test_peak_indices])

    # Show peak properties
    for idx in range(profile_test_peak_indices.size):
        peak_index = profile_test_peak_indices[idx]
        peak_position_str = "{}".format(peak_index)

        peak_value = profile_test[peak_index]
        peak_value_str = "{:.4f}".format(peak_value)

        peak_width = profile_test_peak_properties["widths"][idx]
        peak_width_str = "{:f}".format(peak_width)

        axs.annotate(
                "Position: {}\nAmplitude: {}\nWidth: {}".format(
                    peak_position_str,
                    peak_value_str,
                    peak_width_str,
                    ),
            (peak_index+x_label_offset, peak_value+y_label_offset))

    plt.title(plot_title)
    plt.xlabel("Radius [pixel units]")
    plt.ylabel("Mean intensity [photon count]")
    plt.ylim([-0.1, 1.1])

    plt.show()


if __name__ == '__main__':
    """
    Run cancer predictions on preprocessed data
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--x_offset", type=float, default=0, required=False,
            help="x-offset for centering error.")
    parser.add_argument(
            "--y_offset", type=float, default=0, required=False,
            help="y-offset for centering error.")
    parser.add_argument(
            "--peak_position", type=float, default=0, required=False,
            help="Peak position in pixel units.")

    # Collect arguments
    args = parser.parse_args()

    x_offset = args.x_offset
    y_offset = args.y_offset
    peak_position = args.peak_position

    test_azimuthal_integration_centering_error(
            x_offset=x_offset,
            y_offset=y_offset,
            peak_position=peak_position,
            )
