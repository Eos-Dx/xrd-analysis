"""Script to visually check calibration settings:
    center coordinates
    sample-to-detector distance
"""
import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from skimage import io

from eosdxanalysis.calibration.materials import q_peaks_ref_dict

from eosdxanalysis.calibration.utils import real_position_from_q
from eosdxanalysis.calibration.units_conversion import DEFAULT_SAMPLE_DISTANCE_COLUMN_NAME
from eosdxanalysis.calibration.units_conversion import MM2M

VALID_FILE_FORMATS = [
        "txt",
        "tif",
        "tiff",
        "npy",
        ]


def run_visual_calibration_check(
        image_fullpath,
        calibration_material=None,
        center=None,
        sample_distance_mm=None,
        wavelength_nm=None,
        pixel_size=None,
        origin=None,
        save=None,
        ):
    """
    Runs a visual calibration check.
    """
    # Look up calibration material q-peaks reference data
    try:
        q_peaks_ref = q_peaks_ref_dict[calibration_material]
    except KeyError as err:
        print("Calibration material {} not found!".format(
                                        calibration_material))

    # Average the doublets
    q_doublets = np.array(q_peaks_ref.get("doublets"))
    if q_doublets.size > 0:
        q_doublets_avg = np.array(np.mean(q_doublets)).flatten()
    q_singlets = np.array(q_peaks_ref.get("singlets")).flatten()
    # Join the singlets and doublets averages into a single array
    q_peaks_avg = np.sort(np.concatenate([q_singlets, q_doublets_avg]))

    if q_doublets.size == 2:
        q_doublets_avg = np.mean(q_doublets)
    else:
        raise ValueError("Incorrect amount of doublets found.")

    # Convert from per angstrom to per nm
    q_doublets_per_nm = q_doublets_avg*1e1

    try:
        file_format = os.path.splitext(image_fullpath)[1][1:]
    except:
        raise ValueError("Error getting file format extension. \
                Check if input file path is valid.")

    if file_format not in VALID_FILE_FORMATS:
        raise ValueError("{} is not a valid file format! \
                Choose from: {}.".format(
                    file_format,
                    VALID_FILE_FORMATS,
                    ))

    # Load calibration image
    if file_format == "txt":
        image = np.loadtxt(image_fullpath, dtype=np.float64)
    if file_format == "npy":
        image = np.load(image_fullpath)
    elif file_format in ["tif", "tiff"]:
        image = io.imread(image_fullpath).astype(np.float64)

    doublet_peak_position = real_position_from_q(
            q_per_nm=q_doublets_per_nm,
            sample_distance_mm=sample_distance_mm,
            wavelength_nm=wavelength_nm)
    doublet_peak_index = doublet_peak_position * 1e-3 / pixel_size
    title = "2D Measurement [dB+1]"
    fig = plt.figure(title)
    plt.title(title)

    if origin == "lower":
        image = np.rot90(np.rot90(image))
        # Swap center coordinates as a hack
        center = center[1], center[0]

    plt.imshow(20*np.log10(image.astype(np.float64)+1), cmap="jet", origin=origin)
    # Beam center
    plt.scatter(center[1], center[0], color="green")
    # Doublet ring
    circle = plt.Circle(
            (center[1], center[0]),
            radius=doublet_peak_index,
            linestyle="--",
            fill=False,
            color="red",
            label="Doublet peak location:\n" + \
                    "Real-Space: {} pixel lengths\n".format(doublet_peak_index) + \
                    "Theoretical: {} ".format(
                        np.around(q_doublets_avg*10, decimals=1)) + \
                    r"$\mathrm{{nm}^{-1}}$"
            )
    ax = plt.gca()
    ax.add_artist(circle)

    # Single rings
    # Convert from per angstrom to per nm
    q_peaks_arr = np.array(q_peaks_avg) * 1e1
    for q_peak in q_peaks_arr[:10]:
        radius=real_position_from_q(
                q_per_nm=q_peak,
                sample_distance_mm=sample_distance_mm,
                wavelength_nm=wavelength_nm) / (pixel_size * 1e3)
        circle = plt.Circle(
            (center[1], center[0]),
            radius=radius,
            linestyle="--",
            fill=False,
            color="green",
            )
        ax = plt.gca()
        ax.add_artist(circle)

    plt.xlabel("Horizontal Position [pixel length]")
    plt.ylabel("Vertical Position [pixel length]")
    plt.show()

    if save:
        # Get the parent path of the calibration image
        image_path = os.path.dirname(image_fullpath)
        image_filename = os.path.basename(image_fullpath)
        parent_path = os.path.dirname(image_path)

        # Create the calibration results directory
        calibration_results_dir = "calibration_results"
        calibration_results_path = os.path.join(
                parent_path, calibration_results_dir)
        os.makedirs(calibration_results_path, exist_ok=True)

        # Set the calibration results output file properties
        output_filename = "{}.json".format(image_filename)
        output_filepath = os.path.join(
                calibration_results_path, output_filename)

        # Construct calibration results file content
        sample_distance_m = sample_distance_mm * MM2M
        if origin == "lower":
            center = 256 - center[0], 256 - center[1]
        results_dict = {
                DEFAULT_SAMPLE_DISTANCE_COLUMN_NAME: sample_distance_m,
                "beam_center": list(center),
                }

        # Write calibration results to file
        with open(output_filepath, "w") as outfile:
            outfile.writelines(json.dumps(results_dict, indent=4))


if __name__ == "__main__":
    """
    Commandline interface

    Directory specifications:
    - Specify the image full path

    Parameters specifications:
    - Provide the full path to the params file, or
    - provide a JSON-encoded string of parameters.

    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--image_fullpath", type=str, required=True,
            help="The full path to the raw image data")
    parser.add_argument(
            "--calibration_material", type=str, required=True,
            help="The calibration material")
    parser.add_argument(
            "--pixel_size", type=float, required=True,
            help="The physical pixel size in meters.")
    parser.add_argument(
            "--wavelength_nm", type=float, required=True,
            help="The wavelength in nanometers.")
    parser.add_argument(
            "--center_xy", type=str, default=None,
            help="The center of the diffraction pattern in xy notation.")
    parser.add_argument(
            "--center_mat", type=str, default=None,
            help="The center of the diffraction pattern in matrix notation.")
    parser.add_argument(
            "--sample_distance_mm", type=float,
            help="The sample-to-detector distance in [mm].")
    parser.add_argument(
            "--save", action="store_true",
            help="Save the center in matrix notation and \
                    sample-to-detector distance in [mm].")


    args = parser.parse_args()

    # Set variables based on input arguments
    image_fullpath = args.image_fullpath
    calibration_material = args.calibration_material
    wavelength_nm = args.wavelength_nm
    pixel_size = args.pixel_size

    # Center in xy notation
    center_xy_arg = args.center_xy
    if center_xy_arg:
        center_xy = np.array(center_xy_arg.strip("( )").split(",")).astype(float)
    else:
        center_xy = None
    # Center in matrix (row, column) notation
    center_mat_arg = args.center_mat
    if center_mat_arg:
        center_mat = np.array(center_mat_arg.strip("( )").split(",")).astype(float)
    else:
        center_mat = None
    if type(center_xy) != type(None):
        center = center_xy
        origin = "lower"

    sample_distance_mm = args.sample_distance_mm
    save = args.save

    # Run visuali calibration check
    run_visual_calibration_check(
        image_fullpath,
        calibration_material=calibration_material,
        center=center,
        sample_distance_mm=sample_distance_mm,
        wavelength_nm=wavelength_nm,
        pixel_size=pixel_size,
        origin=origin,
        save=save,
        )
