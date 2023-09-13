"""
Code to calibrate X-ray diffraction setup using calibration samples data
"""
import argparse
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from skimage import io
from skimage.transform import warp_polar
from skimage.transform import warp
from skimage.transform import EuclideanTransform

from scipy.signal import find_peaks
from scipy.ndimage import convolve1d

from eosdxanalysis.calibration.materials import q_peaks_ref_dict

from eosdxanalysis.calibration.materials import CALIBRATION_MATERIAL_LIST

from eosdxanalysis.calibration.units_conversion import radial_profile_unit_conversion
from eosdxanalysis.calibration.units_conversion import real_position_from_q
from eosdxanalysis.calibration.units_conversion import DEFAULT_SAMPLE_DISTANCE_COLUMN_NAME

from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.azimuthal_integration import azimuthal_integration
from eosdxanalysis.preprocessing.azimuthal_integration import DEFAULT_DET_XSIZE

from eosdxanalysis.preprocessing.utils import find_center
from eosdxanalysis.preprocessing.utils import relative_center

from eosdxanalysis.preprocessing.image_processing import pad_image
from eosdxanalysis.preprocessing.utils import quadrant_fold


DEFAULT_SINGLET_HEIGHT = 5
DEFAULT_SINGLET_WIDTH = 2
DEFAULT_DOUBLET_APPROX_MIN_FACTOR = 0.5
DEFAULT_DOUBLET_APPROX_MAX_FACTOR = 2.0
DEFAULT_DOUBLET_WIDTH = 4
DEFAULT_DOUBLET_HEIGHT = 8
DEFAULT_STRIP_WIDTH = 16

VALID_FILE_FORMATS = [
        "txt",
        "tif",
        "tiff",
        "npy",
        ]

def sample_detector_distance(
        image,
        center=None,
        beam_rmax=None,
        distance_approx=None,
        pixel_size=None,
        wavelength_nm=None,
        calibration_material=None,
        doublet_approx_min_factor=DEFAULT_DOUBLET_APPROX_MIN_FACTOR,
        doublet_approx_max_factor=DEFAULT_DOUBLET_APPROX_MAX_FACTOR,
        doublet_width=DEFAULT_DOUBLET_WIDTH,
        doublet_height=DEFAULT_DOUBLET_HEIGHT,
        visualize=False,
        start_radius=None,
        end_radius=None,
        padding=None,
        save=False,
        image_fullpath=None,
        doublet_only=False):
    """
    Calculate the sample-to-detector distance for a calibration sample

    Steps performed:
    1. Use sample-to-detector distance estimate to estimate location of AgBH doublet
    2. Run find peaks in this location to find the doublet
    3. Use doublet peak location to determine sample-to-detector distance

    Parameters
    ----------

    image : ndarray

    center : (float, float)
        Center of the diffraction pattern

    beam_rmax : int
        Maximum beam extent

    distance_approx : float
        Approximate sample-to-detector distance

    output_shape : (int, int)
        output shape of polar warp.

    visualize : bool
        Flag to display plots image and azimuthal integration profile.
    """
    if calibration_material not in CALIBRATION_MATERIAL_LIST:
        raise ValueError("{} not in calibration material library.".format(
            calibration_material))

    # Look up calibration material q-peaks reference data
    try:
        q_peaks_ref = q_peaks_ref_dict[calibration_material]
    except KeyError as err:
        print("Calibration material {} not found!".format(
                                        calibration_material))

    wavelength_angstroms = wavelength_nm*1e1

    # Set center if None
    if type(center) is type(None):
        center = find_center(image, rmin=start_radius, rmax=end_radius)

    radial_profile = azimuthal_integration(
            image, center=center, beam_rmax=beam_rmax,
            start_radius=start_radius, end_radius=end_radius, fill=np.nan)

    # Use the doublet peak location only
    doublet_peak_index = None
    if doublet_only:
        # Convert doublet average in q units to 2*theta units
        # Average the doublets
        q_doublets = np.array(q_peaks_ref.get("doublets"))
        if q_doublets.size == 2:
            q_doublets_avg = np.mean(q_doublets)
        else:
            raise ValueError("Incorrect amount of doublets found.")

        # Calculate theta_n
        theta_n = np.arcsin(q_doublets_avg*wavelength_angstroms/(4*np.pi))
        # Calculate the approximate distance of the doublet average
        doublet_distance_approx = distance_approx * np.tan(2*theta_n).reshape(-1,1)
        # Convert to pixel units
        doublet_pixel_location_approx = int(np.round(doublet_distance_approx / pixel_size))

        # Create a subset of the radial intensity to look for doublet
        start_index = \
                int(np.round(doublet_approx_min_factor*doublet_pixel_location_approx))
        end_index = \
                int(np.round(doublet_approx_max_factor*doublet_pixel_location_approx))
        # Ensure there is no overshoot with the end index
        end_index = radial_profile.size - 1 if end_index >= radial_profile.size else end_index
        radial_profile_subset = radial_profile[start_index:end_index]

        if doublet_height is None:
            # Get the height of the peak based on distance estimate
            doublet_height = radial_profile[doublet_pixel_location_approx]

        doublet_peak_indices_approx, properties = find_peaks(
                radial_profile_subset, width=doublet_width,
                height=doublet_height)

        # Check how many prominent peaks were found
        prominences = properties.get("prominences")
        try:
            if prominences.size >= 1:
                # Get the peak index of the doublet in the main array
                doublet_peak_index = doublet_peak_indices_approx[0] + start_index
        except:
            raise ValueError("Doublet peak not found.")

        # Now use location of doublet to calculate sample-to-detector distance
        doublet_distance = doublet_peak_index * pixel_size

        sample_distance_m = doublet_distance / np.tan(2*theta_n)
    else:
        # Average the doublets
        doublets = np.array(q_peaks_ref.get("doublets"))
        if doublets.size > 0:
            doublets_avg = np.array(np.mean(doublets)).flatten()
        singlets = np.array(q_peaks_ref.get("singlets")).flatten()
        # Join the singlets and doublets averages into a single array
        q_peaks_avg = np.sort(np.concatenate([singlets, doublets_avg]))

        singlet_height = DEFAULT_SINGLET_HEIGHT
        singlet_width = DEFAULT_SINGLET_WIDTH

        peak_finding_loop = True
        while peak_finding_loop:
            # Find peaks
            singlet_peak_indices_approx, properties = find_peaks(
                    radial_profile, width=singlet_width,
                    height=singlet_height)

            if visualize:
                # Plot radial intensity profile
                plt.plot(np.arange(radial_profile.size),
                        20*np.log10(radial_profile+1))
                # Plot found peaks
                plt.scatter(
                        singlet_peak_indices_approx,
                        20*np.log10(
                            radial_profile[singlet_peak_indices_approx]+1))
                plt.show()
            print("Found peaks at pixel radius locations:")
            print(singlet_peak_indices_approx)

            correct = input("Are the found peak locations correct? (Y/n) ")
            if correct in ("Y", "y", "yes", "Yes"):
                # Stop the loop
                peak_finding_loop = False
            else:
                print("Please enter in a new height and width.")
                singlet_height_input = input("Height: ")
                singlet_width_input = input("Width: " )
                if singlet_height_input != "":
                    singlet_height = int(singlet_height_input)
                if singlet_width_input != "":
                    singlet_width = int(singlet_width_input)

        # Match found peaks with reference peaks
        q_peaks_found = q_peaks_avg[:len(singlet_peak_indices_approx)]

        # Check if found peaks are correct
        peak_id_loop = True
        while peak_id_loop:
            print("Found q-peaks:")
            print(q_peaks_found)

            print("q-peaks reference:")
            print("index\tq-value")
            print("=====\t=======")
            for idx in range(len(q_peaks_avg)):
                print("{}\t{}".format(idx,q_peaks_avg[idx]))

            correct = input("Are the found peak q values correct? (Y/n) ")

            if correct in ("Y", "y", "yes", "Yes"):
                # Stop the loop
                peak_id_loop = False
                q_peaks_found_indices = singlet_peak_indices_approx
            else:
                q_peaks_found_indices_list = input(
                    "Please enter the indices of the found peaks: ")
                q_peaks_found_indices = q_peaks_found_indices_list.split(",")
                q_peaks_found_indices = np.array(q_peaks_found_indices, dtype=int)
                q_peaks_found = q_peaks_avg[q_peaks_found_indices]

        # Use Angstroms units
        theta_n = np.arcsin(q_peaks_found*wavelength_angstroms/(4*np.pi))
        Y = np.tan(2*theta_n).reshape(-1,1)
        # Set x values as the measured r peaks
        X = (singlet_peak_indices_approx[:len(q_peaks_found)] * \
                pixel_size).reshape(-1,1)

        # Now perform linear regression, line goes through the origin
        # so intercept = 0
        linreg = LinearRegression(fit_intercept=False)
        linreg.fit(X, Y)
        score = linreg.score(X, Y)
        if print_result:
            print("R^2 score:",score)

        # Get the slope
        coef = linreg.coef_
        slope = coef[0][0]
        # The slope is the inverse of the sample-to-detector distance
        sample_distance_m = 1/slope

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
        results_dict = {
                DEFAULT_SAMPLE_DISTANCE_COLUMN_NAME: sample_distance_m,
                "beam_center": list(center),
                "score": score if not doublet_only else None,
                }

        # Write calibration results to file
        with open(output_filepath, "w") as outfile:
            outfile.writelines(json.dumps(results_dict, indent=4))

    if visualize:
        # Convert to q-range
        sample_distance_mm = sample_distance_m * 1e3
        q_range = radial_profile_unit_conversion(
                radial_count=radial_profile.size,
                sample_distance_mm=sample_distance_mm,
                wavelength_nm=wavelength_nm,
                pixel_size=pixel_size,
                radial_units="q_per_nm")

        title = "Beam masked image [dB+1]"
        fig = plt.figure(title)
        plt.title(title)

        plt.imshow(20*np.log10(image.astype(np.float64)+1), cmap="gray")
        # Beam center
        plt.scatter(center[1], center[0], color="green")
        if doublet_peak_index:
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
            plt.legend()

        plt.xlabel("Horizontal Position [pixel length]")
        plt.ylabel("Vertical Position [pixel length]")

        title = "Radial Intensity Profile versus real-space distance"
        fig = plt.figure(title)
        plt.title(title)

        # Plot azimuthal integration 1-D profile [pixel length]
        plt.plot(np.arange(radial_profile.size),
                20*np.log10(radial_profile+1),
                label="1-D profile")

        # Plot a marker for the most prominent peak
        if doublet_peak_index:
            plt.scatter(doublet_peak_index,
                    20*np.log10(radial_profile[doublet_peak_index]+1),
                    color="red", marker=".", s=250,
                    label="Doublet peak at {} pixel lengths".format(doublet_peak_index))
            plt.legend()

        plt.xlabel("Position [pixel length]")
        plt.ylabel(r"Mean intensity [photon count, dB+1]")

        # Plot azimuthal integration 1-D profile [q]

        title = "Radial Intensity Profile versus q"
        fig = plt.figure(title)
        plt.title(title)

        plt.plot(q_range, 20*np.log10(radial_profile+1),
                label="1-D profile")

        if doublet_peak_index:
            # Plot a marker for the most prominent peak
            plt.scatter(q_range[doublet_peak_index],
                    20*np.log10(radial_profile[doublet_peak_index]+1),
                    color="red", marker=".", s=250,
                    label="Doublet peak location:\n" +\
                            "Theoretical: {} ".format(
                                np.around(q_doublets_avg*10, decimals=1)) + \
                            r"$\mathrm{{nm}^{-1}}$"
                        )
            plt.legend()

        plt.xlabel(r"q $\mathrm{{nm}^{-1}}$")
        plt.ylabel(r"Mean intensity [photon count, dB+1]")

        plt.show()

    return sample_distance_m

def sample_distance_calibration_on_a_file(
            image_fullpath=None,
            calibration_material=None,
            wavelength_nm=None,
            pixel_size=None,
            beam_rmax=None,
            distance_approx=None,
            center=None,
            doublet_height=DEFAULT_DOUBLET_HEIGHT,
            doublet_width=DEFAULT_DOUBLET_WIDTH,
            visualize=False,
            start_radius=None,
            end_radius=None,
            save=False,
            print_result=False,
            doublet_only=False,
            sample_distance=None,
            ):

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

    # Run calibration procedure
    sample_distance = sample_detector_distance(
            image,
            calibration_material=calibration_material,
            beam_rmax=beam_rmax,
            pixel_size=pixel_size,
            wavelength_nm=wavelength_nm,
            distance_approx=distance_approx,
            center=center,
            doublet_height=doublet_height,
            doublet_width=doublet_width,
            visualize=visualize,
            start_radius=start_radius,
            end_radius=end_radius,
            save=save,
            image_fullpath=image_fullpath,
            doublet_only=doublet_only)

    if print_result:
        print("{} m".format(sample_distance))

    return sample_distance

def detector_spacing_calibration(
        image_fullpath=None,
        calibration_material=None,
        wavelength_nm=None,
        pixel_size=None,
        doublet_height=DEFAULT_DOUBLET_HEIGHT,
        doublet_width=DEFAULT_DOUBLET_WIDTH,
        sample_distance=None,
        strip_width=DEFAULT_STRIP_WIDTH,
        beam_center=None,
        filter_size=16,
        start_radius=None,
        end_radius=None,
        save=False,
        doublet_only=False):
    """
    Calculate the detector spacing
    """
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

    # Look up calibration material q-peaks reference data
    try:
        q_peaks_ref = q_peaks_ref_dict[calibration_material]
    except KeyError as err:
        print("Calibration material {} not found!".format(
                                        calibration_material))

    wavelength_angstroms = wavelength_nm*1e1

    singlets = np.array(q_peaks_ref.get("singlets")).flatten()

    # Average the doublets
    doublets = np.array(q_peaks_ref.get("doublets"))
    if doublets.size > 0:
        doublets_avg = np.array(np.mean(doublets)).flatten()
    singlets = np.array(q_peaks_ref.get("singlets")).flatten()
    # Join the singlets and doublets averages into a single array
    q_peaks_avg = np.sort(np.concatenate([singlets, doublets_avg]))

    # Load calibration image
    if file_format == "txt":
        image = np.loadtxt(image_fullpath, dtype=np.float64)
    if file_format == "npy":
        image = np.load(image_fullpath)
    elif file_format in ["tif", "tiff"]:
        image = io.imread(image_fullpath).astype(np.float64)

    start_row = int(beam_center[0] - strip_width/2)
    end_row = int(beam_center[0] + strip_width/2)
    horizontal_strip = image[start_row:end_row, :]

    horizontal_profile = np.mean(horizontal_strip, axis=0)

    if type(filter_size) == int:
        weights = np.ones(filter_size)/filter_size
        filtered_profile = convolve1d(horizontal_profile, weights, cval=np.nan)
    else:
        filtered_profile = horizontal_profile

    doublet_q_per_nm = doublets_avg[0] * 10

    if doublet_only:
        doublet_peak_indices_approx, properties = find_peaks(
                filtered_profile, width=doublet_width,
                height=doublet_height)

        try:
            peak_location = doublet_peak_indices_approx[0]
        except:
            raise ValueError("Doublet peak not found.")

        if visualize:
            plt.plot(filtered_profile)
            plt.scatter(peak_location, filtered_profile[peak_location])
            plt.show()

        sample_distance_mm = sample_distance * 1e3

        # Calculate distance from beam center to doublet peak in pixel units
        beam_doublet_distance_mm = real_position_from_q(
                q_per_nm=doublet_q_per_nm, sample_distance_mm=sample_distance_mm,
                wavelength_nm=wavelength_nm)

        beam_doublet_distance_m = beam_doublet_distance_mm * 1e-3

        detector_size_m = image.shape[0]*pixel_size
        beam_position_m = beam_center[1]*pixel_size
        doublet_position_det2_m = peak_location*pixel_size
        # beam_doublet_distance = (detector_size - beam_position) + detector_spacing + \
        #       doublet_position_det2_m
        detector_spacing_m = beam_doublet_distance_m + beam_position_m - detector_size_m - \
                doublet_position_det2_m
    else:
        raise NotImplementedError

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
        results_dict = {
                "detector_spacing_m": detector_spacing_m,
                }

        # Write calibration results to file
        with open(output_filepath, "w") as outfile:
            outfile.writelines(json.dumps(results_dict, indent=4))

    if print_result:
        print("{} m".format(detector_spacing_m))

    return detector_spacing_m


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
            "--center", type=str, default=None,
            help="The center of the diffraction pattern.")
    parser.add_argument(
            "--beam_rmax", type=int,
            help="The radius to block out the beam (in pixel units).")
    parser.add_argument(
            "--doublet_width", type=float, default=DEFAULT_DOUBLET_WIDTH,
            help="The doublet width to look for.")
    parser.add_argument(
            "--doublet_height", type=float, default=DEFAULT_DOUBLET_HEIGHT,
            help="The doublet height to look for.")
    parser.add_argument(
            "--distance_approx", type=float,
            help="The approximate sample-to-detector distance.")
    parser.add_argument(
            "--start_radius", type=int, default=0,
            help="The maximum radius to analyze.")
    parser.add_argument(
            "--end_radius", type=int, default=None,
            help="The maximum radius to analyze.")
    parser.add_argument(
            "--visualize", action="store_true",
            help="Plot calibration results to screen")
    parser.add_argument(
            "--save", action="store_true",
            help="Store distance to file.")
    parser.add_argument(
            "--print_result", action="store_true",
            help="Print sample distance.")
    parser.add_argument(
            "--doublet_only", action="store_true",
            help="Calibrate distance according to doublet peak location only.")
    parser.add_argument(
            "--secondary_detector", action="store_true",
            help="Calibrate the secondary detector position.")
    parser.add_argument(
            "--beam_center", type=str, default=None,
            help="The position of the beam on detector 1.")
    parser.add_argument(
            "--sample_distance", type=float, default=None,
            help="The distance between the sample and the detector (multi-detector case).")
    parser.add_argument(
            "--filter_size", type=int,
            help="The size of the uniform filter.")

    args = parser.parse_args()

    # Set variables based on input arguments
    image_fullpath = args.image_fullpath
    calibration_material = args.calibration_material
    wavelength_nm = args.wavelength_nm
    pixel_size = args.pixel_size
    beam_rmax = args.beam_rmax
    if type(beam_rmax) == type(None):
        beam_rmax = 0
    center_arg = args.center
    if center_arg:
        center = np.array(center_arg.strip("( )").split(",")).astype(float)
    else:
        center = None
    distance_approx = args.distance_approx
    doublet_height = args.doublet_height
    doublet_width = args.doublet_width
    visualize = args.visualize
    start_radius = args.start_radius
    end_radius = args.end_radius
    save = args.save
    print_result= args.print_result
    doublet_only = args.doublet_only
    secondary_detector = args.secondary_detector

    # In case primary_detector = False
    sample_distance = args.sample_distance
    beam_center_arg = args.beam_center
    if beam_center_arg:
        beam_center = np.array(beam_center_arg.strip("( )").split(",")).astype(float)
    else:
        beam_center = None
    filter_size = args.filter_size

    if not secondary_detector:
        sample_distance_calibration_on_a_file(
                image_fullpath=image_fullpath,
                calibration_material=calibration_material,
                wavelength_nm=wavelength_nm,
                pixel_size=pixel_size,
                beam_rmax=beam_rmax,
                distance_approx=distance_approx,
                center=center,
                doublet_height=doublet_height,
                doublet_width=doublet_width,
                visualize=visualize,
                start_radius=start_radius,
                end_radius=end_radius,
                save=save,
                print_result=print_result,
                doublet_only=doublet_only,
                )
    else:
        detector_spacing_calibration(
                image_fullpath=image_fullpath,
                calibration_material=calibration_material,
                wavelength_nm=wavelength_nm,
                pixel_size=pixel_size,
                sample_distance=sample_distance,
                beam_center=beam_center,
                doublet_height=doublet_height,
                doublet_width=doublet_width,
                start_radius=start_radius,
                end_radius=end_radius,
                filter_size=filter_size,
                save=save,
                doublet_only=doublet_only,
                )
