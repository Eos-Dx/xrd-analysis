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
from scipy.ndimage import uniform_filter1d

from eosdxanalysis.calibration.materials import q_peaks_ref_dict

from eosdxanalysis.calibration.utils import radial_profile_unit_conversion
from eosdxanalysis.calibration.utils import real_position_from_q

from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.azimuthal_integration import azimuthal_integration

from eosdxanalysis.preprocessing.utils import find_center

from eosdxanalysis.preprocessing.image_processing import pad_image
from eosdxanalysis.preprocessing.utils import quadrant_fold

DEFAULT_FILE_FORMAT = "txt"



def sample_detector_distance(
        image, center=None, beam_rmax=None,
        distance_approx=None, pixel_size=None,
        wavelength_nm=None, calibration_material=None,
        doublet_approx_min_factor=None,
        doublet_approx_max_factor=None,
        doublet_width=None, doublet_height=None,
        visualize=False, start_radius=None,
        end_radius=None, padding=None, height=None, save=False,
        image_fullpath=None, doublet_only=False):
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

    rmax : int
        Maximum radius to analyze

    distance_approx : float
        Approximate sample-to-detector distance

    output_shape : (int, int)
        output shape of polar warp.

    visualize : bool
        Flag to display plots image and azimuthal integration profile.
    """

    # Look up calibration material q-peaks reference data
    try:
        q_peaks_ref = q_peaks_ref_dict[calibration_material]
    except KeyError as err:
        print("Calibration material {} not found!".format(
                                        calibration_material))

    wavelength_angstroms = wavelength_nm*1e1

    radial_profile = azimuthal_integration(
            image, center=center, beam_rmax=beam_rmax,
            start_radius=start_radius, end_radius=end_radius)

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
        if prominences.size >= 1:
            # Get the peak index of the doublet in the main array
            doublet_peak_index = doublet_peak_indices_approx[0] + start_index
        else:
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
                singlet_width_input = input("Width:" )
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
        X = (singlet_peak_indices_approx[:len(q_peaks_found_indices)] * \
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
                "sample_distance_m": sample_distance_m,
                "beam_center": center,
                }

        # Write calibration results to file
        with open(output_filepath, "w") as outfile:
            outfile.writelines(json.dumps(results_dict, indent=4))

    if visualize:
        # Calculate to q-range
        q_range = radial_profile_unit_conversion(
                radial_profile.size,
                wavelength_nm,
                sample_distance_m,
                radial_units="q_per_nm")


        title = "Beam masked image [dB+1]"
        fig = plt.figure(title)
        plt.title(title)

        plt.imshow(20*np.log10(masked_image.astype(np.float64)+1), cmap="gray")
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


def sample_distance_calibration_dir(
            image_fullpath=None,
            calibration_material=None,
            wavelength_nm=None,
            pixel_size=None,
            beam_rmax=None,
            rmax=None,
            distance_approx=None,
            center=None,
            doublet_height=None,
            doublet_width=None,
            visualize=False,
            start_radius=None,
            end_radius=None,
            save=False,
            print_result=False,
            doublet_only=False,
            sample_distance=None,
            file_format=DEFAULT_FILE_FORMAT):

    if file_format != "txt" and file_format != "tiff" and file_format != "npy":
        raise ValueError("Choose ``txt``, ``npy``, or ``tiff`` file format.")

    # Instantiate Calibration class
    calibrator = Calibration(calibration_material=material,
            wavelength_nm=wavelength_nm, pixel_size=pixel_size)

    # Load calibration image
    if file_format == "txt":
        image = np.loadtxt(image_fullpath, dtype=np.float64)
    if file_format == "npy":
        image = np.load(image_fullpath)
    elif file_format == "tiff":
        image = io.imread(image_fullpath).astype(np.float64)

    # Run calibration procedure
    sample_distance = calibrator.sample_detector_distance(
            image,
            beam_rmax=beam_rmax,
            rmax=rmax,
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
        sample_distance=None,
        beam_center=None,
        filter_size=16,
        strip_width=8,
        save=False,
        doublet_only=False):
    """
    Calculate the detector spacing
    """
    if file_format != "txt" and file_format != "tiff" and file_format != "npy":
        raise ValueError("Choose ``txt``, ``npy``, or ``tiff`` file format.")

    # Instantiate Calibration class
    calibrator = Calibration(calibration_material=material,
            wavelength_nm=wavelength_nm, pixel_size=pixel_size)

    # Load calibration image
    if file_format == "txt":
        image = np.loadtxt(image_fullpath, dtype=np.float64)
    if file_format == "npy":
        image = np.load(image_fullpath)
    elif file_format == "tiff":
        image = io.imread(image_fullpath).astype(np.float64)

    # Get q-peaks reference
    q_peaks_ref_per_ang = calibrator.q_peaks_ref

    # Average the doublets
    doublets = np.array(q_peaks_ref_per_ang.get("doublets"))
    if doublets.size > 0:
        doublet_q_per_ang = np.mean(doublets)
        doublet_q_per_nm = 10*doublet_q_per_ang

    start_row = int(beam_center[0] - strip_width/2)
    end_row = int(beam_center[0] + strip_width/2)
    horizontal_strip = image[start_row:end_row, :]

    horizontal_profile = np.mean(horizontal_strip, axis=0)
    filtered_profile =  uniform_filter1d(horizontal_profile, filter_size)

    if doublet_only:
        try:
            peak_location = np.where(filtered_profile == np.max(filtered_profile))[0][0]
        except ValueError as err:
            raise ValueError("Doublet peak not found.")

        if visualize:
            plt.plot(filtered_profile)
            plt.scatter(peak_location, filtered_profile[peak_location])
            plt.show()

        # Calculate distance from beam center to doublet peak in pixel units
        beam_doublet_distance_m = real_position_from_q(
                q_per_nm=doublet_q_per_nm, sample_distance_m=sample_distance,
                wavelength_nm=wavelength_nm)

        detector_size_m = 256*pixel_size
        beam_position_m = beam_center[1]*pixel_size
        doublet_position_det2_m = peak_location*pixel_size
        # beam_doublet_distance = (detector_size - beam_position) + detector_spacing + \
        #       doublet_position_det2_m
        detector_spacing_m = beam_doublet_distance_m - (detector_size_m - beam_position_m) - \
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
            "--material", type=str, default="silver_behenate",
            help="The calibration material")
    parser.add_argument(
            "--pixel_size", type=float,
            help="The physical pixel size in meters.")
    parser.add_argument(
            "--wavelength_nm", type=float, required=True,
            help="The wavelength in nanometers.")
    parser.add_argument(
            "--center", type=str, default=None,
            help="The center of the diffraction pattern.")
    parser.add_argument(
            "--beam_rmax", type=int,
            help="The radius to block out the beam.")
    parser.add_argument(
            "--rmax", type=int,
            help="The radius to block out the beam.")
    parser.add_argument(
            "--doublet_width", type=int,
            help="The doublet width to look for.")
    parser.add_argument(
            "--doublet_height", type=int,
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
            "--height", type=float, default=None,
            help="The height of the doublet peak")
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
            help="The y-position of the beam.")
    parser.add_argument(
            "--sample_distance", type=float, default=None,
            help="The distance between the sample and the detector.")
    parser.add_argument(
            "--filter_size", type=int,
            help="The size of the uniform filter.")
    parser.add_argument(
            "--file_format", type=str, default=DEFAULT_FILE_FORMAT, required=False,
            help="The data file format:``txt`` (default), or  ``tiff``.")

    args = parser.parse_args()

    # Set variables based on input arguments
    image_fullpath = args.image_fullpath
    material = args.material
    wavelength_nm = args.wavelength_nm
    pixel_size = args.pixel_size
    beam_rmax = args.beam_rmax
    center = ",".split(args.center) if args.center else None
    rmax = args.rmax
    distance_approx = args.distance_approx
    doublet_height = args.doublet_height
    doublet_width = args.doublet_width
    visualize = args.visualize
    start_radius = args.start_radius
    end_radius = args.end_radius
    save = args.save
    print_result= args.print_result
    doublet_only = args.doublet_only
    file_format = args.file_format
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
        sample_distance_calibration_dir(
                image_fullpath=image_fullpath,
                calibration_material=material,
                wavelength_nm=wavelength_nm,
                pixel_size=pixel_size,
                beam_rmax=beam_rmax,
                rmax=rmax,
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
                file_format=file_format,
                )
    else:
        detector_spacing_calibration(
                image_fullpath=image_fullpath,
                calibration_material=material,
                wavelength_nm=wavelength_nm,
                sample_distance=sample_distance,
                beam_center=beam_center,
                filter_size=filter_size,
                save=save,
                doublet_only=doublet_only,
                )
