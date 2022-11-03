"""
Code to calibrate X-ray diffraction setup using calibration samples data
"""
import argparse

import numpy as np

from sklearn.linear_model import LinearRegression

from skimage.transform import warp_polar
from skimage.transform import warp
from skimage.transform import EuclideanTransform

from scipy.signal import find_peaks

from eosdxanalysis.calibration.materials import q_peaks_ref_dict

from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.beam_utils import azimuthal_integration
from eosdxanalysis.preprocessing.center_finding import find_center
from eosdxanalysis.preprocessing.image_processing import unwarp_polar
from eosdxanalysis.preprocessing.image_processing import crop_image
from eosdxanalysis.preprocessing.image_processing import quadrant_fold

PIXEL_WIDTH = 55e-6 # Pixel width in meters (it is 55 um)
WAVELENGTH = 1.5418E-10 # Wavelength in meters (1.5418 Angstroms)


class Calibration(object):
    """
    Calibration class to perform calibration for a set of calibration images

    q units are per Angstrom
    """

    def __init__(self, calibration_material, wavelength=1.5418e-10,
            pixel_width=PIXEL_WIDTH):
        """
        Initialize Calibration class
        """
        # Store source wavelength
        self.wavelength = wavelength

        # Store calibration material name
        self.calibration_material = calibration_material

        # Look up calibration material q-peaks reference data
        try:
            self.q_peaks_ref = q_peaks_ref_dict[calibration_material]
        except KeyError as err:
            print("Calibration material {} not found!".format(
                                            calibration_material))
        return super().__init__()

    def single_sample_detector_distance(
            self, image, beam_rmax=10, r_max=None, distance_approx=10e-3,
            output_shape=(360, 128), visualize=False):
        """
        Calculate the sample-to-detector distance for a single sample

        Steps performed:
        - Centerize the image
        - Warp to polar
        - Convert to 1D intensity vs. radius (pixel)
        - For each q-peak reference value, calculate
          the sample-to-detector distance
        - Return the mean, standard deviation, and values

        """
        wavelength = self.wavelength
        q_peaks_ref = self.q_peaks_ref

        # Find the image center
        center = find_center(image)
        array_center = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

        radial_intensity = azimuthal_integration(
                image, center=center, output_shape=output_shape)


#        # Set a maximum radius which we are interested in
#        if r_max is None:
#            raise ValueError("Must specify maximum radius.")
#        final_r_pixel = int(r_max)
# -
#        # Set a minimum radius which we are interested in
#        start_r_pixel = int(r_min) if r_min else 0

        # Set up radius linspace, where r is in pixels
        r_space_pixel = np.arange(radial_intensity.size)

        # Idea: Find the widest peak

        # Find the radial peaks
        # Find the peak of the first doublet
        # First, get a list of possible doublets with a width of oversample*8
        test_doublet_peak_indices, properties = find_peaks(
                radial_intensity, width=6)
        prominences = properties.get("prominences")
        if len(prominences) > 2:
            prominent_max = np.max(prominences)
            prominent_index = prominences.tolist().index(prominent_max)
        else:
            prominent_max = prominences
            prominent_index = 0
        # Take the the most prominent peak as the first doublet
        prominent_peak_index = test_doublet_peak_indices[prominent_index]

        # Now find all the peaks
        all_radial_peak_indices, _ = find_peaks(radial_intensity)

        # Average the doublets
        doublets = np.array(q_peaks_ref.get("doublets"))
        if doublets.size > 0:
            doublets_avg = np.array(np.mean(doublets)).flatten()
        singlets = np.array(q_peaks_ref.get("singlets")).flatten()
        # Join the singlets and doublets averages into a single array
        q_peaks_avg = np.sort(np.concatenate([singlets, doublets_avg]))

        # The first doublet will be the last peak
        final_index = all_radial_peak_indices.tolist().index(prominent_peak_index)
        # Count how many we are missing before the first doublet
        num_missing = len(q_peaks_avg[:-1]) - len(all_radial_peak_indices[:final_index])

        if num_missing < 0:
            raise ValueError("We found more peaks than in the reference!")
        elif num_missing == 0:
            radial_peak_indices = all_radial_peak_indices
        elif num_missing > 0:
            # Take subset
            # Note: need to do :final_index+1 since slicing is right-exclusive,
            # and num_missing-1: since slicing is left-inclusive
            radial_peak_indices = all_radial_peak_indices[num_missing:final_index+1]
            q_peaks_avg_subset = q_peaks_avg[num_missing:final_index+1]

        if visualize:
            import matplotlib.pyplot as plt

            title = "Beam masked image"
            fig = plt.figure(title)
            plt.title(title)
            mask = create_circular_mask(
                    image.shape[1], image.shape[0], center=center,
                    rmax=beam_rmax)
            masked_image = np.copy(image)
            masked_image[mask] = 0

            plt.imshow(20*np.log10(masked_image.astype(np.float64)+1), cmap="gray")
            plt.scatter(center[1], center[0], color="green")

            title = "Azimuthal integrated 1-d profile"
            fig = plt.figure(title)
            plt.title(title)

            plt.plot(20*np.log10(radial_intensity+1))
            plt.scatter(r_space_pixel[all_radial_peak_indices].ravel(),
                        20*np.log10(radial_intensity[all_radial_peak_indices]+1).ravel(),
                        color="green", marker="+", s=500)
            plt.scatter(r_space_pixel[prominent_peak_index],
                    20*np.log10(radial_intensity[prominent_peak_index]+1),
                    color="red", marker=".", s=500)
            plt.scatter(r_space_pixel[radial_peak_indices],
                    20*np.log10(radial_intensity[radial_peak_indices]+1),
                    color="orange", marker="|", s=500)
            plt.show()

        # Set up linear regression inputs
        # Set y values based on derviations
        # Convert wavelength to angstroms, same units as q_peaks
        wavelength_angstroms = wavelength*1e10
        theta_n = np.arcsin(q_peaks_avg_subset*wavelength_angstroms/(4*np.pi))
        Y = np.tan(2*theta_n).reshape(-1,1)
        # Set x values as the measured r peaks
        X = r_space_pixel[radial_peak_indices].reshape(-1,1)

        # Now perform linear regression, line goes through the origin
        # so intercept = 0
        linreg = LinearRegression(fit_intercept=False)
        linreg.fit(X, Y)
        score = linreg.score(X, Y)

        # Get the slope
        coef = linreg.coef_
        slope = coef[0][0]
        # The slope is the inverse of the sample-to-detector distance
        distance_pixel = 1/slope
        distance = distance_pixel * PIXEL_WIDTH

        return distance, linreg, score


    def sample_set_detector_distance(self):
        """
        Calculate the sample-to-detector distance for an entire sample set
        """
        pass


if __name__ == "__main__":
    """
    Commandline interface

    Directory specifications:
    - Specify the image full path

    Parameters specifications:
    - Provide the full path to the params file, or
    - provide a JSON-encoded string of parameters.

    """
    print("Start calibration...")

    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--image_fullpath", default=None,
            help="The full path to the raw image data")
    parser.add_argument(
            "--material", default="silver_behenate",
            help="The calibration material")
    parser.add_argument(
            "--pixel_width", default=PIXEL_WIDTH,
            help="The physical pixel size in meters.")
    parser.add_argument(
            "--wavelength", default=WAVELENGTH,
            help="The wavelength meters.")
    parser.add_argument(
            "--visualize", action="store_true",
            help="Plot calibration results to screen")

    args = parser.parse_args()

    # Set variables based on input arguments
    # Set path info
    image_fullpath = args.image_fullpath
    if image_fullpath is None:
        raise ValueError("Must provide full path to image calibration file.")

    # Set material info
    material = args.material
    # Set wavelength
    wavelength = args.wavelength
    # Set pixel width
    pixel_width = args.pixel_width
    # Set visualization option
    visualize = args.visualize

    # Instantiate Calibration class
    calibrator = Calibration(calibration_material=material,
            wavelength=wavelength, pixel_width=pixel_width)

    # Load image
    image = np.loadtxt(image_fullpath, dtype=np.uint32)

    # Run calibration procedure
    detector_distance, linreg, score  = calibrator.single_sample_detector_distance(
            image, beam_rmax=10, r_max=90, distance_approx=10e-3, visualize=visualize)

    detector_distance_mm = detector_distance * 1e3

    print("Detector distance:", detector_distance_mm, "[mm]")
    print("R^2:", score)

    # Save
    print("Done calibrating")
