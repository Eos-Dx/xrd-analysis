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
from eosdxanalysis.preprocessing.center_finding import find_center
from eosdxanalysis.preprocessing.image_processing import unwarp_polar
from eosdxanalysis.preprocessing.image_processing import crop_image
from eosdxanalysis.preprocessing.image_processing import quadrant_fold

PIXEL_WIDTH_M = 55e-6 # Pixel width in meters (it is 55 um)


class Calibration(object):
    """
    Calibration class to perform calibration for a set of calibration images

    q units are per Angstrom
    """

    def __init__(self, calibration_material, wavelen_angstrom=1.5418):
        """
        Initialize Calibration class
        """
        # Store source wavelength
        self.wavelen_angstrom = wavelen_angstrom

        # Store calibration material name
        self.calibration_material = calibration_material

        # Look up calibration material q-peaks reference data
        try:
            self.q_peaks_ref = q_peaks_ref_dict[calibration_material]
        except KeyError as err:
            print("Calibration material {} not found!".format(
                                            calibration_material))
        return super().__init__()

    def single_sample_detector_distance(self, image, r_min=0, r_max=None,
                                        distance_approx=10e-3, oversample=2,
                                        visualize=False):
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
        wavelen_angstrom = self.wavelen_angstrom
        q_peaks_ref = self.q_peaks_ref

        # Centerize the image
        center = find_center(image)
        array_center = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)
        # Find eye rotation using original image
        translation = (array_center[1] - center[1], array_center[0] - center[0])

        # Center the image
        translation_tform = EuclideanTransform(translation=translation)
        centered_image = warp(image, translation_tform.inverse)

        # Warp to polar
        # Set a maximum radius which we are interested in
        if r_max is None:
            raise ValueError("Must specify maximum radius.")
        final_r_pixel = int(r_max)

        # Set a minimum radius which we are interested in
        start_r_pixel = int(r_min) if r_min else 0

        # Oversample output image for better interpolation
        output_shape = (oversample*centered_image.shape[0], oversample*centered_image.shape[1])
        # Convert to 2D polar image
        polar_image = warp_polar(centered_image, radius=final_r_pixel,
                                output_shape=output_shape, preserve_range=True)

        # Crop image based on start_r_pixel (rows is theta, cols is radius)
        polar_image_cropped = polar_image[:, oversample*start_r_pixel:]

        # Convert to 1D intensity vs. radius (pixel) and rescale by shape
        radial_intensity = np.sum(polar_image_cropped, axis=0)/output_shape[0]
        # Set up radius linspace, where r is in pixels
        r_space_pixel = np.linspace(start_r_pixel, final_r_pixel,
                                                len(radial_intensity))

        # Find the radial peaks
        # Find the peak of the first doublet
        # First, get a list of possible doublets with a width of oversample*8
        test_doublet_peak_indices, properties = find_peaks(radial_intensity, width=oversample*8)
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
            radial_peak_indices = all_radial_peak_indices[num_missing-1:final_index+1]
            q_peaks_avg_subset = q_peaks_avg[num_missing-1:final_index+1]

        if visualize:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            plt.imshow(20*np.log10(polar_image+1), cmap="gray")


            fig = plt.figure()
            beam_mask = create_circular_mask(
                    centered_image.shape[1],centered_image.shape[0],rmax=r_min)
            beam_masked_img = np.copy(centered_image)
            beam_masked_img[beam_mask] = 0

            plt.imshow(20*np.log10(beam_masked_img.astype(np.float64)+1), cmap="gray")
            plt.scatter(beam_masked_img.shape[1]/2-0.5, beam_masked_img.shape[0]/2-0.5, color="green")

            fig = plt.figure()

            plt.scatter(r_space_pixel, 20*np.log10(radial_intensity+1))
            plt.scatter(r_space_pixel[all_radial_peak_indices],
                        20*np.log10(radial_intensity[all_radial_peak_indices]+1),
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
        theta_n = np.arcsin(q_peaks_avg_subset*wavelen_angstrom/(4*np.pi))
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

        distance_m = distance_pixel * PIXEL_WIDTH_M

        return distance_m, linreg, score


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
    # Set visualization option
    visualize = args.visualize

    # Instantiate Calibration class
    calibrator = Calibration(calibration_material=material)

    # Load image
    image = np.loadtxt(image_fullpath, dtype=np.uint32)

    # Run calibration procedure
    detector_distance_m, linreg, score  = calibrator.single_sample_detector_distance(
            image, r_min=0, r_max=90, distance_approx=10e-3, visualize=visualize)

    detector_distance_mm = detector_distance_m * 1e3

    print("Detector distance:", detector_distance_mm, "[mm]")
    print("R^2:", score)

    # Save
    print("Done calibrating")
