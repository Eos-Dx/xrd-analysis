"""
Code to calibrate X-ray diffraction setup using calibration samples data
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

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

PIXEL_SIZE = 55e-6 # Pixel width in meters (it is 55 um)
WAVELENGTH = 1.5418E-10 # Wavelength in meters (1.5418 Angstroms)
WAVELEN_ANGSTROMS = WAVELENGTH * 1e10 # Wavelength in Angstroms
BEAM_RMAX = 10 # Pixel radius to block out beam
RMAX = 110 # Pixel radius to ignore beyond this value
DISTANCE_APPROX = 10e-3
DOUBLET_WIDTH = 5
DOUBLET_APPROX_MIN_FACTOR = 0.5
DOUBLET_APPROX_MAX_FACTOR = 2.0
OUTPUT_SHAPE = (360,128)
DEFAULT_RADIUS = 128


class Calibration(object):
    """
    Calibration class to perform calibration for a set of calibration images

    q units are per Angstrom
    """

    def __init__(self, calibration_material, wavelength=WAVELENGTH,
            pixel_size=PIXEL_SIZE):
        """
        Initialize Calibration class
        """
        # Store source wavelength
        self.wavelength = wavelength

        # Store calibration material name
        self.calibration_material = calibration_material

        # Store pixel width
        self.pixel_size = pixel_size

        # Look up calibration material q-peaks reference data
        try:
            self.q_peaks_ref = q_peaks_ref_dict[calibration_material]
        except KeyError as err:
            print("Calibration material {} not found!".format(
                                            calibration_material))
        return super().__init__()

    def sample_detector_distance(
            self, image, center=None, beam_rmax=BEAM_RMAX, rmax=RMAX,
            distance_approx=DISTANCE_APPROX, output_shape=OUTPUT_SHAPE,
            doublet_approx_min_factor=DOUBLET_APPROX_MIN_FACTOR,
            doublet_approx_max_factor=DOUBLET_APPROX_MAX_FACTOR,
            doublet_width=DOUBLET_WIDTH,
            visualize=False, radius=None):
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
        wavelength = self.wavelength
        wavelength_angstroms = wavelength*1e10
        q_peaks_ref = self.q_peaks_ref
        pixel_size = self.pixel_size

        # Calculate 1-d azimuthal integration profile
        if center is None:
            # Find the image center
            center = find_center(image)

        if radius is None:
            radius = int(np.max(image.shape)/2)

        # Mask the beam
        mask = create_circular_mask(
                image.shape[1], image.shape[0], center=center,
                rmax=beam_rmax)
        masked_image = np.copy(image)
        # Set the masked part to the minimum of the beam area
        # to avoid creating another peak
        masked_image[mask] = masked_image[mask].min()

        radial_intensity = azimuthal_integration(
                masked_image, center=center, radius=radius)

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
        end_index = radial_intensity.size - 1 if end_index >= radial_intensity.size else end_index
        radial_intensity_subset = radial_intensity[start_index:end_index]

        # Find the doublet in this subset
        doublet_peak_indices_approx, properties = find_peaks(
                radial_intensity_subset, width=doublet_width,
                height=0.05*radial_intensity.max())

        # Check how many prominent peaks were found
        prominences = properties.get("prominences")
        if prominences.size >= 1:
            # Get the peak index of the doublet in the main array
            doublet_peak_index = doublet_peak_indices_approx[0] + start_index
        else:
            raise ValueError("Doublet peak not found.")

        # Now use location of doublet to calculate sample-to-detector distance
        doublet_distance = doublet_peak_index * pixel_size

        L = doublet_distance / np.tan(2*theta_n)

        if visualize:
            title = "Beam masked image"
            fig = plt.figure(title)
            plt.title(title)

            plt.imshow(20*np.log10(masked_image.astype(np.float64)+1), cmap="gray")
            plt.scatter(center[1], center[0], color="green")

            title = "Azimuthal integrated 1-d profile"
            fig = plt.figure(title)
            plt.title(title)

            # Convert to q-values
            r_range = np.arange(radial_intensity.size) * PIXEL_SIZE
            theta_range = 0.5*np.arctan2(L, r_range)
            q_range = 4 * np.pi * np.sin(theta_range) / WAVELEN_ANGSTROMS

            # Plot azimuthal integration 1-D profile
            plt.plot(q_range, 20*np.log10(radial_intensity+1),
                    label="1-D profile")

            # Plot a marker for the most prominent peak
            plt.scatter(q_range[doublet_peak_index],
                    20*np.log10(radial_intensity[doublet_peak_index]+1),
                    color="red", marker=".", s=250,
                    label="Doublet peak")

            plt.xlabel(r"q $\mathrm{\AA^{-1}}$")
            plt.ylabel(r"Mean intensity [photon count, dB+1]")

            plt.xlim([q_range.max(), q_range.min()])

            plt.legend()
            plt.show()

        return L

    def single_sample_detector_distance_old(
            self, image, beam_rmax=10, rmax=None, distance_approx=10e-3,
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

        # Mask the beam
        mask = create_circular_mask(
                image.shape[1], image.shape[0], center=center,
                rmax=beam_rmax)
        masked_image = np.copy(image)
        # Set the masked part to the minimum of the beam area
        # to avoid creating another peak
        masked_image[mask] = masked_image[mask].min()

        radial_intensity = azimuthal_integration(
                masked_image, center=center, output_shape=output_shape)

#        # Set a maximum radius which we are interested in
#        if rmax is None:
#            raise ValueError("Must specify maximum radius.")
#        final_r_pixel = int(rmax)
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
        final_index = int(np.where(all_radial_peak_indices == prominent_peak_index)[0])
        # Count how many we are missing before the first doublet
        num_missing = len(q_peaks_avg[:-1]) - len(all_radial_peak_indices[:final_index])

        if num_missing < 0:
            raise ValueError("We found more peaks than in the reference!")
        elif num_missing == 0:
            radial_peak_indices = all_radial_peak_indices[:final_index+1]
            q_peaks_avg_subset = q_peaks_avg
        elif num_missing > 0:
            # Take subset
            radial_peak_indices = all_radial_peak_indices[num_missing-1:final_index+1]
            q_peaks_avg_subset = q_peaks_avg[num_missing:final_index+2]

        if visualize:
            title = "Beam masked image"
            fig = plt.figure(title)
            plt.title(title)

            plt.imshow(20*np.log10(masked_image.astype(np.float64)+1), cmap="gray")
            plt.scatter(center[1], center[0], color="green")

            title = "Azimuthal integrated 1-d profile"
            fig = plt.figure(title)
            plt.title(title)


            # Plot azimuthal integration 1-D profile
            plt.plot(20*np.log10(radial_intensity+1),
                    label="1-D profile")

            # Plot markers for found peaks
            plt.scatter(r_space_pixel[all_radial_peak_indices].ravel(),
                        20*np.log10(radial_intensity[all_radial_peak_indices]+1).ravel(),
                        color="green", marker="+", s=250,
                        label="found peaks")

            # Plot a marker for the most prominent peak
            plt.scatter(r_space_pixel[prominent_peak_index],
                    20*np.log10(radial_intensity[prominent_peak_index]+1),
                    color="red", marker=".", s=250,
                    label="prominent peak")

            # Plot markers for peaks used for fitting
            plt.scatter(r_space_pixel[radial_peak_indices],
                    20*np.log10(radial_intensity[radial_peak_indices]+1),
                    color="orange", marker="x", s=250,
                    label="peaks for fitting")

            plt.xlabel("Radius [pixels]")
            plt.ylabel("Intensity [dB+1]")

            plt.legend()
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
        distance = distance_pixel * PIXEL_SIZE

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
            "--pixel_size", default=PIXEL_SIZE,
            help="The physical pixel size in meters.")
    parser.add_argument(
            "--wavelength", default=WAVELENGTH,
            help="The wavelength meters.")
    parser.add_argument(
            "--center", type=str, default=None,
            help="The center of the diffraction pattern.")
    parser.add_argument(
            "--beam_rmax", type=int, default=BEAM_RMAX,
            help="The radius to block out the beam.")
    parser.add_argument(
            "--rmax", type=int, default=RMAX,
            help="The radius to block out the beam.")
    parser.add_argument(
            "--doublet_width", type=int, default=DOUBLET_WIDTH,
            help="The doublet width to look for.")
    parser.add_argument(
            "--distance_approx", type=float, default=DISTANCE_APPROX,
            help="The approximate sample-to-detector distance.")
    parser.add_argument(
            "--radius", type=int, default=DEFAULT_RADIUS,
            help="The maximum radius to analyze.")
    parser.add_argument(
            "--visualize", action="store_true",
            help="Plot calibration results to screen")

    args = parser.parse_args()

    # Set variables based on input arguments
    # Set path info
    image_fullpath = args.image_fullpath
    if image_fullpath is None:
        raise ValueError("Must provide full path to image calibration file.")

    material = args.material
    wavelength = args.wavelength
    pixel_size = args.pixel_size
    beam_rmax = args.beam_rmax
    center = ",".split(args.center) if args.center else None
    rmax = args.rmax
    distance_approx = args.distance_approx
    doublet_width = args.doublet_width
    visualize = args.visualize
    radius = args.radius

    # Instantiate Calibration class
    calibrator = Calibration(calibration_material=material,
            wavelength=wavelength, pixel_size=pixel_size)

    # Load image
    image = np.loadtxt(image_fullpath, dtype=np.uint32)

    # Run calibration procedure
    try:
        detector_distance = calibrator.sample_detector_distance(
                image,
                beam_rmax=beam_rmax,
                rmax=rmax,
                distance_approx=distance_approx,
                center=center,
                doublet_width=doublet_width,
                visualize=visualize,
                radius=radius)

        detector_distance_mm = detector_distance * 1e3

        print("Detector distance:", detector_distance_mm, "[mm]")

        # Save
        print("Done calibrating")
    except Exception as err:
        print("Calibration failed.")
