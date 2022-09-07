"""
Code for fitting data to curves
"""
import os
import argparse
import numpy as np
from collections import OrderedDict

from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.signal import convolve2d

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from eosdxanalysis.models.utils import cart2pol
from eosdxanalysis.models.utils import radial_intensity_1d
from eosdxanalysis.models.utils import angular_intensity_1d
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.simulations.utils import feature_pixel_location


class PolynomialFit(object):
    """
    Class for polynomial fitting
    """

    @classmethod
    def fit_poly(self, input_points, outputs, degree=2):
        """
        Given inputs, outputs, and polynomial degree,
        perform a linear regression.
        If input_points are multidimensional,
        then polynomial will be multidimensional.
        For example, 2D will have x, y, x*y, x^2, etc.
        """
        # Fit the polynomial
        poly = PolynomialFeatures(degree=degree)
        
        # Transform the input to features
        in_features = poly.fit_transform(input_points)

        model = LinearRegression(fit_intercept=False)
        # Perform linear regression
        model.fit(in_features, outputs)
        return model, poly

class GaussianDecomposition(object):
    """
    Class for decomposing keratin diffraction patterns
    as a sum of Gaussians with angular spread.
    """

    def __init__(self, image=None, p0_dict=None, p_lower_bounds_dict=None, p_upper_bounds_dict=None):
        """
        Initialize `GaussianDecomposition` class.
        """
        if type(image) == np.ndarray:
            # Calculate image center coordinates
            center = image.shape[0]/2-0.5, image.shape[1]/2-0.5
            self.center = center
        # Initialize parameters
        self.parameter_init(image, p0_dict, p_lower_bounds_dict, p_upper_bounds_dict)
        return super().__init__()

    def estimate_parameters(self, image=None, width=4, position_tol=0.2):
        """
        Estimate Gaussian fit parameters based on provided image

        Parameters
        ----------
        image : can be None, pulled from init
        width : averaging width used to produce 1D profiles
        position_tol : factor used to check if detected peak locations are incorrect

        Returns
        -------
        (p0_dict, p_lower_bounds_dict, p_upper_bounds_dict): tuple

        Also stores these parameters in class parameters.

        Notes:
        - Use horizontal and vertical radial intensity profiles, and their differences,
          to calculate properties of isotropic and anisotropic Gaussians.
        - The 9A and 5A peaks are hypothesized to be the sum of isotropic and anisotropic Gaussians
        - The 5-4A ring and background intensities are hypothesized to be isotropic Gaussians
        - That makes a total of 6 Gaussians

        """

        # Get 1D radial intensity in positive horizontal direction
        horizontal_intensity_1d = radial_intensity_1d(image, width=width)
        # Get 1D radial intensity in positive vertical direction
        vertical_intensity_1d = radial_intensity_1d(image.T[:,::-1], width=width)
        # Take the difference of the horizontal and vertical 1D intensity profiles
        # to estimate some anisotropic Gaussian properties
        intensity_diff_1d = horizontal_intensity_1d - vertical_intensity_1d

        # Estimate the 9A isotropic and anisotropic Gaussian function properties
        # - Anisotropic
        # - Isotropic

        peaks_aniso_9A, _ = find_peaks(intensity_diff_1d)

        # Ensure that at least one peak was found
        try:
            # Get peak location
            peak_location_radius_9A = peaks_aniso_9A[0]
        except IndexError as err:
            print("No peaks found for 9A parameters estimation.")
            raise err
        # Ensure that peak_9A is close to theoretical value
        peak_location_radius_9A_theory = feature_pixel_location(9e-10)
        if abs(peak_location_radius_9A - peak_location_radius_9A_theory) > position_tol * peak_location_radius_9A_theory:
            # Use the theoretical value
            peak_location_radius_9A = peak_location_radius_9A_theory
            # raise ValueError("First peak is too far from theoretical value of 9A peak location.")

        # Estimate the 9A peak widths (full-width at half maximum)
        width_results_aniso_9A = peak_widths(intensity_diff_1d, peaks_aniso_9A)
        peak_width_aniso_9A = width_results_aniso_9A[0][0]
        peak_std_aniso_9A = peak_width_aniso_9A / (2*np.sqrt(2*np.log(2))) # convert FWHM to standard deviation

        # Estimate the 9A anisotropic peak amplitude
        # TODO: Interpolate using map_coordinates
        peak_amplitude_aniso_9A = intensity_diff_1d[int(peak_location_radius_9A)]

        # Estimate the anisotropic part of the angular intensity
        # TODO: invert and use `find_peaks` with the greatest prominence
        angular_intensity_9A_1d = angular_intensity_1d(image, radius=peak_location_radius_9A, width=width)
        # Estimate the 9A isotropic peak amplitude
        peak_amplitude_iso_9A = angular_intensity_9A_1d.min()
        angular_intensity_aniso_9A_1d = angular_intensity_9A_1d - peak_amplitude_iso_9A 

        # Integrate the normalized anisotropic angular intensity
        alpha_intensity_9A_1d = 1/360 * np.sum(angular_intensity_aniso_9A_1d/angular_intensity_aniso_9A_1d.max())
        cos_power_9A = 1/alpha_intensity_9A_1d 

        # Estimate the width of the 9A isotropic peak
        width_results_iso_9A = peak_widths(vertical_intensity_1d, [peak_location_radius_9A])
        peak_width_iso_9A = width_results_iso_9A[0]
        peak_std_iso_9A = peak_width_iso_9A / (2*np.sqrt(2*np.log(2))) # convert FWHM to standard deviation

        ########################################################################
        # Estimate the 5A isotropic and anisotropic Gaussian function properties
        ########################################################################

        # Locate the 5A peaks and calculate some properties
        intensity_diff_5A = vertical_intensity_1d - horizontal_intensity_1d

        ##########################################################
        # Estimate the 5-4A isotropic Gaussian function properties
        ##########################################################

        ################################################################
        # Estimate the background isotropic Gaussian function properties
        ################################################################

        p0_dict = OrderedDict({
                    # 9A parameters
                    "peak_location_radius_9A":      peak_location_radius_9A, # Peak pixel radius
                    # 9A isometric (ring) parameters
                    "peak_std_iso_9A":              peak_std_iso_9A, # Standard deviation
                    "peak_amplitude_iso_9A":        peak_amplitude_iso_9A, # Amplitude
                    # 9A anisotropic (equatorial peaks) parameters
                    "peak_std_aniso_9A":            peak_std_aniso_9A, # Standard deviation
                    "peak_amplitude_aniso_9A":      peak_amplitude_aniso_9A, # Amplitude
                    "cos_power_9A":                 cos_power_9A, # Cosine power
#                      # 5A parameters
#                      "peak_location_radius_5A":      peak_location_radius_5A, # Peak pixel radius
#                      # 5A isometric (ring) parameters
#                      "peak_std_iso_5A":              peak_std_iso_5A, # Standard deviation
#                      "peak_amplitude_iso_5A":        peak_amplitude_iso_5A, # Amplitude
#                      # 5A anisotropic (equatorial peaks) parameters
#                      "peak_std_aniso_5A":            peak_std_aniso_5A, # Standard deviation
#                      "peak_amplitude_aniso_5A":      peak_amplitude_aniso_5A, # Amplitude
#                      "cos_power_5A":                 cos_power_5A, # Cosine power
#                      # 5-4A isotropic parameters
#                      "peak_location_radius_5_4A":    peak_location_radius_5_4A, # Peak pixel radius
#                      "peak_std_iso_5_4A":            peak_std_iso_5_4A, # Width
#                      "peak_amplitude_iso_5_4A":      peak_amplitude_iso_5_4A, # Amplitude
#                      # Background isotropic parameters
#                      "peak_std_iso_bg":              peak_std_iso_bg, # Width
#                      "peak_amplitude_iso_bg":        peak_amplitude_iso_bg, # Amplitude
                })

        # Lower bounds
        if "p_lower_bounds_dict" not in self.__dict__:
            p_min_factor = 0.7
            p_lower_bounds_dict = OrderedDict()
            for key, value in p0_dict.items():
                # Set lower bounds
                p_lower_bounds_dict[key] = p_min_factor*value

        # Upper bounds
        if "p_upper_bounds_dict" not in self.__dict__:
            p_upper_bounds_dict = OrderedDict()
            p_max_factor = 1.3
            for key, value in p0_dict.items():
                # Set upper bounds
                p_upper_bounds_dict[key] = p_max_factor*value

        self.p0_dict = p0_dict
        self.p_lower_bounds_dict = p_lower_bounds_dict
        self.p_upper_bounds_dict = p_upper_bounds_dict

        return p0_dict, p_lower_bounds_dict, p_upper_bounds_dict

    def parameter_init(self, image=None, p0_dict=None, p_lower_bounds_dict=None, p_upper_bounds_dict=None):
        """
        P parameters:
        - peak_radius
        - width
        - amplitude
        """
        # Estimate parameters based on image if image is provided
        if type(image) == np.ndarray:
            return self.estimate_parameters(image)

        # No image is provided, so use default parameter estimates
        # for a typical keratin diffraction pattern
        if not p0_dict:
            p0_dict = OrderedDict({
                    # 9A equatorial peaks parameters
                    "peak_location_radius_9A":  feature_pixel_location(9e-10), # Peak pixel radius
                    "peak_std_9A":              10, # Width
                    "peak_amplitude_9A":        1000, # Amplitude
                    "cos_power_9A":             8.0, # Cosine power
                    # 5A meridional peaks parameters
                    "peak_location_radius_5A":  feature_pixel_location(5e-10), # Peak pixel radius
                    "peak_std_5A":              10, # Width
                    "peak_amplitude_5A":        1000, # Amplitude
                    "cos_power_5A":             4.0, # Cosine power
                    # 5-4A isotropic region parameters
                    "peak_location_radius_5_4A":feature_pixel_location(4.5e-10), # Peak pixel radius
                    "peak_std_5_4A":            20, # Width
                    "peak_amplitude_5_4A":      800, # Amplitude
                    # Background noise parameters
                    "peak_std_bg":              30, # Width
                    "peak_amplitude_bg":        200, # Amplitude
                })

        # Lower bounds
        if not p_lower_bounds_dict:
            p_min_factor = 0.7
            p_lower_bounds_dict = OrderedDict()
            for key, value in p0_dict.items():
                # Set lower bounds
                p_lower_bounds_dict[key] = p_min_factor*value

        # Upper bounds
        if not p_upper_bounds_dict:
            p_upper_bounds_dict = OrderedDict()
            p_max_factor = 1.3
            for key, value in p0_dict.items():
                # Set upper bounds
                p_upper_bounds_dict[key] = p_max_factor*value

        self.p0_dict = p0_dict
        self.p_lower_bounds_dict = p_lower_bounds_dict
        self.p_upper_bounds_dict = p_upper_bounds_dict

        return p0_dict, p_lower_bounds_dict, p_upper_bounds_dict

    def radial_gaussian(self, r, theta, peak_radius, peak_angle,
                peak_std, peak_amplitude, arc_angle, resolution=(512,512)):
        """
        Isotropic and radial Gaussian

        Currently uses convolution with a high-resolution arc specified by arc-angle.
        Future will possibly use analytic expression for convolution of a Gaussian with an arc.
        Assumes quadrant-fold symmetry.
        Assumes standard deviation is the same for x and y directions.

        .. Parameters

        :param r: Polar point radius to evaluate function at (``r`` must be same shape as ``theta``)
        :type r: array_like

        :param theta: Polar point angle to evaluate function at (``r`` must be same shape as ``theta``)
        :type theta: array_like

        :param peak_radius: Location of the Gaussian peak from the center
        :type peak_radius: float

        :param peak_angle: 0 or np.pi/2 to signify equatorial or meridional peak location (respectively)
        :type peak_angle: float

        :param peak_std: Gaussian standard deviation
        :type peak_std: float

        :param peak_amplitude: Gaussian peak amplitude
        :type peak_amplitude: float

        :param arc_angle: arc angle in radians, `0` is anisotropic, `pi` is fully isotropic
        :type arc_angle: float

        .. Returns
        :return: Value of Gaussian function at polar input points ``(r, theta)``
        :rtype: array_like (same shape as ``r`` and ``theta``)

        """
        # Set up Gaussian at peak_radius, peak_angle position
        # Convert from polar to Cartesian coordinates
        peak_x = peak_radius*np.cos(peak_angle)
        peak_y = peak_radius*np.sin(peak_angle)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        # Assume standard deviation is the same in x and y
        peak_std_x = peak_std
        peak_std_y = peak_std
        # Assume x and y are uncoorrelated
        rho = 0 # correlation coefficient
        # Set Gaussian origin to (0,0)
        mu_x = mu_y = 0

        # Generate a Gaussian at the origin
        gau = peak_amplitude*np.exp( 1/(2*np/pi*peak_std_x*peak_std_y*np.sqrt(1-rho**2)) * \
                ( -1/(2*(1-rho**2)) * ( ((x - mu_x)/peak_std_x)**2 - \
                2*rho*(x - mu_x)/peak_std_x*(y - mu_y)/peak_std_y +
                ((y - mu_y)/peak_std_y)**2 )))
        # Generate a high-resolution arc to perform convolution with
        arc = arc(peak_radius, peak_angle, arc_angle, resolution=resolution)
        # Perform convolution to get radial Gaussian,
        # ensuring output is the same as the low-res gaussian first input
        radial_gau = convolve2d(gau, arc, mode="same")

        return radial_gau

    def keratin_function(self, polar_point,
            peak_location_radius_9A, peak_std_9A, peak_amplitude_9A, cos_power_9A,
            peak_location_radius_5A, peak_std_5A, peak_amplitude_5A, cos_power_5A,
            peak_location_radius_5_4A, peak_std_5_4A, peak_amplitude_5_4A,
            peak_std_bg, peak_amplitude_bg):
        """
        Generate entire kertain diffraction pattern at the points
        (r, theta), with 16 parameters as the arguements to 4 calls
        of radial_gaussian.

        .. Parameters

        :param polar_point: (r, theta) where r and theta are a meshgrid
        :type polar_point: tuple
        p0 ... p15: parameters to 4 radial_gaussians. Each 4 parameters are
        peak_location, peak_width, peak_amplitude, and angular spread.

        Returns
        -------
            Returns a contiguous flattened array suitable for use with
            `scipy.optimize.curve_fit`.
        """
        r, theta = polar_point
        # Create four Gaussians, then sum
        # Set phases
        phase_9A = 0.0 # Equatorial peak
        phase_5A = np.pi/2 # Meridional peak
        phase_5_4A = 0.0 # Don't care since beta = 0
        phase_bg = 0.0 # Don't care since beta = 0
        # Set betas
        beta_9A = 1.0 # Anisotropic
        beta_5A = 1.0 # Anisotropic
        beta_5_4A = 0.0 # Isotropic
        beta_bg = 0.0 # Isotropic
        # Fix background noise peak radius to zero
        peak_location_radius_bg = 0
        # Set cosine powers
        cos_power_5_4A = 0.0 # Don't care since beta = 0
        cos_power_bg = 0.0 # Don't care since beta = 0

        # 9A peaks
        pattern_9A = self.radial_gaussian(r, theta, phase_9A,
                peak_location_radius_9A, peak_std_9A, peak_amplitude_9A,
                beta_9A, cos_power_9A)
        # 5A peaks
        pattern_5A = self.radial_gaussian(r, theta, phase_5A,
                peak_location_radius_5A, peak_std_5A, peak_amplitude_5A,
                beta_5A, cos_power_5A)
        # 5-4 A anisotropic ring
        pattern_5_4A = self.radial_gaussian(r, theta, phase_5_4A,
                peak_location_radius_5_4A, peak_std_5_4A, peak_amplitude_5_4A,
                beta_5_4A, cos_power_5_4A)
        # Background noise
        pattern_bg = self.radial_gaussian(r, theta, phase_bg,
                peak_location_radius_bg, peak_std_bg, peak_amplitude_bg,
                beta_bg, cos_power_bg)
        # Additive model
        pattern = pattern_9A + pattern_5A + pattern_5_4A + pattern_bg
        return pattern.ravel()

    def best_fit(self, image):
        """
        Use `scipy.optimize.curve_fit` to decompose a keratin diffraction pattern
        into a sum of Gaussians with angular spread.

        Function to optimize: `self.keratin_function`
        """
        # Get parameters and bounds
        p0 = np.fromiter(self.p0_dict.values(), dtype=np.float64)

        p_bounds = (
                # Lower bounds
                np.fromiter(self.p_lower_bounds_dict.values(), dtype=np.float64),
                # Upper bounds
                np.fromiter(self.p_upper_bounds_dict.values(), dtype=np.float64),
            )

        # Check if image size is square
        if image.shape[0] != image.shape[1]:
            raise ValueError("Image shape must be square.")

        # Generate the meshgrid
        if not getattr(self, "meshgrid", None):
            RR, TT = self.gen_meshgrid(image.shape)
        else:
            RR, TT = self.meshgrid

        # Remove meshgrid components that are in the beam center
        beam_rmax = 25
        mask = create_circular_mask(image.shape[0], image.shape[1], rmax=beam_rmax)

        RR_masked = RR[~mask].astype(np.float64)
        TT_masked = TT[~mask].astype(np.float64)
        image_masked = image[~mask].astype(np.float64)

        xdata = (RR_masked.ravel(), TT_masked.ravel())
        ydata = image_masked.ravel().astype(np.float64)
        popt, pcov = curve_fit(self.keratin_function, xdata, ydata, p0, bounds=p_bounds)

        return popt, pcov, RR, TT

    def fit_error(self, image, fit):
        """
        Returns the square error between a function and
        its approximation.
        """
        return np.sum(np.square(image - fit))

    def objective(self, p, image, r, theta):
        """
        Generate a keratin diffraction pattern with the given
        parameters list p and return the fit error
        """
        fit = self.keratin_function((r, theta), *p)
        return fit_error(p, image, fit, r, theta)

    def gen_meshgrid(self, shape):
        """
        Generate a meshgrid
        """
        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_end
        y_start = x_start
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT, RR = cart2pol(XX, YY)

        self.meshgrid = RR, TT

        return RR, TT


if __name__ == '__main__':
    """
    Run curve_fitting on a file or entire folder.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_filepath", default=None, required=True,
            help="The filepath of the raw file to perform fitting on")
    parser.add_argument(
            "--output_filepath", default=None, required=True,
            help="The filepath of the output file")

    # Collect arguments
    args = parser.parse_args()
    input_filepath = args.input_filepath
    output_filepath = args.output_filepath

    # Load image
    input_filename = os.path.basename(input_filepath)
    image = np.loadtxt(input_filepath, dtype=np.float64)

    # Now get "best-fit" diffraction pattern
    gauss_class = GaussianDecomposition()
    popt, pcov, RR, TT = gauss_class.best_fit(image)
    decomp_image  = gauss_class.keratin_function((RR, TT), *popt).reshape(*image.shape)

    # Save output
    if output_filepath:
        np.savetxt(output_filepath, decomp_image, fmt="%d")
