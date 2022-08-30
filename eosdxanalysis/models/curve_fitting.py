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

    def estimate_parameters(self, image=None, width=4, position_tol=5):
        """
        Estimate Gaussian fit parameters based on provided image

        - Use horizontal and vertical radial intensity profiles, and their differences,
          to calculate properties of isotropic and isotropic Gaussians.
        """
        # Get 1D radial intensity in positive horizontal direction
        horizontal_intensity_1d = radial_intensity_1d(image, width=width)

        # Get 1D radial intensity in positive vertical direction
        vertical_intensity_1d = radial_intensity_1d(image.T[:,::-1], width=width)

        # Estimate the 9A peak location and calculate peak properties
        intensity_diff_1d_9A = horizontal_intensity_1d - vertical_intensity_1d
        peaks_aniso_9A, _ = find_peaks(intensity_diff_1d_9A)

        # Ensure that at least one peak was found
        try:
            # Get peak location
            peak_location_radius_9A = peaks_aniso_9A[0]
        except IndexError as err:
            print("No peaks found for 9A parameters estimation.")
            raise err
        # Ensure that peak_9A is close to theoretical value
        peak_location_raidius_9A_theory = feature_pixel_location(9e-10)
        if abs(peak_location_radius_9A - peak_location_raidius_9A_theory) > position_tol:
            raise ValueError("9A peak is too far from theoretical value.")

        # Estimate the 9A peak widths (full-width at half maximum)
        width_results_aniso_9A = peak_widths(intensity_diff_1d_9A, peaks_aniso_9A)
        peak_width_aniso_9A = width_results_aniso_9A[0][0]
        peak_std_9A = peak_width_aniso_9A / (2*np.sqrt(2*np.log(2))) # convert FWHM to standard deviation

        # Estimate the 9A anisotropic peak amplitude
        peak_amplitude_aniso_9A = intensity_diff_1d_9A[peak_location_radius_9A]

        # Estimate the anisotropic part of the angular intensity
        # TODO: invert and use `find_peaks` with the greatest prominence
        angular_intensity_9A_1d = angular_intensity_1d(image, radius=peak_location_radius_9A, width=width)
        angular_intensity_aniso_9A_1d = angular_intensity_9A_1d - angular_intensity_9A_1d.min()

        # Integrate the normalized anisotropic angular intensity
        alpha_intensity_9A_1d = 1/360 * np.sum(angular_intensity_aniso_9A_1d/angular_intensity_aniso_9A_1d.max())
        cos_power_9A = 1/alpha_intensity_9A_1d 

        # Locate the 5A peaks and calculate some properties
        intensity_diff_5A = vertical_intensity_1d - horizontal_intensity_1d


        # 9A maxima
        # 5A maxima
        # 5-4A ring
        # Background noise

#         self.p0_dict = p0_dict
#         self.p_lower_bounds_dict = p_lower_bounds_dict
#         self.p_upper_bounds_dict = p_upper_bounds_dict
#
    def parameter_init(self, image=None, p0_dict=None, p_lower_bounds_dict=None, p_upper_bounds_dict=None):
        """
        P parameters:
        - peak_radius
        - width
        - amplitude
        """
        # Estimate parameters based on image if image is provided
        if type(image) == np.ndarray:
            self.estimate_parameters(image)
            return

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

    def radial_gaussian(self, r, theta, phase,
                peak_location_radius, peak_std, peak_amplitude, beta, cos_power):
        """
        Isotropic and radial Gaussian

        Inputs: (fixed parameters)
        - phase: 0 or np.pi/2 to signify equatorial or meridional peak location
        - alpha: isotropy parameter, 1 is isotropic, 0 is anisotropic
        - beta: anisotropy parameter, 1 is anisotropic, 0 is isotropic
        - peak_location_radius: location of the Gaussian peak from the center
        - peak_std: Gaussian standard deviation
        """
        gau_iso = peak_amplitude*np.exp(-((r - peak_location_radius) / peak_std)**2)
        gau = gau_iso * np.power(np.square(np.cos(theta+phase)), beta*cos_power)
        return gau

    def keratin_function(self, polar_point,
            peak_location_radius_9A, peak_std_9A, peak_amplitude_9A, cos_power_9A,
            peak_location_radius_5A, peak_std_5A, peak_amplitude_5A, cos_power_5A,
            peak_location_radius_5_4A, peak_std_5_4A, peak_amplitude_5_4A,
            peak_std_bg, peak_amplitude_bg):
        """
        Generate entire kertain diffraction pattern at the points
        (r, theta), with 16 parameters as the arguements to 4 calls
        of radial_gaussian.

        Inputs:
        - polar_point: tuple (r, theta) where r and theta are a meshgrid
        - p0 ... p15: parameters to 4 radial_gaussians. Each 4 parameters are
          peak_location, peak_width, peak_amplitude, and angular spread.

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
