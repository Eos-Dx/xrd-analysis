"""
Code for fitting data to curves
"""
import os
import argparse
import numpy as np
from collections import OrderedDict

from scipy.optimize import curve_fit

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from eosdxanalysis.models.utils import cart2pol
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

    def __init__(self):
        """
        Initialize `GaussianDecomposition` class.
        """
        return super().__init__()

    # P parameters:
    #  - peak_radius
    #  - width
    #  - amplitude
    #  - cos_power

    # Intial parameters guess for a typical keratin diffraction pattern
    p0_dict = OrderedDict({
            # 9A equatorial peaks minimum parameters
            "peak_radius_9A":       feature_pixel_location(9e-10), # Peak pixel radius
            "width_9A":             8, # Width
            "amplitude_9A":         1131.2, # Amplitude
            "cosine_power_9A":      16, # cosine power
            # 5A meridional peaks minimum parameters
            "peak_radius_5A":       feature_pixel_location(5e-10), # Peak pixel radius
            "width_5A":             9, # Width
            "amplitude_5A":         396, # Amplitude
            "cosine_power_5A":      0.1, # cosine power
            # 5-4A isotropic region minimum parameters
            "peak_radius_5_4A":     feature_pixel_location(4.5e-10), # Peak pixel radius
            "width_5_4A":           18, # Width
            "amplitude_5_4A":       370, # Amplitude
            # Background noise minimum parameters
            "peak_radius_bg":       40, # Peak pixel radius
            "width_bg":             84, # Width
            "amplitude_bg":         886, # Amplitude
        })

    # Lower bounds
    p_lower_bounds_dict = OrderedDict({
            # 9A equatorial peaks
            "peak_radius_9A":       25, # Peak pixel radius
            "width_9A":             1, # Width
            "amplitude_9A":         10, # Amplitude
            "cosine_power_9A":      1, # cosine power
            # 5A meridional peaks
            "peak_radius_5A":       50, # Peak pixel radius
            "width_5A":             1, # Width
            "amplitude_5A":         0, # Amplitude
            "cosine_power_5A":      0.1, # cosine power
            # 5-4A isotropic region
            "peak_radius_5_4A":     50, # Peak pixel radius
            "width_5_4A":           2, # Width
            "amplitude_5_4A":       20, # Amplitude
            # Background noise
            "peak_radius_bg":       30, # Peak pixel radius
            "width_bg":             10, # Width
            "amplitude_bg":          0, # Amplitude
        })

    # Upper bounds
    p_upper_bounds_dict = OrderedDict({
            # 9A equatorial peaks maximum parameters
            "peak_radius_9A":       40, # Peak pixel radius
            "width_9A":             30, # Width
            "amplitude_9A":         2000, # Amplitude
            "cosine_power_9A":      30, # cosine power
            # 5A meridional peaks maximum parameters
            "peak_radius_5A":       70, # Peak pixel radius
            "width_5A":             10, # Width
            "amplitude_5A":         2000, # Amplitude
            "cosine_power_5A":      12, # cosine power
            # 5-4A isotropic region maximum parameters
            "peak_radius_5_4A":     90, # Peak pixel radius
            "width_5_4A":           100, # Width
            "amplitude_5_4A":       2000, # Amplitude
            # Background noise maximum parameters
            "peak_radius_bg":       50, # Peak pixel radius
            "width_bg":             300, # Width
            "amplitude_bg":         1000, # Amplitude
        })

    @classmethod
    def radial_gaussian(self, r, theta, phase, beta,
                peak_radius, width, amplitude, cos_power):
        """
        Isotropic and anisotropic radial Gaussian

        Inputs: (fixed parameters)
        - beta: isotropic = 0, anisotropic = 1
        - phase: either 0 or np.pi/2 to signify equatorial or meridional peak location
        """
        gau = amplitude*np.exp(-((r - peak_radius) / width)**2)
        gau *= np.power(np.cos(theta + phase)**2, beta*cos_power)
        return gau

    @classmethod
    def keratin_function(self, polar_point,
            peak_radius_9A, width_9A, amplitude_9A, cosine_power_9A,
            peak_radius_5A, width_5A, amplitude_5A, cosine_power_5A,
            peak_radius_5_4A, width_5_4A, amplitude_5_4A,
            peak_radius_bg, width_bg, amplitude_bg):
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
        phase_5_4A = 0.0 # Don't care
        phase_bg = 0.0 # Don't care
        # Set betas
        beta_9A = True # Anisotropic
        beta_5A = True # Anisotropic
        beta_5_4A = False # Isotropic
        beta_bg = False # Isotropic

        # Fix isotropic cosine power parameter to zero
        cosine_power_5_4A = 0 # 5-4A cosine power
        cosine_power_bg = 0 # Background noise cosine power

        approx_9A = self.radial_gaussian(r, theta, phase_9A, beta_9A,
                peak_radius_9A, width_9A, amplitude_9A, cosine_power_9A)
        approx_5A = self.radial_gaussian(r, theta, phase_5A, beta_5A,
                peak_radius_5A, width_5A, amplitude_5A, cosine_power_5A)
        approx_5_4A = self.radial_gaussian(r, theta, phase_5_4A, beta_5_4A,
                peak_radius_5_4A, width_5_4A, amplitude_5_4A, cosine_power_5_4A)
        approx_bg = self.radial_gaussian(r, theta, phase_bg, beta_bg,
                peak_radius_bg, width_bg, amplitude_bg, cosine_power_bg)
        approx = approx_9A + approx_5A + approx_5_4A + approx_bg
        return approx.ravel()

    @classmethod
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

    @classmethod
    def fit_error(self, image, fit):
        """
        Returns the square error between a function and
        its approximation.
        """
        return np.sum(np.square(image - fit))

    @classmethod
    def objective(self, p, image, r, theta):
        """
        Generate a keratin diffraction pattern with the given
        parameters list p and return the fit error
        """
        fit = self.keratin_function((r, theta), *p)
        return fit_error(p, image, fit, r, theta)

    @classmethod
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
    popt, pcov, RR, TT = GaussianDecomposition.best_fit(image)
    decomp_image  = GaussianDecomposition.keratin_function((RR, TT), *popt).reshape(*image.shape)

    # Save output
    if output_filepath:
        np.savetxt(output_filepath, decomp_image, fmt="%d")
