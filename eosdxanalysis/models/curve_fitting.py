"""
Code for fitting data to curves
"""
import os
import argparse
import numpy as np

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
    p0 = [
            # 9A equatorial peaks
            31.6, # Peak pixel radius
            8, # Width
            1131.2, # Amplitude
            16, # cosine power
            # 5A meridional peaks
            58.9, # Peak pixel radius
            5, # Width
            60, # Amplitude
            8, # cosine power
            # 5-4A isotropic region
            66.3, # Peak pixel radius
            18, # Width
            25, # Amplitude
            0, # cosine power
            # Background noise
            0, # Peak pixel radius
            70, # Width
            12, # Amplitude
            0, # cosine power
            ]

    # TODO: These bounds should all be a function of exposure time,
    # sample-to-detector distance, and molecular spacings
    # Order is: 9A, 5A, 5-4A, bg
    p_bounds = (
            # Minimum bounds
            np.array([
                #
                # 9A minimum bounds
                #
                25, # 9A peak_radius minimum
                1, # 9A width minimum
                10, # 9A amplitude minimum
                2, # 9A cos^2n power minimum
                #
                # 5A minimum bounds
                #
                50, # 5A peak_radius minimum
                1, # 5A width minimum
                50, # 5A amplitude minimum
                2, # 5A cos^2n power minimum
                #
                # 5-4A minimum bounds
                #
                50, # 5-4A peak_radius minimum
                2, # 5-4A width minimum
                20, # 5-4A amplitude minimum
                -1, # 5-4A cos^2n power minimum
                #
                # bg minimum bounds
                #
                -1, # bg peak_radius minimum
                10, # bg width minimum
                10, # bg amplitude minimum
                -1, # bg cos^2n power minimum
                ],
                dtype=np.float64,
            ),
            # Maximum bounds
            np.array([
                #
                # 9A maximum bounds
                #
                40, # 9A peak_radius maximum
                30, # 9A width maximum
                2000, # 9A amplitude maximum
                30, # 9A cos^2n power maximum
                #
                # 5A maximum bounds
                #
                70, # 5A peak_radius maximum
                10, # 5A width maximum
                2000, # 5A amplitude maximum
                12, # 5A cos^2n power maximum
                #
                # 5-4A maximum bounds
                #
                90, # 5-4A peak_radius maximum
                100, # 5-4A width maximum
                2000, # 5-4A amplitude maximum
                1, # 5-4A cos^2n power maximum
                #
                # bg maximum bounds
                #
                1, # bg peak_radius maximum
                300, # bg width maximum
                500, # bg amplitude maximum
                1, # bg cos^2n power maximum
                ],
                dtype=np.float64,
            ),
        )

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
    def keratin_function(self, polar_point, p0, p1, p2, p3, p4, p5, p6, p7,
            p8, p9, p10, p11, p12, p13, p14, p15):
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

        approx_9A = self.radial_gaussian(r, theta, phase_9A, beta_9A, p0, p1, p2, p3)
        approx_5A = self.radial_gaussian(r, theta, phase_5A, beta_5A, p4, p5, p6, p7)
        approx_5_4A = self.radial_gaussian(r, theta, phase_5_4A, beta_5_4A, p8, p9, p10, p11)
        approx_bg = self.radial_gaussian(r, theta, phase_bg, beta_bg, p12, p13, p14, p15)
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
        p0 = self.p0
        p_bounds = self.p_bounds

        # Check if image size is square
        if image.shape[0] != image.shape[1]:
            raise ValueError("Image shape must be square.")

        # Generate the meshgrid
        if not getattr(self, "meshgrid", None):
            RR, TT = self.gen_meshgrid()

        # Remove meshgrid components that are in the beam center
        beam_rmax = 25
        mask = create_circular_mask(image.shape[0], image.shape[1], rmax=beam_rmax)

        RR_masked = RR[~mask].astype(np.float64)
        TT_masked = TT[~mask].astype(np.float64)
        image_masked = image[~mask].astype(np.float64)

        xdata = (RR_masked.ravel(), TT_masked.ravel())
        ydata = image_masked.ravel()
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
    def gen_meshgrid(self):
        """
        Generate a meshgrid
        """
        # Generate a meshgrid the same size as the image
        size = image.shape[0]
        x_end = size/2 - 0.5
        x_start = -x_end
        y_end = x_end
        y_start = x_start
        YY, XX = np.mgrid[y_start:y_end:size*1j, x_start:x_end:size*1j]
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
    image = np.loadtxt(input_filepath, dtype=np.uint32)

    # Now get "best-fit" diffraction pattern
    popt, pcov, RR, TT = GaussianDecomposition.best_fit(image)
    decomp_image  = GaussianDecomposition.keratin_function((RR, TT), *popt).reshape(*image.shape)

    # Save output
    if output_filepath:
        np.savetxt(output_filepath, decomp_image, fmt="%d")
