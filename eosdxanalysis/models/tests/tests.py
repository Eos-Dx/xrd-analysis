"""
Tests for models module
"""
import os
import shutil
import glob
import unittest
import numpy as np
import numpy.ma as ma
import subprocess

from collections import OrderedDict

from scipy.special import jn_zeros
from scipy.special import jv
from scipy.io import loadmat
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from scipy.signal import peak_widths

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from eosdxanalysis.models.curve_fitting import PolynomialFit
from eosdxanalysis.models.curve_fitting import GaussianDecomposition
from eosdxanalysis.models.curve_fitting import estimate_background_noise
from eosdxanalysis.models.curve_fitting import gaussian_iso
from eosdxanalysis.models.curve_fitting import keratin_function
from eosdxanalysis.models.utils import gen_jn_zerosmatrix
from eosdxanalysis.models.utils import l1_metric
from eosdxanalysis.models.utils import pol2cart
from eosdxanalysis.models.utils import cart2pol
from eosdxanalysis.models.utils import radial_intensity_1d
from eosdxanalysis.models.utils import angular_intensity_1d
from eosdxanalysis.models.utils import draw_antialiased_arc
from eosdxanalysis.models.utils import draw_antialiased_circle
from eosdxanalysis.models.utils import gen_meshgrid
from eosdxanalysis.models.feature_engineering import EngineeredFeatures
from eosdxanalysis.models.polar_sampling import sampling_grid
from eosdxanalysis.models.polar_sampling import freq_sampling_grid
from eosdxanalysis.models.polar_sampling import rmatrix_SpaceLimited
from eosdxanalysis.models.polar_sampling import thetamatrix_SpaceLimited
from eosdxanalysis.models.polar_sampling import rhomatrix_SpaceLimited
from eosdxanalysis.models.polar_sampling import psimatrix_SpaceLimited
from eosdxanalysis.models.fourier_analysis import YmatrixAssembly
from eosdxanalysis.models.fourier_analysis import pfft2_SpaceLimited
from eosdxanalysis.models.fourier_analysis import ipfft2_SpaceLimited
from eosdxanalysis.models.stats import plot_feature_histograms
from eosdxanalysis.models.polar_gaussian import polar_gaussian

from eosdxanalysis.preprocessing.utils import create_circular_mask

from eosdxanalysis.simulations.utils import feature_pixel_location

TEST_PATH = os.path.dirname(__file__)
MODULE_PATH = os.path.join(TEST_PATH, "..")
MODULE_DATA_PATH = os.path.join(MODULE_PATH, "data")
JN_ZEROSMATRIX_TEST_DIR = "test_jn_zerosmatrix"
JN_ZEROSMATRIX_FILENAME = "jn_zeros_501_501.npy"

# Machine parameters
DISTANCE = 10e-3 # meters
WAVELENGTH = 1.5418e-10 # meters
PIXEL_WIDTH = 55e-6 # meters


class TestPolynomialFit(unittest.TestCase):
    """
    Test create_circular_mask
    """

    def test_polynomial_fit_2d_deterministic(self):
        """
        Test polynomial fitting for a 2D deterministic case
        """
        # mxn system
        m = 10
        n = 10
        X_, Y_ = np.mgrid[:m, :n]

        X = X_.flatten()
        Y = Y_.flatten()

        # Best-fit function should be close to:
        # 100*1 -0.8*X**2 + -0.9*Y**2 + 0.08*X*Y^2 - 0.03*X^3
        Z_ = 100*1 -0.8*X_**2 -0.9 *Y_**2 + \
            0.08*X_*Y_**2 - 0.03*X_**3
        Z = Z_.flatten()

        input_pts = np.stack([X,Y]).T

        # Get the linear regression model and polynomial fit
        model, poly = PolynomialFit.fit_poly(input_pts, Z, degree=3)

        # Check that R-squared is 1
        Rsquared = model.score(poly.transform(input_pts), Z)
        self.assertTrue(np.isclose(Rsquared, 1.0))

    def test_polynomial_fit_1d_deterministic_masked_inputs(self):
        """
        Test polynomial fitting for a 1D deterministic case
        with masked inputs and/or masked outputs
        """
        X = np.arange(100)
        Y_ = 10+8*X-1*X**2-0.1*X**3

        # Mask the output only in two center values
        Y = ma.masked_inside(Y_, 4,5)
        mask = Y.mask

        # Reshape input to 1D vector
        input_pts1 = X.reshape(-1, 1)

        # Get the linear regression model and polynomial fit
        model, poly = PolynomialFit.fit_poly(input_pts1, Y, degree=3)

        # Check that R-squared is 1
        Rsquared1 = model.score(poly.transform(input_pts1), Y)
        self.assertTrue(np.isclose(Rsquared1, 1.0))

        # Make predictions for existing input points
        Y_predicted = model.predict(poly.transform(input_pts1))

        # Check that Y_predicted is not masked
        self.assertFalse(np.array_equal(Y_predicted, Y_predicted[~mask]))

        # Now compute with both inputs and outputs masked
        X_masked = ma.array(X, mask=mask)
        input_pts2 = X.reshape(-1, 1)

        # Get the linear regression model and polynomial fit
        model2, poly2 = PolynomialFit.fit_poly(input_pts2, Y, degree=3)

        # Check that the ceoffficients are the same
        self.assertTrue(np.array_equal(model.coef_, model2.coef_))

        # Check that R-suqared is the same
        Rsquared2 = model.score(poly.transform(input_pts1), Y)
        self.assertTrue(np.isclose(Rsquared1,Rsquared2))


class TestGaussianDecomposition(unittest.TestCase):
    """
    Test `GaussianDecomposition` class
    """

    def setUp(self):
        """
        Set up test class
        """
        # Set up data path
        TEST_DATA_PATH = os.path.join(TEST_PATH, "data", "GaussianDecomposition")
        self.TEST_DATA_PATH = TEST_DATA_PATH

        # Set parameters for a synthetic keratin diffraction pattern
        p_synth_dict = OrderedDict({
                # 9A equatorial peaks parameters
                "peak_location_radius_9A":  feature_pixel_location(9.8e-10), # Peak pixel radius
                "peak_std_9A":              7, # Width
                "peak_amplitude_9A":        401, # Amplitude
                "arc_angle_9A":             1e-2, # Arc angle
                # 5A meridional peaks parameters
                "peak_location_radius_5A":  feature_pixel_location(5.1e-10), # Peak pixel radius
                "peak_std_5A":              1, # Width
                "peak_amplitude_5A":        13, # Amplitude
                "arc_angle_5A":             np.pi/4, # Arc angle
                # 5-4A isotropic region parameters
                "peak_location_radius_5_4A":feature_pixel_location(4.5e-10), # Peak pixel radius
                "peak_std_5_4A":            17, # Width
                "peak_amplitude_5_4A":      113, # Amplitude
                # Background noise
                "constant_bg":              223, # Amplitude
                # Rotation
                "rotation_angle":           45, # Angle degrees
            })

        # Set mesh size
        size = 256
        RR, TT = gen_meshgrid((size,size))

        # Generate synthetic image
        synth_image = keratin_function((RR, TT), *p_synth_dict.values()).reshape(RR.shape)

        self.p_synth_dict = p_synth_dict
        self.size = size
        self.RR, self.TT = RR, TT
        self.synth_image = synth_image

    def test_radial_gaussian_trivial_centered(self):
        """
        Test radial gaussian function for a standard normal Gaussian
        centered at the origin
        """
        SPACING_9A = 9.8e-10 # 9.8A molecular spacing theory location in meters
        # Set space parameters
        size = 256
        shape = size, size
        center = np.array(shape)/2-0.5

        # Generate polar meshgrid
        gau_class = GaussianDecomposition()
        RR, TT = gau_class.gen_meshgrid(shape)

        # Set radial gaussian parameters
        peak_angle = 0.0 # equatorial
        peak_radius = 0.0 # pixel distance from center
        peak_std = 10.0
        peak_amplitude = 100.0
        arc_angle = 0 # Fully anisotropic

        gau = gau_class.radial_gaussian(RR, TT,
                peak_radius, peak_angle, peak_std,
                peak_amplitude, arc_angle)

        self.fail("Finish test")

    def test_radial_gaussian_equatorial_fully_anisotropic(self):
        """
        Test radial gaussian function for a simple equatorial anisotropic Gaussian pattern
        with quadrant-fold symmetry
        """
        SPACING_9A = 9.8e-10 # 9.8A molecular spacing theory location in meters
        # Set space parameters
        size = 256
        shape = size, size

        # Generate polar meshgrid
        gau_class = GaussianDecomposition()
        RR, TT = gau_class.gen_meshgrid(shape)

        # Set radial gaussian parameters
        peak_angle = 0.0 # equatorial
        peak_radius = pixel_feature_location(SPACING_9A)
        peak_std = 10
        peak_amplitude = 100
        arc_theta = 0 # Fully anisotropic

        self.fail("Finish test")

    def test_radial_gaussian_fully_isotropic(self):
        """
        Test radial gaussian function for a fully isotropic Gaussian
        not centered at the origin
        """
        SPACING_9A = 9.8e-10 # 9.8A molecular spacing theory location in meters
        # Set space parameters
        size = 256
        shape = size, size

        # Generate polar meshgrid
        gau_class = GaussianDecomposition()
        RR, TT = gau_class.gen_meshgrid(shape)

        # Set radial gaussian parameters
        peak_angle = 0.0 # equatorial
        peak_radius = pixel_feature_location(SPACING_9A)
        peak_std = 10
        peak_amplitude = 100
        arc_theta = 0 # Fully anisotropic

        self.fail("Finish test")

    def test_radial_gaussian_peak_amplitude(self):
        """
        Ensure that the peak_amplitude value corresponds to gau(0)
        """
        # Set radial gaussian parameters
        peak_angle = 0.0 # equatorial
        peak_radius = 0.0
        peak_std = 10
        peak_amplitude = 13
        arc_angle = 0 # Fully anisotropic

        gau_class = GaussianDecomposition()

        r = theta = 0.0
        gau = gau_class.radial_gaussian(r, theta,
                peak_radius, peak_angle, peak_std,
                peak_amplitude, arc_angle)

        self.assertTrue(np.isclose(gau, peak_amplitude))

    def test_radial_gaussian_peak_std(self):
        """
        Ensure that the peak_std value corresponds to the
        standard deviation
        """
        # Set space parameters
        size = 256
        shape = size, size

        # Set radial gaussian parameters
        peak_angle = 0.0 # equatorial
        peak_radius = 0.0
        peak_std = 20
        peak_amplitude = 17
        arc_angle = 0 # Fully anisotropic

        gau_class = GaussianDecomposition()
        RR, TT = gau_class.gen_meshgrid(shape)

        XX = RR*np.cos(TT)

        gau = gau_class.radial_gaussian(RR, TT,
                peak_radius, peak_angle, peak_std,
                peak_amplitude, arc_angle)

        self.fail("Finish test")
        self.assertTrue(np.isclose(peak_std, test_std))

    def test_cli(self):
        """
        Simple test to check if there are no errors when running main
        """
        # Set up the command
        command = ["python", "eosdxanalysis/models/curve_fitting.py",
                    ]
        # Run the command
        subprocess.run(command)

    def test_cli_known_sample(self):
        """
        Test `GaussianDecomposition` on a known sample
        """
        # Set paths
        input_path = os.path.join(self.TEST_DATA_PATH, "input")
        output_path = os.path.join(self.TEST_DATA_PATH, "output")

        # Set up the command
        command = ["python", "eosdxanalysis/models/curve_fitting.py",
                    "--input_path", input_path,
                    "--output_path", output_path,
                    ]
        # Run the command
        subprocess.run(command)

        # Check all output files
        if False:
            # Check that the output is the same as the test output file
            known_output_filename = "GaussianDecomp_CRQF_A00005.txt"
            known_output_filepath = os.path.join(self.TEST_DATA_PATH, "output", known_output_filename)
            known_output = np.loadtxt(known_output_filepath, dtype=np.uint32)
            test_output = np.loadtxt(output_filepath, dtype=np.uint32)

            self.assertTrue(np.isclose(known_output, test_output).all())

        self.fail("Finish writing test.")

    def test_known_sample_bounds(self):
        """
        Test `GaussianDecomposition` on quality measurement of a normal
        specimen to ensure optimal parameters are not near bounds.
        """
        # Set input filepath
        input_filename = "CRQF_A00005.txt"
        input_filepath = os.path.join(self.TEST_DATA_PATH, "input", input_filename)
        # Set output filepath
        output_filename = "test_GaussianDecomp_CRQF_A00005.txt"
        output_filepath = os.path.join(self.TEST_DATA_PATH, "output", output_filename)

        # Calculate optimum parameters
        image = np.loadtxt(input_filepath, dtype=np.float64)

        gauss_class = GaussianDecomposition(image)
        popt_dict, pcov, RR, TT = gauss_class.best_fit(image)
        popt = np.fromiter(popt_dict.values(), dtype=np.float64)
        decomp_image  = gauss_class.keratin_function((RR, TT), *popt).reshape(image.shape)

        # Check that the optimal parameters are not close to the upper or lower bounds
        p0_values = np.fromiter(gauss_class.p0_dict.values(), dtype=np.float64)
        p_lower_bounds_values = np.fromiter(gauss_class.p_lower_bounds_dict.values(), dtype=np.float64)
        p_upper_bounds_values = np.fromiter(gauss_class.p_upper_bounds_dict.values(), dtype=np.float64)

        self.assertFalse(np.isclose(popt, p_lower_bounds_values).all())
        self.assertFalse(np.isclose(popt, p_upper_bounds_values).all())

        self.fail("Finish writing test.")

    def test_gaussian_decomposition_good_sample_manual_guess(self):
        """
        Evaluate performance of automatic parameter estimation
        """
        # Set manual parameter guess
        p0_guess_dict = OrderedDict({
                # 9A equatorial peaks parameters
                "peak_location_radius_9A":  feature_pixel_location(9.8e-10), # Peak pixel radius
                "peak_std_9A":              8, # Width
                "peak_amplitude_9A":        400, # Amplitude
                "arc_angle_9A":             1e-1, # Arc angle
                # 5A meridional peaks parameters
                "peak_location_radius_5A":  feature_pixel_location(5.1e-10), # Peak pixel radius
                "peak_std_5A":              2, # Width
                "peak_amplitude_5A":        100, # Amplitude
                "arc_angle_5A":             np.pi/4, # Arc angle
                # 5-4A isotropic region parameters
                "peak_location_radius_5_4A":feature_pixel_location(4.15e-10), # Peak pixel radius
                "peak_std_5_4A":            15, # Width
                "peak_amplitude_5_4A":      100, # Amplitude
                # Background noise parameters
                "peak_std_bg":              200, # Width
                "peak_amplitude_bg":        200, # Amplitude
            })

        # Set mesh size
        size = 256
        RR, TT = gen_meshgrid((size,size))

        # Generate synthetic guess image
        synth_image = keratin_function(
                (RR, TT), *p0_guess_dict.values()).reshape(RR.shape)

        # Set input filepath
        # input_filename = "CRQF_AA00832.txt"
        input_filename = "CRQF_AA00730-4.txt"
        input_filepath = os.path.join(self.TEST_DATA_PATH, "input", input_filename)

        # Load input image
        image = np.loadtxt(input_filepath, dtype=np.float64)

        gauss_class = GaussianDecomposition(
                image, p0_dict=p0_guess_dict, params_init_method="estimation")

        # Find Gaussian fit
        popt_dict, pcov = gauss_class.best_fit()
        popt = np.fromiter(popt_dict.values(), dtype=np.float64)
        decomp_image  = keratin_function((RR, TT), *popt).reshape(RR.shape)

        # Mask the gaussian image for comparison purposes
        rmin = 25
        rmax = 90
        mask = create_circular_mask(
                image.shape[0], image.shape[1], rmin=rmin, rmax=rmax)
        decomp_image_masked = decomp_image.copy()
        decomp_image_masked[~mask] = 0

        # Ensure the decomp_image and synth_image are different
        self.assertFalse(np.isclose(decomp_image, synth_image).all())
        self.assertFalse(np.array_equal(decomp_image, synth_image))

        # Get squared error for best fit image
        error = gauss_class.fit_error(image, decomp_image_masked)
        error_ratio = error/np.sum(np.square(image))

        p_lower_bounds = np.fromiter(
                gauss_class.p_lower_bounds_dict.values(), dtype=np.float64)
        p_upper_bounds = np.fromiter(
                gauss_class.p_upper_bounds_dict.values(), dtype=np.float64)

        self.assertFalse(np.isclose(popt, p_lower_bounds).all())
        self.assertFalse(np.isclose(popt, p_upper_bounds).all())

        # Ensure that error ratio is below 1%
        self.assertTrue(error_ratio < 0.01)

    def test_synthetic_keratin_pattern_manual_guess(self):
        """
        Generate a synthetic diffraction pattern
        and ensure the Gaussian fit error is small
        """
        # Set parameters for a synthetic keratin diffraction pattern
        p_synth_dict = OrderedDict({
                # 9A equatorial peaks parameters
                "peak_location_radius_9A":  feature_pixel_location(9.8e-10), # Peak pixel radius
                "peak_std_9A":              8, # Width
                "peak_amplitude_9A":        400, # Amplitude
                "arc_angle_9A":             1e-1, # Arc angle
                # 5A meridional peaks parameters
                "peak_location_radius_5A":  feature_pixel_location(5.1e-10), # Peak pixel radius
                "peak_std_5A":              2, # Width
                "peak_amplitude_5A":        100, # Amplitude
                "arc_angle_5A":             np.pi/4, # Arc angle
                # 5-4A isotropic region parameters
                "peak_location_radius_5_4A":feature_pixel_location(4.5e-10), # Peak pixel radius
                "peak_std_5_4A":            15, # Width
                "peak_amplitude_5_4A":      100, # Amplitude
                # Background noise parameters
                "contant_bg":        200, # Amplitude
                # Rotation
                "rotation_angle":           10*np.pi/180, # Pattern rotation angle
            })

        # Lower bounds
        p_lower_bounds_dict = OrderedDict()
        p_upper_bounds_dict = OrderedDict()
        p_guess_dict = OrderedDict()

        p_min_factor = 0.7
        p_max_factor = 1.3
        p_guess_factor = 1.1

        for key, value in p_synth_dict.items():
            # Set lower bounds
            p_lower_bounds_dict[key] = p_min_factor*value
            # Set upper bounds
            p_upper_bounds_dict[key] = p_max_factor*value
            # Set guess values
            p_guess_dict[key] = p_guess_factor*value

        # Set mesh size
        size = 256
        RR, TT = gen_meshgrid((size,size))
        TTcopy = TT.copy()

        # Generate synthetic image
        synth_image = keratin_function((RR, TT), *p_synth_dict.values()).reshape(RR.shape)

        # Instantiate gauss class
        gauss_class = GaussianDecomposition(
                synth_image, params_init_method="ideal")

        # Overwrite initial guess and bounds for fitting parameters
        gauss_class.p0_dict = p_guess_dict
        gauss_class.p_lower_bounds_dict = p_lower_bounds_dict
        gauss_class.p_upper_bounds_dict = p_upper_bounds_dict

        # Find Gaussian fit
        popt_dict, pcov = gauss_class.best_fit()
        popt = np.fromiter(popt_dict.values(), dtype=np.float64)
        decomp_image  = keratin_function((RR, TT), *popt).reshape(RR.shape)

        # Ensure TT is not modified
        self.assertTrue(np.array_equal(TTcopy, TT))

        # Get squared error
        error = gauss_class.fit_error(synth_image, decomp_image)
        error_ratio = error/np.sum(np.square(synth_image))
        # Get R-Factor
        r_factor = np.sum(
                np.abs(np.sqrt(synth_image) - np.sqrt(decomp_image))) \
                / np.sum(np.sqrt(synth_image))

        p_lower_bounds = np.fromiter(p_lower_bounds_dict.values(), dtype=np.float64)
        p_upper_bounds = np.fromiter(p_upper_bounds_dict.values(), dtype=np.float64)

        self.assertFalse(np.isclose(popt, p_lower_bounds).all())
        self.assertFalse(np.isclose(popt, p_upper_bounds).all())

        # Ensure that the optimization is not returning the initial guess
        self.assertFalse(np.array_equal(popt_dict, p_guess_dict))

        # Ensure that the R-Factor is below 1%
        self.assertTrue(r_factor < 0.01)

        # Ensure that error ratio is below 1%
        self.assertTrue(error_ratio < 0.01)

        p_synth = np.fromiter(p_synth_dict.values(), dtype=np.float64)

        # Ensure that popt values are close to p_dict values
        self.assertTrue(np.isclose(popt, p_synth).all())

    def test_synthetic_keratin_pattern_9A_zero_arc_angle(self):
        """
        Generate a synthetic diffraction pattern
        and ensure the Gaussian fit error is small
        """
        # Set parameters for a synthetic keratin diffraction pattern
        p_synth_dict = OrderedDict({
                # 9A equatorial peaks parameters
                "peak_location_radius_9A":  feature_pixel_location(9.8e-10), # Peak pixel radius
                "peak_std_9A":              8, # Width
                "peak_amplitude_9A":        400, # Amplitude
                "arc_angle_9A":             0, # Arc angle
                # 5A meridional peaks parameters
                "peak_location_radius_5A":  feature_pixel_location(5.1e-10), # Peak pixel radius
                "peak_std_5A":              2, # Width
                "peak_amplitude_5A":        100, # Amplitude
                "arc_angle_5A":             np.pi/4, # Arc angle
                # 5-4A isotropic region parameters
                "peak_location_radius_5_4A":feature_pixel_location(4.5e-10), # Peak pixel radius
                "peak_std_5_4A":            15, # Width
                "peak_amplitude_5_4A":      100, # Amplitude
                # Background noise parameters
                "contant_bg":        200, # Amplitude
                # Rotation
                "rotation_angle":           10*np.pi/180, # Pattern rotation angle
            })

        # Test that the synthetic 9.8 A peaks are not circular

        self.fail("Finish writing test.")

    def test_synthetic_keratin_pattern_auto_guess(self):
        """
        Generate a synthetic diffraction pattern
        and ensure the Gaussian fit error is small
        """
        # Set parameters for a synthetic keratin diffraction pattern
        p_synth_dict = OrderedDict({
                # 9A equatorial peaks parameters
                "peak_location_radius_9A":  feature_pixel_location(9.8e-10), # Peak pixel radius
                "peak_std_9A":              8, # Width
                "peak_amplitude_9A":        400, # Amplitude
                "arc_angle_9A":             1e-1, # Arc angle
                # 5A meridional peaks parameters
                "peak_location_radius_5A":  feature_pixel_location(5.1e-10), # Peak pixel radius
                "peak_std_5A":              2, # Width
                "peak_amplitude_5A":        100, # Amplitude
                "arc_angle_5A":             np.pi/4, # Arc angle
                # 5-4A isotropic region parameters
                "peak_location_radius_5_4A":feature_pixel_location(4.5e-10), # Peak pixel radius
                "peak_std_5_4A":            15, # Width
                "peak_amplitude_5_4A":      100, # Amplitude
                # Background noise parameters
                "peak_std_bg":              200, # Width
                "peak_amplitude_bg":        200, # Amplitude
            })

        # Set mesh size
        size = 256
        RR, TT = gen_meshgrid((size,size))

        # Generate synthetic image
        synth_image = keratin_function((RR, TT), *p_synth_dict.values()).reshape(RR.shape)
        gauss_class = GaussianDecomposition(synth_image)

        # Get initial paramter guess and bounds
        p0_dict = gauss_class.p0_dict
        p_lower_bounds_dict = gauss_class.p_lower_bounds_dict
        p_upper_bounds_dict = gauss_class.p_upper_bounds_dict

        # Find Gaussian fit
        popt_dict, pcov = gauss_class.best_fit()
        popt = np.fromiter(popt_dict.values(), dtype=np.float64)
        decomp_image  = keratin_function((RR, TT), *popt).reshape(RR.shape)

        # Get squared error
        error = gauss_class.fit_error(synth_image, decomp_image)
        error_ratio = error/np.sum(np.square(synth_image))

        p_lower_bounds = np.fromiter(p_lower_bounds_dict.values(), dtype=np.float64)
        p_upper_bounds = np.fromiter(p_upper_bounds_dict.values(), dtype=np.float64)

        self.assertFalse(np.isclose(popt, p_lower_bounds).all())
        self.assertFalse(np.isclose(popt, p_upper_bounds).all())

        # Ensure automatic parameter guesses are close to the known values
        for key, value in p_synth_dict.items():
            known_value = p_synth_dict[key]
            estimated_value = p0_dict[key]
            # Only check for closeness if a value is bigger than 1
            if known_value > 1 or estimated_value > 1:
                self.assertTrue(np.isclose(estimated_value, known_value, rtol=0.05),
                        "Estimate is too far: {}, {}, {}".format(
                            key, known_value, estimated_value))

        # Ensure that error ratio is below 1%
        self.assertTrue(error_ratio < 0.01)

        p_synth = np.fromiter(p_synth_dict.values(), dtype=np.float64)

        # Ensure that popt values are close to p_dict values
        self.assertTrue(np.isclose(popt, p_synth).all())

    def test_estimate_background_noise(self):
        """
        Test background-noise peak amplitude and standard deviation
        across noise study measurements.
        """
        # Generate a test Gaussian
        size = 256
        shape = size, size
        RR, TT = gen_meshgrid(shape)
        a = 117
        std = 31
        test_gaussian = a*np.exp(-1/2*( (RR/std)**2) )

        # Estimate the parameters
        peak_amplitude, peak_std = estimate_background_noise(test_gaussian)

        # Ensure the estimated parameters are identically close to the known
        # parameters
        self.assertTrue(np.isclose(peak_amplitude, a, rtol=0.05))
        self.assertTrue(np.isclose(peak_std, std, rtol=0.05))

    def test_estimate_background_noise_nonzero(self):
        """
        Test background-noise peak amplitude and standard deviation
        across noise study measurements.
        This tests situations with missing data (0 intensity)
        """
        # Generate a test Gaussian
        size = 256
        shape = size, size
        RR, TT = gen_meshgrid(shape)
        a = 117
        std = 31
        test_gaussian = a*np.exp(-1/2*( (RR/std)**2) )

        # Set inside to zero
        r_min = 25
        test_gaussian[RR < 25] = 0

        # Estimate the parameters
        peak_amplitude, peak_std = estimate_background_noise(test_gaussian)

        # Ensure the estimated parameters are identically close to the known
        # parameters
        self.assertTrue(np.isclose(peak_amplitude, a, rtol=0.05))
        self.assertTrue(np.isclose(peak_std, std, rtol=0.05))

    def test_estimate_parameters(self):
        """
        Test the estimate of all parameters for a synthetic pattern
        """
        p_synth_dict = self.p_synth_dict
        gauss_class = self.gauss_class

        p0_dict = gauss_class.p0_dict

        # Ensure that p0_dict values are near p_synth_dict values
        for key, value in p0_dict.items():
            known_parameter = p_synth_dict[key]
            test_parameter = p0_dict[key]
            # If parameters are small, no need to check
            large_parameters_flag = known_parameter >= 1 or test_parameter >= 1
            if large_parameters_flag:
                if not np.isclose(test_parameter, known_parameter, rtol=0.05):
                    print(key, known_parameter, test_parameter)
                self.assertTrue(np.isclose(test_parameter, known_parameter, rtol=0.05))

    def test_cli_batch_gaussian_decomposition(self):
        """
        Test main cli on test data
        """
        test_dir = "batch_gaussian_decomposition"
        input_path = os.path.join(self.TEST_DATA_PATH, "input", test_dir)

        # Set up the command
        command = ["python", "eosdxanalysis/models/curve_fitting.py",
                    "--input_path", input_path,
                    "--fitting_method", "gaussian-decomposition",
                    ]
        # Run the command
        subprocess.run(command)


        # Remove any directories created
        output_path_list = glob.glob(os.path.join(input_path, "gaussian_decomposition_*"))
        for output_path in output_path_list:
            shutil.rmtree(output_path)

        self.fail("Finish writing test.")


class TestStats(unittest.TestCase):
    """
    Test ``stats`` code
    """

    def setUp(self):
        """
        Set up for ``TestStats`` tests.
        """
        # Set up data path
        TEST_DATA_PATH = os.path.join(TEST_PATH, "data", "feature_stats")
        self.TEST_DATA_PATH = TEST_DATA_PATH

    def test_plot_feature_histograms_cli(self):
        """
        Test CLI
        """
        self.fail("Finish writing test.")

    @unittest.skipIf(os.environ.get("VISUAL_TESTING") != "True",
        "Skip unless testing visuals")
    def test_plot_feature_histograms_visualize(self):
        """
        Test ``plot_feature_histograms`` visualization with some test data
        """
        input_path = os.path.join(self.TEST_DATA_PATH, "input")
        input_filename = "gauss_fit_ideal_features_only_2022_09_20.csv"
        input_filepath = os.path.join(input_path, input_filename)

        # Plot feature histograms
        plot_feature_histograms(input_filepath, visualize=True)

    def test_plot_feature_histograms_save_nonempty_output(self):
        """
        Test ``plot_feature_histograms`` file output with some test data,
        checks if there are non-empty png files.
        """
        input_path = os.path.join(self.TEST_DATA_PATH, "input")
        input_filename = "gauss_fit_ideal_features_only_2022_09_20.csv"
        input_filepath = os.path.join(input_path, input_filename)
        output_path = os.path.join(self.TEST_DATA_PATH, "output")
        # Create output path
        os.makedirs(output_path, exist_ok=True)

        # Plot feature histograms
        plot_feature_histograms(input_filepath, output_path, save=True)

        # Test that ``output_path`` is not empty
        output_file_suffix = "*.png"
        output_filepath_list = glob.glob(os.path.join(output_path, output_file_suffix))

        # Ensure there are some output files
        self.assertTrue(len(output_filepath_list) > 0)
        
        # Ensure the output files are not empty
        for output_filepath in output_filepath_list:
            self.assertTrue(os.stat(output_filepath).st_size > 0)
            # Remove the output file
            os.remove(output_filepath)


class TestUtils(unittest.TestCase):

    def setUp(self):
        # Table values taken from: https://mathworld.wolfram.com/BesselFunctionZeros.html
        # Rows are the kth zeros (start from k=1)
        # Columns are J_n, start from (n=0)
        known_zeros = np.array(
            [[2.4048, 3.8317, 5.1356, 6.3802, 7.5883, 8.7715],
            [5.5201, 7.0156, 8.4172, 9.7610, 11.0647, 12.3386],
            [8.6537, 10.1735, 11.6198, 13.0152, 14.3725, 15.7002],
            [11.7915, 13.3237, 14.7960, 16.2235, 17.6160, 18.9801],
            [14.9309, 16.4706, 17.9598, 19.4094, 20.8269, 22.2178],
            ])
        self.known_zeros = known_zeros

    def test_draw_antialiased_circle(self):
        """
        Ensure we get a proper circle
        """
        radius = 100
        test_circle = draw_antialiased_circle(radius)

        # Check the arc length, ensure it is within 10% of expected arc length
        # s = r * theta
        known_arc_length = radius*2*np.pi
        test_arc_length = np.sum(test_circle)

        # Second argument is used as the rtol reference
        self.assertTrue(np.isclose(test_arc_length, known_arc_length, rtol=0.1))

        # Generate a meshgrid
        center = np.array(test_circle.shape)/2-0.5
        YY, XX = np.ogrid[:test_circle.shape[0], :test_circle.shape[1]]
        RR = np.sqrt((XX-center[1])**2 + (YY-center[0])**2)

        bool_known_circle = np.zeros_like(test_circle)
        bool_known_circle[(RR <= 100+1) & (RR >= 100)] = 1

        bool_test_circle = test_circle >= 0

        bool_overlap_circle = bool_known_circle == bool_test_circle

        bool_overlap_arc_length = np.sum(bool_overlap_circle)

        # Check that the overlap arc length is also close to the known arc length
        # Second argument is used as the rtol reference
        self.assertTrue(np.isclose(bool_overlap_arc_length, known_arc_length, rtol=0.1))

    def test_draw_antialiased_arc_zero_arc_angle_spread(self):
        """
        Zero spread should raise a ValueError
        """
        size = 15
        shape = size, size
        center = np.array(shape)/2-0.5

        arc_radius = 100
        arc_start_angle = 0
        arc_angle_spread = 0

        self.assertRaises(ValueError, draw_antialiased_arc,
                arc_radius, arc_start_angle, arc_angle_spread, shape)

    def test_gen_jn_zerosmatrix(self):
        # Test if values are close to table values with relative tolerance
        known_zeros = self.known_zeros
        nthorder = 6
        kzeros = 5
        zeromatrix = gen_jn_zerosmatrix((nthorder,kzeros))
        rtol=1e-3
        self.assertTrue(True == np.isclose(zeromatrix, known_zeros.T,rtol=rtol).all())

    def test_bessel_zeros_excluding_origin(self):
        """
        In `scipy.special.jn_zeros`, we are excluding zeros at x = 0.
        In the 2D Polar Discrete Fourier Transform Matlab code,
        we are also exluding zeros at x = 0.
        Make sure we are using consistent scheme.
        """
        known_zeros = self.known_zeros
        # J_0: J_n for n = 0
        # Get the first 3 zeros
        n0 = 0
        J_0_zeros_calc = jn_zeros(n0, 3)
        # Check that jn_zeros generates the correct zeros
        # starting after x = 0
        J_0_zeros_ref = known_zeros[:3, n0]
        self.assertTrue(np.isclose(J_0_zeros_calc, J_0_zeros_ref, atol=1e-3).all())

        # J_1: J_n for n = 1
        # Get the first 3 zeros
        n1 = 1
        J_1_zeros_calc = jn_zeros(n1, 3)
        # Check that jn_zeros generates the correct zeros
        # starting after x = 0
        J_1_zeros_ref = known_zeros[:3, n1]
        self.assertTrue(np.isclose(J_1_zeros_calc, J_1_zeros_ref, atol=1e-3).all())

    def test_radial_intensity_1d(self):
        """
        Test radial intensity 1d function
        """
        test_image = np.zeros((256,256))
        test_image[:,128:] = np.sin(np.arange(128))

        horizontal_1d = radial_intensity_1d(test_image, width=4)

        self.assertTrue(np.array_equal(horizontal_1d, np.sin(np.arange(128))))

    def test_angular_intensity_1d(self):
        """
        Test angular intensity 1d function with a simple example
        """
        # Create test image
        size = 256
        test_image = np.zeros((size,size))
        # Create meshgrid
        YY, XX = np.ogrid[:size, :size]
        center = (size/2-0.5, size/2-0.5)
        # Calculate meshgrid of radii
        dist_from_center = np.sqrt((YY - center[0])**2 + (XX - center[1])**2)
        # Calculate meshgrid of angles
        angle = np.arctan2(YY-center[0], XX-center[1])
        # Create a ring sinusoid
        radius_start = 50
        radius_end = 75
        ring = (dist_from_center >= 50) & (dist_from_center <= 75)
        test_image[ring] = np.cos(2*angle)[ring]

        N = 360
        ring_intensity = angular_intensity_1d(test_image, radius=(radius_start+radius_end)/2, N=N)

        # Ensure the ring intensity follows a sinusoid, within a certain tolerance
        expected_intensity = np.cos(2*np.linspace(-np.pi + 2*np.pi/N/2, np.pi - 2*np.pi/N/2,
            num=N, endpoint=True))

        self.assertTrue(np.isclose(ring_intensity, expected_intensity).all())


class TestFeatureEngineering(unittest.TestCase):

    def test_amorphous_scattering_template(self):
        """
        Amorphous scattering of the scattering template should be zero
        """
        template_filename = "amorphous-scattering-template.txt"
        template_path = os.path.join(MODULE_DATA_PATH, template_filename)
        template = np.loadtxt(template_path)

        known_intensity = 0

        feature_class = EngineeredFeatures(template, params=None)

        amorphous_intensity = feature_class.feature_amorphous_scattering_intensity_ratio()

        self.assertTrue(np.isclose(amorphous_intensity, known_intensity))

    def test_9a_peak_location(self):
        """
        Test the function to find the 9A peak location
        """
        # Create a test image
        test_image = np.zeros((256,256))
        # Set a peak in the 9A region of interest
        SPACING_9A = 9.8e-10 # meters
        theory_peak_location = feature_pixel_location(SPACING_9A,
                distance=DISTANCE, wavelength=WAVELENGTH, pixel_width=PIXEL_WIDTH)
        center = (test_image.shape[0]/2-0.5, test_image.shape[1]/2-0.5)
        peak_row = int(center[0])
        peak_col = int(center[1]+theory_peak_location)
        test_image[peak_row,peak_col] = 1

        feature_class = EngineeredFeatures(test_image, params=None)

        roi_peak_location, _, _, _, _ = feature_class.feature_9a_peak_location()

        self.assertEqual(roi_peak_location, peak_col)

    def test_5a_peak_location(self):
        # Create a test image
        size = 256
        test_image = np.zeros((size,size))
        center = test_image.shape[0]/2-0.5, test_image.shape[1]/2-0.5

        # Set a peak in the 5A region of interest
        SPACING_5A = 5e-10 # meters
        theory_peak_location = feature_pixel_location(SPACING_5A,
                distance=DISTANCE, wavelength=WAVELENGTH, pixel_width=PIXEL_WIDTH)
        known_peak_location = int(center[0] - theory_peak_location)
        test_image[known_peak_location,int(center[1])] = 1

        feature_class = EngineeredFeatures(test_image, params=None)

        roi_peak_location, _, _, _ = feature_class.feature_5a_peak_location()

        self.assertEqual(roi_peak_location, known_peak_location)

    def test_9a_ratio(self):
        # Create a test image
        size = 256
        test_image = np.zeros((size,size))
        center = test_image.shape[0]/2-0.5, test_image.shape[1]/2-0.5
        # Create some blobs in the 9.8A region of interest
        SPACING_9A = 9.8e-10 # meters
        theory_peak_location = feature_pixel_location(SPACING_9A,
                distance=DISTANCE, wavelength=WAVELENGTH, pixel_width=PIXEL_WIDTH)
        rect_w = 6
        rect_h = 20
        # Set the right area to 2
        test_image[int(center[0]-rect_h/2):int(center[0]+rect_h/2),
                int(center[1]+theory_peak_location-rect_w/2):int(center[1]+theory_peak_location+rect_w/2)] = 2

        # Set the top area to 1
        test_image[int(center[1]-theory_peak_location-rect_w/2):int(center[1]-theory_peak_location+rect_w/2),
                int(center[0]-rect_h/2):int(center[0]+rect_h/2)] = 1

        feature_class = EngineeredFeatures(test_image, params=None)

        # Get the 9A peak ratio
        test_ratio, test_rois, test_centers, test_anchors = \
                feature_class.feature_9a_ratio()
        known_ratio = 2 # 2/1 = 2

        # Ensure the intensity is as expected
        self.assertEqual(test_ratio, known_ratio)

        # Ensure the rois are as expected
        roi_right = test_rois[0]
        roi_top = test_rois[2]

        self.assertEqual(np.mean(roi_top),1)
        self.assertEqual(np.mean(roi_right),2)

    def test_feature_5a_9a_peak_location_ratio(self):
        # Create a test image
        test_image = np.zeros((256,256))
        # set a peak in the 5A region of interest
        peak_5a_row = 60
        peak_5a_radius = 256//2-60
        center_cols = slice(127, 128)
        test_image[peak_5a_row,center_cols] = 1

        # Create some blobs in the 9.8A region of interest
        rect_w = 4
        rect_l = 18
        start_radius = 25
        peak_9a_radius = start_radius + rect_w

        # Set the right area to 2
        test_image[128-rect_l//2:128+rect_l//2,
                128+start_radius:128+start_radius+rect_w] = 2

        feature_class = EngineeredFeatures(test_image, params=None)
        # ratio = peak_5a_radius/peak_9a_radius
        peak_location_ratio = feature_class.feature_5a_9a_peak_location_ratio()

        self.assertTrue(np.isclose(peak_location_ratio, peak_5a_radius/peak_9a_radius))

    def test_fwhm(self):
        """
        Test full-width half maximum function

        Test function is a triangular function as follows:
        (max = 2.1)
               /\
              /  \
             /    \
        _________________
            |  |   |
           -2  0   2

        """
        size = 101
        min_val = 0
        max_val = 2
        half_size = size//2
        test_array = np.zeros(size)
        test_array[:half_size+1] = np.linspace(min_val,max_val+max_val/(size-1),half_size+1)
        test_array[half_size+1:] = test_array[:half_size][::-1]

        known_fwhm = half_size

        feature_class = EngineeredFeatures
        test_fwhm, max_val, max_loc, half_max_val, half_max_loc = feature_class.fwhm(test_array)

        self.assertEqual(known_fwhm, test_fwhm)

class TestL1Metric(unittest.TestCase):

    def test_l1_metric(self):
        # Create test matrices
        A = np.array([
            [1,0],
            [0,1],
            ])
        B = np.array([
            [0,1,],
            [1,0,],
            ])
        distance = 4/A.size

        l1 = l1_metric(A,B)

        self.assertEqual(distance, l1)


class TestPolarSamplingGrid(unittest.TestCase):
    """
    Test Python port of functions to create polar grid for
    2D Polar Discrete Fourier Transform
    """

    def setUp(self):
        """
        Load Jn zeros from file
        """
        testdata_path = os.path.join(
                TEST_PATH,
                JN_ZEROSMATRIX_TEST_DIR)
        jn_zerosmatrix_fullpath = os.path.join(testdata_path,
                JN_ZEROSMATRIX_FILENAME)
        jn_zerosmatrix = np.load(jn_zerosmatrix_fullpath)
        self.jn_zerosmatrix = jn_zerosmatrix
        self.testdata_path = testdata_path

    def test_rmatrix_generation_space_limited(self):
        """
        Ensure the correct rmatrix is generated
        """
        jn_zerosmatrix = self.jn_zerosmatrix
        # Set up test sampling parameters for space-limited function
        N1 = 16
        N2 = 15
        R = 1

        # Generate the rmatrix
        rmatrix = rmatrix_SpaceLimited(N2, N1, R)

        # Check some values manually
        p = -7
        k = 1
        rpk_manual = jn_zerosmatrix[abs(p), k-1]/jn_zerosmatrix[abs(p), N1-1]*R

        known_value = 0.18455912342241768
        self.assertEqual(rmatrix[0, 0], rpk_manual)
        self.assertTrue(np.isclose(rmatrix[0,0], known_value))
        self.assertTrue(np.isclose(rpk_manual, known_value))

        p = -6
        k = 1
        rpk_manual = jn_zerosmatrix[abs(p), k-1]/jn_zerosmatrix[abs(p), N1-1]*R

        known_value = 0.16955932
        self.assertEqual(rmatrix[1, 0], rpk_manual)
        self.assertTrue(np.isclose(rmatrix[1,0], known_value))
        self.assertTrue(np.isclose(rpk_manual, known_value))

        p = 0
        k = 5
        rpk_manual = jn_zerosmatrix[abs(p), k-1]/jn_zerosmatrix[abs(p), N1-1]*R

        known_value = 0.30174070727973007
        self.assertEqual(rmatrix[7, 4], rpk_manual)
        self.assertTrue(np.isclose(rmatrix[7,4], known_value))
        self.assertTrue(np.isclose(rpk_manual, known_value))

    @unittest.skipIf(os.environ.get("VISUAL_TESTING") != "True",
        "Skip unless testing visuals")
    def test_polar_sampling_plot(self):
        """
        Visualize the sampling grid
        """
        jn_zerosmatrix = self.jn_zerosmatrix
        # Set up test sampling parameters for space-limited function
        N1 = 16
        N2 = 15
        R = 1

        # Generate the radial matrix
        rmatrix = rmatrix_SpaceLimited(N2, N1, R)

        # Generate the angular matrix
        thetamatrix = thetamatrix_SpaceLimited(N2, N1)

        # Plot polar
        import matplotlib.pyplot as plt

        fig1, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter(thetamatrix, rmatrix)
        ax.set_rlabel_position(-22.5)

        plt.show()

    def test_polar_sampling_different_size(self):
        """
        Ensure that we can generate a polar sampling grid
        for N1 and N2 values that are significantly different.

        There was a bug where only N1 = N2 + 1 worked.
        """

        jn_zerosmatrix = self.jn_zerosmatrix
        # Set up test sampling parameters for space-limited function
        N1 = 20
        N2 = 15
        R = 1

        # Generate the radial matrix
        rmatrix = rmatrix_SpaceLimited(N2, N1, R)

        # Check that the radial matrix shape is (N2, N1-1)
        self.assertTrue(np.array_equal(rmatrix.shape, (N2, N1-1)))

        # Generate the angular matrix
        thetamatrix = thetamatrix_SpaceLimited(N2, N1)

        # Check that the angular matrix shape is (N2, N1-1)
        self.assertTrue(np.array_equal(thetamatrix.shape, (N2, N1-1)))

    def test_rhomatrix_SpaceLimited(self):
        """
        Ensure the correct rhomatrix is generated
        """
        jn_zerosmatrix = self.jn_zerosmatrix
        # Set up test sampling parameters for space-limited function
        N1 = 100
        N2 = 101
        R = 1

        # Generate the rhomatrix
        rhomatrix = rhomatrix_SpaceLimited(N2, N1, R)

        # Load the known rhomatrix
        rhomatrix_fullpath = os.path.join(self.testdata_path,
                "rhomatrix_101_100.mat")
        known_rhomatrix = loadmat(rhomatrix_fullpath).get("rhomatrix")

        # Check that they're equal
        self.assertTrue(np.isclose(rhomatrix, known_rhomatrix).all())

    def test_psimatrix_SpaceLimited(self):
        """
        Ensure the correct psimatrix is generated
        """
        jn_zerosmatrix = self.jn_zerosmatrix
        # Set up test sampling parameters for space-limited function
        N1 = 100
        N2 = 101
        R = 1

        # Generate the psimatrix
        psimatrix = psimatrix_SpaceLimited(N2, N1)

        # Load the known psimatrix
        psimatrix_fullpath = os.path.join(self.testdata_path,
                "psimatrix_101_100.mat")
        known_psimatrix = loadmat(psimatrix_fullpath).get("psimatrix")

        # Check that they're equal
        self.assertTrue(np.isclose(psimatrix, known_psimatrix).all())


class TestFourierAnalysis(unittest.TestCase):

    def setUp(self):
        """
        Load Jn zeros from file
        """
        testdata_path = os.path.join(
                TEST_PATH,
                JN_ZEROSMATRIX_TEST_DIR)
        jn_zerosmatrix_fullpath = os.path.join(testdata_path,
                JN_ZEROSMATRIX_FILENAME)
        jn_zerosmatrix = np.load(jn_zerosmatrix_fullpath)
        self.jn_zerosmatrix = jn_zerosmatrix
        self.testdata_path = testdata_path

    def test_Ymatrix_Assembly_size(self):
        """
        Check that Ymatrix size is correct for non-trivial input values
        """
        jn_zerosmatrix = self.jn_zerosmatrix

        # Set parameters
        count = 5
        N1 = 4
        known_shape = (N1-1, N1-1)

        for n in range(count):
            # Extract jn zeros array
            jn_zerosarray = jn_zerosmatrix[n, :N1]

            # Generate the Ymatrix
            ymatrix = YmatrixAssembly(n, N1, jn_zerosarray)

            # Check that the matrix size is correct
            self.assertTrue(np.array_equal(ymatrix.shape, known_shape))

    def test_Ymatrix_Assembly_manual_values(self):
        """
        Test the Ymatrix against manual calculations
        """
        jn_zerosmatrix = self.jn_zerosmatrix

        # Set parameters
        n = 0
        N1 = 4
        known_shape = (N1-1, N1-1)

        # Extract jn zeros array
        jn_zerosarray = jn_zerosmatrix[n, :N1]

        # Generate the Ymatrix
        ymatrix = YmatrixAssembly(n, N1, jn_zerosarray)

        # Load the known ymatrix
        ymatrix_fullpath = os.path.join(self.testdata_path,
                "ymatrix_0_4.mat")
        known_ymatrix = loadmat(ymatrix_fullpath).get("ymatrix")

        # Check that they're equal
        self.assertTrue(np.isclose(ymatrix, known_ymatrix).all())

    def test_Ymatrix_Assembly_vectorized_manual_values(self):
        """
        Test the Ymatrix against manual calculations
        """
        jn_zerosmatrix = self.jn_zerosmatrix

        # Set parameters
        n = np.array([0,1])
        N1 = 4
        known_shape = (n.size, N1-1, N1-1)

        # Extract jn zeros array
        jn_zeros = jn_zerosmatrix[n, :N1]

        # Generate the Ymatrix
        ymatrix = YmatrixAssembly(n, N1, jn_zeros)

        # Check the shape
        self.assertEqual(ymatrix.shape, known_shape)

        # Load the known ymatrices
        # n = 0, N1 = 4
        ymatrix_0_4_fullpath = os.path.join(self.testdata_path,
                "ymatrix_0_4.mat")
        known_ymatrix_0_4 = loadmat(ymatrix_0_4_fullpath ).get("ymatrix")
        # n = +1, N1 = 4
        ymatrix_1_4_fullpath = os.path.join(self.testdata_path,
                "ymatrix_1_4.mat")
        known_ymatrix_1_4 = loadmat(ymatrix_1_4_fullpath ).get("ymatrix")

        # Check that they're equal
        self.assertTrue(np.isclose(ymatrix[0,...],
                                    known_ymatrix_0_4).all())
        self.assertTrue(np.isclose(ymatrix[1,...],
                                    known_ymatrix_1_4).all())

    def test_pfft2_SpaceLimited_continuous_input(self):
        """
        Test 2D Discrete Polar Fourier Transform
        for a space-limited continuous function
        follows `test_gaussian.m` example
        """
        # Set sampling rates
        N1 = 4 # Radial sampling rate
        N2 = 5 # Angular sampling rate
        # Space limit
        R = 10

        # Gaussian
        a = 0.1
        gau = lambda x, a : np.exp(-(a*x)**2)

        rmatrix = rmatrix_SpaceLimited(N2, N1, R)

        f = gau(rmatrix, a)

        dft = pfft2_SpaceLimited(f, N1, N2, R)

        # Check against known result
        # Load the known DFT matrix
        dft_fullpath = os.path.join(self.testdata_path,
                "dft_gaussian.mat")
        known_dft = loadmat(dft_fullpath).get("dft_gaussian")

        self.assertTrue(np.isclose(dft, known_dft).all())

    def test_pfft2_SpaceLimited_discrete_input(self):
        """
        Test 2D Discrete Polar Fourier Transform
        for a space-limited discrete function
        """
        # Set sampling rates
        N1 = 4 # Radial sampling rate
        N2 = 5 # Angular sampling rate
        # Space limit
        R = 10

        # Gaussian
        a = 0.1
        gau = lambda x, a : np.exp(-(a*x)**2)

        # Let's create a real measurement of our Gaussian
        # Set up our resolution
        dx = 0.2
        dy = 0.2

        # Let's create a meshgrid,
        # note that x and y have even length
        x = np.arange(-R+dx/2, R+dx/2, dx)
        y = np.arange(-R+dx/2, R+dx/2, dy)
        XX, YY = np.meshgrid(x, y)

        RR = np.sqrt(XX**2 + YY**2)

        discrete_image = np.exp(-(a*RR)**2)

        origin = (discrete_image.shape[0]/2-0.5, discrete_image.shape[1]/2-0.5)

        # Now sample the discrete image according to the Baddour polar grid
        # First get rmatrix and thetamatrix
        thetamatrix, rmatrix = sampling_grid(N1, N2, R)
        # Now convert rmatrix to Cartesian coordinates
        Xcart = rmatrix*np.cos(thetamatrix)/dx
        Ycart = rmatrix*np.sin(thetamatrix)/dy
        # Now convert Cartesian coordinates to the array notation
        # by shifting according to the origin
        Xindices = Xcart + origin[0]
        Yindices = origin[1] - Ycart

        cart_sampling_indices = [Yindices, Xindices]

        fdiscrete = map_coordinates(discrete_image, cart_sampling_indices)
        fcontinuous = gau(rmatrix, a)

        # Check that these two are close
        self.assertTrue(np.isclose(fdiscrete, fcontinuous).all())

        dft = pfft2_SpaceLimited(fdiscrete, N1, N2, R)

        # Check against known result
        # Load the known DFT matrix
        dft_fullpath = os.path.join(self.testdata_path,
                "dft_gaussian.mat")
        known_dft = loadmat(dft_fullpath).get("dft_gaussian")

        self.assertTrue(np.isclose(dft, known_dft).all())

    def test_ipfft2_SpaceLimited_continuous_input(self):
        """
        Test 2D Discrete Inverse Polar Fourier Transform
        for a space-limited continuous function
        """
        # Set sampling rates
        N1 = 4 # Radial sampling rate
        N2 = 5 # Angular sampling rate
        # Space limit
        R = 10

        # Gaussian
        a = 0.1
        # Frequency domain
        gau2 = lambda x, a : np.pi/(a**2)*np.exp(-((x/a)**2)/4)

        rmatrix = rmatrix_SpaceLimited(N2, N1, R)
        rhomatrix = rhomatrix_SpaceLimited(N2, N1, R)
        
        # Frequency domain
        freq_gaussian = gau2(rhomatrix, 1.0)

        # Take inverse DFT of frequency domain Gaussian
        idft = ipfft2_SpaceLimited(freq_gaussian, N1, N2, R)

        # Compare to results from Matlab code
        idft_gaussian_fullpath = os.path.join(self.testdata_path,
                "idft_gaussian.mat")
        idft_gaussian = loadmat(idft_gaussian_fullpath).get("idft_gaussian")

        self.assertTrue(np.isclose(idft, idft_gaussian).all())


class TestPolarGaussian(unittest.TestCase):
    """
    Tests of the polar_gaussian function
    """

    def test_polar_gaussian(self):
        """
        Test
        """
        size = 256
        shape = size, size
        RR, TT = gen_meshgrid(shape)

        # Set function parameters
        peak_radius = 0
        peak_angle = 0
        peak_radial_std = 5
        peak_azimuthal_std = 5
        peak_amplitude = 100

        gau = polar_gaussian(RR, TT, peak_radius, peak_angle, peak_radial_std,
                peak_azimuthal_std, peak_amplitude)

        import matplotlib.pyplot as plt
        plt.imshow(gau)
        plt.show()
        import ipdb
        ipdb.set_trace()

        self.fail("Finish writing test")


if __name__ == '__main__':
    unittest.main()
