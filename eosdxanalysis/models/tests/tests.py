"""
Tests for models module
"""
import os
import unittest
import numpy as np
import numpy.ma as ma
import subprocess

from collections import OrderedDict

from scipy.special import jn_zeros
from scipy.special import jv
from scipy.io import loadmat
from scipy.ndimage import map_coordinates

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from eosdxanalysis.models.curve_fitting import PolynomialFit
from eosdxanalysis.models.curve_fitting import GaussianDecomposition
from eosdxanalysis.models.utils import gen_jn_zerosmatrix
from eosdxanalysis.models.utils import l1_metric
from eosdxanalysis.models.utils import pol2cart
from eosdxanalysis.models.utils import cart2pol
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
        TEST_DATA_PATH = os.path.join(TEST_PATH, "data", "GaussianDecomposition")
        self.TEST_DATA_PATH = TEST_DATA_PATH

    def test_bg_noise_peak_location(self):
        """
        Background noise peak should be close to 0 (right at the origin
        or center of the image)
        """
        # Set input filepath
        test_filename = "CRQF_A00005.txt"
        test_filepath = os.path.join(self.TEST_DATA_PATH, "input", test_filename)
        # Find best-fit parameters
        test_image = np.loadtxt(test_filepath, dtype=np.uint32)
        popt, pcov, RR, TT = GaussianDecomposition.best_fit(test_image)

        # Check that background-noise peak location is close to zero
        bg_peak_location = popt[-4]
        self.assertTrue(np.isclose(bg_peak_location, 0))

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
        # Set input filepath
        input_filename = "CRQF_A00005.txt"
        input_filepath = os.path.join(self.TEST_DATA_PATH, "input", input_filename)
        # Set output filepath
        output_filename = "test_GaussianDecomp_CRQF_A00005.txt"
        output_filepath = os.path.join(self.TEST_DATA_PATH, "output", output_filename)

        # Set up the command
        command = ["python", "eosdxanalysis/models/curve_fitting.py",
                    "--input_filepath", input_filepath,
                    "--output_filepath", output_filepath,
                    ]
        # Run the command
        subprocess.run(command)

        # Check that the output is the same as the test output file
        known_output_filename = "GaussianDecomp_CRQF_A00005.txt"
        known_output_filepath = os.path.join(self.TEST_DATA_PATH, "output", known_output_filename)
        known_output = np.loadtxt(known_output_filepath, dtype=np.uint32)
        test_output = np.loadtxt(output_filepath, dtype=np.uint32)

        self.assertTrue(np.isclose(known_output, test_output).all())

    def test_known_sample_bounds(self):
        """
        Test `GaussianDecomposition` on a known sample
        to ensure optimal parameters are not near bounds
        """
        # Set input filepath
        input_filename = "CRQF_A00005.txt"
        input_filepath = os.path.join(self.TEST_DATA_PATH, "input", input_filename)
        # Set output filepath
        output_filename = "test_GaussianDecomp_CRQF_A00005.txt"
        output_filepath = os.path.join(self.TEST_DATA_PATH, "output", output_filename)

        # Calculate optimum parameters
        image = np.loadtxt(input_filepath, dtype=np.float64)

        popt, pcov, RR, TT = GaussianDecomposition.best_fit(image)
        decomp_image  = GaussianDecomposition.keratin_function((RR, TT), *popt).reshape(image.shape)

        # Check that the optimal parameters are not close to the upper or lower bounds
        p0 = np.fromiter(GaussianDecomposition.p0_dict.values(), dtype=np.float64)
        p_lower_bounds = np.fromiter(GaussianDecomposition.p_lower_bounds_dict.values(), dtype=np.float64)
        p_upper_bounds = np.fromiter(GaussianDecomposition.p_upper_bounds_dict.values(), dtype=np.float64)

        self.assertFalse(np.isclose(popt, p_lower_bounds).all())
        self.assertFalse(np.isclose(popt, p_upper_bounds).all())

        # Check that the output is the same as the test output file
        known_output_filename = "GaussianDecomp_CRQF_A00005.txt"
        known_output_filepath = os.path.join(self.TEST_DATA_PATH, "output", known_output_filename)
        known_output = np.loadtxt(known_output_filepath, dtype=np.uint32)
        test_output = np.loadtxt(output_filepath, dtype=np.uint32)

        self.assertTrue(np.isclose(known_output, test_output).all())

    def test_synthetic_keratin_pattern(self):
        """
        Generate a synthetic diffraction pattern
        and ensure the Gaussian fit error is small
        """
        # Set parameters for a synthetic keratin diffraction pattern
        p_dict = OrderedDict({
                # 9A equatorial peaks minimum parameters
                "peak_radius_9A":       feature_pixel_location(9e-10), # Peak pixel radius
                "width_9A":             1, # Width
                "amplitude_9A":         100, # Amplitude
                "cosine_power_9A":      8, # cosine power
                # 5A meridional peaks minimum parameters
                "peak_radius_5A":       feature_pixel_location(5e-10), # Peak pixel radius
                "width_5A":             2, # Width
                "amplitude_5A":         20, # Amplitude
                "cosine_power_5A":      6, # cosine power
                # 5-4A isotropic region minimum parameters
                "peak_radius_5_4A":     feature_pixel_location(4.9e-10), # Peak pixel radius
                "width_5_4A":           5, # Width
                "amplitude_5_4A":       50, # Amplitude
                # Background noise minimum parameters
                "peak_radius_bg":       0, # Peak pixel radius
                "width_bg":             100, # Width
                "amplitude_bg":         100, # Amplitude
            })

        # Set mesh size
        size = 256
        RR, TT = GaussianDecomposition.gen_meshgrid((size,size))

        # Generate synthetic image
        synth_image = GaussianDecomposition.keratin_function((RR, TT), *p_dict.values()).reshape(RR.shape)

        # Set guess parameters
        p0_dict = OrderedDict({
                # 9A equatorial peaks minimum parameters
                "peak_radius_9A":       feature_pixel_location(9e-10), # Peak pixel radius
                "width_9A":             1, # Width
                "amplitude_9A":         100, # Amplitude
                "cosine_power_9A":      8, # cosine power
                # 5A meridional peaks minimum parameters
                "peak_radius_5A":       feature_pixel_location(5e-10), # Peak pixel radius
                "width_5A":             2, # Width
                "amplitude_5A":         20, # Amplitude
                "cosine_power_5A":      6, # cosine power
                # 5-4A isotropic region minimum parameters
                "peak_radius_5_4A":     feature_pixel_location(4.9e-10), # Peak pixel radius
                "width_5_4A":           5, # Width
                "amplitude_5_4A":       50, # Amplitude
                # Background noise minimum parameters
                "peak_radius_bg":       0, # Peak pixel radius
                "width_bg":             100, # Width
                "amplitude_bg":         100, # Amplitude
            })

        # Find Gaussian fit
        popt, pcov, RR, TT = GaussianDecomposition.best_fit(synth_image)
        decomp_image  = GaussianDecomposition.keratin_function((RR, TT), *popt).reshape(RR.shape)

        # Get squared error
        error = GaussianDecomposition.fit_error(synth_image, decomp_image)
        error_ratio = error/np.sum(np.square(synth_image))

        # Ensure that error ratio is below 1%
        self.assertTrue(error_ratio < 0.01)

        # Ensure that popt values are close to p_dict values
        self.assertTrue(np.array_equal(popt, p_dict.values()))


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


if __name__ == '__main__':
    unittest.main()
