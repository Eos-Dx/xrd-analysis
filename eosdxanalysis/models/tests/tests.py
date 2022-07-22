"""
Tests for models module
"""
import os
import unittest
import numpy as np
import numpy.ma as ma

from scipy.special import jn_zeros
from scipy.special import jv
from scipy.io import loadmat
from scipy.ndimage import map_coordinates

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from eosdxanalysis.models.curve_fitting import PolynomialFit
from eosdxanalysis.models.utils import gen_jn_zerosmatrix
from eosdxanalysis.models.utils import l1_metric
from eosdxanalysis.models.feature_engineering import feature_5a_peak_location
from eosdxanalysis.models.feature_engineering import feature_9a_ratio
from eosdxanalysis.models.polar_sampling import sampling_grid
from eosdxanalysis.models.polar_sampling import rmatrix_SpaceLimited
from eosdxanalysis.models.polar_sampling import thetamatrix_SpaceLimited
from eosdxanalysis.models.polar_sampling import rhomatrix_SpaceLimited
from eosdxanalysis.models.fourier_analysis import YmatrixAssembly
from eosdxanalysis.models.fourier_analysis import pfft2_SpaceLimited

TEST_PATH = os.path.join("eosdxanalysis", "models", "tests")
JN_ZEROSMATRIX_TEST_DIR = "test_jn_zerosmatrix"
JN_ZEROSMATRIX_FILENAME = "jn_zerosmatrix.npy"


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

    def test_5a_peak_location(self):
        # Create a test image
        test_image = np.zeros((256,256))
        # set a peak in the 5A region of interest
        peak_row = 60
        center_col = 128
        test_image[peak_row,center_col] = 1

        roi_peak_location, _, _, _ = feature_5a_peak_location(test_image)
        abs_peak_location = roi_peak_location

        self.assertEqual(abs_peak_location, peak_row)

    def test_9a_ratio(self):
        # Create a test image
        test_image = np.zeros((256,256))
        # Create some blobs in the 9.8A region of interest
        rect_w = 4
        rect_l = 18
        start_radius = 25
        # Set the top area to 1
        test_image[128-start_radius-rect_w:128-start_radius,
                128-rect_l//2:128+rect_l//2] = 1

        # Set the right area to 2
        test_image[128-rect_l//2:128+rect_l//2,
                128+start_radius:128+start_radius+rect_w] = 2

        # Get the 9A peak ratio
        test_ratio, test_rois, test_centers, test_anchors = feature_9a_ratio(test_image,
                start_radius=start_radius)
        known_ratio = 2 # 2/1 = 2

        # Ensure the intensity is as expected
        self.assertEqual(test_ratio, known_ratio)

        # Ensure the rois are as expected
        roi_right = test_rois[0]
        roi_top = test_rois[2]

        self.assertEqual(np.mean(roi_top),1)
        self.assertEqual(np.mean(roi_right),2)


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
        Ensure the correct rmatrix is generated
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
            jn_zerosarray = jn_zerosmatrix[n, :]

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
        n = np.array([-1,0,1])
        N1 = 4
        known_shape = (N1-1, N1-1)

        # Extract jn zeros array
        jn_zeros = jn_zerosmatrix[n, :N1]

        # Generate the Ymatrix
        ymatrix = YmatrixAssembly(n, N1, jn_zeros)

        # Check the shape
        self.assertEqual(ymatrix.shape, (N1-1, N1-1, n.size))

        # Load the known ymatrices
        # n = -1, N1 = 4
        ymatrix_neg1_4_fullpath = os.path.join(self.testdata_path,
                "ymatrix_neg1_4.mat")
        known_ymatrix_neg1_4 = loadmat(ymatrix_neg1_4_fullpath ).get("ymatrix")
        # n = 0, N1 = 4
        ymatrix_0_4_fullpath = os.path.join(self.testdata_path,
                "ymatrix_0_4.mat")
        known_ymatrix_0_4 = loadmat(ymatrix_0_4_fullpath ).get("ymatrix")
        # n = +1, N1 = 4
        ymatrix_1_4_fullpath = os.path.join(self.testdata_path,
                "ymatrix_1_4.mat")
        known_ymatrix_p1_4 = loadmat(ymatrix_1_4_fullpath ).get("ymatrix")

        # Check that they're equal
        self.assertTrue(np.isclose(ymatrix[...,0],
                                    known_ymatrix_neg1_4).all())
        self.assertTrue(np.isclose(ymatrix[...,1],
                                    known_ymatrix_0_4).all())
        self.assertTrue(np.isclose(ymatrix[...,2],
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
        rmatrix, thetamatrix = sampling_grid(N1, N2, R)
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


if __name__ == '__main__':
    unittest.main()
