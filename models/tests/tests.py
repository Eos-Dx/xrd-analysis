"""
Tests for models module
"""
import os
import unittest
import numpy as np
import numpy.ma as ma

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from models.curve_fitting import PolynomialFit
from models.utils import gen_zeromatrix


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

    def test_bessel_zeros(self):
        # Test if values are close to table values with relative tolerance
        nthorder = 6
        kzeros = 5
        zeromatrix = gen_zeromatrix((nthorder,kzeros))
        # Table values taken from: https://mathworld.wolfram.com/BesselFunctionZeros.html
        known_zeros = np.array(
            [[2.4048, 3.8317, 5.1356, 6.3802, 7.5883, 8.7715],
            [5.5201, 7.0156, 8.4172, 9.7610, 11.0647, 12.3386],
            [8.6537, 10.1735, 11.6198, 13.0152, 14.3725, 15.7002],
            [11.7915, 13.3237, 14.7960, 16.2235, 17.6160, 18.9801],
            [14.9309, 16.4706, 17.9598, 19.4094, 20.8269, 22.2178],
            ])
        rtol=1e-3
        self.assertTrue(True == np.isclose(zeromatrix, known_zeros.T,rtol=rtol).all())


if __name__ == '__main__':
    unittest.main()
