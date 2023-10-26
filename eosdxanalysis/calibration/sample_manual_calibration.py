"""Convert to q-units based on presence of fat peak (14.1 per nm)
or water peak (20 per nm)


q -> angle -> position on detector

Steps:
    1. Get position on detector using Gaussian fitting
    2. Calculate sample distance using ``sample_distance_from_q``
    3. Convert units using ``radial_profile_unit_conversion``
"""

import numpy as np

from scipy.optimize import curve_fit


# Set peak values [nm^-1]
q_fat = 14.1
q_h2o = 20


class GaussianFit(object):
    """
    Gaussian fit
    """

    def __init__(self):
        """
        """
        pass

    @classmethod
    def best_fit(self, xdata, ydata, p0=None, p_bounds=None):
        """
        xdata = space
        ydata = sample values
        """
        popt, pcov = curve_fit(
                self.gaussian_function, xdata, ydata,
                p0=p0, p_bounds=p_bounds)
        return popt, pcov

    @classmethod
    def initial_parameters(self, calculated_distance=None):
        pass

    @classmethod
    def parameter_bounds(self, calculated_distance=None):
        pass

    @classmethod
    def gaussian_function(self, x, mu, sigma, amplitude):
        """
        """
        result = amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2)
        return result
