"""
Code for fitting data to curves
"""
import os
import argparse
import glob
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import peak_widths

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from eosdxanalysis.models.utils import cart2pol
from eosdxanalysis.models.utils import radial_intensity_1d
from eosdxanalysis.models.utils import angular_intensity_1d
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.simulations.utils import feature_pixel_location

BG_NOISE_STD = 20
DEFAULT_5_4A_STD = 20
cmap = "hot"

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

    def __init__(
            self, image, p0_dict=None, p_lower_bounds_dict=None,
            p_upper_bounds_dict=None):
        """
        Initialize `GaussianDecomposition` class.
        """
        self.image = image

        # Calculate image center coordinates
        center = image.shape[0]/2-0.5, image.shape[1]/2-0.5
        self.center = center
        # Initialize parameters
        self.parameter_init(p0_dict, p_lower_bounds_dict, p_upper_bounds_dict)
        return super().__init__()

    def estimate_parameters(self, width=4, position_tol=0.2):
        """
        Estimate Gaussian fit parameters based on provided image.
        Used for providing accurate initial guesses to speed up curve fitting
        algorithm.


        Parameters
        ----------

        image : array_like
            ``image`` can be None, in which case it is pulled from init

        width : int
            averaging width used to produce 1D profiles

        position_tol : float
            Factor used to check if detected peak locations are incorrect.

        Returns
        -------

        p0_dict, p_lower_bounds_dict, p_upper_bounds_dict : tuple
            tuple of initial paremeters guess, as well as lower and upper
            bounds on the parameters.


        Notes
        -----
        - Use horizontal and vertical radial intensity profiles, and their
          differences, to calculate properties of isotropic and anisotropic
          Gaussians.
        - The 9A and 5A peaks are hypothesized to be the sum of isotropic and
          anisotropic Gaussians
            - For now, use only anisotropic functions for these, increasing
              error will correspond to less normal specimens
        - The 5-4A ring and background intensities are hypothesized to be
          isotropic Gaussians
        - That makes a total of 6 Gaussians

        Also stores these parameters in class parameters.

        """
        image = self.image
        RR, TT = self.meshgrid

        # Get 1D radial intensity in positive horizontal direction
        horizontal_intensity_1d = radial_intensity_1d(image, width=width)
        # Get 1D radial intensity in positive vertical direction
        vertical_intensity_1d = radial_intensity_1d(
                image.T[:, ::-1], width=width)

        # Take the difference of the horizontal and vertical 1D intensity
        # profiles to estimate some anisotropic Gaussian properties
        intensity_diff_1d = horizontal_intensity_1d - vertical_intensity_1d

        if "p0_dict" not in self.__dict__:
            # Call all functions to estimate individual parameters
            # Note that some estimator functions have dependcies on the output
            # of other estimator functions

            # Estimate background intensity parameters
            peak_amplitude_bg, peak_std_bg = estimate_background_noise(image)

            # Estimate 9A parameters
            peak_location_radius_9A, peaks_aniso_9A = \
                    self.estimate_peak_location_radius_9A(
                            image, position_tol, horizontal_intensity_1d,
                            vertical_intensity_1d, intensity_diff_1d)
            peak_std_9A = self.estimate_peak_std_9A(
                    image, peak_location_radius_9A, peaks_aniso_9A,
                    horizontal_intensity_1d, vertical_intensity_1d,
                    intensity_diff_1d)
            peak_amplitude_9A = self.estimate_peak_amplitude_9A(
                    image, peak_location_radius_9A, horizontal_intensity_1d,
                    vertical_intensity_1d, intensity_diff_1d)
            arc_angle_9A = self.estimate_arc_angle_9A(
                    image, peak_location_radius_9A, horizontal_intensity_1d,
                    vertical_intensity_1d, intensity_diff_1d)

            # Create a radial gaussian estimate based on the 9A parameters
            radial_gaussian_estimate_9A = radial_gaussian(
                    RR, TT, peak_location_radius_9A, 0,
                    peak_std_9A, peak_amplitude_9A, arc_angle_9A)
            horizontal_intensity_9A = radial_intensity_1d(
                    radial_gaussian_estimate_9A)

            # Estimate 5A parameters
            # NOTE: Here we flip intensity_diff_1d using a minus sign
            peak_location_radius_5A, peaks_aniso_5A = \
                self.estimate_peak_location_radius_5A(
                            image, position_tol, horizontal_intensity_1d,
                            vertical_intensity_1d, -intensity_diff_1d)
            peak_std_5A = self.estimate_peak_std_5A(
                    image, peak_location_radius_5A, peaks_aniso_5A,
                    horizontal_intensity_1d, vertical_intensity_1d,
                    -intensity_diff_1d)
            peak_amplitude_5A = self.estimate_peak_amplitude_5A(
                    image, peak_location_radius_5A, horizontal_intensity_1d,
                    vertical_intensity_1d, -intensity_diff_1d)
            arc_angle_5A = self.estimate_arc_angle_5A(
                    image, peak_location_radius_5A, horizontal_intensity_1d,
                    vertical_intensity_1d, -intensity_diff_1d)

            # Estimate 5-4A parameters
            peak_location_radius_5_4A, peaks_iso_5_4A = \
                self.estimate_peak_location_radius_5_4A(
                        image, position_tol, horizontal_intensity_1d,
                        vertical_intensity_1d, intensity_diff_1d)
            peak_std_5_4A = self.estimate_peak_std_5_4A(
                    image, peak_location_radius_5_4A, peaks_iso_5_4A,
                    horizontal_intensity_1d, vertical_intensity_1d,
                    intensity_diff_1d, horizontal_intensity_9A)
            peak_amplitude_5_4A = self.estimate_peak_amplitude_5_4A(
                    image, peak_location_radius_5_4A, horizontal_intensity_1d,
                    vertical_intensity_1d,
                    intensity_diff_1d)

            # Set up initial parameters dictionary with None for each value
            p0_dict = OrderedDict({
                    # 9A equatorial peaks parameters
                    "peak_location_radius_9A":   peak_location_radius_9A,
                    "peak_std_9A":               peak_std_9A,
                    "peak_amplitude_9A":         peak_amplitude_9A,
                    "arc_angle_9A":              arc_angle_9A,
                    # 5A meridional peaks parameters
                    "peak_location_radius_5A":   peak_location_radius_5A,
                    "peak_std_5A":               peak_std_5A,
                    "peak_amplitude_5A":         peak_amplitude_5A,
                    "arc_angle_5A":              arc_angle_5A,
                    # 5-4A isotropic region parameters
                    "peak_location_radius_5_4A": peak_location_radius_5_4A,
                    "peak_std_5_4A":             peak_std_5_4A,
                    "peak_amplitude_5_4A":       peak_amplitude_5_4A,
                    # Background noise parameters
                    "peak_std_bg":               peak_std_bg,
                    "peak_amplitude_bg":         peak_amplitude_bg,
                })

        # Lower bounds
        if "p_lower_bounds_dict" not in self.__dict__:
            p_min_factor = 1e-2
            p_lower_bounds_dict = OrderedDict()
            for key, value in p0_dict.items():
                # Set lower bounds
                p_lower_bounds_dict[key] = p_min_factor*value

            # Set arc_angle parameters individually
            p_lower_bounds_dict["arc_angle_9A"] = 1e-6
            p_lower_bounds_dict["arc_angle_5A"] = 1e-6

        # Upper bounds
        if "p_upper_bounds_dict" not in self.__dict__:
            p_upper_bounds_dict = OrderedDict()
            p_max_factor = 1e2
            for key, value in p0_dict.items():
                # Set upper bounds
                p_upper_bounds_dict[key] = p_max_factor*value

            # Set arc_angle parameters individually
            p_upper_bounds_dict["arc_angle_9A"] = np.pi
            p_upper_bounds_dict["arc_angle_5A"] = np.pi

        self.p0_dict = p0_dict
        self.p_lower_bounds_dict = p_lower_bounds_dict
        self.p_upper_bounds_dict = p_upper_bounds_dict

        return p0_dict, p_lower_bounds_dict, p_upper_bounds_dict

    def estimate_peak_location_radius_9A(
            self, image, position_tol, horizontal_intensity_1d,
            vertical_intensity_1d, intensity_diff_1d):
        """
        Estimate the 9A peak distance from the center

        :param image: The input image
        :type image: 2D ndarray

        :param:

        """
        # Estimate the 9A peak using the intensity difference
        # which isolates the anisotropic 9A peak
        peaks_aniso, _ = find_peaks(intensity_diff_1d)

        # Ensure that at least one peak was found
        try:
            # Get peak location
            peak_location = peaks_aniso[0]
        except IndexError as err:
            print("No peaks found for 9A parameters estimation.")
            raise err
        # Ensure that peak_9A is close to theoretical value
        peak_location_theory = feature_pixel_location(9.8e-10)
        if abs(peak_location - peak_location_theory) > \
                position_tol * peak_location_theory:
            # Use the theoretical value
            peak_location = peak_location_theory
        return peak_location, peaks_aniso

    def estimate_peak_std_9A(
            self, image, peak_location_radius_9A, peaks_aniso_9A,
            horizontal_intensity_1d, vertical_intensity_1d, intensity_diff_1d):
        """
        Estimate the 9A peak widths (full-width at half maximum)
        """
        width_results_aniso_9A = peak_widths(intensity_diff_1d, peaks_aniso_9A)
        peak_width_aniso_9A = width_results_aniso_9A[0][0]
        # convert FWHM to standard deviation
        peak_std_aniso_9A = peak_width_aniso_9A / (2*np.sqrt(2*np.log(2)))

        return peak_std_aniso_9A

    def estimate_peak_amplitude_9A(
            self, image, peak_location_radius_9A, horizontal_intensity_1d,
            vertical_intensity_1d, intensity_diff_1d):
        """
        Estimate the 9A peak amplitude
        """
        return intensity_diff_1d[int(peak_location_radius_9A)]

    def estimate_arc_angle_9A(self, image, peak_location_radius_9A,
            horizontal_intensity_1d, vertical_intensity_1d, intensity_diff_1d):
        """
        Estimate the 9A maxima arc angle, related to the plateau size
        """
        # Estimate the anisotropic part of the angular intensity
        angular_intensity_9A_1d = angular_intensity_1d(image, radius=peak_location_radius_9A)

        # Estimate the arc_angle for the 9A anisotropic peak
        angular_peaks, peak_properties = find_peaks(
                angular_intensity_9A_1d, plateau_size=(None, None))
        plateau_sizes = peak_properties["plateau_sizes"]

        # If peaks are found, convert from arc length to radians (s = r*theta)
        try:
            arc_length = plateau_sizes[0]
            arc_angle = arc_length*np.pi/180/peak_location_radius_9A
            # If peak value is 0, set angle to near 0 to avoid bounds issues
            if np.isclose(arc_angle, 0):
                arc_angle = 1e-6
        except IndexError as err:
            # No peaks found case, set angle to near pi to avoid bounds issues
            arc_angle = np.pi-1e-6

        return arc_angle

    def estimate_peak_location_radius_5A(self, image, position_tol,
            horizontal_intensity_1d, vertical_intensity_1d, intensity_diff_1d):
        """
        Estimate the 5A peak using the intensity difference
        which isolates the anisotropic 5A peak
        """
        peaks_aniso, _ = find_peaks(intensity_diff_1d)

        # Ensure that at least one peak was found
        try:
            # Get peak location
            peak_location = peaks_aniso[0]
        except IndexError as err:
            print("No peaks found for 5A parameters estimation.")
            raise err
        # Ensure that peak_5A is close to theoretical value
        peak_location_theory = feature_pixel_location(5.1e-10)
        if abs(peak_location - peak_location_theory) > position_tol * peak_location_theory:
            # Use the theoretical value
            peak_location = int(peak_location_theory)
            # raise ValueError("First peak is too far from theoretical value of 5A peak location.")
        return peak_location, peaks_aniso

    def estimate_peak_std_5A(self, image, peak_location_radius_5A,
            peaks_aniso_5A, horizontal_intensity_1d,
            vertical_intensity_1d, intensity_diff_1d):
        """
        Estimate the 5A peak widths (full-width at half maximum)
        """
        width_results_aniso_5A = peak_widths(intensity_diff_1d, peaks_aniso_5A)
        peak_width_aniso_5A = width_results_aniso_5A[0][0]
        peak_std_aniso_5A = peak_width_aniso_5A / (2*np.sqrt(2*np.log(2))) # convert FWHM to standard deviation

        return peak_std_aniso_5A

    def estimate_peak_amplitude_5A(self, image, peak_location_radius_5A,
            horizontal_intensity_1d, vertical_intensity_1d, intensity_diff_1d):
        """
        Estimate the 5A peak amplitude
        """
        return intensity_diff_1d[int(peak_location_radius_5A)]

    def estimate_arc_angle_5A(self, image, peak_location_radius_5A, horizontal_intensity_1d,
            vertical_intensity_1d, intensity_diff_1d):
        """
        Estimate the 5A maxima arc angle, related to the plateau size
        """
        # Estimate the anisotropic part of the angular intensity
        angular_intensity_5A_1d = angular_intensity_1d(image, radius=peak_location_radius_5A)

        # Estimate the arc_angle for the 5A anisotropic peak
        angular_peaks, peak_properties = find_peaks(
                angular_intensity_5A_1d, plateau_size=(None, None))
        plateau_sizes = peak_properties["plateau_sizes"]

        # If peaks are found, convert from arc length to radians (s = r*theta)
        try:
            arc_length = plateau_sizes[0]
            arc_angle = arc_length*np.pi/180/peak_location_radius_5A
            # If peak value is 0, set angle to near 0 to avoid bounds issues
            if np.isclose(arc_angle, 0):
                arc_angle = 1e-6
        except IndexError as err:
            # No peaks found case, set angle to near pi to avoid bounds issues
            arc_angle = np.pi-1e-6

        return arc_angle

    def estimate_peak_location_radius_5_4A(
            self, image, position_tol, horizontal_intensity_1d,
            vertical_intensity_1d, intensity_diff_1d):
        """
        Estimates the 5-4A peak distance from the center

        Parameters
        ----------
        image : 2D ndarray
            The input image

        Returns
        -------
        peak_location : int
        peak_results : tuple
            First output of `scipy.signal.find_peaks`

        """
        # Estimate the 5-4A peak using the horizontal intensity
        peaks_iso, _ = find_peaks(horizontal_intensity_1d)

        # Ensure that at least one peak was found
        try:
            # Get farthest peak location
            peak_location = peaks_iso[-1]
        except IndexError as err:
            print("No peaks found for 5-4A parameters estimation.")
            raise err
        # Ensure that peak_5-4A is close to theoretical value
        peak_location_theory = feature_pixel_location(5.1e-10)
        if abs(peak_location - peak_location_theory) > \
                position_tol * peak_location_theory:
            # Use the theoretical value
            peak_location = peak_location_theory

        return peak_location, peaks_iso

    def estimate_peak_std_5_4A(
            self, image, peak_location_radius_5_4A, peaks_iso_5_4A,
            horizontal_intensity_1d, vertical_intensity_1d, intensity_diff_1d,
            horizontal_intensity_9A):
        """
        Estimate the 5_4A peak widths (full-width at half maximum)
        """
        # Subtract the 1d horizontal intensity profile from the radial gaussian
        # 9A estimate
        horizontal_intensity_5_4A_1d_estimate = \
                horizontal_intensity_1d - horizontal_intensity_9A
        # Look at the horizontal intensity
        widths, width_heights, left_ips, right_ips = peak_widths(
                horizontal_intensity_5_4A_1d_estimate,
                [int(peak_location_radius_5_4A)])

        # Take the last peak
        peak_width_iso_5_4A = widths[-1]
        # convert FWHM to standard deviation
        peak_std_iso_5_4A = peak_width_iso_5_4A / (2*np.sqrt(2*np.log(2)))

        if np.isclose(peak_std_iso_5_4A, 0):
            peak_std_iso_5_4A = DEFAULT_5_4A_STD

        return peak_std_iso_5_4A

    def estimate_peak_amplitude_5_4A(
            self, image, peak_location_radius_5_4A, horizontal_intensity_1d,
            vertical_intensity_1d, intensity_diff_1d):
        return horizontal_intensity_1d[int(peak_location_radius_5_4A)]

    def parameter_init(self, p0_dict=None, p_lower_bounds_dict=None, p_upper_bounds_dict=None):
        """
        P parameters:
        - peak_radius
        - width
        - amplitude
        """
        image = self.image
        # Generate meshgrid
        meshgrid = gen_meshgrid(image.shape)
        self.meshgrid = meshgrid
        # Estimate parameters based on image if image is provided
        if type(image) == np.ndarray:
            return self.estimate_parameters()

    def best_fit(self):
        """
        Use `scipy.optimize.curve_fit` to decompose a keratin diffraction pattern
        into a sum of Gaussians with angular spread.

        Function to optimize: `self.keratin_function`
        """
        image = self.image
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
            RR, TT = gen_meshgrid(image.shape)
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
        popt, pcov = curve_fit(keratin_function, xdata, ydata, p0, bounds=p_bounds)

        # Create popt_dict so we can have keys
        popt_dict = OrderedDict()
        idx = 0
        for key, value in self.p0_dict.items():
            popt_dict[key] = popt[idx]
            idx += 1

        return popt_dict, pcov

    def fit_error(self, image, fit):
        """
        Returns the square error between a function and
        its approximation.
        """
        return np.sqrt(np.sum(np.square(image - fit)))

    def objective(self, p, image, r, theta):
        """
        Generate a keratin diffraction pattern with the given
        parameters list p and return the fit error
        """
        fit = keratin_function((r, theta), *p)
        return fit_error(p, image, fit, r, theta)

    @classmethod
    def parameter_list(self):
        """
        Returns the list of parameter names
        """
        params = [
                "peak_location_radius_9A",
                "peak_std_9A",
                "peak_amplitude_9A",
                "arc_angle_9A",
                "peak_location_radius_5A",
                "peak_std_5A",
                "peak_amplitude_5A",
                "arc_angle_5A",
                "peak_location_radius_5_4A",
                "peak_std_5_4A",
                "peak_amplitude_5_4A",
                "peak_std_bg",
                "peak_amplitude_bg",
                ]
        return params

def radial_gaussian(r, theta, peak_radius, peak_angle,
            peak_std, peak_amplitude, arc_angle):
    """
    Isotropic and radial Gaussian

    Currently uses convolution with a high-resolution arc specified by arc-angle.
    Future will possibly use analytic expression for convolution of a Gaussian with an arc.
    Assumes quadrant-fold symmetry.
    Assumes standard deviation is the same for x and y directions.
    A peak_angle of 0 corresponds to horizontal (equatorial) arcs.
    A peak_angle of pi/2 corresponds to vertical (meridional) arcs.
    Algorithm starts with equatorial peaks, then rotates as neeed.

    Parameters
    ----------

    r : array_like
        Polar point radius to evaluate function at (``r`` must be same shape as ``theta``)

    :param theta: Polar point angle to evaluate function at (``r`` must be same shape as ``theta``)
    :type theta: array_like

    :param peak_radius: Location of the Gaussian peak from the center
    :type peak_radius: float

    :param peak_angle: 0 or np.pi/2 to signify equatorial or meridional peak location (respectively).
        Future will take a continuous input.
    :type peak_angle: float

    :param peak_std: Gaussian standard deviation
    :type peak_std: float

    :param peak_amplitude: Gaussian peak amplitude
    :type peak_amplitude: float

    :param arc_angle: arc angle in radians, `0` is anisotropic, `pi` is fully isotropic
    :type arc_angle: float

    Returns
    -------

    :return: Value of Gaussian function at polar input points ``(r, theta)``
    :rtype: array_like (same shape as ``r`` and ``theta``)

    """
    # Check if arc_angle is between 0 and pi
    if arc_angle < 0 or arc_angle > np.pi:
        raise ValueError("arc_angle must be between 0 and pi")

    # Check if peak_angle is between 0 and pi/2
    if peak_angle < 0 or peak_angle > np.pi/2:
        raise ValueError("peak_angle must be between 0 and pi/2")

    # Check if size of r and theta inputs are equal
    if np.array(r).shape != np.array(theta).shape:
        raise ValueError("Inputs r and theta must have the same shape")

    # Take the modulus of the arc_angle
    arc_angle %= np.pi
    # Force theta to be between -pi and pi
    theta += np.pi
    theta %= 2*np.pi
    theta -= np.pi
    # If the arc angle is 0 or pi, return the isometric Gaussian
    if np.isclose(arc_angle, 0.0):
        # Construct an isometric Gaussian centered at peak_radius
        gau = peak_amplitude*np.exp( -1/2*((r-peak_radius)/peak_std)**2)
        return gau
    # Else we need to convolve with an arc
    else:
        # Create an array of zeros
        gau = np.zeros_like(r)

        # Handle two cases: peak_angle = 0 or pi/2
        if np.isclose(peak_angle, 0):
            # Add the endpoints
            theta1 = -arc_angle/2
            theta2 = arc_angle/2
            # Convert to cartesian coordinates
            x = r*np.cos(theta)
            y = r*np.sin(theta)

            # Create quadrant-folded arc using masks
            # Create masks for right side
            # Create mask for top_right side
            mask_top_right = (theta >= arc_angle/2) & (theta <= np.pi/2)
            gau[mask_top_right] = peak_amplitude*np.exp(
                    -1/(2*peak_std**2)*( (x - peak_radius*np.cos(theta2))**2 + \
                            (y - peak_radius*np.sin(theta2))**2) )[mask_top_right]
            # Create mask for bottom_right side
            mask_bottom_right = (theta <= -arc_angle/2) & (theta >= -np.pi/2)
            gau[mask_bottom_right] = peak_amplitude*np.exp(
                    -1/(2*peak_std**2)*( (x - peak_radius*np.cos(theta1))**2 + \
                            (y - peak_radius*np.sin(theta1))**2) )[mask_bottom_right]
            # Create mask for inside right side
            mask_in_right = (theta > -arc_angle/2) & (theta < arc_angle/2)
            gau[mask_in_right] = peak_amplitude*np.exp( -1/2*((r-peak_radius)/peak_std)**2)[mask_in_right]

            # Create masks for left side
            # Create mask for top_left side
            mask_top_left = (theta <= np.pi-arc_angle/2) & (theta >= np.pi/2)
            gau[mask_top_left] = peak_amplitude*np.exp(
                    -1/(2*peak_std**2)*( (x - peak_radius*np.cos(np.pi-arc_angle/2))**2 + \
                            (y - peak_radius*np.sin(np.pi-arc_angle/2))**2) )[mask_top_left]
            # Create mask for bottom_left side
            mask_bottom_left = (theta >= -np.pi+arc_angle/2) & (theta <= -np.pi/2)
            gau[mask_bottom_left] = peak_amplitude*np.exp(
                    -1/(2*peak_std**2)*( (x - peak_radius*np.cos(-np.pi+arc_angle/2))**2 + \
                            (y - peak_radius*np.sin(-np.pi+arc_angle/2))**2) )[mask_bottom_left]
            # Create mask for inside left side
            mask_in_left = (theta > np.pi-arc_angle/2) | (theta < -np.pi+arc_angle/2)
            gau[mask_in_left] = peak_amplitude*np.exp( -1/2*((r-peak_radius)/peak_std)**2)[mask_in_left]

            # Set the output as the quadrant-folded arc
            output = gau

        elif np.isclose(peak_angle, np.pi/2):
            # Add the endpoints
            theta1 = np.pi/2-arc_angle/2
            theta2 = np.pi/2+arc_angle/2
            # Convert to cartesian coordinates
            x = r*np.cos(theta)
            y = r*np.sin(theta)

            # Create quadrant-folded arc using masks
            # Create masks for top side
            # Create mask for top right side
            mask_right_top = (theta <= peak_angle - arc_angle/2) & (theta >= 0)
            gau[mask_right_top] = peak_amplitude*np.exp(
                    -1/(2*peak_std**2)*( (x - peak_radius*np.cos(theta1))**2 + \
                            (y - peak_radius*np.sin(theta1))**2) )[mask_right_top]
            # Create mask for top left side
            mask_left_top = (theta >= peak_angle + arc_angle/2) & (theta >= 0)
            gau[mask_left_top] = peak_amplitude*np.exp(
                    -1/(2*peak_std**2)*( (x - peak_radius*np.cos(theta2))**2 + \
                            (y - peak_radius*np.sin(theta2))**2) )[mask_left_top]
            # Create mask for top inside
            mask_in_top = (theta > peak_angle - arc_angle/2) & (theta < peak_angle + arc_angle/2)
            gau[mask_in_top] = peak_amplitude*np.exp( -1/2*((r-peak_radius)/peak_std)**2)[mask_in_top]

            # Create masks for bottom side
            # Create mask for bottom right side
            mask_right_bottom = (theta >= -peak_angle+arc_angle/2) & (theta <= 0)
            gau[mask_right_bottom] = peak_amplitude*np.exp(
                    -1/(2*peak_std**2)*( (x - peak_radius*np.cos(-np.pi/2+arc_angle/2))**2 + \
                            (y - peak_radius*np.sin(-np.pi/2+arc_angle/2))**2) )[mask_right_bottom]
            # Create mask for bottom left side
            mask_left_bottom = (theta <= -peak_angle-arc_angle/2) & (theta <= 0)
            gau[mask_left_bottom] = peak_amplitude*np.exp(
                    -1/(2*peak_std**2)*( (x - peak_radius*np.cos(-np.pi/2-arc_angle/2))**2 + \
                            (y - peak_radius*np.sin(-np.pi/2-arc_angle/2))**2) )[mask_left_bottom]
            # Create mask for bottom inside
            mask_in_bottom = (theta > -peak_angle-arc_angle/2) & (theta < -peak_angle+arc_angle/2)
            gau[mask_in_bottom] = peak_amplitude*np.exp( -1/2*((r-peak_radius)/peak_std)**2)[mask_in_bottom]

            # Set the output as the quadrant-folded arc
            output = gau

        else:
            raise ValueError("peak_angle must be 0 or pi/2.")

        return output

def keratin_function(
        polar_point,
        peak_location_radius_9A, peak_std_9A, peak_amplitude_9A, arc_angle_9A,
        peak_location_radius_5A, peak_std_5A, peak_amplitude_5A, arc_angle_5A,
        peak_location_radius_5_4A, peak_std_5_4A, peak_amplitude_5_4A,
        peak_std_bg, peak_amplitude_bg):
    """
    Generate entire kertain diffraction pattern at the points
    (r, theta), with parameters as the arguements to 4 calls
    of radial_gaussian.

    .. Parameters

    :param polar_point: (r, theta) where r and theta are polar coordinates
    :type polar_point: 2-tuple of array_like

    .. Returns

    :returns a: Returns a contiguous flattened array suitable for use
        with ``scipy.optimize.curve_fit``.
    :rtype: array_like

    """
    r, theta = polar_point

    # Set peak position angle parameters
    peak_angle_9A = 0
    peak_angle_5A = np.pi/2
    peak_angle_5_4A = 0 # Don't care
    peak_angle_bg = 0 # Don't care
    # Set peak arc angle parameters for isotropic cases
    arc_angle_5_4A = 0 # Don't care
    arc_angle_bg = 0 # Don't care
    # Set peak location radius for background noise case (Airy disc from pinhole)
    peak_location_radius_bg = 0

    # 9A peaks
    pattern_9A = radial_gaussian(r, theta, peak_location_radius_9A,
            peak_angle_9A, peak_std_9A, peak_amplitude_9A,
            arc_angle_9A)
    # 5A peaks
    pattern_5A = radial_gaussian(r, theta, peak_location_radius_5A,
            peak_angle_5A, peak_std_5A, peak_amplitude_5A,
            arc_angle_5A)
    # 5-4 A anisotropic ring
    pattern_5_4A = radial_gaussian(r, theta, peak_location_radius_5_4A,
            peak_angle_5_4A, peak_std_5_4A, peak_amplitude_5_4A,
            arc_angle_5_4A)
    # Background noise
    pattern_bg = radial_gaussian(r, theta, peak_location_radius_bg,
            peak_angle_bg, peak_std_bg, peak_amplitude_bg,
            arc_angle_bg)
    # Additive model
    pattern = pattern_9A + pattern_5A + pattern_5_4A + pattern_bg

    return pattern.ravel()


def gen_meshgrid(shape):
    """
    Generate a meshgrid
    """
    # Generate a meshgrid the same size as the image
    x_end = shape[1]/2 - 0.5
    x_start = -x_end
    y_end = x_start
    y_start = x_end
    YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
    TT, RR = cart2pol(XX, YY)

    return RR, TT

def gaussian_iso(r, a, std):
    """
    Define the isometric Gaussian function as follows:

    .. math ::
        \mathtt{a}\exp(-1/2(r/\sgiam)^2)

    Parameters
    ----------
    r : float
        The polar radial coordinate to evaluate the function at

    a : float
        The amplitude

    std : float
        The standard deviation of the isotropic Gaussian
    """
    return a*np.exp( -1/2*( (r/std)**2) )


def estimate_background_noise(image):
    """
    Fit an isotropic Gaussian to the image, excluding zero-value points
    There are several options. We use `scipy.optimize.curve_fit`.
    """
    # Set up inputs to curve_fit
    # Generate the meshgrid
    RR, TT = gen_meshgrid(image.shape)
    # Get 1D slice for radial intensity
    r = np.arange(0,int(image.shape[0]/2))

    # Get 1D vertical intensity profile
    vertical_intensity_1d = radial_intensity_1d(image.T[:,::-1])

    # Generate non-zero pixels mask
    mask = vertical_intensity_1d > 0
    r_masked = r[mask].astype(np.float64)
    vertical_intensity_1d_masked = vertical_intensity_1d[mask].astype(np.float64)

    # Prepare independent and dependent inputs to optimizer
    xdata = r_masked.ravel()
    ydata = vertical_intensity_1d_masked.ravel().astype(np.float64)

    # Generate initial guess for peak_amplitude and peak_std
    peak_results, _ = find_peaks(vertical_intensity_1d)
    peak_width_results = peak_widths(vertical_intensity_1d, peak_results)
    try:
        peak_amplitude_guess = r[peak_results[0]]
        peak_std_guess = np.min(peak_width_results[0])
    except IndexError:
        # No peaks found, use median of vertical_intensity_1d as peak amplitude guess
        peak_amplitude_guess = np.median(vertical_intensity_1d)
        peak_std_guess = BG_NOISE_STD

    p0 = [peak_amplitude_guess, peak_std_guess]

    # Run curve fitting procedure
    popt, pcov = curve_fit(gaussian_iso, xdata, ydata, p0)
    peak_amplitude, peak_std = popt

    return peak_amplitude, peak_std


def gaussian_decomposition(input_path, output_path=None):
    """
    Runs batch gaussian decomposition
    """
    # Get full paths to files and created sorted list
    file_path_list = glob.glob(os.path.join(input_path,"*.txt"))
    file_path_list.sort()

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Set output path with a timestamp if not specified
    if not output_path:
        output_dir = "gaussian_decomposition_{}".format(timestamp)
        output_path = os.path.join(input_path, "..", output_dir)

    output_data_dir = "decomp"
    output_data_path = os.path.join(output_path, output_data_dir)
    output_images_dir = "decomp_images"
    output_images_path = os.path.join(output_path, output_images_dir)

    # Create output data and images paths
    os.makedirs(output_data_path, exist_ok=True)
    os.makedirs(output_images_path, exist_ok=True)

    # Get list of parameters
    param_list = GaussianDecomposition.parameter_list()

    # Construct empty list for storing data
    row_list = []

    # Loop over files
    for file_path in file_path_list:
        # Load data
        filename = os.path.basename(file_path)
        image = np.loadtxt(file_path, dtype=np.float64)

        # Now get ``best-fit`` diffraction pattern
        gauss_class = GaussianDecomposition(image)
        popt_dict, pcov = gauss_class.best_fit()
        RR, TT = gen_meshgrid(image.shape)
        popt = np.fromiter(popt_dict.values(), dtype=np.float64)

        decomp_image  = keratin_function((RR, TT), *popt).reshape(*image.shape)

        # Get fit error
        error = gauss_class.fit_error(image, decomp_image)
        error_ratio = error/np.sqrt(np.sum(np.square(image)))

        # Construct dataframe row
        # - filename
        # - optimum fit parameters
        row = [filename, error, error_ratio] + popt.tolist()
        row_list.append(row)

        # Save output
        output_filename = "GD_{}".format(filename)
        output_file_path = os.path.join(output_data_path, output_filename)
        np.savetxt(output_file_path, decomp_image, fmt="%d")

        # Save image preview
        save_image_filename = "GD_{}.png".format(filename)
        save_image_fullpath = os.path.join(output_images_path,
                save_image_filename)
        plt.imsave(save_image_fullpath, decomp_image, cmap=cmap)

    # Create dataframe to store parameters

    # Construct pandas dataframe columns
    columns = ["Filename", "Error", "Error_Ratio"] + param_list

    df = pd.DataFrame(data=row_list, columns=columns)

    # Save dataframe
    csv_filename = "GD_results.csv"
    csv_output_path = os.path.join(output_path, csv_filename)
    df.to_csv(csv_output_path)

if __name__ == '__main__':
    """
    Run curve_fitting on a file or entire folder.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None, required=True,
            help="The path containing raw files to perform fitting on")
    parser.add_argument(
            "--fitting_method", default="gaussian-decomposition", required=False,
            help="The fitting method to perform on the raw files."
            " Options are: `gaussian-decomposition`.")

    # Collect arguments
    args = parser.parse_args()
    input_path = args.input_path
    fitting_method = args.fitting_method

    if fitting_method == "gaussian-decomposition":
        gaussian_decomposition(input_path)

    if fitting_method == "polynomial":
        raise NotImplementedError("Not fully implemeneted yet.")
