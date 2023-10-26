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

from eosdxanalysis.calibration.units_conversion import real_position_from_q
from eosdxanalysis.calibration.units_conversion import sample_distance_from_q
from eosdxanalysis.calibration.units_conversion import radial_profile_unit_conversion


M2MM = 1e3
MM2M = 1e-3

# Set peak values [nm^-1]
q_fat_per_nm = 14.1
q_h2o_per_nm = 20

# Tissue categories
peaks_dict = {
        "control-like": q_fat_per_nm,
        "tumor-like": q_h2o_per_nm,
        "mixed": [q_fat_per_nm, q_h2o_per_nm],
        }


class TissueGaussianFit(object):
    """
    Gaussian fit
    """

    def __init__(self, wavelength_nm=None, pixel_size=None):
        """
        Set up Gaussian fitting class object
        """
        # Store machine parameters
        self.wavelength_nm = wavelength_nm
        self.pixel_size = pixel_size
        return super().__init__()

    def best_fit(self, xdata, ydata, p0=None, bounds=None):
        """
        xdata = detector space
        ydata = sample values
        """
        popt, pcov = curve_fit(
                self.gaussian_function, xdata, ydata,
                p0=p0, bounds=bounds)
        return popt, pcov

    def initial_parameters(
            self,
            radial_profile,
            calculated_distance=None,
            tissue_category=None):

        """
        Compute initial guess for parameters
        """
        # Get machine parameters
        wavelength_nm = self.wavelength_nm
        pixel_size = self.pixel_size

        # Validate tissue category
        if tissue_category not in peaks_dict.keys():
            raise ValueError(f"{tissue_category} not a valid tissue category.")

        # Collect sample parameters
        sample_distance_mm = calculated_distance * M2MM
        q_peak_per_nm = peaks_dict.get(tissue_category)

        # Compute initial guess for where the peak is
        mu0_mm = real_position_from_q(
                q_per_nm=q_peak_per_nm,
                sample_distance_mm=sample_distance_mm,
                wavelength_nm=wavelength_nm)
        # Convert to pixel space
        mu0 = mu0_mm * MM2M / pixel_size

        # Compute initial guess for peak amplitude
        # Convert real position to pixel location
        peak_guess_idx = int(mu0_mm * MM2M / pixel_size)
        amplitude0 = radial_profile[peak_guess_idx]

        # Compute initial guess for sigma by calculating full-width at half max
        # using amplitude0 as the guess for max
        half_max0 = amplitude0/2
        # Compute half width half_max0 locations right and left
        hwhm_right0 = np.where(radial_profile[peak_guess_idx:] < half_max0)[0][0]
        hwhm_left0 = np.where(radial_profile[:peak_guess_idx] > half_max0)[0][0]
        fwhm0 = hwhm_right0 - hwhm_left0
        sigma0 = fwhm0

        # Collect parameters
        p0 = np.array([mu0, sigma0, amplitude0])

        return p0

    def parameter_bounds(
            self,
            radial_profile,
            p0,
            calculated_distance=None,
            tissue_category=None):
        """
        Use the initial guess and some multiplicative factors
        to generate parameter bounds for curve fitting.
        """

        bound_lower = 0.8 * p0
        bound_upper = 1.2 * p0
        bounds = bound_lower, bound_upper

        return bounds

    def initial_parameters_and_bounds(
            self,
            radial_profile,
            calculated_distance=None,
            tissue_category=None):

        p0 = self.initial_parameters(
            radial_profile,
            calculated_distance=calculated_distance,
            tissue_category=tissue_category)
        bounds = self.parameter_bounds(
            radial_profile,
            p0,
            calculated_distance=calculated_distance,
            tissue_category=tissue_category)

        return p0, bounds

    @classmethod
    def gaussian_function(self, x, mu, sigma, amplitude):
        """
        Gaussian function to use for peak finding.
        """
        result = amplitude * np.exp(-0.5 * ((x - mu) / sigma)**2)
        return result

    def calculate_q_range_from_peak_fitting(
            self,
            radial_profile,
            calculated_distance=None,
            tissue_category=None):
        """
        Perform peak fitting.
        """
        # Validate tissue category
        if tissue_category not in peaks_dict.keys():
            raise ValueError(f"{tissue_category} not a valid tissue category.")

        # Collect sample parameters
        q_peak_per_nm = peaks_dict.get(tissue_category)

        # Get machine parameters
        wavelength_nm = self.wavelength_nm
        pixel_size = self.pixel_size

        p0, bounds = self.initial_parameters_and_bounds(
            radial_profile,
            calculated_distance=calculated_distance,
            tissue_category=tissue_category)

        # Run curve fitting
        mu0 = p0[0]
        sigma = p0[1]
        x_start = int(mu0 - sigma/2)
        x_end = int(mu0 + sigma/2)
        xdata = np.arange(x_start, x_end)
        ydata = radial_profile[xdata]
        popt, pcov = self.best_fit(xdata, ydata, p0=p0, bounds=bounds)

        # TODO: Check for bad fit

        # Collect parameters
        mu_fit, sigma_fit, amplitude_fit = popt

        # Convert the Gaussian peak location from detector space to real space
        real_position_mm = mu_fit * pixel_size * M2MM

        # Calculate sample distance from Gaussian peak location
        sample_distance_mm = sample_distance_from_q(
                q_per_nm=q_peak_per_nm,
                wavelength_nm=wavelength_nm,
                real_position_mm=real_position_mm)

        # Calculate the q-range
        q_range = radial_profile_unit_conversion(
                radial_count=radial_profile.size,
                sample_distance_mm=sample_distance_mm,
                wavelength_nm=wavelength_nm,
                pixel_size=pixel_size,
                radial_units="q_per_nm")

        return q_range
