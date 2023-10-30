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
from scipy.signal import find_peaks

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

    def best_fit(self, xdata, ydata, p0=None, bounds=None, tissue_category=None):
        """
        xdata = detector space
        ydata = sample values
        """
        if tissue_category in ["control-like", "tumor-like"]:
            popt, pcov = curve_fit(
                    self.gaussian_function, xdata, ydata,
                    p0=p0, bounds=bounds)
        elif tissue_category == "mixed":
            popt, pcov = curve_fit(
                    self.double_gaussian_function, xdata, ydata,
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

        if tissue_category in ["control-like", "tumor-like"]:
            # Compute initial guess for where the peak is
            mu0_est_mm = real_position_from_q(
                    q_per_nm=q_peak_per_nm,
                    sample_distance_mm=sample_distance_mm,
                    wavelength_nm=wavelength_nm)
            # Convert to pixel space
            mu0_est = mu0_est_mm * MM2M / pixel_size

            # Compute initial guess for peak amplitude
            # Convert real position to pixel location
            peak_guess_idx = int(mu0_est_mm * MM2M / pixel_size)
            amplitude0_est = radial_profile[peak_guess_idx]

            # Compute initial guess for sigma by calculating full-width at half max
            # using amplitude0 as the guess for max
            sub_max0_est = 2/3 * amplitude0_est
            # Compute half width half_max0 locations right and left
            swsm_right0_est = np.where(
                    radial_profile[peak_guess_idx:] < sub_max0_est)[0][0]
            swsm_left0_est_offset = np.where(
                    radial_profile[:peak_guess_idx] < sub_max0_est)[0][-1]
            swsm_left0_est = radial_profile[:peak_guess_idx].size \
                    - swsm_left0_est_offset
            swsm0_est = np.min([swsm_right0_est, swsm_left0_est])
            sigma0_est = 3/2 * swsm0_est

            # Now use find_peaks to find maximum near this m0
            if tissue_category == "control-like":
                # Now get better estimates using find_peaks
                subset_left_idx = int(mu0_est - sigma0_est)
                subset_right_idx = int(mu0_est + sigma0_est)
                radial_profile_subset = radial_profile[
                        subset_left_idx:subset_right_idx]
                width = 0
            elif tissue_category == "tumor-like":
                # Now get better estimates using find_peaks
                subset_left_idx = int(mu0_est - sigma0_est * 2/3)
                subset_right_idx = int(mu0_est + sigma0_est * 2/3)
                radial_profile_subset = radial_profile[
                        subset_left_idx:subset_right_idx]
                width = 2
            peak_indices_approx, properties = find_peaks(radial_profile_subset)

            if peak_indices_approx.size == 1:
                peak_idx = peak_indices_approx[-1] + subset_left_idx

                mu0 = peak_idx
                amplitude0 = radial_profile[peak_idx]
                # Compute refined guess for sigma by calculating full-width at half max
                # using amplitude0 as the guess for max
                half_max0 = 2/3 * amplitude0
                # Compute half width half_max0 locations right and left
                swsm_right0 = np.where(radial_profile[peak_idx:] < half_max0)[0][0]
                swsm_left0_offset = np.where(
                        radial_profile[:peak_idx] < half_max0)[0][-1]
                swsm_left0 = radial_profile[:peak_idx].size - swsm_left0_offset
                swsm0 = np.min([swsm_right0, swsm_left0])
                sigma0 = 3/2 * swsm0

            else:
                mu0 = mu0_est
                sigma0 = sigma0_est
                amplitude0 = amplitude0_est

        elif tissue_category == "mixed":
            # Mixed tissue type has two peaks
            # Compute initial guess for where the peaks are
            mu0_fat_est_mm = real_position_from_q(
                    q_per_nm=q_peak_per_nm[0],
                    sample_distance_mm=sample_distance_mm,
                    wavelength_nm=wavelength_nm)
            mu0_h2o_est_mm = real_position_from_q(
                    q_per_nm=q_peak_per_nm[1],
                    sample_distance_mm=sample_distance_mm,
                    wavelength_nm=wavelength_nm)
            # Convert to pixel space
            mu0_fat_est = mu0_fat_est_mm * MM2M / pixel_size
            mu0_h2o_est = mu0_h2o_est_mm * MM2M / pixel_size

            # Compute initial guess for peak amplitude
            # Convert real position to pixel location
            peak_fat_guess_idx = int(mu0_fat_est_mm * MM2M / pixel_size)
            amplitude0_fat_est = radial_profile[peak_fat_guess_idx]
            peak_h2o_guess_idx = int(mu0_h2o_est_mm * MM2M / pixel_size)
            amplitude0_h2o_est = radial_profile[peak_h2o_guess_idx]

            # Compute initial guess for sigma by calculating full-width at half max
            # using the amplitude0 guesses for max
            try:
                # Fat
                sub_max0_fat_est = 2/3 * amplitude0_fat_est
                # Compute half width half_max0 locations right and left
                swsm_right0_fat_est = np.where(
                        radial_profile[peak_fat_guess_idx:] < sub_max0_fat_est)[0][0]
                swsm_left0_fat_est_offset = np.where(
                        radial_profile[:peak_fat_guess_idx] < sub_max0_fat_est)[0][-1]
                swsm_left0_fat_est = radial_profile[:peak_fat_guess_idx].size \
                        - swsm_left0_fat_est_offset
                swsm0_fat_est = np.min([swsm_right0_fat_est, swsm_left0_fat_est])
                sigma0_fat_est = 3/2 * swsm0_fat_est
            except:
                sigma0_fat_est = 10
            try:
                # Water
                sub_max0_h2o_est = 2/3 * amplitude0_h2o_est
                # Compute half width half_max0 locations right and left
                swsm_right0_h2o_est = np.where(
                        radial_profile[peak_h2o_guess_idx:] < sub_max0_h2o_est)[0][0]
                swsm_left0_h2o_est_offset = np.where(
                        radial_profile[:peak_h2o_guess_idx] < sub_max0_h2o_est)[0][-1]
                swsm_left0_h2o_est = radial_profile[:peak_h2o_guess_idx].size \
                        - swsm_left0_h2o_est_offset
                swsm0_h2o_est = np.min([swsm_right0_h2o_est, swsm_left0_h2o_est])
                sigma0_h2o_est = 3/2 * swsm0_h2o_est
            except:
                sigma0_h2o_est = 10

            try:
                # Now use find_peaks to find maximum near the m0 values
                # Now get better estimates using find_peaks
                # Fat
                subset_fat_left_idx = int(mu0_fat_est - sigma0_fat_est)
                subset_fat_right_idx = int(mu0_fat_est + sigma0_fat_est)
                radial_profile_fat_subset = radial_profile[
                        subset_fat_left_idx:subset_fat_right_idx]
                width_fat = 0
                peak_fat_indices_approx, properties_fat = find_peaks(radial_profile_fat_subset)
            except:
                # Fat
                mu0_fat = mu0_fat_est
                sigma0_fat = sigma0_fat_est
                amplitude0_fat = amplitude0_fat_est
            try:
                # Water
                subset_h2o_left_idx = int(mu0_h2o_est - sigma0_h2o_est)
                subset_h2o_right_idx = int(mu0_h2o_est + sigma0_h2o_est)
                radial_profile_h2o_subset = radial_profile[
                        subset_h2o_left_idx:subset_h2o_right_idx]
                width_h2o = 0
                peak_h2o_indices_approx, properties_h2o = find_peaks(radial_profile_h2o_subset)
            except:
                # Water
                mu0_h2o = mu0_h2o_est
                sigma0_h2o = sigma0_h2o_est
                amplitude0_h2o = amplitude0_h2o_est

            # Fat
            if peak_fat_indices_approx.size == 1:
                peak_fat_idx = peak_fat_indices_approx[-1] + subset_fat_left_idx

                mu0_fat = peak_fat_idx
                amplitude0_fat = radial_profile[peak_fat_idx]
                # Compute refined guess for sigma by calculating full-width at half max
                # using amplitude0 as the guess for max
                half_max0_fat = 2/3 * amplitude0_fat
                # Compute half width half_max0 locations right and left
                swsm_right0_fat = np.where(radial_profile[peak_fat_idx:] < half_max0_fat)[0][0]
                swsm_left0_fat_offset = np.where(
                        radial_profile[:peak_fat_idx] < half_max0_fat)[0][-1]
                swsm_left0_fat = radial_profile[:peak_fat_idx].size - swsm_left0_fat_offset
                swsm0_fat = np.min([swsm_right0_fat, swsm_left0_fat])
                sigma0_fat = 3/2 * swsm0_fat

            else:
                mu0_fat = mu0_fat_est
                sigma0_fat = sigma0_fat_est
                amplitude0_fat = amplitude0_fat_est
            # Water
            if peak_h2o_indices_approx.size == 1:
                peak_h2o_idx = peak_h2o_indices_approx[-1] + subset_h2o_left_idx

                mu0_h2o = peak_h2o_idx
                amplitude0_h2o = radial_profile[peak_h2o_idx]
                # Compute refined guess for sigma by calculating full-width at half max
                # using amplitude0 as the guess for max
                half_max0_h2o = 2/3 * amplitude0_h2o
                # Compute half width half_max0 locations right and left
                swsm_right0_h2o = np.where(radial_profile[peak_h2o_idx:] < half_max0_h2o)[0][0]
                swsm_left0_h2o_offset = np.where(
                        radial_profile[:peak_h2o_idx] < half_max0_h2o)[0][-1]
                swsm_left0_h2o = radial_profile[:peak_h2o_idx].size - swsm_left0_h2o_offset
                swsm0_h2o = np.min([swsm_right0_h2o, swsm_left0_h2o])
                sigma0_h2o = 3/2 * swsm0_h2o

            else:
                mu0_h2o = mu0_h2o_est
                sigma0_h2o = sigma0_h2o_est
                amplitude0_h2o = amplitude0_h2o_est

        # Collect parameters
        p0 = np.array(
                    [mu0_fat,
                    sigma0_fat,
                    amplitude0_fat,
                    mu0_h2o,
                    sigma0_h2o,
                    amplitude0_h2o])

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

        bound_lower = 0.5 * p0
        bound_upper = 2.0 * p0
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

    @classmethod
    def double_gaussian_function(
            self,
            x,
            mu0,
            sigma0,
            amplitude0,
            mu1,
            sigma1,
            amplitude1):
        """
        Gaussian function to use for peak finding.
        """
        result = self.gaussian_function(x, mu0, sigma0, amplitude0) + \
                self.gaussian_function(x, mu1, sigma1, amplitude1)
        return result

    def calculate_sample_distance(
            self,
            radial_profile,
            calculated_distance=None,
            tissue_category=None):
        """Calculate the sample distance based on peak fitting
        """
        wavelength_nm = self.wavelength_nm
        pixel_size = self.pixel_size

        p0, bounds = self.initial_parameters_and_bounds(
            radial_profile,
            calculated_distance=calculated_distance,
            tissue_category=tissue_category)

        sample_distance_mm = calculated_distance * M2MM
        q_peak_per_nm = peaks_dict.get(tissue_category)


        # TODO: Check for bad fit

        if tissue_category in ["control-like", "tumor-like"]:
            # Run curve fitting
            # Narrow the input range
            mu0 = p0[0]
            sigma0 = p0[1]
            if tissue_category == "control-like":
                x_start = int(mu0 - sigma0)
                x_end = int(mu0 + sigma0)
                xdata = np.arange(x_start, x_end)
            elif tissue_category == "tumor-like":
                x_start = int(mu0 - sigma0/2)
                x_end = int(mu0 + sigma0/2)
                xdata = np.arange(x_start, x_end)

            ydata = radial_profile[xdata]
            popt, pcov = self.best_fit(
                    xdata,
                    ydata, p0=p0,
                    bounds=bounds,
                    tissue_category=tissue_category)

            # Collect parameters
            mu_fit, sigma_fit, amplitude_fit = popt

            # Convert the Gaussian peak location from detector space to real space
            real_position_mm = mu_fit * pixel_size * M2MM

            # Calculate sample distance from Gaussian peak location
            sample_distance_mm = sample_distance_from_q(
                    q_per_nm=q_peak_per_nm,
                    wavelength_nm=wavelength_nm,
                    real_position_mm=real_position_mm)

        elif tissue_category == "mixed":
            # Run curve fitting
            # Narrow the input range
            mu0_fat = p0[0]
            sigma0_fat = p0[1]
            mu0_h2o = p0[3]
            sigma0_h2o = p0[4]
            x_start = int(mu0_fat - sigma0_fat/2)
            x_end = int(mu0_h2o + sigma0_h2o/2)
            xdata = np.arange(x_start, x_end)

            ydata = radial_profile[xdata]
            try:
                popt, pcov = self.best_fit(
                        xdata,
                        ydata, p0=p0,
                        bounds=bounds,
                        tissue_category=tissue_category)

                # Collect parameters
                mu_fat_fit, sigma_fat_fit, amplitude_fat_fit, \
                        mu_h2o_fit, sigma_h2o_fit, amplitude_h2o_fit = popt

                # Convert the Gaussian peak location from detector space to real space
                real_position_fat_mm = mu_fat_fit * pixel_size * M2MM
                real_position_h2o_mm = mu_h2o_fit * pixel_size * M2MM

                # Calculate sample distance from Gaussian peak location
                sample_distance_fat_mm = sample_distance_from_q(
                        q_per_nm=q_peak_per_nm[0],
                        wavelength_nm=wavelength_nm,
                        real_position_mm=real_position_fat_mm)
                sample_distance_h2o_mm = sample_distance_from_q(
                        q_per_nm=q_peak_per_nm[1],
                        wavelength_nm=wavelength_nm,
                        real_position_mm=real_position_h2o_mm)
                sample_distance_mm = np.mean([
                        sample_distance_fat_mm,
                        sample_distance_h2o_mm])

            except RuntimeError:
                # Use provided sample distance
                sample_distance_mm = calculated_distance * M2MM

        return sample_distance_mm

    def calculate_q_range_from_peak_fitting(
            self,
            radial_profile,
            calculated_distance=None,
            tissue_category=None):
        """
        Perform peak fitting.
        """
        # Get machine parameters
        wavelength_nm = self.wavelength_nm
        pixel_size = self.pixel_size

        # Validate tissue category
        if tissue_category not in peaks_dict.keys():
            raise ValueError(f"{tissue_category} not a valid tissue category.")

        sample_distance_mm = self.calculate_sample_distance(
                radial_profile,
                calculated_distance,
                tissue_category,
                )

        # Calculate the q-range
        q_range = radial_profile_unit_conversion(
                radial_count=radial_profile.size,
                sample_distance_mm=sample_distance_mm,
                wavelength_nm=wavelength_nm,
                pixel_size=pixel_size,
                radial_units="q_per_nm")

        return q_range, sample_distance_mm
