"""
Functions to help with calibration code
"""

import numpy as np
import pandas as pd

from sklearn.base import OneToOneFeatureMixin
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

from eosdxanalysis.calibration.sample_manual_calibration import TissueGaussianFit

from eosdxanalysis.calibration.utils import radial_profile_unit_conversion


DEFAULT_Q_RANGE_COLUMN_NAME = "q_range"
DEFAULT_SAMPLE_DISTANCE_COLUMN_NAME = "calculated_distance"
DEFAULT_RECALCULATED_DISTANCE_COLUMN_NAME = "recalculated_distance"
DEFAULT_RADIAL_PROFILE_DATA_COLUMN_NAME = "radial_profile_data"
DEFAULT_TISSUE_CATEGORY_COLUMN_NAME = "tissue_category"
MM2M = 1e-3

class MomentumTransferUnitsConversion(
        OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Adapted from scikit-learn transforms
    Sample units conversion.
    Converts from real-space pixel units to momentum transfer (q) units.
    """
    def __init__(self, *, copy=True,
            wavelength_nm=None, pixel_size=None,
            q_range_column_name=DEFAULT_Q_RANGE_COLUMN_NAME,
            sample_distance_column_name=DEFAULT_SAMPLE_DISTANCE_COLUMN_NAME,
            radial_profile_data_column_name=DEFAULT_RADIAL_PROFILE_DATA_COLUMN_NAME):
        """
        Parameters
        ----------
        copy : bool
            Creates copy of array if True (default = False).

        sample_distance_m : array_like
            Array of sample distances in meters. Must be same shape as X.
        """
        self.copy = copy
        self.wavelength_nm = wavelength_nm
        self.pixel_size = pixel_size
        self.q_range_column_name = q_range_column_name
        self.sample_distance_column_name = sample_distance_column_name
        self.radial_profile_data_column_name = radial_profile_data_column_name

    def fit(self, X, y=None, sample_weight=None):
        """Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        return self

    def transform(self, X, copy=True):
        """Transforms radial data from intensity versus pixel position
        to intensity versus q value.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        wavelength_nm = self.wavelength_nm
        pixel_size = self.pixel_size
        q_range_column_name = self.q_range_column_name
        sample_distance_column_name = self.sample_distance_column_name
        radial_profile_data_column_name = self.radial_profile_data_column_name

        if type(X) != type(pd.DataFrame()):
            raise ValueError("Input ``X`` must be a dataframe.")

        if copy is True:
            X = X.copy()

        # Extract sample distance and radial profile data
        sample_distance_mm = X[sample_distance_column_name].values * 1e3
        profile_data = X[radial_profile_data_column_name].values

        for idx in X.index:
            profile = X.loc[idx, "radial_profile_data"]
            radial_count = profile.size
            q_range = radial_profile_unit_conversion(
                    radial_count=radial_count,
                    sample_distance_mm=sample_distance_mm,
                    wavelength_nm=wavelength_nm,
                    pixel_size=pixel_size,
                    radial_units="q_per_nm").T

        # Add q-ranges into dataset
        X[q_range_column_name] = q_range.tolist()

        return X


class GaussianFittingMomentumTransferUnitsConversion(
        OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Adapted from scikit-learn transforms
    Sample units conversion.
    Converts from real-space pixel units to momentum transfer (q) units.

    Uses peak fitting.
    """
    def __init__(self, *, copy=True,
            wavelength_nm=None, pixel_size=None,
            q_range_column_name=DEFAULT_Q_RANGE_COLUMN_NAME,
            sample_distance_column_name=DEFAULT_SAMPLE_DISTANCE_COLUMN_NAME,
            recalculated_distance_column_name=DEFAULT_RECALCULATED_DISTANCE_COLUMN_NAME,
            tissue_category_column_name=DEFAULT_TISSUE_CATEGORY_COLUMN_NAME,
            radial_profile_data_column_name=DEFAULT_RADIAL_PROFILE_DATA_COLUMN_NAME):
        """
        Parameters
        ----------
        copy : bool
            Creates copy of array if True (default = False).

        sample_distance_m : array_like
            Array of sample distances in meters. Must be same shape as X.
        """
        self.copy = copy
        self.wavelength_nm = wavelength_nm
        self.pixel_size = pixel_size
        self.q_range_column_name = q_range_column_name
        self.sample_distance_column_name = sample_distance_column_name
        self.recalculated_distance_column_name = recalculated_distance_column_name
        self.tissue_category_column_name = tissue_category_column_name
        self.radial_profile_data_column_name = radial_profile_data_column_name

    def fit(self, X, y=None, sample_weight=None):
        """Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        return self

    def transform(self, X, copy=True):
        """Transforms radial data from intensity versus pixel position
        to intensity versus q value.

        Parameters
        ----------
        X : {array-like, sparse matrix of shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, default=None
            Copy the input X or not.
        Returns
        -------
        X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Transformed array.
        """
        wavelength_nm = self.wavelength_nm
        pixel_size = self.pixel_size
        q_range_column_name = self.q_range_column_name
        sample_distance_column_name = self.sample_distance_column_name
        recalculated_distance_column_name = self.recalculated_distance_column_name
        tissue_category_column_name = self.tissue_category_column_name
        radial_profile_data_column_name = self.radial_profile_data_column_name

        if type(X) != type(pd.DataFrame()):
            raise ValueError("Input ``X`` must be a dataframe.")

        if copy is True:
            X = X.copy()

        fitter = TissueGaussianFit(
                wavelength_nm=wavelength_nm,
                pixel_size=pixel_size)

        X[q_range_column_name] = np.nan
        X[q_range_column_name] = X[q_range_column_name].astype(object)

        for idx in X.index:
            sample_distance = X.loc[idx, sample_distance_column_name]
            profile_data = X.loc[idx, radial_profile_data_column_name]
            tissue_category = X.loc[idx, tissue_category_column_name]

            # Strip nans
            profile_data = profile_data[~np.isnan(profile_data)]

            q_range, recalculated_sample_distance_mm = \
                    fitter.calculate_q_range_from_peak_fitting(
                        profile_data,
                        calculated_distance=sample_distance,
                        tissue_category=tissue_category)
            recalculated_distance = recalculated_sample_distance_mm * MM2M

            # Add q-ranges into dataframe
            X.at[idx, q_range_column_name] = q_range.ravel().tolist()
            # Add recalculated sample distance into dataframe
            X.at[idx, recalculated_distance_column_name] = recalculated_distance

        return X

class DiffractionUnitsConversion(object):
    """
    Class to handle converting units for X-ray diffraction measurements

    Notes
    -----

    Machine diagram:
               /|   tan(2*theta) =
              / |       bragg_peak_pixel_location/sample_distance
             /  |
        ____/_\_|   angle = 2*theta
       ^   ^    ^
      (1) (2)  (3)

    (1) Source
    (2) Sample
    (3) Detector

    """

    def __init__(
            self, source_wavelength_nm=None, pixel_length=None,
            sample_distance_mm=None, n=1):
        """
        Initialize the DiffractionUnitConversion class
        """
        # Store machine parameters
        self.source_wavelength_nm = source_wavelength_nm
        self.pixel_length = pixel_length
        self.sample_distance_mm = sample_distance_mm

        # Set defaults
        self.n = n

        return super().__init__()

    def theta_from_molecular_spacing(self, molecular_spacing, n=1):
        """
        Calculate two*theta from molecular spacing (crystal plane spacing)

        Parameters
        ----------
        molecular_spacing : float
            The crystal plane spacing in meters.

        n : int
            Order of the Bragg peak

        Notes
        -----

        Two*theta is the angle from the beam axis to the location of a Bragg
        peak on a diffraction pattern.
        """
        # Set the source wavelength
        source_wavelength_nm = self.source_wavelength_nm

        # Check if source wavelength was provided
        if not source_wavelength_nm:
            raise ValueError("Source wavelength is required!")

        # Calculate the Bragg angle for the n-th order Bragg peak
        theta = np.arcsin(n*source_wavelength_nm/(2*molecular_spacing))

        # Store theta and two*theta
        self.theta = theta

        return theta

    def two_theta_from_molecular_spacing(self, molecular_spacing, n=1):
        """
        Calculate two*theta from molecular spacing (crystal plane spacing)

        Parameters
        ----------
        molecular_spacing : float
            The crystal plane spacing in meters.

        n : int
            Order of the Bragg peak

        Notes
        -----

        Two*theta is the angle from the beam axis to the location of a Bragg
        peak on a diffraction pattern.
        """
        # Get or calculate theta
        theta = self.theta_from_molecular_spacing(molecular_spacing, n)

        # Calculate two*theta
        two_theta = 2*theta

        # Store two*theta
        self.two_theta = two_theta

        return two_theta

    def q_spacing_from_theta(self, theta):
        """
        Calculate q-spacing from molecular spacing
        """
        # Set the source wavelength
        source_wavelength_nm = self.source_wavelength_nm

        # Check if source wavelength was provided
        if not source_wavelength_nm:
            raise ValueError("Source wavelength is required!")

        q_spacing = 4*np.pi/source_wavelength_nm * np.sin(theta)
        return q_spacing

    def q_spacing_from_molecular_spacing(self, molecular_spacing, n=1):
        """
        Calculate q-spacing from two-theta
        """
        # Set the source wavelength
        source_wavelength_nm = self.source_wavelength_nm

        # Check if source wavelength was provided
        if not source_wavelength_nm:
            raise ValueError("Source wavelength is required!")

        # Get or calculate theta
        try:
            theta = self.theta
        except AttributeError:
            theta = theta_from_molecular_spacing(self, molecular_spacing, n)

        # Calculate q-spacing from theta
        q_spacing = q_spacing_from_theta(theta)

        return q_spacing

    def bragg_peak_location_from_molecular_spacing(
            self, molecular_spacing):
        """
        Function to calculate the distance from the beam axis to a bragg peak
        in units of pixel lengths

        Parameters
        ----------
        molecular_spacing : ndarray

        Returns
        -------
        pixel_location : float
            Distance from diffraction pattern center to bragg peak
            corresponding the ``molecular_spacing``

        Notes
        -----
        Since our sampling rate corresponds to pixels, pixel units directly
        correspond to array indices, i.e.  distance of 1 pixel = distance of 1
        array element

        """
        # Get machine parameters
        source_wavelength_nm = self.source_wavelength_nm
        sample_distance_mm = self.sample_distance_mm

        # Check if required machine parameters were provided
        if not all([self.source_wavelength_nm, self.sample_distance_mm]):
            raise ValueError(
                    "Missing source wavelength and sample-to-detector distance!"
                    " You must initialize the class with machine parameters for"
                    " this method!")

        # Calculate the distance in meters
        two_theta = self.two_theta_from_molecular_spacing(molecular_spacing)
        bragg_peak_location = sample_distance_mm * np.tan(two_theta)

        return bragg_peak_location

    def bragg_peak_pixel_location_from_molecular_spacing(
            self, molecular_spacing):
        """
        Function to calculate the distance from the beam axis to a bragg peak
        in units of pixel lengths

        Parameters
        ----------
        molecular_spacing : ndarray

        Returns
        -------
        pixel_location : float
            Distance from diffraction pattern center to bragg peak
            corresponding the ``molecular_spacing``

        Notes
        -----
        Since our sampling rate corresponds to pixels, pixel units directly
        correspond to array indices, i.e.  distance of 1 pixel = distance of 1
        array element

        """
        # Get pixel length machine parameter
        pixel_length = self.pixel_length

        # Check if required pixel length machine parameter was provided
        if not pixel_length:
            raise ValueError(
                    "You must initialize the class with ``pixel_length`` for"
                    " this method!")

        # Calculate the distance in meters
        bragg_peak_location = self.bragg_peak_location_from_molecular_spacing(
                molecular_spacing)

        # Convert Bragg peak location to pixel length units
        bragg_peak_pixel_location = bragg_peak_location / pixel_length

        return bragg_peak_pixel_location

