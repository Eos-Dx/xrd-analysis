"""
Functions to help with calibration code
"""

import numpy as np


PIXEL_SIZE = 55e-6 # Pixel width in meters (it is 55 um)
WAVELENGTH = 1.5418E-10 # Wavelength in meters (1.5418 Angstroms)
WAVELEN_ANGSTROMS = WAVELENGTH * 1e10 # Wavelength in Angstroms


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
            self, source_wavelength=None, pixel_length=None,
            sample_distance=None, n=1):
        """
        Initialize the DiffractionUnitConversion class
        """
        # Store machine parameters
        self.source_wavelength = source_wavelength
        self.pixel_length = pixel_length
        self.sample_distance = sample_distance

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
        source_wavelength = self.source_wavelength

        # Check if source wavelength was provided
        if not source_wavelength:
            raise ValueError("Source wavelength is required!")

        # Calculate the Bragg angle for the n-th order Bragg peak
        theta = np.arcsin(n*source_wavelength/(2*molecular_spacing))

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
        source_wavelength = self.source_wavelength

        # Check if source wavelength was provided
        if not source_wavelength:
            raise ValueError("Source wavelength is required!")

        q_spacing = 4*np.pi/source_wavelength * np.sin(theta)
        return q_spacing

    def q_spacing_from_molecular_spacing(self, molecular_spacing, n=1):
        """
        Calculate q-spacing from two-theta
        """
        # Set the source wavelength
        source_wavelength = self.source_wavelength

        # Check if source wavelength was provided
        if not source_wavelength:
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
        source_wavelength = self.source_wavelength
        sample_distance = self.sample_distance

        # Check if required machine parameters were provided
        if not all([self.source_wavelength, self.sample_distance]):
            raise ValueError(
                    "Missing source wavelength and sample-to-detector distance!"
                    " You must initialize the class with machine parameters for"
                    " this method!")

        # Calculate the distance in meters
        two_theta = self.two_theta_from_molecular_spacing(molecular_spacing)
        bragg_peak_location = sample_distance * np.tan(two_theta)

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

def radial_profile_unit_conversion(radial_count=None,
        sample_distance=None,
        wavelength_nm=None,
        radial_units="q_per_nm"):
    """
    Convert radial profile from pixel lengths to:
    - q_per_nm
    - two_theta
    - um

    Parameters
    ----------

    radial_count : int
        Number of radial points.

    sample_distance : float
        Meters.

    radial_units : str
        Choice of "q_per_nm" (default), "two_theta", or "um".
    """
    radial_range_m = np.arange(radial_count) * PIXEL_SIZE
    radial_range_m = radial_range_m.reshape(-1,1)
    # sample_distance = np.array(sample_distance).reshape(-1,1)
    sample_distance_mm = sample_distance * 1e3
    radial_range_mm = radial_range_m * 1e3


    if radial_units == "q_per_nm":
        q_range_per_nm = q_conversion(
            real_position_mm=radial_range_mm,
            sample_distance_mm=sample_distance_mm,
            wavelength_nm=wavelength_nm)
        return q_range_per_nm

    if radial_units == "two_theta":
        two_theta_range = two_theta_conversion(
                sample_distance_mm, radial_range_mm)
        return two_theta_range

    if radial_units == "um":
        return radial_range_m * 1e6

def two_theta_conversion(real_position_mm=None, sample_distance_mm=None):
    """
    Convert real position to two*theta
    """
    two_theta = np.arctan2(real_position_mm, sample_distance_mm)
    return two_theta

def q_conversion(
        real_position_mm=None, sample_distance_mm=None, wavelength_nm=None):
    """
    Convert real position to q
    """
    two_theta = two_theta_conversion(
            real_position_mm=real_position_mm,
            sample_distance_mm=sample_distance_mm)
    theta = two_theta / 2
    q = 4*np.pi*np.sin(theta) / wavelength_nm
    return q

def real_position_from_two_theta(two_theta=None, sample_distance_m=None):
    """
    two_theta : float
        radians

    sample_distance_m
    """
    position_m = sample_distance_m * np.tan(two_theta)
    return position_m

def real_position_from_q(q_per_nm=None, sample_distance_m=None, wavelength_nm=None):
    """
    """
    theta = np.arcsin(q_per_nm * wavelength_nm / 4 / np.pi)
    two_theta = 2*theta
    position_m = real_position_from_two_theta(
            two_theta=two_theta, sample_distance_m=sample_distance_m)
    return position_m
