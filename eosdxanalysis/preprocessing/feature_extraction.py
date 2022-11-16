"""
Feature extraction functions
"""
import numpy as np

from eosdxanalysis.preprocessing.utils import create_circular_mask

from eosdxanalysis.calibration.utils import DiffractionUnitsConversion


class FeatureExtraction(object):
    """
    Class to handle feature extraction
    """

    def __init__(
            self, image, source_wavelength=None, pixel_length=None,
            sample_to_detector_distance=None):

        """
        Initializes the FeatureExtraction class with image path and
        machine parameters.

        Parameters
        ----------

        image : (n,2)-ndarray
            Image data

        source_wavelength: number

        pixel_length : number

        sample_to_detector_distance : number

        Notes
        -----
        See ``eosdxanalysis.calibration.utils.DiffractionUnitsConversion`` for
        explanation of machine parameters.

        """
        # Store input image
        self.image = image

        # Store machine parameters
        self.source_wavelength = source_wavelength
        self.pixel_length = pixel_length
        self.sample_to_detector_distance = sample_to_detector_distance

        return super().__init__()

    def feature_image_intensity(self):
        """
        Computes the intensity of the image

        Parameters
        ----------

        image : ndarray
            The measurement image data

        Output
        ------

        image_intensity : number
            The total intensity

        """
        # Reference the stored image
        image = self.image

        # Compute the total image intensity
        image_intensity = np.sum(image)

        return image_intensity

    def feature_annulus_intensity(self, center=None, rmin=None, rmax=None):
        """
        Computes the total intensity of an annulus

        Parameters
        ----------

        center : 2-tuple (number)
            Row and column location of the annulus center

        rmin : number
            Minimum annulus radius in pixel units

        rmax : number
            Maximum annulus radius in pixel units

        Output
        ------

        annulus_intensity : number
            The intensity of the annulus
        """
        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(shape[0], shape[1], center=center,
                rmin=rmin, rmax=rmax)

        # Compute the annulus intensity
        annulus_intensity = np.sum(image[annulus_mask])

        return annulus_intensity

    def feature_annulus_intensity_angstrom(
            self, pixel_length=None, center=None, amin=None, amax=None):
        """
        Calculate the intensity of an annulus specified by start and end radii
        in Angstrom units.

        Parameters
        ----------

        distance_to_detector : number
            Distance from sample to detector

        center : 2-tuple (number)
            Row and column location of the annulus center

        amin : number
            Minimum annulus angstrom in pixel units

        amax : number
            Maximum annulus angstrom in pixel units

        Output
        ------

        annulus_intensity : number
            The intensity of the annulus


        Notes
        -----

        ``rmin`` and ``rmax`` are swapped from ``amin`` and ``amax`` since we
        are in reciprocal space.

        """
        # Get machine parameters
        source_wavelength = self.source_wavelength
        sample_to_detector_distance = self.sample_to_detector_distance

        # Check if required machine parameters were provided
        if not all([self.source_wavelength, self.sample_to_detector_distance]):
            raise ValueError("You must initialize the units class with machine"
                    " parameters for this method!")

        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        # Initialize the units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=sample_to_detector_distance)

        # Set rmin to the larger angstrom spacing (reciprocal space)
        rmin = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amax)
        # Set rmax to the smaller angstrom spacing (repciprocal space)
        rmax = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amin)

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(shape[0], shape[1], center=center,
                rmin=rmin, rmax=rmax)

        # Compute the annulus intensity
        annulus_intensity = np.sum(image[annulus_mask])

        return annulus_intensity
