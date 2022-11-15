"""
Feature extraction functions
"""
import numpy as np

from eosdxanalysis.preprocessing.utils import create_circular_mask


class FeatureExtraction(object):
    """
    Class to handle feature extraction
    """

    def __init__(self, image, distance=None):
        """
        Initializes the FeatureExtraction class with image path and
        sample-to-detector distance.

        Parameters
        ----------

        image_path : str
            Path to the image file

        distance : number
            Sample-to-detector distance
        """
        # Store the inputs
        self.image = image
        self.distance = distance

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
        self.image_intensity = image_intensity

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
            self, distance=None, center=None, amin=None, mmax=None):
        """
        Calculate the intensity of an annulus specified by start and end radii
        in Angstrom units.

        Parameters
        ----------

        distance : number
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
        # Reference the stored image
        image = self.image

        # Get the image shape
        shape = image.shape

        rmin = somefunction(amax)
        rmax = somefunction(amin)

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(shape[0], shape[1], center=center,
                rmin=rmin, rmax=rmax)

        # Compute the annulus intensity
        annulus_intensity = np.sum(image[annulus_mask])

        return annulus_intensity
