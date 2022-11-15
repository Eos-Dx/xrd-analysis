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

        image : ndarray
            The measurement image data

        Output
        ------

        intensity : number
            The total intensity
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
