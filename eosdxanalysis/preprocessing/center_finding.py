"""
Methods for finding the center of the diffraction pattern.
"""
import numpy as np
from skimage.transform import warp_polar

from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import unwarp_polar

RMIN_BEAM=0
RMAX_BEAM=50



def circular_average(image, center, order=1):
    """
    Returns a circularly averaged version of the input image.

    Inputs:
    - image: 2D array of intensities
    - center: tuple of coordinates for center of 2D array
    """
    radius = np.ceil(np.min([image.shape[0]/2, image.shape[1]/2]))
    # Get the polar warped image
    polar_image = warp_polar(image, center=center, radius=radius,
            output_shape=(360, radius), order=order)
    # Take the mean across all angles (first index)
    polar_image_avg_1d = np.mean(polar_image, axis=0)
    # Copy to 2D
    polar_image_avg_2d = np.tile(polar_image_avg_1d, (360, 1))
    # Convert 1D intensity profile to 2D image
    image_avg_2d = unwarp_polar(polar_image_avg_2d, output_shape = image.shape, order=order)

    return image_avg_2d

def radial_histogram(intensities,center):
    """
    Given a 2D array of intensities, return the radial histogram
    for each intensity value along with the corresponding radii

    Smallest radius is 1 pixel, so 4 pixels is
    the smallest annulus.

    Inputs:
    - intensities: 2D array
    - center: tuple of coordinates for center of 2D array

    """
    ycenter, xcenter = center

    # Create meshgrid
    X,Y = np.meshgrid(np.arange(intensities.shape[1]),np.arange(intensities.shape[0]))
    # Calculate radii
    R = np.sqrt(np.square(X-xcenter)+np.square(Y-ycenter))

    # Get unique radii
    radii = np.unique(R).flatten()

    # Store the total intensity for each annulus
    radial_intensities = np.zeros(radii.shape)


    for idx in np.arange(radii.shape[0]):
        annulus_intensity = np.sum(intensities[(R >= radii[idx]) & (R < radii[idx]+1.0)])
        radial_intensities[idx] = annulus_intensity

    return radii, radial_intensities
