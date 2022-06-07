import numpy as np
from scipy.ndimage import generic_filter
from preprocessing.utils import create_circular_mask

"""
Methods for denoising the diffraction pattern
"""

def stray_filter(img, factor=10.0, filter_mask=np.array([]), rmin=2, rmax=3):
    """
    Custom noise filter

    Given an nxn image (odd) return mean of surrounding area
    if the center pixel intensity is stray (much greater than neighborhood intensity)
    """
    # Calculate 1D index of center value
    center_index = img.size//2
    # Store center value
    center_value = img.flatten()[center_index]
    
    if filter_mask.size == 0:
        # Get neighbors between 2 and 3 pixels away
        filter_mask = create_circular_mask(img.shape[0],img.shape[1],rmin=rmin,rmax=rmax)

    neighbors = np.copy(img)
    
    # Take mean of neighbors within the mask region
    neighbors_mean = np.mean(neighbors[filter_mask])
    
    # If center is greater than the mean multiplied by a factor,
    # it's stray, return neighborhood mean
    if center_value > neighbors_mean*factor:
        return neighbors_mean
    # Not a stray, return original value
    else:
        return center_value

def filter_strays(img, block_size=7, factor=10.0, rmin=2, rmax=3):
    """
    Remove stray high-intensity pixels
    Block size must be odd

    Alternative: can specify percentile instead of mean factor
    """
    if block_size % 2 != 1.0:
        raise ValueError("Block size must be odd!")
    cleaned_img = np.copy(img)

    filter_mask = create_circular_mask(block_size,block_size,rmin=rmin,rmax=rmax)

    # Apply the stray_detector as a custom filter
    generic_filter(img, stray_filter, size=block_size,
                                 output=cleaned_img, mode='reflect',
                                 extra_keywords={
                                     "filter_mask": filter_mask,
                                     })
    
    return cleaned_img
