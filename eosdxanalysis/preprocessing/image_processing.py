import numpy as np
import skimage
from scipy.ndimage import map_coordinates

"""
Functions for image processing
"""

def pad_image(img, method="prerotation", padding=None):
    """
    Add zeros to image
    Default behavior is if padding is not specified,
    then pad image symmetrically to prevent image loss in case of rotation
    and cropping to original size.

    padding should be a 4-tuple:
    (rows to prepend, rows to append, cols to prepend, cols to append)
    """
    nrows, ncols = img.shape

    if padding == None and method == "prerotation":
        # Calculate padding for worst-case scenario
        # when rotation is 45 degrees
        max_dim = np.max([nrows,ncols])
        pad_side = int(np.around((np.sqrt(2)-1)*(max_dim-1)/2))
        padding = (pad_side,)*4

    # Pad according to padding
    new_img = np.zeros((padding[0]+padding[1]+nrows,
            padding[2]+padding[3]+ncols))

    # Write the old image shifted into the new image
    new_img[padding[0]:nrows+padding[0],
            padding[2]:ncols+padding[2]] = img

    return new_img

def crop_image(img,height,width,center=None):
    """
    Crops an image to a given height and width about a center
    """
    # Check that height and width are not larger than original image size
    if height > img.shape[0] or width > img.shape[1]:
        raise ValueError("Height and width cannot be greater than input image!")

    # Check if image and crop sizes are even
    if any([img.shape[0] % 2 != 0, img.shape[1] % 2 != 0,
            height % 2 != 0, width % 2 != 0]):
        raise NotImplementedError("Only even sized input and output images allowed!")

    # Ensure that specified center is half-integer
    if center:
        center_row, center_col = center
        if any([center_row*2 % 2 != 1, center_col*2 % 2 != 1]):
            raise ValueError("Specified center must be half-integer values!")
    elif center == None:
        center_row = img.shape[0]/2-0.5
        center_col = img.shape[1]/2-0.5
        center = (center_row, center_col)

    row_slice = slice(int(center[0] - height/2 + 0.5),
                        int(center[0] + height/2 + 0.5))
    col_slice = slice(int(center[1] - width/2 + 0.5),
                        int(center[1] + width/2 + 0.5))

    try:
        cropped_img = img[row_slice, col_slice]
    except IndexError as err:
        print("Error cropping image! Check that your inputs make sense.")
        raise err

    return cropped_img

def bright_pixel_count(image, qmin=0, qmax=1, offset=0):
    """
    Return the number of pixels between qmin and qmax
    """
    # Validate inputs
    if qmin > qmax:
        raise ValueError("qmin must be less than qmax")
    if qmin < 0 or qmin > 1 or qmax < 0 or qmax > 1:
        raise ValueError("qmin and qmax must be between 0 and 1 inclusive")

    image_max = image.max()
    image_min = image.min()

    # Rescale image to be between 0 and 1
    image_rescaled = (image - offset) / (image_max - offset)

    if qmin == 0 and qmax == 1:
        qcount = image_rescaled.size
    elif qmin == 0 and qmax != 1:
        qcount = (image_rescaled <= qmax).sum()
    elif qmin != 0 and qmax != 1:
        qcount = ((image_rescaled >= qmin) & (image_rescaled <= qmax)).sum()
    elif qmin != 0 and qmax == 1:
        qcount = (image_rescaled >= qmin).sum()

    return qcount
