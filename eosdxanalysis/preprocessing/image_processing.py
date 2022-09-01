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

def quadrant_fold(image):
    # Create copies of flipped image
    flip_horizontal = image[:,::-1]
    flip_vertical = image[::-1,:]
    flip_both = flip_vertical[:,::-1]

    # Take average of all image copies
    quad_folded = (image + flip_horizontal + flip_vertical + flip_both)/4

    return quad_folded

def unwarp_polar(img, origin=None, output_shape=None, rmax=None, order=1):
    """
    Convert image from polar to cartesian.
    Reverse of `skimage.transform.warp_polar`

    Cartesian origin is upper-left corner.
    - Y is height (rows)
    - X is width (columns)

    Adapted from:
    https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/3
    """
    if output_shape is None:
        output_shape = img.shape
    output = np.zeros(output_shape, dtype=img.dtype)
    out_h, out_w = output.shape

    if origin is None:
        origin = np.array(output.shape)/2 - 0.5

    # Create a grid of x and y coordinates with origin
    YY, XX = np.ogrid[:out_h, :out_w]
    YY = YY - origin[0]
    XX = XX - origin[1]

    # Create a grid of r and theta coordinates
    RR = np.sqrt(YY**2+XX**2)
    TT = np.arccos(XX/RR)

    # Add 2*pi if YY < 0 to make theta range from 0 to 2*pi
    TT[YY.ravel() < 0, :] = np.pi*2 - TT[YY.ravel() < 0, :]
    # Rescale theta to go from 0 to 2*pi
    TT *= (img.shape[1]-1)/(2*np.pi)

    # Crop according to rmax
    if rmax is not None:
        RR *= (img.shape[0]-1)/(rmax)

    # Convert image from polar to cartesian
    map_coordinates(img, (RR, TT), order=order, output=output)

    return output
