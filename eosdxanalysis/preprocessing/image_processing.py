import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
from skimage.transform import AffineTransform
from skimage.transform import warp

"""
Functions for image processing
"""

def pad_image(img, method="prerotation", padding=None, center=None, nan=False):
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
        diag = int(np.around(np.sqrt(img.shape[0]**2 + img.shape[1]**2)))
        padding = (diag, diag-img.shape[0], diag, diag-img.shape[1])

    # Pad according to padding
    values = np.nan if nan else 0
    new_img = np.full((padding[0]+padding[1]+nrows,
            padding[2]+padding[3]+ncols), np.nan)

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

def enlarge_image(image, center, nan=True):
    # Enlarge the image
    padding_amount = (np.sqrt(2)*np.max(image.shape)).astype(int)
    padding_top = padding_amount
    padding_bottom = padding_amount
    padding_left = padding_amount
    padding_right = padding_amount
    padding = (padding_top, padding_bottom, padding_left, padding_right)
    enlarged_image = pad_image(
            image, padding=padding, nan=nan)
    new_center = (padding_top + center[0], padding_left + center[1])

    return enlarged_image, new_center, padding_amount

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

def concatenate_detector_images(image1, image2, detector_spacing, pixel_size):
    """Code to concatenate images from two detectors.
    Uses skimage.transform.EuclideanTransform to shift second detector
    by arbitrary number of pixel lengths.
    """
    if image1.shape[0] != image2.shape[0]:
        raise ValueError("Input images must have the same number of rows.")

    # Set translation parameters
    trans_x = detector_spacing / pixel_size
    trans_y = 0

    # Construct transform matrix
    tform = AffineTransform(translation=(trans_x, trans_y))

    # Set output shape
    output_shape = (image2.shape[0], image2.shape[1] + int(np.floor(trans_x)))

    # Translate image2
    image2_trans = warp(
            image2,
            tform.inverse,
            cval=np.nan,
            preserve_range=True,
            output_shape=output_shape)

    # Concatenate images
    images_combined = np.hstack([image1, image2_trans])

    return images_combined
