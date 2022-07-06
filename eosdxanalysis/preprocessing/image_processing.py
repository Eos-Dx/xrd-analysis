import numpy as np
import cv2
import skimage
from scipy.ndimage import map_coordinates

"""
Functions for image processing
"""

def centerize(img, center, method="standard"):
    """
    Enlarges image with zero-intensity pixels so that specified image center
    is the exact center of the enlarged image


    Standard method adapted from Muscle X
    center is tuple of (row, column) coordinates (can be float)

    Note that "standard" method will pad image with zeros
    to prep for rotation

    Point method minimally pads image with zeros.
    """

    nrows, ncols = img.shape

    if center == ((nrows-1)/2, (ncols-1)/2):
        """
        Image is already centered
        """
        return img, center

    if method == "standard":
        """
        Perform standard re-centering, includes rasterization
        """
        dim = int(2.8*np.max([ncols, nrows]))
        new_img = np.zeros((dim,dim))
        new_img[0:nrows,0:ncols] = img

        #Translate image to appropriate position
        transx = int(((dim/2) - center[1]))
        transy = int(((dim/2) - center[0]))
        m_trans = np.float32([[1,0,transx],[0,1,transy]])
        rows,cols = new_img.shape
        translated_Img = cv2.warpAffine(new_img, m_trans ,(cols,rows))

        return translated_Img, (int(dim / 2), int(dim / 2))

    if method == "point":
        """
        Perform point-based re-centering
        Assume the image pixels are points on a grid
        (0,0) maps to point at center of first pixel
        """
        # If dimensions are even, round center to nearest half-integer
        mod_center = [center[0], center[1]]
        if nrows % 2 == 0:
            mod_center[0] = np.round(center[0]*2)/2
        else:
            mod_center[0] = np.round(center[0])
        if ncols % 2 == 0:
            mod_center[1] = np.round(center[1]*2)/2
        else:
            mod_center[1] = np.round(center[1])

        # Calculate displacement from shape center to specified center
        trans_row = (nrows-1)/2 - mod_center[0]
        trans_col = (ncols-1)/2 - mod_center[1]

        # Calculate the radii of the new image
        row_radius = (nrows-1)/2 + np.abs(trans_row)
        col_radius = (ncols-1)/2 + np.abs(trans_col)

        # Calculate the shape of the new image
        new_rows = int(np.around(row_radius*2))+1
        new_cols = int(np.around(col_radius*2))+1

        # Calculate the position of the old image in the new image
        row_start = int(np.around(mod_center[1] - row_radius + trans_row))
        row_end = row_start + nrows
        col_start = int(np.around(mod_center[1] - col_radius + trans_col))
        col_end = col_start + ncols

        # Copy the old image to the new image
        new_img = np.zeros((new_rows,new_cols))
        new_img[row_start:row_end,col_start:col_end] = img

        return new_img, ((new_rows-1)/2, (new_cols-1)/2)

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

def convert_to_cv2_img(orig_img):
    """
    Converts raw intensity data to 16-bit grayscale image
    """
    img_float = orig_img/np.max(orig_img)
    uint_img = np.array(img_float * (2**16-1), dtype = np.uint16)
    cv2_img = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)

    return cv2_img

def rotate_image(img, center=None, angle=None, method="standard"):
    """
    Rotates an image about its center
    Note that cv2 uses x,y for images so we take the transpose
    """
    if center == None:
        nrows, ncols = img.shape
        center = ((nrows-1)/2,(ncols-1)/2)

    # Convert to cv2 center where upper-left corner of image is origin
    # which is (-0.5, -0.5) in our point-centered coordinate system
    cv2_center = (center[0] + 0.0, center[1] + 0.0)

    rot_mat = cv2.getRotationMatrix2D(cv2_center,angle,1.0)
    # Convert to cv2 image
    # uint_img = img.astype(np.uint16)
    # img = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)[:,:,0]
    img = convert_to_cv2_img(img)[:,:,0]
    img_shape_T = (img.shape[1],img.shape[0])

    if method == "standard":
        rotated_img = cv2.warpAffine(convert_to_cv2_img(img.T), rot_mat, img_shape_T)[:,:,0]
        rotated_img = rotated_img.T
        return rotated_img
    elif method == "nearest":
        flag = cv2.INTER_NEAREST
        rotated_img = cv2.warpAffine(img.T, rot_mat, img.shape, flag)
        return rotated_img
    elif method == "elastic":
        # Use skimage.transform.rotate
        rotated_img = skimage.transform.rotate(img, angle, resize=True, order=0)
        return rotated_img
    else:
        raise NotImplemented("Choose an existing rotation method.")


def crop_image(img,height,width,center=None):
    """
    Crops an image to a given height and width about a center
    # TODO: check for overruns
    """
    if center == None:
        center_row = height//2
        center_col = width//2
    else:
        center_row = int(center[0])
        center_col = int(center[1])

    row_start = center_row-height//2
    row_end = center_row+height//2
    col_start = center_col-width//2
    col_end = center_col+width//2

    try:
        cropped_img = img[row_start:row_end,col_start:col_end] 
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
    Note that notation here is different from `warp_polar`:
    Here we use the standard Cartesian grid.

    Adapted from:
    https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/3
    """
    if output_shape is None:
        output_shape = img.shape
    output = np.zeros(output_shape, dtype=img.dtype)
    if origin is None:
        origin = np.array(output.shape)/2 - 0.5
    out_h, out_w = output.shape
    # Create a grid of x and y coordinates
    ys, xs = np.mgrid[:out_h, :out_w] - origin[:,None,None]
    # Create a grid of r and theta coordinates
    rs = np.sqrt(ys**2+xs**2)
    thetas = np.arccos(xs/rs)
    thetas[ys<0] = np.pi*2 - thetas[ys<0]
    thetas *= (img.shape[1]-1)/(np.pi*2)
    if rmax is not None:
        rs *= (img.shape[0]-1)/(rmax)
    map_coordinates(img, (rs, thetas), order=order, output=output)
    # Rescale image
    output *= np.sum(img)/np.sum(output)
    return output
