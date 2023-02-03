"""
Utility functions
"""

import numpy as np

from skimage.transform import rotate

from eosdxanalysis.preprocessing.image_processing import quadrant_fold

def create_circular_mask(nrows, ncols, center=None, rmin=0, rmax=None, mode="min"):
    """
    Creates a circular mask for a pixel array with shape (nrows,ncols)
    Center can be specified as float tuple (row_center, col_center)
    rmin and rmax can be float
    Mask elements are True when distance from center is between rmin and rmax (inclusive)
    
    If dimension is even, center is in between pixels, location is a half-index.
    If dimension is odd, take middle of center pixel, location is a whole index.

    `rmax` for `min` mode is the distance from the center to the nearest edge
    with a 0.5 distance buffer.

    Compare to opencv circle:
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
    """
    if center is None: # use the middle of the image
        center = (nrows/2-0.5, ncols/2-0.5)
    if rmax is None: # use the smallest distance between the center and image walls
        if mode == "min":
            rmax = np.min([center[0], center[1], ncols-1-center[0], nrows-1-center[1]])
        else:
            rmax = np.max([center[0], center[1], ncols-1-center[0], nrows-1-center[1]])

    # Create a grid of coordinates
    Rows, Cols = np.ogrid[:nrows, :ncols]
    
    # Calculate distance from center for each coordinate
    dist_from_center = np.sqrt((Rows - center[0])**2 + (Cols-center[1])**2)

    # Calculate mask elements
    # Mask elements are True if they lie within rmin and rmax
    mask = (dist_from_center <= rmax) & (dist_from_center >= rmin)
    return mask

def gen_rotation_line(center=[0,0],angle=0.0,radius=100.0):
    """
    Function to generate line for plotting purposes using x,y coordinates.
    Parameters:
    - center: (x0,y0) where origin is top-left (north-west or nw)
         horizontal is x0, vertical down is y0
    - angle: (theta) [degrees] for standard lower-left origin
    - radius: (radius)
    Returns:
    - line_points: ([x1,x2],[y1,y2]) a tuple of the line endpoints
    """
    # Calculate unit vectors
    ux = np.cos(angle*np.pi/180.0)
    uy = np.sin(angle*np.pi/180.0)
    # Calculator line endpoints
    x1 = radius*ux
    y1 = radius*uy
    x2 = -x1
    y2 = -y1
    # Recalculate
    y0, x0 = center
    x1 = x0 + x1
    x2 = x0 + x2
    y1 = y0 - y1
    y2 = y0 - y2

    return np.array([[x1,x2],[y1,y2]])

def get_angle(feature1, feature2):
    """
    Calculate the angle between two features,
    angle is for standard origin lower-left
    """
    vertical_coord =  feature1[0] - feature2[0]
    horizontal_coord = feature2[1] - feature1[1]

    # Use arctan2 to get angle
    angle = np.arctan2(vertical_coord,horizontal_coord)*180.0/np.pi
    return angle

def count_intervals(num_array, count=0):
    """
    Given an array of integers, return the number of intervals
    Example:
    - num_array = [0,1,2,6,7,10,12,16,17,18]
    - output: 5
    """

    # If array has 0 elements, the count is 0
    if len(num_array) == 0:
        return 0

    # If array has 1 element, the count is 1
    if len(num_array) == 1:
        count = count + 1
        return count

    # Now we handle the general case,
    # where array has at least two elements

    # Get first element of the array
    num = num_array[0]

    # Now we check if the next element is consecutive
    if num_array[1] != num+1:
        # The next element is not consecutive
        # Add 1 to the count
        count = count + 1
        # Feed shortened array with new count
        # back to the count_intervals function
        return count_intervals(num_array[1:], count)
    else:
        # The next element is consecutive
        # so we do not add to the count
        return count_intervals(num_array[1:], count)

def find_maxima(img, mask_center, rmin=0, rmax=None):
    # Create create circular mask for beam region of interest (roi)
    roi_mask = create_circular_mask(img.shape[0], img.shape[1],
            center=mask_center, rmin=rmin, rmax=rmax)

    img_roi = np.copy(img)
    img_roi[~roi_mask]=0

    # Find pixels with maximum intensity within beam of interest (roi)
    # Take tranpose so each rows is coordinates for each point
    max_indices = np.array(np.where(img_roi == np.max(img_roi))).T

    return max_indices

def zerocross1d(x, y, getIndices=False):
    """
      Find the zero crossing points in 1d data.

      Find the zero crossing events in a discrete data set.
      Linear interpolation is used to determine the actual
      locations of the zero crossing between two data points
      showing a change in sign. Data point which are zero
      are counted in as zero crossings if a sign change occurs
      across them. Note that the first and last data point will
      not be considered whether or not they are zero.

      Parameters
      ----------
      x, y : arrays
          Ordinate and abscissa data values.
      getIndices : boolean, optional
          If True, also the indicies of the points preceding
          the zero crossing event will be returned. Defeualt is
          False.

      Returns
      -------
      xvals : array
          The locations of the zero crossing events determined
          by linear interpolation on the data.
      indices : array, optional
          The indices of the points preceding the zero crossing
          events. Only returned if `getIndices` is set True.

      Notes
      -----
      Copied with modifications from PyAstronomy library. MIT License.
    """

    # Check sorting of x-values
    if np.any((x[1:] - x[0:-1]) <= 0.0):
        raise(ValueError("The x-values must be sorted in ascending order!"))

    # Indices of points *before* zero-crossing
    indi = np.where(y[1:]*y[0:-1] < 0.0)[0]

    # Find the zero crossing by linear interpolation
    dx = x[indi+1] - x[indi]
    dy = y[indi+1] - y[indi]
    zc = -y[indi] * (dx/dy) + x[indi]

    # What about the points, which are actually zero
    zi = np.where(y == 0.0)[0]
    # Do nothing about the first and last point should they
    # be zero
    zi = zi[np.where((zi > 0) & (zi < x.size-1))]
    # Select those point, where zero is crossed (sign change
    # across the point)
    zi = zi[np.where(y[zi-1]*y[zi+1] < 0.0)]

    # Concatenate indices
    zzindi = np.concatenate((indi, zi))
    # Concatenate zc and locations corresponding to zi
    zz = np.concatenate((zc, x[zi]))

    # Sort by x-value
    sind = np.argsort(zz)
    zz, zzindi = zz[sind], zzindi[sind]

    if not getIndices:
        return zz
    else:
        return zz, zzindi


def radial_integration(image, center=None, output_shape=(360,128)):
    """
    Performs radial integration

    Parameters
    ----------

    image : ndarray

    output_shape : 2-tuple int

    Returns
    -------

    profile_1d : (n,1)-array float
    """
    polar_image = warp_polar(
            image, center=center, radius=output_shape[1]-0.5,
            output_shape=output_shape, preserve_range=True)
    profile_1d = np.mean(polar_image, axis=1)
    return profile_1d

def polar_meshgrid(
        output_shape=(256,256), r_count=10, theta_count=10,
        rmin=20, rmax=110, quadrant_fold=True, half_step_rotation=True):
    """
    Creates a polar meshgrid with unique label per cell.

    Parameters
    ----------
    output_shape : (int, int)
        Shape of polar meshgrid

    r_count : int
        Number of annuli

    theta_count : int
        Number of sectors

    rmin : int
        Start radius

    rmax : int
        End radius

    quadrant_fold : bool
        Return a quadrant-folded image if ``True``. Default is ``False``.

    half_step_rotation : bool
        Rotate meshgrid by half sector angle size.

    Returns
    -------

    meshgrid : ndarray
        Returns ndarray of shape ``output_shape`` with unique label per cell.

    Notes
    -----
    Cell label values start at 1 to distinguish unused image portions
    """
    output = np.zeros(output_shape)
    shape = output.shape

    # Set up radius and theta ranges
    r_range = np.linspace(rmin, rmax, r_count+1)
    theta_range = np.linspace(-np.pi, np.pi, theta_count+1)

    # Generate a meshgrid the same size as the image
    x_end = shape[1]/2 - 0.5
    x_start = -x_end
    y_end = x_start
    y_start = x_end
    YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
    TT = np.arctan2(YY, XX)
    RR = np.sqrt(XX**2 + YY**2)

    # Create image mask
    image_mask = create_circular_mask(
            shape[0], shape[1], rmin=rmin, rmax=rmax)

    # Initialize cell values
    cell_value = 1

    for jdx in range(theta_count):
        # Calculate sector start and end angles based on symmetric sector angle
        theta_min = theta_range[jdx]
        theta_max = theta_range[jdx+1]

        for idx in range(r_count):
            # Set sector rmin rmax
            sector_rmin = r_range[idx]
            sector_rmax = r_range[idx+1]

            # Create a mask for the annulus
            annulus_mask = create_circular_mask(
                    shape[0], shape[1], rmin=sector_rmin, rmax=sector_rmax)


            # Create a mask for the cell, combining annulus mask
            # with theta bounds
            cell_mask = annulus_mask & \
                (TT >= theta_min) & (TT < theta_max)

            # Set the cell value
            output[cell_mask] = cell_value

            # Increment the cell value
            cell_value += 1

        # Apply image mask
        output[~image_mask] = 0

    # Rotate image
    output = np.rot90(np.rot90(output))

    # Rotate image by half step
    if half_step_rotation:
        # Get rotation angle in degrees
        rotation_angle = 360/16/2
        # Rotate image
        output = rotate(output, -rotation_angle)

    # Get quadrant folded image
    if quadrant_fold:
        # Get middle index
        mdx = int(shape[0]/2)
        # Use upper right as template
        upper_right = output[:mdx,mdx:]
        # Set upper left as right mirror image
        upper_left = np.fliplr(upper_right)
        # Set lower left as upper left mirror image
        lower_left = np.flipud(upper_left)
        # Set lower right as upper right mirror image
        lower_right = np.flipud(upper_right)

        # Set image quadrants
        # Set upper left
        output[:mdx,:mdx] = upper_left
        # Set lower left
        output[mdx:,:mdx] = lower_left
        # Set lower right
        output[mdx:, mdx:] = lower_right

    return output
