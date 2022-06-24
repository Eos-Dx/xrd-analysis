import numpy as np

"""
Utility functions
"""

def create_circular_mask(nrows, ncols, center=None, rmin=0, rmax=None, mode="min"):
    """
    Creates a circular mask for a pixel array with shape (nrows,ncols)
    Center can be specified as float tuple (row_center, col_center)
    rmin and rmax can be float
    Mask elements are True when distance from center is between rmin and rmax (inclusive)
    
    If dimension is even, center is in between pixels, location is a half-index.
    If dimension is odd, take middle of center pixel, location is a whole index.

    Compare to opencv circle:
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
    """
    if center is None: # use the middle of the image
        center = (nrows/2-0.5, ncols/2-0.5)
    if rmax is None: # use the smallest distance between the center and image walls
        if mode == "min":
            rmax = np.min([center[0]+0.5, center[1]+0.5, ncols-1-center[0]+0.5, nrows-1-center[1]+0.5])
        else:
            rmax = np.max([center[0]+0.5, center[1]+0.5, ncols-1-center[0]+0.5, nrows-1-center[1]+0.5])

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
