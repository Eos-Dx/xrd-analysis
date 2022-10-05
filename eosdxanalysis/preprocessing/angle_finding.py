"""
Code to find the rotation angle for image registration.
"""

import numpy as np

from skimage.transform import rotate
from scipy.optimize import minimize


def find_rotation_angle(image, r, width, height):
    """
    Finds the rotation angle of an image based on rectangles in the 9 A
    region of interest. The result is the angle that the image was rotated
    by to maximize the intensity in the rectangle rois.

    Parameters
    ----------

    image : ndarray

    r : int
        center of the rectangle from the origin

    width : int
        horizontal size of the rectangle

    height : int
        vertical size of the rectangle

    Returns
    -------

    theta : float (degrees)

    """
    theta_init = 0
    result = minimize(
            rotated_rectangle_roi_sum,
            theta_init,
            args = (-image, r, width, height),
            )
    theta = result.x[0]

    return theta

def rotated_rectangle_roi_sum(angle, image, r, width, height):
    """
    Rotates an image and calculates the total intensity in the
    rectangle rois.

    Parameters
    ----------

    angle : float (degrees)
        The angle to rotate the image by

    image : ndarray

    r : int
        center of the rectangle from the origin

    width : int
        horizontal size of the rectangle

    height : int
        vertical size of the rectangle

    Returns
    -------

    roi_sum : float

    """
    rotated_image = rotate(image, angle[0])
    roi_sum = rectangle_roi_sum(rotated_image, r, width, height)
    return roi_sum

def rectangle_roi_sum(image, r, width, height):
    """
    Calculates the total intensity in the rectangle rois.

    Parameters
    ----------

    image : ndarray

    r : int
        center of the rectangle from the origin

    width : int
        horizontal size of the rectangle

    height : int
        vertical size of the rectangle

    Returns
    -------

    roi_sum : float

    """
    shape = image.shape
    center = (shape[0]/2-0.5, shape[1]/2-0.5)

    # Define the right rectangular roi
    right_row_start = int(center[0] - height/2)
    right_row_end = int(center[0] + height/2)
    right_col_start = int((r + center[1]) - width/2)
    right_col_end = int((r + center[1]) + width/2)

    right_rectangle_rows = slice(right_row_start, right_row_end)
    right_rectangle_cols = slice(right_col_start, right_col_end)

    right_rectangle_sum = image[right_rectangle_rows, right_rectangle_cols].sum()

    # Define the left rectangular roi
    left_row_start = int(center[0] - height/2)
    left_row_end = int(center[0] + height/2)
    left_col_start = int((center[1] - r) - width/2)
    left_col_end = int((center[1] - r) + width/2)

    left_rectangle_rows = slice(left_row_start, left_row_end)
    left_rectangle_cols = slice(left_col_start, left_col_end)

    left_rectangle_sum = image[left_rectangle_rows, left_rectangle_cols].sum()

    rectangle_sum = right_rectangle_sum + left_rectangle_sum

    return rectangle_sum
