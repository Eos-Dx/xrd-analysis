import numpy as np

"""
Calculate features using preprocessing functions
"""

def feature_9a_ratio(image, start_radius=25, roi_l=18, roi_w=4):
    """
    Calulate the ratio of vertical region of interest (roi) intensities
    over horizontal window roi intensities in the 9A region

    Define a rectangular region of interest as in the ascii graphic
     _ <- roi_w
    | | -
    | | |
    | | | <- roi_l
    | | |
    |_| -

    The window shown is defined for the right 9.8A peak (eye).
    It's height is roi_l.
    It's width is roi_w.
    The window is the same for the left 9.8A peak (eye).
    The window is rotated +/-90 degrees about the image center
    to calculate an intensity ratio.
    See ascii diagram below:
               ___
              |___|

         _             _
        | |           | |
        |_|           |_|

               ___
              |___|
    """
    # Calculate center of image
    shape = image.shape
    row_isodd = shape[0]%2
    col_isodd = shape[1]%2
    row_center = shape[0]/2+row_isodd
    col_center = shape[1]/2+col_isodd

    # Calculate center of roi's
    roi_right_center = (int(row_center),
                        int(col_center + start_radius + roi_w/2))
    roi_left_center = (int(row_center),
                        int(col_center - start_radius - roi_w/2))
    roi_top_center = (int(row_center - start_radius - roi_w/2),
                        int(col_center))
    roi_bottom_center = (int(row_center + start_radius + roi_w/2),
                        int(col_center))

    centers = (
            roi_right_center,
            roi_left_center,
            roi_top_center,
            roi_bottom_center,
            )
    
    # Calculate slice indices
    roi_right_rows = (int(roi_right_center[0]-roi_l/2),
                    int(roi_right_center[0]+roi_l/2))
    roi_right_cols = (int(roi_right_center[1]-roi_w/2),
                    int(roi_right_center[1]+roi_w/2))
    
    roi_left_rows = (int(roi_left_center[0]-roi_l/2),
                    int(roi_left_center[0]+roi_l/2))
    roi_left_cols = (int(roi_left_center[1]-roi_w/2),
                    int(roi_left_center[1]+roi_w/2))

    roi_top_rows = (int(roi_top_center[0]-roi_w/2),
                    int(roi_top_center[0]+roi_w/2))
    roi_top_cols = (int(roi_top_center[1]-roi_l/2),
                    int(roi_top_center[1]+roi_l/2))
    
    roi_bottom_rows = (int(roi_bottom_center[0]-roi_w/2),
                        int(roi_bottom_center[0]+roi_w/2))
    roi_bottom_cols = (int(roi_bottom_center[1]-roi_l/2),
                        int(roi_bottom_center[1]+roi_l/2))
    
    # Calculate anchors (upper-left corner of each roi)
    anchors = [
        (roi_right_rows[0], roi_right_cols[0],roi_l,roi_w),
        (roi_left_rows[0], roi_left_cols[0],roi_l,roi_w),
        (roi_top_rows[0], roi_top_cols[0],roi_w,roi_l),
        (roi_bottom_rows[0], roi_bottom_cols[0],roi_w,roi_l),
              ]
    
    # Calculate windows
    roi_right = image[roi_right_rows[0]:roi_right_rows[1],
                          roi_right_cols[0]:roi_right_cols[1]]
    roi_left = image[roi_left_rows[0]:roi_left_rows[1],
                          roi_left_cols[0]:roi_left_cols[1]]
    roi_top = image[roi_top_rows[0]:roi_top_rows[1],
                          roi_top_cols[0]:roi_top_cols[1]]
    roi_bottom = image[roi_bottom_rows[0]:roi_bottom_rows[1],
                          roi_bottom_cols[0]:roi_bottom_cols[1]]
    
    rois = (roi_right, roi_left, roi_top, roi_bottom)

    roi_intensity_horizontal = np.sum(roi_right)+np.sum(roi_left)
    roi_intensity_vertical = np.sum(roi_top)+np.sum(roi_bottom)
    
    intensity_ratio = roi_intensity_horizontal/roi_intensity_vertical

    return intensity_ratio, rois, centers, anchors


def feature_5a_peak_location(image, row_min=6, row_max=78, roi_w=5):
    """
    Calculate the location of the 5A peak in the given image.

    Define a rectangular region of interest as in the ascii graphic
     _ <- roi_w
    | | -
    | | |
    | | | <- row_max - row_min
    | | |
    |_| -

    The window is defined from the top of the image.
    It's height is row_max - row_min.
    It's width is roi_w.
    """
    # Calculate center of image
    shape = image.shape
    row_isodd = shape[0]%2
    col_isodd = shape[1]%2
    row_center = shape[0]/2+row_isodd
    col_center = shape[1]/2+col_isodd


    # Calculate the slice indices
    roi_rows = (row_min, row_max)
    roi_cols = (int(col_center-roi_w/2),int(col_center+roi_w/2))

    # Calculate the center of the roi
    roi_center = (roi_rows[1] - roi_rows[0],
                    roi_cols[1] - roi_cols[0])

    # Calculate anchor (upper-left corner of roi)
    anchor = (row_min, int(col_center-roi_w/2))
    
    # Calculate the roi
    roi = image[roi_rows[0]:roi_rows[1],
                          roi_cols[0]:roi_cols[1]]
    
    # Average across columns
    roi_avg = np.mean(roi,axis=1)
    
    # Calculate peak location using maximum of centroid
    roi_peak_location_list = np.where(roi_avg == np.max(roi_avg))
    # The row number of the peak in the roi
    roi_peak_location = np.mean(roi_peak_location_list)
    # The absolute row number of the peak in the image
    peak_location = roi_peak_location + row_min
    
    return peak_location, roi, roi_center, anchor
