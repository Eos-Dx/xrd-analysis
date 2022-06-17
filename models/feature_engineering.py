import numpy as np

"""
Calculate features using preprocessing functions
"""

def feature_9a_ratio(image, radius=25, roi_h=10, roi_w=10):
    """
    Calulate the ratio of vertical window roi intensities over
    horizontal window roi intensities in the 9A region
    """
    # Calculate center of image
    shape = image.shape
    row_isodd = shape[0]%2
    col_isodd = shape[1]%2
    row_center = shape[0]/2+row_isodd
    col_center = shape[1]/2+col_isodd
    
    # Calculate center of roi's
    eye_right_center = (row_center + row_isodd, col_center + col_isodd + radius)
    eye_left_center = (row_center - row_isodd, col_center - col_isodd - radius)
    eye_north_center = (row_center + row_isodd - radius, col_center + col_isodd)
    eye_south_center = (row_center - row_isodd + radius, col_center + col_isodd)
    
    centers = [
        eye_right_center,
        eye_left_center,
        eye_north_center,
        eye_south_center,
    ]
    
    # Calculate slice indices
    eye_right_roi_rows = (int(eye_right_center[0]-roi_h/2),int(eye_right_center[0]+roi_h/2))
    eye_right_roi_cols = (int(eye_right_center[1]-roi_w/2),int(eye_right_center[1]+roi_w/2))
    
    eye_left_roi_rows = (int(eye_left_center[0]-roi_h/2),int(eye_left_center[0]+roi_h/2))
    eye_left_roi_cols = (int(eye_left_center[1]-roi_w/2),int(eye_left_center[1]+roi_w/2))

    eye_north_roi_rows = (int(eye_north_center[0]-roi_w/2),int(eye_north_center[0]+roi_w/2))
    eye_north_roi_cols = (int(eye_north_center[1]-roi_h/2),int(eye_north_center[1]+roi_h/2))
    
    eye_south_roi_rows = (int(eye_south_center[0]-roi_w/2),int(eye_south_center[0]+roi_w/2))
    eye_south_roi_cols = (int(eye_south_center[1]-roi_h/2),int(eye_south_center[1]+roi_h/2))
    
    # Calculate anchors (upper-left corner of each roi)
    anchors = [
        (eye_right_roi_rows[0], eye_right_roi_cols[0],roi_h,roi_w),
        (eye_left_roi_rows[0], eye_left_roi_cols[0],roi_h,roi_w),
        (eye_north_roi_rows[0], eye_north_roi_cols[0],roi_w,roi_h),
        (eye_south_roi_rows[0], eye_south_roi_cols[0],roi_w,roi_h),
              ]
    
    # Calculate windows
    eye_right_roi = image[eye_right_roi_rows[0]:eye_right_roi_rows[1],
                          eye_right_roi_cols[0]:eye_right_roi_cols[1]]
    eye_left_roi = image[eye_left_roi_rows[0]:eye_left_roi_rows[1],
                          eye_left_roi_cols[0]:eye_left_roi_cols[1]]
    eye_north_roi = image[eye_north_roi_rows[0]:eye_north_roi_rows[1],
                          eye_north_roi_cols[0]:eye_north_roi_cols[1]]
    eye_south_roi = image[eye_south_roi_rows[0]:eye_south_roi_rows[1],
                          eye_south_roi_cols[0]:eye_south_roi_cols[1]]
    
    rois = np.array([eye_right_roi, eye_left_roi, eye_north_roi, eye_south_roi])

    eye_intensity_horizontal = np.sum(eye_right_roi)+np.sum(eye_left_roi)
    eye_intensity_vertical = np.sum(eye_north_roi)+np.sum(eye_south_roi)
    
    intensity_ratio = eye_intensity_horizontal/eye_intensity_vertical
    
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
