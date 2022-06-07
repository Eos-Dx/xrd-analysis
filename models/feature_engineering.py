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
    
    intensity_ratio = eye_intensity_vertical/eye_intensity_horizontal
    
    return intensity_ratio, rois, centers, anchors


def feature_5a_peak_location(image, rmin=70, roi_h=20, roi_w=10):
    """
    Calculate the location of the 5A peak
    normalized by image radius
    """
    # Known location of 5A peak
    radius_5a = 65
    # Calculate center of image
    shape = image.shape
    row_isodd = shape[0]%2
    col_isodd = shape[1]%2
    row_center = shape[0]/2+row_isodd
    col_center = shape[1]/2+col_isodd
    
    # Calculate radius
    radius = rmin + roi_h/2

    # Calculate the center of the roi mask, a vertical strip
    roi_center = (row_center + row_isodd - radius, col_center + col_isodd)
    
    # Calculate the slice indices
    roi_rows = (int(roi_center[0]-roi_h/2),int(roi_center[0]+roi_h/2))
    roi_cols = (int(roi_center[1]-roi_w/2),int(roi_center[1]+roi_w/2))

    # Calculate anchor (upper-left corner of roi)
    anchor = (roi_rows[0], roi_cols[0],roi_h,roi_w)
    
    # Calculate the roi
    roi = image[roi_rows[0]:roi_rows[1],
                          roi_cols[0]:roi_cols[1]]
    
    # Average across columns
    roi_avg = np.mean(roi,axis=1)
    
    # Calculate peak location using maximum of centroid
    peak_locations = np.where(roi_avg == np.max(roi_avg))
    # The row number of the peak in the roi
    peak_location = np.mean(peak_locations)
    
    # Now we need to offset to get location in image
    peak_location_radius = row_center - peak_location
    
    peak_location_feature = (peak_location_radius - radius_5a)/radius_5a
    
    return peak_location_feature, roi, roi_center, anchor
