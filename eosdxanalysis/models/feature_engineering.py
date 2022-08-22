"""
Calculate features using preprocessing functions
"""
import os
import numpy as np

from eosdxanalysis.simulations.utils import feature_pixel_location

MODULE_PATH = os.path.dirname(__file__)
MODULE_DATA_PATH = os.path.join(MODULE_PATH, "data")
TEMPLATE_FILENAME = "amorphous-scattering-template.txt"
TEMPLATE_PATH = os.path.join(MODULE_DATA_PATH, TEMPLATE_FILENAME)

# Molecular properties
SPACING_9A = 9.8e-10 # meters

# Machine properties
DISTANCE = 10e-3 # meters
WAVELENGTH = 1.5418e-10 # meters
PIXEL_WIDTH = 55e-6 # meters

class EngineeredFeatures(object):
    """
    Class to calculate engineered features.
    """

    def __init__(self, image, params):
        """
        Initialize engineered features class with an image
        and parameters.

        Inputs:
        - image (quad-folded image as 2D numpy array)
        - params (dictionary)

        Returns:
        - Class instance
        """
        # Ensure image size is 256x256
        if image.shape != (256,256):
            raise ValueError("This class is only designed for images of size 256x256")

        self.image = image
        self.params = params

        # Calculte image center coordinates
        center = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)
        self.center = center

        return super().__init__()

    def feature_5a_peak_location(self, row_min=0, row_max=78, roi_w=6):
        """
        Calculate the location (radius) of the 5A peak in the given image.

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
        The window is centered horizontally.

        Inputs:
        - row_min: minimum row number
        - row_max: maximum row number
        - roi_w: roi width

        """

        image = self.image
        # Calculate center of image
        center = self.center

        # Calculate the roi rows and columns
        roi_rows = (row_min, row_max)
        roi_cols = (int(center[1]-roi_w/2), int(center[1]+roi_w/2))

        # Calculate the center of the roi
        roi_center = (roi_rows[1] - roi_rows[0],
                        roi_cols[1] - roi_cols[0])

        # Calculate anchor (upper-left corner of roi)
        anchor = (row_min, int(center[1]-roi_w/2))

        # Calculate the roi
        roi = image[slice(*roi_rows),
                              slice(*roi_cols)]

        # Average across columns
        roi_avg = np.mean(roi,axis=1)

        # Calculate peak location using maximum of centroid
        roi_peak_location_list = np.where(roi_avg == np.max(roi_avg))
        # Calculate the row number of the peak in the roi
        roi_peak_location = np.mean(roi_peak_location_list)
        # Calculaate the row number of the peak in the image
        peak_location = roi_peak_location + row_min

        return peak_location, roi, roi_center, anchor

    def feature_9a_peak_location(self, roi_h=20, roi_w=6):
        """
        Calculate the location (radius) of the 9A peak in the given image.

        Define a rectangular region of interest as in the ascii graphic
         _ <- roi_w
        | | -
        | | |
        | | | <- roi_h
        | | |
        |_| -

        The window shown is defined for the right 9.8A peak (eye).
        It's height is roi_h.
        It's width is roi_w.

        Inputs:
        - roi_h: roi height
        - roi_w: roi width

        """
        theory_peak_location = feature_pixel_location(SPACING_9A,
                distance=DISTANCE, wavelength=WAVELENGTH, pixel_width=PIXEL_WIDTH)

        image = self.image
        # Calculate center of image
        center = self.center

        # Calculate the roi rows and columns
        roi_rows = (int(center[0] - roi_h/2), int(center[0] + roi_h/2))
        roi_cols = (int(center[1] + theory_peak_location - roi_w/2),
                    int(center[1] + theory_peak_location + roi_w/2))

        # Calculate the center of the roi
        roi_center = (roi_rows[1] - roi_rows[0],
                        roi_cols[1] - roi_cols[0])

        # Calculate anchor (upper-left corner of roi)
        anchor = (int(center[0] - roi_h/2), int(center[1] - roi_w/2))

        # Calculate the roi
        roi = image[slice(*roi_rows),
                              slice(*roi_cols)]

        # Average across rows
        roi_avg = np.mean(roi,axis=0)

        # Calculate peak location using maximum of centroid
        roi_peak_location_list = np.where(roi_avg == np.max(roi_avg))
        # Calculate the column number of the peak in the roi
        roi_peak_location = np.mean(roi_peak_location_list)
        # Calculate the horizontal pixel radius of the peak location
        peak_location = roi_peak_location + roi_cols[0]

        return peak_location, roi, roi_center, anchor

    def feature_9a_ratio(self, roi_h=20, roi_w=6):
        """
        Calulate the ratio of vertical region of interest (roi) intensities
        over horizontal window roi intensities in the 9A region

        Define a rectangular region of interest as in the ascii graphic
         _ <- roi_w
        | | -
        | | |
        | | | <- roi_h
        | | |
        |_| -

        The window shown is defined for the right 9.8A peak (eye).
        It's height is roi_h.
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

        image = self.image
        # Calculate center of image
        center = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

        theory_peak_location = feature_pixel_location(SPACING_9A,
                distance=DISTANCE, wavelength=WAVELENGTH, pixel_width=PIXEL_WIDTH)

        # Calculate center of roi's
        roi_right_center = (int(center[0]),
                            int(center[1] + theory_peak_location + roi_w/2))
        roi_left_center = (int(center[0]),
                            int(center[1] - theory_peak_location - roi_w/2))
        roi_top_center = (int(center[0] - theory_peak_location - roi_w/2),
                            int(center[1]))
        roi_bottom_center = (int(center[0] + theory_peak_location + roi_w/2),
                            int(center[1]))

        centers = (
                roi_right_center,
                roi_left_center,
                roi_top_center,
                roi_bottom_center,
                )

        # Calculate slice indices
        roi_right_rows = (int(roi_right_center[0]-roi_h/2),
                        int(roi_right_center[0]+roi_h/2))
        roi_right_cols = (int(roi_right_center[1]-roi_w/2),
                        int(roi_right_center[1]+roi_w/2))

        roi_left_rows = (int(roi_left_center[0]-roi_h/2),
                        int(roi_left_center[0]+roi_h/2))
        roi_left_cols = (int(roi_left_center[1]-roi_w/2),
                        int(roi_left_center[1]+roi_w/2))

        roi_top_rows = (int(roi_top_center[0]-roi_w/2),
                        int(roi_top_center[0]+roi_w/2))
        roi_top_cols = (int(roi_top_center[1]-roi_h/2),
                        int(roi_top_center[1]+roi_h/2))

        roi_bottom_rows = (int(roi_bottom_center[0]-roi_w/2),
                            int(roi_bottom_center[0]+roi_w/2))
        roi_bottom_cols = (int(roi_bottom_center[1]-roi_h/2),
                            int(roi_bottom_center[1]+roi_h/2))

        # Calculate anchors (upper-left corner of each roi)
        anchors = [
            (roi_right_rows[0], roi_right_cols[0],roi_h,roi_w),
            (roi_left_rows[0], roi_left_cols[0],roi_h,roi_w),
            (roi_top_rows[0], roi_top_cols[0],roi_w,roi_h),
            (roi_bottom_rows[0], roi_bottom_cols[0],roi_w,roi_h),
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


    def feature_5a_9a_peak_location_ratio(self,
                                        row_min_5a=6, row_max_5a=78, roi_w_5a=4,
                                        start_radius_9a=25, roi_l_9a=18, roi_w_9a=4):
        """
        Calculate the locations (radii) of the 5.1A and 9.8A peaks.
        Then take the ratio of their locations (radii).
        """

        image = self.image
        ######################################
        # Calculate the 5.1A location radius #
        ######################################
        peak_5a_location, _, roi_5a_center, _ = self.feature_5a_peak_location(
                            row_min=row_min_5a, row_max=row_max_5a, roi_w=roi_w_5a)
        peak_5a_radius = image.shape[0]//2 - peak_5a_location + 0.5

        ########################
        # Define the 9.8A roi #
        ########################
        # Calculate center of horizontal/equitorial roi's
        roi_9a_right_center = (int(image.shape[0]/2),
                            int(image.shape[1]/2 + start_radius_9a + roi_w_9a/2))
        roi_9a_left_center = (int(image.shape[0]/2),
                            int(image.shape[1]/2 - start_radius_9a - roi_w_9a/2))
        # Calculate slice indices
        roi_9a_right_rows = (int(roi_9a_right_center[0]-roi_l_9a/2),
                        int(roi_9a_right_center[0]+roi_l_9a/2))
        roi_9a_right_cols = (int(roi_9a_right_center[1]-roi_w_9a/2),
                        int(roi_9a_right_center[1]+roi_w_9a/2))

        roi_9a_left_rows = (int(roi_9a_left_center[0]-roi_l_9a/2),
                        int(roi_9a_left_center[0]+roi_l_9a/2))
        roi_9a_left_cols = (int(roi_9a_left_center[1]-roi_w_9a/2),
                        int(roi_9a_left_center[1]+roi_w_9a/2))

        # Calculate windows
        roi_9a_right = image[roi_9a_right_rows[0]:roi_9a_right_rows[1],
                              roi_9a_right_cols[0]:roi_9a_right_cols[1]]
        roi_9a_left = image[roi_9a_left_rows[0]:roi_9a_left_rows[1],
                              roi_9a_left_cols[0]:roi_9a_left_cols[1]]
        # Combine the windows
        roi_9a = roi_9a_right + roi_9a_left[:,::-1]
        # Calculate the peak location (radius)
        roi_9a_peak_location_list = np.where(roi_9a == np.max(roi_9a))
        roi_9a_peak_location = np.mean(roi_9a_peak_location_list, axis=1)

        # Convert from roi coordinates to radius in image
        peak_9a_radius = np.sqrt(np.square(start_radius_9a + roi_9a_peak_location[0] + 0.5) + \
                np.square(roi_l_9a/2 - roi_9a_peak_location[1] + 0.5))

        #############################################################################
        # Calculate the ratio of the 5.1A peak location over the 9.8A peak location #
        #############################################################################

        ratio = peak_5a_radius/peak_9a_radius
        return ratio

    def feature_amorphous_scattering_intensity_ratio(self):
        """
        Calculate the amorphous scattering intensity,
        which is the intensity in areas outside of the main features
        of the 9A and 5A peaks
        """

        image = self.image
        # Import the template
        template = np.loadtxt(TEMPLATE_PATH, dtype=bool)
        # Get the intensity outside of the template
        amorphous_intensity = np.sum(image[~template])
        # Normalize by the total intensity
        total_intensity = np.sum(image)
        amorphous_intensity_ratio = amorphous_intensity/total_intensity

        return amorphous_intensity_ratio
