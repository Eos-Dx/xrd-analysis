"""
Calculate features using preprocessing functions
"""
import os
import glob
from datetime import datetime
import numpy as np

from eosdxanalysis.models.curve_fitting import GaussianShoulderFitting

from eosdxanalysis.simulations.utils import feature_pixel_location

from eosdxanalysis.models.utils import radial_intensity_1d

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

    def __init__(self, image, params=None):
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

        # Calculate the peak maximum
        peak_max = np.max(roi)

        return peak_location, peak_max, roi, roi_center, anchor

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

    @classmethod
    def fwhm(self, input_array, direction=+1):
        """
        Calculates the full-width at half maximum for a 1D input array
        Uses a one-sided half-max

        Inputs:
        - input_array: 1D array
        - direction: keyword argument, default is one-sided half-max
          in the positive index direction

        Outputs:
        - Full-width half max
        - Maximum value
        - Maximum location
        - Half-maximum value
        - Half-maximum location
        """
        # Ensure input array is 1D
        if len(input_array.flatten().shape) != 1:
            raise ValueError("Input array must be 1D!")

        # Find the maximum value in the input array
        max_val = np.max(input_array)
        max_val_loc_array = np.where(input_array == max_val)[0]

        # Ensure that only one maximum is found
        if len(max_val_loc_array) > 1:
            raise ValueError("More than one maximum found!")

        max_val_loc = max_val_loc_array[0]

        # Calculate the half-maximum value
        half_max = max_val/2

        # Take a subarray to calculate the one-sided half-max location
        sub_array = input_array[max_val_loc:]

        # Sub-array from max value to half-max
        max_to_half_max_array  = np.where(sub_array >= half_max)[0]

        # Get the location of the half-max value in the sub-array
        try:
            # The one-sided half-maximum width
            half_max_sub_loc = max_to_half_max_array[-1]
        except IndexError as err:
            # Handle the case where the half-max is not found
            raise ValueError("Half-max value not found!")
        except Exception as err:
            print("An error occured finding half-max location!")
            raise err

        # Calculate the half-maximum location in the original array
        half_max_loc = half_max_sub_loc + max_val_loc

        # Calculate the full-width at half-maximum from the one-sided width
        full_width_half_max = 2 * half_max_sub_loc

        return  full_width_half_max, max_val, max_val_loc, half_max, half_max_loc

    def features_5A_shoulder(self, method="approximate"):
        """
        Given an image, approximate the 5.1 A peak and beyond as a sum of two
        Gaussians. The hypothesis is that measurements of healthy specimens
        present with a single Gaussian, whereas measurements of cancerous
        specimens present with a second Gaussian slightly beyond 5.1 A
        (perhaps 4.9 A)

        Parameters
        ----------

        method : str
            ``method`` can be `optimal` or `approximate`.

        Returns
        -------

        shoulder_params : tuple of Gaussian parameters
            5.1 A peak location, standard deviation, and amplitude, as well as
            4.9 A peak location, standard deviation, and amplitude.

        """

        if method == "ideal":
            # Use initial parameter guesses close to the ideal
            # Get the rotated image
            image = np.rot90(self.image)
            # Calculate the radial intensity of the positive meridian
            vertical_intensity_1d = radial_intensity_1d(image)

            # Calculate the full 1D angular integrated profile
            from eosdxanalysis.models.utils import angular_intensity_1d

            intensity = np.zeros((95,))
            for idx in range(95):
                radius = idx + 25
                angular_profile = angular_intensity_1d(image, radius=radius, N=360)
                intensity[idx] = np.sum(angular_profile)

            import matplotlib.pyplot as plt
            plt.plot(intensity)
            plt.show()

            return

        elif method == "estimation":
            raise NotImplementedError(
            "``estimation`` shoulder feature calculations not implemented yet.")
        else:
            raise ValueError("Specify a valid method.")

        return vertical_intensity_1d, features_dict

    @classmethod
    def features_5A_shoulder_params_list(self):
        """
        Returns list of 5A shoulder parameters
        """

        shoulder_parameters_list = [
                "gauss_51nm_amplitude",
                "gauss_51nm_std",
                "gauss_51nm_mean",
                "gauss_49nm_amplitude",
                "gauss_49nm_std",
                "gauss_49nm_mean",
                ]

        return shoulder_parameters_list

def shoulder_analysis(input_path, output_path=None,
        method="ideal"):
    """
    Runs batch shoulder analysis
    """
    # Get full paths to files and created sorted list
    file_path_list = glob.glob(os.path.join(input_path,"*.txt"))
    file_path_list.sort()

    # Set timestamp
    timestr = "%Y%m%dT%H%M%S.%f"
    timestamp = datetime.utcnow().strftime(timestr)

    # Set output path with a timestamp if not specified
    if not output_path:
        output_dir = "shoulder_analysis_{}".format(timestamp)
        output_path = os.path.join(input_path, "..", output_dir)

    plot_orig_dir = "plot_orig"
    plot_orig_path = os.path.join(output_path, plot_orig_dir)
    plot_fit_dir = "plot_fit"
    plot_fit_path = os.path.join(output_path, plot_fit_dir)

    # Create output image paths
    os.makedirs(plot_orig_path, exist_ok=True)
    os.makedirs(plot_fit_path, exist_ok=True)

    # Get list of parameters
    param_list = EngineeredFeatures.features_5A_shoulder_params_list()

    # Construct empty list for storing data
    row_list = []

    # Loop over files
    for file_path in file_path_list:
        # Load data
        filename = os.path.basename(file_path)
        image = np.loadtxt(file_path, dtype=np.float64)

        # Get features
        feature_class = EngineeredFeatures(image)

        try:
            radial_profile_1d_orig, popt_dict = \
                    feature_class.features_5A_shoulder(method=method)
            popt = np.fromiter(popt_dict.values(), dtype=np.float64)
        except RuntimeError as err:
            print("Could not find Gaussian fit for {}.".format(filename))
            print(err)
            popt = np.array([0]*6)
        except TypeError:
            continue

        # Get best fit profile
        radial_profile_1d_fit = EngineeredFeatures.shoulder_pattern(popt)

        # Get squared error for best fit
        error = gauss_class.fit_error(
                radial_profile_1d_orig, radial_profile_1d_fit)
        error_ratio = error/np.sum(np.square(radial_profile_1d_orig))
        r_factor = np.sum(
                np.abs(np.sqrt(radial_profile_1d_orig) \
                        - np.sqrt(radial_profile_1d_fit))) \
                / np.sum(np.sqrt(radial_profile_1d_orig))

        # Construct dataframe row
        # - filename
        # - optimum fit parameters
        row = [filename, error, error_ratio, r_factor] + popt.tolist()
        row_list.append(row)

        # Save original 1D plot
        save_plot_filename = "{}_{}.png".format(output_prefix, filename)
        save_plot_fullpath = os.path.join(plot_fit_path,
                save_plot_filename)
        fig = plt.figure(facecolor="white")
        plt.plot(radial_intensity_1d_orig)
        plt.savefig(save_plot_fullpath)
        fig.clear()
        plt.close(fig)

        # Save Gaussian sum 1D plot
        save_plot_filename = "{}_{}.png".format(output_prefix, filename)
        save_plot_fullpath = os.path.join(plot_fit_path,
                save_plot_filename)
        fig = plt.figure(facecolor="white")
        plt.plot(radial_intensity_1d_fit)
        plt.savefig(save_plot_fullpath)
        fig.clear()
        plt.close(fig)

    # Create dataframe to store parameters

    # Construct pandas dataframe columns
    columns = ["Filename", "Error", "Error_Ratio", "R_Factor"] + param_list

    df = pd.DataFrame(data=row_list, columns=columns)

    # Save dataframe
    csv_filename = "{}.csv".format(output_prefix)
    csv_output_path = os.path.join(output_path, csv_filename)
    df.to_csv(csv_output_path)

if __name__ == '__main__':
    """
    Run feature analysis on a file or entire folder.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--input_path", default=None, required=True,
            help="The path containing raw files to perform fitting on")
    parser.add_argument(
            "--output_path", default=None, required=False,
            help="The output path to store results.")
    parser.add_argument(
            "--feature", default="shoulder", required=True,
            help="Specify the feature to calculate. Use ``all`` for all.")
    parser.add_argument(
            "--params_init_method", default="ideal", required=False,
            help="For Gaussian fitting, the default method to initialize the"
            " parameters Options are: ``ideal`` and ``approximate``.")

    # Collect arguments
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    feature = args.feature
    params_init_method = args.params_init_method

    if feature == "gaussian-decomposition":
        gaussian_decomposition(input_path, output_path, params_init_method)
    else:
        raise NotImplementedError("``{}`` not implemented.".format(feature))
