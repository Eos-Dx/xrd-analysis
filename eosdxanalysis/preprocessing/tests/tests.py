import os
import shutil
import unittest
import numpy as np
import pandas as pd

import subprocess
import json
import glob

from skimage.io import imsave, imread
from skimage.transform import warp_polar
from scipy.ndimage import center_of_mass

from scipy.interpolate import RegularGridInterpolator

from sklearn.pipeline import make_pipeline

from eosdxanalysis.preprocessing.image_processing import pad_image
from eosdxanalysis.preprocessing.image_processing import crop_image
from eosdxanalysis.preprocessing.image_processing import enlarge_image
from eosdxanalysis.preprocessing.image_processing import bright_pixel_count

from eosdxanalysis.preprocessing.center_finding import circular_average

from eosdxanalysis.preprocessing.denoising import filter_outlier_pixel_values
from eosdxanalysis.preprocessing.denoising import filter_outlier_pixel_values_dir
from eosdxanalysis.preprocessing.denoising import find_outlier_pixel_values
from eosdxanalysis.preprocessing.denoising import FilterOutlierPixelValues

from eosdxanalysis.preprocessing.utils import count_intervals
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import gen_rotation_line
from eosdxanalysis.preprocessing.utils import get_angle
from eosdxanalysis.preprocessing.utils import polar_meshgrid
from eosdxanalysis.preprocessing.utils import unwarp_polar
from eosdxanalysis.preprocessing.utils import find_center
from eosdxanalysis.preprocessing.utils import find_centroid

from eosdxanalysis.preprocessing.preprocess import PreprocessData
from eosdxanalysis.preprocessing.preprocess import ABBREVIATIONS
from eosdxanalysis.preprocessing.preprocess import OUTPUT_MAP
from eosdxanalysis.preprocessing.preprocess import INVERSE_OUTPUT_MAP

from eosdxanalysis.preprocessing.peak_finding import find_2d_peak
from eosdxanalysis.preprocessing.peak_finding import find_1d_peaks

from eosdxanalysis.preprocessing.angle_finding import find_rotation_angle

from eosdxanalysis.preprocessing.beam_utils import first_valley_location
from eosdxanalysis.preprocessing.beam_utils import beam_extent

from eosdxanalysis.preprocessing.feature_extraction import FeatureExtraction

from eosdxanalysis.preprocessing.azimuthal_integration import azimuthal_integration
from eosdxanalysis.preprocessing.azimuthal_integration import AzimuthalIntegration
from eosdxanalysis.preprocessing.azimuthal_integration import azimuthal_integration_dir
from eosdxanalysis.preprocessing.azimuthal_integration import radial_intensity_sum

from eosdxanalysis.simulations.utils import feature_pixel_location

from eosdxanalysis.calibration.units_conversion import DiffractionUnitsConversion

TEST_PATH = os.path.dirname(__file__)
MODULE_PATH = os.path.join(TEST_PATH, "..")
TEST_IMAGE_DIR = "test_images"
TEST_IMAGE_PATH = os.path.join(TEST_PATH, TEST_IMAGE_DIR)


def intensity_profile_function(coords,N=16):
    max_intensity = 1000
    amplitude = max_intensity//4
    return amplitude*np.cos(np.pi*coords/N) + \
            amplitude*np.sin(np.pi*3*coords/N) + \
            amplitude*np.cos(np.pi*5*coords/N) + \
            amplitude*np.sin(np.pi*7*coords/N) + \
            max_intensity

def gen_1d_intensity_profile(N=16):
    """
    Helper function to generate 1D intensity profile for testing
    """
    coords = np.arange(0,N)
    return intensity_profile_function(coords,N)

def gen_2d_intensity_profile(N=16):
    """
    Helper function to generate 1D intensity profile for testing
    """
    X,Y = np.meshgrid(np.arange(N),np.arange(N))
    Z = np.sin(X)+np.cos(Y)
    return intensity_profile_function(Z,N)


class TestCreateCircularMask(unittest.TestCase):
    """
    Test create_circular_mask
    """

    def test_circular_mask_odd_square_output(self):
        """
        Test circular mask for shape (odd,odd)
        """
        # Specify 5x5 shape
        nrows = 5
        ncols = 5
        shape = (nrows,ncols)

        # Create (5,5) circular mask
        mask_5x5 = create_circular_mask(shape[0], shape[1])

        # Store expected result for circular mask of (5,5) array
        # with default center, rmin, and rmax
        test_5x5_mask = np.array([
            [False,  False, True, False, False],
            [False,  True,  True, True,  False],
            [True,   True,  True, True,  True],
            [False,  True,  True, True,  False],
            [False,  False, True, False, False],
        ])

        # Test if the generated mask equals the expected mask
        self.assertTrue(np.array_equal(mask_5x5, test_5x5_mask))

    def test_circular_mask_even_square_output(self):
        """
        Test circular mask for shape (even,even)
        """
        # Specify 4x4 shape
        nrows = 4
        ncols = 4
        shape = (nrows,ncols)

        # Create (4,4) circular mask
        mask_4x4 = create_circular_mask(shape[0], shape[1])

        # Store expected result for circular mask of (4,4) array
        # with default center, rmin, and rmax
        test_4x4_mask = np.array([
            [False, False, False, False],
            [False,  True,  True, False],
            [False,  True,  True, False],
            [False, False, False, False],
        ])

        # Test if the generated mask equals the expected mask
        self.assertTrue(np.array_equal(mask_4x4, test_4x4_mask))

    def test_circular_mask_center_specified_even_square_output(self):
        # Specify 4x4 shape
        nrows = 4
        ncols = 4
        row_center = 1.5
        col_center = 1.5
        shape = (nrows,ncols)

        # Create (4,4) circular mask
        mask_4x4 = create_circular_mask(shape[0], shape[1], center=(row_center, col_center))

        # Store expected result for circular mask of (4,4) array
        # with default center, rmin, and rmax
        test_4x4_mask = np.array([
            [False, False, False, False],
            [False,  True,  True, False],
            [False,  True,  True, False],
            [False, False, False, False],
        ])

        # Test if the generated mask equals the expected mask
        self.assertTrue(np.array_equal(mask_4x4, test_4x4_mask))

    def test_circular_mask_center_specified_odd_square_output(self):
        # Specify 5x5 shape
        nrows = 5
        ncols = 5
        row_center = 2
        col_center = 2
        shape = (nrows,ncols)

        # Create (5,5) circular mask
        mask_5x5 = create_circular_mask(shape[0], shape[1], center=(row_center, col_center))

        # Store expected result for circular mask of (5,5) array
        # with default center, rmin, and rmax
        test_5x5_mask = np.array([
            [False, False,  True, False, False],
            [False,  True,  True,  True, False],
            [ True,  True,  True,  True,  True],
            [False,  True,  True,  True, False],
            [False, False,  True, False, False],
        ])

        # Test if the generated mask equals the expected mask
        self.assertTrue(np.array_equal(mask_5x5, test_5x5_mask))

    def test_circular_mask_off_center_specified_even_square_output(self):
        # Specify 4x4 shape
        nrows = 4
        ncols = 4
        row_center = 2
        col_center = 2
        shape = (nrows,ncols)

        # Create (4,4) circular mask
        mask_4x4 = create_circular_mask(shape[0], shape[1], center=(row_center, col_center), mode="min")

        # Store expected result for circular mask of (4,4) array
        # with default center, rmin, and rmax
        test_4x4_mask = np.array([
            [False, False, False, False],
            [False, False, True, False],
            [False,  True, True,  True],
            [False, False, True, False],
        ])

        # Test if the generated mask equals the expected mask
        self.assertTrue(np.array_equal(mask_4x4, test_4x4_mask))


class TestPreprocessingCLI(unittest.TestCase):
    """
    Test to ensure preprocessing can be done from commandline
    """

    def setUp(self):
        test_path = os.path.join(TEST_IMAGE_PATH, "test_cli_images")
        self.test_path = test_path

        # Specify parameters file without plans
        params_file = os.path.join(test_path, "params.txt")
        self.params_file = params_file
        with open(params_file, "r") as param_fp:
            params = param_fp.read()
        self.params = params

        # Specify parameters file with plans
        params_with_plans_file = os.path.join(test_path, "params_with_plans.txt")
        self.params_with_plans_file = params_with_plans_file
        with open(params_with_plans_file, "r") as param_fp:
            params_with_plans = param_fp.read()
        self.params_with_plans = params_with_plans

        # Create test images
        input_path="input"

        # Set up test image
        test_image = np.zeros((256,256), dtype=np.uint16)
        center = (test_image.shape[0]/2-0.5, test_image.shape[1]/2-0.5)
        test_image[127:129, 127:129] = 9
        test_image[(127-50):(129-50), (127-50+2):(129-50+2)] = 5
        test_image[(127+50):(129+50), (127+50-2):(129+50-2)] = 5

        # Set the filename
        filename = "test_cli.txt"
        # Set the full output path
        fullpath = os.path.join(test_path, input_path, filename)
        # Save the image to file
        np.savetxt(fullpath, test_image, fmt="%d")

        # Set the input and output pathectories
        test_input_path = os.path.join(test_path, "input")
        test_output_path = os.path.join(test_path, "output")
        # Create the test output pathectory
        os.makedirs(test_output_path, exist_ok=True)

        self.test_input_path = test_input_path
        self.test_output_path = test_output_path

        input_file_path_list = glob.glob(os.path.join(test_input_path, "*.txt"))
        input_file_path_list.sort()

        self.input_file_path_list = input_file_path_list

    def test_preprocess_cli_input_dir_output_dir_params_file_plans_in_params_file(self):
        """
        Run preprocessing using commandline, providing input data directory
        and output directory.
        """
        params_with_plans_file = self.params_with_plans_file
        test_input_path = self.test_input_path
        test_output_path = self.test_output_path
        input_file_path_list = self.input_file_path_list

        # Construct plans list
        plan = "centerize_rotate_quad_fold"
        plan_abbr = ABBREVIATIONS.get(plan)
        output_style = INVERSE_OUTPUT_MAP.get(plan)

        # Set up the command
        command = ["python", "eosdxanalysis/preprocessing/preprocess.py",
                    "--input_path", test_input_path,
                    "--output_path", test_output_path,
                    "--params_file", params_with_plans_file,
                    ]

        # Run the command
        subprocess.run(command)

        # Check that output files exist
        # First get list of files
        num_files = len(input_file_path_list)

        # Check that number of files is > 0
        self.assertTrue(num_files > 0)
        # Check that number of input and output files is the same
        plan_output_path = os.path.join(test_output_path, output_style)
        plan_output_file_path_list = glob.glob(os.path.join(plan_output_path, "*.txt"))
        plan_output_file_path_list.sort()

        self.assertEqual(num_files, len(plan_output_file_path_list))

        for idx in range(num_files):
            # Load data
            input_image = np.loadtxt(input_file_path_list[idx])
            output_image = np.loadtxt(plan_output_file_path_list[idx])

            # Check that data are positive
            self.assertTrue(input_image[input_image > 0].all())
            self.assertTrue(output_image[output_image > 0].all())

            # Check that the maximum value of the output is less than the
            # maximum value of the input
            self.assertTrue(np.max(output_image) < np.max(input_image))

            # Check that output means are smaller than input means
            self.assertTrue(np.mean(output_image) < np.mean(input_image))

    def test_preprocess_cli_input_dir_output_dir_params_file_params_csv_string(self):
        """
        Run preprocessing using commandline, providing input data directory
        and output directory.
        """
        params_file = self.params_file
        test_input_path = self.test_input_path
        test_output_path = self.test_output_path
        input_file_path_list = self.input_file_path_list

        # Construct plans list
        plan = "centerize_rotate_quad_fold"
        plans = [plan,]
        plan_abbr = ABBREVIATIONS.get(plan)
        output_style = INVERSE_OUTPUT_MAP.get(plan)

        # Set up the command
        command = ["python", "eosdxanalysis/preprocessing/preprocess.py",
                    "--input_path", test_input_path,
                    "--output_path", test_output_path,
                    "--params_file", params_file,
                    "--plans", ",".join(plans),
                    ]

        # Run the command
        subprocess.run(command)

        # Check that output files exist
        # First get list of files
        num_files = len(input_file_path_list)

        # Check that number of files is > 0
        self.assertTrue(num_files > 0)
        # Check that number of input and output files is the same
        plan_output_path = os.path.join(test_output_path, output_style)
        plan_output_file_path_list = glob.glob(os.path.join(plan_output_path, "*.txt"))
        plan_output_file_path_list.sort()

        self.assertEqual(num_files, len(plan_output_file_path_list))

        for idx in range(num_files):
            # Load data
            input_image = np.loadtxt(input_file_path_list[idx])
            output_image = np.loadtxt(plan_output_file_path_list[idx])

            # Check that data are positive
            self.assertTrue(input_image[input_image > 0].all())
            self.assertTrue(output_image[output_image > 0].all())

            # Check that the maximum value of the output is less than the
            # maximum value of the input
            self.assertTrue(np.max(output_image) < np.max(input_image))

            # Check that output means are smaller than input means
            self.assertTrue(np.mean(output_image) < np.mean(input_image))

    def tearDown(self):
        # Delete the output folder
        shutil.rmtree(self.test_output_path)


class TestPreprocessData(unittest.TestCase):

    def setUp(self):
        # Set test path
        test_path = os.path.join(TEST_IMAGE_PATH, "test_preprocessing_images")
        self.test_path = test_path

        # Specify parameters file without plans
        params_file = os.path.join(test_path, "params.txt")
        self.params_file = params_file
        with open(params_file, "r") as param_fp:
            params = param_fp.read()
        self.params = params

        # Specify parameters file with plans
        params_with_plans_file = os.path.join(test_path, "params_with_plans.txt")
        self.params_with_plans_file = params_with_plans_file
        with open(params_with_plans_file, "r") as param_fp:
            params_with_plans = param_fp.read()
        self.params_with_plans = params_with_plans

        # Specify parameters file with centerize rotate plans
        params_centerize_rotate_file = os.path.join(test_path, "params_centerize_rotate.txt")
        self.params_centerize_rotate_file = params_centerize_rotate_file
        with open(params_centerize_rotate_file, "r") as param_fp:
            params_centerize_rotate = param_fp.read()
        self.params_centerize_rotate = params_centerize_rotate

        # Specify parameters file with hot spot filtering
        params_hot_spot_filter_file = os.path.join(test_path, "params_hot_spot_filter.txt")
        self.params_hot_spot_filter_file = params_hot_spot_filter_file

        with open(params_hot_spot_filter_file, "r") as param_fp:
            params_hot_spot_filter = param_fp.read()
        self.params_hot_spot_filter = params_hot_spot_filter

        # Specify parameters file with hot spot filtering
        params_hot_spot_no_filter_file = os.path.join(test_path, "params_hot_spot_no_filter.txt")
        self.params_hot_spot_no_filter_file = params_hot_spot_no_filter_file

        with open(params_hot_spot_no_filter_file, "r") as param_fp:
            params_hot_spot_no_filter = param_fp.read()
        self.params_hot_spot_no_filter = params_hot_spot_no_filter

        # Create test images
        INPUT_DIR="input"

        # Set up test image
        test_image = np.zeros((256, 256), dtype=np.uint16)
        center = (test_image.shape[0]/2-0.5, test_image.shape[1]/2-0.5)
        test_image[127:129, 127:129] = 9
        test_image[92, 163] = 5
        test_image[163, 92] = 5

        self.test_image = test_image

        # Set the filename
        filename = "test_cli.txt"
        # Set the full output path
        output_file_path = os.path.join(test_path, INPUT_DIR, filename)
        # Save the image to file
        np.savetxt(output_file_path, test_image, fmt="%d")

        # Set the input and output paths
        test_input_path = os.path.join(test_path, "input")
        test_output_path = os.path.join(test_path, "output")
        # Create the output path
        os.makedirs(test_output_path, exist_ok=True)

        self.test_input_path = test_input_path
        self.test_output_path = test_output_path

        input_file_path_list = glob.glob(os.path.join(test_input_path, "*.txt"))
        input_file_path_list.sort()

        self.input_file_path_list = input_file_path_list

    def test_preprocess_single_image_rotate_centerize(self):
        params_centerize_rotate = self.params_centerize_rotate
        test_input_path = self.test_input_path
        test_output_path = self.test_output_path
        input_file_path_list = self.input_file_path_list

        input_filename = "synthetic_sparse_image.txt"
        input_filename_fullpath = os.path.join(test_input_path, input_filename)
        output_filename_fullpath = os.path.join(test_output_path,
                "centered_rotated", "CR_" + input_filename)

        params = json.loads(params_centerize_rotate)

        preprocessor = PreprocessData(filename=input_filename,
                input_path=test_input_path, output_path=test_output_path, params=params)

        # Preprocess data, saving to a file
        preprocessor.preprocess()

        # Load data
        input_image = np.loadtxt(input_filename_fullpath)

        preprocessed_image = np.loadtxt(output_filename_fullpath)

        # Check if image is non-zero
        self.assertFalse(np.array_equal(preprocessed_image, np.zeros(preprocessed_image.shape)))

        # Check center of saved file
        calculated_center = find_center(preprocessed_image)
        center = (127.5,127.5)

        self.assertTrue(np.isclose(center, calculated_center, atol=2.0).all())

        # Check the rotation angle
        angle = preprocessor.find_eye_rotation_angle(input_image, calculated_center)

        self.assertTrue(np.isclose(angle, 135, atol=2.0))

    def test_preprocess_sample_924_remeasurements(self):
        """
        Ensure that rotation angle is close over 10 measurements
        """
        input_filepath_list = glob.glob(os.path.join(self.test_input_path, "*924.txt"))
        input_filepath_list.sort()
        params_filename = "params_centerize_rotate.txt"
        params_filepath = os.path.join(self.test_path, params_filename)

        with open(params_filepath, "r") as params_fp:
            params = json.loads(params_fp.read())

        centers = []
        angles = []
        for input_filepath in input_filepath_list:
            preprocessor = PreprocessData(filename=input_filepath,
                    input_path=self.test_input_path, output_path=self.test_output_path, params=params)
            preprocessor.preprocess()
            # Load the image file
            input_filename = os.path.basename(input_filepath)
            output_filename_fullpath = os.path.join(self.test_output_path,
                    "centered_rotated", "CR_" + input_filename)
            input_image = np.loadtxt(input_filepath)
            output_image = np.loadtxt(output_filename_fullpath)

            # Call this on input image, not output image
            centered_rotated_image, center, angle_degrees = preprocessor.centerize_and_rotate(input_image)

            centers.append(center)
            angles.append(angle_degrees)

        # Ensure that angles and centers are close to each other
        for idx in range(len(angles)):
            self.assertTrue(np.isclose(angles[idx], angles[0], rtol=0.05))
            self.assertTrue(np.isclose(centers[idx], centers[0], rtol=0.01).all())

    def test_preprocess_hot_spot_filter(self):
        params_hot_spot_filter = self.params_hot_spot_filter
        params_hot_spot_no_filter = self.params_hot_spot_no_filter
        test_input_path = self.test_input_path
        test_output_parent_path = self.test_output_path
        input_file_path_list = self.input_file_path_list

        # Set output paths
        test_output_filtered_path = os.path.join(test_output_parent_path, "filtered")
        test_output_unfiltered_path = os.path.join(test_output_parent_path, "unfiltered")

        # Create output paths
        os.makedirs(test_output_filtered_path, exist_ok=True)
        os.makedirs(test_output_unfiltered_path, exist_ok=True)

        input_filename = "AA00950.txt"
        input_filename_fullpath = os.path.join(test_input_path, input_filename)
        output_filtered_filename_fullpath = os.path.join(test_output_filtered_path,
                "original", "O_" + input_filename)
        output_unfiltered_filename_fullpath = os.path.join(test_output_unfiltered_path,
                "original", "O_" + input_filename)

        params_filter = json.loads(params_hot_spot_filter)
        params_no_filter = json.loads(params_hot_spot_no_filter)

        preprocessor_filter = PreprocessData(
                filename=input_filename,
                input_path=test_input_path,
                output_path=test_output_filtered_path,
                params=params_filter)
        preprocessor_no_filter = PreprocessData(
                filename=input_filename,
                input_path=test_input_path,
                output_path=test_output_unfiltered_path,
                params=params_no_filter)

        # Preprocess data, saving to a file
        preprocessor_filter.preprocess()
        preprocessor_no_filter.preprocess()

        # Load data
        input_image = np.loadtxt(input_filename_fullpath)

        preprocessed_filtered_image = np.loadtxt(output_filtered_filename_fullpath)
        preprocessed_unfiltered_image = np.loadtxt(output_unfiltered_filename_fullpath)

        # Check if image is non-zero
        self.assertFalse(
                np.array_equal(
                    preprocessed_filtered_image,
                    np.zeros(preprocessed_filtered_image.shape)))
        self.assertFalse(
                np.array_equal(
                    preprocessed_unfiltered_image,
                    np.zeros(preprocessed_unfiltered_image.shape)))

        # Check if images are the same
        self.assertFalse(
                np.array_equal(preprocessed_filtered_image, preprocessed_unfiltered_image))

        # Finding the hot spots
        threshold = 400
        filtered_hot_spot_coords = find_outlier_pixel_values(
                preprocessed_filtered_image, threshold=threshold, absolute=True)
        unfiltered_hot_spot_coords = find_outlier_pixel_values(
                preprocessed_unfiltered_image, threshold=threshold, absolute=True)

        self.assertTrue(unfiltered_hot_spot_coords.size > 0)
        self.assertTrue(filtered_hot_spot_coords.size == 0)


    def tearDown(self):
        # Delete the output folder
        shutil.rmtree(self.test_output_path)


class TestCenterFinding(unittest.TestCase):

    def test_find_centroid_single_point(self):
        """
        Calculate the centroid of a single point
        """
        # Create (1,2) array
        single_point = np.array([[4,5]])
        centroid = find_centroid(single_point)

        self.assertTrue(np.array_equal(single_point.flatten(), centroid))

    def test_find_centroid_multiple_points(self):
        """
        Calculate the centroid of a series of points
        """
        points = np.array([[2,4],[4,2]])
        known_centroid = (3,3)
        calculated_centroid = find_centroid(points)

        self.assertTrue(np.array_equal(known_centroid, calculated_centroid))

    def test_find_center_max_centroid(self):
        # Original filename: 20220330/A00041.txt
        test_filename = "test_preprocess_center.txt"
        test_dir = "test_preprocessing_images"
        test_image_path = os.path.join(TEST_IMAGE_PATH, test_dir, "input", test_filename)
        # Set known center using centroid of max pixels in beam region of interest
        known_center = (126.125, 132.375) # Using centroid of max pixels
        test_image = np.loadtxt(test_image_path)
        calculated_center = find_center(test_image, method="max_centroid")

        self.assertTrue(np.array_equal(calculated_center, known_center))

    def test_find_center_off_center(self):
        # Create a 4x4 image with a single on pixel, rest are off
        test_image = np.zeros((4,4))
        test_image[2,2] = 1

        known_center = (2,2)
        calculated_center = find_center(test_image, method="max_centroid")
        self.assertTrue(np.array_equal(calculated_center, known_center))

    def test_find_trunc_limit(self):
        intensities = gen_1d_intensity_profile()
        self.assertEqual('foo'.upper(), 'FOO')

    def test_circular_average_trivial_example(self):
        size = 8
        test_image = np.ones((size,size))
        center = test_image.shape[0]/2-0.5, test_image.shape[1]/2-0.5
        avg_image = circular_average(test_image,center)

        # Check that the circularly averaged image is close to identical
        self.assertTrue(np.isclose(test_image, avg_image).all())

    def test_circular_average(self):
        test_image = np.array([
            [4,6,6,4,],
            [5,10,10,5],
            [5,10,10,5],
            [4,6,6,4,],
            ])
        center = test_image.shape[0]/2-0.5, test_image.shape[1]/2-0.5
        avg_image = circular_average(test_image,center)

        # Check that the circular mean is close to the original mean
        self.assertTrue(np.isclose(np.mean(test_image), np.mean(avg_image)))


class TestUtils(unittest.TestCase):

    def test_interval_count_multiple_intervals(self):
        # Set up multiple interval example
        num_array = np.array([0,1,2,3,6,7,9,12,14,15,16])
        count = count_intervals(num_array)
        self.assertEqual(count,5)

    def test_empty_interval(self):
        num_array = np.array([])
        count = count_intervals(num_array)
        self.assertEqual(count,0)

    def test_single_element_interval(self):
        num_array = np.array([3])
        count = count_intervals(num_array)
        self.assertEqual(count,1)

    def test_get_angle_0_degrees(self):
        """
        Two features are on a horizontal line, so angle should be zero
        """
        known_angle = 0.0

        feature_1 = [10, 10]
        feature_2 = [10, 20]

        test_angle = get_angle(feature_1, feature_2)

        self.assertTrue(np.array_equal(known_angle, test_angle))

    def test_get_angle_90_degrees(self):
        """
        Two features are on a vertical line, so angle should be 90
        based on input order.
        """
        known_angle = 90.0

        feature_1 = [20, 10]
        feature_2 = [10, 10]

        test_angle = get_angle(feature_1, feature_2)

        self.assertTrue(np.array_equal(known_angle, test_angle))

    def test_get_angle_n90_degrees(self):
        """
        Two features are on a vertical line, so angle should be -90
        based on input order.
        """
        known_angle = -90.0

        feature_1 = [10, 10]
        feature_2 = [20, 10]

        test_angle = get_angle(feature_1, feature_2)

        self.assertTrue(np.array_equal(known_angle, test_angle))


class TestDenoising(unittest.TestCase):

    def test_filter_hot_spots_median_method(self):
        """
        Test hot spot filter using the median method
        """
        size = 256
        test_image = np.ones((size,size))
        hot_spot_coords = (20,40)
        hot_spot_value = 10
        test_image[hot_spot_coords] = hot_spot_value

        self.assertEqual(np.max(test_image), hot_spot_value)

        threshold = 5
        filtered_image = filter_outlier_pixel_values(
                test_image, threshold=threshold, absolute=True,
                fill_method="median")

        self.assertTrue(np.array_equal(filtered_image, np.ones((size,size))))

    def test_filter_hot_spots_invalid_method(self):
        """
        Test hot spot filter using an invalid method
        """
        size = 256
        filter_size = 5
        test_image = np.ones((size,size))
        hot_spot_coords = (20,40)
        hot_spot_value = 10
        test_image[hot_spot_coords] = hot_spot_value

        # Set the known result which has the 5x5 neighborhood of the hot spot
        # set to zero
        known_result = test_image.copy()
        hot_spot_roi_rows = slice(
                hot_spot_coords[0]-filter_size//2,
                hot_spot_coords[0]+filter_size//2+1)
        hot_spot_roi_cols = slice(
                hot_spot_coords[1]-filter_size//2,
                hot_spot_coords[1]+filter_size//2+1)
        known_result[hot_spot_roi_rows, hot_spot_roi_cols] = 0

        self.assertEqual(np.max(test_image), hot_spot_value)

        threshold = 5
        with self.assertRaises(ValueError):
            filtered_image = filter_outlier_pixel_values(
                    test_image, threshold, absolute=True,
                    filter_size=filter_size,
                    fill_method="invalid")

    def test_filter_hot_spots_zero_method(self):
        """
        Test hot spot filter using the zero method
        """
        size = 256
        filter_size = 5
        test_image = np.ones((size,size))
        hot_spot_coords = (20,40)
        hot_spot_value = 10
        test_image[hot_spot_coords] = hot_spot_value

        # Set the known result which has the 5x5 neighborhood of the hot spot
        # set to zero
        known_result = test_image.copy()
        hot_spot_roi_rows = slice(
                hot_spot_coords[0]-filter_size//2,
                hot_spot_coords[0]+filter_size//2+1)
        hot_spot_roi_cols = slice(
                hot_spot_coords[1]-filter_size//2,
                hot_spot_coords[1]+filter_size//2+1)
        known_result[hot_spot_roi_rows, hot_spot_roi_cols] = 0

        self.assertEqual(np.max(test_image), hot_spot_value)

        threshold = 5
        filtered_image = filter_outlier_pixel_values(
                test_image, threshold=threshold, absolute=True,
                filter_size=filter_size,
                fill_method="zero")

        self.assertFalse(np.array_equal(filtered_image, test_image))

        self.assertTrue(np.array_equal(filtered_image, known_result))

    def test_dead_pixel_repair_single_high_value_pixel_nan_copy(self):
        """
        Test dead pixel repair on an image with a single high value pixel
        away from the diffraction pattern center with beam radius = 0
        and no diffraction pattern specified.
        """
        size = 5
        test_image = np.ones((size,size))
        known_repaired_image = test_image.copy()

        # Create test image
        test_image[int(size/2),int(size/2)] = 1e6
        # Create known result
        known_repaired_image[int(size/2),int(size/2)] = np.nan

        # Repair test image
        repaired_image = filter_outlier_pixel_values(
                test_image,
                threshold=1e3,
                absolute=True,
                fill_method="nan",
                filter_size=1)

        # Check that repaired image is non-zero
        self.assertTrue(np.nansum(repaired_image) == size**2-1)

        # Check that repaired image matches known result
        self.assertTrue(
                np.array_equal(
                    repaired_image, known_repaired_image, equal_nan=True))

    def test_dead_pixel_repair_single_high_value_pixel_median_copy(self):
        """
        Test dead pixel repair on an image with a single high value pixel
        away from the diffraction pattern center with beam radius = 0
        and no diffraction pattern specified.
        """
        size = 5
        test_image = np.ones((size,size))
        known_repaired_image = test_image.copy()

        # Create test image
        test_image[int(size/2),int(size/2)] = 1e6
        # Create known result
        known_repaired_image = np.ones((size,size))

        # Repair test image
        repaired_image = filter_outlier_pixel_values(
                test_image,
                threshold=1e3,
                absolute=True,
                fill_method="median",
                filter_size=1)

        # Check that repaired image is non-zero
        self.assertTrue(np.sum(repaired_image) == size**2)

        # Check that repaired image matches known result
        self.assertTrue(
                np.array_equal(
                    repaired_image, known_repaired_image))

    def test_dead_pixel_repair_single_high_value_pixel_away_from_center(self):
        """
        Test dead pixel repair on an image with a single high value pixel
        away from the center.
        """
        size = 5
        test_image = np.ones((size,size))
        known_repaired_image = test_image.copy()

        # Create test image
        test_image[int(size/2),int(size/2)] = 1e6
        # Create known result
        known_repaired_image[int(size/2),int(size/2)] = np.nan

        # Repair test image
        repaired_image = filter_outlier_pixel_values(
                test_image,
                threshold=1e3,
                absolute=True,
                fill_method="nan",
                filter_size=1)

        # Check that repaired image is non-zero
        self.assertTrue(np.nansum(repaired_image) == size**2-1)

        # Check that repaired image matches known result
        self.assertTrue(
                np.array_equal(
                    repaired_image, known_repaired_image, equal_nan=True))

    def test_dead_pixel_repair_transform_single_high_value_pixel_nan_copy(self):
        """
        Test dead pixel repair on an image with a single high value pixel
        away from the diffraction pattern center with beam radius = 0
        and no diffraction pattern specified.
        """
        size = 5
        test_image = np.ones((size,size))
        known_repaired_image = test_image.copy()

        # Create test image
        test_image[int(size/2),int(size/2)] = 1e6
        # Create known result
        known_repaired_image[int(size/2),int(size/2)] = np.nan

        pixel_repair = FilterOutlierPixelValues(
                threshold=1e3, absolute=True, filter_size=1, fill_method="nan")

        # Reshape
        test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1]))
        known_repaired_image = known_repaired_image.reshape(
            (1, known_repaired_image.shape[0], known_repaired_image.shape[1]))

        # Repair test image
        repaired_image = pixel_repair.transform(
                test_image,
                copy=True)

        # Check that repaired image is non-zero
        self.assertTrue(np.nansum(repaired_image) == size**2-1)

        # Check that repaired image matches known result
        self.assertTrue(
                np.array_equal(
                    repaired_image, known_repaired_image, equal_nan=True))

    def test_dead_pixel_repair_transform_single_high_value_pixel_median_copy(self):
        """
        Test dead pixel repair on an image with a single high value pixel
        away from the diffraction pattern center with beam radius = 0
        and no diffraction pattern specified.
        """
        size = 5
        test_image = np.ones((size,size))

        # Create test image
        test_image[int(size/2),int(size/2)] = 1e6
        # Create known result
        known_repaired_image = np.ones((size,size))

        # Set repair parameters
        threshold = 1e3
        absolute = True
        filter_size = 1
        fill_method = "median"

        pixel_repair = FilterOutlierPixelValues(
                threshold=threshold, absolute=absolute, filter_size=filter_size,
                fill_method=fill_method)

        # Reshape
        test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1]))
        known_repaired_image = known_repaired_image.reshape(
            (1, known_repaired_image.shape[0], known_repaired_image.shape[1]))

        # Repair test image
        repaired_image_array = pixel_repair.transform(
                test_image,
                copy=True)

        repaired_image = repaired_image_array[0, ...]

        # Check that repaired image is non-zero
        self.assertTrue(np.sum(repaired_image) == size**2)

        # Check that repaired image matches known result
        self.assertTrue(
                np.allclose(
                    repaired_image, known_repaired_image))

    def test_dead_pixel_repair_transform_single_high_value_pixel_away_from_center(self):
        """
        Test dead pixel repair on an image with a single high value pixel
        away from the center.
        """
        size = 5
        test_image = np.ones((size,size))
        known_repaired_image = test_image.copy()

        # Create test image
        test_image[int(size/2),int(size/2)] = 1e6
        # Create known result
        known_repaired_image[int(size/2),int(size/2)] = np.nan

        pixel_repair = FilterOutlierPixelValues(
                threshold=1e3, absolute=True, filter_size=1, fill_method="nan")

        # Reshape
        test_image = test_image.reshape((1, test_image.shape[0], test_image.shape[1]))
        known_repaired_image = known_repaired_image.reshape(
            (1, known_repaired_image.shape[0], known_repaired_image.shape[1]))

        # Repair test image
        repaired_image = pixel_repair.transform(
                test_image,
                copy=True)

        # Check that repaired image is non-zero
        self.assertTrue(np.nansum(repaired_image) == size**2-1)

        # Check that repaired image matches known result
        self.assertTrue(
                np.array_equal(
                    repaired_image, known_repaired_image, equal_nan=True))

    def test_dead_pixel_repair_filter_outlier_pixel_values_dir(self):
        """
        """
        TEST_IMAGE_DIR = "test_images"
        TEST_DIR = "test_dead_pixel_repair_images"
        INPUT_DIR = "input"
        OUTPUT_DIR = "output"
        REPAIRED_DIR = "repaired"

        # Set input path
        input_path = os.path.join(TEST_IMAGE_PATH, TEST_DIR, INPUT_DIR)
        # Set output path
        output_path = os.path.join(TEST_IMAGE_PATH, TEST_DIR, OUTPUT_DIR)
        # Set known repaired path
        known_repaired_path = os.path.join(
                TEST_IMAGE_PATH, TEST_DIR, REPAIRED_DIR)

        # Create output path if it does not exist
        # os.makedirs(output_path, exist_ok=True)
        # print("Created {}".format(output_path))

        # Run dead pixel repair for directory
        filter_outlier_pixel_values_dir(
            input_path=input_path,
            output_path=output_path,
            filter_size=1,
            absolute=True,
            overwrite=False,
            threshold=1e3,
            verbose=False,
            fill_method="nan")

        # Get list of output file paths
        output_filepath_list = glob.glob(os.path.join(output_path, "*"))
        output_filepath_list.sort()

        # Get list of repaired file paths
        known_repaired_filepath_list = glob.glob(os.path.join(known_repaired_path, "*"))
        known_repaired_filepath_list.sort()

        # Check that there are some output files
        self.assertTrue(len(output_filepath_list) > 0)

        # Check the output files
        for idx in range(len(output_filepath_list)):
            output_filepath = output_filepath_list[idx]
            output_image = np.loadtxt(output_filepath)

            known_repaired_filepath = known_repaired_filepath_list[idx]
            known_repaired_image = np.loadtxt(known_repaired_filepath)

            # Ensure images are non-zero
            self.assertTrue(np.nansum(output_image) > 0)
            self.assertTrue(np.nansum(known_repaired_image) > 0)

            # Ensure output and known repaired image are equal
            self.assertTrue(
                    np.array_equal(
                        output_image, known_repaired_image, equal_nan=True))

        # Delete output files
        shutil.rmtree(output_path)


class TestImageProcessing(unittest.TestCase):

    def test_sensible_intensity_ranges(self):
        self.fail("Finish writing test")

    def test_local_threshold_determinism_separate_measurements(self):
        self.fail("Finish writing test")

    def test_pad_image_prerotation_method(self):
        """
        Test a few images with default prerotation method
        """
        dim = 1
        image = np.arange(dim**2).reshape((dim,dim))

        padded_image = pad_image(image, method="prerotation")

        self.assertEqual(padded_image.shape, (2,2))

        dim = 2
        image = np.arange(dim**2).reshape((dim,dim))

        padded_image = pad_image(image, method="prerotation")

        self.assertEqual(padded_image.shape, (6,6))

        dim = 3
        image = np.arange(dim**2).reshape((dim,dim))

        padded_image = pad_image(image, method="prerotation")

        self.assertEqual(padded_image.shape, (8,8))

        dim = 4
        image = np.arange(dim**2).reshape((dim,dim))

        padded_image = pad_image(image, method="prerotation")

        self.assertEqual(padded_image.shape, (12,12))

        dim = 10
        image = np.arange(dim**2).reshape((dim,dim))

        padded_image = pad_image(image, method="prerotation")

        self.assertEqual(padded_image.shape, (28,28))

    def test_unwarp_polar(self):
        """
        Test unwarp_polar function
        """
        # Create test polar image
        test_intensity_1d = np.zeros((1,100))
        test_intensity_1d[0,9] = 10
        test_image_polar = np.repeat(test_intensity_1d, 100, axis=0)

        output_shape=(256,256)

        test_image = unwarp_polar(test_image_polar.T, output_shape=output_shape)

        # Check that the test image is not all zeros
        self.assertTrue(np.any(test_image))

        test_image_warp_polar = warp_polar(test_image)

        # Test that maximum is at index 9
        # First, get the indices of the maximum locations along the columns
        max_indices_start_image = np.argmax(test_image_polar, axis=1)
        max_indices_final_image = np.argmax(test_image_warp_polar, axis=1)

        self.assertTrue(np.all(max_indices_start_image == 9))
        self.assertTrue(np.all(max_indices_final_image == 9))

    def test_unwarp_polar_scaled(self):
        """
        Test unwarp_polar function for scaling r by 2
        """
        # Create test polar image
        test_intensity_1d = np.zeros((1,100))
        test_intensity_1d[0,9] = 10
        test_image_polar = np.repeat(test_intensity_1d, 100, axis=0)

        output_shape=(256,256)

        test_image = unwarp_polar(test_image_polar.T,
                            output_shape=output_shape, rmax=200)
        test_image_warp_polar = warp_polar(test_image)

        # Test that maximum is at index 9
        # First, get the indices of the maximum locations along the columns
        max_indices_start_image = np.argmax(test_image_polar, axis=1)
        max_indices_final_image = np.argmax(test_image_warp_polar, axis=1)

        # The 10th value is the max in the start image
        self.assertTrue(np.all(max_indices_start_image == 9))
        # The 18th and 19th values in the final image should be near the max,
        # that is location 18.5, which rounds down to 18
        self.assertTrue(np.all(max_indices_final_image == np.round(18.5)))

    def test_unwarp_polar_small_even_example(self):
        """
        Test for a small even example
        """
        # Create test polar image
        test_intensity_1d = np.zeros((1,10))
        test_intensity_1d[0,3] = 10
        test_image_polar = np.repeat(test_intensity_1d, 10, axis=0)

        output_shape=(10,10)

        test_image = unwarp_polar(test_image_polar.T,
                            output_shape=output_shape, rmax=None)
        test_image_warp_polar = warp_polar(test_image, output_shape=output_shape)

        # Ensure that the cartesian center of mass is the center
        center = (output_shape[0]/2 - 0.5, output_shape[1]/2 - 0.5)
        calculated_center_of_mass = center_of_mass(test_image)
        self.assertTrue(np.array_equal(center, calculated_center_of_mass))

        # Ensure maximum cartesian values are close to 5 at the horizontal strip
        strip = test_image[test_image.shape[0]//2,:]
        max_value = np.max(strip)
        self.assertTrue(np.isclose(max_value, 5, atol=1))

        # Ensure that warping back to polar gives the maximum near r=3
        radial = np.sum(test_image_warp_polar, axis=0)
        radial_max = np.max(radial)
        radial_max_index = np.where(radial == radial_max)
        self.assertTrue(np.isclose(radial_max_index, 3, atol=1))

    def test_unwarp_polar_small_odd_example(self):
        """
        Test for a small odd example
        """
        # Create test polar image
        test_intensity_1d = np.zeros((1,9))
        test_intensity_1d[0,3] = 10
        test_image_polar = np.repeat(test_intensity_1d, 9, axis=0)

        output_shape=(9,9)

        test_image = unwarp_polar(test_image_polar.T,
                            output_shape=output_shape, rmax=None)
        test_image_warp_polar = warp_polar(test_image, output_shape=output_shape)

        # Ensure that the cartesian center of mass is the center
        center = (output_shape[0]/2 - 0.5, output_shape[1]/2 - 0.5)
        calculated_center_of_mass = center_of_mass(test_image)
        self.assertTrue(np.array_equal(center, calculated_center_of_mass))

        # Ensure maximum cartesian values are close to 5 at the horizontal strip
        strip = test_image[test_image.shape[0]//2,:]
        max_value = np.max(strip)
        self.assertTrue(np.isclose(max_value, 10, atol=1))

        # Ensure that warping back to polar gives the maximum near r=3
        radial = np.sum(test_image_warp_polar, axis=0)
        radial_max = np.max(radial)
        radial_max_index = np.where(radial == radial_max)
        self.assertTrue(np.isclose(radial_max_index, 3, atol=1))

    def test_crop_image_original_size(self):
        """
        Trivial test to ensure crop to original shape outputs
        the original image.
        """
        # Create even-shaped image
        side = 256
        even_image = np.arange(side**2).reshape(side,side)
        # Crop image
        cropped_even_image = crop_image(even_image, side, side, center=(side/2-0.5,side/2-0.5))

        # Check that original image and cropped image are identical
        self.assertTrue(np.array_equal(even_image, cropped_even_image))

    def test_crop_image_smaller_size(self):
        """
        Test to ensure cropped image output size is as intended
        """
        # Create even-shaped image
        side = 8
        even_image = np.arange(side**2).reshape(side,side)
        # Crop image
        cropped_even_image = crop_image(even_image, side//2, side//2)

        # Check the output shape of the image
        self.assertTrue(np.array_equal(cropped_even_image.shape, (side//2, side//2)))

        # Check the first value is correct
        extracted_first_value = cropped_even_image[0,0]
        known_first_value = even_image[side//4, side//4]
        self.assertEqual(extracted_first_value, known_first_value)

    def test_crop_image_smaller_size_off_center(self):
        """
        Test to ensure cropped image output size is as intended
        """
        # Create even-shaped image
        side = 8
        even_image = np.arange(side**2).reshape(side,side)
        # Crop image
        center = (2.5,2.5)
        cropped_even_image = crop_image(even_image, side//2, side//2, center=center)

        # Check the output shape of the image
        self.assertTrue(np.array_equal(cropped_even_image.shape, (side//2, side//2)))

        # Check the first value is correct
        extracted_first_value = cropped_even_image[0,0]
        known_first_value = even_image[int(side//4-center[0]/2+0.5),
                                        int(side//4-center[1]/2+0.5)]
        self.assertEqual(extracted_first_value, known_first_value)

    def test_crop_image_odd_size(self):
        # Create odd-shaped image
        side = 9
        odd_image = np.arange(side**2).reshape(side,side)

        with self.assertRaises(NotImplementedError):
            # Crop image
            cropped_odd_image = crop_image(odd_image, side//2, side//2)

    def test_bright_pixel_count(self):
        """
        Test that bright pixel count returns the correct
        number of pixels for each color
        """
        test_image = np.array([
            [1,1,1,4],
            [1,1,1,4],
            [1,1,1,4],
            [1,1,1,4],
            ])

        known_black_count = 12
        known_yellow_count = 4

        calculated_black_count = bright_pixel_count(test_image, qmax=0.5)
        calculated_yellow_count = bright_pixel_count(test_image, qmin=0.5)

        self.assertEqual(calculated_black_count, known_black_count)
        self.assertEqual(calculated_yellow_count, known_yellow_count)


class TestPeakFinding(unittest.TestCase):

    def test_gaussian_2d_peak_finding_single_max_value(self):
        """
        Test gaussian peak finding for a matrix with 1 at the center, rest 0
        """
        # Set up the test image
        test_image = np.array([
            [0,0,0,0,0,],
            [0,0,0,0,0,],
            [0,0,1,0,0,],
            [0,0,0,0,0,],
            [0,0,0,0,0,],
            ])

        known_peak_location = (np.array(test_image.shape)[:2]/2) - 0.5

        # Find the peak location
        window_size = 3
        test_peak_location = find_2d_peak(test_image, window_size=window_size)

        # Check if peak location is correct
        self.assertTrue(np.array_equal(known_peak_location, test_peak_location))

    def test_gaussian_2d_peak_finding_double_max_value(self):
        """
        Test gaussian peak finding for a matrix with two 1s
        """
        # Set up the test image
        test_image = np.array([
            [0,0,0,0,0,],
            [0,1,0,0,0,],
            [0,0,0,0,0,],
            [0,0,0,1,0,],
            [0,0,0,0,0,],
            ])

        known_peak_location = (np.array(test_image.shape)[:2]/2) - 0.5

        # Find the peak location
        window_size = 3
        test_peak_location = find_2d_peak(test_image, window_size=window_size)

        # Check if peak location is correct
        self.assertTrue(np.array_equal(known_peak_location, test_peak_location))

    def test_gaussian_2d_peak_finding_noisy_example(self):
        # Set up the test image with a noisy peak at the center
        test_image = np.array([
            [0,0,0,0,0,],
            [1,1,0,1,0,],
            [1,9,6,7,0,],
            [0,1,1,1,0,],
            [0,0,0,0,0,],
            ])

        known_peak_location = (np.array(test_image.shape)[:2]/2) - 0.5

        # Find the peak location
        window_size = 3
        test_peak_location = find_2d_peak(test_image, window_size=window_size)

        # Check if peak location is correct
        self.assertTrue(np.array_equal(known_peak_location, test_peak_location))

    def test_gaussian_1d_peak_finding_single_max_value(self):
        """
        Test gaussian peak finding for an array with 1 at the center, rest 0
        """
        # Set up the test array
        test_array = np.array([
            0,0,1,0,0,],)

        known_peak_location = (test_array.size/2) - 0.5

        # Find the peak location
        window_size = 3
        test_peak_locations = find_1d_peaks(test_array, window_size=window_size)

        # Check that only one peak location is found
        self.assertEqual(test_peak_locations.size, 1)

        # Check if peak location is correct
        self.assertTrue(np.array(known_peak_location == test_peak_locations).all())

    def test_gaussian_1d_peak_finding_double_max_value(self):
        """
        Test gaussian peak finding for an array with two 1s
        """
        # Set up the test array
        test_array = np.array([
            0,1,0,1,0,],)

        known_peak_location = (test_array.size/2) - 0.5

        # Find the peak location
        window_size = 3
        test_peak_locations = find_1d_peaks(test_array, window_size=window_size)

        # Check that only one peak location is found
        self.assertEqual(test_peak_locations.size, 1)

        # Check if peak location is correct
        self.assertTrue(np.array(known_peak_location == test_peak_locations).all())

    def test_gaussian_1d_peak_finding_noisy_example(self):
        # Set up the test array with a noisy peak at the center
        test_array = np.array([
            1,9,6,7,0,],)

        known_peak_location = (test_array.size/2) - 0.5

        # Find the peak location
        window_size = 3
        test_peak_locations = find_1d_peaks(test_array, window_size=window_size)

        # Check that only one peak location is found
        self.assertEqual(test_peak_locations.size, 1)

        # Check if peak location is correct
        self.assertTrue(np.array(known_peak_location == test_peak_locations).all())

    def test_gaussian_1d_peak_finding_two_peaks(self):
        """
        Test gaussian peak finding for an array with two 1s
        """
        # Set up the test array
        test_array = np.zeros((10,1))
        test_array[1] = 1
        test_array[8] = 1
        known_peak_locations = [1,8]

        # Find the peak location
        window_size = 3
        test_peak_locations = find_1d_peaks(test_array, window_size=window_size)

        # Check that two peak locations are found
        self.assertEqual(test_peak_locations.size, 2)

        # Check if peak location is correct
        self.assertTrue(np.array(known_peak_locations == test_peak_locations).all())


class TestOutputSaturationBugFix(unittest.TestCase):
    """
    Issue #64:
    Preprocessing sometimes outputs images with values all 0 or 2147483648
    """

    def setUp(self):
        """
        Set up some paths
        """
        test_dirname = "test_output_saturation_images"
        test_parent_path = os.path.join(TEST_IMAGE_PATH, test_dirname)

        samples_dir = "samples"
        samples_path = os.path.join(test_parent_path, samples_dir)

        output_dir = "output"
        output_path = os.path.join(test_parent_path, output_dir)
        # Create the output directory
        os.makedirs(output_path, exist_ok=True)

        saturated_dir = "saturated_preprocessed_samples"
        saturated_path = os.path.join(test_parent_path, saturated_dir)

        control_dir = "controls"
        control_path = os.path.join(test_parent_path, control_dir)

        saturation_value = 2147483648

        self.test_parent_path = test_parent_path
        self.samples_path = samples_path
        self.output_path = output_path
        self.saturated_path = saturated_path
        self.control_path = control_path
        self.saturation_value = saturation_value

    def test_output_saturation_occurs_with_known_problem_samples(self):
        """
        Ensure preprocessed images do not saturate
        """
        samples_path = self.samples_path
        saturated_path = self.saturated_path
        output_path = self.output_path
        saturation_value = self.saturation_value

        # Check that saturation indeed occured previously
        saturated_filepath_list = glob.glob(
                                    os.path.join(saturated_path, "*.txt"))

        # Check that files list is not empty
        self.assertTrue(saturated_filepath_list)

        for saturated_filepath in saturated_filepath_list:
            data = np.loadtxt(saturated_filepath)
            unique = np.unique(data)
            # Ensure that we get only two values
            self.assertEqual(unique.size, 2)
            # Ensure that the first ordered value is 0
            self.assertEqual(unique[0], 0)
            # Ensure that the second ordered value is saturation_value
            self.assertEqual(unique[1], saturation_value)

    def test_no_output_saturation_control_samples(self):
        """
        Test that output saturation does not occur for known control samples
        """
        test_parent_path = self.test_parent_path
        control_path = self.control_path
        output_path = self.output_path
        saturation_value = self.saturation_value

        # Control
        control_name = "A00001.txt"

        # Set up parameters and plans
        params_file = "params.txt"
        params_path = os.path.join(test_parent_path, params_file)
        with open(params_path, "r") as params_fp:
            params = json.loads(params_fp.read())

        plans = ["centerize_rotate_quad_fold"]
        output_style = INVERSE_OUTPUT_MAP[plans[0]]
        plan_output_path = os.path.join(output_path, output_style)

        # Run preprocessing
        preprocessor = PreprocessData(
                        input_dir=control_path, output_dir=output_path, params=params)
        preprocessor.preprocess(plans=plans)

        # Now ensure that preprocessed output file is not saturated
        output_style_abbreviation = ABBREVIATIONS.get(output_style)
        output_filepath = os.path.join(
                plan_output_path, "{}_{}".format(output_style_abbreviation, control_name))
        data = np.loadtxt(output_filepath)
        unique = np.unique(data)
        # Ensure that we get only two values
        self.assertGreater(unique.size, 2)
        # Ensure that the saturation_value is not in the file
        self.assertNotIn(saturation_value, unique)

    def test_output_saturation_bugfix(self):
        """
        Run preprocessing on output_dir and ensure saturation does not occur
        """
        # Set to allow RuntimeWarning to raise an error
        np.seterr(all='raise')

        # Set up variables
        test_parent_path = self.test_parent_path
        input_path = self.samples_path
        output_path = self.output_path
        saturation_value = self.saturation_value

        # Set up parameters and plans
        params_file = "params.txt"
        params_path = os.path.join(test_parent_path, params_file)
        with open(params_path, "r") as params_fp:
            params = json.loads(params_fp.read())

        plans = ["centerize_rotate_quad_fold"]
        output_style = INVERSE_OUTPUT_MAP.get(plans[0])
        plan_output_path = os.path.join(output_path, output_style)
        plan_abbr = ABBREVIATIONS.get(output_style)

        # Run preprocessing
        preprocessor = PreprocessData(
                        input_dir=input_path, output_dir=output_path, params=params)
        preprocessor.preprocess(plans=plans)

        # Now ensure that preprocessed files are not saturated
        output_filepath_list = glob.glob(
                                    os.path.join(plan_output_path, "{}*.txt".format(plan_abbr)))

        # Ensure that output_filepath_list is not empty
        self.assertTrue(output_filepath_list)

        for output_filepath in output_filepath_list:
            data = np.loadtxt(output_filepath)
            unique = np.unique(data)
            # Ensure that we get only two values
            self.assertGreater(unique.size, 2)
            # Ensure that the saturation_value is not in the file
            self.assertNotIn(saturation_value, unique)

    def tearDown(self):
        # Delete the output folder
        shutil.rmtree(self.output_path)


class TestAngleFinding(unittest.TestCase):

    def setUp(self):
        # Set test path
        test_path = os.path.join(TEST_IMAGE_PATH, "test_angle_finding")
        self.test_path = test_path

        # Create test images
        INPUT_DIR="input"

        # Set the input and output paths
        test_input_path = os.path.join(test_path, "input")

        self.test_input_path = test_input_path

    def test_find_rotation_angle_synthetic_rotated(self):
        test_input_path = self.test_input_path

        input_filename = "rotated_synthetic.txt"
        input_filename_fullpath = os.path.join(test_input_path, input_filename)

        # Load data
        test_image = np.loadtxt(input_filename_fullpath)

        r = int(feature_pixel_location(9.8e-10))
        width = 10
        height = 10

        angle = find_rotation_angle(test_image, r, width, height)

        # Check the rotation angle
        self.assertTrue(np.isclose(angle, 45, atol=2.0))

    def test_find_rotation_angle_sample_924_remeasurements(self):
        """
        Ensure that rotation angle is close over 10 measurements
        """
        test_input_path = self.test_input_path

        # Load centered images
        input_filepath_list = glob.glob(os.path.join(test_input_path, "C_924*.txt"))
        input_filepath_list.sort()

        r = int(feature_pixel_location(9.8e-10))
        width = 10
        height = 20

        angles = []
        for input_filepath in input_filepath_list:
            # Load the image file
            input_filename = os.path.basename(input_filepath)
            image = np.loadtxt(input_filepath, dtype=np.uint32)

            # Calculate the rotation angle
            angle_degrees = find_rotation_angle(image, r, width, height)
            angles.append(angle_degrees)

        # Ensure that angles are close to each other and that the calculated
        # angles are between 45 and 90 degrees
        for idx in range(len(angles)):
            self.assertTrue(np.isclose(angles[idx], np.mean(angles), atol=10))
            self.assertTrue((angles[idx] < 90) & (angles[idx] > 45))


class TestBeamUtils(unittest.TestCase):

    def setUp(self):
        # Set test path
        test_path = os.path.join(TEST_IMAGE_PATH, "test_beam_utils")
        self.test_path = test_path

        # Create test images
        INPUT_DIR="input"

        # Set the input and output paths
        test_input_path = os.path.join(test_path, "input")

        self.test_input_path = test_input_path


    def test_beam_radius_924_measurements(self):
        """
        Test dynamic beam detection on remeasurements data
        """
        test_input_path = self.test_input_path

        # Load images
        input_filepath_list = glob.glob(
                os.path.join(test_input_path, "C_924*.txt"))
        input_filepath_list.sort()

        known_radius = 21

        for input_filepath in input_filepath_list:
            # Load the image file
            input_filename = os.path.basename(input_filepath)
            image = np.loadtxt(input_filepath, dtype=np.uint32)

            # Calculate beam radius
            calculated_radius = beam_radius(image)

            # Check if the beam radius is accurate to within 1 pixel
            self.assertTrue(np.isclose(calculated_radius, known_radius, atol=1))


    def test_first_valley_location(self):
        """
        Test first valley location function against a test image with known
        first valley location.
        """
        size = 256
        test_image_radius_20 = np.ones((size, size))
        array_center = np.array(test_image_radius_20.shape)/2 - 0.5

        x = np.linspace(-array_center[1], array_center[1], num=size)
        y = np.linspace(-array_center[0], array_center[0], num=size)

        YY, XX = np.meshgrid(y, x)
        RR = np.sqrt(YY**2 + XX**2)

        known_first_valley_location = 20
        test_image_radius_20[
                (RR < (known_first_valley_location + 0.5)) & \
                        (RR > (known_first_valley_location - 0.5))] = 0
        calculated_first_valley_location, profile_1d = first_valley_location(
                test_image_radius_20)

        self.assertTrue(
                np.isclose(
                    calculated_first_valley_location, known_first_valley_location))

    def test_first_valley_location_no_valley(self):
        """
        Test first valley location function against a test image with no
        vallies.
        """
        size = 256
        array_center = np.array([size]*2)/2 - 0.5

        x = np.linspace(-array_center[1], array_center[1], num=size)
        y = np.linspace(-array_center[0], array_center[0], num=size)

        YY, XX = np.meshgrid(y, x)
        RR = np.sqrt(YY**2 + XX**2)

        test_image = np.exp(-(RR**2))

        first_valley, _ = first_valley_location(test_image)

        # Ensure the first valley is ``None``
        self.assertIsNone(first_valley)

    def test_beam_extent(self):
        """
        Test beam extent function against a test image with a known valley and
        inflection point.
        """
        size = 256
        array_center = np.array([size]*2)/2 - 0.5

        x = np.linspace(-array_center[1], array_center[1], num=size)
        y = np.linspace(-array_center[0], array_center[0], num=size)

        YY, XX = np.meshgrid(y, x)
        RR = np.sqrt(YY**2 + XX**2)

        # Set the test function as sin(r)^2 with some scaling
        test_image = np.sin(np.pi*RR/20)**2

        # The period is 40
        # The first valley is at 20
        # The subsequent inflection point is at 25
        known_inflection_point = 25

        inflection_point, _, _ = beam_extent(test_image)

        self.assertEqual(inflection_point, known_inflection_point)

    def test_azimuthal_integration_linear_radial_function(self):
        """
        """

        self.fail("Finish writing test.")

    def test_azimuthal_integration_constant_function(self):
        """
        Ensure that a constant image function results in a constant
        azimuthal profile.
        """

        size = 256
        shape = (size, size)
        array_center = np.array([size]*2)/2 - 0.5

        # Set the test function as constant value 1
        test_image = np.ones(shape)

        radial_profile = azimuthal_integration(test_image)

        known_profile = np.ones(radial_profile.shape)

        self.assertTrue(np.array_equal(radial_profile, known_profile))

class TestFeatureExtraction(unittest.TestCase):
    """
    FeatureExtraction tests
    """

    def test_feature_extraction_init(self):
        """
        Test FeatureExtraction for a test image of ones.
        Test passes if it no error raised.
        """
        size = 256
        shape = size, size
        test_image = np.ones(shape)

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

    def test_feature_extraction_image_intensity_ones(self):
        """
        Test FeatureExtraction for a test image of ones.
        Ensure the calculated image intensity is correct.
        """
        size = 256
        shape = size, size
        test_image = np.ones(shape)
        # Calculate the known intensity
        known_intensity = size*size

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the intensit
        calculated_intensity = feature_extraction.feature_image_intensity()

        # Ensure the calculated intensity is correct
        self.assertEqual(calculated_intensity, known_intensity)

    def test_feature_extraction_image_intensity_zeros(self):
        """
        Test FeatureExtraction for a test image of zeros.
        Ensure the calculated image intensity is correct.
        """
        size = 256
        shape = size, size
        test_image = np.zeros(shape)
        # Calculate the known intensity
        known_intensity = 0

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the intensit
        calculated_intensity = feature_extraction.feature_image_intensity()

        # Ensure the calculated intensity is correct
        self.assertEqual(calculated_intensity, known_intensity)

    def test_feature_extraction_percentile_intensity_ones(self):
        """
        Test FeatureExtraction for a test image of ones.
        Ensure the calculated percentile intensity is correct.
        """
        size = 256
        shape = size, size
        test_image = np.ones(shape)
        percentile = 99.99
        # Calculate the known intensity
        known_intensity = 1

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the intensit
        calculated_intensity = \
                feature_extraction.feature_percentile_intensity(percentile)

        # Ensure the calculated intensity is correct
        self.assertEqual(calculated_intensity, known_intensity)

    def test_feature_extraction_percentile_intensity_zeros(self):
        """
        Test FeatureExtraction for a test image of zeros.
        Ensure the calculated percentile intensity is correct.
        """
        size = 256
        shape = size, size
        test_image = np.zeros(shape)
        percentile = 99.99
        # Calculate the known intensity
        known_intensity = 0

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the intensit
        calculated_intensity = \
                feature_extraction.feature_percentile_intensity(percentile)

        # Ensure the calculated intensity is correct
        self.assertEqual(calculated_intensity, known_intensity)

    def test_feature_extraction_percentile_intensity_range(self):
        """
        Test FeatureExtraction for a test image of range.
        Ensure the calculated percentile intensity is correct.
        """
        size = 256
        shape = size, size
        test_image = np.arange(size*size).reshape(shape)
        percentile = 99.99
        # Calculate the known intensity
        known_intensity = percentile*(size*size-1)/100

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the intensit
        calculated_intensity = \
                feature_extraction.feature_percentile_intensity(percentile)

        # Ensure the calculated intensity is correct
        self.assertEqual(calculated_intensity, known_intensity)

    def test_feature_extraction_annulus_intensity_ones(self):
        """
        Test FeatureExtraction for a test image with an annulus of ones.
        Ensure the calculated annulus intensity is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set annulus properties
        rmin = size/4
        rmax = size/2

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=rmin, rmax=rmax)

        # Set the annulus values equal to 1
        test_image[annulus_mask] = 1

        # Calculate the known intensity based on area
        area = np.pi*(rmax**2 - rmin**2)
        known_intensity = area

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the annulus intensity
        calculated_intensity = feature_extraction.feature_annulus_intensity(
                rmin=rmin, rmax=rmax)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_extraction_annulus_intensity_zeros(self):
        """
        Test FeatureExtraction for a test image with an annulus of zeros.
        Ensure the calculated annulus intensity is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set annulus properties
        rmin = size/4
        rmax = size/2

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=rmin, rmax=rmax)

        # Set the known intensity to zero
        known_intensity = 0

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the annulus intensity
        calculated_intensity = feature_extraction.feature_annulus_intensity(
                rmin=rmin, rmax=rmax)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_extraction_annulus_intensity_angstrom_zeros(self):
        """
        Test FeatureExtraction for a test image with an annulus of zeros.
        Ensure the calculated annulus intensity is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set annulus properties
        amin = 8.8e-10
        amax = 10.8e-10

        # Convert from molecular spacings in angstroms to pixel lengths in
        # detector space (recpiprocal units)

        # Set machine parameters
        source_wavelength = 1.5418e-10
        pixel_length = 55e-6
        sample_to_detector_distance = 10e-3

        # Initialize the units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate rmin and rmax
        rmin = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amax)
        rmax = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amin)

        # Set the known intensity to zero
        known_intensity = 0

        # Initiate the class
        feature_extraction = FeatureExtraction(
                test_image, source_wavelength=source_wavelength,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate the annulus intensity
        calculated_intensity = \
                feature_extraction.feature_annulus_intensity_angstroms(
                        pixel_length=pixel_length, amin=amin, amax=amax)

        # Ensure the calculated intensity is correct
        self.assertEqual(calculated_intensity, known_intensity)

    def test_feature_extraction_annulus_intensity_angstrom_ones(self):
        """
        Test FeatureExtraction for a test image with an annulus of ones.
        Ensure the calculated annulus intensity is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set annulus properties
        amin = 8.8e-10
        amax = 10.8e-10

        # Convert from molecular spacings in angstroms to pixel lengths in
        # detector space (recpiprocal units)

        # Set machine parameters
        source_wavelength = 1.5418e-10
        pixel_length = 55e-6
        sample_to_detector_distance = 10e-3

        # Initialize the units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate rmin and rmax
        rmin = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amax)
        rmax = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amin)

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=rmin, rmax=rmax)

        # Set the annulus values equal to 1
        test_image[annulus_mask] = 1

        # Calculate the known intensity based on area
        area = np.pi*(rmax**2 - rmin**2)
        known_intensity = area


        # Initiate the class
        feature_extraction = FeatureExtraction(
                test_image, source_wavelength=source_wavelength,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate the annulus intensity
        calculated_intensity = \
                feature_extraction.feature_annulus_intensity_angstroms(
                        pixel_length=pixel_length, amin=amin, amax=amax)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_sector_intensity_ones_90_degrees(self):
        """
        Test feature sector intensity with all ones
        """
        # Set test image properties
        size = 256
        shape = size, size

        # Set annulus properties
        rmin = size/4
        rmax = size/2
        theta_min = -np.pi/4
        theta_max = np.pi/4

        # Generate sector mask
        # Create a mask for the annulus
        annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=rmin, rmax=rmax)

        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Get sector indices
        sector_indices = (TT > theta_min) & (TT < theta_max) & annulus_mask

        # Create test image
        test_image = np.zeros(shape)
        # Set sector pixels to one
        test_image[sector_indices] = 1

        # Calculate the known intensity based on area
        area = np.pi*(rmax**2 - rmin**2)*(theta_max - theta_min)/(2*np.pi)
        known_intensity = area

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the annulus intensity
        calculated_intensity = \
                feature_extraction.feature_sector_intensity(
                        rmin=rmin, rmax=rmax, theta_min=theta_min,
                        theta_max=theta_max)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_sector_intensity_ones_90_degrees_negative_x(self):
        """
        Test feature sector intensity with all ones with 90 degree sector
        symmetric about the x < 0 axis. Should raise ValueError.
        """
        # Set test image properties
        size = 256
        shape = size, size

        # Set annulus properties
        rmin = size/4
        rmax = size/2
        theta_min = 3*np.pi/4
        theta_max = 5*np.pi/4

        # Generate sector mask
        # Create a mask for the annulus
        annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=rmin, rmax=rmax)

        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Get sector indices
        sector_indices = (TT > theta_min) & (TT < theta_max) & annulus_mask

        # Create test image
        test_image = np.zeros(shape)
        # Set sector pixels to one
        test_image[sector_indices] = 1

        # Calculate the known intensity based on area
        area = np.pi*(rmax**2 - rmin**2)*(theta_max - theta_min)/(2*np.pi)
        known_intensity = area

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the annulus intensity
        self.assertRaises(ValueError,
                feature_extraction.feature_sector_intensity,
                        rmin=rmin, rmax=rmax, theta_min=theta_min,
                        theta_max=theta_max)

    def test_feature_sector_intensity_angstrom_ones(self):
        """
        Test FeatureExtraction for a test image with an annulus of ones.
        Ensure the calculated annulus intensity is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size

        # Set annulus properties
        amin = 8.8e-10
        amax = 10.8e-10
        theta_min = -np.pi/4
        theta_max = np.pi/4

        # Convert from molecular spacings in angstroms to pixel lengths in
        # detector space (recpiprocal units)

        # Set machine parameters
        source_wavelength = 1.5418e-10
        pixel_length = 55e-6
        sample_to_detector_distance = 10e-3

        # Initialize the units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate rmin and rmax
        rmin = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amax)
        rmax = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amin)

        # Generate sector mask
        # Create a mask for the annulus
        annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=rmin, rmax=rmax)

        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Get sector indices
        sector_indices = (TT > theta_min) & (TT < theta_max) & annulus_mask

        # Create test image
        test_image = np.zeros(shape)
        # Set sector pixels to one
        test_image[sector_indices] = 1

        # Calculate the known intensity based on area
        area = np.pi*(rmax**2 - rmin**2)*(theta_max - theta_min)/(2*np.pi)
        known_intensity = area

        # Initiate the class
        feature_extraction = FeatureExtraction(
                test_image, pixel_length=pixel_length,
                source_wavelength=source_wavelength,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate the annulus intensity
        calculated_intensity = \
                feature_extraction.feature_sector_intensity(
                        amin=amin, amax=amax, theta_min=theta_min,
                        theta_max=theta_max)

        # Ensure calculated intensity is non-zero
        self.assertTrue(calculated_intensity > 0)

        # Ensure known intensity is non-zero
        self.assertTrue(known_intensity > 0)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_sector_intensity_equator_pair_ones(self):
        """
        Test FeatureExtraction for a test image with a pair of equator sectors
        of ones.
        Ensure the calculated intensity of the sector pairs is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set equator sector pair properties
        rmin = size/4
        rmax = size/2
        sector_angle = np.pi/4

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=rmin, rmax=rmax)

        # Set the sector pixel values to 1
        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Calculate sector start and end angles based on symmetric sector angle
        theta_min = -sector_angle/2
        theta_max = sector_angle/2

        # Get right sector indices
        right_sector_indices = \
                (TT > theta_min) & (TT < theta_max) & annulus_mask
        # Get left sector indices based on left-right symmetry
        left_sector_indices = np.fliplr(right_sector_indices)

        test_image[right_sector_indices] = 1
        test_image[left_sector_indices] = 1

        # Calculate the known intensity based on sector areas
        area_right_sector = np.pi*(rmax**2-rmin**2)*sector_angle/(2*np.pi)
        area_left_sector = area_right_sector
        known_intensity = area_right_sector + area_left_sector

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the annulus intensity
        calculated_intensity = feature_extraction.feature_sector_intensity_equator_pair(
                rmin=rmin, rmax=rmax, sector_angle=sector_angle)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_sector_intensity_equator_pair_zeros(self):
        """
        Test FeatureExtraction for a test image with a pair of equator sectors
        of zeros.
        Ensure the calculated intensity of the sector pairs is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set equator sector pair properties
        rmin = size/4
        rmax = size/2
        sector_angle = np.pi/4

        # Set the known intensity to zero
        known_intensity = 0

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the annulus intensity
        calculated_intensity = feature_extraction.feature_sector_intensity_equator_pair(
                rmin=rmin, rmax=rmax, sector_angle=sector_angle)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_sector_intensity_meridian_pair_ones(self):
        """
        Test FeatureExtraction for a test image with a pair of meridian sectors
        of ones.
        Ensure the calculated intensity of the sector pairs is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set equator sector pair properties
        rmin = size/4
        rmax = size/2
        sector_angle = np.pi/4

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=rmin, rmax=rmax)

        # Set the sector pixel values to 1
        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Calculate sector start and end angles based on symmetric sector angle
        theta_min = -sector_angle/2 + np.pi/2
        theta_max = sector_angle/2 + np.pi/2

        # Get top sector indices
        top_sector_indices = \
                (TT > theta_min) & (TT < theta_max) & annulus_mask
        # Get bottom sector indices based on bottom-top symmetry
        bottom_sector_indices = np.flipud(top_sector_indices)

        test_image[top_sector_indices] = 1
        test_image[bottom_sector_indices] = 1

        # Calculate the known intensity based on sector areas
        area_top_sector = np.pi*(rmax**2-rmin**2)*sector_angle/(2*np.pi)
        area_bottom_sector = area_top_sector
        known_intensity = area_top_sector + area_bottom_sector

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the annulus intensity
        calculated_intensity = feature_extraction.feature_sector_intensity_meridian_pair(
                rmin=rmin, rmax=rmax, sector_angle=sector_angle)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_sector_intensity_meridian_pair_zeros(self):
        """
        Test FeatureExtraction for a test image with a pair of meridian sectors
        of zeros.
        Ensure the calculated intensity of the sector pairs is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set equator sector pair properties
        rmin = size/4
        rmax = size/2
        sector_angle = np.pi/4

        # Set the known intensity
        known_intensity = 0

        # Initiate the class
        feature_extraction = FeatureExtraction(test_image)

        # Calculate the annulus intensity
        calculated_intensity = feature_extraction.feature_sector_intensity_meridian_pair(
                rmin=rmin, rmax=rmax, sector_angle=sector_angle)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_sector_intensity_equator_pair_angstrom_ones(self):
        """
        Test FeatureExtraction for a test image with a pair of equator sectors
        of ones, given sector bounds in angstroms.
        Ensure the calculated intensity of the sector pairs is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set equator sector pair properties
        amin = 8.8e-10
        amax = 10.8e-10
        sector_angle = np.pi/4

        # Set machine parameters
        source_wavelength = 1.5418e-10
        pixel_length = 55e-6
        sample_to_detector_distance = 10e-3

        # Initialize the units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate rmin and rmax
        rmin = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amax)
        rmax = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amin)

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=rmin, rmax=rmax)

        # Set the sector pixel values to 1
        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Calculate sector start and end angles based on symmetric sector angle
        theta_min = -sector_angle/2
        theta_max = sector_angle/2

        # Get right sector indices
        right_sector_indices = \
                (TT > theta_min) & (TT < theta_max) & annulus_mask
        # Get left sector indices based on left-right symmetry
        left_sector_indices = np.fliplr(right_sector_indices)

        test_image[right_sector_indices] = 1
        test_image[left_sector_indices] = 1

        # Calculate the known intensity based on sector areas
        area_right_sector = np.pi*(rmax**2-rmin**2)*sector_angle/(2*np.pi)
        area_left_sector = area_right_sector
        known_intensity = area_right_sector + area_left_sector

        # Initiate the class
        feature_extraction = FeatureExtraction(
                test_image, pixel_length=pixel_length,
                source_wavelength=source_wavelength,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate the annulus intensity
        calculated_intensity = feature_extraction.feature_sector_intensity_equator_pair_angstroms(
                pixel_length=pixel_length, amin=amin, amax=amax, sector_angle=sector_angle)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_sector_intensity_equator_pair_angstrom_zeros(self):
        """
        Test FeatureExtraction for a test image with a pair of equator sectors
        of zeros, given sector bounds in angstroms.
        Ensure the calculated intensity of the sector pairs is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set equator sector pair properties
        amin = 8.8e-10
        amax = 10.8e-10
        sector_angle = np.pi/4

        # Set machine parameters
        source_wavelength = 1.5418e-10
        pixel_length = 55e-6
        sample_to_detector_distance = 10e-3

        # Set the known intensity to zero
        known_intensity = 0

        # Initiate the class
        feature_extraction = FeatureExtraction(
                test_image, pixel_length=pixel_length,
                source_wavelength=source_wavelength,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate the annulus intensity
        calculated_intensity = feature_extraction.feature_sector_intensity_equator_pair_angstroms(
                pixel_length=pixel_length, amin=amin, amax=amax, sector_angle=sector_angle)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_sector_intensity_meridian_pair_angstrom_ones(self):
        """
        Test FeatureExtraction for a test image with a pair of meridian sectors
        of ones, given sector bounds in angstroms.
        Ensure the calculated intensity of the sector pairs is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set equator sector pair properties
        amin = 8.8e-10
        amax = 10.8e-10
        sector_angle = np.pi/4

        # Set machine parameters
        source_wavelength = 1.5418e-10
        pixel_length = 55e-6
        sample_to_detector_distance = 10e-3

        # Initialize the units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate rmin and rmax
        rmin = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amax)
        rmax = units_class.bragg_peak_pixel_location_from_molecular_spacing(
                amin)

        # Create a mask for the annulus
        annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=rmin, rmax=rmax)

        # Set the sector pixel values to 1
        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Calculate sector start and end angles based on symmetric sector angle
        theta_min = -sector_angle/2 + np.pi/2
        theta_max = sector_angle/2 + np.pi/2

        # Get top sector indices
        top_sector_indices = \
                (TT > theta_min) & (TT < theta_max) & annulus_mask
        # Get bottom sector indices based on bottom-top symmetry
        bottom_sector_indices = np.fliplr(top_sector_indices)

        test_image[top_sector_indices] = 1
        test_image[bottom_sector_indices] = 1

        # Calculate the known intensity based on sector areas
        area_top_sector = np.pi*(rmax**2-rmin**2)*sector_angle/(2*np.pi)
        area_bottom_sector = area_top_sector
        known_intensity = area_top_sector + area_bottom_sector

        # Initiate the class
        feature_extraction = FeatureExtraction(
                test_image, pixel_length=pixel_length,
                source_wavelength=source_wavelength,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate the annulus intensity
        calculated_intensity = feature_extraction.feature_sector_intensity_meridian_pair_angstroms(
                pixel_length=pixel_length, amin=amin, amax=amax, sector_angle=sector_angle)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))

    def test_feature_sector_intensity_meridian_pair_angstrom_zeros(self):
        """
        Test FeatureExtraction for a test image with a pair of meridian sectors
        of zeros, given sector bounds in angstroms.
        Ensure the calculated intensity of the sector pairs is correct.
        """
        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape)

        # Set equator sector pair properties
        amin = 8.8e-10
        amax = 10.8e-10
        sector_angle = np.pi/4

        # Set machine parameters
        source_wavelength = 1.5418e-10
        pixel_length = 55e-6
        sample_to_detector_distance = 10e-3

        # Set known intensity to zero
        known_intensity = 0

        # Initiate the class
        feature_extraction = FeatureExtraction(
                test_image, pixel_length=pixel_length,
                source_wavelength=source_wavelength,
                sample_to_detector_distance=sample_to_detector_distance)

        # Calculate the annulus intensity
        calculated_intensity = feature_extraction.feature_sector_intensity_meridian_pair_angstroms(
                pixel_length=pixel_length, amin=amin, amax=amax, sector_angle=sector_angle)

        # Ensure the calculated intensity is correct
        self.assertTrue(
                np.isclose(calculated_intensity, known_intensity, rtol=0.05))


class TestFeatureExtractionCLI(unittest.TestCase):
    """
    FeatureExtraction commandline interface tests
    """

    def setUp(self):

        # Generate the test image
        size = 256
        shape = size, size
        test_image = np.zeros(shape, dtype=np.uint32)

        # Set equator sector pair properties
        equator_rmin = size/8
        equator_rmax = size/6
        equator_sector_angle = np.pi/4

        # Create a mask for the annulus
        equator_annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=equator_rmin, rmax=equator_rmax)

        # Set the sector pixel values to 1
        # Generate a meshgrid the same size as the image
        x_end = shape[1]/2 - 0.5
        x_start = -x_end
        y_end = x_start
        y_start = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        TT = np.arctan2(YY, XX)

        # Calculate sector start and end angles based on symmetric sector angle
        equator_theta_min = -equator_sector_angle/2
        equator_theta_max = equator_sector_angle/2

        # Get right sector indices
        right_sector_indices = \
                (TT > equator_theta_min) & (TT < equator_theta_max) \
                & equator_annulus_mask
        # Get left sector indices based on left-right symmetry
        left_sector_indices = np.fliplr(right_sector_indices)

        # Set equator sectors to 10
        test_image[right_sector_indices] = 10
        test_image[left_sector_indices] = 10

        # Set meridian sector properties
        meridian_rmin = size/4
        meridian_rmax = size/3
        meridian_sector_angle = np.pi/3

        # Calculate sector start and end angles based on symmetric sector angle
        meridian_theta_min = -meridian_sector_angle/2 + np.pi/2
        meridian_theta_max = meridian_sector_angle/2 + np.pi/2

        # Create a mask for the annulus
        meridian_annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=meridian_rmin, rmax=meridian_rmax)

        # Get top sector indices
        top_sector_indices = \
                (TT > meridian_theta_min) & (TT < meridian_theta_max) \
                & meridian_annulus_mask
        # Get bottom sector indices based on left-right symmetry
        bottom_sector_indices = np.flipud(top_sector_indices)

        # Set equator sectors to 10
        test_image[top_sector_indices] = 5
        test_image[bottom_sector_indices] = 5

        # Set ring properties
        ring_rmin = size/3
        ring_rmax = 4*size/9

        # Create a mask for the annulus
        ring_annulus_mask = create_circular_mask(
                shape[0], shape[1], rmin=ring_rmin, rmax=ring_rmax)

        test_image[ring_annulus_mask] = 1

        # Create test directory
        feature_extraction_test_dir = "test_feature_extraction"
        feature_extraction_test_path = os.path.join(
                TEST_IMAGE_PATH, feature_extraction_test_dir)
        os.makedirs(feature_extraction_test_path, exist_ok=True)

        # Store test directory
        self.test_path = feature_extraction_test_path

        test_image_filename = "test_image.txt"
        test_image_filepath = os.path.join(feature_extraction_test_path,
                test_image_filename)

        # Save test image to file
        np.savetxt(test_image_filepath, test_image, fmt="%d")

    def test_feature_extraction_cli_ones_test_image(self):
        """
        Test feature extraction commandline interface for a test image with
        ones.
        """
        self.fail("Finish writing test.")

    def tearDown(self):
        # Construct test directory
        feature_extraction_test_dir = "test_feature_extraction"
        feature_extraction_test_path = os.path.join(
                TEST_PATH, feature_extraction_test_dir)
        os.makedirs(feature_extraction_test_path, exist_ok=True)

        test_image_filename = "test_image.txt"
        test_image_filepath = os.path.join(feature_extraction_test_path,
                test_image_filename)

        # Delete test directory
        test_path = self.test_path
        shutil.rmtree(test_path)

class TestPolarMeshgrid(unittest.TestCase):
    """
    Test polar meshgrid
    """

    def test_2_radii(self):
        """
        Test polar meshgrid for a simple 2-radii case
        """
        size = 256
        output_shape = (size, size)
        r_count = 2
        theta_count = 1
        rmin = 0
        rmax = output_shape[0]/2

        meshgrid = polar_meshgrid(
                output_shape=output_shape,
                r_count=r_count,
                theta_count=theta_count,
                rmin=rmin,
                rmax=rmax,
                )

        inner_mask = create_circular_mask(
                size, size, rmin=rmin, rmax=rmax/2)
        outer_mask = create_circular_mask(
                size, size, rmin=rmax/2, rmax=rmax)

        known_image = np.zeros(output_shape)
        known_image[inner_mask] = 1
        known_image[outer_mask] = 2

        self.assertTrue(np.array_equal(meshgrid, known_image))

    def test_4_sectors(self):
        """
        Test polar meshgrid for a simple 4-sector case
        """
        size = 256
        output_shape = (size, size)
        r_count = 1
        theta_count = 4
        rmin = 0
        rmax = output_shape[0]/2

        meshgrid = polar_meshgrid(
                output_shape=output_shape,
                r_count=r_count,
                theta_count=theta_count,
                rmin=rmin,
                rmax=rmax,
                )

        # Create image mask
        image_mask = create_circular_mask(
                size, size, rmin=rmin, rmax=rmax)

        # Create known image
        known_image = np.zeros(output_shape)
        # Set middle index
        mdx = int(output_shape[0]/2)
        # Set upper-right quadrant value to 1
        known_image[:mdx,mdx:] = 1
        # Set upper-left quadrant value to 2
        known_image[:mdx,:mdx] = 2
        # Set lower-left quadrant value to 3
        known_image[mdx:,:mdx] = 3
        # Set lower-right quadrant value to 4
        known_image[mdx:,mdx:] = 4

        # Apply image mask
        known_image[~image_mask] = 0

        self.assertTrue(np.array_equal(meshgrid, known_image))


class TestAzimuthalIntegration(unittest.TestCase):

    def test_azimuthal_integration_peak_position_conservation(self):
        """
        Use a synthetic pattern with known peak location
        and check that azimuthal integration yields a peak in the
        same location.
        """
        shape = (256,256)
        x_start = -shape[1]/2 - 0.5
        x_end = -x_start
        y_start = x_start
        y_end = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        RR = np.sqrt(XX**2 + YY**2)

        known_peak_location = 20
        test_image = np.exp(-(RR - known_peak_location)**2)

        center = np.array(shape)/2 - 0.5

        profile_1d = azimuthal_integration(test_image, center=center)

        calculated_peak_location = np.where(profile_1d == np.nanmax(profile_1d))[0]

        self.assertEqual(calculated_peak_location, known_peak_location)

    def test_azimuthal_integration_half_image_peak_position_conservation(self):
        """
        Use a synthetic half-image test pattern with known peak location
        and check that azimuthal integration yields a peak in the
        same location with the expected value.
        """
        shape = (256,256)
        x_start = -shape[1]/2 - 0.5
        x_end = -x_start
        y_start = x_start
        y_end = x_end
        YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        RR = np.sqrt(XX**2 + YY**2)
        TT = np.arctan2(YY, XX)

        known_peak_location = 20
        test_image = np.exp(-(RR - known_peak_location)**2)
        test_image[TT < 0] = 0

        center = np.array(shape)/2 - 0.5

        profile_1d = azimuthal_integration(
                test_image,
                center=center,
                azimuthal_point_count=90,
                start_angle=-np.pi,
                end_angle=0)

        calculated_peak_location = np.where(profile_1d == np.nanmax(profile_1d))[0]

        self.assertEqual(calculated_peak_location, known_peak_location)

    def test_azimuthal_integration_half_image_peak_value_conservation(self):
        """
        Use a synthetic half-image test pattern with known peak location
        and check that azimuthal integration yields a peak in the
        same location with the expected value.
        """
        shape = (256,256)
        x_start = -shape[1]/2 - 0.5
        x_end = -x_start
        y_start = x_start
        y_end = x_end
        rows, cols = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
        RR = np.sqrt(rows**2 + cols**2)
        TT = np.arctan2(-rows, cols)

        known_peak_location = 20
        test_image_full_ring = np.exp(-(RR - known_peak_location)**2)
        test_image_half_ring = test_image_full_ring.copy()
        test_image_half_ring[TT < 0] = 0

        center = np.array(shape)/2 - 0.5

        profile_1d_full_ring = azimuthal_integration(
                test_image_full_ring,
                center=center,
                azimuthal_point_count=180,
                start_angle=-np.pi,
                end_angle=np.pi)
        profile_1d_half_ring = azimuthal_integration(
                test_image_half_ring,
                center=center,
                azimuthal_point_count=90,
                start_angle=0,
                end_angle=np.pi)

        peak_value_full_ring = np.nanmax(profile_1d_full_ring)
        peak_value_half_ring = np.nanmax(profile_1d_half_ring)

        self.assertTrue(np.isclose(peak_value_full_ring, peak_value_half_ring, atol=1e-2))

        calculated_peak_location_full_ring = np.where(
                profile_1d_full_ring == np.nanmax(profile_1d_full_ring))[0]
        calculated_peak_location_half_ring = np.where(
                profile_1d_half_ring == np.nanmax(profile_1d_half_ring))[0]

        self.assertEqual(calculated_peak_location_full_ring, known_peak_location)
        self.assertEqual(calculated_peak_location_half_ring, known_peak_location)

    def test_azimuthal_integration_dir(self):
        """
        """
        TEST_IMAGE_DIR = "test_images"
        TEST_DIR = "test_azimuthal_integration"
        INPUT_DIR = "input"
        OUTPUT_DIR = "output"
        PREPROCESSED_DATA_DIR = "preprocessed_data"
        KNOWN_RESULTS_DIR = "known_results"

        # Set input path
        input_path = os.path.join(TEST_IMAGE_PATH, TEST_DIR, INPUT_DIR)
        # Set output path
        output_path = os.path.join(
                TEST_IMAGE_PATH, TEST_DIR, OUTPUT_DIR)
        # Test data path
        test_data_path = os.path.join(output_path, PREPROCESSED_DATA_DIR)
        # Set known results path
        known_results_path = os.path.join(
                TEST_IMAGE_PATH, TEST_DIR, KNOWN_RESULTS_DIR)

        size = 256
        shape = (size, size)
        center = np.array(shape)/2 - 0.5

        # Run azimuthal integration on input directory
        azimuthal_integration_dir(
                input_path=input_path,
                output_path=output_path,
                center=center,
                beam_rmax=0,
                )

        # Load input images
        input_filepath_list = glob.glob(
                os.path.join(input_path, "*.*"))
        input_filepath_list.sort()

        # Load output images
        test_data_filepath_list = glob.glob(
                os.path.join(test_data_path, "*.*"))
        test_data_filepath_list.sort()

        # Load known results images
        known_results_filepath_list = glob.glob(
                os.path.join(known_results_path, "*.*"))
        known_results_filepath_list .sort()

        # Check that there are a non-zero number of files
        self.assertTrue(len(input_filepath_list) > 0)
        self.assertTrue(len(test_data_filepath_list) > 0)
        self.assertTrue(len(known_results_filepath_list) > 0)

        for idx in range(len(input_filepath_list)):
            # Load the azimuthal integration test results
            output_filepath = test_data_filepath_list[idx]
            test_results_radial_data = np.loadtxt(output_filepath)

            # Load known azimuthal integration results
            known_results_filepath = known_results_filepath_list[idx]
            known_results_radial_data = np.loadtxt(known_results_filepath)

            # Check that the test results equal the known results
            self.assertTrue(
                    np.array_equal(
                        test_results_radial_data, known_results_radial_data,
                        equal_nan=True))

    def test_azimuthal_integration_scaling(self):
        """
        Ensure azimuthal integration scales properly
        """
        # Create test image such that the inner annulus is 1
        # and the outer annulus is 0
        # The resulting azimuthal integration profile should
        # be a step function
        size = 256
        shape = (size, size)
        test_image = np.zeros(shape)
        mask = create_circular_mask(size, size, rmin=0, rmax=size/4)
        test_image[mask] = 1

        end_radius = int(np.max(test_image.shape)/2)

        center = np.array(shape)/2 - 0.5

        profile_1d = azimuthal_integration(
                test_image, center=center, end_radius=end_radius)
        profile_size = profile_1d.size
        step_function = np.zeros(end_radius)
        step_function[:end_radius//2] = 1

        # Take the difference
        diff = abs(step_function - profile_1d)

        # Test that the 1-D integrated profile is close to a step function
        self.assertTrue(np.isclose(np.sum(diff), 0, atol=1))

        # In this test we receive only 5 values
        # 0., 0.0032228, 0.55451952, 0.99823135, and 1
        values = np.unique(profile_1d)
        self.assertEqual(values.size, 5)

    def test_azimuthal_integration_transformer_scaling(self):
        """
        Ensure azimuthal integration scales properly
        """
        # Create test image such that the inner annulus is 1
        # and the outer annulus is 0
        # The resulting azimuthal integration profile should
        # be a step function
        size = 256
        shape = (size, size)
        test_image = np.zeros(shape)
        mask = create_circular_mask(size, size, rmin=0, rmax=size/4)
        test_image[mask] = 1

        end_radius = int(np.max(test_image.shape)/2)

        center = np.array(shape)/2 - 0.5

        test_image_array = test_image.reshape(
                (1, test_image.shape[0], test_image.shape[1]))

        azimuthalintegration = AzimuthalIntegration(
                center=center, end_radius=end_radius)

        results = azimuthalintegration.transform(
                test_image_array)
        profile_1d = results[0, ...]
        profile_size = profile_1d.size
        step_function = np.zeros(end_radius)
        step_function[:end_radius//2] = 1

        # Take the difference
        diff = abs(step_function - profile_1d)

        # Test that the 1-D integrated profile is close to a step function
        self.assertTrue(np.isclose(np.sum(diff), 0, atol=1))

        # In this test we receive only 5 values
        # 0., 0.0032228, 0.55451952, 0.99823135, and 1
        values = np.unique(profile_1d)
        self.assertEqual(values.size, 5)

    def test_azimuthal_integration_output_shape(self):
        """
        Ensure azimuthal integration scales properly
        """
        # Create test image such that the inner annulus is 1
        # and the outer annulus is 0
        # The resulting azimuthal integration profile should
        # be a step function
        size = 256
        shape = (size, size)
        test_image = np.zeros(shape)
        mask = create_circular_mask(size, size, rmin=0, rmax=size/4)
        test_image[mask] = 1

        start_radius = 10
        end_radius = int(np.max(test_image.shape)/2)

        center = np.array(shape)/2 - 0.5

        profile_1d = azimuthal_integration(
                test_image, center=center, start_radius=start_radius,
                end_radius=end_radius)
        profile_size = profile_1d.size

        self.assertEqual(profile_size, end_radius-start_radius)


class TestPreprocessingPipeline(unittest.TestCase):

    def test_image_repair_pipeline(self):
        """
        Test pipeline with image repair transformer
        """
        # Create test image
        size = 256
        test_image = np.ones((size,size))
        hot_spot_coords = (20,40)
        hot_spot_value = 10
        test_image[hot_spot_coords] = hot_spot_value

        self.assertEqual(np.max(test_image), hot_spot_value)

        X = test_image.reshape((1, test_image.shape[0], test_image.shape[1]))

        threshold = 5
        absolute = True
        fill_method = "median"
        filter_size = 1

        # Instantiate the image repair transformer
        imrepair = FilterOutlierPixelValues(
                threshold=threshold,
                absolute=absolute,
                fill_method=fill_method,
                filter_size=filter_size)
        # Create a classifier from the pipeline
        clf = make_pipeline(imrepair)

        # Transform the data
        X_repaired = clf.transform(X)

        repaired_image = X_repaired[0, ...]

        # Check if the image repair was successful
        self.assertTrue(np.array_equal(repaired_image, np.ones((size,size))))

    def test_image_repair_azimuthal_integration_pipeline(self):
        """Test pipeline with image repair and azimuthal integration
        transformers.
        """
        # Create test image such that the inner annulus is 1
        # and the outer annulus is 0
        # The resulting azimuthal integration profile should
        # be a step function
        size = 256
        shape = (size, size)
        test_image = np.zeros(shape)
        mask = create_circular_mask(size, size, rmin=0, rmax=size/4)
        test_image[mask] = 1

        # Add hot spot
        hot_spot_coords = (20,40)
        hot_spot_value = 10
        test_image[hot_spot_coords] = hot_spot_value

        X = test_image.reshape((1, test_image.shape[0], test_image.shape[1]))

        # Set azimuthal integration parameters
        end_radius = int(np.max(test_image.shape)/2)
        center = np.array(shape)/2 - 0.5

        # Create step function for known result of azimuthal integration
        step_function = np.zeros(end_radius)
        step_function[:end_radius//2] = 1

        # Set image repair parameters
        threshold = 5
        absolute = True
        fill_method = "median"
        filter_size = 1

        # Instantiate the image repair transformer
        imrepair = FilterOutlierPixelValues(
                threshold=threshold,
                absolute=absolute,
                fill_method=fill_method,
                filter_size=filter_size)
        azint = AzimuthalIntegration(
                center=center, end_radius=end_radius)
        # Create a classifier from the pipeline
        clf = make_pipeline(imrepair, azint)

        # Transform the data
        X_trans = clf.transform(X)

        profile_1d = X_trans[0, ...]

        # Check if the preprocessing pipeline was successful

        # Take the difference
        diff = abs(step_function - profile_1d)

        # Test that the 1-D integrated profile is close to a step function
        self.assertTrue(np.isclose(np.sum(diff), 0, atol=1))

if __name__ == '__main__':
    unittest.main()
