import os
import unittest
import numpy as np
import pandas as pd

import subprocess
import json
import glob

from skimage.io import imsave, imread
from skimage.transform import warp_polar
from scipy.ndimage import center_of_mass

from eosdxanalysis.preprocessing.image_processing import centerize
from eosdxanalysis.preprocessing.image_processing import pad_image
from eosdxanalysis.preprocessing.image_processing import rotate_image
from eosdxanalysis.preprocessing.image_processing import unwarp_polar
from eosdxanalysis.preprocessing.image_processing import crop_image

from eosdxanalysis.preprocessing.center_finding import radial_mean
from eosdxanalysis.preprocessing.center_finding import find_center
from eosdxanalysis.preprocessing.center_finding import find_centroid

from eosdxanalysis.preprocessing.denoising import stray_filter
from eosdxanalysis.preprocessing.denoising import filter_strays

from eosdxanalysis.preprocessing.utils import count_intervals
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import gen_rotation_line

from eosdxanalysis.preprocessing.preprocess import PreprocessData
from eosdxanalysis.preprocessing.preprocess import ABBREVIATIONS
from eosdxanalysis.preprocessing.preprocess import OUTPUT_MAP
from eosdxanalysis.preprocessing.preprocess import INVERSE_OUTPUT_MAP

from eosdxanalysis.preprocessing.peak_finding import find_2d_peak
from eosdxanalysis.preprocessing.peak_finding import find_1d_peaks

TEST_IMAGE_DIR = os.path.join("eosdxanalysis","preprocessing","tests","test_images")


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
        test_dir = os.path.join(TEST_IMAGE_DIR, "test_cli_images")
        self.test_dir = test_dir

        # Specify parameters file

        params_file = os.path.join(test_dir, "params.txt")
        self.params_file = params_file
        with open(params_file, "r") as param_fp:
            params = param_fp.read()
        self.params = params

        # Set the input and output directories
        test_input_dir = os.path.join(test_dir, "input")
        test_output_dir = os.path.join(test_dir, "output")

        self.test_input_dir = test_input_dir
        self.test_output_dir = test_output_dir

        input_files_fullpaths = glob.glob(os.path.join(test_input_dir, "*.txt"))

        self.input_files_fullpaths = input_files_fullpaths

    def test_preprocess_cli(self):
        """
        Run preprocessing using commandline, providing input data directory
        and output directory.
        """
        params = self.params
        params_file = self.params_file
        test_input_dir = self.test_input_dir
        test_output_dir = self.test_output_dir
        input_files_fullpaths = self.input_files_fullpaths

        # Construct plans list
        plan = "centerize_rotate_quad_fold"
        plan_abbr = ABBREVIATIONS.get(plan)
        output_style = INVERSE_OUTPUT_MAP.get(plan)

        # Set up the command
        command = ["python", "eosdxanalysis/preprocessing/preprocess.py",
                    "--input_dir", test_input_dir,
                    "--output_dir", test_output_dir,
                    "--params_file", params_file,
                    "--plans", plan,
                    ]

        # Run the command
        subprocess.run(command)

        # Check that output files exist
        # First get list of files
        num_files = len(input_files_fullpaths)

        # Check that number of files is > 0
        self.assertTrue(num_files > 0)
        # Check that number of input and output files is the same
        plan_output_dir = os.path.join(test_output_dir, output_style)
        plan_output_files_fullpaths = glob.glob(os.path.join(plan_output_dir, "*.txt"))

        self.assertEqual(num_files, len(plan_output_files_fullpaths))

        for idx in range(num_files):
            # Load data
            input_image = np.loadtxt(input_files_fullpaths[idx])
            output_image = np.loadtxt(plan_output_files_fullpaths[idx])

            # Check that data are positive
            self.assertTrue(input_image[input_image > 0].all())
            self.assertTrue(output_image[output_image > 0].all())

            # Check that the maximum value of the output is less than the
            # maximum value of the input
            self.assertTrue(np.max(output_image) < np.max(input_image))

            # Check that output means are smaller than input means
            self.assertTrue(np.mean(output_image) < np.mean(input_image))

    def tearDown(self):
        """
        Remove any output files
        """
        test_output_dir = self.test_output_dir
        # os.remove(os.path.join(test_output_dir, "*"))


class TestPreprocessData(unittest.TestCase):

    def setUp(self):
        params = {
            # Set parameters
            # Image size
            "h":256,
            "w":256,
            # Region of interest for beam center on raw images
            # rmax_beam=50
            "beam_rmax":25,
            # Annulus region of interest for XRD pattern
            # rmin=30
            # rmax=120
            "rmin":25,
            "rmax":90,
            # Annulus region of interest for 9A feature region
            # reyes_min=40
            # reyes_max=80
            "eyes_rmin":30,
            "eyes_rmax":45,
            # Maximum distance from 9A feature maximum intensity location
            # for blob analysis
            # reyes_max_blob=30
            "eyes_blob_rmax":20,
            # Percentile used to analyze 9A features as blobs
            "eyes_percentile":99,
            # Local threshold block size
            # local_threshold_block_size = 27
            "local_thresh_block_size":21,
        }
        self.params = params

    def test_preprocess_single_image_rotate_centerize(self):
        # Load the test image
        # Original filename: 20220330/A00041.txt
        test_filename = "test_preprocess_center.txt"

        cwd = os.getcwd()
        test_dir = os.path.join(cwd, TEST_IMAGE_DIR)
        test_img = os.path.join(test_dir, test_filename)

        preprocessed_filename = "preprocessed_{}".format(test_filename)
        preprocessed_filename_fullpath = os.path.join(test_dir,preprocessed_filename)

        params = self.params
        preprocessor = PreprocessData(filename=test_img,
                input_dir=test_dir, output_dir=test_dir, params=params)

        # Preprocess data, saving to a file
        preprocessor.preprocess()

        # Check center of saved file
        preprocessed_image = np.loadtxt(preprocessed_filename_fullpath)
        calculated_center = find_center(preprocessed_image)
        # Proper centerized images would have center near here:
        center = (128,128)

        self.assertTrue(np.isclose(center, calculated_center, atol=1.0).all())


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
        # Set known center using centroid of max pixels in beam region of interest
        known_center = (126.125, 132.375) # Using centroid of max pixels
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_img = np.loadtxt(os.path.join(test_dir, TEST_IMAGE_DIR, test_filename))
        calculated_center = find_center(test_img, method="max_centroid")

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

    def test_radial_mean(self):
        intensities = np.array([[4,6,4,],[5,10,5],[4,6,4]])
        center = (1,1)
        rmean = radial_mean(intensities,center)
        rmean_ref = np.array([0,1,2,3,4,5,6])
        self.fail("Finish writing test")
        self.assertIsNone(np.testing.assert_array_equal(rmean, rmean_ref))


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


class TestDenoising(unittest.TestCase):

    def test_stray_detector_high_intensity(self):
        # Create a test array where center value is much higher
        # than neighborhood average
        test_array = np.array([[1,2,3],[1,105,0],[2,25,2]])
        center_value = test_array[1,1]
        # Apply filter to center value
        filtered_center = stray_filter(test_array, factor=5.0)

        # Ensure filter changed value to mean of neighbors
        self.assertFalse(center_value == filtered_center)

    def test_stray_detector_low_intensity(self):
        # Create a test array where center value is close to
        # neighborhood average
        test_array = np.array([[1,2,3],[1,20,0],[2,25,2]])
        center_value = test_array[1,1]
        # Apply filter to center value
        filtered_center = stray_filter(test_array, factor=5.0)

        # Ensure filter did not change center value
        self.assertTrue(center_value == filtered_center)


class TestImageProcessing(unittest.TestCase):

    def test_sensible_intensity_ranges(self):
        self.fail("Finish writing test")

    def test_local_threshold_determinism_separate_measurements(self):
        self.fail("Finish writing test")

    def test_centerize_photon_count(self):
        """
        Ensure that the centerized image has a photon intensity count
        that is close to the original (some differences expected due
        to rasterization).
        """
        # Generate random 10x10 image
        rng = np.random.default_rng(seed=100)
        original = rng.integers(low=0, high=100, size=(10,10))
        # Pick center at random
        center = rng.random((1,2))*10
        center = tuple(center.flatten())

        centerized, new_center = centerize(original, center)
        original_count = np.sum(original)
        centerized_count = np.sum(centerized)

        self.assertTrue(np.isclose(original_count, centerized_count, atol=2))

    def test_centerize_point_method_precentered(self):
        """
        Test centerize point method for precentered image
        """
        # Create an image
        dim = 6
        image = np.arange(dim**2).reshape((dim,dim))
        # choose center 
        center = ((dim-1)/2,(dim-1)/2)

        centerized_image, new_center = centerize(image, center, method="point")

        self.assertEqual(center, new_center)
        self.assertEqual(image.shape, centerized_image.shape)
        self.assertTrue(np.array_equal(image, centerized_image))

    def test_centerize_point_method_slight_off_center(self):
        """
        Test centerize point method for slightly off-center image
        Should give back original image
        """
        # Create an image
        dim = 6
        image = np.arange(dim**2).reshape((dim,dim))
        # choose center 
        center = (2.6,2.6)

        centerized_image, new_center = centerize(image, center, method="point")

        known_center = (2.5,2.5)
        self.assertEqual(known_center, new_center)
        self.assertEqual(image.shape, centerized_image.shape)
        self.assertTrue(np.array_equal(image, centerized_image))

    def test_centerize_point_method_rounding_edge_case(self):
        """
        Numpy rounds both 1.5 and 2.5 to 2.0, make sure
        code gives correct results.
        """
        self.fail("Finish writing test")


    def test_centerize_point_method_off_center(self):
        """
        Test centerize point method for significantly off-center image
        """
        # Create an image
        dim = 4
        image = np.arange(dim**2).reshape((dim,dim))
        # choose center 
        center = (2.1,1.6)
        # This should be like center equivalent or mod_center
        center_equiv = (2.0,1.5)

        centerized_image, new_center = centerize(image, center, method="point")
        centerized_image_equiv, new_center_equiv = centerize(image, center_equiv, method="point")

        self.assertEqual(new_center, new_center_equiv)
        self.assertEqual(centerized_image.shape, centerized_image_equiv.shape)
        self.assertTrue(np.array_equal(centerized_image, centerized_image_equiv))

    def test_centerize_point_method_center_3x3(self):
        """
        Test centerize point method for 3x3 images where center could be
        any of the pixel centers (whole indices)
        """
        # Create an image
        dim = 3
        image = np.arange(1,dim**2+1).reshape((dim,dim))
        """
        Original image:

            [[1, 2, 3]
             [4, 5, 6]
             [7, 8, 9]],
        """

        # Comments show matrix coordinates of center for original image
        known_results = np.array([
            # (0,0) the 1
            np.array([[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 1, 2, 3],
             [0, 0, 4, 5, 6],
             [0, 0, 7, 8, 9]]),
            # (0,1) the 2
            np.array([[0, 0, 0],
             [0, 0, 0],
             [1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]),
            # (0,2) the 3
            np.array([[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [1, 2, 3, 0, 0],
             [4, 5, 6, 0, 0],
             [7, 8, 9, 0, 0]]),
            # (1,0) the 4
            np.array([[0, 0, 1, 2, 3],
             [0, 0, 4, 5, 6],
             [0, 0, 7, 8, 9]]),
            # (1,1) the 5
            np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9],]),
            # (1,2) the 6
            np.array([[1, 2, 3, 0, 0],
             [4, 5, 6, 0, 0],
             [7, 8, 9, 0, 0],]),
            # (2,0) the 7
            np.array([[0, 0, 1, 2, 3],
             [0, 0, 4, 5, 6],
             [0, 0, 7, 8, 9],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],]),
            # (2,1) the 8
            np.array([[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9],
             [0, 0, 0],
             [0, 0, 0],]),
            # (2,2) the 9
            np.array([[1, 2, 3, 0, 0],
             [4, 5, 6, 0, 0],
             [7, 8, 9, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],]),
            ],dtype=object)

        # Loop over possibilites for center
        for center_row in np.arange(dim):
            for center_col in np.arange(dim):
                center = (center_row, center_col)

                centerized_image, new_center = centerize(image, center, method="point")

                # Ensure the number to be centered is now at new center
                self.assertEqual(image[int(np.around(center[0])),int(np.around(center[1]))],
                        centerized_image[int(np.around(new_center[0])),int(np.around(new_center[1]))])

                # Ensure the entire image is still present
                self.assertEqual(np.sum(centerized_image), np.sum(image))

                # Test to make sure results are accurate using known results
                self.assertTrue(np.array_equal(known_results[center_row*dim+center_col], centerized_image))

    def test_pad_image_prerotation_method(self):
        """
        Test a few images with default prerotation method
        """
        dim = 1
        image = np.arange(dim**2).reshape((dim,dim))

        padded_image = pad_image(image, method="prerotation")

        self.assertEqual(padded_image.shape, (1,1))

        dim = 2
        image = np.arange(dim**2).reshape((dim,dim))

        padded_image = pad_image(image, method="prerotation")

        self.assertEqual(padded_image.shape, (2,2))

        dim = 3
        image = np.arange(dim**2).reshape((dim,dim))

        padded_image = pad_image(image, method="prerotation")

        self.assertEqual(padded_image.shape, (3,3))

        dim = 4
        image = np.arange(dim**2).reshape((dim,dim))

        padded_image = pad_image(image, method="prerotation")

        self.assertEqual(padded_image.shape, (6,6))

        dim = 10
        image = np.arange(dim**2).reshape((dim,dim))

        padded_image = pad_image(image, method="prerotation")

        self.assertEqual(padded_image.shape, (14,14))

    def test_rotate_image_nearest(self):
        """
        Test rotate_image function with nearest method
        which uses cv2.INTER_NEAREST flag
        """
        # Create 4x4 array
        dim=4
        image = np.array([
            [0,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,0],
        ])

        # Create known 90 degree rotation of 2x2 array
        rot_image_known = np.array([
            [0,0,0,0],
            [0,0,1,0],
            [0,1,0,0],
            [0,0,0,0],
        ])

        angle = 90.0

        rotated_image_nearest = rotate_image(image, angle=angle, method="nearest")
        rotated_image_standard = rotate_image(image, angle=angle, method="standard")

        self.assertTrue(np.array_equal(rot_image_known, rotated_image_nearest))

    def test_rotate_image_elastic(self):
        """
        Test rotate_image function with nearest method
        which uses cv2.INTER_NEAREST flag
        """
        # Create 4x4 array
        dim=4
        image = np.array([
            [0,0,0,0],
            [0,10,0,0],
            [0,0,10,0],
            [0,0,0,0],
        ])

        # Create known 90 degree rotation of 2x2 array
        rot_image_known = np.array([
            [0,0,0,0],
            [0,0,10,0],
            [0,10,0,0],
            [0,0,0,0],
        ])

        angle = 45.0

        rotated_image_elastic = rotate_image(image, angle=angle, method="elastic")
        rotated_image_standard = rotate_image(image, angle=angle, method="standard")

        print(rotated_image_elastic)

        self.assertTrue(np.array_equal(rot_image_known, rotated_image_elastic))

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
        test_parent_path = os.path.join(TEST_IMAGE_DIR, test_dirname)

        samples_dir = "samples"
        samples_path = os.path.join(test_parent_path, samples_dir)

        output_dir = "output"
        output_path = os.path.join(test_parent_path, output_dir)

        saturated_dir = "preprocessed_samples"
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
        preprocessor.save()

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
        output_style = INVERSE_OUTPUT_MAP[plans[0]]
        plan_output_path = os.path.join(output_path, output_style)

        # Run preprocessing
        preprocessor = PreprocessData(
                        input_dir=input_path, output_dir=plan_output_path, params=params)
        preprocessor.preprocess(plans=plans)
        preprocessor.save()

        # Now ensure that preprocessed files are not saturated
        output_filepath_list = glob.glob(
                                    os.path.join(plan_output_path, "*.txt"))

        # Ensure that output_filepath_list is not empty
        self.assertTrue(output_filepath_list)

        for output_filepath in output_filepath_list:
            data = np.loadtxt(output_filepath)
            unique = np.unique(data)
            # Ensure that we get only two values
            self.assertGreater(unique.size, 2)
            # Ensure that the saturation_value is not in the file
            self.assertNotIn(saturation_value, unique)


if __name__ == '__main__':
    unittest.main()
