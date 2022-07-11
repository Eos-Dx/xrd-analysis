"""
Calibration module tests

q units are per Angstrom
"""
import os

import unittest
import numpy as np

from eosdxanalysis.preprocessing.center_finding import find_center
from eosdxanalysis.preprocessing.image_processing import centerize

from eosdxanalysis.calibration.calibration import Calibration


TEST_IMAGE_PATH = os.path.join("eosdxanalysis","calibration","tests","test_images")


class TestCalibration(unittest.TestCase):
    """
    Test Calibration class

    q units are per Angstrom
    """

    def setUp(self):
        # Set paths and directories
        self.silver_behenate_test_dir = "silver_behenate_test_images"
        self.silver_behenate_test_input_dir = "input"
        self.silver_behenate_test_ouput_dir = "output"
        self.silver_behenate_test_image_filename = \
                "synthetic_calibration_silver_behenate.txt"

        # Clear output files
        output_filepath = os.path.join(TEST_IMAGE_PATH,
                                    self.silver_behenate_test_ouput_dir)

    def test_synthetic_silver_behenate_images(self):
        test_dir = self.silver_behenate_test_dir
        test_input_dir = self.silver_behenate_test_input_dir
        test_output_dir = self.silver_behenate_test_ouput_dir
        test_image_filename = self.silver_behenate_test_image_filename

        # Load the test data
        test_image_fullpath = os.path.join(TEST_IMAGE_PATH,
                test_dir, test_input_dir, test_image_filename)
        test_image = np.loadtxt(test_image_fullpath)
        known_distance_mm = 10 # mm

        # Set up the calibrator class
        calibrator = Calibration(calibration_material="silver_behenate")

        # Calculate the detector distance (units in meters)
        detector_distance_m, linreg = \
                calibrator.single_sample_detector_distance(test_image, r_max=80)

        detector_distance_mm = detector_distance_m * 1e3

        self.assertTrue(np.isclose(detector_distance_mm, known_distance_mm))

if __name__ == '__main__':
    unittest.main()
