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


TEST_IMAGE_DIR = os.path.join("eosdxanalysis","calibration","tests","test_images")


class TestCalibration(unittest.TestCase):
    """
    Test Calibration class

    q units are per Angstrom
    """

    def setUp(self):
        self.silver_behenate_test_dir = "silver_behenate_test_images"
        self.silver_behenate_test_input_dir = "input"
        self.silver_behenate_test_ouput_dir = "output"
        self.silver_behenate_test_image = \
                "synthetic_calibration_silver_behenate.txt"

    def test_synthetic_silver_behenate_images(self):
        # Load the test data
        image_fullpath = os.path.join(TEST_IMAGE_DIR,
                self.silver_behenate_test_dir,
                self.silver_behenate_test_input_dir,
                self.silver_behenate_test_image)
        test_image = np.loadtxt(image_fullpath)

        # Set up the calibrator class
        calibrator = Calibration(calibration_material="silver_behenate")

        # Calculate the detector distance (units in meters)
        detector_distance_angstrom, linreg = \
                calibrator.single_sample_detector_distance(test_image, r_max=80)

        detector_distance_mm = detector_distance_angstrom / 1e10 * 1e3

        self.assertTrue(np.isclose(detector_distance_mm, 10e-3))

if __name__ == '__main__':
    unittest.main()
