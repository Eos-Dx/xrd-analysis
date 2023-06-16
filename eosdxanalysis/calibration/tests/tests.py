"""
Calibration module tests

q units are per Angstrom
"""
import os

import unittest
import numpy as np

from eosdxanalysis.preprocessing.utils import find_center

from eosdxanalysis.calibration.calibration import SampleDistanceCalibration

from eosdxanalysis.calibration.utils import DiffractionUnitsConversion


TEST_IMAGE_PATH = os.path.join("eosdxanalysis","calibration","tests","test_images")


class TestSampleDistanceCalibration(unittest.TestCase):
    """
    Test SampleDistanceCalibration class

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
        calibrator = SampleDistanceCalibration(calibration_material="silver_behenate")

        # Calculate the detector distance (units in meters)
        detector_distance_m, linreg = \
                calibrator.single_sample_detector_distance(test_image, r_max=80)

        detector_distance_mm = detector_distance_m * 1e3

        # Set the tolerance
        # See: eosdxanalysis.simulation.diffraction for 1-pixel shift at 10 mm
        tol_mm = 0.36

        self.assertTrue(np.isclose(detector_distance_mm, known_distance_mm, atol=tol_mm),
                msg=f"SampleDistanceCalibration distance incorrect! "
                    f"Calculated distance {np.round(detector_distance_mm, decimals=2)} mm "
                        f"!= {known_distance_mm} mm")


class TestDiffractionUnitsConversion(unittest.TestCase):

    def test_two_theta_from_molecular_spacing(self):
        """
        Test 2*theta is correct for some chosen values
        """

        # Set machine parameters
        source_wavelength = 1
        sample_to_detector_distance = 1e-3

        # Set molecular spacing
        molecular_spacing = 2.3

        # Set known 2*theta
        known_two_theta = 2*np.arcsin(source_wavelength/(2*molecular_spacing))

        conversion_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength,
                sample_to_detector_distance=sample_to_detector_distance)

        two_theta = conversion_class.two_theta_from_molecular_spacing(
                molecular_spacing)

        self.assertEqual(two_theta, known_two_theta)

    def test_bragg_peak_pixel_location_from_molecular_spacing(self):
        """
        Test Bragg peak pixel location function
        """
        # Set parameters
        source_wavelength = 1.5418e-10
        pixel_length = 55e-6
        distance = 10e-3
        molecular_spacing = 9.8e-10

        # Set known location
        known_pixel_location = 29

        # Initialize diffraction units class
        units_class = DiffractionUnitsConversion(
                source_wavelength=source_wavelength, pixel_length=pixel_length,
                sample_to_detector_distance=distance)

        bragg_peak_pixel_location = \
                units_class.bragg_peak_pixel_location_from_molecular_spacing(
                        molecular_spacing)

        self.assertTrue(
                np.isclose(
                    bragg_peak_pixel_location, known_pixel_location, atol=0.5))


if __name__ == '__main__':
    unittest.main()
