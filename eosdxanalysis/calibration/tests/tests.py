"""
Calibration module tests

q units are per Angstrom
"""
import os

import unittest
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from eosdxanalysis.preprocessing.utils import find_center

from eosdxanalysis.calibration.calibration import sample_detector_distance

from eosdxanalysis.calibration.units_conversion import DiffractionUnitsConversion
from eosdxanalysis.calibration.units_conversion import MomentumTransferUnitsConversion
from eosdxanalysis.calibration.units_conversion import real_position_from_q


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

        # Calculate the detector distance (units in meters)
        detector_distance_m, linreg = \
                sample_detector_distance(test_image, end_radius=80)

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


class TestMomentumTransferUnitsConversion(unittest.TestCase):

    def test_agbh_first_q_peak_units_conversion(self):
        """
        AgBH has the first peak at 1.076 per nm
        """
        wavelength_nm = 0.1540562
        pixel_size = 55e-6
        unitsconversion = MomentumTransferUnitsConversion(
                wavelength_nm=wavelength_nm,
                pixel_size=pixel_size,
                )

        array_len = int(np.sqrt(2)*256)
        q_per_nm = 1.076
        sample_distance_mm = 180

        # Calculate real position
        position_mm = real_position_from_q(
                q_per_nm=q_per_nm,
                sample_distance_mm=sample_distance_mm,
                wavelength_nm=wavelength_nm,
                )

        # Calculate pixel position
        position_m = position_mm / 1e3
        pixel_position = position_m / pixel_size

        # Create a Gaussian at this pixel position
        X = np.exp(-((np.arange(array_len) - pixel_position)/(array_len/4))**2)

        sample_distance_m = np.array(sample_distance_mm) * 1e-3

        # Store data in dataframe
        profile_data = [X]
        sample_distance_data = [sample_distance_m]

        data = {
                "profile_data": profile_data,
                "sample_distance_m": sample_distance_data,
                }
        df = pd.DataFrame(data=data)

        # Convert units
        df_results = unitsconversion.transform(
                df,
                copy=True)

        q_range = df_results.loc[0]["q_range"]

        # Check that the size of X_q is correct
        self.assertEqual(q_range.size, array_len)

        # Check that the peak value is correct
        peak_value = np.nanmax(X)
        test_peak_pixel_location = np.where(X == peak_value)[0]

        # Check that the pixel location is the same
        self.assertTrue(np.isclose(test_peak_pixel_location, pixel_position, atol=0.5))

        test_q_per_nm = q_range[test_peak_pixel_location]
        self.assertTrue(np.isclose(test_q_per_nm, q_per_nm, atol=0.01))

        # Check the q-peak value is close to the reference value
        # Interpolate
        q_interp = interp1d(q_range, X)
        test_q_peak_value = q_interp(test_q_per_nm)
        self.assertTrue(np.isclose(test_q_peak_value, peak_value))


if __name__ == '__main__':
    unittest.main()
