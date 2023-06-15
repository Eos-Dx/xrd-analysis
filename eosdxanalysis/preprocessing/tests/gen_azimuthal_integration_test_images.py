import os
import numpy as np

from eosdxanalysis.preprocessing.azimuthal_integration import azimuthal_integration


FILE_PATH = os.path.dirname(__file__)
MODULE_PATH = os.path.join(FILE_PATH, "..")
MODULE_DATA_PATH = os.path.join(MODULE_PATH, "data")
TEST_IMAGE_DIR = "test_images"
TEST_DIR = "test_azimuthal_integration"
TEST_DATA_DIR = "input"
KNOWN_RESULTS_DIR = "known_results"

# Set output path
test_data_path = os.path.join(FILE_PATH, TEST_IMAGE_DIR, TEST_DIR, TEST_DATA_DIR)
# Set known results path
known_results_data_path = os.path.join(FILE_PATH, TEST_IMAGE_DIR, TEST_DIR, KNOWN_RESULTS_DIR)

# Create directory
os.makedirs(test_data_path, exist_ok=True)
os.makedirs(known_results_data_path, exist_ok=True)

shape = (256,256)
x_start = -shape[1]/2 - 0.5
x_end = -x_start
y_start = x_start
y_end = x_end
YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]
RR = np.sqrt(XX**2 + YY**2)
TT = np.arctan2(YY, XX)

known_peak_location = 20

# Test image 1: 2D Gaussian
test_image1 = np.exp(-(RR - known_peak_location)**2)

center = np.array(shape)/2 - 0.5

test_radial_profile1 = azimuthal_integration(test_image1, center=center)

# Test image 2: Half 2D Gaussian
test_image2 = np.exp(-(RR - known_peak_location)**2)
test_image2[TT < 0] = 0

center = np.array(shape)/2 - 0.5

test_radial_profile2 = azimuthal_integration(
        test_image2,
        center=center,
        azimuthal_point_count=90,
        start_angle=-np.pi,
        end_angle=0)

test_image_list = [test_image1, test_image2]
test_radial_profile_list = [test_radial_profile1, test_radial_profile2]

for idx in range(len(test_image_list)):
    test_image = test_image_list[idx]
    test_radial_profile = test_radial_profile_list[idx]

    # Save the test image
    test_image_filepath = "test_image{}.txt".format(idx)
    test_image_output_filepath =  os.path.join(
            test_data_path, test_image_filepath)
    np.savetxt(test_image_output_filepath, test_image1)

    # Save the azimuthal integration profile
    known_results_data_filename = "radial_profile{}.txt".format(idx)
    known_results_data_output_filepath = os.path.join(
            known_results_data_path, known_results_data_filename)
    np.savetxt(known_results_data_output_filepath, test_radial_profile)
