import os
import numpy as np


FILE_PATH = os.path.dirname(__file__)
MODULE_PATH = os.path.join(FILE_PATH, "..")
MODULE_DATA_PATH = os.path.join(MODULE_PATH, "data")
TEST_IMAGE_DIR = "test_images"
TEST_DIR = "test_dead_pixel_repair_images"
TEST_DATA_DIR = "input"
REPAIRED_DIR = "repaired"

# Set output path
test_data_path = os.path.join(FILE_PATH, TEST_IMAGE_DIR, TEST_DIR, TEST_DATA_DIR)

# Set repaired_output_path
repaired_data_path = os.path.join(FILE_PATH, TEST_IMAGE_DIR, TEST_DIR, REPAIRED_DIR)

# Create directory
os.makedirs(test_data_path, exist_ok=True)
os.makedirs(repaired_data_path, exist_ok=True)

# Generate test images
size = 5

for idx in range(size):
    # Create test image
    test_image = np.ones((size,size))
    test_image[idx,idx] = 1e6

    # Create repaired image
    repaired_image = np.ones((size,size))
    repaired_image[idx,idx] = np.nan

    # Save test image
    test_filename = "test_image_{}.txt".format(idx)
    test_filepath = os.path.join(test_data_path, test_filename)
    np.savetxt(test_filepath, test_image)

    # Save repaired image
    repaired_filename = "repaired_image_{}.txt".format(idx)
    repaired_filepath = os.path.join(repaired_data_path, repaired_filename)
    np.savetxt(repaired_filepath, repaired_image)
