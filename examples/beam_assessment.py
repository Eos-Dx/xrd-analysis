"""
Code to assess beam size
"""
import os
import glob

import numpy as np
from skimage.transform import warp_polar

import matplotlib.pyplot as plt


# Set paths
LIBRARY_NAME = "eosdxanalysis"
MODULE_NAME = "preprocessing"
EXAMPLES_PATH = os.path.dirname(__file__)
MODULE_PATH = os.path.join(EXAMPLES_PATH, "..", LIBRARY_NAME, MODULE_NAME)
TEST_IMAGE_PATH = "/home/jfriedman/EosDx/code/xrd-analysis/eosdxanalysis/preprocessing/tests/test_images/"
test_path = os.path.join(TEST_IMAGE_PATH, "test_beam_removal")

# Create test images
INPUT_DIR="input"

# Set the input and output paths
test_input_path = os.path.join(test_path, INPUT_DIR)

input_filepath_list = glob.glob(os.path.join(test_input_path, "C_924*.txt"))
input_filepath_list.sort()

def azimuthal_integration(image, output_shape=(360,128)):
    polar_image = warp_polar(image, output_shape=output_shape, preserve_range=True)
    profile_1d = np.sum(polar_image, axis=0)
    return profile_1d

for input_filepath in input_filepath_list:

    image = np.loadtxt(input_filepath)
    profile_1d = azimuthal_integration(image)
    grad2 = np.gradient(np.gradient(profile_1d))

    fig, axs = plt.subplots(2, 1, sharex=True)
    title = "Beam Assessment: " + os.path.splitext(os.path.basename(input_filepath))[0]
    fig.suptitle(title)

    axs[0].plot(profile_1d)
    axs[0].set_title("Azimuthal Integration")

    ymax = np.median(profile_1d)*2
    axs[0].set_ylim(0, ymax)

    axs[1].plot(grad2, color="red")
    axs[1].set_title("Second Derivative")

    ymax = 10000
    axs[1].set_ylim(-ymax, ymax)

    plt.show()
