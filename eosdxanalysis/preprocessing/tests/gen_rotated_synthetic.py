"""
Generates a synthetic test image
"""
import os
import numpy as np
from collections import OrderedDict

from skimage.transform import rotate

from eosdxanalysis.models.curve_fitting import gen_meshgrid
from eosdxanalysis.models.curve_fitting import keratin_function
from eosdxanalysis.simulations.utils import feature_pixel_location


FILE_PATH = os.path.dirname(__file__)
MODULE_PATH = os.path.join(FILE_PATH, "..")
MODULE_DATA_PATH = os.path.join(MODULE_PATH, "data")
TEST_IMAGE_DIR = "test_images"
TEST_DIR = "test_preprocessing_images"
INPUT_DIR = "input"
FILENAME = "rotated_synthetic.txt"

output_path = os.path.join(FILE_PATH, TEST_IMAGE_DIR, TEST_DIR, INPUT_DIR)

p_synth_dict = OrderedDict({
    # 9A equatorial peaks parameters
    "peak_location_radius_9A":  feature_pixel_location(9.8e-10), # Peak pixel radius
    "peak_std_9A":              7, # Width
    "peak_amplitude_9A":        401, # Amplitude
    "arc_angle_9A":             1e-2, # Arc angle
    # 5A meridional peaks parameters
    "peak_location_radius_5A":  feature_pixel_location(5.1e-10), # Peak pixel radius
    "peak_std_5A":              1, # Width
    "peak_amplitude_5A":        13, # Amplitude
    "arc_angle_5A":             np.pi/4, # Arc angle
    # 5-4A isotropic region parameters
    "peak_location_radius_5_4A":feature_pixel_location(4.5e-10), # Peak pixel radius
    "peak_std_5_4A":            17, # Width
    "peak_amplitude_5_4A":      113, # Amplitude
    # Background noise parameters
    "peak_std_bg":              211, # Width
    "peak_amplitude_bg":        223, # Amplitude
    })

# Set mesh size
size = 256
RR, TT = gen_meshgrid((size,size))

# Generate synthetic image
synth_image = keratin_function((RR, TT), *p_synth_dict.values()).reshape(RR.shape)

angle = 135
rotated_image = rotate(synth_image, angle)

output_filepath = os.path.join(output_path, FILENAME)
np.savetxt(output_filepath, rotated_image, fmt="%d")
