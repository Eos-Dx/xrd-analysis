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

from eosdxanalysis.preprocessing.image_processing import pad_image
from eosdxanalysis.preprocessing.image_processing import unwarp_polar
from eosdxanalysis.preprocessing.image_processing import crop_image

from eosdxanalysis.preprocessing.center_finding import circular_average
from eosdxanalysis.preprocessing.center_finding import find_center
from eosdxanalysis.preprocessing.center_finding import find_centroid

from eosdxanalysis.preprocessing.denoising import stray_filter
from eosdxanalysis.preprocessing.denoising import filter_strays

from eosdxanalysis.preprocessing.utils import count_intervals
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.preprocessing.utils import gen_rotation_line
from eosdxanalysis.preprocessing.utils import get_angle

from eosdxanalysis.preprocessing.preprocess import PreprocessData
from eosdxanalysis.preprocessing.preprocess import ABBREVIATIONS
from eosdxanalysis.preprocessing.preprocess import OUTPUT_MAP
from eosdxanalysis.preprocessing.preprocess import INVERSE_OUTPUT_MAP

from eosdxanalysis.preprocessing.peak_finding import find_2d_peak
from eosdxanalysis.preprocessing.peak_finding import find_1d_peaks

from eosdxanalysis.preprocessing.angle_finding import find_rotation_angle

from eosdxanalysis.simulations.utils import feature_pixel_location

# dirname = os.path.dirname(__file__)
# filepath = os.path.join(dirname, "C:\Users\Benjamin\xrd-analysis\eosdxanalysis\preprocessing\tests\test_images\test_beam_removal\input\C_924-1.txt")
# file = open("C_924-1.txt", "r")
#
# print(file.read())

TEST_PATH = os.path.dirname(__file__)
MODULE_PATH = os.path.join(TEST_PATH, "..")
TEST_IMAGE_DIR = "tests/test_images/test_angle_finding/input/C_924-1.txt"
TEST_IMAGE_PATH = os.path.join(TEST_PATH, TEST_IMAGE_DIR)

myFile = open(TEST_IMAGE_PATH, os.O_RDONLY)
myData = os.read(myFile, 105)
myStr = myData.decode("UTF-8")
print(myStr)

