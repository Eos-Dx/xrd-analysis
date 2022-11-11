"""
Code to calculate number of bright pixels in an image
"""

import os
import glob

import numpy as np

from eosdxanalysis.preprocessing.image_processing import quantile_count

# Set the minimum brightness threshold
qmin = 0.75

# Set the input path
input_path = ""
files_list = glob.glob(os.path.join(input_path, "*.txt"))
files_list.sort()

for filepath in files_list:
    image = np.loadtxt(filepath, dtype=np.uint32)
    yellow_count = quantile_count(image, qmin=qmin)
    filename = os.path.basename(filepath)
    print("{},{}".format(filename, yellow_count))
