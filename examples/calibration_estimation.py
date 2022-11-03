"""
Code for generating image previews to show theoretical 9.8 A peak locations
using multiple values for the sample-to-detector distance
"""
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from eosdxanalysis.simulations.utils import feature_pixel_location

input_path = ""

files_list = glob.glob(os.path.join(input_path, "*.txt"))
files_list.sort()

distance_mm_list = [
        14.32,
        16.17,
        18.28,
        ]

for filepath in files_list:
    image = np.loadtxt(filepath, dtype=np.uint32)
    title = "Theoretical 9.8 A peak locations for {}".format(os.path.basename(filepath))
    fig = plt.figure(title, figsize=(8,8))
    plt.title(title)
    plt.imshow(image, cmap="hot")

    for idx in range(len(distance_mm_list)):
        distance_mm = distance_mm_list[idx]
        distance_m = distance_mm * 1e-3
        peak_location_9A = feature_pixel_location(
                9.8e-10, distance=distance_m)
        plt.scatter(
            [127.5 + peak_location_9A],
            [127.5],
            label="Theoretical 9.8 A peak location for {} mm distance".format(
                distance_mm)
            )
    plt.legend()
    plt.show()
