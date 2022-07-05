"""
Script for generating synthetic silver behenate (AgBH) test images

- First, the q-peaks reference is read in
- Then, each peak is represented as a narrow Gaussian
- The total is then a sum of Gaussians
- The final two q-peak locations represents a doublet

Note that this synthetic image is not physically accurate,
it should only be used by peak-detection algorithms.
"""
import os

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Import q-peaks reference
from eosdxanalysis.calibration.materials import q_peaks_ref_dict


# Set the material
material = "silver_behenate"
INNER_RINGS_COUNT = 11
DOUBLET_COUNT = 2

# Set input and output directories
INPUT_DIR="input"
OUTPUT_DIR="output"

# Set up test image parameters
h=256
w=h
min_val=0
max_val=255
q_min=0
q_max = 1.5
wavelen_ang = 1.5418 # 1.5418 Angstrom

# Set up q-space and initialize intensity array
Q_COUNT = 256
intensities = np.zeros((Q_COUNT, 1))
q_space = np.linspace(q_min, q_max, Q_COUNT).reshape(-1,1)

# Extract q-peaks reference
q_peaks_ref = q_peaks_ref_dict[material]

# All but last two peaks, create 1/q intensity fall
# Last two are a doublet, intensity should be
inner_rings = q_peaks_ref[:-2]
doublet = q_peaks_ref[-2:]

# Ensure inner rings and doublet count are good
try:
    assert(len(inner_rings) == INNER_RINGS_COUNT)
    assert(len(doublet) == DOUBLET_COUNT)
except ValueError as err:
    print("Peak counts are incorrect, check against reference!")
    raise err


# Set inner ring values according to q-peaks reference as sum of Gaussians
# with a 1/q_peak amplitude scaling
# and a width scaling of the first peak location/4
GAUSS_INNER_SCALE = q_peaks_ref[0]/4
for q_peak_location in inner_rings:
    gaussian_peak = norm.pdf(q_space, q_peak_location,
                                        GAUSS_INNER_SCALE)/q_peak_location
    intensities = intensities + gaussian_peak

# Set outer doublet value similarly, with a wider Gaussian
GAUSS_DOUBLET_SCALE = q_peaks_ref[0]/(len(inner_rings))
doublet_gaussian_peak = norm.pdf(q_space, np.mean(doublet),
                                        GAUSS_DOUBLET_SCALE)/q_peak_location
intensities = intensities + doublet_gaussian_peak

# Rescale intensities
intensities_rescaled = max_val*intensities/np.max(intensities)
intensities_rescaled = intensities_rescaled.astype(np.uint32)

# Save the intensity vs. q data
np.savetxt(os.path.join(INPUT_DIR, "intensity_vs_q.txt"),
                            intensities_rescaled, fmt="%d")

# Save the intensity vs. q scatter plot
# Set up figure properties and title
fig = plt.figure(dpi=100)
fig.set_facecolor("white")
fig.suptitle("Synthetic AgBH Calibration Image")
plt.scatter(q_space, intensities_rescaled)
plt.title("Intensity vs. q")
# Save the figure
plt.savefig(os.path.join(INPUT_DIR, "intensity_vs_q.png"))

# Convert from q-space to 2*theta space (degrees)
two_theta_space = 2*np.arcsin(q_space*wavelen_ang/4/np.pi)*180/np.pi

# Save the intensity vs. two*theta scatter plot
# Set up figure properties and title
fig = plt.figure(dpi=100)
fig.set_facecolor("white")
fig.suptitle("Synthetic AgBH Calibration Image")
plt.scatter(two_theta_space, intensities_rescaled)
plt.title("Intensity vs. two*theta")
# Save the figure
plt.savefig(os.path.join(INPUT_DIR, "intensity_vs_two_theta.png"))
