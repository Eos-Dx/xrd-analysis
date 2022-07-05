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
from skimage.transform import warp_polar
from scipy.ndimage import map_coordinates

from eosdxanalysis.preprocessing.image_processing import unwarp_polar
from eosdxanalysis.calibration.materials import q_peaks_ref_dict

"""
Set up parameters
"""

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

# set equipment parameters
WAVELEN_ANG = 1.5418 # 1.5418 Angstrom
DETECTOR_DISTANCE = 10e-3 # 10 mm sample-to-detector distance
PIXEL_WIDTH = 55e-6 # 55 um pixel width

# Set up q-space and initialize intensity array
Q_COUNT = 256
intensities_unscaled = np.zeros((Q_COUNT, 1))
q_space = np.linspace(q_min, q_max, Q_COUNT).reshape(-1,1)


"""
Create 1D radial intensity
"""

# Extract q-peaks reference
q_peaks_ref = q_peaks_ref_dict[material]

# Set ring values according to q-peaks reference as sum of Gaussians
# with a 1/q_peak_location amplitude scaling
# and a width scaling of q_peak location/4
GAUSS_SCALE = q_peaks_ref[0]/4
for q_peak_location in q_peaks_ref:
    gaussian_peak = norm.pdf(q_space, q_peak_location,
                                        GAUSS_SCALE)/q_peak_location
    intensities_unscaled = intensities_unscaled + gaussian_peak

# Rescale intensities
intensities = max_val*intensities_unscaled/np.max(intensities_unscaled)
intensities = intensities.astype(np.uint32)

# Save the intensity vs. q data
np.savetxt(os.path.join(INPUT_DIR, "radial_intensity_vs_q.txt"),
                            intensities, fmt="%d")


"""
Plot 1D radial intensity vs. q
"""

# Save the intensity vs. q scatter plot
# Set up figure properties and title
fig = plt.figure(dpi=100)
fig.set_facecolor("white")
fig.suptitle("Synthetic AgBH Calibration Image")
plt.scatter(q_space, intensities)
plt.title("Intensity vs. q")
plt.xlabel("q")
plt.ylabel("Intensity (photon count)")
# Save the figure
plt.savefig(os.path.join(INPUT_DIR, "radial_intensity_vs_q.png"))


"""
Plot 1D radial intenstiy vs. two*theta
"""

# Convert from q-space to 2*theta space (degrees)
two_theta_space = 2*np.arcsin(q_space*WAVELEN_ANG/4/np.pi)*180/np.pi

# Save the intensity vs. two*theta scatter plot
# Set up figure properties and title
fig = plt.figure(dpi=100)
fig.set_facecolor("white")
fig.suptitle("Synthetic AgBH Calibration Image")
plt.scatter(two_theta_space, intensities)
plt.title("Intensity vs. two*theta")
plt.xlabel("two*theta")
plt.ylabel("Intensity (photon count)")
# Save the figure
plt.savefig(os.path.join(INPUT_DIR, "radial_intensity_vs_two_theta.png"))


"""
Plot 1D radial intensity vs. pixel radius
"""

R_COUNT=Q_COUNT
R_MIN=0
# R_MAX is based on the maximum two*theta value
# Convert two_theta to radians, and get R_MAX in pixel units
R_MAX=DETECTOR_DISTANCE*np.tan(np.pi/180*np.max(two_theta_space))/PIXEL_WIDTH
r_space = np.linspace(R_MIN, R_MAX, R_COUNT).reshape(-1,1)

# Save the intensity vs. pixel radius scatter plot
# Set up figure properties and title
fig = plt.figure(dpi=100)
fig.set_facecolor("white")
fig.suptitle("Synthetic AgBH Calibration Image")
plt.scatter(r_space, intensities)
plt.title("Intensity vs. pixel radius")
plt.xlabel("radius (pixels)")
plt.ylabel("Intensity (photon count)")
# Save the figure
plt.savefig(os.path.join(INPUT_DIR, "radial_intensity_vs_pixel_radius.png"))


"""
Create 2D polar intensity image
"""
polar_intensities = np.repeat(intensities.T, R_COUNT, axis=0)

# Save the data
np.savetxt(os.path.join(INPUT_DIR, "polar_intensity_2d.txt"),
                            polar_intensities, fmt="%d")

# Plot and save
# Set up figure properties and title
fig = plt.figure(dpi=100)
fig.set_facecolor("white")
fig.suptitle("Synthetic AgBH Calibration Image")
plt.imshow(polar_intensities, cmap="gray")
plt.title("2D Polar Intensity: 2*Theta vs. R")
# Save the figure
plt.savefig(os.path.join(INPUT_DIR, "polar_intensity_2d.png"))


"""
Create 2D Cartesian image
"""

image = unwarp_polar(polar_intensities.T, output_shape=(h,w), rmax=R_MAX)

# Crop the image to (h,w) size
row_start, row_end = (int(image.shape[0]/2-h/2), int(image.shape[0]/2+h/2))
col_start, col_end = (int(image.shape[1]/2-w/2), int(image.shape[1]/2+w/2))
assert(row_end - row_start == h)
assert(col_end - col_start == w)
# image = image[row_start:row_end, col_start:col_end]

# Save the data
np.savetxt(os.path.join(INPUT_DIR, "synthetic_calibration_silver_behenate.txt"),
                                        image, fmt="%d")

# Plot and save
# Set up figure properties and title
fig = plt.figure(dpi=100)
fig.set_facecolor("white")
fig.suptitle("Synthetic AgBH Calibration Image")
plt.imshow(image, cmap="gray")
# Save the figure
plt.savefig(os.path.join(INPUT_DIR, "synthetic_calibration_silver_behenate.png"))
