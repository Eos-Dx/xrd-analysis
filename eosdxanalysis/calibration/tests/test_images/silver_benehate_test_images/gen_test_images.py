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

# set equipment parameters
WAVELEN_ANG = 1.5418 # 1.5418 Angstrom
DETECTOR_DISTANCE = 10e-3 # 10 mm sample-to-detector distance
PIXEL_WIDTH = 55e-6 # 55 um pixel width

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
GAUSS_SCALE = q_peaks_ref[0]/4
for q_peak_location in q_peaks_ref:
    gaussian_peak = norm.pdf(q_space, q_peak_location,
                                        GAUSS_SCALE)/q_peak_location
    intensities = intensities + gaussian_peak

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
two_theta_space = 2*np.arcsin(q_space*WAVELEN_ANG/4/np.pi)*180/np.pi

# Save the intensity vs. two*theta scatter plot
# Set up figure properties and title
fig = plt.figure(dpi=100)
fig.set_facecolor("white")
fig.suptitle("Synthetic AgBH Calibration Image")
plt.scatter(two_theta_space, intensities_rescaled)
plt.title("Intensity vs. two*theta")
# Save the figure
plt.savefig(os.path.join(INPUT_DIR, "intensity_vs_two_theta.png"))


# Create a 2D polar image
R_COUNT=Q_COUNT
R_MIN=0
# R_MAX is based on the maximum two*theta value
# Convert two_theta to radians, and get R_MAX in pixel units
R_MAX=int(DETECTOR_DISTANCE*np.tan(np.pi/180*np.max(two_theta_space))/PIXEL_WIDTH)
polar_intensities = np.repeat(intensities_rescaled.T, R_COUNT, axis=0)

# Save the data
np.savetxt(os.path.join(INPUT_DIR, "polar_intensities.txt"),
                            polar_intensities, fmt="%d")

# Plot and save
# Set up figure properties and title
fig = plt.figure(dpi=100)
fig.set_facecolor("white")
fig.suptitle("Synthetic AgBH Calibration Image")
plt.imshow(polar_intensities, cmap="gray")
plt.title("2D Polar Intensity: 2*Theta vs. R")
# Save the figure
plt.savefig(os.path.join(INPUT_DIR, "polar_intensities.png"))

# Convert 2D polar image into 2D Cartesian image

# linear_polar and polar_linear via:
# https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/2

def linear_polar(img, o=None, r=None, output=None, order=1, cont=0):
    if o is None: o = np.array(img.shape[:2])/2 - 0.5
    if r is None: r = (np.array(img.shape[:2])**2).sum()**0.5/2
    if output is None:
        shp = int(round(r)), int(round(r*2*np.pi))
        output = np.zeros(shp, dtype=img.dtype)
    elif isinstance(output, tuple):
        output = np.zeros(output, dtype=img.dtype)
    out_h, out_w = output.shape
    out_img = np.zeros((out_h, out_w), dtype=img.dtype)
    rs = np.linspace(0, r, out_h)
    ts = np.linspace(0, np.pi*2, out_w)
    xs = rs[:,None] * np.cos(ts) + o[1]
    ys = rs[:,None] * np.sin(ts) + o[0]
    map_coordinates(img, (ys, xs), order=order, output=output)
    return output

def polar_linear(img, o=None, r=None, output=None, order=1, cont=0):
    if r is None: r = img.shape[0]
    if output is None:
        output = np.zeros((r*2, r*2), dtype=img.dtype)
    elif isinstance(output, tuple):
        output = np.zeros(output, dtype=img.dtype)
    if o is None: o = np.array(output.shape)/2 - 0.5
    out_h, out_w = output.shape
    ys, xs = np.mgrid[:out_h, :out_w] - o[:,None,None]
    rs = (ys**2+xs**2)**0.5
    ts = np.arccos(xs/rs)
    ts[ys<0] = np.pi*2 - ts[ys<0]
    ts *= (img.shape[1]-1)/(np.pi*2)
    map_coordinates(img, (rs, ts), order=order, output=output)
    return output

image = polar_linear(polar_intensities.T)

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
