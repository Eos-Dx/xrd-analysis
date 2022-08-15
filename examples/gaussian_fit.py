"""
Gaussian fit to keratin diffraction patterns

Gaussians:
    1. Background noise
    2. Circular aperture diffraction pattern
    3. Amorphous scattering
    4. 9 A: Looks like a D-shape, how to characterize? Convolution with an arc?
    5. 5 A: 2D Gaussian convolution with an arc
    6. 5-4 A: 2D Gaussian convolution with a circle. Or look into 1D Gaussian rotated
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import abel

from eosdxanalysis.models.utils import cart2pol

PIXEL_WIDTH = 55e-6 # 55 um in [meters]
WAVELENGTH = 1.5418e-10 # 1.5418 Angstrom in [meters]
DISTANCE = 10e-3 # Distance from sample to detector in [meters]

cmap="hot"


def feature_pixel_location(spacing, distance=DISTANCE, wavelength=WAVELENGTH,
            pixel_width=PIXEL_WIDTH):
    """
    Function to calculate the pixel distance from the center to a
    specified feature defined by spacing.

    Inputs:
    - spacing: [meters]
    - distance: sample-to-detector distance [meters]
    - wavelength: source wavelength [meters]

    Returns:
    - distance from center to feature location in pixel units

    Note that since our sampling rate corresponds to pixels,
    pixel units directly correspond to array indices,
    i.e.  distance of 1 pixel = distance of 1 array element
    """
    twoTheta = np.arcsin(wavelength/spacing)
    d_inv = distance * np.tan(twoTheta)
    d_inv_pixels = d_inv / PIXEL_WIDTH
    return d_inv_pixels

# Generate coordinate grid
YY, XX = np.mgrid[-10:10:256j, -10:10:256j]
TT, RR = cart2pol(XX, YY)


"""
Import keratin diffraction image
"""
# Import sample image
MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
DATA_FILENAME = "CRQF_A00005.txt"

image_path = os.path.join(MODULE_PATH, DATA_DIR, DATA_FILENAME)
image = np.loadtxt(image_path, dtype=np.uint32)

# Calculate center coordinates of image in array index notation
center = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

# Plot image
plot_title = DATA_FILENAME
fig = plt.figure(plot_title)
plt.imshow(image, cmap=cmap)
plt.title(plot_title)

# Specify isotropic Gaussian function for 5-4A
r0 = 1.5
width = 0.2
gau_iso = np.exp(-((RR - r0) / width)**2)

# Anisotropic radial Gaussian
r0 = 5
width = 0.5
gau_aniso = np.exp(-((RR - r0) / width)**2)
gau_aniso *= (1 + np.cos(2*TT))/2

"""
Calculate angles corresponding to isotropic 5-4 A feature
"""
d5 = 5e-10 # 5 Angstroms in meters
d5_inv_pixels = feature_pixel_location(d5)


plot_title = "5A feature"
fig = plt.figure(plot_title)
plt.imshow(image, cmap=cmap)
plt.scatter(center[1], center[0] - d5_inv_pixels)
plt.title(plot_title)

"""
9A 
"""

d9 = 9e-10 # 9 Angstroms in meters
d9_inv_pixels = feature_pixel_location(d9)

center = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

plot_title = "9A feature"
fig = plt.figure(plot_title)
plt.imshow(image, cmap=cmap)
plt.scatter(center[1] - d9_inv_pixels, center[0])
plt.title(plot_title)

"""
Plot
"""

plot_title = "9A and 5A features"
fig = plt.figure(plot_title)
plt.imshow(image, cmap=cmap)
plt.scatter(center[1] - d9_inv_pixels, center[0], c="black")
plt.scatter(center[1], center[0] - d5_inv_pixels, c="black")
plt.title(plot_title)

plt.show()
