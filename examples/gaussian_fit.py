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
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize

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


"""
Import keratin diffraction image
"""
# Import sample image
MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
DATA_FILENAME = "CRQF_A00005.txt"

image_path = os.path.join(MODULE_PATH, DATA_DIR, DATA_FILENAME)
image = np.loadtxt(image_path, dtype=np.uint32)

filtered_img = gaussian_filter(image, 2)

# Calculate center coordinates of image in array index notation
center = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

# Plot image
plot_title = DATA_FILENAME
fig = plt.figure(plot_title)
plt.imshow(image, cmap=cmap)
plt.title(plot_title)


"""
Calculate pixel radius corresponding to 5 A peak
"""
d5 = 5e-10 # 5 Angstroms in [meters]
d5_inv_pixels = feature_pixel_location(d5)

"""
Calculate pixel radius corresponding to 9 A peak
"""
d9 = 9e-10 # 9 Angstroms in [meters]
d9_inv_pixels = feature_pixel_location(d9)

"""
Calculate start and end radius corresponding to isotropic 5-4 A feature
"""
# We already have d5_inv_pixels,
# so just calculate d4
d4 = 4e-10 # 4 Angstroms in [meters]
d4_inv_pixels = feature_pixel_location(d4)

"""
Calculate radius corresponding to 4.5 A, which we take
to be the center of the 5-4 A isotropic ring
"""
d5_4 = 4.5e-10 # 4.5 Angstroms in [meters]
d5_4_inv_pixels = feature_pixel_location(d5_4)




"""
Gaussian fit
"""

# Generate coordinate grid
size = 256
x_end = size/2 - 0.5
x_start = -x_end
y_end = x_end
y_start = x_start
YY, XX = np.mgrid[y_start:y_end:size*1j, x_start:x_end:size*1j]
TT, RR = cart2pol(XX, YY)

def radial_gaussian(r, theta, peak_radius, width, amplitude,
            cos_power=0, phase=0, iso=False):
    """
    Isotropic and anisotropic radial Gaussian
    """
    gau = amplitude*np.exp(-((r - peak_radius) / width)**2)
    # If the function is isotropic, return
    if iso:
        return gau
    # Anisotropic case
    gau *= np.power(np.cos(theta + phase), 2**cos_power)
    return gau

def fit_error(func, approx):
    """
    Returns the square error between a function and
    its approximation.
    """
    return np.sum(np.square(func - approx))

def objective(p, image, r, theta):
    """
    """
    # Create four Gaussians, then sum
    approx_9A = radial_gaussian(r, theta, p[0], p[1], p[2], p[3], p[4], p[5])
    approx_5A = radial_gaussian(r, theta, p[6], p[7], p[8], p[9], p[10], p[11])
    approx_5_4A = radial_gaussian(r, theta, p[12], p[13], p[14], p[15], p[16], p[17])
    approx_bg = radial_gaussian(r, theta,  p[18], p[19], p[20], p[21], p[22], p[23])
    approx = approx_9A + approx_5A + approx_5_4A + approx_bg
    return fit_error(image, approx)


# Specify isotropic Gaussian function for 5-4 A
width_5_4 = 18.0
A5_4 = 7
power_2n5_4 = 1
phase_5_4 = 0
gau_5_4 = radial_gaussian(RR, TT, d5_4_inv_pixels, width_5_4, A5_4, iso=True)

# Specify anisotropic Gaussian function for 9 A
width_9 = 8
A9 = 10
power_2n9 = 4
phase_9 = 0
gau_9 = radial_gaussian(RR, TT, d9_inv_pixels, width_9, A9, power_2n9)

# Specify anisotropic Gaussian function for 5 A
width_5 = 5.0
A5 = 2
power_2n5 = 3
phase_5 = np.pi/2
gau_5 = radial_gaussian(RR, TT, d5_inv_pixels, width_5, A5, power_2n5, phase_5)

# Specify background noise Gaussian
width_bg = 70.0
Abg = 6
gau_bg = radial_gaussian(RR, TT, 0, width_bg, Abg, iso=True)

# Add gaussians
gau_approx = gau_5_4 + gau_9 + gau_5 + gau_bg

"""
Fit Error Analysis
"""

p0 = [
    d9_inv_pixels, # peak_radius
    width_9, # width
    A9, #amplitude
    power_2n9, # cos_power=0
    phase_9, # phase=0
    0, # iso=False
    d5_inv_pixels, # peak_radius
    width_5, # width
    A5, #amplitude
    power_2n5, # cos_power=0
    phase_5, # phase=0
    0, # iso=False
    d5_4_inv_pixels, # peak_radius
    width_5_4, # width
    A5_4, #amplitude
    power_2n5_4, # cos_power=0
    phase_5_4, # phase=0
    1, # iso=False
    0, # peak_radius
    width_bg, # width
    Abg, #amplitude
    0, # cos_power=0
    0, # phase=0
    1, # iso=False
    ]

# Object function: objective(p, image, radius, theta)
p_opt = minimize(objective, p0, args = (image, RR, TT))


"""
Plot
"""

plot_title = DATA_FILENAME 
fig = plt.figure(plot_title)
plt.imshow(image, cmap=cmap)
plt.scatter(center[1] - d9_inv_pixels, center[0], c="green", label="9 A")
plt.scatter(center[1], center[0] - d5_inv_pixels, c="blue", label="5 A")
plt.plot([center[1] + d5_inv_pixels, center[1] + d4_inv_pixels],
        [center[0], center[0]], c="white", label="5-4 A")
plt.scatter(center[1] + d5_4_inv_pixels, center[0], c="black", label="4.5 A")
plt.legend()
plt.title(plot_title)

plt.savefig(DATA_FILENAME + "_features.png", cmap=cmap)


plot_title = "Filtered " + DATA_FILENAME
fig = plt.figure(plot_title)
plt.imshow(filtered_img, cmap=cmap)
plt.scatter(center[1] - d9_inv_pixels, center[0], c="green", label="9 A")
plt.scatter(center[1], center[0] - d5_inv_pixels, c="blue", label="5 A")
plt.plot([center[1] + d5_inv_pixels, center[1] + d4_inv_pixels],
        [center[0], center[0]], c="white", label="5-4 A")
plt.scatter(center[1] + d5_4_inv_pixels, center[0], c="black", label="4.5 A")
plt.legend()
plt.title(plot_title)

plt.savefig("filtered_" + DATA_FILENAME + "_features.png", cmap=cmap)


plot_title = "Gaussian approximation " + DATA_FILENAME 
fig = plt.figure(plot_title)
plt.imshow(gau_approx, cmap=cmap)
plt.scatter(center[1] - d9_inv_pixels, center[0], c="green", label="9 A")
plt.scatter(center[1], center[0] - d5_inv_pixels, c="blue", label="5 A")
plt.plot([center[1] + d5_inv_pixels, center[1] + d4_inv_pixels],
        [center[0], center[0]], c="white", label="5-4 A")
plt.scatter(center[1] + d5_4_inv_pixels, center[0], c="black", label="4.5 A", zorder=2.0)
plt.legend()
plt.title(plot_title)

plt.savefig("gaussian_fit_" + DATA_FILENAME + "_features.png", cmap=cmap)

plt.show()
