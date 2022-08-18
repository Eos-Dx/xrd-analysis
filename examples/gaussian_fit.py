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
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.optimize import curve_fit

from eosdxanalysis.models.utils import cart2pol
from eosdxanalysis.models.curve_fitting import GaussianDecomposition
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.simulations.utils import feature_pixel_location

cmap="hot"


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
Get best-fit parameters
"""

popt, pcov, RR, TT = GaussianDecomposition.best_fit(image)
decomp_image  = GaussianDecomposition.keratin_function((RR, TT), *popt).reshape(image.shape)

p0 = np.fromiter(GaussianDecomposition.p0_dict.values(), dtype=np.float64)
p_lower_bounds = np.fromiter(GaussianDecomposition.p_lower_bounds_dict.values(), dtype=np.float64)
p_upper_bounds = np.fromiter(GaussianDecomposition.p_upper_bounds_dict.values(), dtype=np.float64)

# Manual fit
p0_dict = GaussianDecomposition.p0_dict
p_guess_dict = p0_dict.fromkeys(p0_dict, 0)
p_guess = np.fromiter(p0_dict.values(), dtype=np.float64)
gau_approx = GaussianDecomposition.keratin_function((RR, TT), *p_guess).reshape(image.shape)




"""
Plot
"""
# Set up meshgrid for 3D plotting
shape = image.shape
x_end = shape[1]/2 - 0.5
x_start = -x_end
y_end = x_end
y_start = x_start
YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]

# Set up feature pixel locations
d9_inv_pixels = feature_pixel_location(9e-10)
d5_inv_pixels = feature_pixel_location(5e-10)
d4_inv_pixels = feature_pixel_location(4e-10)
d5_4_inv_pixels = feature_pixel_location(4.5e-10)

# 3D Plot of filtered image
plot_title = "3D Filtered Image"
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title(plot_title)
surf = ax.plot_surface(XX, YY, filtered_img, cmap=cmap,
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 2e3)
plt.title(plot_title)

# Original image
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

plt.savefig(DATA_FILENAME + "_features.png")

# Filtered image
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

# plt.savefig("filtered_" + DATA_FILENAME + "_features.png")

# Manual Gaussian approximation
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

# plt.savefig("gaussian_fit_" + DATA_FILENAME + "_features.png")

# Optimum Gaussian approximation
plot_title = "Optimum fit " + DATA_FILENAME
fig = plt.figure(plot_title)
plt.imshow(decomp_image, cmap=cmap)
plt.scatter(center[1] - d9_inv_pixels, center[0], c="green", label="9 A")
plt.scatter(center[1], center[0] - d5_inv_pixels, c="blue", label="5 A")
plt.plot([center[1] + d5_inv_pixels, center[1] + d4_inv_pixels],
        [center[0], center[0]], c="white", label="5-4 A")
plt.scatter(center[1] + d5_4_inv_pixels, center[0], c="black", label="4.5 A", zorder=2.0)
plt.legend()
plt.title(plot_title)

# plt.savefig("optimum_fit_" + DATA_FILENAME + "_features.png")

# 3D Plot of optimal image
plot_title = "3D Optimal Image"
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title(plot_title)
surf = ax.plot_surface(XX, YY, decomp_image, cmap=cmap,
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 2e3)
plt.title(plot_title)

plt.show()
