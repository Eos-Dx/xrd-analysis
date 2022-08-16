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
from scipy.optimize import curve_fit

import abel

from eosdxanalysis.models.utils import cart2pol
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

def radial_gaussian(r, theta, phase, beta,
            peak_radius, width, amplitude, cos_power):
    """
    Isotropic and anisotropic radial Gaussian

    Inputs: (fixed parameters)
    - beta: isotropic = 0, anisotropic = 1
    - phase: either 0 or np.pi/2 to signify equatorial or meridional peak location
    """
    gau = amplitude*np.exp(-((r - peak_radius) / width)**2)
    gau *= np.power(np.cos(theta + phase)**2, beta*cos_power)
    return gau

def keratin_function(polar_point, p0, p1, p2, p3, p4, p5, p6, p7,
        p8, p9, p10, p11, p12, p13, p14, p15):
    """
    Generate entire kertain diffraction pattern
    p has 4 * 4 elements
    which are the arguements to 4 calls of radial_gaussian
    """
    r, theta = polar_point
    # Create four Gaussians, then sum
    # Set phases
    phase_9A = 0.0 # Equatorial peak
    phase_5A = np.pi/2 # Meridional peak
    phase_5_4A = 0.0 # Don't care
    phase_bg = 0.0 # Don't care
    # Set betas
    beta_9A = True # Anisotropic
    beta_5A = True # Anisotropic
    beta_5_4A = False # Isotropic
    beta_bg = False # Isotropic

    approx_9A = radial_gaussian(r, theta, phase_9A, beta_9A, p0, p1, p2, p3)
    approx_5A = radial_gaussian(r, theta, phase_5A, beta_5A, p4, p5, p6, p7)
    approx_5_4A = radial_gaussian(r, theta, phase_5_4A, beta_5_4A, p8, p9, p10, p11)
    approx_bg = radial_gaussian(r, theta, phase_bg, beta_bg, p12, p13, p14, p15)
    approx = approx_9A + approx_5A + approx_5_4A + approx_bg
    return approx.ravel()

def fit_error(p, data, approx, r, theta):
    """
    Returns the square error between a function and
    its approximation.
    p has 4 * 6 elements
    which are the arguements to 4 calls of radial_gaussian
    """
    return np.sum(np.square(data - approx))

def objective(p, data, r, theta):
    """
    Generate a kertain diffraction pattern and
    return how good the fit is.
    """
    approx = keratin_function((r, theta), p)
    return fit_error(p, data, approx, r, theta)



# Specify isotropic Gaussian function for 5-4 A
width_5_4 = 18.0
A5_4 = 7
power_5_4 = 2**1
phase_5_4 = 0
beta_5_4 = False
gau_5_4 = radial_gaussian(RR, TT, phase_5_4, beta_5_4, d5_4_inv_pixels, width_5_4, A5_4, power_5_4)

# Specify anisotropic Gaussian function for 9 A
width_9 = 8
A9 = 10
power_9 = 2**4
phase_9 = 0
beta_9 = True
gau_9 = radial_gaussian(RR, TT, phase_9, beta_9, d9_inv_pixels, width_9, A9, power_9)

# Specify anisotropic Gaussian function for 5 A
width_5 = 5.0
A5 = 2
power_5 = 2**3
phase_5 = np.pi/2
beta_5 = True
gau_5 = radial_gaussian(RR, TT, phase_5, beta_5, d5_inv_pixels, width_5, A5, power_5)

# Specify background noise Gaussian
width_bg = 70.0
Abg = 6
power_bg = 0.0
phase_bg = 0.0
bg_inv_pixels = 0.0
beta_bg = False
gau_bg = radial_gaussian(RR, TT, phase_bg, beta_bg, bg_inv_pixels, width_bg, Abg, power_bg)

# Add gaussians
gau_approx = gau_5_4 + gau_9 + gau_5 + gau_bg

"""
Fit Error Analysis
"""

# P parameters:
#  - peak_radius
#  - width
#  - amplitude
#  - cos_power

# Intial parameters guess
p0 = [
        # 9A equatorial peaks
        30.3, # Peak pixel radius
        7.45, # Width
        1131.2, # Amplitude
        2.0, # cosine power
        # 5A meridional peaks
        6.272287340528552591e+01, # Peak pixel radius
        6.598313320885878852e+00, # Width
        2.739652038901909918e+02, # Amplitude
        2.000000000494934760e+00, # cosine power
        # 5-4A isotropic region
        6.456137737533244092e+01, # Peak pixel radius
        3.319087173851783490e+01, # Width
        8.551915935756256886e+02, # Amplitude
        9.999999999999784062e-02, # cosine power
        # Background noise
        9.995967745626450907e-02, # Peak pixel radius
        2.999690035706361755e+02, # Width
        5.261225560942693136e+01, # Amplitude
        0.01, # cosine power
        ]

# TODO: These bounds should all be a function of exposure time,
# sample-to-detector distance, and molecular spacings
# Order is: 9A, 5A, 5-4A, bg
p_bounds = (
        # Minimum bounds
        np.array([
            # 9A minimum bounds
            25, # 9A peak_radius minimum
            1, # 9A width minimum
            10, # 9A amplitude minimum
            2, # 9A cos^2n power minimum
            # 5A minimum bounds
            50, # 5A peak_radius minimum
            1, # 5A width minimum
            50, # 5A amplitude minimum
            2, # 5A cos^2n power minimum
            # 5-4A minimum bounds
            50,  # 5-4A peak_radius minimum
            2,  # 5-4A width minimum
            20, # 5-4A amplitude minimum
            -1, # 5-4A cos^2n power minimum
            # bg minimum bounds
            -1, # bg peak_radius minimum
            10, # bg width minimum
            10, # bg amplitude minimum
            -1, # bg cos^2n power minimum
            ],
            dtype=np.float64,
        ),
        # Maximum bounds
        np.array([
            # 9A maximum bounds
            40, # 9A peak_radius maximum
            30, # 9A width maximum
            2000, # 9A amplitude maximum
            20, # 9A cos^2n power maximum
            # 5A maximum bounds
            70, # 5A peak_radius maximum
            10, # 5A width maximum
            2000, # 5A amplitude maximum
            12, # 5A cos^2n power maximum
            # 5-4A maximum bounds
            90, # 5-4A peak_radius maximum
            100, # 5-4A width maximum
            2000, # 5-4A amplitude maximum
            1, # 5-4A cos^2n power maximum
            # bg maximum bounds
            1, # bg peak_radius maximum
            300, # bg width maximum
            500, # bg amplitude maximum
            1, # bg cos^2n power maximum
            ],
            dtype=np.float64,
        ),
    )

# Use `scipy.optimize.minimize`
# Object function: objective(p, image, radius, theta)
# p_opt = minimize(objective, p0, args = (image, RR, TT))

# Use `scipy.optimize.curve_fit`
# Function: kertain_function

# Remove meshgrid components that are in the beam center
beam_rmax = 25
mask = create_circular_mask(size, size, rmax=beam_rmax)

RR_masked = RR[~mask].astype(np.float64)
TT_masked = TT[~mask].astype(np.float64)
image_masked = image[~mask].astype(np.float64)

xdata = (RR_masked.ravel(), TT_masked.ravel())
ydata = image_masked.ravel()
popt, pcov = curve_fit(keratin_function, xdata, ydata, p0, bounds=p_bounds)

# Now get "best-fit" diffraction pattern
decomp_opt  = keratin_function((RR, TT), *popt).reshape(size,size)

# Save optimal parameters and image
# np.savetxt("optimal_parameters.txt", popt)
# np.savetxt("best_fit.txt", decomp_opt)

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

plt.savefig(DATA_FILENAME + "_features.png")


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

plt.savefig("filtered_" + DATA_FILENAME + "_features.png")


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

plt.savefig("gaussian_fit_" + DATA_FILENAME + "_features.png")

plot_title = "Optimum fit " + DATA_FILENAME
fig = plt.figure(plot_title)
plt.imshow(decomp_opt, cmap=cmap)
plt.scatter(center[1] - d9_inv_pixels, center[0], c="green", label="9 A")
plt.scatter(center[1], center[0] - d5_inv_pixels, c="blue", label="5 A")
plt.plot([center[1] + d5_inv_pixels, center[1] + d4_inv_pixels],
        [center[0], center[0]], c="white", label="5-4 A")
plt.scatter(center[1] + d5_4_inv_pixels, center[0], c="black", label="4.5 A", zorder=2.0)
plt.legend()
plt.title(plot_title)

plt.savefig("optimum_fit_" + DATA_FILENAME + "_features.png")

plt.show()
