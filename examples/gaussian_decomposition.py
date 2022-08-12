"""
Sample calculation of the 2D Polar Discrete Fourier Transform
"""
import os
import time

import numpy as np

from scipy.ndimage import map_coordinates
from scipy.ndimage import uniform_filter
from scipy.ndimage import sobel
from scipy.ndimage import gaussian_filter
from skimage.transform import warp_polar
from skimage.transform import warp
from skimage.transform import EuclideanTransform
from skimage.transform import AffineTransform
from skimage.transform import rotate
from scipy.special import jv
from scipy.interpolate import griddata
from scipy.signal import wiener
from scipy.signal import convolve
from scipy import signal
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt

from eosdxanalysis.models.fourier_analysis import pfft2_SpaceLimited
from eosdxanalysis.models.fourier_analysis import ipfft2_SpaceLimited
from eosdxanalysis.models.polar_sampling import sampling_grid
from eosdxanalysis.models.polar_sampling import freq_sampling_grid
from eosdxanalysis.models.utils import pol2cart
from eosdxanalysis.models.utils import cart2pol
from eosdxanalysis.preprocessing.image_processing import unwarp_polar

t0 = time.time()

"""
Gaussian Synthesis functions
"""

# Construct Cartesian and polar meshgrids
# y goes from 1 to -1, rows
# x goes from -1 to 1, columns
YY, XX = np.mgrid[1:-1:256j, -1:1:256j]
TT, RR = cart2pol(XX, YY)

# Create Gaussians of each feature:
# 9A< 5A, and 5-4A

# Construct 9A Gaussian peak
pos_5A = np.dstack([XX, YY])
rv_5A = multivariate_normal([0.0, 0.0], [[0.001, 0.0], [0.0, 0.001]])
gaussian_5A = rv_5A.pdf(pos_5A)

# Draw a curve for convolution along  rmin_9A < r < rmax_9A
# and theta_min_9A < theta < theta_max_9A

# Set some constants
r_5A = 0.5
dr_5A = 0.01
curve_5A = np.zeros(gaussian_5A.shape)
theta1_5A = np.pi/2
theta2_5A = -np.pi/2
dtheta_5A = np.pi/4

# Set curve = 1 in defined template regions
# Define radial region of interest
radial_5A = (r_5A - dr_5A < RR) & (RR < r_5A + dr_5A)
# Define disjoint angular regions of interest
angular1_5A = (theta1_5A - dtheta_5A/2 < TT) & (TT < theta1_5A + dtheta_5A/2)
angular2_5A = (theta2_5A - dtheta_5A/2 < TT) & (TT < theta2_5A + dtheta_5A/2)
# Combine regions of interest
region_5A = radial_5A & (angular1_5A | angular2_5A)
# Set curve = 1
curve_5A[region_5A] += 1

# Take 5A convolution
gaussian_5A_feature = convolve(gaussian_5A, curve_5A, mode="same")
# Normalize
gaussian_5A_feature /= np.max(gaussian_5A_feature)


# Construct 9A Gaussian peak

region_radial = (0.5 + 0.01 < RR) & (RR < 0.5 + 0.02)
region_angular_1 = (np.pi/2 - np.pi/8 < TT) & (TT < np.pi/2 + np.pi/8)
region_angular_2 = (np.pi/2 - np.pi - np.pi/8 < TT) & (TT < np.pi/2 - np.pi + np.pi/8)
region1 = region_radial & region_angular_1
region2 = region_radial & region_angular_2


if False:
    curve2 = np.zeros(gaussian.shape)

    rv2 = multivariate_normal([0.0, 0.0], [[0.0001, 0.0], [0.0, 0.0001]])
    rv3 = multivariate_normal([0.0, 0.0], [[0.5, 0.0], [0.0, 0.5]])
    gaussian2 = rv2.pdf(pos)
    bg_noise = rv3.pdf(pos)

    # Translate gaussian from (0,0) to (x0,y0)
    # x0, y0 = (0.25*gaussian.shape[1], 0)
    # translation = (x0, y0)
    # translation_tform = EuclideanTransform(translation=translation)
    # translated_image = warp(gaussian, translation_tform.inverse)


    # 5-4 A region diffuse scattering
    region3_radial = (0.5 < RR)
    curve[region3_radial] += 0.5

    # Now put Gaussian decay
    curve[region3_radial] *= np.exp(-(2*RR[region3_radial])**2)
    # Add in background noise
    curve += bg_noise/np.max(bg_noise)/10

    # Convolution
    polar_gaussian = convolve(gaussian, curve, mode="same")
    # Normalize
    polar_gaussian /= np.max(polar_gaussian)

    # Convolution 2
    polar_gaussian2 = convolve(gaussian2, curve2, mode="same")
    # Normalize
    polar_gaussian2 /= np.max(polar_gaussian2)

cmap = "hot"

if True:

    plot_title = "2D Gaussian"
    fig = plt.figure(plot_title)
    plt.imshow(gaussian_5A, cmap=cmap)
    plt.title(plot_title)

    plot_title = "5A Peaks Template"
    fig = plt.figure(plot_title)
    plt.imshow(curve_5A, cmap=cmap)
    plt.title(plot_title)

    plot_title = "5A Gaussian Peaks"
    fig = plt.figure(plot_title)
    plt.imshow(gaussian_5A_feature, cmap=cmap)
    plt.title(plot_title)

    # 3D Gaussian 5A Feature
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.canvas.manager.set_window_title("3D Gaussian 5A Feature")
    surf = ax.plot_surface(XX, YY, gaussian_5A_feature, cmap=cmap,
                                   linewidth=0, antialiased=False)
    clb = fig.colorbar(surf)
    # ax.set_zlim(0, 1.5)
    plt.title("3D Gaussian 5A Feature")

    if False:
        fig = plt.figure()
        plt.imshow(polar_gaussian, cmap=cmap)

        plt.imsave("synthetic_peaks.png", polar_gaussian, cmap=cmap)

        # 3D Plot of result
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.canvas.manager.set_window_title("2D Gaussian Polar Convolution")
        surf = ax.plot_surface(XX, YY, polar_gaussian, cmap=cmap,
                                       linewidth=0, antialiased=False)
        clb = fig.colorbar(surf)
        # ax.set_zlim(0, 1.5)
        plt.title("2D Gaussian Polar Convolution")

# plt.show()

# exit(0)

"""
Radial profiles
"""
# Import sample image
MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
DATA_FILENAME = "CRQF_A00005.txt"

image_path = os.path.join(MODULE_PATH, DATA_DIR, DATA_FILENAME)
image = np.loadtxt(image_path, dtype=np.uint32)

# Plot image
plot_title = DATA_FILENAME
fig = plt.figure(plot_title)
plt.imshow(image, cmap=cmap)
plt.title(plot_title)

# Define window properties
center = (image.shape[0]//2, image.shape[1]//2)
window_thickness = 4*2
window_length = image.shape[0]//2
row_start = int(center[0] - window_thickness/2)
row_end = int(center[0] + window_thickness/2)
window_rows = slice(row_start, row_end)
col_start = int(center[1])
col_end = int(image.shape[1])
window_cols = slice(col_start, col_end)

# Rotate image twice to take horizontal windows
image_45 = rotate(image, angle=45, resize=False, preserve_range=True, mode='constant')
image_90 = rotate(image, angle=90, resize=False, preserve_range=True, mode='constant')

# Take radial profiles, average across rows axis=0
radial_0 = image[window_rows, window_cols]
radial_0 = np.mean(radial_0, axis=0)
radial_45 = image_45[window_rows, window_cols]
radial_45 = np.mean(radial_45, axis=0)
radial_90 = image_90[window_rows, window_cols]
radial_90 = np.mean(radial_90, axis=0)

# Plot intensity vs. pixel radius

plot_title = DATA_FILENAME + " Radial profile: 0 degrees"
fig = plt.figure(plot_title)
plt.plot(np.arange(window_length), radial_0)
plt.title(plot_title)

plot_title = DATA_FILENAME + " Radial profile: 45 degrees"
fig = plt.figure(plot_title)
plt.plot(np.arange(window_length), radial_45)
plt.title(plot_title)

plot_title = DATA_FILENAME + " Radial profile: 90 degrees"
fig = plt.figure(plot_title)
plt.plot(np.arange(window_length), radial_90)
plt.title(plot_title)


# Full radial profile
polar_image = warp_polar(image, radius=window_length, preserve_range=True)
radial_profile = np.sum(polar_image, axis=0)

# Plot polar image
plot_title = "Polar warped image"
fig = plt.figure(plot_title)
plt.imshow(polar_image, cmap=cmap)
plt.title(plot_title)

# Plot radial profile
plot_title = "Radial profile"
fig = plt.figure(plot_title)
plt.plot(np.arange(window_length), radial_profile)
plt.title(plot_title)


"""
Swirl
"""
if False:
    # Translate gaussian from (0,0) to (x0,y0)
    x0, y0 = (0.25*gaussian.shape[1], 0)
    theta_center = np.arctan2(y0, x0)
    translation = (x0, y0)
    translation_tform = EuclideanTransform(translation=translation)
    translated_image = warp(gaussian, translation_tform.inverse)

    # Transform to polar coordinates
    Nsize = int(256)
    polar_image = warp_polar(translated_image.T, radius=Nsize, output_shape=(Nsize,Nsize))

    # Unwarp
    output_shape = gaussian.shape
    sheared_image = unwarp_polar(polar_image, output_shape=output_shape).T

    fig = plt.figure(1)
    plt.imshow(translated_image)

    fig = plt.figure(2)
    plt.imshow(polar_image)

    fig = plt.figure(4)
    plt.imshow(sheared_image)

    plt.show()

    exit(0)


"""
Keratin pattern
"""

MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
DATA_FILENAME = "CRQF_A00005.txt"

image_path = os.path.join(MODULE_PATH, DATA_DIR, DATA_FILENAME)
image = np.loadtxt(image_path, dtype=np.uint32)
# filtered_img = wiener(image, 15)

# Create meshgrid for 3D plot
YY, XX = np.mgrid[:256, :256]

# Estimate background noise
intensities = np.unique(image)
# Take the second element
noise_floor = intensities[1]
noise_floor_subtracted_img = image.copy()
noise_floor_subtracted_img[image > 0] = image[image > 0] - noise_floor

# filtered_img = wiener(noise_floor_subtracted_img, 15)
# Take low-pass filter (remove high frequencies)
# Get IIR filter coefficients for Butterworth filter
# sos = signal.butter(3, 0.1, 'lowpass', output='sos')
# Filter image
# filtered_img = signal.sosfilt(sos, noise_floor_subtracted_img)
# filtered_img = uniform_filter(noise_floor_subtracted_img, size=21)
filtered_img = noise_floor_subtracted_img

# Percentile clipping
# percentile = np.percentile(image, 99)
# filtered_img[image > percentile] = percentile

filtered_img = gaussian_filter(filtered_img, 2)
# filtered_img = wiener(filtered_img, 15)
for idx in range(0):
    # filtered_img = wiener(filtered_img, 2)
    filtered_img = gaussian_filter(filtered_img, 2)

"""
Polar DFT Grid
"""
if False:

    N1 = 201 # radial sampling count
    N2 = 15 # angular sampling count
    R = 90

    # Sample the image according to the Baddour polar grid
    # Get rmatrix and thetamatrix
    thetamatrix, rmatrix = sampling_grid(N1, N2, R)
    # Convert to Cartesian
    Xcart, Ycart = pol2cart(thetamatrix, rmatrix)
    # Wrap/extend for plotting
    Xcart_ext = np.vstack([Xcart, Xcart[0,:]])
    Ycart_ext = np.vstack([Ycart, Ycart[0,:]])

    # row, col
    origin = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

    Xindices = Xcart + origin[0]
    Yindices = origin[1] - Ycart

    cart_sampling_indices = [Yindices, Xindices]

    # Interpolate
    img_sampled = map_coordinates(filtered_img, cart_sampling_indices)

"""
Take polar DFT
"""
if False:
    pdft = pfft2_SpaceLimited(img_sampled, N1, N2, R)
    t1 = time.time()
    print("Time to calculate the polar transform:", np.round(t1-t0, decimals=2), "s")

    # Wrap/extend for plotting
    pdft_ext = np.vstack([pdft, pdft[0,:]])

    psimatrix, rhomatrix = freq_sampling_grid(N1, N2, R)
    FX, FY = pol2cart(psimatrix, rhomatrix)
    FX_ext = np.vstack([FX, FX[0,:]])
    FY_ext = np.vstack([FY, FY[0,:]])

    # Convert to frequeny domain Cartesian coordinates with scaling
    FXX, FYY = pol2cart(psimatrix, rhomatrix)

"""
Take Inverse polar DFT
"""
if False:
    ipdft = ipfft2_SpaceLimited(img_sampled, N1, N2, R)
    ipdft_ext = np.vstack([ipdft, ipdft[0,:]])

"""
Classic DFT
"""
if False:
    dft = np.fft.fft2(image)
    # Get frequencies
    FFrows = np.fft.fftshift(np.broadcast_to(np.fft.fftfreq(dft.shape[0],d=2), dft.shape).T)
    FFcols = np.fft.fftshift(np.broadcast_to(np.fft.fftfreq(dft.shape[1],d=2), dft.shape))
    # Rows and Cols
    Frowindices, Fcolindices = np.mgrid[:image.shape[0], :image.shape[1]]

"""
Plots
"""

# 2D Plot of original image
fig = plt.figure("2D Plot original")
plt.imshow(image, cmap=cmap)
plt.title("Original image")

# 2D Plot of noise-floor-subtracted original image
fig = plt.figure("2D Plot noise-floor-subtracted original")
plt.imshow(noise_floor_subtracted_img, cmap=cmap)
plt.title("Original image (noise-floor-subtracted)")

# 2D Plot of low-pass-filtered image
fig = plt.figure("2D Plot low-pass image")
plt.imshow(filtered_img, cmap=cmap)
plt.title("Low-pass filtered original image")

# 3D Plot of original image
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("3D Original Image")
surf = ax.plot_surface(XX, YY, image , cmap=cmap,
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 2e3)
plt.title("3D Original Image")

# 3D Plot of filtered image
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("3D Filtered Image")
surf = ax.plot_surface(XX, YY, filtered_img, cmap=cmap,
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 2e3)
plt.title("3D Filtered Image")

if False:

    # 3D Plot of DFT
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.canvas.manager.set_window_title("3D Polar DFT")
    surf = ax.plot_surface(FX_ext, FY_ext, 20*np.log10(np.abs(pdft_ext)), cmap="gray",
                                   linewidth=0, antialiased=False)
    clb = fig.colorbar(surf)
    # ax.set_zlim(0, 1.5)
    plt.title("3D Polar DFT (Frequency Domain, Polar Grid)")

    # 2D Classic DFT
    fig = plt.figure("2D Classic DFT")
    plt.imshow(20*np.log10(np.abs(np.fft.fftshift(dft))), cmap="gray")
    plt.title("2D Classic DFT")

    # 3D Plot classic DFT
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.canvas.manager.set_window_title("3D Classic DFT")
    surf = ax.plot_surface(Fcolindices, Frowindices, 20*np.log10(np.fft.fftshift(np.abs(dft))), cmap="gray",
                                   linewidth=0, antialiased=False)
    clb = fig.colorbar(surf)
    # ax.set_zlim(0, 1.5)
    plt.title("3D Classic DFT (Frequency Domain, Cartesian Grid)")

    # 3D Plot of inverse polar DFT
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.canvas.manager.set_window_title("3D Inverse Polar DFT")
    # Take the real part
    surf = ax.plot_surface(Xcart_ext, Ycart_ext, np.abs(ipdft_ext), cmap="gray",
                                   linewidth=0, antialiased=False)
    clb = fig.colorbar(surf)
    # ax.set_zlim(0, 1.5)
    plt.title("3D Inverse Polar DFT (Spatial Domain, Polar Grid)")

plt.show()
