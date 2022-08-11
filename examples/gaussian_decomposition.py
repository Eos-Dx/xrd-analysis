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

YY, XX = np.mgrid[-1:1:256j, -1:1:256j]

# TT, RR = cart2pol(XX, YY)

# Create 2D Gaussian

sigma_x = 0.1
sigma_y = 0.1
mu_x = 0
mu_y = 0
rho = 1

# XPP = mu_r * XX/RR
# YPP = mu_r * YY/RR
pos = np.dstack([XX, YY])
rv = multivariate_normal([0.0, 0.0], [[0.001, 0.0], [0.0, 0.001]])
rv2 = multivariate_normal([0.0, 0.0], [[0.0001, 0.0], [0.0, 0.0001]])
rv3 = multivariate_normal([0.0, 0.0], [[0.5, 0.0], [0.0, 0.5]])
gaussian = rv.pdf(pos)
gaussian2 = rv2.pdf(pos)
bg_noise = rv3.pdf(pos)

# Translate gaussian from (0,0) to (x0,y0)
# x0, y0 = (0.25*gaussian.shape[1], 0)
# translation = (x0, y0)
# translation_tform = EuclideanTransform(translation=translation)
# translated_image = warp(gaussian, translation_tform.inverse)

# Draw a curve for convolution along 0.45 < r < 0.55
# and pi/4 < theta < pi/2
TT, RR = cart2pol(XX, YY)
curve = np.zeros(gaussian.shape)
curve2 = np.zeros(gaussian.shape)


# TODO: Make these curves gaussian instead of constant-valued
# First set entire radius values = 1
# 5A region peaks
region_radial = (0.5 + 0.01 < RR) & (RR < 0.5 + 0.02)
region_angular_1 = (np.pi/2 - np.pi/8 < TT) & (TT < np.pi/2 + np.pi/8)
region_angular_2 = (np.pi/2 - np.pi - np.pi/8 < TT) & (TT < np.pi/2 - np.pi + np.pi/8)
region1 = region_radial & region_angular_1
region2 = region_radial & region_angular_2
# curve2[region1] += 1
# curve2[region2] += 1

# 9A region
region2_radial = (0.25 - 0.02 < RR) & (RR < 0.25 + 0.02)
region2_angular_1 = (-np.pi/8 < TT) & (TT < np.pi/8)
region2_angular_2 = (- np.pi + np.pi/8 > TT) | (TT > np.pi - np.pi/8)
region1 = region2_radial & region2_angular_1
region2 = region2_radial & region2_angular_2
curve[region1] += 1
curve[region2] += 1

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

if False:

    fig = plt.figure()
    plt.imshow(gaussian, cmap=cmap)

    fig = plt.figure()
    plt.imshow(curve, cmap=cmap)

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

"""
Swirl
"""
# Translate gaussian from (0,0) to (x0,y0)
x0, y0 = (0.25*gaussian.shape[1], 0)
theta_center = np.arctan2(y0, x0)
translation = (x0, y0)
translation_tform = EuclideanTransform(translation=translation)
translated_image = warp(gaussian, translation_tform.inverse)

# Transform to polar coordinates
N = int(256)
polar_image = warp_polar(translated_image.T, radius=N, output_shape=(N,N))

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
DATA_FILENAME = "CRQF_A00823.txt"
# DATA_FILENAME = "CR_AA00248.txt"

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
percentile = np.percentile(image, 75)
filtered_img[image > percentile] = percentile

filtered_img = gaussian_filter(filtered_img, 2)
for idx in range(5):
    filtered_img = wiener(filtered_img, 2)
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
plt.imshow(image, cmap="gray")
plt.title("Original image")

# 2D Plot of noise-floor-subtracted original image
fig = plt.figure("2D Plot noise-floor-subtracted original")
plt.imshow(noise_floor_subtracted_img, cmap="gray")
plt.title("Original image (noise-floor-subtracted)")

# 2D Plot of low-pass-filtered image
fig = plt.figure("2D Plot low-pass image")
plt.imshow(filtered_img, cmap="gray")
plt.title("Low-pass filtered original image")

# 3D Plot of noise-floor-offset image
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("3D Original Image")
surf = ax.plot_surface(XX, YY, noise_floor_subtracted_img , cmap="gray",
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 1e3)
plt.title("3D Noise-Floor-Offset Original Image")

# 3D Plot of filtered image
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("3D Filtered Image")
surf = ax.plot_surface(XX, YY, filtered_img, cmap="gray",
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
# ax.set_zlim(0, 1e3)
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
