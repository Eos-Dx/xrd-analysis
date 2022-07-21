"""
Sample calculation of the 2D Polar Discrete Fourier Transform
"""
import os
import time

import numpy as np

from scipy.ndimage import map_coordinates
from skimage.transform import warp_polar
from scipy.special import jv

from scipy.signal import wiener

import matplotlib.pyplot as plt

from eosdxanalysis.models.fourier_analysis import pfft2_SpaceLimited
from eosdxanalysis.models.polar_sampling import sampling_grid

t0 = time.time()

MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
DATA_FILENAME = "CRQF_A00823.txt"

N1 = 150 # Radial sampling count
N2 = 151 # Angular sampling count, must be odd
R = 90

image_path = os.path.join(MODULE_PATH, DATA_DIR, DATA_FILENAME)
image = np.loadtxt(image_path, dtype=np.uint32)

# Original image
fig = plt.figure()
plt.imshow(image)
plt.title("Original Cartesian sampling of {}".format(DATA_FILENAME))

# Wiener filtered image
filtered_img = wiener(image, 5)
fig = plt.figure()
plt.imshow(filtered_img)
plt.title("Wiener Filtered of Original Cartesian sampling of {}".format(DATA_FILENAME))

# Now take 2D FFT to compare filtering

# Sample our image according on the Baddour grid
dx = 0.2
dy = 0.2

# Let's create a meshgrid,
# note that x and y have even length
#x = np.arange(-R+dx/2, R+dx/2, dx)
#y = np.arange(-R+dx/2, R+dx/2, dy)
#XX, YY = np.meshgrid(x, y)
#RR = np.sqrt(XX**2 + YY**2)

origin = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

# Now sample the discrete image according to the Baddour polar grid
# First get rmatrix and thetamatrix
rmatrix, thetamatrix = sampling_grid(N1, N2, R)
# Now convert rmatrix to Cartesian coordinates
Xcart = rmatrix*np.cos(thetamatrix)
Ycart = rmatrix*np.sin(thetamatrix)
# Now convert Cartesian coordinates to the array notation
# by shifting according to the origin
Xindices = Xcart + origin[0]
Yindices = origin[1] - Ycart

cart_sampling_indices = [Yindices, Xindices]

fdiscrete = map_coordinates(filtered_img, cart_sampling_indices)

# Plot the original image with the Baddour polar sampling grid in Cartesian coordinates
fig = plt.figure()
plt.imshow(filtered_img)
plt.scatter(Xindices, Yindices, s=1.0)
plt.title("Filtered image with polar DFT sampling grid")

# plt.show()

# exit(0)


t1 = time.time()

print("Start-up time to sample Baddour polar grid:",np.round(t1-t0, decimals=2), "s")

# Calculate the polar dft in frequency space (rho, theta)
pdft = pfft2_SpaceLimited(fdiscrete, N1, N2, R)

# Convert the polar dft to Cartesian frequency space

t2 = time.time()

print("Time to calculate the polar transform:", np.round(t2-t1, decimals=2), "s")


# Warped
polar_image = warp_polar(filtered_img)
fig = plt.figure()
plt.imshow(polar_image)
plt.title("Polar warped sampling of filtered {}".format(DATA_FILENAME))
plt.colorbar()

# Polar DFT
fig = plt.figure()
plt.imshow(20*np.log10(np.abs(pdft)))
plt.title("Magnitude DFT of filtered {} [dB]\n in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))
plt.colorbar()

# Compare to 2D FFT
fft = np.fft.fftshift(np.fft.fft2(filtered_img))

fig = plt.figure()
plt.imshow(20*np.log10(np.abs(fft)))
plt.title("Magnitude FFT of filtered {}\n [dB] in frequency domain".format(DATA_FILENAME))

plt.show()
