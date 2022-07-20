"""
Sample calculation of the 2D Polar Discrete Fourier Transform
"""
import os

import numpy as np

from scipy.ndimage import map_coordinates
from skimage.transform import warp_polar

import matplotlib.pyplot as plt

from eosdxanalysis.models.fourier_analysis import pfft2_SpaceLimited
from eosdxanalysis.models.polar_sampling import sampling_grid

MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
DATA_FILENAME = "CRQF_A00823.txt"

N1 = 100
N2 = 101
R = 90

image_path = os.path.join(MODULE_PATH, DATA_DIR, DATA_FILENAME)
image = np.loadtxt(image_path, dtype=np.uint32)

# Sample our image according on the Baddour grid
dx = 0.2
dy = 0.2

# Let's create a meshgrid,
# note that x and y have even length
x = np.arange(-R+dx/2, R+dx/2, dx)
y = np.arange(-R+dx/2, R+dx/2, dy)
XX, YY = np.meshgrid(x, y)

RR = np.sqrt(XX**2 + YY**2)

origin = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

# Now sample the discrete image according to the Baddour polar grid
# First get rmatrix and thetamatrix
rmatrix, thetamatrix = sampling_grid(N1, N2, R)
# Now convert rmatrix to Cartesian coordinates
Xcart = rmatrix*np.cos(thetamatrix)/dx
Ycart = rmatrix*np.sin(thetamatrix)/dy
# Now convert Cartesian coordinates to the array notation
# by shifting according to the origin
Xindices = Xcart + origin[0]
Yindices = origin[1] - Ycart

cart_sampling_indices = [Yindices, Xindices]

fdiscrete = map_coordinates(image, cart_sampling_indices)


# Calculate the polar dft
pdft = pfft2_SpaceLimited(fdiscrete, N1, N2, R)

# Original image
fig = plt.figure()
plt.imshow(image)
plt.title("Original Cartesian sampling of {}".format(DATA_FILENAME))

# Warped
polar_image = warp_polar(image)
fig = plt.figure()
plt.imshow(polar_image)
plt.title("Polar warped sampling of {}".format(DATA_FILENAME))

# Polar DFT
fig = plt.figure()
#plt.imshow(20*np.log10(np.abs(pdft)))
#plt.title("Magnitude DFT of {} [dB]\n in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))
plt.imshow(np.abs(pdft))
plt.title("Magnitude DFT of {}\n in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))

fig = plt.figure()
#plt.imshow(20*np.log10(np.real(pdft)))
#plt.title("Real part of DFT of {}\n [dB] in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))
plt.imshow(np.real(pdft))
plt.title("Real part of DFT of {}\n in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))

fig = plt.figure()
#plt.imshow(20*np.log10(np.imag(pdft)))
#plt.title("Imaginary part of DFT of {} [dB]\n in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))
plt.imshow(np.imag(pdft))
plt.title("Imaginary part of DFT of {}\n in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))

fig = plt.figure()
plt.imshow(np.angle(pdft))
plt.title("Phase of DFT of {}\n in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))

# Compare to 2D FFT
fft = np.fft.fftshift(np.fft.fft2(image))

fig = plt.figure()
plt.imshow(20*np.log10(np.abs(fft)))
plt.title("Magnitude FFT of {}\n [dB] in frequency domain".format(DATA_FILENAME))

plt.show()
