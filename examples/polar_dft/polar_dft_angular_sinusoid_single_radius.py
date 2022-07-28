"""
Sample calculation of the 2D Polar Discrete Fourier Transform
"""
import os
import time

import numpy as np

from scipy.ndimage import map_coordinates
from skimage.transform import warp_polar
from scipy.special import jv

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from eosdxanalysis.models.fourier_analysis import pfft2_SpaceLimited
from eosdxanalysis.models.polar_sampling import sampling_grid

t0 = time.time()

MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
DATA_FILENAME = "CRQF_A00823.txt"

N1 = 100
N2 = 101
R = 90

# image_path = os.path.join(MODULE_PATH, DATA_DIR, DATA_FILENAME)
# image = np.loadtxt(image_path, dtype=np.uint32)

function_description = "angular sinusoid"

dx = 0.2
dy = 0.2

# Let's create a meshgrid,
# note that x and y have even length
x = np.arange(-R+dx/2, R+dx/2, dx)
y = np.arange(-R+dx/2, R+dx/2, dy)
XX, YY = np.meshgrid(x, y)

RR = np.sqrt(XX**2 + YY**2)
TT = np.arctan2(YY, XX)

image = np.sin(TT)

origin = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

fig = plt.figure()
plt.imshow(image, cmap="gray")
plt.show()

# Now sample the discrete image according to the Baddour polar grid
# First get rmatrix and thetamatrix
thetamatrix, rmatrix = sampling_grid(N1, N2, R)
# Now convert rmatrix to Cartesian coordinates
Xcart = rmatrix*np.cos(thetamatrix)/dx
Ycart = rmatrix*np.sin(thetamatrix)/dy
# Now convert Cartesian coordinates to the array notation
# by shifting according to the origin
Xindices = Xcart + origin[0]
Yindices = origin[1] - Ycart

cart_sampling_indices = [Yindices, Xindices]

fdiscrete = map_coordinates(image, cart_sampling_indices)

t1 = time.time()

print("Start-up time to sample Baddour polar grid:",np.round(t1-t0, decimals=2), "s")

# Calculate the polar dft
pdft = pfft2_SpaceLimited(fdiscrete, N1, N2, R)

t2 = time.time()

print("Time to calculate the polar transform:", np.round(t2-t1, decimals=2), "s")

# Compare to 2D FFT
fft = np.fft.fftshift(np.fft.fft2(image))



"""
Plotting
"""



# Create periodic color map
colors = ["black", "white", "black"]
periodic_cmap = LinearSegmentedColormap.from_list("", colors)



# Original image
fig = plt.figure()
plt.imshow(image, cmap="gray")
plt.title("Original Cartesian sampling of radial J_1")
plt.colorbar()

# Warped
polar_image = warp_polar(image)
fig = plt.figure()
plt.imshow(polar_image, cmap="gray")
plt.title("Polar warped sampling of radial J_1")
plt.colorbar()

# Polar DFT
fig = plt.figure()
#plt.imshow(20*np.log10(np.abs(pdft)))
#plt.title("Magnitude DFT of {} [dB]\n in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))
plt.imshow(np.abs(pdft), cmap="gray")
plt.title("Magnitude DFT of {}\n in polar frequency domain using Baddour coordinates".format(function_description))
plt.colorbar()

fig = plt.figure()
#plt.imshow(20*np.log10(np.real(pdft)))
#plt.title("Real part of DFT of {}\n [dB] in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))
plt.imshow(np.real(pdft), cmap=periodic_cmap)
plt.title("Real part of DFT of \n in polar frequency domain using Baddour coordinates".format(function_description))
plt.colorbar()

fig = plt.figure()
#plt.imshow(20*np.log10(np.imag(pdft)))
#plt.title("Imaginary part of DFT of {} [dB]\n in polar frequency domain using Baddour coordinates".format(DATA_FILENAME))
plt.imshow(np.imag(pdft), cmap=periodic_cmap)
plt.title("Imaginary part of DFT of {}\n in polar frequency domain using Baddour coordinates".format(function_description))
plt.colorbar()

fig = plt.figure()
plt.imshow(np.angle(pdft), cmap=periodic_cmap)
plt.title("Phase of DFT of {}\n in polar frequency domain using Baddour coordinates".format(function_description))
plt.colorbar()

fig = plt.figure()
plt.imshow(20*np.log10(np.abs(fft)), cmap="gray")
plt.title("Magnitude FFT of {}\n [dB] in frequency domain".format(function_description))
plt.colorbar()

plt.show()
