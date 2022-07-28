"""
Sample calculation of the 2D Polar Discrete Fourier Transform
"""
import os
import time

import numpy as np

from scipy.ndimage import map_coordinates
from skimage.transform import warp_polar
from scipy.special import jv
from scipy.interpolate import griddata

import matplotlib.pyplot as plt

from eosdxanalysis.models.fourier_analysis import pfft2_SpaceLimited
from eosdxanalysis.models.polar_sampling import sampling_grid
from eosdxanalysis.models.polar_sampling import freq_sampling_grid
from eosdxanalysis.models.utils import pol2cart
from eosdxanalysis.models.utils import cart2pol
from eosdxanalysis.preprocessing.image_processing import unwarp_polar

t0 = time.time()

MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
DATA_FILENAME = "CRQF_A00823.txt"

N1 = 102
N2 = 101
R = 90

# image_path = os.path.join(MODULE_PATH, DATA_DIR, DATA_FILENAME)
# image = np.loadtxt(image_path, dtype=np.uint32)

dx = 0.2
dy = 0.2

# Let's create a meshgrid,
# note that x and y have even length
x = np.arange(-R+dx/2, R+dx/2, dx)
y = np.arange(-R+dx/2, R+dx/2, dy)
XX, YY = np.meshgrid(x, y)

TT, RR = cart2pol(XX, YY)

n=20
# image = jv(n, RR)*np.cos(TT)
image = jv(n, RR)

# Sample our image according on the Baddour grid
dx = 0.2
dy = 0.2

origin = (image.shape[0]/2-0.5, image.shape[1]/2-0.5)

"""
Baddour polar grid sampling
"""

# Now sample the discrete image according to the Baddour polar grid
# First get rmatrix and thetamatrix
thetamatrix, rmatrix = sampling_grid(N1, N2, R)
# Now convert rmatrix to Cartesian coordinates
Xcart, Ycart = pol2cart(thetamatrix, rmatrix)
# Now convert Cartesian coordinates to the array notation
# by shifting according to the origin
Xindices = Xcart/dx + origin[0]
Yindices = origin[1] - Ycart/dy

cart_sampling_indices = [Yindices, Xindices]

fdiscrete = map_coordinates(image, cart_sampling_indices)

t1 = time.time()

print("Start-up time to sample Baddour polar grid:",np.round(t1-t0, decimals=2), "s")

"""
Polar DFT
"""

# Calculate the polar dft
pdft = pfft2_SpaceLimited(fdiscrete, N1, N2, R)

t2 = time.time()

print("Time to calculate the polar transform:", np.round(t2-t1, decimals=2), "s")

# Plot the polar DFT on the sample grid in Cartesian frequency space

# Get rhomatrix and psimatrix
psimatrix, rhomatrix = freq_sampling_grid(N1, N2, R)

# Set output size
output_shape = image.shape
output_origin = (output_shape[0]/2-0.5, output_shape[1]/2-0.5)

# Set the maximum Rho of the output plot, same R as original image
R_fraction = R/image.shape[0]/2/dx
Rho = np.max(rhomatrix)
Fdx = R_fraction
Fdy = R_fraction

# Convert to frequeny domain Cartesian coordinates with scaling
FXX, FYY = pol2cart(psimatrix, rhomatrix)

# Create a meshgrid for interpolation
FYnew, FXnew = np.mgrid[-Rho:Rho:pdft.shape[0]*1j, -Rho:Rho:pdft.shape[1]*1j]

# Interpolate
polar_points = np.vstack([FXX.ravel(), FYY.ravel()]).T
polar_values = pdft.ravel()
pdft_cart = griddata(polar_points, polar_values, (FXnew, FYnew), method='cubic')

"""
Plots
"""

# Original image
fig = plt.figure()
plt.imshow(image)
plt.scatter(Xindices, Yindices, s=1.0)
plt.title("Original Cartesian sampling of radial J_1")

# Warped
polar_image = warp_polar(image)
fig = plt.figure()
plt.imshow(polar_image)
plt.title("Polar warped sampling of radial J_1")

# Polar DFT
fig = plt.figure()
plt.imshow(np.abs(pdft), cmap="gray")
plt.title("Magnitude DFT of radial J_1\n in polar frequency domain using Baddour coordinates")

# Frequency domain DFT cartesian grid
fig = plt.figure()
plt.imshow(20*np.log10(np.abs(pdft_cart)), cmap="gray")
clb = plt.colorbar()
plt.title("Polar DFT in frequency domain cartesian grid [dB]")

# Compare to 2D FFT
fft = np.fft.fftshift(np.fft.fft2(image))

fig = plt.figure()
plt.imshow(20*np.log10(np.abs(fft)))
plt.title("Magnitude FFT of {}\n [dB] in frequency domain".format(DATA_FILENAME))

plt.show()
