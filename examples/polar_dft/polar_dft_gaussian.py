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
from eosdxanalysis.models.fourier_analysis import ipfft2_SpaceLimited
from eosdxanalysis.models.polar_sampling import sampling_grid
from eosdxanalysis.models.polar_sampling import freq_sampling_grid
from eosdxanalysis.models.utils import pol2cart
from eosdxanalysis.models.utils import cart2pol
from eosdxanalysis.preprocessing.image_processing import unwarp_polar

t0 = time.time()

N1 = 383
N2 = 15
R = 40
a = 0.1

# Gaussian input function and it's polar DFT
gau = lambda x, a: np.exp(-(x/a)**2)
gau2 = lambda x, a: np.pi/(a**2)*np.exp(-((x/a)**2)/4)


"""
Baddour polar grid sampling
"""

# Now sample the discrete image according to the Baddour polar grid
# First get rmatrix and thetamatrix
thetamatrix, rmatrix = sampling_grid(N1, N2, R)
# Get rhomatrix and psimatrix
psimatrix, rhomatrix = freq_sampling_grid(N1, N2, R)

# Now convert rmatrix to Cartesian coordinates
Xcart, Ycart = pol2cart(thetamatrix, rmatrix)
# Convert to frequeny domain Cartesian coordinates with scaling
FXX, FYY = pol2cart(psimatrix, rhomatrix)


# Create the image
fdiscrete = gau(rmatrix, 1.0)
XX, YY = np.mgrid[-R:R:N1*1j, -R:R:N1*1j]
TT, RR = cart2pol(XX, YY)

t1 = time.time()

print("Start-up time to sample Baddour polar grid:",np.round(t1-t0, decimals=2), "s")

"""
Polar DFT
"""
TrueFunc = gau2(rhomatrix, 1.0)

# Calculate the polar dft
pdft = pfft2_SpaceLimited(fdiscrete, N1, N2, R)

t2 = time.time()

print("Time to calculate the polar transform:", np.round(t2-t1, decimals=2), "s")

"""
Classic DFT
"""
fcart = gau(RR, 1.0)
dft = np.fft.fftshift(np.fft.fft2(fcart))

"""
Interpolate DFT
"""

# Create a meshgrid for interpolation
Rho = R
FYnew, FXnew = np.mgrid[-Rho:Rho:pdft.shape[0]*1j, -Rho:Rho:pdft.shape[1]*1j]

# Interpolate
polar_points = np.vstack([FXX.ravel(), FYY.ravel()]).T
polar_values = pdft.ravel()
pdft_inter = griddata(polar_points, polar_values, (FXnew, FYnew), method='linear')

"""
Take Inverse DFT of DFT to get back original function
"""

ipdft = ipfft2_SpaceLimited(pdft, N1, N2, R)

"""
Plots
"""

# 3D Surface of DFT in Frequency Domain
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("DFT")
surf = ax.plot_surface(FXX, FYY, np.abs(pdft), cmap="gray",
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 1.5)
plt.title("DFT surface in frequency domain")

# Polar DFT
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("Continuous FT")
surf = ax.plot_surface(FXX, FYY, np.abs(TrueFunc), cmap="gray",
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 1.5)
plt.title("Continuous FT surface in frequency domain")

# Plot the DFT matrix
fig = plt.figure("DFT Matrix")
plt.imshow(np.abs(pdft), cmap="gray")
plt.title("DFT Matrix")

# Interpolate
fig = plt.figure("DFT Matrix Interpolated")
plt.imshow(np.abs(pdft_inter), cmap="gray")
plt.title("DFT Matrix Interpolated")

# Plot the Inverse DFT
fig = plt.figure("Inverse DFT")
plt.imshow(np.abs(ipdft), cmap="gray")
plt.title("Inverse DFT")

# 3D Plot of Inverse DFT of DFT
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("Inverse DFT")
surf = ax.plot_surface(Xcart, Ycart, np.abs(ipdft), cmap="gray",
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 1.5)
plt.title("Inverse DFT of DFT")

# Plot the classic DFT
fig = plt.figure("Classic DFT")
plt.imshow(np.abs(dft), cmap="gray")
plt.title("Classic DFT")

plt.show()
