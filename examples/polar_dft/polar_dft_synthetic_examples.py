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

"""
Take Inverse DFT of DFT to get back original function
"""

N1 = 101 # radial sampling count
N2 = 51 # angular sampling count
R = 5

fsignal = np.zeros((N2, N1-1), dtype=np.complex128)
M = (N1-1)/2
fsignal[0, 0] = 5
# fsignal[int(N2/4+0.5), 0] = 5

ipdft = ipfft2_SpaceLimited(fsignal, N1, N2, R)
# Extend for 3D plotting shading purposes
ipdft_ext = np.vstack([ipdft, ipdft[0,:]])

thetamatrix, rmatrix = sampling_grid(N1, N2, R)

Xcart, Ycart = pol2cart(thetamatrix, rmatrix)
# Extend for 3D plotting shading purposes
Xcart_ext = np.vstack([Xcart, Xcart[0,:]])
Ycart_ext = np.vstack([Ycart, Ycart[0,:]])

# Take DFT of frequency signal
pdft = pfft2_SpaceLimited(ipdft, N1, N2, R)
psimatrix, rhomatrix = freq_sampling_grid(N1, N2, R)

FX, FY = pol2cart(psimatrix, rhomatrix)
FX_ext = np.vstack([FX, FX[0,:]])
FY_ext = np.vstack([FY, FY[0,:]])

"""
Synthetic patterns similar to our data
"""

gau = lambda x, a: np.exp(-(x/a)**2)
gau_2d = lambda x, y, a: np.exp(-((x/a)**2 + (y/a)**2))

a = 1.0

# The space-domain signal is sampled on the Baddour polar sampling grid
gau_signal = gau_2d(Xcart - R/2, Ycart, a) + gau_2d(Xcart + R/2, Ycart, a)

# For reference, we plot the same space-domain signal in the Cartesian grid
# Create a meshgrid
YY, XX = np.mgrid[-R:R:50j, -R:R:50j]
gau_cart = gau_2d(XX - R/2, YY, a) + gau_2d(XX + R/2, YY, a)

gau_pdft = pfft2_SpaceLimited(gau_signal, N1, N2, R)
gau_pdft_ext = np.vstack([gau_pdft, gau_pdft[0,:]])

# Now take inverse DFT of DFT

gau_ipdft = ipfft2_SpaceLimited(gau_pdft, N1, N2, R)
# Extend for 3D plotting shading purposes
gau_ipdft_ext = np.vstack([gau_ipdft, gau_ipdft[0,:]])

# Compare to input
assert(np.isclose(gau_signal, np.real_if_close(gau_ipdft, tol=1e18)).all())

# Classic DFT

gau_dft = np.fft.fft2(gau_cart)

"""
Plots
"""

# 3D Plot of Inverse DFT of DFT
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("Inverse DFT")
surf = ax.plot_surface(Xcart_ext, Ycart_ext, np.abs(ipdft_ext), cmap="gray",
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 1.5)
plt.title("Inverse DFT of frequency signal (Space Domain, Cartesian Grid)")

# 2D Plot
#fig = plt.figure("2D Plot Example")
#plt.scatter(Xcart, Ycart, c=np.abs(ipdft), s=0.1)
#plt.title("Inverse DFT of frequency signal")

# 2D Plot
fig = plt.figure("2D Plot Image")
plt.imshow(np.abs(ipdft), cmap="gray")
plt.ylabel("Angular Sampling")
plt.xlabel("Radial Sampling")
plt.title("Inverse DFT Image (Space Domain, Polar Grid)")

# DFT
fig = plt.figure("DFT Polar Grid")
plt.imshow(np.abs(pdft), cmap="gray")
plt.ylabel("Angular Sampling")
plt.xlabel("Radial Sampling")
plt.title("(Frequency Domain, Polar Grid)")

"""
Double Gaussian plots
"""

# Double Gaussian example Cartesian space domain
fig = plt.figure("Double Gaussian")
plt.imshow(gau_cart, cmap="gray")
plt.title("Double Gaussian Cartesian Sampling (Space Domain)")

# 3D Double Gaussian Space Domain, Cartesian
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("3D: Double Gaussian")
surf = ax.plot_surface(XX, YY, np.abs(gau_cart), cmap="gray",
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 1.5)
plt.title("Double Gaussian (Space Domain, Cartesian Grid)")

# Double Gaussian example PDFT
fig = plt.figure("DFT Double Gaussian")
plt.imshow(np.abs(gau_pdft), cmap="gray")
plt.title("Double Gaussian DFT (Frequency Domain)")

# Double Guassian example DFT dB
fig = plt.figure("DFT Double Gaussian")
plt.imshow(20*np.log10(np.abs(gau_pdft)), cmap="gray")
plt.title("Double Gaussian DFT (Frequency Domain) [dB]")

# 3D Plot of Double Gaussian DFT
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("3D: DFT of Double Gaussian")
surf = ax.plot_surface(FX_ext, FY_ext, np.abs(gau_pdft_ext), cmap="gray",
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 15)
plt.title("DFT of Double Gaussian (Frequency Domain)")

# 3D Plot of Inverse DFT of DFT
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("Inverse DFT Double Gaussian")
surf = ax.plot_surface(Xcart_ext, Ycart_ext, np.abs(gau_ipdft_ext), cmap="gray",
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 1.5)
plt.title("Inverse DFT of Double Gaussian DFT (Space Domain)")

# 2D Plot of original and IDFT of DFT
fig, axs = plt.subplots(1,2)
fig.canvas.manager.set_window_title("Double Gaussian Plot After DFT and IDFT (Frequency Domain, Polar Grid)")
fig.suptitle("Double Gaussian Before and After DFT + IDFT (Polar Grid)")
axs[0].imshow(np.abs(gau_ipdft), cmap="gray")
axs[0].set_title("IDFT of DFT of Double Gaussian")
axs[1].imshow(gau_signal, cmap="gray")
axs[1].set_title("Original Double Gaussian")

# Plot the classic 2D DFT
fig = plt.figure("Classic DFT of Double Gaussian")
plt.imshow(np.fft.fftshift(np.abs(gau_dft)), cmap="gray")
plt.title("Classic DFT of Double Gaussian")
plt.xlabel("X Frequency")
plt.ylabel("Y Frequency")

plt.show()
