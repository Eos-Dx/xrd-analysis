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

N1 = 51
N2 = 51
R = 5

fsignal = np.zeros((N2, N1-1))
M = (N1-1)/2
fsignal[0, 0] = 10

ipdft = ipfft2_SpaceLimited(fsignal, N1, N2, R)

thetamatrix, rmatrix = sampling_grid(N1, N2, R)

Xcart, Ycart = pol2cart(thetamatrix, rmatrix)

# Take DFT of frequency signal
pdft = pfft2_SpaceLimited(ipdft, N1, N2, R)
psimatrix, rhomatrix = freq_sampling_grid(N1, N2, R)

FX, FY = pol2cart(psimatrix, rhomatrix)

"""
Plots
"""

# 3D Plot of Inverse DFT of DFT
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig.canvas.manager.set_window_title("Inverse DFT")
surf = ax.plot_surface(Xcart, Ycart, np.abs(ipdft), cmap="gray",
                               linewidth=0, antialiased=False)
clb = fig.colorbar(surf)
ax.set_zlim(0, 1.5)
plt.title("Inverse DFT of frequency signal")

# 2D Plot
fig = plt.figure("2D Plot")
plt.scatter(Xcart, Ycart, c=ipdft, s=0.1)
plt.title("Inverse DFT of frequency signal")

# DFT
fig = plt.figure("DFT")
plt.imshow(np.abs(pdft), cmap="gray")

plt.show()
