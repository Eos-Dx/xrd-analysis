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
from scipy.signal import wiener

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
Keratin pattern
"""
N1 = 101 # radial sampling count
N2 = 15 # angular sampling count
R = 90

MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
DATA_FILENAME = "CRQF_A00823.txt"

image_path = os.path.join(MODULE_PATH, DATA_DIR, DATA_FILENAME)
image = np.loadtxt(image_path, dtype=np.uint32)
# filtered_img = wiener(image, 15)

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
img_sampled = map_coordinates(image, cart_sampling_indices)

"""
Take polar DFT
"""
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
Classic DFT
"""
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

plt.show()
