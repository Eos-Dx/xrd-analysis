"""
Sample calculation of the 2D Polar Discrete Fourier Transform
"""
import os
import sys
import time

import numpy as np

from scipy.ndimage import map_coordinates
from skimage.transform import warp_polar
from scipy.special import jv

from scipy.signal import wiener

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from eosdxanalysis.models.fourier_analysis import pfft2_SpaceLimited
from eosdxanalysis.models.polar_sampling import sampling_grid

t0 = time.time()

MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
DATA_FILENAME = "CRQF_A00823.txt"


"""
1D DFT
"""

dim = 2**4

# Do 1D case first
L = 2
spacing = L/dim
xspace = np.arange(-L/2 + spacing/2, L/2 + spacing/2, spacing)
func1d = np.cos(2*np.pi*xspace)

func1d_dft = np.fft.fftshift(np.real_if_close(np.fft.fft(func1d), tol=1e18))
power1d = np.square(np.abs(func1d_dft))/dim
freq1d = np.fft.fftshift(np.fft.fftfreq(func1d_dft.shape[0], spacing))

if False:
    fig = plt.figure()
    plt.scatter(xspace, func1d)

    fig = plt.figure()
    plt.scatter(freq1d, power1d)
    plt.show()

"""
2D Sinusoid
"""

if False:
    Lx = 2
    Ly = Lx
    dimx = 2**6
    dimy = dimx
    cstepx = dimx*1j
    cstepy = cstepx

    xspacing = Lx/dimx
    yspacing = xspacing

    YY, XX = np.mgrid[-Lx/2+xspacing/2:Lx/2-xspacing/2:cstepx, -Ly/2+yspacing/2:Ly/2-yspacing/2:cstepy]

    func2d = np.cos(4*2*np.pi*XX)*np.cos(4*2*np.pi*YY)

    # Take 2D DFT

    func2d_dft = np.fft.fftshift(np.real_if_close(np.fft.fft2(func2d), tol=1e18))
    power2d = np.square(np.abs(func2d_dft))/(dimx*dimy)
    freq2dx = np.fft.fftshift(np.fft.fftfreq(func2d_dft.shape[1], xspacing))/dimx
    freq2dy = np.fft.fftshift(np.fft.fftfreq(func2d_dft.shape[0], yspacing))/dimy


    ## Plots

    # Plot original data
    fig = plt.figure()
    plt.imshow(func2d, origin='lower', cmap="gray")
    plt.title("2D Sinusoid")
    plt.xlabel("Horizontal Position [pixels]")
    plt.ylabel("Vertical Position [pixels]")
    plt.colorbar()
    plt.clim(-1,1)

    # Plot 2D DFT magnitude
    left = freq2dx[0]
    right = freq2dx[-1]
    bottom = freq2dy[0]
    top = freq2dy[-1]
    extent = [left, right, bottom, top]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(np.abs(func2d_dft), origin='lower', extent=extent, cmap="gray")
    plt.xlim(freq2dx[1], freq2dx[-1])
    plt.ylim(freq2dy[1], freq2dy[-1])
    plt.title("Fourier Transform of 2D Sinusoid [dB]")
    plt.xlabel("Horizontal Frequency [cycles per pixel]")
    plt.ylabel("Vertical Frequency [cycles per pixel]")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Magnitude [dB]", rotation=270, va="baseline")

    # Show 3D surface maps
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Y, X = np.mgrid[:dimy, :dimx]
    surf = ax.plot_surface(X, Y, func2d, linewidth=0, cmap="gray")
    cbar = fig.colorbar(surf)
    cbar.mappable.set_clim(-1,1)
    ax.view_init(30, +60+180)
    ax.set_title("2D Sinusoid - 3D Surface Plot")
    ax.set_xlabel("Horizontal Position [pixels]")
    ax.set_ylabel("Vertical Position [pixels]")
    ax.set_zlabel("Intensity")
    ax.set_zlim([-1, 1])

    # fig.colorbar(func2d)
    # fig.clim(0,1)


    plt.show()

    exit(0)


"""
Airy function - DFT of unit circle
"""
if False:
    Lx = 2
    Ly = Lx
    dimx = 2**8
    dimy = dimx
    cstepx = dimx*1j
    cstepy = cstepx

    xspacing = Lx/dimx
    yspacing = xspacing

    YY, XX = np.mgrid[-Lx/2+xspacing/2:Lx/2-xspacing/2:cstepx, -Ly/2+yspacing/2:Ly/2-yspacing/2:cstepy]

    RR = np.sqrt(XX**2 + YY**2)

    func2d = np.zeros(RR.shape)
    func2d[RR < 0.1] = 1

    # Take 2D DFT

    func2d_dft = np.fft.fftshift(np.real_if_close(np.fft.fft2(func2d), tol=1e18))
    power2d = np.square(np.abs(func2d_dft))/(dimx*dimy)
    freq2dx = np.fft.fftshift(np.fft.fftfreq(func2d_dft.shape[1], xspacing))/dimx
    freq2dy = np.fft.fftshift(np.fft.fftfreq(func2d_dft.shape[0], yspacing))/dimy

    ## Plots

    # Plot original data
    fig = plt.figure()
    plt.imshow(func2d, origin='lower', cmap="gray")
    plt.title("Disc")
    plt.xlabel("Horizontal Position [pixels]")
    plt.ylabel("Vertical Position [pixels]")
    plt.colorbar()
    plt.clim(0,1)

    # Plot 2D DFT magnitude
    left = freq2dx[0]
    right = freq2dx[-1]
    bottom = freq2dy[0]
    top = freq2dy[-1]
    extent = [left, right, bottom, top]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(np.abs(func2d_dft), origin='lower', extent=extent, cmap="gray")
    plt.xlim(freq2dx[1], freq2dx[-1])
    plt.ylim(freq2dy[1], freq2dy[-1])
    plt.title("Fourier Transform of Disc [dB]")
    plt.xlabel("Horizontal Frequency [cycles per pixel]")
    plt.ylabel("Vertical Frequency [cycles per pixel]")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Magnitude [dB]", rotation=270, va="baseline")

    # Show 3D surface map of original image
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Y, X = np.mgrid[:dimy, :dimx]
    surf = ax.plot_surface(X, Y, func2d, linewidth=0, cmap="gray")
    cbar = fig.colorbar(surf)
    cbar.mappable.set_clim(-1,1)
    ax.view_init(30, +60+180)
    ax.set_title("Disc - 3D Surface Plot")
    ax.set_xlabel("Horizontal Position [pixels]")
    ax.set_ylabel("Vertical Position [pixels]")
    ax.set_zlabel("Intensity")
    ax.set_zlim([0, 1])

    plt.show()

    exit(0)

"""
Do some regular 2D DFT to gain intuition
"""

if False:
    # 2D

    function_description = DATA_FILENAME

    Lx = 2
    Ly = Lx
    dimx = 32
    dimy = dimx
    cstepx = dimx*1j
    cstepy = cstepx

    xspacing = Lx/dimx
    yspacing = xspacing

    YY, XX = np.mgrid[-Lx/2+xspacing/2:Lx/2-xspacing/2:cstepx, -Ly/2+yspacing/2:Ly/2-yspacing/2:cstepy]

    func2d = np.cos(2*np.pi*XX)

    # Take 2D DFT

    func2d_dft = np.fft.fftshift(np.real_if_close(np.fft.fft2(func2d), tol=1e18))
    power2d = np.square(np.abs(func2d_dft))/(dimx*dimy)
    freq2dx = np.fft.fftshift(np.fft.fftfreq(func2d_dft.shape[1], xspacing))/dimx
    freq2dy = np.fft.fftshift(np.fft.fftfreq(func2d_dft.shape[0], yspacing))/dimy


    ## Plots

    # Plot original data
    fig = plt.figure()
    plt.imshow(func2d, origin='lower', cmap="gray")
    plt.title("Horizontal Sinusoid")
    plt.xlabel("Horizontal Position [pixels]")
    plt.ylabel("Vertical Position [pixels]")
    plt.colorbar()
    plt.clim(-1,1)

    # Plot 2D DFT magnitude
    left = freq2dx[0]
    right = freq2dx[-1]
    bottom = freq2dy[0]
    top = freq2dy[-1]
    extent = [left, right, bottom, top]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(np.abs(func2d_dft), origin='lower', extent=extent, cmap="gray")
    plt.xlim(freq2dx[1], freq2dx[-1])
    plt.ylim(freq2dy[1], freq2dy[-1])
    plt.title("Fourier Transform of Horizontal Sinusoid [dB]")
    plt.xlabel("Horizontal Frequency [cycles per pixel]")
    plt.ylabel("Vertical Frequency [cycles per pixel]")
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("Magnitude [dB]", rotation=270, va="baseline")

    # Show 3D surface maps
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    Y, X = np.mgrid[:dimy, :dimx]
    surf = ax.plot_surface(X, Y, func2d, linewidth=0, cmap="gray")
    cbar = fig.colorbar(surf)
    cbar.mappable.set_clim(-1,1)
    ax.view_init(30, +60+180)
    ax.set_title("Horizontal Sinusoid - 3D Surface Plot")
    ax.set_xlabel("Horizontal Position [pixels]")
    ax.set_ylabel("Vertical Position [pixels]")
    ax.set_zlabel("Intensity")
    ax.set_zlim([-1, 1])

    # fig.colorbar(func2d)
    # fig.clim(0,1)


    plt.show()

    exit(0)

"""
Keratin 2D Cartesian FFT with and without filtering
"""
function_description = DATA_FILENAME

# Load keratin xrd image
image_path = os.path.join(MODULE_PATH, DATA_DIR, DATA_FILENAME)
image = np.loadtxt(image_path, dtype=np.uint32)

# Original image
fig = plt.figure()
plt.imshow(image)
plt.title("Original Cartesian sampling of {}".format(function_description))

# Wiener filtered image
filtered_img = wiener(image, 5)
fig = plt.figure()
plt.imshow(filtered_img)
plt.title("Wiener Filtered of Original Cartesian sampling of {}".format(function_description))

# FFT of original image
image_fft = np.fft.fftshift(np.real_if_close(np.fft.fft2(image), tol=1e18))
dimx, dimy = image.shape
xspacing, yspacing = (1, 1)
image_fft_power2d = np.square(np.abs(image_fft))/(dimx*dimy)
image_freq2dx = np.fft.fftshift(np.fft.fftfreq(image_fft.shape[1], xspacing))
image_freq2dy = np.fft.fftshift(np.fft.fftfreq(image_fft.shape[0], yspacing))

# Plot 2D FFT magnitude of original image
left = image_freq2dx[0]
right = image_freq2dx[-1]
bottom = image_freq2dy[0]
top = image_freq2dy[-1]
extent = [left, right, bottom, top]

fig = plt.figure()
plt.imshow(20*np.log10(np.abs(image_fft)), origin='lower', extent=extent)
plt.title("FFT of Original Cartesian sampling of {} [dB]".format(function_description))
plt.xlim(image_freq2dx[1], image_freq2dx[-1])
plt.ylim(image_freq2dy[1], image_freq2dy[-1])


# 2D FFT of filtered image
filtered_img_fft = np.fft.fftshift(np.real_if_close(np.fft.fft2(filtered_img), tol=1e18))
dimx, dimy = filtered_img.shape
xspacing, yspacing = (1, 1)
filtered_img_fft_power2d = np.square(np.abs(filtered_img_fft))/(dimx*dimy)
filtered_img_freq2dx = np.fft.fftshift(np.fft.fftfreq(filtered_img_fft.shape[1], xspacing))
filtered_img_freq2dy = np.fft.fftshift(np.fft.fftfreq(filtered_img_fft.shape[0], yspacing))

# Plot 2D FFT magnitude of filtered image
left = filtered_img_freq2dx[0]
right = filtered_img_freq2dx[-1]
bottom = filtered_img_freq2dy[0]
top = filtered_img_freq2dy[-1]
extent = [left, right, bottom, top]

fig = plt.figure()
plt.imshow(20*np.log10(np.abs(filtered_img_fft)), origin='lower', extent=extent)
plt.title("FFT of Original Cartesian sampling of {} [dB]".format(function_description))
plt.xlim(filtered_img_freq2dx[1], filtered_img_freq2dx[-1])
plt.ylim(filtered_img_freq2dy[1], filtered_img_freq2dy[-1])

plt.show()

"""
2D Polar DFT
"""

N1 = 100
N2 = 101
R = 90

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
