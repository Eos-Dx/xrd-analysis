"""
Gaussian fit to keratin diffraction patterns

Gaussians:
    1. Background noise
    2. Circular aperture diffraction pattern
    3. Amorphous scattering
    4. 9 A: Looks like a D-shape, how to characterize? Convolution with an arc?
    5. 5 A: 2D Gaussian convolution with an arc
    6. 5-4 A: 2D Gaussian convolution with a circle. Or look into 1D Gaussian rotated
"""

import os
import glob
from collections import OrderedDict
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.optimize import curve_fit

from eosdxanalysis.models.utils import cart2pol
from eosdxanalysis.models.feature_engineering import EngineeredFeatures
from eosdxanalysis.models.curve_fitting import GaussianDecomposition
from eosdxanalysis.preprocessing.utils import create_circular_mask
from eosdxanalysis.simulations.utils import feature_pixel_location

t0 = time.time()

cmap="hot"


"""
Import keratin diffraction image
"""
# Import sample image
DATA_DIR = os.path.join("")
OUT_DIR = os.path.join("")

filename_path_list = glob.glob(os.path.join(DATA_DIR,"*.txt"))

size = 256
# Calculate center coordinates of image in array index notation
center = (size/2-0.5, size/2-0.5)

print("Running Gaussian fit algorithm on data set...")
print("Filename | Fit Error | Fit Error (%) | R-like factor (%) | Time (s)")

for filename_path in filename_path_list:
    filename = os.path.basename(filename_path)
    print(filename, end=" ", flush=True)
    image = np.loadtxt(filename_path, dtype=np.uint32)
    filtered_img = gaussian_filter(image, 2)

    """
    Gaussian fit pipeline
    1. Filter
    2. Centerize, rotate
    3. Calculate engineered features
      - Peak location
      - Peak amplitude
      - Standard deviation
        - Full-width at half maximum (quick estimate)
      - Angular spread
    4. Curve fitting
    """

    # preprocessor = PreprocessData()

    # Calculate 9A peak location and maximum
    feature_class = EngineeredFeatures(filtered_img, params=None)
    peak_col_9A, peak_9A_max, roi_9A, roi_center_9A, anchor_9A = feature_class.feature_9a_peak_location()

    peak_radius_9A = peak_col_9A - center[1]


    # Calculate full-width half maximum for 9A peak
    image_slice_9A = filtered_img[int(center[0]), int(peak_col_9A):int(peak_col_9A)+20]
    full_width_half_max, max_val, max_val_loc, half_max, half_max_loc = EngineeredFeatures.fwhm(image_slice_9A)

    sigma_9A = full_width_half_max/(2*np.sqrt(2*np.log(2)))

    """
    Get best-fit parameters
    """

    # Get initial guess and bounds
    gauss_class = GaussianDecomposition()
    p0_dict = gauss_class.p0_dict
    p0 = np.fromiter(p0_dict.values(), dtype=np.float64)
    p_lower_bounds = np.fromiter(gauss_class.p_lower_bounds_dict.values(), dtype=np.float64)
    p_upper_bounds = np.fromiter(gauss_class.p_upper_bounds_dict.values(), dtype=np.float64)

    # Modify initial guess and bounds
    p0_dict["peak_radius_9A"] = peak_radius_9A
    p0_dict["amplitude_9A"] = peak_9A_max
    p0_dict["width_9A"] = sigma_9A

    # Perform iterative curve_fit
    popt, pcov, RR, TT = gauss_class.best_fit(image)
    decomp_image  = gauss_class.keratin_function((RR, TT), *popt).reshape(image.shape)

    # Calculate fit error
    error = gauss_class.fit_error(image, decomp_image)
    error_ratio = error/np.sum(np.square(image))
    print(np.round(error), end=" ")
    print(np.round(error_ratio*100, decimals=2), end=" ")

    # Use R-factor style error 
    r_style_error = np.sum(np.abs(image - decomp_image)) / np.sum(image)
    print(np.round(r_style_error*100, decimals=2), end=" ", flush=True)

    """
    Plot
    """
    # Set up meshgrid for 3D plotting
    shape = image.shape
    x_end = shape[1]/2 - 0.5
    x_start = -x_end
    y_end = x_end
    y_start = x_start
    YY, XX = np.mgrid[y_start:y_end:shape[0]*1j, x_start:x_end:shape[1]*1j]

    # Set up feature pixel locations
    d9_inv_pixels = feature_pixel_location(9e-10)
    d5_inv_pixels = feature_pixel_location(5e-10)
    d4_inv_pixels = feature_pixel_location(4e-10)
    d5_4_inv_pixels = feature_pixel_location(4.5e-10)


    # 3D Plot of filtered image
    plot_title = "3D Filtered Image"
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.canvas.manager.set_window_title(plot_title)
    surf = ax.plot_surface(XX, YY, filtered_img, cmap=cmap,
                                   linewidth=0, antialiased=False)
    clb = fig.colorbar(surf)
    ax.set_zlim(0, 2e3)
    plt.title(plot_title)
    plt.close('all')

    # Original image
    plot_title = filename
    fig = plt.figure(plot_title)
    plt.imshow(image, cmap=cmap)
    plt.scatter(center[1] - d9_inv_pixels, center[0], c="green", label="9 A")
    plt.scatter(center[1], center[0] - d5_inv_pixels, c="blue", label="5 A")
    plt.plot([center[1] + d5_inv_pixels, center[1] + d4_inv_pixels],
            [center[0], center[0]], c="white", label="5-4 A")
    plt.scatter(center[1] + d5_4_inv_pixels, center[0], c="black", label="4.5 A")
    plt.legend()
    plt.title(plot_title)

    plt.savefig(os.path.join(OUT_DIR, filename + "_features.png"))
    plt.close('all')

    # Filtered image
    plot_title = "Filtered " + filename
    fig = plt.figure(plot_title)
    plt.imshow(filtered_img, cmap=cmap)
    plt.scatter(center[1] - d9_inv_pixels, center[0], c="green", label="9 A")
    plt.scatter(center[1], center[0] - d5_inv_pixels, c="blue", label="5 A")
    plt.plot([center[1] + d5_inv_pixels, center[1] + d4_inv_pixels],
            [center[0], center[0]], c="white", label="5-4 A")
    plt.scatter(center[1] + d5_4_inv_pixels, center[0], c="black", label="4.5 A")
    plt.legend()
    plt.title(plot_title)
    plt.close('all')

    # plt.savefig(os.path.join(OUT_DIR, "filtered_" + filename + "_features.png"))

    # Optimum Gaussian approximation
    plot_title = "Optimum fit " + filename
    fig = plt.figure(plot_title)
    plt.imshow(decomp_image, cmap=cmap)
    plt.scatter(center[1] - d9_inv_pixels, center[0], c="green", label="9 A")
    plt.scatter(center[1], center[0] - d5_inv_pixels, c="blue", label="5 A")
    plt.plot([center[1] + d5_inv_pixels, center[1] + d4_inv_pixels],
            [center[0], center[0]], c="white", label="5-4 A")
    plt.scatter(center[1] + d5_4_inv_pixels, center[0], c="black", label="4.5 A", zorder=2.0)
    plt.legend()
    plt.title(plot_title)

    plt.savefig(os.path.join(OUT_DIR, "optimum_fit_" + filename + "_features.png"))
    plt.close('all')

    # 3D Plot of optimal image
    plot_title = "3D Optimal Image"
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.canvas.manager.set_window_title(plot_title)
    surf = ax.plot_surface(XX, YY, decomp_image, cmap=cmap,
                                   linewidth=0, antialiased=False)
    clb = fig.colorbar(surf)
    ax.set_zlim(0, 2e3)
    plt.title(plot_title)

    plt.savefig(os.path.join(OUT_DIR, "optimum_fit_3d_" + filename + "_features.png"))
    # plt.show()
    plt.close('all')

    t1 = time.time()
    print(np.round(t1 - t0), flush=True)
    t0 = t1
