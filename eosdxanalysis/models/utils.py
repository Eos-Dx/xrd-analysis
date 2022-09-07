"""
Helper functions for models
"""
import os

import numpy as np
import pandas as pd
from scipy.special import jn_zeros
from scipy.io import savemat
import scipy
import scipy.cluster.hierarchy as sch

from skimage.transform import rescale
from skimage.transform import warp_polar

from eosdxanalysis.preprocessing.image_processing import crop_image
from eosdxanalysis.preprocessing.image_processing import pad_image
from eosdxanalysis.preprocessing.utils import create_circular_mask

def gen_jn_zerosmatrix(shape, save_mat=False, save_numpy=False, outdir=""):
    """
    Pre-calculate Bessel zeros
    Can save matlab and/or numpy format
    Used for 2D Polar Discrete Fourier Transform
    """
    nthorder, kzeros = shape
    zeromatrix = np.zeros((nthorder,kzeros))

    for idx in range(nthorder):
        zeromatrix[idx,:] = jn_zeros(idx,kzeros)

    if save_mat:
        # Save to matlab file
        mdic = {"zeromatrix": zeromatrix}
        filename = "zeromatrix.mat"
        full_savepath = os.path.join(outdir, filename)
        savemat(full_savepath, mdic)

    if save_numpy:
        filename = "zeromatrix.npy"
        full_savepath = os.path.join(outdir, filename)
        # Save to numpy file
        np.save(full_savepath, zeromatrix)

    return zeromatrix

def pol2cart(theta, rho):
    """
    Function to convert polar to cartesian coordinates
    Similar to Matlab pol2cart
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def cart2pol(x, y):
    """
    Function to convert cartesian to polar coordinates
    Similar to Matlab cart2pol
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return theta, r

def radial_intensity_1d(image, width=4):
    """
    Returns the 1D radial intensity of positive horizontal strip (averaged).
    For any other strip or quadrant, transpose and/or reverse rows/columns order.
       __
     /    \
    |   ===|
     \ __ /

    """
    # Calculate 1D radial intensity in positive horizontal direction
    center = image.shape[0]/2-0.5, image.shape[1]/2-0.5
    row_start, row_end = int(np.ceil(center[0] - width/2)), int(np.ceil(center[0] + width/2))
    col_start, col_end = int(np.ceil(center[1])), image.shape[1]
    intensity_strip = image[row_start:row_end, col_start:col_end]
    intensity_1d = np.mean(intensity_strip, axis=0) # Average across rows

    return intensity_1d

def angular_intensity_1d(image, radius=None, width=4):
    # If radius is not specified, do half of smallest image shape
    if radius is None:
        smallest_shape = np.min(image.shape)
        radius = smallest_shape/2
    # Calculate image center
    center = image.shape[0]/2-0.5, image.shape[1]/2-0.5

    # Use warp_polar
    polar_image = warp_polar(image, radius=radius+int(width/2),
            output_shape=(360, image.shape[1]))
    col_start, col_end = int(np.floor(radius - width/2)), int(np.ceil(radius + width/2))
    # Average across the columns corresponding to the annulus
    angular_profile_1d = np.mean(polar_image[:, col_start:col_end], axis=1)
    return angular_profile_1d

def dirac_arc(radius, start_angle, angle_spread, output_shape):
    """
    Creates a 2D direc delta function in the shape of an arc

    :param radius: Radius of the arc
    :type radius: float

    :param start_angle: The starting angle of the arc in radians.
        The angle in the positive x direction is `0` radians,
        positive angle is counter-clockwise.
    :type start_angle: float

    :param angle_spread: The angular spread of the arc in radians.
        This is a positive number from `0` to `2pi`.
    :type angle_spread: float

    :param output_shape: Shape of the output
    :type output_shape: ndarray

    :returns arc: The dirac arc
    :rtype: ndarray

    """
    return

def draw_antialiased_circle(outer_radius):
    """
    Adapted via: https://stackoverflow.com/a/37714284

    This will always output a shape 2*(outer_radius+1)

    :param outer_radius: radius of circle
    :type outer_radius: int

    :returns circle: Numpy array of antialiased circle of shape ``2*(outer_radius+1)``
    :rtype: ndarray

    """
    point_array = np.zeros((outer_radius+1, outer_radius+1))

    i = 0
    j = outer_radius
    last_fade_amount = 0
    fade_amount = 0

    MAX_OPAQUE = 1.0

    while i < j:
        height = np.sqrt(np.max(outer_radius * outer_radius - i * i, 0))
        fade_amount = MAX_OPAQUE * (np.ceil(height) - height)

        if fade_amount < last_fade_amount:
            # Opaqueness reset so drop down a row.
            j -= 1
        last_fade_amount = fade_amount

        # We're fading out the current j row, and fading in the next one down.
        point_array[i,j] = MAX_OPAQUE - fade_amount
        point_array[i,j-1] = fade_amount

        i += 1

    # Fully construct the lower-right quadrant by adding the transpose
    quad_lower_right = point_array + point_array.T

    # Flip the lower-right quadrant vertically to get the upper-right quadrant
    quad_upper_right = quad_lower_right[::-1, :]
    # Stack the upper-right and lower-right quadrants to get the right half
    circle_right = np.vstack((quad_upper_right, quad_lower_right))
    # Flip the right half horizontally (row-wise, i.e. reverse order of columns)
    # to get the left half of the circle
    circle_left = circle_right[:,::-1]
    # Finally, stack the left and right halves to form the full circle
    circle = np.hstack((circle_left, circle_right))

    return circle

def l1_metric(A, B):
    """
    Calculates the L1 metric (distance) of two matrices
    based on the L1 norm as defined below.
    """
    if A.size != B.size:
        raise ValueError("Matrix sizes must agree!")
    return 1/A.size*np.sum(abs(A.ravel()-B.ravel()))

def l1_metric_normalized(A, B):
    """
    Calculates the L1 metric (distance) of two matrices
    based on the L1 norm as defined below, with normalization.
    """
    if A.size != B.size:
        raise ValueError("Matrix sizes must agree! Shapes are {}, {}.".format(A.shape, B.shape))
    A_vec = A.ravel()/np.sum(A)
    B_vec = B.ravel()/np.sum(B)
    return np.sum(abs(A_vec-B_vec))

def cluster_corr(corr_array, inplace=False):
    # via: https://wil.yegelwel.com/cluster-correlation-matrix/
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

def calculate_min_distance(image1_post, image2_post_unmasked, mask, scale=1.0,
                        tol=1e-2, max_iters=1e1, iterations=0):
    iterations = iterations+1

    h, w = image1_post.shape
    distance = l1_metric_normalized(image1_post, image2_post_unmasked)

    if iterations > max_iters:
        return distance

    # These images may be have odd shape
    image2_smallscale_unmasked = rescale(image2_post_unmasked, scale*0.9, anti_aliasing=False)
    image2_largescale_unmasked = rescale(image2_post_unmasked, scale*1.1, anti_aliasing=False)

    # First centerize the smaller image so that it is even
    image2_smallscale_center = (image2_smallscale_unmasked.shape[0]/2,
                            image2_smallscale_unmasked.shape[1]/2,)
    image2_smallscale_centerized, _ = centerize(image2_smallscale_unmasked,
                            image2_smallscale_center)
    # Crop centerized image to 256x256
    image2_smallscale_unmasked = crop_image(image2_smallscale_centerized, h, w)
    # Crop the larger image to 256x256
    image2_largescale_unmasked = crop_image(image2_largescale_unmasked, h, w)

    # Mask images
    image2_smallscale = image2_smallscale_unmasked.copy()
    image2_largescale = image2_largescale_unmasked.copy()
    image2_smallscale[~mask] = 0
    image2_largescale[~mask] = 0

    distance_smallscale = l1_metric_normalized(image1_post, image2_smallscale)
    distance_largescale = l1_metric_normalized(image1_post, image2_largescale)

    if np.abs(distance_smallscale - distance) < tol:
        return distance_smallscale
    elif np.abs(distance_largescale - distance) < tol:
        return distance_largescale
    elif distance_smallscale < distance:
        # Free up memory
        del image2_smallscale_centerized
        del image2_largscale_unmasked
        del image2_smallscale
        del image2_largescale
        # Set new scale
        new_scale = scale*0.9
        return calculate_min_distance(
                image1_post, image2_smallscale_unmasked, mask,
                scale=new_scale, tol=tol, iterations=iterations)
    else:
        # Free up memory
        del image2_smallscale_centerized
        del image2_smallscale_unmasked
        del image2_smallscale
        del image2_largescale
        # Set new scale
        new_scale = scale*1.1
        return calculate_min_distance(
                image1_post, image2_largescale_unmasked, mask,
                scale=new_scale, tol=tol, iterations=iterations)

#def l1_metric_optimized(image1, image2, params, plan=None):
#    """
#    Function which computes the L1 distance between two images
#    that may include a sample-to-detector distance shift.
#    The algorithm performs a binary search to minimize the distance
#    between one image and resized version of the other image.
#    """
#    # Set the tolerance for convergence criterion
#    TOL=1e-6
#
#    if plan is None:
#        plan = [
#                "local_thresh_quad_fold",
#                ]
#        output_style = [
#                "local_thresh_quad_folded",
#                ]
#
#    params1 = params.copy()
#    params2 = params.copy()
#    del params2["crop_style"]
#
#    # Preprocess both images according to parameters, not cropping image2
#    image1_preprocessor = PreprocessDataArray(image1, params=params1)
#    image2_preprocessor = PreprocessDataArray(image2, params=params2)
#
#    # Preprocess images
#    image1_preprocessor.preprocess(plan, mask_style="both")
#    image2_preprocessor.preprocess(plan, mask_style=None)
#    image1_post = image1_preprocessor.cache.get(output_style[0])
#    image2_post_unmasked = image2_preprocessor.cache.get(output_style[0])
#
#    # Crop image2
#    image2_preprocessor.preprocess(plan, mask_style="both")
#    image2_post = image2_preprocessor.cache.get(output_style[0])
#
#    # Create mask
#    h, w = image1.shape
#    rmin = params.get("rmin")
#    rmax = params.get("rmax")
#    mask = create_circular_mask(h, w, rmin=rmin, rmax=rmax)
#
#    distance = calculate_min_distance(
#                    image1_post, image2_post_unmasked, mask, tol=TOL)
#
#    return distance
