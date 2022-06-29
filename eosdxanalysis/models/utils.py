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

from eosdxanalysis.preprocessing.preprocess import PreprocessDataArray
from eosdxanalysis.preprocessing.image_processing import crop_image
from eosdxanalysis.preprocessing.image_processing import centerize
from eosdxanalysis.preprocessing.image_processing import pad_image
from eosdxanalysis.preprocessing.utils import create_circular_mask

def gen_zeromatrix(shape, save_mat=False, save_numpy=False, outdir=""):
    """
    Pre-calculate Bessel zeros
    Can save matlab and/or numpy format
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

def l1_metric_optimized(image1, image2, params, plan=None):
    """
    Function which computes the L1 distance between two images
    that may include a sample-to-detector distance shift.
    The algorithm performs a binary search to minimize the distance
    between one image and resized version of the other image.
    """
    # Set the tolerance for convergence criterion
    TOL=1e-6

    if plan is None:
        plan = [
                "local_thresh_quad_fold",
                ]
        output_style = [
                "local_thresh_quad_folded",
                ]

    params1 = params.copy()
    params2 = params.copy()
    del params2["crop_style"]

    # Preprocess both images according to parameters, not cropping image2
    image1_preprocessor = PreprocessDataArray(image1, params=params1)
    image2_preprocessor = PreprocessDataArray(image2, params=params2)

    # Preprocess images
    image1_preprocessor.preprocess(plan, mask_style="both")
    image2_preprocessor.preprocess(plan, mask_style=None)
    image1_post = image1_preprocessor.cache.get(output_style[0])
    image2_post_unmasked = image2_preprocessor.cache.get(output_style[0])

    # Crop image2
    image2_preprocessor.preprocess(plan, mask_style="both")
    image2_post = image2_preprocessor.cache.get(output_style[0])

    # Create mask
    h, w = image1.shape
    rmin = params.get("rmin")
    rmax = params.get("rmax")
    mask = create_circular_mask(h, w, rmin=rmin, rmax=rmax)

    distance = calculate_min_distance(
                    image1_post, image2_post_unmasked, mask, tol=TOL)

    return distance
