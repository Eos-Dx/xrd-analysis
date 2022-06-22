"""
Helper functions for models
"""
import os

import numpy as np
from scipy.special import jn_zeros
from scipy.io import savemat


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
