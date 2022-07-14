"""
Generate a polar sampling grid according to the
2D Polar Discrete Fourier Transform described in
https://peerj.com/articles/cs-257/#fig-1
"""
import os
import numpy as np

MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
JN_ZEROSMATRIX_FILENAME = "jn_zerosmatrix.npy"

# R = 1, N1 = 16 and N2 = 15

def thetamatrix_SpaceLimited(N2, N1):
    """
    Generates the angular meshgrid
    """
    b2 = 2*np.pi/N2;
    progr2 = np.linspace(-np.pi + b2/2, np.pi - b2/2, N2)
    theta = progr2*np.ones((1,N1-1))
    return theta

def rmatrix_SpaceLimited(N2, N1, R, jn_zerosmatrix=None):
    """
    Generates the radial meshgrid
    """
    # If the jn_zerosmatrix is not given, read from file
    if jn_zerosmatrix is None:
        jn_zerosmatrix_path = os.path.join(
                MODULE_PATH,
                DATA_DIR,
                JN_ZEROSMATRIX_FILENAME)
        jn_zerosmatrix = np.load(jn_zerosmatrix_path)

    M=(N2-1)//2;

    rmatrix = np.zeros((N2,N1-1))

    for pprime in range(N2):
        p = pprime - M
        for k in range(N1-1):
            zero2 = jn_zerosmatrix[abs(p),:N1]
            jpk = zero2[k]
            jpN1 = zero2[N1-1]
            rmatrix[pprime,k] = (jpk/jpN1)*R
    return rmatrix

def sampling_grid(R, N1, N2):
    """
    Returns polar sampling grid according to
    2D Polar Discrete Fourier Transform
    """
    pass
