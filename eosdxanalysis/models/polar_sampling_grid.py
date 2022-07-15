"""
Generate a polar sampling grid according to the
2D Polar Discrete Fourier Transform described in
https://peerj.com/articles/cs-257/#fig-1

N1 is the radial sampling count
N2 is the angular sampling count

The grid size is (N2, N1-1)
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
    thetamatrix shape is (N2, N1-1)

    theta_p = p*2*pi/N2
    where -M <= p <= M
    """
    # Set up a symmetric theta range based on prescription from paper
    # (symmetric about theta = 0)
    # Set up theta start, stop and step values
    theta_step = 2*np.pi/N2
    theta_min = -np.pi + theta_step/2
    theta_max = np.pi + theta_step/2
    # Set up the theta range
    theta_range = np.arange(theta_min, theta_max, theta_step).reshape(-1,1)
    # Create a meshgrid
    thetamatrix = theta_range * np.ones((1,N1-1))
    return thetamatrix

def rmatrix_SpaceLimited(N2, N1, R, jn_zerosmatrix=None):
    """
    Generates the radial meshgrid
    rmatrix shape is (N2, N1-1)

    rpk = jpk/jpN1 * R
    where -M <= p <=M
    and 1 <= k <= N1-1

    - R is the space limit, size of domain
    - rpk is the kth zero of the pth Bessel function of the first kind
    """
    # If the jn_zerosmatrix is not given, read from file
    if jn_zerosmatrix is None:
        jn_zerosmatrix_path = os.path.join(
                MODULE_PATH,
                DATA_DIR,
                JN_ZEROSMATRIX_FILENAME)
        jn_zerosmatrix = np.load(jn_zerosmatrix_path)

    M = (N2-1)//2;

    rmatrix = np.zeros((N2,N1-1))

    for pprime in range(N2):
        p = pprime - M
        zero2 = jn_zerosmatrix[abs(p), :N1]
        jpN1 = zero2[N1-1]
        jpk = zero2[:N1-1]
        rmatrix[pprime, :N1-1] = (jpk[:N1-1]/jpN1)*R

    return rmatrix

def sampling_grid(R, N1, N2):
    """
    Returns polar sampling grid according to
    2D Polar Discrete Fourier Transform
    """
    pass
