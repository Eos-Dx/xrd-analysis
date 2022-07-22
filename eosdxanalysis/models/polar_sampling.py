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
    Note that N2 = 2*M+1

    - R is the space limit, size of domain
    - jpk is the kth zero of the pth Bessel function of the first kind
    """
    # If the jn_zerosmatrix is not given, read from file
    if jn_zerosmatrix is None:
        jn_zerosmatrix_path = os.path.join(
                MODULE_PATH,
                DATA_DIR,
                JN_ZEROSMATRIX_FILENAME)
        jn_zerosmatrix = np.load(jn_zerosmatrix_path)

    M = (N2-1)//2;

    # Get jpk and jpN1 from jn_zerosmatrix
    p_range = abs(np.arange(N2)-M)
    # Get all Jn zeros we interested in
    # Note that k goes from 1 to N1-1
    # So we are getting 1 to N1 and then splitting
    # the results into jpk and jpN1
    jpk_jpN1_combined = jn_zerosmatrix[p_range, :N1]
    # Get jpk by slicing
    jpk = jpk_jpN1_combined[:,:N1-1]
    # Get jpN1 by slicing (last column)
    jpN1 = jpk_jpN1_combined[:,-1].reshape(-1,1)
    # Now get rmatrix. Size of rmatrix = size of jpk.
    rmatrix = jpk/jpN1*R

    return rmatrix

def rhomatrix_SpaceLimited(N2, N1, R, jn_zerosmatrix=None):
    """
    Generates the radial meshgrid
    rmatrix shape is (N2, N1-1)

    rhoqm = jqm/R
    where -M <= q <=M
    and 1 <= m <= N1-1
    Note that N2 = 2*M+1

    - R is the space limit, size of domain
    - jqm is the kth zero of the pth Bessel function of the first kind
    """
    # If the jn_zerosmatrix is not given, read from file
    if jn_zerosmatrix is None:
        jn_zerosmatrix_path = os.path.join(
                MODULE_PATH,
                DATA_DIR,
                JN_ZEROSMATRIX_FILENAME)
        jn_zerosmatrix = np.load(jn_zerosmatrix_path)

    M = (N2-1)//2;

    # Get jpk and jpN1 from jn_zerosmatrix
    p_range = abs(np.arange(N2)-M)
    # Get all Jn zeros we interested in
    # Note that k goes from 1 to N1-1
    # So we are getting 1 to N1 and then splitting
    # the results into jpk and jpN1
    jqm = jn_zerosmatrix[p_range, :N1-1]
    # Now get rmatrix. Size of rmatrix = size of jpk.
    rhomatrix = jqm/R

    return rhomatrix

def psimatrix_SpaceLimited(N2, N1):
    """
    Generates the angular meshgrid
    psimatrix shape is (N2, N1-1)

    psi_q = q*2*pi/N2
    where -M <= q <= M
    """
    # Set up a symmetric psi range based on prescription from paper
    # (symmetric about psi = 0)
    # Set up psi start, stop and step values
    psi_step = 2*np.pi/N2
    psi_min = -np.pi + psi_step/2
    psi_max = np.pi + psi_step/2
    # Set up the psi range
    psi_range = np.arange(psi_min, psi_max, psi_step).reshape(-1,1)
    # Create a meshgrid
    psimatrix = psi_range * np.ones((1,N1-1))
    return psimatrix

def sampling_grid(N1, N2, R):
    """
    Returns polar sampling grid according to
    2D Polar Discrete Fourier Transform
    """
    rmatrix = rmatrix_SpaceLimited(N2, N1, R)
    thetamatrix = thetamatrix_SpaceLimited(N2, N1)
    return rmatrix, thetamatrix

def freq_sampling_grid(N1, N2, R):
    """
    Returns polar sampling grid in the frequency domain
    according to the 2D Polar Discrete Fourier Transform
    """
    rhomatrix = rhomatrix_SpaceLimited()
    psimatrix = psimatrix_SpaceLimited()
    return rhomatrix, psimatrix
