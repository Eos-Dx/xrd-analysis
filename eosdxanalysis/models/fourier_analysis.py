"""
Implements various Fourier Transform techniques
"""
import os

import numpy as np
from scipy.special import jv
from scipy.ndimage import map_coordinates
from skimage.transform import warp_polar

from eosdxanalysis.models.polar_sampling_grid import sampling_grid
from eosdxanalysis.models.utils import pol2cart
from eosdxanalysis.preprocessing.image_processing import unwarp_polar

MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
JN_ZEROSMATRIX_FILENAME = "jn_zerosmatrix.npy"


def pfft2_SpaceLimited(cartesian_image, N1, N2, R):
    """
    Function to perform the 2D Polar Discrete Fast Fourier Transform

    Input is a cartesian image, maximum radius R, and sampling grid
    parameters N1 and N2.

    N1 is the radial sampling rate.
    N2 is the angular sampling rate.

    The sampling grid has size (N2, N1-1)
    """
    if N2 % 2 == 0:
        raise ValueError("N2 must be odd!")

    M = int((N2-1)//2)

    rmatrix, thetamatrix = sampling_grid(N1, N2, R)

    # Convert to equispaced polar coordinates
    grid_shape = (N2, N1-1)

    # Sample according to polar sampling grid
    # Convert rmatrix and thetamatrix to (row, col) indices
    # rindices = rmatrix / (R) * (N1-1)
    # thetaindices = (thetamatrix + np.pi) / (2*np.pi) * N2
    # polarindices = [thetaindices, rindices]

    # Now convert the sampling grid to cartesian coordinates
    X, Y = pol2cart(rmatrix, thetamatrix)

    # Convert cartesian coordinates to array notation
    center = (cartesian_image.shape[0]/2-0.5, cartesian_image.shape[1]/2-0.5)
    row_indices = X + center[0]
    col_indices = center[1] - Y

    cart_indices = [row_indices, col_indices]

    # Sample our image on the Baddour polar grid
    fpprimek = map_coordinates(cartesian_image, cart_indices, cval=0, order=1, mode='grid-wrap')

    """
    1D Fast Fourier Transform (FFT)
    """
    # Shift rows (i.e. move last half of rows to the front),
    # perform 1D FFT, then shift back
    fnk = np.roll( np.fft.fft( np.roll(fpprimek, M+1, axis=0), N2, axis=0), -(M+1), axis=0)

    """
    1D Discrete Hankel Transform (DHT)
    """
    # Perform the 1D DHT
    Fnl = dht(fnk, N2, N1, R)

    """
    1D Inverse Fast Fourier Transform (IFFT)
    """
    TwoDFT = np.roll( np.fft.ifft( np.roll(Fnl, M+1, axis=0), N2, axis=0), -(M+1), axis=0)

    return TwoDFT

def dht(fnk, N2, N1, R, jn_zerosmatrix=None):
    """
    Performs the 1D Discrete Hankel Transform (DHT)
    """
    # If the jn_zerosmatrix is not given, read from file
    if jn_zerosmatrix is None:
        jn_zerosmatrix_path = os.path.join(
                MODULE_PATH,
                DATA_DIR,
                JN_ZEROSMATRIX_FILENAME)
        jn_zerosmatrix = np.load(jn_zerosmatrix_path)

    M = int((N2-1)//2)

    fnl = np.zeros(fnk.shape, dtype=np.complex64)
    Fnl = np.zeros(fnk.shape, dtype=np.complex64)

    # n ranges from -M to M inclusive
    for n in np.arange(-M, M+1):
        # Use index notation, so that n = -M corresponds to index 0
        ii=n+M
        zero2=jn_zerosmatrix[abs(n),:]
        jnN1=zero2[N1-1];

        if n < 0:
            Y = ((-1)^abs(n))*YmatrixAssembly(abs(n),N1,zero2);
        else:
            Y = YmatrixAssembly(abs(n),N1,zero2);

        fnl[ii,:] = ( Y @ fnk[ii,:].T ).T;
        Fnl[ii,:] = fnl[ii,:] * (2*np.pi*(np.power(1j, -n)))*(R**2/jnN1);

    return Fnl

def YmatrixAssembly(n, N1, jn_zerosarray):
    """
    Assemble the Y-matrix for performing the 1D DHT step
    of the 2D Polar Discrete Fourier Transform

    Inputs:
    - Y is the N-1 x N-1 transformation matrix to be assembled
    - n is the order of the bessel function
    - N1 is the size of the transformation matrix
    - jn_zerosarray are the bessel zeros of Jn only,
      with size (N+1,1)

    Output:
    - Ymatrix of shape (N1-1, N1-1) with indices m, k
    """

    # jnN1 is the last element
    jnN1 = jn_zerosarray[N1-1]
    # jnk is all but the last element, row vector
    jnk = jn_zerosarray[:N1-1].reshape(1,-1)
    # jnm is a column vector (m is a row index)
    jnm = jnk.T

    denominator = jnN1*(jv(n+1, jnk)**2)
    Jn_arg = jnm*jnk/jnN1
    Ymatrix = 2/denominator*jv(n, Jn_arg)

    return Ymatrix
