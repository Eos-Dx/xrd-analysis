"""
Implements various Fourier Transform techniques
"""
import os

import numpy as np
from scipy.special import jv
from scipy.ndimage import map_coordinates

from eosdxanalysis.models.polar_sampling import sampling_grid
from eosdxanalysis.models.polar_sampling import freq_sampling_grid

MODULE_PATH = os.path.dirname(__file__)
DATA_DIR = "data"
JN_ZEROSMATRIX_FILENAME = "jn_zerosmatrix.npy"


def pfft2_SpaceLimited(discrete_sampled_function, N1, N2, R):
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

    """
    1D Fast Fourier Transform (FFT)
    """
    # Shift rows (i.e. move last half of rows to the front),
    # perform 1D FFT, then shift back
    # fnk = np.roll( np.fft.fft( np.roll(fpprimek, M+1, axis=0), N2, axis=0), -(M+1), axis=0)
    fnk = np.roll( np.fft.fft( np.roll(discrete_sampled_function, M+1, axis=0), N2, axis=0), -(M+1), axis=0)

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
    # Set up sign array
    sign = np.ones(N2)
    nrange = np.arange(-M, M+1)
    iirange=nrange+M
    sign[nrange < 0] = (-1)**abs(nrange[nrange < 0])

    jn_zerosmatrix_sub = jn_zerosmatrix[abs(nrange),:]

    jnN1 = jn_zerosmatrix_sub[iirange, N1-1]

    Y = np.einsum('i,ijk->ijk', sign, YmatrixAssembly(abs(nrange), N1, jn_zerosmatrix_sub))

    fnl = np.einsum('nk,nlk->nl', fnk, Y)
    Fnl = np.einsum('nl,n->nl', fnl, (2*np.pi*(np.power(1j, -nrange)))*(R**2/jnN1))

    return Fnl

def YmatrixAssembly(n, N1, jn_zeros):
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
    - Ymatrix of shape (n', N1-1, N1-1) with indices n', m, k
    """
    if np.issubdtype(type(n), np.ndarray):
        if len(n.shape) == 1:
            shape = (n.size, N1-1, N1-1)
            n = n.reshape(-1,1)
        else:
            raise NotImplementedError("Input array n with more than 1 dimension {} is not implemented.".format(n.shape))
    elif np.issubdtype(type(n), np.integer):
        shape = (1, N1-1, N1-1)
        jn_zeros = jn_zeros[:N1].reshape(1,N1)
        n = np.array([[n]])
    else:
        raise ValueError("Input `n` must be an int or array of int, not {}.".format(type(n)))

    # jnN1 is the last element
    jnN1 = jn_zeros[:, N1-1,...].reshape(shape[0],1)
    # jnk is all but the last element, row vector
    jnk = jn_zeros[:, :N1-1,...].reshape((shape[0], shape[1]))
    # jnm is a column vector (m is a row index)
    jnm = jnk

    denominator = np.reshape(jnN1*(jv(n+1, jnk)**2), (shape[0], shape[1], 1))
    # We have n',k,m indices now, use Einstein notation to get 3D array
    Jn_arg = np.einsum('nk,nm->nkm', jnk/jnN1, jnm).reshape(shape)
    Jn_term = (jv(n, Jn_arg.reshape(shape[0], shape[1]*shape[2]))).reshape(shape)
    # Multiply again using Einstein notation

    Ymatrix = np.einsum('ijk,ijk->ikj', Jn_term, 2/denominator)

    if Ymatrix.shape[0] == 1:
        Ymatrix = Ymatrix.squeeze()

    return Ymatrix

def ipfft2_SpaceLimited(discrete_sampled_function, N1, N2, R):
    """
    Inverse Polar Fourier Transform

    Input is a cartesian image, maximum radius R, and sampling grid
    parameters N1 and N2.

    N1 is the radial sampling rate.
    N2 is the angular sampling rate.

    The sampling grid has size (N2, N1-1)
    Input is a cartesian image, maximum radius R, and sampling grid
    """
    if N2 % 2 == 0:
        raise ValueError("N2 must be odd!")

    M = int((N2-1)//2)

    # Convert to equispaced polar coordinates
    grid_shape = (N2, N1-1)

    """
    1D Fast Fourier Transform (FFT)
    """
    # Shift rows (i.e. move last half of rows to the front),
    # perform 1D FFT, then shift back
    FNL = np.roll( np.fft.fft( np.roll(discrete_sampled_function, M+1, axis=0), N2, axis=0), -(M+1), axis=0)

    """
    1D Discrete Hankel Transform (DHT)
    """
    # Perform the 1D DHT
    fnk = idht(FNL, N2, N1, R)

    """
    1D Inverse Fast Fourier Transform (IFFT)
    """
    IDFT = np.roll( np.fft.ifft( np.roll(fnk, M+1, axis=0), N2, axis=0), -(M+1), axis=0)

    return IDFT

def idht(FNL, N2, N1, R, jn_zerosmatrix=None):
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

    Fnk = np.zeros(FNL.shape, dtype=np.complex64)
    fnk = np.zeros(FNL.shape, dtype=np.complex64)

    # n ranges from -M to M inclusive
    # Set up sign array
    sign = np.ones(N2)
    nrange = np.arange(-M, M+1)
    # Use index notation, so that n = -M corresponds to index 0
    iirange=nrange+M
    sign[nrange < 0] = (-1)**abs(nrange[nrange < 0])

    jn_zerosmatrix_sub = jn_zerosmatrix[abs(nrange),:]

    jnN1 = jn_zerosmatrix_sub[iirange, N1-1]

    Y = np.einsum('i,ijk->ijk', sign, YmatrixAssembly(abs(nrange), N1, jn_zerosmatrix_sub))
    Fnk = np.einsum('nk,nlk->nl', FNL, Y)
    fnk = np.einsum('nl,n->nl', Fnk, (jnN1)*np.power(1j,nrange))/(2*np.pi*(R**2))

    return fnk
