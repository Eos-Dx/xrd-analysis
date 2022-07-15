"""
Implements various Fourier Transform techniques
"""
import numpy as np
from scipy.special import jv

from eosdxanalysis.models.polar_sampling_grid import rmatrix_SpaceLimited
from eosdxanalysis.models.polar_sampling_grid import thetamatrix_SpaceLimited

class PolarDiscreteFourierTransform2D(object):
    """
    Class to perform all steps of the
    2D Polar Discrete Fourier Transform
    """

    def __init__(self, N2=None, N1=None, R=None):
        """
        Initialize class
        """
        self.N2 = N2
        self.N1 = N1
        self.R = R
        self.M = int((N2-1)//2)
        return super().__init__()

    def pdfft2_SpaceLimited(self, cartesian_image,
            N1=None, N2=None, R=None):
        """
        Given a polar function that is space limited, compute the
        2D Polar Discrete Fast Fourier Transform.
        """
        N1 = N1 if N1 else self.N1
        N2 = N2 if N2 else self.N2
        R = R if R else self.R

        rmatrix = rmatrix_SpaceLimited(N2, N1, R)
        thetamatrix = thetamatrix_SpaceLimited(N2, N1, R)

        # fpprimek = cartesian_image sampled according to polar_sampling_grid

        # Shift along columns (i.e. move last half of rows to the front),
        # perform 1D FFT, then shift back
        # fnk = np.roll( np.fft( np.roll(fpprimek, M+1, axis=1), N2, axis=1), -(M+1), axis=1)

        # Now perform the 1D DHT


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
    """
    Ymatrix = np.zeros((N1-1,N1-1))

    jnN1 = jn_zerosarray[N1]

    for m in range(N1-1):
        jnm = jn_zerosarray[m]
        for k in range(N1-1):
            jnk = jn_zerosarray[k]
            denominator = jnN1*(jv(n+1, jnk)**2)
            Jn_arg = jnm*jnk/jnN1
            Ymatrix[m,k] = 2/denominator*jv(n, Jn_arg)

    return Ymatrix
