import numpy as np
from scipy import fft
from scipy.integrate import simpson as simps


def slope_removal_custom(curve):
    """
    Removes the slope from a curve by rotating the curve such that the first \
    and last values are set to zero.

    :param curve: The input curve that requires slope removal.
    :type curve: numpy.ndarray
    :return: A tuple containing the flattened curve and the transformed x \
    values.
    :rtype: tuple (numpy.ndarray, numpy.ndarray)
    """
    curve = curve - curve[0]
    curve = list(curve)
    length = len(curve)
    x_range = np.zeros(length)
    curve_flat = np.zeros(length)
    angle = np.arctan(curve[-1] / (length - 1))
    for j in range(length):
        curve_flat[j] = curve[j] * np.cos(angle) - j * np.sin(angle)
        x_range[j] = j * np.cos(angle) + curve[j] * np.sin(angle)
    return curve_flat, x_range


def slope_removal(y):
    """
    Removes the slope of a curve by subtracting a linear function that \
    connects the first and last values.

    :param y: The input curve where slope needs to be removed.
    :type y: numpy.ndarray
    :return: The curve with slope removed.
    :rtype: numpy.ndarray
    """
    # Number of points
    n = len(y)

    # Get first and last values
    y1 = y[0]
    yn = y[-1]

    # Create a linear function passing through (1, y1) and (n, yn)
    linear_func = np.linspace(y1, yn, n)

    # Subtract the linear function from the original curve
    adjusted_y = y - linear_func

    return adjusted_y


def fourier_custom(curve, order):
    """
    Applies a custom Fourier transform to extract the Fourier coefficients up \
    to a given order.

    :param curve: The input curve to transform using Fourier series.
    :type curve: numpy.ndarray
    :param order: The number of Fourier terms (harmonics) to consider in the \
    expansion.
    :type order: int
    :return: Concatenated list of 'an' (cosine) and 'bn' (sine) coefficients.
    :rtype: numpy.ndarray
    """
    L = len(curve)
    x = np.arange(0, L)
    an_list = np.zeros(order)
    bn_list = np.zeros(order)

    for n in range(1, order + 1):
        cos_term = np.cos(2 * np.pi * n * x / L)
        sin_term = np.sin(2 * np.pi * n * x / L)

        an_list[n - 1] = 2.0 / L * simps(curve * cos_term, x)
        bn_list[n - 1] = 2.0 / L * simps(curve * sin_term, x)

    return np.concatenate([an_list, bn_list])


def fourier_fft(curve, order):
    """
    Applies Fourier transform using FFT (Fast Fourier Transform) and extracts \
    the real and imaginary coefficients.

    :param curve: The input curve to transform using FFT.
    :type curve: numpy.ndarray
    :param order: The number of Fourier terms (harmonics) to consider in the \
    expansion.
    :type order: int
    :return: Concatenated list of real and imaginary parts of the Fourier \
    coefficients.
    :rtype: numpy.ndarray
    """
    # 0 order is ignored
    coeff = fft.fft(curve)[1 : order + 1]  # noqa: E203
    return np.concatenate([coeff.real, coeff.imag])
