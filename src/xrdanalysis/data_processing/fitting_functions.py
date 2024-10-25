import numpy as np


# Define the Gaussian function
def Gauss(x, amp, cen, wid):
    """
    Gaussian function to model a peak.

    :param x: The independent variable (e.g., position or time).
    :type x: numpy.ndarray or float
    :param amp: The amplitude of the Gaussian peak.
    :type amp: float
    :param cen: The center of the Gaussian peak.
    :type cen: float
    :param wid: The width (standard deviation) of the Gaussian peak.
    :type wid: float
    :return: The value of the Gaussian function at the given x.
    :rtype: numpy.ndarray or float
    """
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid**2))


def bg_decay_fn(x, fourth_degree_coef, constant):
    """
    Background decay function with a fourth-degree term and a constant.

    :param x: The independent variable.
    :type x: numpy.ndarray or float
    :param fourth_degree_coef: Coefficient for the fourth-degree decay term.
    :type fourth_degree_coef: float
    :param constant: Constant term for the background.
    :type constant: float
    :return: The value of the background decay function at the given x.
    :rtype: numpy.ndarray or float
    """
    return fourth_degree_coef / x**4 + constant


def poly_gauss(x, *params):
    """
    Composite function that sums Gaussian functions with a polynomial \
    background.

    :param x: The independent variable.
    :type x: numpy.ndarray or float
    :param params: Parameters for multiple Gaussian functions and the \
    background. The first part of params consists of 3 parameters per \
    Gaussian (amplitude, center, width), followed by 2 parameters for \
    the background (fourth-degree coefficient, constant).
    :type params: list or numpy.ndarray
    :return: The value of the composite function at the given x.
    :rtype: numpy.ndarray
    """
    x = np.array(x)

    # Last two parameters are for the polynomial background
    poly = params[-2:]
    gaussians = params[:-2]

    fourth_degree_coef = poly[0]
    constant = poly[1]

    # Compute the background decay
    y = bg_decay_fn(x, fourth_degree_coef, constant)

    # Split the rest of the parameters into sets of 3 (amp, cen, wid) for
    # Gaussians
    num_gaussians = (len(gaussians)) // 3
    gaussians = [
        params[i : i + 3] for i in range(0, num_gaussians * 3, 3)  # noqa: E203
    ]

    # Sum up all Gaussian contributions
    y += sum([Gauss(x, *params) for params in gaussians])

    return y
