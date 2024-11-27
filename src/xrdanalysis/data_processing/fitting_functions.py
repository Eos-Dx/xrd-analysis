import numpy as np
from scipy.special import wofz  # For Voigt function
from scipy.special import gamma
from scipy.stats import norm


def inverse_power_background(x, power, coef, constant):
    """
    Generate background using inverse power law with constant offset.

    :param x: Independent variable
    :type x: numpy.ndarray or float
    :param coef: Coefficient of inverse power term
    :type coef: float
    :param power: Power of inverse function
    :type power: float
    :param constant: Constant offset
    :type constant: float
    :return: Background function values
    :rtype: numpy.ndarray or float
    """
    return coef / (x**power) + constant


def gaussian_peak(x, amp, cen, sigma):
    """
    Generate Gaussian peak function.

    :param x: Independent variable
    :type x: numpy.ndarray or float
    :param amp: Peak amplitude
    :type amp: float
    :param cen: Peak center
    :type cen: float
    :param sigma: Peak width (standard deviation)
    :type sigma: float
    :return: Gaussian peak values
    :rtype: numpy.ndarray or float
    """
    return amp * np.exp(-((x - cen) ** 2) / (2 * sigma**2))


def lorentzian_peak(x, amp, cen, gamma):
    """
    Generate Lorentzian peak function.

    :param x: Independent variable
    :type x: numpy.ndarray or float
    :param amp: Peak amplitude
    :type amp: float
    :param cen: Peak center
    :type cen: float
    :param gamma: Peak half-width at half-maximum
    :type gamma: float
    :return: Lorentzian peak values
    :rtype: numpy.ndarray or float
    """
    return amp * (gamma**2 / ((x - cen) ** 2 + gamma**2))


def voigt_peak(x, amp, cen, sigma, gamma):
    """
    Generate Voigt peak function.

    :param x: Independent variable
    :type x: numpy.ndarray or float
    :param amp: Peak amplitude
    :type amp: float
    :param cen: Peak center
    :type cen: float
    :param sigma: Gaussian width
    :type sigma: float
    :param gamma: Lorentzian half-width
    :type gamma: float
    :return: Voigt peak values
    :rtype: numpy.ndarray or float
    """
    z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
    return amp * wofz(z).real / (sigma * np.sqrt(2 * np.pi))


def skewed_voigt(x, amp, mu, sigma, gamma, alpha):
    """
    Compute the skewed Voigt profile.
    Parameters:
        x (array-like): Input values.
        mu (float): Center of the profile.
        sigma (float): Gaussian standard deviation.
        gamma (float): Lorentzian half-width at half-maximum.
        alpha (float): Skewness parameter (positive for right skew, negative \
        for left skew).
    Returns:
        array-like: Skewed Voigt profile values.
    """
    voigt = voigt_peak(x, amp, mu, sigma, gamma)
    skew_factor = norm.cdf(alpha * (x - mu) / sigma)
    return amp * 2 * voigt * skew_factor


def gamma_distribution(x, amp, position, k, theta):
    """
    Compute the Gamma distribution with amplitude and position parameters.
    Parameters:
        x (array-like): Input values (x > position).
        k (float): Shape parameter.
        theta (float): Scale parameter.
        amplitude (float): Amplitude of the distribution.
        position (float): Shift (location) parameter.
    Returns:
        array-like: Gamma distribution PDF values.
    """
    if k <= 0 or theta <= 0:
        raise ValueError(
            "Shape (k) and scale (theta) parameters must be positive."
        )

    adjusted_x = x - position
    print(adjusted_x)
    # Ensure valid range for the adjusted x
    adjusted_x[adjusted_x < 0] = 0
    return (
        amp
        * (adjusted_x ** (k - 1) * np.exp(-adjusted_x / theta))
        / (gamma(k) * theta**k)
    )


def create_poly_peak_fn(background_power=4, peak_types=None):
    def poly_peak_fn(x, *params):
        """
        Composite function supporting multiple peaks and flexible background.

        :param x: Independent variable
        :type x: numpy.ndarray or float
        :param params: Parameters for multiple peaks and background
        :type params: list or numpy.ndarray
        :param peak_types: List of peak types ('gauss', 'lorentz', 'voigt')
        :type peak_types: list, optional
        :param background_type: Type of background function
        :type background_type: str, optional
        :return: Composite peak function with background
        :rtype: numpy.ndarray or float
        """
        x = np.array(x)

        # Last parameters are for background
        # coef, constant
        background = inverse_power_background(
            x, background_power, params[-2], params[-1]
        )
        peak_params = params[:-2]

        # Track parameter index
        param_idx = 0

        # Add peaks based on specified types
        if peak_types is None:
            local_peak_types = ["voigt"] * (len(peak_params) // 4)
        else:
            local_peak_types = peak_types

        y = background
        for peak_type in local_peak_types:
            if peak_type == "gauss":
                y += gaussian_peak(
                    x, *peak_params[param_idx : param_idx + 3]  # noqa: E203
                )
                param_idx += 3

            elif peak_type == "lorentz":
                y += lorentzian_peak(
                    x, *peak_params[param_idx : param_idx + 3]  # noqa: E203
                )
                param_idx += 3

            elif peak_type == "gamma":
                y += gamma_distribution(
                    x, *peak_params[param_idx : param_idx + 4]  # noqa: E203
                )
                param_idx += 4

            elif peak_type == "voigt":
                y += voigt_peak(
                    x, *peak_params[param_idx : param_idx + 4]  # noqa: E203
                )
                param_idx += 4

            elif peak_type == "skewed_voigt":
                y += skewed_voigt(
                    x, *peak_params[param_idx : param_idx + 5]  # noqa: E203
                )
                param_idx += 5

        return y

    return poly_peak_fn
