import numpy as np
from scipy.special import wofz  # For Voigt function
from scipy.special import gamma
from scipy.stats import norm


class FittingFunction:
    def get_param_count(self):
        """
        Get the number of parameters for the fitting function.

        :return: Number of parameters.
        :rtype: int
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def min_bounds(self):
        """
        Get the bounds for the fitting function parameters.

        :return: Bounds for the parameters.
        :rtype: tuple
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def max_bounds(self):
        """
        Get the maximum bounds for the fitting function parameters.

        :return: Maximum bounds for the parameters.
        :rtype: tuple
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def initial_guess(self):
        """
        Get the initial guess for the fitting function parameters.

        :return: Initial guess for the parameters.
        :rtype: list
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def returned_values(self):
        """
        Get the returned values for the fitting function parameters.

        :return: List of returned values.
        :rtype: list
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def calculate(self, x):
        """
        Calculate the fitting function value.

        :param x: Independent variable.
        :type x: numpy.ndarray or float
        :return: Fitting function value.
        :rtype: numpy.ndarray or float
        """
        raise NotImplementedError("Subclasses should implement this method.")


class FittingParameter:
    """
    Class to represent a fitting parameter with its properties.

    :param name: Name of the parameter.
    :type name: str
    :param value: Initial value of the parameter.
    :type value: float
    :param min_value: Minimum value of the parameter.
    :type min_value: float
    :param max_value: Maximum value of the parameter.
    :type max_value: float
    """

    def __init__(
        self, value, min_value=-np.inf, max_value=np.inf, returned=True
    ):
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.returned = returned


class ConstantBackground(FittingFunction):
    """
    Class to represent a constant background with its properties.

    :param name: Name of the background function.
    :type name: str
    :param func: Function to compute the background.
    :type func: callable
    :param params: Parameters for the background function.
    :type params: list
    """

    def __init__(self, constant: FittingParameter):
        self.constant = constant
        self.param_count = 1

    def get_param_count(self):
        """
        Get the number of parameters for the background function.

        :return: Number of parameters.
        :rtype: int
        """
        return self.param_count

    def min_bounds(self):
        """
        Get the bounds for the background function parameters.

        :return: Bounds for the parameters.
        :rtype: tuple
        """
        return [self.constant.min_value]

    def max_bounds(self):
        """
        Get the maximum bounds for the background function parameters.

        :return: Maximum bounds for the parameters.
        :rtype: tuple
        """
        return [self.constant.max_value]

    def initial_guess(self):
        """
        Get the initial guess for the background function parameters.

        :return: Initial guess for the parameters.
        :rtype: list
        """
        return [self.constant.value]

    def returned_values(self):
        """
        Get the returned values for the background function parameters.

        :return: List of returned values.
        :rtype: list
        """
        return [self.constant.returned]

    def calculate(self, x, *params):
        """
        Calculate the background function value.

        :param x: Independent variable.
        :type x: numpy.ndarray or float
        :return: Background function value.
        :rtype: numpy.ndarray or float
        """
        constant = params[0]
        return np.full_like(x, constant)


class InversePowerBackground(FittingFunction):
    """
    Class to represent a background function with its properties.

    :param name: Name of the background function.
    :type name: str
    :param func: Function to compute the background.
    :type func: callable
    :param params: Parameters for the background function.
    :type params: list
    """

    def __init__(
        self,
        power: FittingParameter,
        coef: FittingParameter,
        constant: FittingParameter,
    ):
        self.power = power
        self.coef = coef
        self.constant = constant
        self.param_count = 3

    def get_param_count(self):
        """
        Get the number of parameters for the background function.

        :return: Number of parameters.
        :rtype: int
        """
        return self.param_count

    def min_bounds(self):
        """
        Get the bounds for the background function parameters.

        :return: Bounds for the parameters.
        :rtype: tuple
        """
        return [
            self.power.min_value,
            self.coef.min_value,
            self.constant.min_value,
        ]

    def max_bounds(self):
        """
        Get the maximum bounds for the background function parameters.

        :return: Maximum bounds for the parameters.
        :rtype: tuple
        """
        return [
            self.power.max_value,
            self.coef.max_value,
            self.constant.max_value,
        ]

    def initial_guess(self):
        """
        Get the initial guess for the background function parameters.

        :return: Initial guess for the parameters.
        :rtype: list
        """
        return [self.power.value, self.coef.value, self.constant.value]

    def returned_values(self):
        """
        Get the returned values for the background function parameters.

        :return: List of returned values.
        :rtype: list
        """
        return [
            self.power.returned,
            self.coef.returned,
            self.constant.returned,
        ]

    def calculate(self, x, *params):
        """
        Calculate the background function value.

        :param x: Independent variable.
        :type x: numpy.ndarray or float
        :return: Background function value.
        :rtype: numpy.ndarray or float
        """
        power = params[0]
        coef = params[1]
        constant = params[2]

        return coef / (x**power) + constant


class GaussianPeak(FittingFunction):
    """
    Class to represent a Gaussian peak with its properties.

    :param name: Name of the peak function.
    :type name: str
    :param amp: Amplitude of the peak.
    :type amp: float
    :param cen: Center of the peak.
    :type cen: float
    :param sigma: Width of the peak (standard deviation).
    :type sigma: float
    """

    def __init__(
        self,
        amp: FittingParameter,
        cen: FittingParameter,
        sigma: FittingParameter,
    ):
        self.amp = amp
        self.cen = cen
        self.sigma = sigma
        self.param_count = 3

    def get_param_count(self):
        """
        Get the number of parameters for the Gaussian peak function.

        :return: Number of parameters.
        :rtype: int
        """
        return self.param_count

    def min_bounds(self):
        """
        Get the bounds for the Gaussian peak parameters.

        :return: Bounds for the parameters.
        :rtype: tuple
        """
        return [self.amp.min_value, self.cen.min_value, self.sigma.min_value]

    def max_bounds(self):
        """
        Get the maximum bounds for the Gaussian peak parameters.

        :return: Maximum bounds for the parameters.
        :rtype: tuple
        """
        return [self.amp.max_value, self.cen.max_value, self.sigma.max_value]

    def initial_guess(self):
        """
        Get the initial guess for the Gaussian peak parameters.

        :return: Initial guess for the parameters.
        :rtype: list
        """
        return [self.amp.value, self.cen.value, self.sigma.value]

    def returned_values(self):
        """
        Get the returned values for the Gaussian peak parameters.

        :return: List of returned values.
        :rtype: list
        """
        return [self.amp.returned, self.cen.returned, self.sigma.returned]

    def calculate(self, x, *params):
        """
        Calculate the Gaussian peak value.

        :param x: Independent variable.
        :type x: numpy.ndarray or float
        :return: Gaussian peak value.
        :rtype: numpy.ndarray or float
        """
        amp = params[0]
        cen = params[1]
        sigma = params[2]

        return amp * np.exp(-((x - cen) ** 2) / (2 * sigma**2))


class LorentzianPeak(FittingFunction):
    """
    Class to represent a Lorentzian peak with its properties.

    :param name: Name of the peak function.
    :type name: str
    :param amp: Amplitude of the peak.
    :type amp: float
    :param cen: Center of the peak.
    :type cen: float
    :param gamma: Half-width at half-maximum of the peak.
    :type gamma: float
    """

    def __init__(
        self,
        amp: FittingParameter,
        cen: FittingParameter,
        gamma: FittingParameter,
    ):
        self.amp = amp
        self.cen = cen
        self.gamma = gamma
        self.param_count = 3

    def get_param_count(self):
        """
        Get the number of parameters for the Gaussian peak function.

        :return: Number of parameters.
        :rtype: int
        """
        return self.param_count

    def min_bounds(self):
        """
        Get the bounds for the Lorentzian peak parameters.

        :return: Bounds for the parameters.
        :rtype: tuple
        """
        return [self.amp.min_value, self.cen.min_value, self.gamma.min_value]

    def max_bounds(self):
        """
        Get the maximum bounds for the Lorentzian peak parameters.

        :return: Maximum bounds for the parameters.
        :rtype: tuple
        """
        return [self.amp.max_value, self.cen.max_value, self.gamma.max_value]

    def initial_guess(self):
        """
        Get the initial guess for the Lorentzian peak parameters.

        :return: Initial guess for the parameters.
        :rtype: list
        """
        return [self.amp.value, self.cen.value, self.gamma.value]

    def returned_values(self):
        """
        Get the returned values for the Lorentzian peak parameters.

        :return: List of returned values.
        :rtype: list
        """
        return [self.amp.returned, self.cen.returned, self.gamma.returned]

    def calculate(self, x, *params):
        """
        Calculate the Lorentzian peak value.

        :param x: Independent variable.
        :type x: numpy.ndarray or float
        :return: Lorentzian peak value.
        :rtype: numpy.ndarray or float
        """
        amp = params[0]
        cen = params[1]
        gamma = params[2]

        return amp * (gamma**2 / ((x - cen) ** 2 + gamma**2))


class VoigtPeak(FittingFunction):
    """
    Class to represent a Voigt peak with its properties.

    :param name: Name of the peak function.
    :type name: str
    :param amp: Amplitude of the peak.
    :type amp: float
    :param cen: Center of the peak.
    :type cen: float
    :param sigma: Gaussian width of the peak.
    :type sigma: float
    :param gamma: Lorentzian half-width of the peak.
    :type gamma: float
    """

    def __init__(
        self,
        amp: FittingParameter,
        cen: FittingParameter,
        sigma: FittingParameter,
        gamma: FittingParameter,
    ):
        self.amp = amp
        self.cen = cen
        self.sigma = sigma
        self.gamma = gamma
        self.param_count = 4

    def get_param_count(self):
        """
        Get the number of parameters for the Gaussian peak function.

        :return: Number of parameters.
        :rtype: int
        """
        return self.param_count

    def min_bounds(self):
        """
        Get the bounds for the Voigt peak parameters.

        :return: Bounds for the parameters.
        :rtype: tuple
        """
        return [
            self.amp.min_value,
            self.cen.min_value,
            self.sigma.min_value,
            self.gamma.min_value,
        ]

    def max_bounds(self):
        """
        Get the maximum bounds for the Voigt peak parameters.

        :return: Maximum bounds for the parameters.
        :rtype: tuple
        """
        return [
            self.amp.max_value,
            self.cen.max_value,
            self.sigma.max_value,
            self.gamma.max_value,
        ]

    def initial_guess(self):
        """
        Get the initial guess for the Voigt peak parameters.

        :return: Initial guess for the parameters.
        :rtype: list
        """
        return [
            self.amp.value,
            self.cen.value,
            self.sigma.value,
            self.gamma.value,
        ]

    def returned_values(self):
        """
        Get the returned values for the Voigt peak parameters.

        :return: List of returned values.
        :rtype: list
        """
        return [
            self.amp.returned,
            self.cen.returned,
            self.sigma.returned,
            self.gamma.returned,
        ]

    def calculate(self, x, *params):
        """
        Calculate the Voigt peak value.

        :param x: Independent variable.
        :type x: numpy.ndarray or float
        :return: Voigt peak value.
        :rtype: numpy.ndarray or float
        """
        amp = params[0]
        cen = params[1]
        sigma = params[2]
        gamma = params[3]

        z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
        return amp * wofz(z).real / (sigma * np.sqrt(2 * np.pi))


class SkewedVoigtPeak(FittingFunction):
    """
    Class to represent a skewed Voigt peak with its properties.

    :param name: Name of the peak function.
    :type name: str
    :param amp: Amplitude of the peak.
    :type amp: float
    :param mu: Center of the peak.
    :type mu: float
    :param sigma: Gaussian width of the peak.
    :type sigma: float
    :param gamma: Lorentzian half-width of the peak.
    :type gamma: float
    :param alpha: Skewness parameter.
    :type alpha: float
    """

    def __init__(
        self,
        amp: FittingParameter,
        cen: FittingParameter,
        sigma: FittingParameter,
        gamma: FittingParameter,
        alpha: FittingParameter,
    ):
        self.amp = amp
        self.cen = cen
        self.sigma = sigma
        self.gamma = gamma
        self.alpha = alpha
        self.param_count = 5

    def get_param_count(self):
        """
        Get the number of parameters for the Gaussian peak function.

        :return: Number of parameters.
        :rtype: int
        """
        return self.param_count

    def min_bounds(self):
        """
        Get the bounds for the skewed Voigt peak parameters.

        :return: Bounds for the parameters.
        :rtype: tuple
        """
        return [
            self.amp.min_value,
            self.cen.min_value,
            self.sigma.min_value,
            self.gamma.min_value,
            self.alpha.min_value,
        ]

    def max_bounds(self):
        """
        Get the maximum bounds for the skewed Voigt peak parameters.

        :return: Maximum bounds for the parameters.
        :rtype: tuple
        """
        return [
            self.amp.max_value,
            self.cen.max_value,
            self.sigma.max_value,
            self.gamma.max_value,
            self.alpha.max_value,
        ]

    def initial_guess(self):
        """
        Get the initial guess for the skewed Voigt peak parameters.

        :return: Initial guess for the parameters.
        :rtype: list
        """
        return [
            self.amp.value,
            self.cen.value,
            self.sigma.value,
            self.gamma.value,
            self.alpha.value,
        ]

    def returned_values(self):
        """
        Get the returned values for the skewed Voigt peak parameters.

        :return: List of returned values.
        :rtype: list
        """
        return [
            self.amp.returned,
            self.cen.returned,
            self.sigma.returned,
            self.gamma.returned,
            self.alpha.returned,
        ]

    def calculate(self, x, *params):
        """
        Calculate the skewed Voigt peak value.

        :param x: Independent variable.
        :type x: numpy.ndarray or float
        :return: Skewed Voigt peak value.
        :rtype: numpy.ndarray or float
        """
        amp = params[0]
        cen = params[1]
        sigma = params[2]
        gamma = params[3]
        alpha = params[4]

        z = ((x - cen) + 1j * gamma) / (sigma * np.sqrt(2))
        voigt = amp * wofz(z).real / (sigma * np.sqrt(2 * np.pi))
        skew_factor = norm.cdf(alpha * (x - cen) / sigma)
        return amp * 2 * voigt * skew_factor


class GammaDistributionPeak(FittingFunction):
    """
    Class to represent a Gamma distribution with its properties.

    :param name: Name of the distribution function.
    :type name: str
    :param amp: Amplitude of the distribution.
    :type amp: float
    :param position: Shift (location) parameter.
    :type position: float
    :param k: Shape parameter (must be positive).
    :type k: float
    :param theta: Scale parameter (must be positive).
    :type theta: float
    """

    def __init__(
        self,
        amp: FittingParameter,
        position: FittingParameter,
        k: FittingParameter,
        theta: FittingParameter,
    ):
        self.amp = amp
        self.position = position
        self.k = k
        self.theta = theta
        self.param_count = 4

    def get_param_count(self):
        """
        Get the number of parameters for the Gaussian peak function.

        :return: Number of parameters.
        :rtype: int
        """
        return self.param_count

    def min_bounds(self):
        """
        Get the bounds for the Gamma distribution parameters.

        :return: Bounds for the parameters.
        :rtype: tuple
        """
        return [
            self.amp.min_value,
            self.position.min_value,
            self.k.min_value,
            self.theta.min_value,
        ]

    def max_bounds(self):
        """
        Get the maximum bounds for the Gamma distribution parameters.

        :return: Maximum bounds for the parameters.
        :rtype: tuple
        """
        return [
            self.amp.max_value,
            self.position.max_value,
            self.k.max_value,
            self.theta.max_value,
        ]

    def initial_guess(self):
        """
        Get the initial guess for the Gamma distribution parameters.

        :return: Initial guess for the parameters.
        :rtype: list
        """
        return [
            self.amp.value,
            self.position.value,
            self.k.value,
            self.theta.value,
        ]

    def returned_values(self):
        """
        Get the returned values for the Gamma distribution parameters.

        :return: List of returned values.
        :rtype: list
        """
        return [
            self.amp.returned,
            self.position.returned,
            self.k.returned,
            self.theta.returned,
        ]

    def calculate(self, x, *params):
        """
        Calculate the Gamma distribution value.

        :param x: Independent variable.
        :type x: numpy.ndarray or float
        :return: Gamma distribution value.
        :rtype: numpy.ndarray or float
        """
        amp = params[0]
        position = params[1]
        k = params[2]
        theta = params[3]

        if k <= 0 or theta <= 0:
            raise ValueError(
                "Shape (k) and scale (theta) parameters must be positive."
            )

        adjusted_x = x - position

        # Ensure valid range for the adjusted x
        adjusted_x[adjusted_x < 0] = 0
        return (
            amp
            * (adjusted_x ** (k - 1) * np.exp(-adjusted_x / theta))
            / (gamma(k) * theta**k)
        )


class FittingFunctionProducer:
    """
    Class to produce fitting functions based on the provided FittingFunction
    instances.
    """

    def __init__(self, functions: list[FittingFunction]):
        self.functions = functions

    def produce_function(self):
        """
        Calculate the combined function value for all fitting functions.
        """

        def combined_function(x, *params):
            first_param = 0
            last_param = 0
            result = np.zeros_like(x)
            for function in self.functions:
                last_param += function.get_param_count()
                function.get_param_count()
                result += function.calculate(
                    x, *params[first_param:last_param]
                )
                first_param = last_param
            return result

        return combined_function

    def bounds(self):
        """
        Get the bounds for all fitting functions parameters.
        :return: Bounds for the parameters.
        :rtype: tuple
        """
        min_bounds = []
        for function in self.functions:
            min_bounds.extend(function.min_bounds())

        max_bounds = []
        for function in self.functions:
            max_bounds.extend(function.max_bounds())

        return (min_bounds, max_bounds)

    def returned_values(self):
        """
        Get the returned values for all fitting functions parameters.
        :return: List of returned values.
        :rtype: list
        """
        returned_values = []
        for function in self.functions:
            returned_values.extend(function.returned_values())
        true_indices = [i for i, val in enumerate(returned_values) if val]
        return true_indices

    def initial_guess(self):
        """
        Get the initial guess for all fitting functions parameters.
        :return: Initial guess for the parameters.
        :rtype: list
        """
        initial_guess = []
        for function in self.functions:
            initial_guess.extend(function.initial_guess())
        return initial_guess
