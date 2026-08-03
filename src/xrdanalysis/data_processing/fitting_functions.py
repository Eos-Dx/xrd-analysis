"""Composable analytical functions used by curve-fitting transformers."""

import numpy as np
from scipy.special import gamma, wofz
from scipy.stats import norm


class FittingFunction:
    """Interface implemented by every fitting-function component."""

    def get_param_count(self):
        """Return number of parameters consumed by ``calculate``."""
        raise NotImplementedError("Subclasses should implement this method.")

    def min_bounds(self):
        """Return lower parameter bounds in calculation order."""
        raise NotImplementedError("Subclasses should implement this method.")

    def max_bounds(self):
        """Return upper parameter bounds in calculation order."""
        raise NotImplementedError("Subclasses should implement this method.")

    def initial_guess(self):
        """Return initial parameter values in calculation order."""
        raise NotImplementedError("Subclasses should implement this method.")

    def returned_values(self):
        """Return flags selecting parameters exposed to callers."""
        raise NotImplementedError("Subclasses should implement this method.")

    def calculate(self, x):
        """Evaluate the fitting component."""
        raise NotImplementedError("Subclasses should implement this method.")


class FittingParameter:
    """Value, bounds, and output-selection metadata for one parameter."""

    def __init__(self, value, min_value=-np.inf, max_value=np.inf, returned=True):
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.returned = returned


def _parameter_values(function, attribute):
    """Collect one metadata attribute in a component's declared order."""
    return [
        getattr(getattr(function, name), attribute)
        for name in function._parameter_names
    ]


class ConstantBackground(FittingFunction):
    """Constant background component."""

    _parameter_names = ("constant",)

    def __init__(self, constant: FittingParameter):
        self.constant = constant
        self.param_count = 1

    def get_param_count(self):
        return self.param_count

    def min_bounds(self):
        return _parameter_values(self, "min_value")

    def max_bounds(self):
        return _parameter_values(self, "max_value")

    def initial_guess(self):
        return _parameter_values(self, "value")

    def returned_values(self):
        return _parameter_values(self, "returned")

    def calculate(self, x, *params):
        constant = params[0]
        return np.full_like(x, constant)


class InversePowerBackground(FittingFunction):
    """Inverse-power background plus constant offset."""

    _parameter_names = ("power", "coef", "constant")

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
        return self.param_count

    def min_bounds(self):
        return _parameter_values(self, "min_value")

    def max_bounds(self):
        return _parameter_values(self, "max_value")

    def initial_guess(self):
        return _parameter_values(self, "value")

    def returned_values(self):
        return _parameter_values(self, "returned")

    def calculate(self, x, *params):
        power = params[0]
        coef = params[1]
        constant = params[2]
        return coef / (x**power) + constant


class GaussianPeak(FittingFunction):
    """Gaussian peak parameterized by amplitude, center, and sigma."""

    _parameter_names = ("amp", "cen", "sigma")

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
        return self.param_count

    def min_bounds(self):
        return _parameter_values(self, "min_value")

    def max_bounds(self):
        return _parameter_values(self, "max_value")

    def initial_guess(self):
        return _parameter_values(self, "value")

    def returned_values(self):
        return _parameter_values(self, "returned")

    def calculate(self, x, *params):
        amp = params[0]
        cen = params[1]
        sigma = params[2]
        return amp * np.exp(-((x - cen) ** 2) / (2 * sigma**2))


class LorentzianPeak(FittingFunction):
    """Lorentzian peak parameterized by amplitude, center, and gamma."""

    _parameter_names = ("amp", "cen", "gamma")

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
        return self.param_count

    def min_bounds(self):
        return _parameter_values(self, "min_value")

    def max_bounds(self):
        return _parameter_values(self, "max_value")

    def initial_guess(self):
        return _parameter_values(self, "value")

    def returned_values(self):
        return _parameter_values(self, "returned")

    def calculate(self, x, *params):
        amp = params[0]
        cen = params[1]
        gamma_value = params[2]
        return amp * (gamma_value**2 / ((x - cen) ** 2 + gamma_value**2))


class VoigtPeak(FittingFunction):
    """Voigt peak with Gaussian and Lorentzian widths."""

    _parameter_names = ("amp", "cen", "sigma", "gamma")

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
        return self.param_count

    def min_bounds(self):
        return _parameter_values(self, "min_value")

    def max_bounds(self):
        return _parameter_values(self, "max_value")

    def initial_guess(self):
        return _parameter_values(self, "value")

    def returned_values(self):
        return _parameter_values(self, "returned")

    def calculate(self, x, *params):
        amp = params[0]
        cen = params[1]
        sigma = params[2]
        gamma_value = params[3]
        z = ((x - cen) + 1j * gamma_value) / (sigma * np.sqrt(2))
        return amp * wofz(z).real / (sigma * np.sqrt(2 * np.pi))


class SkewedVoigtPeak(FittingFunction):
    """Legacy skewed-Voigt peak, including its amplitude-squared scaling."""

    _parameter_names = ("amp", "cen", "sigma", "gamma", "alpha")

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
        return self.param_count

    def min_bounds(self):
        return _parameter_values(self, "min_value")

    def max_bounds(self):
        return _parameter_values(self, "max_value")

    def initial_guess(self):
        return _parameter_values(self, "value")

    def returned_values(self):
        return _parameter_values(self, "returned")

    def calculate(self, x, *params):
        amp = params[0]
        cen = params[1]
        sigma = params[2]
        gamma_value = params[3]
        alpha = params[4]
        z = ((x - cen) + 1j * gamma_value) / (sigma * np.sqrt(2))
        voigt = amp * wofz(z).real / (sigma * np.sqrt(2 * np.pi))
        skew_factor = norm.cdf(alpha * (x - cen) / sigma)
        return amp * 2 * voigt * skew_factor


class GammaDistributionPeak(FittingFunction):
    """Shifted Gamma-distribution peak."""

    _parameter_names = ("amp", "position", "k", "theta")

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
        return self.param_count

    def min_bounds(self):
        return _parameter_values(self, "min_value")

    def max_bounds(self):
        return _parameter_values(self, "max_value")

    def initial_guess(self):
        return _parameter_values(self, "value")

    def returned_values(self):
        return _parameter_values(self, "returned")

    def calculate(self, x, *params):
        amp = params[0]
        position = params[1]
        k = params[2]
        theta = params[3]
        if k <= 0 or theta <= 0:
            raise ValueError("Shape (k) and scale (theta) parameters must be positive.")

        adjusted_x = x - position
        adjusted_x[adjusted_x < 0] = 0
        numerator = amp * (adjusted_x ** (k - 1) * np.exp(-adjusted_x / theta))
        return numerator / (gamma(k) * theta**k)


class FittingFunctionProducer:
    """Aggregate fitting components into one curve-fit callable."""

    def __init__(self, functions: list[FittingFunction]):
        self.functions = functions

    def get_function_count(self):
        return len(self.functions)

    def get_function_param_counts(self):
        return [function.get_param_count() for function in self.functions]

    def produce_function(self):
        def combined_function(x, *params):
            first_param = 0
            last_param = 0
            result = np.zeros_like(x)
            for function in self.functions:
                last_param += function.get_param_count()
                result += function.calculate(x, *params[first_param:last_param])
                first_param = last_param
            return result

        return combined_function

    def bounds(self):
        min_bounds = []
        for function in self.functions:
            min_bounds.extend(function.min_bounds())

        max_bounds = []
        for function in self.functions:
            max_bounds.extend(function.max_bounds())
        return (min_bounds, max_bounds)

    def returned_values(self):
        returned_values = []
        for function in self.functions:
            returned_values.extend(function.returned_values())
        return [index for index, value in enumerate(returned_values) if value]

    def initial_guess(self):
        initial_guess = []
        for function in self.functions:
            initial_guess.extend(function.initial_guess())
        return initial_guess
