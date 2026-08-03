"""Compatibility and numerical contracts for fitting-function models."""

import inspect
import io
import pickle

import joblib
import numpy as np
import pytest

from xrdanalysis.data_processing import fitting_functions as fitting
from xrdanalysis.data_processing import transformers


def _parameter(value, lower=-np.inf, upper=np.inf, returned=True):
    return fitting.FittingParameter(value, lower, upper, returned)


def _models():
    return {
        "constant": fitting.ConstantBackground(_parameter(0.25, -1, 1)),
        "inverse": fitting.InversePowerBackground(
            _parameter(2, 0, 4),
            _parameter(3, 0, 5, False),
            _parameter(0.25, -1, 1),
        ),
        "gaussian": fitting.GaussianPeak(
            _parameter(2, 0, 5),
            _parameter(1, -2, 2, False),
            _parameter(0.5, 0.1, 2),
        ),
        "lorentzian": fitting.LorentzianPeak(
            _parameter(2, 0, 5),
            _parameter(1, -2, 2, False),
            _parameter(0.3, 0.1, 1),
        ),
        "voigt": fitting.VoigtPeak(
            _parameter(2, 0, 5),
            _parameter(1, -2, 2, False),
            _parameter(0.5, 0.1, 2),
            _parameter(0.3, 0.1, 1, False),
        ),
        "skew": fitting.SkewedVoigtPeak(
            _parameter(2, 0, 5),
            _parameter(1, -2, 2, False),
            _parameter(0.5, 0.1, 2),
            _parameter(0.3, 0.1, 1, False),
            _parameter(1.2, -3, 3),
        ),
        "gamma": fitting.GammaDistributionPeak(
            _parameter(2, 0, 5),
            _parameter(0.5, -1, 1, False),
            _parameter(2, 0.1, 5),
            _parameter(1.5, 0.1, 4),
        ),
    }


@pytest.mark.parametrize(
    "name, parameters",
    [
        ("FittingFunction", []),
        ("FittingParameter", ["value", "min_value", "max_value", "returned"]),
        ("ConstantBackground", ["constant"]),
        ("InversePowerBackground", ["power", "coef", "constant"]),
        ("GaussianPeak", ["amp", "cen", "sigma"]),
        ("LorentzianPeak", ["amp", "cen", "gamma"]),
        ("VoigtPeak", ["amp", "cen", "sigma", "gamma"]),
        ("SkewedVoigtPeak", ["amp", "cen", "sigma", "gamma", "alpha"]),
        ("GammaDistributionPeak", ["amp", "position", "k", "theta"]),
        ("FittingFunctionProducer", ["functions"]),
    ],
)
def test_public_classes_keep_canonical_module_and_constructor(name, parameters):
    cls = getattr(fitting, name)

    assert cls.__module__ == fitting.__name__
    assert list(inspect.signature(cls).parameters) == parameters
    if name in {
        "ConstantBackground",
        "InversePowerBackground",
        "GaussianPeak",
        "LorentzianPeak",
        "VoigtPeak",
        "SkewedVoigtPeak",
        "GammaDistributionPeak",
    }:
        assert cls.__bases__ == (fitting.FittingFunction,)
        assert cls in fitting.FittingFunction.__subclasses__()
        assert cls.get_param_count.__qualname__ == f"{name}.get_param_count"


@pytest.mark.parametrize(
    "name, count, lower, upper, guess, returned",
    [
        ("constant", 1, [-1], [1], [0.25], [True]),
        ("inverse", 3, [0, 0, -1], [4, 5, 1], [2, 3, 0.25], [True, False, True]),
        ("gaussian", 3, [0, -2, 0.1], [5, 2, 2], [2, 1, 0.5], [True, False, True]),
        ("lorentzian", 3, [0, -2, 0.1], [5, 2, 1], [2, 1, 0.3], [True, False, True]),
        (
            "voigt",
            4,
            [0, -2, 0.1, 0.1],
            [5, 2, 2, 1],
            [2, 1, 0.5, 0.3],
            [True, False, True, False],
        ),
        (
            "skew",
            5,
            [0, -2, 0.1, 0.1, -3],
            [5, 2, 2, 1, 3],
            [2, 1, 0.5, 0.3, 1.2],
            [True, False, True, False, True],
        ),
        (
            "gamma",
            4,
            [0, -1, 0.1, 0.1],
            [5, 1, 5, 4],
            [2, 0.5, 2, 1.5],
            [True, False, True, True],
        ),
    ],
)
def test_model_parameter_metadata_order(name, count, lower, upper, guess, returned):
    model = _models()[name]

    assert model.get_param_count() == count
    assert model.param_count == count
    assert model.min_bounds() == lower
    assert model.max_bounds() == upper
    assert model.initial_guess() == guess
    assert model.returned_values() == returned


@pytest.mark.parametrize(
    "name, expected",
    [
        ("constant", [0.25, 0.25, 0.25, 0.25]),
        ("inverse", [12.25, 3.25, 1.5833333333333333, 1.0]),
        ("gaussian", [1.2130613194252668, 2.0, 1.2130613194252668, 0.2706705664732254]),
        (
            "lorentzian",
            [0.5294117647058824, 2.0, 0.5294117647058824, 0.16513761467889906],
        ),
        (
            "voigt",
            [
                0.7705662511246848,
                1.0479115640200407,
                0.7705662511246848,
                0.3408921197162848,
            ],
        ),
        (
            "skew",
            [
                0.35467521760358195,
                2.0958231280400814,
                2.727589786895157,
                1.3523905772739953,
            ],
        ),
        (
            "gamma",
            [0.0, 0.31845836025501745, 0.4563707724734151, 0.49050592156192313],
        ),
    ],
)
def test_model_numerical_regression_vectors(name, expected):
    model = _models()[name]
    x = np.array([0.5, 1.0, 1.5, 2.0])

    np.testing.assert_allclose(
        model.calculate(x, *model.initial_guess()), expected, rtol=1e-12, atol=1e-12
    )


def test_producer_preserves_mixed_model_slicing_and_aggregation():
    models = _models()
    producer = fitting.FittingFunctionProducer([models["constant"], models["gaussian"]])
    x = np.array([0.5, 1.0, 1.5, 2.0])

    assert producer.get_function_count() == 2
    assert producer.get_function_param_counts() == [1, 3]
    assert producer.bounds() == ([-1, 0, -2, 0.1], [1, 5, 2, 2])
    assert producer.initial_guess() == [0.25, 2, 1, 0.5]
    assert producer.returned_values() == [0, 1, 3]
    np.testing.assert_allclose(
        producer.produce_function()(x, *producer.initial_guess()),
        [1.4630613194252668, 2.25, 1.4630613194252668, 0.5206705664732254],
    )


def test_models_producer_and_transformer_keep_joblib_paths():
    models = _models()
    producer = fitting.FittingFunctionProducer(list(models.values()))
    transformer = transformers.CurveFittingTransformer("q", "profile", producer)

    for value in [*models.values(), producer, transformer]:
        stream = io.BytesIO()
        joblib.dump(value, stream)
        stream.seek(0)
        restored = joblib.load(stream)
        assert type(restored) is type(value)
        assert type(restored).__module__ == type(value).__module__


def test_legacy_integer_dtype_and_local_closure_behavior_is_explicit():
    models = _models()
    integer_x = np.array([1, 2], dtype=int)
    constant = models["constant"]
    producer = fitting.FittingFunctionProducer([constant, models["gaussian"]])

    np.testing.assert_array_equal(constant.calculate(integer_x, 0.25), [0, 0])
    with pytest.raises(TypeError):
        producer.produce_function()(integer_x, *producer.initial_guess())
    with pytest.raises((AttributeError, TypeError)):
        pickle.dumps(producer.produce_function())


def test_skew_amplitude_squared_behavior_remains_frozen():
    model = _models()["skew"]
    x = np.array([0.5, 1.0, 1.5])
    _, center, sigma, gamma, alpha = model.initial_guess()

    amp_one = model.calculate(x, 1.0, center, sigma, gamma, alpha)
    amp_two = model.calculate(x, 2.0, center, sigma, gamma, alpha)
    np.testing.assert_allclose(amp_two, 4.0 * amp_one)


def test_gamma_validation_scalar_failure_and_input_copy_behavior():
    model = _models()["gamma"]
    x = np.array([-1.0, 0.5, 1.0])
    original = x.copy()

    model.calculate(x, 2.0, 0.5, 2.0, 1.5)
    np.testing.assert_array_equal(x, original)
    with pytest.raises(ValueError, match="must be positive"):
        model.calculate(x, 2.0, 0.5, 0.0, 1.5)
    with pytest.raises(ValueError, match="must be positive"):
        model.calculate(x, 2.0, 0.5, 2.0, 0.0)
    with pytest.raises(TypeError):
        model.calculate(1.0, 2.0, 0.5, 2.0, 1.5)
