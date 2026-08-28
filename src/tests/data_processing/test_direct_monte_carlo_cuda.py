"""Contracts for the optional CUDA direct detector Monte Carlo backend."""

from __future__ import annotations

import numpy as np
import pytest

from xrdanalysis import direct_monte_carlo_cuda as cuda_mc
from xrdanalysis.direct_monte_carlo import NativeDirectMonteCarloPlan


def _simple_plan(*, q_grid: np.ndarray | None = None) -> NativeDirectMonteCarloPlan:
    q_values = np.array([1.0, 2.0]) if q_grid is None else q_grid
    return NativeDirectMonteCarloPlan(
        image_shape=(1, 4),
        csc_indptr=np.array([0, 1, 2, 3, 4]),
        csc_indices=np.array([0, 0, 1, 1]),
        csc_weights=np.ones(4),
        normalization_denominators=np.array([2.0, 2.0]),
        q_grid=q_values,
        q_normalization_band=(float(q_values[0]), float(q_values[0])),
    )


def _image() -> np.ndarray:
    return np.array([[100.0, 120.0, 80.0, 90.0]])


def test_invalid_request_is_rejected_before_cupy_import(
    monkeypatch: pytest.MonkeyPatch,
):
    imports: list[str] = []

    def record_import(name: str):
        imports.append(name)
        raise AssertionError("CuPy import must remain lazy during validation")

    monkeypatch.setattr(cuda_mc.importlib, "import_module", record_import)
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        cuda_mc.direct_detector_monte_carlo_cuda(
            _simple_plan(),
            _image(),
            (1.0,),
            0,
        )
    assert imports == []


def test_missing_cupy_raises_explicit_unavailability(
    monkeypatch: pytest.MonkeyPatch,
):
    def missing_import(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(cuda_mc.importlib, "import_module", missing_import)
    assert not cuda_mc.cuda_backend_available()
    with pytest.raises(cuda_mc.CudaBackendUnavailableError, match="requires.*CuPy"):
        cuda_mc.direct_detector_monte_carlo_cuda(
            _simple_plan(),
            _image(),
            (0.5, 1.0),
            10,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scales": ()}, "scales must be a non-empty"),
        ({"scales": (0.0,)}, "scales must be finite and positive"),
        ({"seed": -1}, "seed must be an unsigned"),
        ({"batch_draws": 0}, "batch_draws must be a positive"),
        ({"device": -1}, "device must be a non-negative"),
    ],
)
def test_request_validation_precedes_cuda_loading(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    message: str,
):
    monkeypatch.setattr(
        cuda_mc,
        "_load_cuda_stack",
        lambda: pytest.fail("CUDA stack must not load for invalid input"),
    )
    parameters = {
        "scales": (1.0,),
        "seed": 0,
        "batch_draws": 16,
        "device": None,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=message):
        cuda_mc.direct_detector_monte_carlo_cuda(
            _simple_plan(),
            _image(),
            parameters.pop("scales"),
            10,
            **parameters,
        )


def test_measurement_runner_rejects_incompatible_contracts_before_cuda_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        cuda_mc,
        "_load_cuda_stack",
        lambda: pytest.fail("CUDA stack must not load for incompatible plans"),
    )
    with pytest.raises(ValueError, match="equal non-zero length"):
        cuda_mc.direct_detector_monte_carlo_cuda_measurements(
            (_simple_plan(),),
            (),
            (1.0,),
            10,
        )
    with pytest.raises(ValueError, match="same q grid"):
        cuda_mc.direct_detector_monte_carlo_cuda_measurements(
            (_simple_plan(), _simple_plan(q_grid=np.array([1.0, 2.1]))),
            (_image(), _image()),
            (1.0,),
            10,
        )


def test_cuda_output_axes_and_seed_repeatability_when_available():
    if not cuda_mc.cuda_backend_available():
        pytest.skip("CuPy CUDA backend is unavailable")
    plan = _simple_plan()
    first = cuda_mc.direct_detector_monte_carlo_cuda(
        plan,
        _image(),
        (0.5, 1.0),
        20,
        seed=37,
        batch_draws=5,
    )
    second = cuda_mc.direct_detector_monte_carlo_cuda(
        plan,
        _image(),
        (0.5, 1.0),
        20,
        seed=37,
        batch_draws=5,
    )
    assert first.shape == (2, 20, 1, 2)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[..., 0], 1.0, rtol=0.0, atol=1e-12)


def test_distinct_measurement_plans_when_cuda_is_available():
    if not cuda_mc.cuda_backend_available():
        pytest.skip("CuPy CUDA backend is unavailable")
    plan = _simple_plan()
    q_grid, profiles = cuda_mc.direct_detector_monte_carlo_cuda_measurements(
        (plan, plan),
        (_image(), _image() + 10.0),
        (1.0,),
        10,
        seed=11,
        batch_draws=4,
    )
    np.testing.assert_array_equal(q_grid, plan.q_grid)
    assert profiles.shape == (1, 10, 2, 2)
