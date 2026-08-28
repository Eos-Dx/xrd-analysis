"""Contracts for the optional native CUDA direct Monte Carlo backend."""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

from xrdanalysis import direct_monte_carlo_cuda_native as native_cuda
from xrdanalysis.direct_monte_carlo import NativeDirectMonteCarloPlan


def _simple_plan() -> NativeDirectMonteCarloPlan:
    return NativeDirectMonteCarloPlan(
        image_shape=(1, 4),
        csc_indptr=np.array([0, 1, 2, 3, 4]),
        csc_indices=np.array([0, 0, 1, 1]),
        csc_weights=np.ones(4),
        normalization_denominators=np.array([2.0, 2.0]),
        q_grid=np.array([1.0, 2.0]),
        q_normalization_band=(1.0, 1.0),
    )


def _image() -> np.ndarray:
    return np.array([[100.0, 120.0, 80.0, 90.0]])


def _require_native_cuda() -> None:
    if native_cuda.native_cuda_backend_available():
        return
    if os.environ.get("XRDANALYSIS_REQUIRE_CUDA_TESTS") == "1":
        pytest.fail("native CUDA tests are required but no backend/device is available")
    pytest.skip("native CUDA backend is unavailable")


def test_invalid_request_is_rejected_before_native_library_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        native_cuda,
        "_load_native_cuda_library",
        lambda: pytest.fail("native CUDA library must not load"),
    )
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        native_cuda.direct_detector_monte_carlo_cuda_native(
            _simple_plan(),
            _image(),
            (1.0,),
            0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"scales": ()}, "scales must be a non-empty"),
        ({"scales": (0.0,)}, "scales must be finite and positive"),
        ({"seed": -1}, "seed must be an unsigned"),
        ({"device": -1}, "device must be a non-negative"),
        ({"threads_per_block": 0}, "threads_per_block must be a positive"),
        ({"threads_per_block": 1025}, "threads_per_block must not exceed"),
        ({"profile_batch_size": 0}, "profile_batch_size must be a positive"),
    ],
)
def test_request_validation_precedes_native_cuda_loading(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    message: str,
):
    monkeypatch.setattr(
        native_cuda,
        "_load_native_cuda_library",
        lambda: pytest.fail("native CUDA library must not load"),
    )
    parameters = {
        "scales": (1.0,),
        "seed": 0,
        "device": 0,
        "threads_per_block": 256,
        "profile_batch_size": 4096,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=message):
        native_cuda.direct_detector_monte_carlo_cuda_native(
            _simple_plan(),
            _image(),
            parameters.pop("scales"),
            10,
            **parameters,
        )


def test_missing_native_cuda_library_is_explicit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(
        "XRDANALYSIS_DIRECT_MONTE_CARLO_CUDA_LIBRARY",
        raising=False,
    )
    monkeypatch.setattr(native_cuda.platform, "system", lambda: "Darwin")
    native_cuda._load_native_cuda_library.cache_clear()
    assert not native_cuda.native_cuda_backend_available()
    with pytest.raises(
        native_cuda.NativeCudaBackendUnavailableError,
        match="requires Linux",
    ):
        native_cuda.direct_detector_monte_carlo_cuda_native(
            _simple_plan(),
            _image(),
            (1.0,),
            10,
        )
    native_cuda._load_native_cuda_library.cache_clear()


def test_native_cuda_status_is_propagated(monkeypatch: pytest.MonkeyPatch):
    class RejectingLibrary:
        @staticmethod
        def xrdmc_cuda_run(*arguments: object) -> int:
            error_buffer = arguments[-2]
            assert isinstance(error_buffer, ctypes.Array)
            error_buffer.value = b"test CUDA failure"
            return 7

    monkeypatch.setattr(
        native_cuda,
        "_load_native_cuda_library",
        lambda: RejectingLibrary(),
    )
    with pytest.raises(
        native_cuda.NativeCudaBackendError,
        match="status 7: test CUDA failure",
    ):
        native_cuda.direct_detector_monte_carlo_cuda_native(
            _simple_plan(),
            _image(),
            (0.5, 1.0),
            10,
        )


def test_large_implicit_host_output_is_rejected_before_library_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        native_cuda,
        "_load_native_cuda_library",
        lambda: pytest.fail("native CUDA library must not load"),
    )
    with pytest.raises(ValueError, match="process patient-sized batches"):
        native_cuda.direct_detector_monte_carlo_cuda_native(
            _simple_plan(),
            _image(),
            (1.0,),
            70_000_000,
        )


def test_caller_provided_output_is_used(monkeypatch: pytest.MonkeyPatch):
    class SuccessfulLibrary:
        @staticmethod
        def xrdmc_cuda_run(*arguments: object) -> int:
            return 0

    monkeypatch.setattr(
        native_cuda,
        "_load_native_cuda_library",
        lambda: SuccessfulLibrary(),
    )
    output = np.zeros((2, 10, 1, 2), dtype=np.float64)
    actual = native_cuda.direct_detector_monte_carlo_cuda_native(
        _simple_plan(),
        _image(),
        (0.5, 1.0),
        10,
        output=output,
    )
    assert actual is output


def test_equivalent_measurement_plans_use_one_cuda_call(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[int, tuple[int, ...]]] = []

    def record_call(
        plan: NativeDirectMonteCarloPlan,
        images: np.ndarray,
        scales: tuple[float, ...],
        draws: int,
        **kwargs: object,
    ) -> np.ndarray:
        calls.append((draws, images.shape))
        return np.zeros((len(scales), draws, images.shape[0], plan.bins))

    monkeypatch.setattr(
        native_cuda,
        "direct_detector_monte_carlo_cuda_native",
        record_call,
    )
    plan = _simple_plan()
    q_grid, profiles = native_cuda.direct_detector_monte_carlo_cuda_native_measurements(
        (plan, plan),
        (_image(), _image() + 10.0),
        (0.5, 1.0),
        10,
    )
    np.testing.assert_array_equal(q_grid, plan.q_grid)
    assert profiles.shape == (2, 10, 2, 2)
    assert calls == [(10, (2, 1, 4))]


def test_native_cuda_output_and_repeatability_when_available():
    _require_native_cuda()
    plan = _simple_plan()
    first = native_cuda.direct_detector_monte_carlo_cuda_native(
        plan,
        _image(),
        (0.5, 1.0),
        100,
        seed=29,
        threads_per_block=128,
        profile_batch_size=13,
    )
    second = native_cuda.direct_detector_monte_carlo_cuda_native(
        plan,
        _image(),
        (0.5, 1.0),
        100,
        seed=29,
        threads_per_block=256,
        profile_batch_size=31,
    )
    assert first.shape == (2, 100, 1, 2)
    np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(first[..., 0], 1.0, rtol=0.0, atol=1e-12)


def test_native_cuda_deterministic_integration_parity_when_available():
    _require_native_cuda()
    actual = native_cuda.integrate_detector_frames_cuda_native(
        _simple_plan(),
        _image(),
    )
    expected = np.array([[1.0, (80.0 + 90.0) / (100.0 + 120.0)]])
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_native_cuda_integration_matches_pyfai_plan_when_available():
    _require_native_cuda()
    pytest.importorskip("pyFAI")
    from pyFAI.detectors import Detector

    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except ImportError:
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    detector = Detector(1e-4, 1e-4)
    integrator = AzimuthalIntegrator(detector=detector)
    integrator.setFit2D(100.0, 16.0, 16.0, wavelength=1.54)
    image = np.arange(32 * 32, dtype=float).reshape(32, 32) + 100.0
    result = integrator.integrate1d(
        image,
        32,
        error_model="poisson",
        method=("bbox", "csr", "cython"),
        unit="q_nm^-1",
        correctSolidAngle=True,
    )
    q_grid = np.asarray(result.radial, dtype=float)
    band = (float(q_grid[12]), float(q_grid[18]))
    from xrdanalysis.direct_monte_carlo import prepare_native_plan

    plan = prepare_native_plan(
        integrator,
        image.shape,
        normalization_denominators=result.sum_normalization,
        q_grid=q_grid,
        q_normalization_band=band,
    )
    actual = native_cuda.integrate_detector_frames_cuda_native(plan, image)[0]
    expected = np.asarray(result.intensity, dtype=float)
    selected = (q_grid >= band[0]) & (q_grid <= band[1])
    expected /= np.median(expected[selected])
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10)


def test_native_cuda_matches_centered_poisson_statistics_when_available():
    _require_native_cuda()
    draws = 20_000
    actual = native_cuda.direct_detector_monte_carlo_cuda_native(
        _simple_plan(),
        _image(),
        (1.0,),
        draws,
        seed=41,
    )[0, :, 0, 1]
    rng = np.random.default_rng(73)
    sampled = rng.poisson(_image(), size=(draws, 1, 4)).reshape(draws, 4)
    expected = sampled[:, 2:].sum(axis=1) / sampled[:, :2].sum(axis=1)
    standard_error = np.sqrt(
        np.var(actual, ddof=1) / draws + np.var(expected, ddof=1) / draws
    )
    assert abs(np.mean(actual) - np.mean(expected)) <= 5.0 * standard_error
    assert 0.90 <= np.var(actual, ddof=1) / np.var(expected, ddof=1) <= 1.10
