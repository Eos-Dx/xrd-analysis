"""Contracts for the native direct detector Monte Carlo backend."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xrdanalysis import direct_monte_carlo as direct_mc
from xrdanalysis._native import build as native_build


@pytest.fixture(scope="session")
def native_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one native library for focused parity tests."""
    if platform.system() not in {"Darwin", "Linux"}:
        pytest.skip("native direct Monte Carlo build is unsupported on this platform")
    root = Path(__file__).resolve().parents[3]
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    output = tmp_path_factory.mktemp("direct-mc-native") / f"libxrdmc{suffix}"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_direct_monte_carlo_native.py"),
            "--output",
            str(output),
        ],
        check=True,
        cwd=root,
    )
    os.environ["XRDANALYSIS_DIRECT_MONTE_CARLO_LIBRARY"] = str(output)
    direct_mc._load_native_library.cache_clear()
    assert direct_mc.native_backend_available()
    return output


def _simple_plan() -> direct_mc.NativeDirectMonteCarloPlan:
    return direct_mc.NativeDirectMonteCarloPlan(
        image_shape=(1, 4),
        csc_indptr=np.array([0, 1, 2, 3, 4]),
        csc_indices=np.array([0, 0, 1, 1]),
        csc_weights=np.ones(4),
        normalization_denominators=np.array([2.0, 2.0]),
        q_grid=np.array([1.0, 2.0]),
        q_normalization_band=(1.0, 1.0),
    )


def test_default_native_threads_reserves_two_cpus(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(direct_mc.os, "cpu_count", lambda: 14)
    assert direct_mc.default_native_threads() == 12


def test_native_builder_prefers_active_conda_openmp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (tmp_path / "include").mkdir()
    (tmp_path / "include/omp.h").touch()
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/libomp.dylib").touch()
    monkeypatch.delenv("LIBOMP_PREFIX", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))

    assert native_build._libomp_prefix() == tmp_path


def test_plan_rejects_empty_normalization_band():
    with pytest.raises(ValueError, match="contains no q-grid bins"):
        direct_mc.NativeDirectMonteCarloPlan(
            image_shape=(1, 1),
            csc_indptr=np.array([0, 1]),
            csc_indices=np.array([0]),
            csc_weights=np.array([1.0]),
            normalization_denominators=np.array([1.0]),
            q_grid=np.array([1.0]),
            q_normalization_band=(2.0, 3.0),
        )


def test_native_output_is_independent_of_thread_count(native_library: Path):
    plan = _simple_plan()
    image = np.array([[100.0, 120.0, 80.0, 90.0]])
    one_thread = plan.run(image, (0.5, 1.0, 1.5), 200, seed=17, threads=1)
    twelve_threads = plan.run(image, (0.5, 1.0, 1.5), 200, seed=17, threads=12)
    np.testing.assert_array_equal(one_thread, twelve_threads)


def test_measurement_runner_preserves_output_axes(native_library: Path):
    plan = _simple_plan()
    q, profiles = direct_mc.direct_detector_monte_carlo_measurements(
        (plan, plan),
        (
            np.array([[100.0, 120.0, 80.0, 90.0]]),
            np.array([[110.0, 130.0, 70.0, 95.0]]),
        ),
        (0.5, 1.0),
        10,
        seed=29,
        threads=2,
    )
    np.testing.assert_array_equal(q, plan.q_grid)
    assert profiles.shape == (2, 10, 2, 2)


def test_native_centered_poisson_matches_reference_statistics(native_library: Path):
    plan = _simple_plan()
    image = np.array([[100.0, 120.0, 80.0, 90.0]])
    draws = 20_000
    actual = plan.run(image, (1.0,), draws, seed=23, threads=4)[0, :, 0, 1]

    rng = np.random.default_rng(41)
    sampled = rng.poisson(image, size=(draws, *image.shape)).reshape(draws, 4)
    expected = sampled[:, 2:].sum(axis=1) / sampled[:, :2].sum(axis=1)
    mean_standard_error = np.sqrt(
        np.var(actual, ddof=1) / draws + np.var(expected, ddof=1) / draws
    )
    assert abs(np.mean(actual) - np.mean(expected)) <= 5.0 * mean_standard_error
    assert 0.90 <= np.var(actual, ddof=1) / np.var(expected, ddof=1) <= 1.10


def test_native_integration_matches_pyfai_bbox_csr(native_library: Path):
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
    mask = np.zeros_like(image, dtype=bool)
    mask[0, 0] = True
    result = integrator.integrate1d(
        image,
        32,
        mask=mask,
        error_model="poisson",
        method=("bbox", "csr", "cython"),
        unit="q_nm^-1",
        correctSolidAngle=True,
    )
    q = np.asarray(result.radial, dtype=float)
    band = (float(q[12]), float(q[18]))
    plan = direct_mc.prepare_native_plan(
        integrator,
        image.shape,
        normalization_denominators=result.sum_normalization,
        q_grid=q,
        q_normalization_band=band,
    )
    actual = direct_mc.integrate_detector_frames(plan, image, threads=4)[0]
    expected = np.asarray(result.intensity, dtype=float)
    selected = (q >= band[0]) & (q <= band[1])
    expected /= np.median(expected[selected])
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10)
