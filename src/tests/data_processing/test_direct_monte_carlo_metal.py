"""Contracts for the native Apple Metal direct Monte Carlo backend."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xrdanalysis import direct_monte_carlo_metal as metal_mc
from xrdanalysis.direct_monte_carlo import NativeDirectMonteCarloPlan
from xrdanalysis.direct_monte_carlo_geometry_metal import (
    GeometryAwareMetalMonteCarlo,
    MetalDetectorGeometry,
)
from xrdanalysis.direct_monte_carlo_metal_session import (
    FrameMaskedPreparedGeometryMetalMonteCarlo,
    GroupedPersistentMetalMonteCarlo,
    PreparedGeometryMetalMonteCarlo,
    PersistentMetalMonteCarlo,
    metal_plan_fingerprint,
)


@pytest.fixture(scope="session")
def metal_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build one Metal bridge and verify runtime shader compilation."""
    if platform.system() != "Darwin":
        pytest.skip("native Metal backend requires macOS")
    root = Path(__file__).resolve().parents[3]
    output = tmp_path_factory.mktemp("direct-mc-metal") / "libxrdmc-metal.dylib"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/build_direct_monte_carlo_metal.py"),
            "--output",
            str(output),
        ],
        check=True,
        cwd=root,
    )
    os.environ["XRDANALYSIS_DIRECT_MONTE_CARLO_METAL_LIBRARY"] = str(output)
    metal_mc._load_metal_library.cache_clear()
    assert metal_mc.metal_backend_available()
    return output


def _native_plan() -> NativeDirectMonteCarloPlan:
    return NativeDirectMonteCarloPlan(
        image_shape=(1, 4),
        csc_indptr=np.array([0, 1, 2, 3, 4]),
        csc_indices=np.array([0, 0, 1, 1]),
        csc_weights=np.ones(4),
        normalization_denominators=np.array([2.0, 2.0]),
        q_grid=np.array([1.0, 2.0]),
        q_normalization_band=(1.0, 1.0),
    )


def _plan() -> metal_mc.NativeMetalMonteCarloPlan:
    return metal_mc.prepare_metal_plan(_native_plan())


def _image() -> np.ndarray:
    return np.array([[100.0, 120.0, 80.0, 90.0]])


def test_plan_conversion_preserves_bin_major_contract():
    plan = _plan()
    np.testing.assert_array_equal(plan.csr_indptr, np.array([0, 2, 4]))
    np.testing.assert_array_equal(plan.csr_indices, np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(plan.csr_weights, np.ones(4))


def test_invalid_request_is_rejected_before_library_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        metal_mc,
        "_load_metal_library",
        lambda: pytest.fail("Metal library must not load"),
    )
    with pytest.raises(ValueError, match="draws must be a positive integer"):
        metal_mc.direct_detector_monte_carlo_metal(
            _plan(),
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
        ({"profile_batch_size": 0}, "profile_batch_size must be a positive"),
    ],
)
def test_validation_precedes_metal_loading(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    message: str,
):
    monkeypatch.setattr(
        metal_mc,
        "_load_metal_library",
        lambda: pytest.fail("Metal library must not load"),
    )
    parameters = {
        "scales": (1.0,),
        "seed": 0,
        "device": 0,
        "profile_batch_size": 4096,
    }
    parameters.update(kwargs)
    with pytest.raises(ValueError, match=message):
        metal_mc.direct_detector_monte_carlo_metal(
            _plan(),
            _image(),
            parameters.pop("scales"),
            10,
            **parameters,
        )


def test_large_implicit_host_output_is_rejected_before_library_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        metal_mc,
        "_load_metal_library",
        lambda: pytest.fail("Metal library must not load"),
    )
    with pytest.raises(ValueError, match="process patient-sized batches"):
        metal_mc.direct_detector_monte_carlo_metal(
            _plan(),
            _image(),
            (1.0,),
            70_000_000,
        )


def test_metal_device_and_deterministic_integration(metal_library: Path):
    assert metal_library.is_file()
    assert metal_mc.metal_device_count() >= 1
    actual = metal_mc.integrate_detector_frames_metal(_plan(), _image())
    expected = np.array([[1.0, (80.0 + 90.0) / (100.0 + 120.0)]])
    np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=3e-7)


def test_metal_random_stream_is_batch_invariant(metal_library: Path):
    plan = _plan()
    first = plan.run(
        _image(),
        (0.5, 1.0),
        100,
        seed=29,
        profile_batch_size=7,
    )
    second = plan.run(
        _image(),
        (0.5, 1.0),
        100,
        seed=29,
        profile_batch_size=64,
    )
    assert first.shape == (2, 100, 1, 2)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[..., 0], 1.0, rtol=0.0, atol=1e-7)


def test_metal_centered_poisson_matches_reference_statistics(metal_library: Path):
    draws = 20_000
    actual = _plan().run(
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


def test_metal_integration_matches_warmed_pyfai_plan(metal_library: Path):
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

    native = prepare_native_plan(
        integrator,
        image.shape,
        normalization_denominators=result.sum_normalization,
        q_grid=q_grid,
        q_normalization_band=band,
    )
    actual = metal_mc.integrate_detector_frames_metal(
        metal_mc.prepare_metal_plan(native),
        image,
    )[0]
    expected = np.asarray(result.intensity, dtype=float)
    selected = (q_grid >= band[0]) & (q_grid <= band[1])
    expected /= np.median(expected[selected])
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


def test_persistent_session_matches_one_shot_and_reuses_random_stream(
    metal_library: Path,
):
    plan = _plan()
    images = np.array(
        [
            [[100.0, 120.0, 80.0, 90.0]],
            [[95.0, 125.0, 85.0, 88.0]],
        ]
    )
    measurement_seeds = (11, 22)
    expected = metal_mc.direct_detector_monte_carlo_metal(
        plan,
        images,
        (0.5, 1.0),
        100,
        seed=29,
        measurement_seeds=measurement_seeds,
        profile_batch_size=7,
    )
    with PersistentMetalMonteCarlo(
        plan,
        images,
        measurement_seeds=measurement_seeds,
        profile_batch_size=7,
    ) as session:
        first = session.run((0.5, 1.0), 100, seed=29)
        second = session.run((0.5, 1.0), 100, seed=29)
        integrated = session.integrate()
        assert not session.closed
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    np.testing.assert_allclose(
        integrated,
        np.array([[1.0, 170.0 / 220.0], [1.0, 173.0 / 220.0]]),
        rtol=3e-7,
        atol=3e-7,
    )
    assert session.closed
    with pytest.raises(RuntimeError, match="session is closed"):
        session.run((1.0,), 1)


def test_persistent_session_enforces_scale_capacity(metal_library: Path):
    with PersistentMetalMonteCarlo(
        _plan(),
        _image(),
        scale_capacity=1,
    ) as session:
        with pytest.raises(ValueError, match="scale_capacity"):
            session.run((0.5, 1.0), 10)


def test_persistent_session_preserves_memmap_output(
    metal_library: Path,
    tmp_path: Path,
):
    output = np.memmap(
        tmp_path / "profiles.mmap",
        mode="w+",
        dtype=np.float64,
        shape=(1, 10, 1, 2),
    )
    with PersistentMetalMonteCarlo(_plan(), _image()) as session:
        returned = session.run((1.0,), 10, seed=37, output=output)
    assert returned is output
    output.flush()
    assert np.isfinite(output).all()


def test_grouped_session_combines_identical_plans_without_stream_changes(
    metal_library: Path,
):
    plan = _plan()
    images = [
        np.array([[100.0, 120.0, 80.0, 90.0]]),
        np.array([[95.0, 125.0, 85.0, 88.0]]),
    ]
    seeds = (11, 22)
    expected = metal_mc.direct_detector_monte_carlo_metal(
        plan,
        np.stack(images),
        (0.5, 1.0),
        100,
        seed=29,
        measurement_seeds=seeds,
        profile_batch_size=7,
    )
    with GroupedPersistentMetalMonteCarlo(
        [plan, plan],
        images,
        measurement_seeds=seeds,
        profile_batch_size=7,
    ) as grouped:
        assert grouped.group_count == 1
        actual = grouped.run((0.5, 1.0), 100, seed=29)
    np.testing.assert_array_equal(actual, expected)


def test_grouped_session_preserves_order_across_distinct_plans(
    metal_library: Path,
):
    first_plan = _plan()
    second_native = _native_plan()
    second_native = NativeDirectMonteCarloPlan(
        image_shape=second_native.image_shape,
        csc_indptr=second_native.csc_indptr,
        csc_indices=second_native.csc_indices,
        csc_weights=np.array([1.0, 1.0, 0.5, 1.5]),
        normalization_denominators=second_native.normalization_denominators,
        q_grid=second_native.q_grid,
        q_normalization_band=second_native.q_normalization_band,
    )
    second_plan = metal_mc.prepare_metal_plan(second_native)
    assert metal_plan_fingerprint(first_plan) != metal_plan_fingerprint(second_plan)
    images = [_image(), np.array([[95.0, 125.0, 85.0, 88.0]])]
    seeds = (11, 22)
    expected = np.stack(
        [
            metal_mc.direct_detector_monte_carlo_metal(
                plan,
                image,
                (1.0,),
                50,
                seed=31,
                measurement_seeds=(measurement_seed,),
                profile_batch_size=9,
            )[0, :, 0]
            for plan, image, measurement_seed in zip(
                (first_plan, second_plan),
                images,
                seeds,
                strict=True,
            )
        ],
        axis=1,
    )
    with GroupedPersistentMetalMonteCarlo(
        [first_plan, second_plan],
        images,
        measurement_seeds=seeds,
        profile_batch_size=9,
    ) as grouped:
        assert grouped.group_count == 2
        actual = grouped.run((1.0,), 50, seed=31)
    np.testing.assert_array_equal(actual[0], expected)


def test_prepared_geometry_executor_uses_external_plans_and_stable_seed(
    metal_library: Path,
):
    first_plan = _plan()
    second_native = _native_plan()
    second_plan = metal_mc.prepare_metal_plan(
        NativeDirectMonteCarloPlan(
            image_shape=second_native.image_shape,
            csc_indptr=second_native.csc_indptr,
            csc_indices=second_native.csc_indices,
            csc_weights=np.array([1.0, 1.0, 0.5, 1.5]),
            normalization_denominators=second_native.normalization_denominators,
            q_grid=second_native.q_grid,
            q_normalization_band=second_native.q_normalization_band,
        )
    )
    images = [_image(), np.array([[95.0, 125.0, 85.0, 88.0]])]
    executor = PreparedGeometryMetalMonteCarlo(
        images,
        measurement_seeds=(11, 22),
        profile_batch_size=3,
    )

    first = executor.run_geometry(
        (first_plan, second_plan),
        7,
        seed=31,
        include_deterministic=True,
    )
    repeated = executor.run_geometry(
        (first_plan, second_plan),
        7,
        seed=31,
        include_deterministic=True,
    )
    changed_seed = executor.run_geometry(
        (first_plan, second_plan),
        7,
        seed=32,
    )

    assert first.profiles.shape == (7, 2, 2)
    assert first.unique_plan_count == 2
    np.testing.assert_array_equal(first.profiles, repeated.profiles)
    np.testing.assert_array_equal(
        first.deterministic_profiles,
        repeated.deterministic_profiles,
    )
    assert not np.array_equal(first.profiles, changed_seed.profiles)


def _geometry_case():
    pytest.importorskip("pyFAI")
    from pyFAI.detectors import Detector

    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except ImportError:
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    detector = Detector(1e-4, 1e-4, max_shape=(32, 32), orientation=3)
    integrator = AzimuthalIntegrator(detector=detector)
    integrator.setFit2D(100.0, 16.0, 16.0, wavelength=1.54)
    row, column = np.indices((32, 32), dtype=float)
    image = 100.0 + 0.25 * row + 0.5 * column + 20.0 * np.exp(
        -((row - 18.0) ** 2 + (column - 11.0) ** 2) / 20.0
    )
    mask = np.zeros_like(image, dtype=np.int8)
    mask[:2, :] = 1
    mask[:, -2:] = 1
    result = integrator.integrate1d(
        image,
        32,
        mask=mask,
        error_model="poisson",
        method=("bbox", "csr", "cython"),
        unit="q_nm^-1",
        correctSolidAngle=True,
    )
    q_grid = np.asarray(result.radial, dtype=float)
    band = (float(q_grid[12]), float(q_grid[18]))
    from xrdanalysis.direct_monte_carlo import prepare_native_plan

    native = prepare_native_plan(
        integrator,
        image.shape,
        normalization_denominators=result.sum_normalization,
        q_grid=q_grid,
        q_normalization_band=band,
    )
    static_plan = metal_mc.prepare_metal_plan(native)
    geometry = MetalDetectorGeometry.from_pyfai(integrator)
    return integrator, image, mask, q_grid, band, static_plan, geometry


def _frame_masked_prepared_geometry_case():
    pytest.importorskip("pyFAI")
    from pyFAI.detectors import Detector

    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except ImportError:
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    def make_integrator():
        detector = Detector(1e-4, 1e-4, max_shape=(32, 32), orientation=3)
        integrator = AzimuthalIntegrator(detector=detector)
        integrator.setFit2D(100.0, 16.0, 16.0, wavelength=1.54)
        return integrator

    row, column = np.indices((32, 32), dtype=float)
    first_image = 100.0 + 0.25 * row + 0.5 * column + 20.0 * np.exp(
        -((row - 18.0) ** 2 + (column - 11.0) ** 2) / 20.0
    )
    images = np.stack((first_image, first_image * 1.03 + 2.0))
    masks = np.zeros(images.shape, dtype=np.uint8)
    masks[0, :2, :] = 1
    masks[0, :, -2:] = 1
    masks[1, 5:8, 4:12] = 1
    masks[1, 20:23, 20:28] = 1

    unmasked_integrator = make_integrator()
    unmasked_result = unmasked_integrator.integrate1d(
        first_image,
        32,
        error_model="poisson",
        method=("bbox", "csr", "cython"),
        unit="q_nm^-1",
        correctSolidAngle=True,
    )
    q_grid = np.asarray(unmasked_result.radial, dtype=float)
    band = (float(q_grid[12]), float(q_grid[18]))
    from xrdanalysis.direct_monte_carlo import prepare_native_plan

    plan = metal_mc.prepare_metal_plan(
        prepare_native_plan(
            unmasked_integrator,
            first_image.shape,
            normalization_denominators=unmasked_result.sum_normalization,
            q_grid=q_grid,
            q_normalization_band=band,
        )
    )
    pixel_normalization = unmasked_integrator.solidAngleArray(first_image.shape)
    return (
        make_integrator,
        images,
        masks,
        q_grid,
        band,
        plan,
        pixel_normalization,
    )


def _masked_pyfai_plan(integrator, image, mask, band):
    result = integrator.integrate1d(
        image,
        32,
        mask=mask,
        error_model="poisson",
        method=("bbox", "csr", "cython"),
        unit="q_nm^-1",
        correctSolidAngle=True,
    )
    from xrdanalysis.direct_monte_carlo import prepare_native_plan

    plan = metal_mc.prepare_metal_plan(
        prepare_native_plan(
            integrator,
            image.shape,
            normalization_denominators=result.sum_normalization,
            q_grid=result.radial,
            q_normalization_band=(
                float(result.radial[12]),
                float(result.radial[18]),
            ),
        )
    )
    return result, plan


def test_frame_masked_prepared_geometry_matches_pyfai_bbox_deterministically(
    metal_library: Path,
):
    assert metal_library.is_file()
    (
        make_integrator,
        images,
        masks,
        _,
        band,
        plan,
        pixel_normalization,
    ) = _frame_masked_prepared_geometry_case()
    expected_profiles = []
    expected_denominators = []
    for image, mask in zip(images, masks, strict=True):
        result, _ = _masked_pyfai_plan(make_integrator(), image, mask, band)
        expected_profiles.append(_normalized_profile(result, band))
        expected_denominators.append(result.sum_normalization)

    with FrameMaskedPreparedGeometryMetalMonteCarlo(
        plan,
        images,
        masks,
        pixel_normalization=pixel_normalization,
        measurement_seeds=(11, 22),
        profile_batch_size=3,
    ) as session:
        actual = session.integrate()
        actual_denominators = session.normalization_denominators

    np.testing.assert_allclose(
        actual,
        np.asarray(expected_profiles),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        actual_denominators,
        np.asarray(expected_denominators),
        rtol=1e-7,
        atol=1e-6,
    )


def test_frame_masked_prepared_geometry_preserves_photon_draw_stream(
    metal_library: Path,
):
    assert metal_library.is_file()
    (
        make_integrator,
        images,
        masks,
        _,
        band,
        plan,
        pixel_normalization,
    ) = _frame_masked_prepared_geometry_case()
    measurement_seeds = (11, 22)
    with FrameMaskedPreparedGeometryMetalMonteCarlo(
        plan,
        images,
        masks,
        pixel_normalization=pixel_normalization,
        measurement_seeds=measurement_seeds,
        scale_capacity=2,
        profile_batch_size=3,
    ) as session:
        actual = session.run((0.5, 1.0), 17, seed=31)

    expected_measurements = []
    for image, mask, measurement_seed in zip(
        images,
        masks,
        measurement_seeds,
        strict=True,
    ):
        _, masked_plan = _masked_pyfai_plan(
            make_integrator(),
            image,
            mask,
            band,
        )
        expected_measurements.append(
            metal_mc.direct_detector_monte_carlo_metal(
                masked_plan,
                image,
                (0.5, 1.0),
                17,
                seed=31,
                measurement_seeds=(measurement_seed,),
                profile_batch_size=3,
            )[:, :, 0]
        )
    expected = np.stack(expected_measurements, axis=2)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-7)


def test_frame_masked_prepared_geometry_rejects_wrong_correction_before_metal(
    monkeypatch: pytest.MonkeyPatch,
):
    (
        _,
        images,
        masks,
        _,
        _,
        plan,
        _,
    ) = _frame_masked_prepared_geometry_case()
    monkeypatch.setattr(
        metal_mc,
        "_load_metal_library",
        lambda: pytest.fail("Metal library must not load"),
    )
    with pytest.raises(ValueError, match="does not reproduce the unmasked pyFAI"):
        FrameMaskedPreparedGeometryMetalMonteCarlo(
            plan,
            images,
            masks,
            pixel_normalization=np.full(plan.image_shape, 2.0),
        )


def test_frame_masked_prepared_geometry_allows_masked_nonfinite_pixels(
    metal_library: Path,
):
    assert metal_library.is_file()
    (
        _,
        images,
        masks,
        _,
        _,
        plan,
        pixel_normalization,
    ) = _frame_masked_prepared_geometry_case()
    images = images.copy()
    images[0, 0, 0] = np.nan
    images[0, 0, 1] = np.inf
    masks = masks.copy()
    masks[0, 0, :2] = 1
    with FrameMaskedPreparedGeometryMetalMonteCarlo(
        plan,
        images,
        masks,
        pixel_normalization=pixel_normalization,
    ) as session:
        actual = session.integrate()
    assert np.isfinite(actual).all()


def _normalized_profile(result, band: tuple[float, float]) -> np.ndarray:
    q_grid = np.asarray(result.radial, dtype=float)
    profile = np.asarray(result.intensity, dtype=float)
    lower = int(np.argmin(np.abs(q_grid - band[0])))
    upper = int(np.argmin(np.abs(q_grid - band[1])))
    return profile / np.median(profile[lower : upper + 1])


def test_geometry_contract_rejects_nonzero_poni_rotation():
    with pytest.raises(ValueError, match="exactly zero PONI rotations"):
        MetalDetectorGeometry(
            distance_m=0.1,
            poni1_m=0.001,
            poni2_m=0.001,
            pixel1_m=1e-4,
            pixel2_m=1e-4,
            wavelength_m=1.54e-10,
            rot1_rad=1e-12,
        )


def test_geometry_contract_rejects_nonuniform_q_grid_before_metal_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        metal_mc,
        "_load_metal_library",
        lambda: pytest.fail("Metal library must not load"),
    )
    image = np.ones((1, 4, 4), dtype=float)
    geometry = MetalDetectorGeometry(
        distance_m=0.1,
        poni1_m=0.001,
        poni2_m=0.001,
        pixel1_m=1e-4,
        pixel2_m=1e-4,
        wavelength_m=1.54e-10,
    )
    with pytest.raises(ValueError, match="uniform q_grid"):
        GeometryAwareMetalMonteCarlo(
            image,
            None,
            [geometry],
            [1.0, 2.0, 4.0],
            (1.0, 2.0),
        )


def test_geometry_contract_allows_masked_nonfinite_pixels(
    metal_library: Path,
):
    assert metal_library.is_file()
    _, image, mask, q_grid, band, _, geometry = _geometry_case()
    image = image.copy()
    image[0, 0] = np.nan
    image[0, 1] = np.inf
    mask = mask.copy()
    mask[0, :2] = 1
    with GeometryAwareMetalMonteCarlo(
        image,
        mask,
        [geometry],
        q_grid,
        band,
    ) as session:
        actual = session.integrate()
    assert np.isfinite(actual).all()


def test_geometry_contract_rejects_unmasked_nonfinite_pixels(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        metal_mc,
        "_load_metal_library",
        lambda: pytest.fail("Metal library must not load"),
    )
    image = np.ones((4, 4), dtype=float)
    image[2, 2] = np.nan
    geometry = MetalDetectorGeometry(
        distance_m=0.1,
        poni1_m=0.001,
        poni2_m=0.001,
        pixel1_m=1e-4,
        pixel2_m=1e-4,
        wavelength_m=1.54e-10,
    )
    with pytest.raises(ValueError, match="unmasked detector pixels"):
        GeometryAwareMetalMonteCarlo(
            image,
            None,
            [geometry],
            [1.0, 2.0, 3.0],
            (1.0, 2.0),
        )


def test_geometry_zero_perturbation_matches_static_bbox_metal(
    metal_library: Path,
):
    assert metal_library.is_file()
    _, image, mask, q_grid, band, static_plan, geometry = _geometry_case()
    expected = metal_mc.integrate_detector_frames_metal(static_plan, image)[0]
    expected_random = static_plan.run(
        image,
        (0.5, 1.0),
        7,
        seed=37,
        profile_batch_size=2,
    )
    with GeometryAwareMetalMonteCarlo(
        image,
        mask,
        [geometry],
        q_grid,
        band,
        scale_capacity=2,
        draw_capacity=7,
        profile_batch_size=2,
    ) as session:
        actual = session.integrate()[0, 0]
        actual_random = session.run((0.5, 1.0), 7, seed=37)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(
        actual_random,
        expected_random,
        rtol=5e-4,
        atol=5e-4,
    )


def test_geometry_centered_poisson_matches_static_metal_statistics(
    metal_library: Path,
):
    assert metal_library.is_file()
    _, image, mask, q_grid, band, static_plan, geometry = _geometry_case()
    draws = 20_000
    expected = static_plan.run(
        image,
        (1.0,),
        draws,
        seed=71,
        profile_batch_size=16,
    )[0, :, 0]
    with GeometryAwareMetalMonteCarlo(
        image,
        mask,
        [geometry],
        q_grid,
        band,
        scale_capacity=1,
        draw_capacity=32,
        profile_batch_size=16,
    ) as session:
        actual = session.run(
            (1.0,),
            draws,
            seed=71,
            draw_chunk_size=32,
        )[0, :, 0]

    expected_scalar = np.mean(expected, axis=1)
    actual_scalar = np.mean(actual, axis=1)
    standard_error = np.sqrt(
        np.var(actual_scalar, ddof=1) / draws
        + np.var(expected_scalar, ddof=1) / draws
    )
    assert abs(np.mean(actual_scalar) - np.mean(expected_scalar)) <= (
        5.0 * standard_error
    )
    variance_ratio = np.var(actual_scalar, ddof=1) / np.var(
        expected_scalar, ddof=1
    )
    assert 0.90 <= variance_ratio <= 1.10


def test_geometry_random_perturbations_match_direct_pyfai_oracle(
    metal_library: Path,
):
    assert metal_library.is_file()
    integrator, image, mask, q_grid, band, _, geometry = _geometry_case()
    draws = 5
    rng = np.random.default_rng(91)
    distance = geometry.distance_m + rng.uniform(-0.003, 0.003, size=(draws, 1))
    poni1 = geometry.poni1_m + rng.uniform(-3e-4, 3e-4, size=(draws, 1))
    poni2 = geometry.poni2_m + rng.uniform(-3e-4, 3e-4, size=(draws, 1))
    with GeometryAwareMetalMonteCarlo(
        image,
        mask,
        [geometry],
        q_grid,
        band,
        draw_capacity=3,
        profile_batch_size=2,
    ) as session:
        actual = session.integrate(
            draws,
            effective_distance_m=distance,
            poni1_m=poni1,
            poni2_m=poni2,
            draw_chunk_size=2,
        )[:, 0]

    from pyFAI.detectors import Detector

    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except ImportError:
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

    q_delta = float(np.mean(np.diff(q_grid)))
    radial_range = (
        float(q_grid[0] - 0.5 * q_delta),
        float(q_grid[-1] + 0.5 * q_delta),
    )
    expected = []
    for draw in range(draws):
        detector = Detector(
            geometry.pixel1_m,
            geometry.pixel2_m,
            max_shape=image.shape,
            orientation=geometry.orientation,
        )
        oracle = AzimuthalIntegrator(
            dist=float(distance[draw, 0]),
            poni1=float(poni1[draw, 0]),
            poni2=float(poni2[draw, 0]),
            rot1=0.0,
            rot2=0.0,
            rot3=0.0,
            detector=detector,
            wavelength=integrator.wavelength,
        )
        result = oracle.integrate1d(
            image,
            q_grid.size,
            radial_range=radial_range,
            mask=mask,
            error_model="poisson",
            method=("bbox", "csr", "cython"),
            unit="q_nm^-1",
            correctSolidAngle=True,
        )
        expected.append(_normalized_profile(result, band))
    np.testing.assert_allclose(
        actual,
        np.asarray(expected),
        rtol=5e-4,
        atol=5e-4,
    )


def test_geometry_rng_is_invariant_to_chunking_and_draw_offset(
    metal_library: Path,
):
    assert metal_library.is_file()
    _, image, mask, q_grid, band, _, geometry = _geometry_case()
    draws = 9
    rng = np.random.default_rng(117)
    distance = geometry.distance_m + rng.uniform(-0.001, 0.001, size=(draws, 1))
    poni1 = geometry.poni1_m + rng.uniform(-1e-4, 1e-4, size=(draws, 1))
    poni2 = geometry.poni2_m + rng.uniform(-1e-4, 1e-4, size=(draws, 1))
    with GeometryAwareMetalMonteCarlo(
        image,
        mask,
        [geometry],
        q_grid,
        band,
        scale_capacity=2,
        draw_capacity=9,
        profile_batch_size=3,
    ) as session:
        expected = session.run(
            (0.5, 1.0),
            draws,
            effective_distance_m=distance,
            poni1_m=poni1,
            poni2_m=poni2,
            seed=43,
            draw_offset=13,
            draw_chunk_size=9,
        )
        chunked = session.run(
            (0.5, 1.0),
            draws,
            effective_distance_m=distance,
            poni1_m=poni1,
            poni2_m=poni2,
            seed=43,
            draw_offset=13,
            draw_chunk_size=2,
        )
        first = session.run(
            (0.5, 1.0),
            4,
            effective_distance_m=distance[:4],
            poni1_m=poni1[:4],
            poni2_m=poni2[:4],
            seed=43,
            draw_offset=13,
        )
        second = session.run(
            (0.5, 1.0),
            draws - 4,
            effective_distance_m=distance[4:],
            poni1_m=poni1[4:],
            poni2_m=poni2[4:],
            seed=43,
            draw_offset=17,
        )
    np.testing.assert_allclose(chunked, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        np.concatenate((first, second), axis=1),
        expected,
        rtol=2e-6,
        atol=2e-6,
    )


def test_nested_geometry_reuses_geometry_without_changing_photon_stream(
    metal_library: Path,
):
    assert metal_library.is_file()
    _, image, mask, q_grid, band, _, geometry = _geometry_case()
    geometry_draws = 3
    photon_replicates = 4
    rng = np.random.default_rng(219)
    distance = geometry.distance_m + rng.uniform(
        -0.001,
        0.001,
        size=(geometry_draws, 1),
    )
    poni1 = geometry.poni1_m + rng.uniform(
        -1e-4,
        1e-4,
        size=(geometry_draws, 1),
    )
    poni2 = geometry.poni2_m + rng.uniform(
        -1e-4,
        1e-4,
        size=(geometry_draws, 1),
    )
    repeated_distance = np.repeat(distance, photon_replicates, axis=0)
    repeated_poni1 = np.repeat(poni1, photon_replicates, axis=0)
    repeated_poni2 = np.repeat(poni2, photon_replicates, axis=0)
    with GeometryAwareMetalMonteCarlo(
        image,
        mask,
        [geometry],
        q_grid,
        band,
        scale_capacity=2,
        draw_capacity=12,
        profile_batch_size=16,
    ) as session:
        expected = session.run(
            (0.5, 1.0),
            geometry_draws * photon_replicates,
            effective_distance_m=repeated_distance,
            poni1_m=repeated_poni1,
            poni2_m=repeated_poni2,
            seed=83,
            draw_offset=17,
        ).reshape(2, geometry_draws, photon_replicates, 1, q_grid.size)
        actual = session.run_nested(
            (0.5, 1.0),
            geometry_draws,
            photon_replicates,
            effective_distance_m=distance,
            poni1_m=poni1,
            poni2_m=poni2,
            seed=83,
            photon_draw_offset=17,
        )
        chunked = session.run_nested(
            (0.5, 1.0),
            geometry_draws,
            photon_replicates,
            effective_distance_m=distance,
            poni1_m=poni1,
            poni2_m=poni2,
            seed=83,
            photon_draw_offset=17,
            geometry_chunk_size=1,
        )
    assert actual.shape == (2, geometry_draws, photon_replicates, 1, q_grid.size)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(chunked, actual, rtol=2e-6, atol=2e-6)
