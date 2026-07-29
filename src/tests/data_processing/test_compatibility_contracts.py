"""Executable import and serialization contracts for supported consumers.

These tests deliberately protect import paths used by DiFRA and notebook/model
consumers.  They are structural compatibility gates; numerical azimuthal
behavior belongs in the dedicated azimuthal-integration tests.
"""

from __future__ import annotations

import importlib
import inspect

import joblib
import numpy as np
import pandas as pd
import pytest

from xrdanalysis.data_processing import (
    DataFrameCleaningPipeline,
    FaultyPixelDetector,
    MCRALSTransformer,
    MeasurementTypeClassifier,
    SpectroSVDTransformer,
)
from xrdanalysis.data_processing.azimuthal_integration import (
    initialize_azimuthal_integrator_df,
    initialize_azimuthal_integrator_poni_text,
    perform_azimuthal_integration,
)
from xrdanalysis.data_processing.pipeline import MLPipeline, MLPipelineMulti
from xrdanalysis.data_processing.transformers import (
    AzimuthalIntegration,
    DetectorJoiner,
    SNRTransformer,
)
from xrdanalysis.data_processing.utility_functions import create_mask


@pytest.mark.parametrize(
    ("module_name", "symbol_names"),
    [
        (
            "xrdanalysis.data_processing.azimuthal_integration",
            (
                "initialize_azimuthal_integrator_df",
                "initialize_azimuthal_integrator_poni",
                "initialize_azimuthal_integrator_poni_text",
                "perform_azimuthal_integration",
            ),
        ),
        (
            "xrdanalysis.data_processing.faulty_pixel_detection",
            ("FaultyPixelDetector",),
        ),
        ("xrdanalysis.data_processing.utility_functions", ("create_mask", "h5_to_df")),
        (
            "xrdanalysis.data_processing.transformers",
            ("AzimuthalIntegration", "DetectorJoiner", "SNRTransformer"),
        ),
        ("xrdanalysis.data_processing.pipeline", ("MLPipeline", "MLPipelineMulti")),
        (
            "xrdanalysis.data_processing.spectrokinetic_transformers",
            ("MCRALSTransformer", "SpectroSVDTransformer"),
        ),
        (
            "xrdanalysis.data_processing.detector_joining",
            ("join_detectors", "average_ignore_zeros"),
        ),
        ("xrdanalysis.data_processing.containers", ("RuleQ",)),
    ],
)
def test_supported_direct_import_paths(module_name, symbol_names):
    """Keep direct paths used by DiFRA, notebooks, and saved models importable."""
    module = importlib.import_module(module_name)
    for symbol_name in symbol_names:
        assert hasattr(module, symbol_name), f"{module_name}.{symbol_name} is missing"


def test_data_processing_package_facade_keeps_supported_symbols():
    """Package-level imports are an existing public API, not implementation detail."""
    package = importlib.import_module("xrdanalysis.data_processing")
    expected = {
        "DataFrameCleaningPipeline": DataFrameCleaningPipeline,
        "FaultyPixelDetector": FaultyPixelDetector,
        "MeasurementTypeClassifier": MeasurementTypeClassifier,
        "MCRALSTransformer": MCRALSTransformer,
        "SpectroSVDTransformer": SpectroSVDTransformer,
    }
    for name, symbol in expected.items():
        assert getattr(package, name) is symbol


def test_spectrokinetic_canonical_public_api_and_signatures_are_exact():
    """Keep canonical numerical API names and call forms stable for consumers."""
    module = importlib.import_module(
        "xrdanalysis.data_processing.spectrokinetic_transformers"
    )
    empty = inspect.Parameter.empty
    expected = {
        "ALSConfig": (
            (
                "n_components",
                "n_start",
                "maxiter",
                "thresh",
                "init_method",
                "opt_s_first",
                "nonneg_c",
                "nonneg_s",
                "uni_s",
                "norm_s",
                "sum_norm",
                "norm_mode",
                "smooth",
                "close_c",
                "w_close_c",
                "hard_s0",
                "w_hard_s0",
                "broadening",
                "broadening_max_pct",
                "correction_spectra",
                "correction_lambda",
                "random_state",
            ),
            (
                2,
                None,
                100,
                1e-3,
                "svd",
                True,
                True,
                True,
                False,
                True,
                False,
                "intensity",
                0.0,
                False,
                0.0,
                True,
                1.0,
                False,
                10.0,
                False,
                0.0,
                42,
            ),
        ),
        "ALSResult": (
            (
                "c",
                "s",
                "model",
                "resid",
                "rss",
                "iter",
                "lof",
                "converged",
                "msg",
                "g",
                "broadening_vec",
                "n_fixed",
                "lambda_corr",
                "meta",
            ),
            (
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                empty,
                None,
                None,
                0,
                0.0,
                "<factory>",
            ),
        ),
        "SpectroSVDTransformer": (
            (
                "matrix_col",
                "delay_col",
                "wavelength_col",
                "delay_mask_col",
                "wavelength_mask_col",
                "max_rank",
                "model_rank",
            ),
            ("spectro_matrix", "delay_axis", "wavelength_axis", None, None, 10, None),
        ),
        "MCRALSTransformer": (
            (
                "matrix_col",
                "delay_col",
                "wavelength_col",
                "delay_mask_col",
                "wavelength_mask_col",
                "decomposition_mode",
                "group_col",
                "group_strategy",
                "allow_group_wavelength_interpolation",
                "n_components",
                "n_start",
                "init_method",
                "maxiter",
                "thresh",
                "opt_s_first",
                "nonneg_c",
                "nonneg_s",
                "uni_s",
                "norm_s",
                "sum_norm",
                "norm_mode",
                "smooth",
                "close_c",
                "w_close_c",
                "presence_mask_col",
                "fixed_spectra",
                "fixed_spectra_col",
                "fixed_wavelength_axis",
                "interpolate_fixed",
                "hard_s0",
                "w_hard_s0",
                "correction_spectra",
                "correction_lambda",
                "broadening",
                "broadening_max_pct",
                "random_state",
                "restart_result",
            ),
            (
                "spectro_matrix",
                "delay_axis",
                "wavelength_axis",
                None,
                None,
                "row",
                None,
                "tile_delay",
                False,
                2,
                None,
                "svd",
                100,
                1e-3,
                True,
                True,
                True,
                False,
                True,
                False,
                "intensity",
                0.0,
                False,
                0.0,
                None,
                None,
                None,
                None,
                False,
                True,
                1.0,
                False,
                0.0,
                False,
                10.0,
                42,
                None,
            ),
        ),
        "compute_lof": (("model", "data"), (empty, empty)),
        "convolve_spectrum": (("spectrum", "sigma"), (empty, empty)),
        "optimize_broadening_single": (
            ("data_row", "c_row", "s", "g_init", "sigma_max", "broadening_vec"),
            (empty, empty, empty, empty, None, None),
        ),
        "solve_C": (
            ("s", "data", "c", "nonneg_c", "null_c", "close_c", "w_close_c", "g"),
            (empty, empty, empty, True, None, False, 0.0, None),
        ),
        "solve_S": (
            (
                "c",
                "data",
                "s",
                "x_s",
                "nonneg_s",
                "uni_s",
                "s0",
                "norm_s",
                "smooth",
                "sum_s",
                "hard_s0",
                "w_hard_s0",
                "norm_mode",
                "g",
            ),
            (
                empty,
                empty,
                empty,
                empty,
                True,
                False,
                None,
                True,
                0.0,
                False,
                True,
                1.0,
                "intensity",
                None,
            ),
        ),
        "solve_C_coupled": (
            ("s", "data", "c", "nonneg_c", "null_c", "close_c", "w_close_c", "n_fixed"),
            (empty, empty, empty, True, None, False, 0.0, 0),
        ),
        "solve_S_coupled": (
            (
                "c",
                "data",
                "s",
                "x_s",
                "nonneg_s",
                "uni_s",
                "s0",
                "norm_s",
                "smooth",
                "sum_s",
                "hard_s0",
                "w_hard_s0",
                "n_fixed",
                "lambda_corr",
                "norm_mode",
            ),
            (
                empty,
                empty,
                empty,
                empty,
                True,
                False,
                None,
                True,
                0.0,
                False,
                True,
                1.0,
                0,
                0.0,
                "intensity",
            ),
        ),
        "run_als_iteration": (
            ("c", "psi", "s", "x_c", "x_s", "config", "null_c", "s0", "g", "n_fixed"),
            (empty, empty, empty, empty, empty, empty, None, None, None, 0),
        ),
        "enforce_unimodal": (("y",), (empty,)),
    }

    assert module.__all__ == list(expected)
    for name, (parameter_names, defaults) in expected.items():
        parameters = tuple(inspect.signature(getattr(module, name)).parameters.values())
        assert tuple(parameter.name for parameter in parameters) == parameter_names
        actual_defaults = tuple(
            "<factory>" if repr(parameter.default) == "<factory>" else parameter.default
            for parameter in parameters
        )
        assert actual_defaults == defaults
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters
        )


def test_difra_facing_function_signatures_are_stable():
    """Protect DiFRA call forms while permitting appended optional keywords."""
    df_signature = inspect.signature(initialize_azimuthal_integrator_df)
    assert list(df_signature.parameters)[:5] == [
        "pixel_size",
        "center_column",
        "center_row",
        "wavelength",
        "sample_distance_mm",
    ]
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in list(df_signature.parameters.values())[:5]
    )
    df_signature.bind(1e-4, 512, 256, 1.54, 1_000.0)

    poni_signature = inspect.signature(initialize_azimuthal_integrator_poni_text)
    assert list(poni_signature.parameters)[:1] == ["ponifile_text"]
    assert poni_signature.parameters["ponifile_text"].default is inspect.Parameter.empty
    poni_signature.bind("Distance: 0.1")

    mask_signature = inspect.signature(create_mask)
    assert list(mask_signature.parameters)[:2] == ["faulty_pixels", "size"]
    assert mask_signature.parameters["size"].default == (256, 256)
    mask_signature.bind([(3, 4)])
    mask_signature.bind([(3, 4)], size=(512, 512))

    detector_signature = inspect.signature(FaultyPixelDetector)
    for name in (
        "temporal_consistency",
        "exclude_beam_center_radius",
        "debug",
    ):
        assert name in detector_signature.parameters
    assert detector_signature.parameters["temporal_consistency"].default == 0.7
    assert detector_signature.parameters["exclude_beam_center_radius"].default is None
    assert detector_signature.parameters["debug"].default is False
    detector_signature.bind(
        temporal_consistency=0.0,
        exclude_beam_center_radius=0.15,
        debug=True,
    )


def test_perform_azimuthal_integration_signature_is_stable():
    """Protect established integration call forms while allowing new keywords."""
    signature = inspect.signature(perform_azimuthal_integration)
    assert list(signature.parameters)[:6] == [
        "row",
        "column",
        "npt",
        "mask",
        "mode",
        "calibration_mode",
    ]
    assert signature.parameters["row"].default is inspect.Parameter.empty
    assert signature.parameters["column"].default == "measurement_data"
    assert signature.parameters["npt"].default == 256
    assert signature.parameters["mode"].default == "1D"
    assert signature.parameters["calibration_mode"].default == "dataframe"
    for name in (
        "thickness_adjustment",
        "thickness_reference_mm",
        "sample_thickness_column",
        "sample_thickness_mm",
        "error_model",
    ):
        assert name in signature.parameters

    signature.bind(object())
    signature.bind(object(), "measurement_data", 256, None, "1D", "dataframe")
    signature.bind(
        object(),
        calibration_mode="poni",
        thickness_adjustment=True,
        thickness_reference_mm=11.0,
        sample_thickness_column="thickness",
        error_model="poisson",
    )


@pytest.mark.parametrize(
    ("instance", "qualified_name"),
    [
        (
            AzimuthalIntegration(),
            "xrdanalysis.data_processing.transformers.AzimuthalIntegration",
        ),
        (DetectorJoiner(), "xrdanalysis.data_processing.transformers.DetectorJoiner"),
        (SNRTransformer(), "xrdanalysis.data_processing.transformers.SNRTransformer"),
        (
            FaultyPixelDetector(),
            "xrdanalysis.data_processing.faulty_pixel_detection.FaultyPixelDetector",
        ),
        (
            SpectroSVDTransformer(),
            "xrdanalysis.data_processing.spectrokinetic_transformers."
            "SpectroSVDTransformer",
        ),
        (
            MCRALSTransformer(),
            "xrdanalysis.data_processing.spectrokinetic_transformers."
            "MCRALSTransformer",
        ),
        (MLPipeline(), "xrdanalysis.data_processing.pipeline.MLPipeline"),
        (MLPipelineMulti(), "xrdanalysis.data_processing.pipeline.MLPipelineMulti"),
    ],
)
def test_current_joblib_round_trip_smoke_preserves_qualified_class_path(
    tmp_path, instance, qualified_name
):
    """Smoke-test current serialization; historical artifact loading needs fixtures."""
    path = tmp_path / "compatibility.joblib"
    joblib.dump(instance, path)
    restored = joblib.load(path)

    assert type(restored) is type(instance)
    assert (
        f"{type(restored).__module__}.{type(restored).__qualname__}" == qualified_name
    )


def test_fitted_mcr_restart_state_survives_joblib_round_trip(tmp_path):
    delay = np.linspace(0.0, 1.0, 8)
    wav = np.linspace(300.0, 500.0, 12)
    c = np.column_stack(
        [np.linspace(0.2, 1.0, delay.size), np.linspace(1.0, 0.3, delay.size)]
    )
    s = np.column_stack(
        [np.linspace(0.1, 1.0, wav.size), np.linspace(1.0, 0.2, wav.size)]
    )
    frame = pd.DataFrame(
        {
            "spectro_matrix": [c @ s.T],
            "delay_axis": [delay],
            "wavelength_axis": [wav],
        }
    )
    base = MCRALSTransformer(n_components=2, maxiter=6, thresh=1e-5, random_state=11)
    base.fit_transform(frame)
    restart = MCRALSTransformer(
        n_components=2,
        init_method="restart",
        restart_result=base._last_result,
        maxiter=6,
        thresh=1e-5,
        random_state=11,
    ).fit(frame)

    path = tmp_path / "fitted-mcr-restart.joblib"
    joblib.dump(restart, path)
    restored = joblib.load(path)
    out = restored.transform(frame)

    assert isinstance(restored, MCRALSTransformer)
    assert restored._last_result is not None
    assert restored._last_result.c.shape == (delay.size, 2)
    assert out.iloc[0]["als_C"].shape == (delay.size, 2)
    assert out.iloc[0]["als_meta"]["init_method"] == "restart"


def test_legacy_transformers_mcr_reexports_stay_available():
    """Older consumers import MCR/SVD from transformers rather than canonical module."""
    transformers = importlib.import_module("xrdanalysis.data_processing.transformers")
    assert transformers.MCRALSTransformer is MCRALSTransformer
    assert transformers.SpectroSVDTransformer is SpectroSVDTransformer
