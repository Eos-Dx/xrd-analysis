"""Executable import and serialization contracts for supported consumers.

These tests deliberately protect import paths used by DiFRA and notebook/model
consumers.  They are structural compatibility gates; numerical azimuthal
behavior belongs in the dedicated azimuthal-integration tests.
"""

from __future__ import annotations

import importlib
import inspect

import joblib
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


def test_difra_facing_function_signatures_are_stable():
    """Protect the exact callable surface imported by DiFRA's compatibility layer."""
    assert list(inspect.signature(initialize_azimuthal_integrator_df).parameters) == [
        "pixel_size",
        "center_column",
        "center_row",
        "wavelength",
        "sample_distance_mm",
    ]
    assert list(
        inspect.signature(initialize_azimuthal_integrator_poni_text).parameters
    ) == ["ponifile_text"]
    assert list(inspect.signature(create_mask).parameters) == ["faulty_pixels", "size"]
    assert list(inspect.signature(FaultyPixelDetector).parameters) == [
        "region_size",
        "outlier_n_std",
        "zero_frac_threshold",
        "temporal_consistency",
        "exclude_beam_center_radius",
        "poni_column",
        "debug",
    ]


def test_perform_azimuthal_integration_signature_is_stable():
    """Protect call ordering and optional controls used by integration consumers."""
    assert list(inspect.signature(perform_azimuthal_integration).parameters) == [
        "row",
        "column",
        "npt",
        "mask",
        "mode",
        "calibration_mode",
        "thres",
        "max_iter",
        "thickness_adjustment",
        "thickness_adjustment_distance",
        "thickness_reference_mm",
        "sample_thickness_column",
        "sample_thickness_mm",
        "calc_cake_stats",
        "angles",
        "poni_dir",
        "error_model",
    ]


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
def test_joblib_round_trip_preserves_qualified_class_path(
    tmp_path, instance, qualified_name
):
    """Moving these classes breaks existing joblib artifacts, so pin their paths."""
    path = tmp_path / "compatibility.joblib"
    joblib.dump(instance, path)
    restored = joblib.load(path)

    assert type(restored) is type(instance)
    assert (
        f"{type(restored).__module__}.{type(restored).__qualname__}" == qualified_name
    )


def test_legacy_transformers_mcr_reexports_stay_available():
    """Older consumers import MCR/SVD from transformers rather than canonical module."""
    transformers = importlib.import_module("xrdanalysis.data_processing.transformers")
    assert transformers.MCRALSTransformer is MCRALSTransformer
    assert transformers.SpectroSVDTransformer is SpectroSVDTransformer
