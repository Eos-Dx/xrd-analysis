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


def test_legacy_transformers_mcr_reexports_stay_available():
    """Older consumers import MCR/SVD from transformers rather than canonical module."""
    transformers = importlib.import_module("xrdanalysis.data_processing.transformers")
    assert transformers.MCRALSTransformer is MCRALSTransformer
    assert transformers.SpectroSVDTransformer is SpectroSVDTransformer
