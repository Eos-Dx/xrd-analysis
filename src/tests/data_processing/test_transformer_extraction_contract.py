"""Compatibility contracts for transformer implementations extracted privately."""

from __future__ import annotations

import importlib
import inspect

import joblib
import pytest
from sklearn.base import clone

from xrdanalysis.data_processing import transformers

_CANONICAL_MODULE = "xrdanalysis.data_processing.transformers"
_MOVED_SIGNATURES = {
    "ColumnStandardizer": "(column)",
    "RuleBasedProfileFilter": (
        "(rules, q_col: 'str' = 'q_range', data_col: 'str' = "
        "'radial_profile_data', type_col: 'str' = 'type_measurement', "
        "evaluation_mode: 'str' = 'nearest', window_half_width: 'float' = 0.0, "
        "keep_unruled: 'bool' = True, keep_column: 'str' = 'passes_rules', "
        "failed_rule_column: 'str' = 'failed_rule', details_column: 'str' = "
        "'rule_details', trace_column: 'str' = 'rule_trace', "
        "rules_checked_column: 'str' = 'rules_checked', drop_failures: 'bool' = "
        "False, reset_index: 'bool' = False)"
    ),
    "ColumnExtractor": "(columns)",
    "ColumnCleaner": "(rules: 'List[Rule]')",
    "QRangeSetter": "(limits: 'Limits' = None)",
    "SlopeRemoval": "(columns=['radial_profile_data'], mode='')",
    "FourierTransform": (
        "(fourier_mode='', order=15, columns=['radial_profile_data'], "
        "remove_beam='false', thresh=1000, padding=0, mask=None, "
        "filter_radius=None, features: 'Optional[Union[List[str], str]]' = None)"
    ),
    "GoodnessTransformer": (
        "(column: 'str' = 'polar_data', skip_bins: 'int' = 30, "
        "hf_cutoff_fraction: 'float' = 0.25, output_col: 'str' = 'goodness', "
        "save_dev: 'bool' = False, diff_col: 'str' = 'data_diff')"
    ),
    "GoodnessFilter": (
        "(goodness_column: 'str' = 'goodness', type_column: 'str' = "
        "'type_measurement', thresholds: 'dict' = None, rule: 'str' = '>', "
        "default_threshold: 'float' = 50.0, verbose: 'bool' = False)"
    ),
    "DataPreparation": (
        "(columns=['calibration_measurement_id', 'study_name', 'study_id', "
        "'cancer_tissue', 'cancer_diagnosis', 'patient_id', 'wavelength', "
        "'pixel_size', 'calibration_manual_distance', 'calculated_distance', "
        "'measurement_data', 'center', 'ponifile'])"
    ),
    "NormScaler": "(scalers: 'Dict[str, StandardScaler]' = None, name='Scaler')",
    "CurveFittingTransformer": "(x_column, y_column, producer, cutoff_ranges=None)",
    "MeasurementCutter": "(column, distances)",
    "ImageResizer": "(column, ref_distance)",
    "CommonRegionCutter": "(column, square=False)",
    "HankelTransformer": "(column, start_radius=0, order=0)",
    "SpecimenStatusToSoftLabels": (
        "(status_col: 'str' = 'specimen_status', rule_cols: "
        "'Optional[List[str]]' = None, output_col: 'str' = "
        "'cancer_status_soft', class_order: 'Optional[List[Union[str, int]]]' = "
        "None, rules: 'Optional[Dict[Tuple[Optional[str], Any, str], "
        "List[float]]]' = None, normalize: 'bool' = True, strict: 'bool' = "
        "False, capitalize_status: 'bool' = True) -> 'None'"
    ),
    "SoftLabelToWeightedSamples": (
        "(soft_col: 'str' = 'cancer_status_soft', label_col: 'str' = "
        "'cancer_status', weight_col: 'str' = 'cancer_status_weighted', "
        "class_names: 'Optional[List[Union[str, int]]]' = None, "
        "label_col_numeric: 'Optional[str]' = None, label_codes: "
        "'Optional[List[Any]]' = None, label_code_map: 'Optional[Dict[Any, Any]]' "
        "= None, min_weight: 'float' = 0.0, normalize: 'bool' = True, "
        "drop_soft_col: 'bool' = False) -> 'None'"
    ),
}


def _instances():
    return [
        transformers.ColumnStandardizer("radial_profile_data"),
        transformers.RuleBasedProfileFilter([]),
        transformers.ColumnExtractor(["radial_profile_data"]),
        transformers.ColumnCleaner([]),
        transformers.QRangeSetter(),
        transformers.SlopeRemoval(),
        transformers.FourierTransform(),
        transformers.GoodnessTransformer(),
        transformers.GoodnessFilter(),
        transformers.DataPreparation(),
        transformers.NormScaler(),
        transformers.CurveFittingTransformer("q_range", "profile", "demo"),
        transformers.MeasurementCutter("profile", []),
        transformers.ImageResizer("image", 1.0),
        transformers.CommonRegionCutter("profile"),
        transformers.HankelTransformer("polar_data"),
        transformers.SpecimenStatusToSoftLabels(),
        transformers.SoftLabelToWeightedSamples(),
    ]


def test_extracted_transformers_keep_canonical_direct_imports_and_signatures():
    module = importlib.import_module(_CANONICAL_MODULE)

    for name, signature in _MOVED_SIGNATURES.items():
        direct = __import__(_CANONICAL_MODULE, fromlist=[name])
        transformer_type = getattr(module, name)
        assert getattr(direct, name) is transformer_type
        assert transformer_type.__module__ == _CANONICAL_MODULE
        assert transformer_type.__qualname__ == name
        assert str(inspect.signature(transformer_type)) == signature


def test_rule_based_profile_filter_remains_sklearn_cloneable():
    original = transformers.RuleBasedProfileFilter(
        [(1.0, ">", 0.5)], evaluation_mode="interp", drop_failures=True
    )

    copied = clone(original)

    assert type(copied) is type(original)
    assert copied.get_params(deep=False) == original.get_params(deep=False)


@pytest.mark.parametrize(
    "instance",
    _instances(),
    ids=lambda instance: type(instance).__name__,
)
def test_extracted_transformer_joblib_round_trips_keep_canonical_paths(
    tmp_path, instance
):
    path = tmp_path / f"{type(instance).__name__}.joblib"
    joblib.dump(instance, path)
    restored = joblib.load(path)

    assert type(restored) is type(instance)
    assert type(restored).__module__ == _CANONICAL_MODULE
    assert type(restored).__qualname__ == type(instance).__name__
    assert b"xrdanalysis.data_processing._transformer_" not in path.read_bytes()
