"""Public compatibility contracts for utility-function extraction."""

from __future__ import annotations

import importlib
import inspect
import pickle

import pandas as pd
import pytest

UTILITY_MODULE = "xrdanalysis.data_processing.utility_functions"

_FUNCTION_SIGNATURES = {
    "combine_h5_to_df": "(file_paths)",
    "h5_to_df": "(file_path)",
    "compute_group_statistics": "(df, label_column, array_column)",
    "plot_group_statistics": (
        "(df, label_column, array_column, selected_labels=None)"
    ),  # noqa: E501
    "unpack_results": "(result)",
    "unpack_results_cake": "(result)",
    "process_angular_ranges": "(angles)",
    "get_angle_span": "(start, end)",
    "prepare_angular_ranges": "(start_angle, end_angle)",
    "perform_weighted_integration": (
        "(data, ai_cached, range_info, npt, interpolation_q_range, mask=None)"
    ),
    "unpack_rotating_angles_results": "(results)",
    "get_center": "(data: numpy.ndarray, threshold=3.0) -> Tuple[float]",
    "mask_beam_center": (
        "(image: numpy.ndarray, thresh: float, padding: int = 0)"
    ),  # noqa: E501
    "calculate_poni_from_pixels": "(poni_str, center_x, center_y)",
    "adjust_poni_centers_coef": "(poni_str, k)",
    "substitute_poni_centers": "(poni_str, poni1, poni2)",
    "format_poni_string": "(poni_str)",
    "generate_poni_from_text": "(ponifile_text)",
    "generate_poni": "(df: pandas.DataFrame, out_dir: str) -> str",
    "interpolate_cluster": (
        "(df: pandas.DataFrame, cluster_label: int, perc_min: float, "
        "perc_max: float, azimuth) -> "
        "xrdanalysis.data_processing.utility_functions.MLCluster"
    ),
    "normalize_scale_cluster": (
        "(cluster: xrdanalysis.data_processing.utility_functions.MLCluster)"
    ),
    "remove_outliers_by_cluster": (
        "(df: pandas.DataFrame, z_score_threshold: float, direction: str, "
        "num_clusters: int) -> pandas.DataFrame"
    ),
    "create_mask": "(faulty_pixels, size=(256, 256))",
    "is_all_none": "(array)",
    "is_nan_pair": "(pair)",
    "prep": "(df)",
    "show_data": "(df: pandas.DataFrame)",
    "custom_splitter_balanced": "(df, split)",
    "viz_roc": (
        "(fig, axes, model_name, predictor, text_on=True, legend_on=True)"
    ),  # noqa: E501
    "metrics": "(tpr, fpr, thresholds, y_score, y_true, roc_auc)",
    "viz_roc_balanced": "(fig, axes, model_name, estimators)",
    "generate_roc_based_metrics": (
        "(y_true, y_score, show_flag=True, min_sensitivity=None, "
        "min_specificity=None)"
    ),
    "calculate_optimal_threshold": (
        "(y_true, y_score, min_sensitivity=None, min_specificity=None, "
        "print_flag=False)"
    ),
    "extract_image_data_values": (
        "(data: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray"
    ),
    "extract_center_from_poni": "(poni_text: str)",
    "filter_points_by_distance": "(row, column, distances=None)",
    "extract_distance_from_poni": "(poni_text: str)",
    "resize_image": "(row, column, ref_dist)",
    "find_common_region": "(df, column, square=False)",
    "cut_common_region": (
        "(row, column, max_up, max_down, max_left, max_right)"
    ),  # noqa: E501
    "filter_dataframe_by_rules": (
        "(df: pandas.DataFrame, rules: dict, q_col: str = 'q_range', "
        "data_col: str = 'radial_profile_data', "
        "type_col: str = 'type_measurement') -> pandas.DataFrame"
    ),
}

_WILDCARD_NAMES = {
    "MLCluster",
    "MultipleLocator",
    "Path",
    "RocCurveDisplay",
    "SCALED_DATA",
    "Tuple",
    "adjust_poni_centers_coef",
    "auc",
    "calculate_optimal_threshold",
    "calculate_poni_from_pixels",
    "combine_h5_to_df",
    "compute_group_statistics",
    "create_mask",
    "custom_splitter_balanced",
    "cut_common_region",
    "cv2",
    "extract_center_from_poni",
    "extract_distance_from_poni",
    "extract_image_data_values",
    "f1_score",
    "filter_dataframe_by_rules",
    "filter_points_by_distance",
    "find_common_region",
    "format_poni_string",
    "generate_poni",
    "generate_poni_from_text",
    "generate_roc_based_metrics",
    "get_angle_span",
    "get_center",
    "h5_to_df",
    "h5py",
    "interpolate_cluster",
    "is_all_none",
    "is_nan_pair",
    "json",
    "label",
    "mask_beam_center",
    "metrics",
    "normalize_scale_cluster",
    "np",
    "pd",
    "perform_weighted_integration",
    "plot_group_statistics",
    "plt",
    "precision_score",
    "prep",
    "prepare_angular_ranges",
    "process_angular_ranges",
    "re",
    "regionprops",
    "remove_outliers_by_cluster",
    "resize_image",
    "roc_curve",
    "show_data",
    "substitute_poni_centers",
    "tempfile",
    "unpack_results",
    "unpack_results_cake",
    "unpack_rotating_angles_results",
    "viz_roc",
    "viz_roc_balanced",
    "wraps",
}


@pytest.mark.parametrize("name, signature", _FUNCTION_SIGNATURES.items())
def test_public_utility_functions_keep_signature_identity_and_pickle_path(
    name,
    signature,
):
    utility = importlib.import_module(UTILITY_MODULE)
    function = getattr(utility, name)

    assert str(inspect.signature(function)) == signature
    assert function.__module__ == UTILITY_MODULE
    assert function.__qualname__ == name
    assert pickle.loads(pickle.dumps(function)) is function


def test_mlcluster_keeps_public_constructor_and_pickle_identity():
    utility = importlib.import_module(UTILITY_MODULE)
    cluster = utility.MLCluster(pd.DataFrame({"value": [1]}), 3, (2, 4))
    restored = pickle.loads(pickle.dumps(cluster))

    assert str(inspect.signature(utility.MLCluster)) == (
        "(df: pandas.DataFrame, q_cluster: int, q_range=None)"
    )
    assert utility.MLCluster.__module__ == UTILITY_MODULE
    assert utility.MLCluster.__qualname__ == "MLCluster"
    assert type(restored) is utility.MLCluster
    assert restored.q_cluster == 3
    assert restored.q_range == (2, 4)


def test_wildcard_import_keeps_the_full_legacy_namespace():
    namespace = {}

    exec(f"from {UTILITY_MODULE} import *", namespace)

    wildcard_names = {name for name in namespace if not name.startswith("_")}
    assert wildcard_names == _WILDCARD_NAMES
