"""Various utility functions used in different parts of codebase"""

import json
import re
import tempfile
from functools import wraps
from pathlib import Path
from typing import Tuple

import cv2
import h5py  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from skimage.measure import label, regionprops
from sklearn.metrics import RocCurveDisplay  # noqa: F401
from sklearn.metrics import auc  # noqa: F401
from sklearn.metrics import f1_score  # noqa: F401
from sklearn.metrics import precision_score  # noqa: F401
from sklearn.metrics import roc_curve  # noqa: F401

import xrdanalysis.data_processing._evaluation_utils as _evaluation_utils
import xrdanalysis.data_processing._utility_angular as _utility_angular
import xrdanalysis.data_processing._utility_hdf as _utility_hdf
import xrdanalysis.data_processing._utility_image as _utility_image
import xrdanalysis.data_processing._utility_statistics as _utility_statistics


def combine_h5_to_df(file_paths):
    """Combine calibration and measurement data from multiple HDF5 files."""
    return _utility_hdf.combine_h5_to_df(file_paths, h5_to_df=h5_to_df)


def h5_to_df(file_path):
    """Convert one supported HDF5 layout into calibration and measurement tables."""
    return _utility_hdf.h5_to_df(file_path)


def compute_group_statistics(df, label_column, array_column):
    """Compute mean and standard-deviation arrays per label."""
    return _utility_statistics.compute_group_statistics(df, label_column, array_column)


def plot_group_statistics(df, label_column, array_column, selected_labels=None):
    """Plot grouped profile statistics with public monkeypatch seams."""
    return _utility_statistics.plot_group_statistics(
        df,
        label_column,
        array_column,
        selected_labels,
        compute_group_statistics=compute_group_statistics,
        plt=plt,
    )


def unpack_results(result):
    (
        above_limits,
        images_above,
        averages_higher,
        below_limits,
        images_below,
        averages_lower,
    ) = result
    """
    Unpack and reorganize result data into a single dictionary with \
    dynamically named columns.

    :param result: A tuple containing lists of above and below limit \
    values, including images and averages.
    :type result: Tuple[List[float], List[Any], List[float], \
    List[float], List[Any], List[float]]
    :returns: A dictionary with dynamically named columns for images \
    and deviations above and below specified limits.
    :rtype: Dict[str, Any]
    """
    # Initialize dictionaries to store the columns
    above_columns = {}
    below_columns = {}

    # Loop through above limits with enumerate for indexing
    for index, limit in enumerate(above_limits):
        above_columns[f"image_above_{limit}"] = images_above[index]
        above_columns[f"deviation_above_{limit}"] = averages_higher[index]

    # Loop through below limits with enumerate for indexing
    for index, limit in enumerate(below_limits):
        below_columns[f"image_below_{limit}"] = images_below[index]
        below_columns[f"deviation_below_{limit}"] = averages_lower[index]
    # Merge above and below columns
    return {**above_columns, **below_columns}


def unpack_results_cake(result):
    (
        above_limits,
        cakes_above,
        images_above,
        averages_higher,
        below_limits,
        cakes_below,
        images_below,
        averages_lower,
    ) = result
    """
    Unpack and reorganize result data into a single dictionary \
    with dynamically named columns, including cake-specific information.

    :param result: A tuple containing lists of above and below limit values, \
    including cake identifiers, images, and averages.
    :type result: Tuple[List[float], List[Any], List[Any], List[float], \
    List[float], List[Any], List[Any], List[float]]
    :returns: A dictionary with dynamically named columns for cakes, \
    images, and deviations above and below specified limits.
    :rtype: Dict[str, Any]
    """
    # Initialize dictionaries to store the columns
    above_columns = {}
    below_columns = {}

    # Loop through above limits with enumerate for indexing
    for index, limit in enumerate(above_limits):
        above_columns[f"image_above_{limit}"] = images_above[index]
        above_columns[f"cake_above_{limit}"] = cakes_above[index]
        above_columns[f"deviation_above_{limit}"] = averages_higher[index]

    # Loop through below limits with enumerate for indexing
    for index, limit in enumerate(below_limits):
        below_columns[f"image_below_{limit}"] = images_below[index]
        below_columns[f"cake_below_{limit}"] = cakes_below[index]
        below_columns[f"deviation_below_{limit}"] = averages_lower[index]
    # Merge above and below columns
    return {**above_columns, **below_columns}


def process_angular_ranges(angles):
    """Normalize and split angular ranges at the +/-180-degree boundary."""
    return _utility_angular.process_angular_ranges(angles)


def get_angle_span(start, end):
    """Calculate angular span while handling wraparound."""
    return _utility_angular.get_angle_span(start, end)


def prepare_angular_ranges(start_angle, end_angle):
    """Prepare public angular helpers for weighted integration."""
    return _utility_angular.prepare_angular_ranges(
        start_angle,
        end_angle,
        get_angle_span=get_angle_span,
        process_angular_ranges=process_angular_ranges,
    )


def perform_weighted_integration(
    data, ai_cached, range_info, npt, interpolation_q_range, mask=None
):
    """Perform weighted azimuthal integration for one angular region."""
    return _utility_angular.perform_weighted_integration(
        data, ai_cached, range_info, npt, interpolation_q_range, mask
    )


def unpack_rotating_angles_results(results):
    """Flatten rotating-angle analysis output into dataframe-ready columns."""
    return _utility_angular.unpack_rotating_angles_results(results)


def get_center(data: np.ndarray, threshold=3.0) -> Tuple[float]:
    """
    Determines the center of the beam in SAXS data.

    :param data: The input SAXS data.
    :type data: np.ndarray
    :param threshold: The threshold factor for identifying the center of the\
    beam. Defaults to 3.0 times the average value of the input data.
    :type threshold: float, optional
    :returns: The coordinates of the center of the beam in the input data. \
    If no center is found, returns (np.nan, np.nan).
    :rtype: tuple
    """
    average_value = np.nanmean(data)

    # Set the threshold to be X higher than the average value
    threshold = threshold * average_value
    binary_image = data > threshold

    # Label connected regions
    labeled_image = label(binary_image)

    # Get region properties for all labeled regions
    regions = regionprops(labeled_image)

    # Find the largest region
    max_area = 0
    max_region = None

    for region in regions:
        if region.area > max_area:
            max_area = region.area
            max_region = region

    # Get the centroid of the largest region
    if max_region is not None:
        center = max_region.centroid
        center = (center[0], center[1])
    else:
        center = (np.nan, np.nan)

    return center


def mask_beam_center(image: np.ndarray, thresh: float, padding: int = 0):
    """
    Isolates and removes a central beam from an image based on a threshold \
    value.

    :param image: Input image to process for beam removal.
    :type image: np.ndarray
    :param thresh: Threshold value to identify the beam region. Pixels above \
    this value are considered part of the beam.
    :type thresh: float
    :param padding: Additional padding around the detected beam region in \
    pixels. Defaults to 0.
    :type padding: int, optional
    :param return_coords: If True, returns the beam coordinates along with the\
        isolated beam image. Defaults to False.
    :type return_coords: bool, optional
    :returns: If return_coords is False, returns only the isolated beam image.\
        If return_coords is True, returns a tuple containing the isolated beam\
        image and a dictionary of beam coordinates and measurements.
    :rtype: Union[np.ndarray, Tuple[np.ndarray, dict]]
    """
    # Create beam mask and find coordinates
    primary_beam_mask = image > thresh

    # Find beam boundaries
    true_indices = np.argwhere(primary_beam_mask)
    min_row = max(0, true_indices[:, 0].min() - padding)
    max_row = min(image.shape[0] - 1, true_indices[:, 0].max() + padding)
    min_col = max(0, true_indices[:, 1].min() - padding)
    max_col = min(image.shape[1] - 1, true_indices[:, 1].max() + padding)

    # Create output array
    beam = np.zeros_like(image)

    # Extract beam region
    beam_region = image[min_row : max_row + 1, min_col : max_col + 1]  # noqa: E203
    beam[min_row : max_row + 1, min_col : max_col + 1] = beam_region  # noqa: E203

    return beam


def calculate_poni_from_pixels(poni_str, center_x, center_y):
    detector_config_str = re.search(r"Detector_config:\s*(\{.*\})", poni_str).group(1)
    detector_config = json.loads(detector_config_str)

    # Get pixel sizes
    pixel1 = detector_config["pixel1"]
    pixel2 = detector_config["pixel2"]

    # Calculate PONI values
    poni1 = center_y * pixel1
    poni2 = center_x * pixel2

    return poni1, poni2


def adjust_poni_centers_coef(poni_str, k):
    centers_poni = ["Poni1", "Poni2"]
    new_ponifile_text = poni_str
    adjusted_poni = []
    for center in centers_poni:
        distance_index = new_ponifile_text.find(center) + len(center) + 1
        end_of_line_index = new_ponifile_text.find("\n", distance_index)
        adjusted = float(new_ponifile_text[distance_index:end_of_line_index]) * k
        adjusted_poni.append(adjusted)

    return adjusted_poni


def substitute_poni_centers(poni_str, poni1, poni2):
    centers_poni = ["Poni1", "Poni2"]
    new_ponifile_text = poni_str
    for center, adjusted in zip(centers_poni, [poni1, poni2]):
        distance_index = new_ponifile_text.find(center) + len(center) + 1
        end_of_line_index = new_ponifile_text.find("\n", distance_index)
        new_ponifile_text = (
            new_ponifile_text[:distance_index]
            + f"{adjusted}"
            + new_ponifile_text[end_of_line_index:]
        )
        adjusted_poni = new_ponifile_text

    return adjusted_poni


def format_poni_string(poni_str):
    # Check if string already contains newlines
    if "\n" in poni_str:
        return poni_str

    # Keywords that should trigger a new line
    keywords = [
        "# Nota:",
        "# Calibration",
        "poni_version:",
        "Detector:",
        "Detector_config:",
        "Distance:",
        "Poni1:",
        "Poni2:",
        "Rot1:",
        "Rot2:",
        "Rot3:",
        "Wavelength:",
        "# Calibrant:",
        "# Image:",
    ]

    # Split the string into parts based on the keywords
    formatted_parts = []
    current_pos = 0

    for keyword in keywords:
        pos = poni_str.find(keyword, current_pos)
        if pos != -1:
            # If not at the start, add the previous part
            if pos > current_pos:
                formatted_parts.append(poni_str[current_pos:pos].strip())
            current_pos = pos

    # Add the last part
    if current_pos < len(poni_str):
        formatted_parts.append(poni_str[current_pos:].strip())

    # Join the parts with newlines
    return "\n".join(formatted_parts)


def generate_poni_from_text(ponifile_text):
    """
    Generates a temporary .poni file from the provided ponifile text.

    :param ponifile_text: Text content of the .poni file
    :type ponifile_text: str
    :returns: Path to the temporary .poni file
    :rtype: str
    """
    ponifile_text = format_poni_string(ponifile_text)
    # Create a temporary file with .poni extension
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".poni"
    ) as temp_file:
        temp_file.write(ponifile_text)
        temp_file_path = temp_file.name

    return temp_file_path


def generate_poni(df: pd.DataFrame, out_dir: str) -> str:
    """Generate .poni files from DataFrame column 'ponifile' and return directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        mid = (
            int(row["calibration_measurement_id"])
            if "calibration_measurement_id" in row
            else int(row["measurement_id"])
        )
        text = str(row["ponifile"]) if "ponifile" in row else ""
        (out / f"{mid}.poni").write_text(text)
    return str(out)


class MLCluster:
    """Minimal MLCluster structure used by tests."""

    def __init__(self, df: pd.DataFrame, q_cluster: int, q_range=None):
        self.df = df
        self.q_cluster = q_cluster
        self.q_range = q_range


def interpolate_cluster(
    df: pd.DataFrame, cluster_label: int, perc_min: float, perc_max: float, azimuth
) -> MLCluster:
    """Prepare a cluster subset with interpolation_q_range and call azimuth.transform."""
    sub = df[df["q_cluster_label"] == cluster_label].copy()
    if sub.empty:
        return MLCluster(sub, cluster_label, None)
    # Compute per-row interpolation range from q_range_max
    qmin = (sub["q_range_max"] * perc_min).astype(int)
    qmax = (sub["q_range_max"] * perc_max).astype(int)
    sub["interpolation_q_range"] = [[int(a), int(b)] for a, b in zip(qmin, qmax)]
    # Call the provided transformer
    azimuth.transform(sub)
    return MLCluster(sub, cluster_label, (int(qmin.iloc[0]), int(qmax.iloc[0])))


def normalize_scale_cluster(cluster: MLCluster):
    """Add normalized and scaled columns to MLCluster.df."""

    def _norm(arr):
        arr = np.asarray(arr)
        m = np.max(np.abs(arr)) or 1.0
        return arr / m

    cluster.df["radial_profile_data_norm"] = cluster.df["radial_profile_data"].apply(
        _norm
    )
    # For tests, a simple copy as 'scaled' is enough
    cluster.df["radial_profile_data_norm_scaled"] = cluster.df[
        "radial_profile_data_norm"
    ]


def remove_outliers_by_cluster(
    df: pd.DataFrame, z_score_threshold: float, direction: str, num_clusters: int
) -> pd.DataFrame:
    """Remove outliers in q_range_max per cluster_label using a simple z-score rule."""

    def _filter(group: pd.DataFrame) -> pd.DataFrame:
        vals = group["q_range_max"].astype(float)
        mu = vals.mean()
        sigma = vals.std(ddof=0) or 1.0
        if direction == "positive":
            mask = vals <= mu + z_score_threshold * sigma
        elif direction == "negative":
            mask = vals >= mu - z_score_threshold * sigma
        elif direction == "both":
            mask = (vals >= mu - z_score_threshold * sigma) & (
                vals <= mu + z_score_threshold * sigma
            )
        else:
            raise ValueError(
                "Invalid direction. Use 'both', 'positive', or 'negative'."
            )
        return group[mask]

    return df.groupby("q_cluster_label", group_keys=False).apply(_filter)


def create_mask(faulty_pixels, size=(256, 256)):
    """
    Creates a mask array to identify faulty pixels.

    :param faulty_pixels: List of (y, x) coordinates representing faulty\
    pixels, or None.
    :type faulty_pixels: list of tuples or None
    :returns: Mask array where 1 indicates a faulty pixel and 0 indicates a\
    good pixel, or None.
    :rtype: numpy.ndarray or None
    """
    if faulty_pixels is not None:
        # Initialize the mask array for a size detector
        mask = np.zeros(size, dtype=np.uint8)
        for y, x in faulty_pixels:
            mask[y, x] = 1
    else:
        mask = None
    return mask


def is_all_none(array):
    """
    Checks if all elements in the input array are None.

    :param array: Input array or iterable.
    :type array: iterable
    :returns: True if all elements are None, False otherwise.
    :rtype: bool
    """

    return all(x is None for x in array)


def is_nan_pair(pair):
    """
    Checks if the input is a tuple of two NaN values.

    :param x: Input to check.
    :type x: tuple or any
    :returns: True if the input is a tuple of two NaN values, False otherwise.
    :rtype: bool
    """
    if isinstance(pair, tuple) and len(pair) == 2:
        return all(np.isnan(x) for x in pair)
    return False


SCALED_DATA = "radial_profile_data_norm_scaled"


def prep(df):
    """
    Prepare dataframe to RF learning or other models.
    It uses SCALED_DATA and age of the patient's measurement.
    """
    dfc = df.copy()
    transformed_data = np.vstack(dfc[SCALED_DATA].values)
    counts = dfc.groupby("patient_id").size()
    dfc["entry_count"] = dfc["patient_id"].map(counts)
    entry = dfc["entry_count"].values.reshape(-1, 1)
    if "age" not in dfc.columns:
        dfc["age"] = [-1] * len(dfc)
    age = dfc["age"].values.reshape(-1, 1)
    return np.concatenate((transformed_data, entry, age), axis=1)


def show_data(df: pd.DataFrame):
    # Initialize an empty DataFrame to append to
    dfki = df.copy()

    # Group the DataFrame by 'q_cluster_label' and 'cancer_tissue'
    grouped = dfki.groupby(["type_measurement", "cancer_diagnosis"])

    # Create a grid of subplots, where the number of rows is determined by the
    # number of clusters
    num_clusters = len(dfki["cancer_diagnosis"].unique())
    fig, axes = plt.subplots(
        nrows=num_clusters, ncols=2, figsize=(5.5, 2.5 * num_clusters)
    )

    # Flatten the axes for easy indexing
    axes = axes.flatten()
    # Iterate over clusters and cancer/non-cancer tissues
    for i, ((cluster, diagnosis), group) in enumerate(grouped):
        ax = axes[i]

        # Plot individual entries
        for _, entry in group.iterrows():
            ax.plot(entry["q_range"], entry["radial_profile_data"])

        # Compute average radial profile data for the current cluster
        # and diagnosis
        average_radial_profile = group["radial_profile_data"].mean()
        ax.plot(
            group.iloc[0]["q_range"],
            average_radial_profile,
            color="black",
            linestyle="--",
            linewidth=2,
        )

        # Add labels and a legend for the current cluster
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        if diagnosis:
            diag = "Cancer"
        else:
            diag = "Non-cancer"

        ax.set_title(f"{cluster} - {diag}")
        ax.set_yscale("log")
        ax.set_xlabel("q, nm$^{-1}$")
        ax.set_ylabel("Intensity")
        ax.grid(True)
        m = np.max(group.iloc[0]["q_range"])
        ax.set_ylim(0.5, 1000)
        # ax.savefig(f'{dis}_{diag}.png', dpi=400)
        if m < 5:
            ax.set_xticks(np.arange(0, m + 0.1, 1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))
        else:
            ax.set_xticks(np.arange(0, m + 0.1, 5))
            ax.xaxis.set_minor_locator(MultipleLocator(1))

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    plt.savefig("plot.png", dpi=400)
    # Show the plots
    plt.show()


@wraps(_evaluation_utils.custom_splitter_balanced)
def custom_splitter_balanced(df, split):
    """Compatibility wrapper retaining the historical public function path."""
    return _evaluation_utils.custom_splitter_balanced(df, split)


custom_splitter_balanced.__module__ = __name__


@wraps(_evaluation_utils.viz_roc)
def viz_roc(fig, axes, model_name, predictor, text_on=True, legend_on=True):
    """Compatibility wrapper retaining the historical public function path."""
    return _evaluation_utils.viz_roc(
        fig, axes, model_name, predictor, text_on, legend_on
    )


viz_roc.__module__ = __name__


@wraps(_evaluation_utils.metrics)
def metrics(tpr, fpr, thresholds, y_score, y_true, roc_auc):
    """Compatibility wrapper retaining the historical public function path."""
    return _evaluation_utils.metrics(tpr, fpr, thresholds, y_score, y_true, roc_auc)


metrics.__module__ = __name__


@wraps(_evaluation_utils.viz_roc_balanced)
def viz_roc_balanced(fig, axes, model_name, estimators):
    """Compatibility wrapper retaining the historical public function path."""
    return _evaluation_utils.viz_roc_balanced(fig, axes, model_name, estimators)


viz_roc_balanced.__module__ = __name__


@wraps(_evaluation_utils.generate_roc_based_metrics)
def generate_roc_based_metrics(
    y_true, y_score, show_flag=True, min_sensitivity=None, min_specificity=None
):
    """Compatibility wrapper retaining the historical public function path."""
    return _evaluation_utils.generate_roc_based_metrics(
        y_true, y_score, show_flag, min_sensitivity, min_specificity
    )


generate_roc_based_metrics.__module__ = __name__


@wraps(_evaluation_utils.calculate_optimal_threshold)
def calculate_optimal_threshold(
    y_true,
    y_score,
    min_sensitivity=None,
    min_specificity=None,
    print_flag=False,
):
    """Compatibility wrapper retaining the historical public function path."""
    return _evaluation_utils.calculate_optimal_threshold(
        y_true, y_score, min_sensitivity, min_specificity, print_flag
    )


calculate_optimal_threshold.__module__ = __name__


def extract_image_data_values(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract image values according to the legacy binary mask convention."""
    return _utility_image.extract_image_data_values(data, mask)


def extract_center_from_poni(poni_text: str):
    """Extract pixel-space beam-centre coordinates from PONI calibration text."""
    return _utility_image.extract_center_from_poni(poni_text)


def filter_points_by_distance(row, column, distances=None):
    """Apply radial-distance filtering with the public PONI parser seam."""
    return _utility_image.filter_points_by_distance(
        row, column, distances, extract_center_from_poni=extract_center_from_poni
    )


def extract_distance_from_poni(poni_text: str):
    """Extract detector distance from PONI calibration text."""
    return _utility_image.extract_distance_from_poni(poni_text)


def resize_image(row, column, ref_dist):
    """Resize an image while preserving public PONI helper seams."""
    return _utility_image.resize_image(
        row,
        column,
        ref_dist,
        extract_distance_from_poni=extract_distance_from_poni,
        adjust_poni_centers_coef=adjust_poni_centers_coef,
        substitute_poni_centers=substitute_poni_centers,
        cv2=cv2,
    )


def find_common_region(df, column, square=False):
    """Find common image bounds with the public PONI parser seam."""
    return _utility_image.find_common_region(
        df, column, square, extract_center_from_poni=extract_center_from_poni
    )


def cut_common_region(row, column, max_up, max_down, max_left, max_right):
    """Crop a common region while preserving public PONI helper seams."""
    return _utility_image.cut_common_region(
        row,
        column,
        max_up,
        max_down,
        max_left,
        max_right,
        extract_center_from_poni=extract_center_from_poni,
        calculate_poni_from_pixels=calculate_poni_from_pixels,
        substitute_poni_centers=substitute_poni_centers,
    )


def filter_dataframe_by_rules(
    df: pd.DataFrame,
    rules: dict,
    q_col: str = "q_range",
    data_col: str = "radial_profile_data",
    type_col: str = "type_measurement",
) -> pd.DataFrame:
    """
    Filter a DataFrame based on interpolation rules at specific q-values.

    This function allows quality control of XRD data by filtering measurements
    based on intensity thresholds at specific q-values. Each measurement type
    (e.g. WAXS, SAXS) can have different filtering criteria.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing XRD data to filter
    rules : dict
        Dictionary mapping measurement types to lists of (q_target, operator, threshold) tuples.
        Example: {
            'WAXS': [(4.0, '>', 0.006), (7.5, '<', 0.009)],
            'SAXS': [(1.5, '>', 0.008), (1.32, '<', 0.02)]
        }
    q_col : str, default='q_range'
        Column name for q_range (iterable of floats)
    data_col : str, default='radial_profile_data'
        Column name for radial_profile_data (iterable of floats)
    type_col : str, default='type_measurement'
        Column name for the measurement type

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only rows that satisfy all rules for their measurement type

    Examples
    --------
    >>> rules = {
    ...     'WAXS': [(4.0, '>', 0.006), (18.5, '<', 0.02)],
    ...     'SAXS': [(1.5, '>', 0.008)]
    ... }
    >>> df_filtered = filter_dataframe_by_rules(df, rules)
    """
    import operator as op

    # Map operator strings to functions
    ops = {
        "<": op.lt,
        "<=": op.le,
        ">": op.gt,
        ">=": op.ge,
        "==": op.eq,
        "!=": op.ne,
    }

    def row_passes_rules(row):
        """Check if a row passes all rules for its measurement type."""
        # Retrieve the list of rules for this row's measurement type
        rule_list = rules.get(row[type_col], [])

        # If no rules for this type, keep the row
        if not rule_list:
            return True

        # Check each (q_target, operator, threshold) rule
        for q_target, op_str, thresh in rule_list:
            # Interpolate intensity at q_target
            intensity = np.interp(q_target, row[q_col], row[data_col])

            # Apply operator
            if not ops[op_str](intensity, thresh):
                return False

        return True

    # Build a mask of rows to keep
    mask = df.apply(row_passes_rules, axis=1)

    # Return the filtered DataFrame
    return df.loc[mask].reset_index(drop=True)
