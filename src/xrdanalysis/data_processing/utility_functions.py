"""Various utility functions used in different parts of codebase"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from skimage.measure import label, regionprops
from sklearn.preprocessing import Normalizer, StandardScaler

from xrdanalysis.data_processing.containers import MLCluster


def get_center(data: np.ndarray, threshold=3.0) -> Tuple[float]:
    """
    Determines the center of the beam in SAXS data.

    :param data: The input SAXS data.
    :type data: np.ndarray
    :param threshold: The threshold factor for identifying the center of the\
        beam.
        Defaults to 3.0 times the average value of the input data.
    :type threshold: float, optional
    :returns: The coordinates of the center of the beam in the input data.
        If no center is found, returns (np.NaN, np.NaN).
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
        center = (np.NaN, np.NaN)

    return center


def generate_poni(df, path):
    """
    Generates .poni files from the provided DataFrame and saves them to the\
    specified directory.

    :param df: Input DataFrame containing calibration measurement IDs and\
        corresponding .poni file content.
    :type df: pd.DataFrame
    :param path: Path to the directory where .poni files will be saved.
    :type path: str
    :returns: Path to the directory where .poni files are saved.
    :rtype: str
    """

    directory_path = os.path.join(os.getcwd(), path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    df_cut = df[["calibration_measurement_id", "ponifile"]]

    df_unique = df_cut.drop_duplicates(subset="calibration_measurement_id")

    df_ponis = df_unique.dropna(subset="ponifile")

    for _, row in df_ponis.iterrows():
        filename = str(row["calibration_measurement_id"]) + ".poni"
        text_content = row["ponifile"]
        if text_content:
            # Write the text content to the file
            with open(
                os.path.join(directory_path, filename),
                "w",
            ) as file:
                file.write(text_content)
    return directory_path


def create_mask(faulty_pixels):
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
        # Initialize the mask array for a 256x256 detector
        mask = np.zeros((256, 256), dtype=np.uint8)
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


def interpolate_cluster(df, cluster_label, perc_min, perc_max, azimuth):
    """
    Interpolates a cluster of azimuthal integration data.

    :param df: Input DataFrame containing data to be interpolated.
    :type df: pd.DataFrame
    :param cluster_label: Label of the cluster to be interpolated.
    :type cluster_label: int
    :param perc_min: The minimum percentage of the maximum q-range for\
        interpolation.
    :type perc_min: float
    :param perc_max: The maximum percentage of the maximum q-range for\
        interpolation.
    :type perc_max: float
    :param azimuth: A transformer to use for azimuthal integration.
    :type azimuth: Callable
    :returns: Interpolated `MLCluster` object.
    :rtype: MLCluster
    """

    cluster_indices = df[df["q_cluster_label"] == cluster_label].index
    dfc = df.loc[cluster_indices].copy()
    q_max_min = np.min(dfc["q_range_max"])
    q_min = q_max_min * perc_min
    q_max = q_max_min * perc_max
    q_range = (q_min, q_max)
    dfc["interpolation_q_range"] = [q_range] * len(dfc)

    dfc = azimuth.transform(dfc)

    return MLCluster(df=dfc, q_cluster=cluster_label, q_range=dfc["q_range"])


def normalize_scale_cluster(
    cluster: MLCluster, std=None, norm=None, do_fit=True
):
    """
    Normalizes and scales a cluster of azimuthal integration data.

    :param cluster: The cluster to be normalized and scaled.
    :type cluster: MLCluster
    :param std: StandardScaler instance for scaling.
        If None, a new instance will be created. Defaults to None.
    :type std: StandardScaler or None, optional
    :param norm: Normalizer instance for normalization.
        If None, a new instance will be created. Defaults to None.
    :type norm: Normalizer or None, optional
    :param do_fit: Whether to fit the scaler. Defaults to True.
    :type do_fit: bool, optional
    """
    if not norm:
        norm = Normalizer(norm="l2")
    if not std:
        std = StandardScaler()
    dfc = cluster.df.copy()
    dfc["radial_profile_data_norm"] = dfc["radial_profile_data"].apply(
        lambda x: norm.transform([x])[0]
    )

    matrix_2d = np.vstack(dfc["radial_profile_data_norm"].values)
    if do_fit:
        scaled_data = std.fit_transform(matrix_2d)
    else:
        scaled_data = std.transform(matrix_2d)

    dfc["radial_profile_data_norm_scaled"] = [arr for arr in scaled_data]

    cluster.df = dfc
    cluster.normalizer = norm
    cluster.std = std


def remove_outliers_by_cluster(
    dataframe, z_score_threshold, direction="negative", num_clusters=4
):
    """
    Removes outliers from a DataFrame by clustering and z-score thresholding.

    :param dataframe: The input DataFrame containing the data.
    :type dataframe: pd.DataFrame
    :param z_score_threshold: The threshold for Z-score based outlier removal.
    :type z_score_threshold: float
    :param direction: The direction of outlier removal, either "both",\
        "positive", or "negative". Defaults to "negative".
    :type direction: str, optional
    :param num_clusters: The number of clusters to use.\
        Defaults to 4.
    :type num_clusters: int, optional
    :returns: A DataFrame with outliers removed based on the specified\
        criteria.
    :rtype: pd.DataFrame
    """
    # Create an empty DataFrame to store non-outlier values
    filtered_data = pd.DataFrame(columns=dataframe.columns)

    for cluster in range(num_clusters):
        cluster_indices = dataframe[
            dataframe["q_cluster_label"] == cluster
        ].index

        # Extract q_range_max values for the current cluster
        q_range_max_values = dataframe.loc[cluster_indices, "q_range_max"]

        # Calculate Z-scores for q_range_max values
        z_scores = zscore(q_range_max_values)

        # Determine indices of non-outliers based on the chosen direction
        if direction == "both":
            non_outlier_indices = np.abs(z_scores) < z_score_threshold
        elif direction == "positive":
            non_outlier_indices = z_scores < z_score_threshold
        elif direction == "negative":
            non_outlier_indices = z_scores > -z_score_threshold
        else:
            raise ValueError(
                "Invalid direction. Use 'both', 'positive', or 'negative'."
            )

        # Select relevant entries from dataframe
        selected_entries = dataframe.loc[cluster_indices[non_outlier_indices]]

        # Exclude empty or all-NA columns
        selected_entries = selected_entries.dropna(how="all", axis=1)

        # Concatenate filtered_data with selected entries
        if filtered_data.empty:
            filtered_data = selected_entries
        else:
            filtered_data = pd.concat(
                [filtered_data, selected_entries], ignore_index=True
            )

    return filtered_data
