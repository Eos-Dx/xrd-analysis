import os
from typing import Tuple

import numpy as np
from skimage.measure import label, regionprops
from sklearn.preprocessing import Normalizer, StandardScaler

from xrdanalysis.data.containers import MLCluster
from xrdanalysis.data.transformers import AzimuthalIntegration


def get_center(data: np.ndarray, threshold=3.0) -> Tuple[float]:
    """
    Takes SAXS data and determines the center of the beam.

    Parameters:
    - data : numpy.ndarray
        The input SAXS data.
    - threshold : float, optional
        The threshold factor for identifying the center of the beam.
        Defaults to 3.0 times the average value of the input data.

    Returns:
    - center : tuple of float
        The coordinates of the center of the beam in the input data.
        If no center is found, returns (np.NaN, np.NaN).
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
    Generate .poni files from the provided DataFrame and save them
    to the specified directory.

    Args:
        df (DataFrame): Input DataFrame containing calibration measurement
        IDs and corresponding .poni file content.
        path (str): Path to the directory where .poni files will be saved.

    Returns:
        str: Path to the directory where .poni files are saved.
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
    Create a mask array to identify faulty pixels.

    Args:
        faulty_pixels (list of tuples or None): List of (y, x)
        coordinates representing faulty pixels.

    Returns:
        numpy.ndarray or None: Mask array where 1 indicates a
        faulty pixel and 0 indicates a good pixel.
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
    Check if all elements in the input array are None.

    Args:
        array (iterable): Input array or iterable.

    Returns:
        bool: True if all elements are None, False otherwise.
    """
    return all(x is None for x in array)


def is_nan_pair(x):
    """
    Check if the input is a tuple of two NaN values.

    Args:
        x (tuple or any): Input to check.

    Returns:
        bool: True if the input is a tuple of two NaN values, False otherwise.
    """
    if isinstance(x, tuple) and len(x) == 2:
        return np.isnan(x[0]) and np.isnan(x[1])
    return False


def interpolate_cluster(
    df, cluster_label, perc_min, perc_max, q_resolution, faulty_pixels_array
):
    """
    Interpolate a cluster of azimuthal integration data.

    Parameters:
        df (DataFrame): Input DataFrame containing data to be interpolated.
        cluster_label (int): Label of the cluster to be interpolated.
        perc_min (float): The minimum percentage of the maximum
        q-range for interpolation.
        perc_max (float): The maximum percentage of the maximum
        q-range for interpolation.
        q_resolution (int): The resolution for interpolation.
        faulty_pixels_array (list): A list of faulty pixel coordinates.

    Returns:
        MLCluster: Interpolated MLCluster object.
    """

    cluster_indices = df[df["q_cluster_label"] == cluster_label].index
    dfc = df.loc[cluster_indices].copy()
    q_max_min = np.min(dfc["q_range_max"])
    q_min = q_max_min * perc_min
    q_max = q_max_min * perc_max
    q_range = (q_min, q_max)
    dfc["interpolation_q_range"] = [q_range] * len(dfc)

    azimuth = AzimuthalIntegration(
        faulty_pixels=faulty_pixels_array, npt=q_resolution
    )
    dfc = azimuth.transform(dfc)

    return MLCluster(df=dfc, q_cluster=cluster_label, q_range=dfc["q_range"])


def normalize_scale_cluster(
    cluster: MLCluster, std=None, norm=None, do_fit=True
):
    """
    Normalize and scale a cluster of azimuthal integration data.

    Parameters:
        cluster (MLCluster): The cluster to be normalized and scaled.
        std (StandardScaler or None, optional): StandardScaler instance
        for scaling. If None, a new instance will be created. Defaults to None.
        norm (Normalizer or None, optional): Normalizer instance for
        normalization. If None, a new instance will be created. Defaults
        to None.
        do_fit (bool, optional): Whether to fit the scaler. Defaults to True.

    Returns:
        None
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
