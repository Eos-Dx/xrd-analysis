"""Various utility functions used in different parts of codebase"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from skimage.measure import label, regionprops
from sklearn.metrics import (
    RocCurveDisplay,
    auc,
    f1_score,
    precision_score,
    roc_curve,
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
    beam_region = image[
        min_row : max_row + 1, min_col : max_col + 1  # noqa: E203
    ]
    beam[min_row : max_row + 1, min_col : max_col + 1] = (  # noqa: E203
        beam_region
    )

    return beam


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

    if "calibration_measurement_id" not in df.columns:
        df["calibration_measurement_id"] = [
            f"{i + 1}_fake" for i in range(len(df))
        ]

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

    # Define colors for cancer and non-cancer tissues
    colors = {"cancer": "red", "non-cancer": "blue"}

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


def custom_splitter_balanced(df, split):
    total_measurements = len(df)
    df_cancer = df[df["cancer_diagnosis"]]
    df_non_cancer = df[~df["cancer_diagnosis"]]
    cancer_fraction = df_cancer.shape[0] / total_measurements
    non_cancer_fraction = df_non_cancer.shape[0] / total_measurements
    train_size = int((1 - split) * total_measurements)
    train_cancer_size = int(train_size * cancer_fraction)
    train_non_cancer_size = int(train_size * non_cancer_fraction)
    patients_in_train_cancer = set()
    patients_in_train_non_cancer = set()
    unique_patient_ids_cancer = df_cancer["patient_id"].unique()
    unique_patient_ids_non_cancer = df_non_cancer["patient_id"].unique()

    np.random.shuffle(unique_patient_ids_cancer)
    np.random.shuffle(unique_patient_ids_non_cancer)

    train_data_cancer = 0
    train_data_non_cancer = 0
    # Iterate over each patient for cancer
    for patient_id in unique_patient_ids_cancer:
        # Get all rows corresponding to the current patient
        patient_rows = df_cancer[df_cancer["patient_id"] == patient_id]
        if train_data_cancer <= train_cancer_size:
            train_data_cancer += len(patient_rows)
            patients_in_train_cancer.add(patient_id)
        else:
            break
    for patient_id in unique_patient_ids_non_cancer:
        # Get all rows corresponding to the current patient
        patient_rows = df_non_cancer[df_non_cancer["patient_id"] == patient_id]
        if train_data_non_cancer <= train_non_cancer_size:
            train_data_non_cancer += len(patient_rows)
            patients_in_train_non_cancer.add(patient_id)
        else:
            break
    unique_patient_ids = df.patient_id.unique()
    patients_in_train = patients_in_train_cancer.union(
        patients_in_train_non_cancer
    )
    patients_in_test = set(unique_patient_ids) - patients_in_train
    train = df[df["patient_id"].isin(patients_in_train)].index
    train_idx = [df.index.get_loc(label) for label in train]
    test = df[df["patient_id"].isin(patients_in_test)].index
    test_idx = [df.index.get_loc(label) for label in test]
    return train_idx, test_idx


def viz_roc(fig, axes, model_name, predictor, text_on=True, legend_on=True):
    def draw_roc(
        y_score,
        y_true,
        title,
        ax,
        patients,
        measurements,
        cancer_measurements,
        text_on=True,
        legend_on=True,
    ):

        display = RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
        display.plot(ax=ax)
        ax.set_title(title)
        if not legend_on:
            ax.get_legend().remove()

            # Optimal threshold closest to perfect classifier
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Compute optimal sensitivity, specificity, and precision
        optimal_sensitivity = tpr[optimal_idx]
        optimal_specificity = 1 - fpr[optimal_idx]
        y_pred = y_score > optimal_threshold
        optimal_precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Print training set statistics

        control_measurements = measurements - cancer_measurements

        # Round to 1 decimal place
        text = f"""Optimal threshold: {optimal_threshold}
        Best performance closest to ideal classifier:
        Sensitivity: {round(optimal_sensitivity * 100, 1)}%
        Specificity: {round(optimal_specificity * 100, 1)}%
        PPV: {round(optimal_precision * 100, 1)}%
        F1-score: {round(f1 * 100, 1)}%
        Patients: {patients}
        Measurements: {measurements}
        Cancer measurements: {cancer_measurements}
        Control measurements: {control_measurements}
        Size of test set: {y_true.shape[0]}
        """
        if text_on:
            ax.text(
                0.2,
                0.5,
                text,
                fontsize=7,
                color="black",
                ha="left",
                va="center",
            )

    clusters = {0: predictor.ML_saxs, 1: predictor.ML_waxs}

    for idx, cluster in clusters.items():
        x_test = cluster.X_test
        y_proba = cluster.model.predict_proba(x_test)
        y_score = y_proba[:, 1]
        y_true = cluster.y_test

        patients = cluster.df["patient_id"].unique().shape[0]
        measurements = cluster.df.shape[0]
        cancer_measurements = cluster.df["cancer_diagnosis"].sum()

        title = f"Cluster {idx}: {model_name}"
        ax = axes[idx]
        draw_roc(
            y_score,
            y_true,
            title,
            ax,
            patients,
            measurements,
            cancer_measurements,
            text_on,
            legend_on,
        )
    plt.show()


def metrics(tpr, fpr, thresholds, y_score, y_true, roc_auc):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    y_pred = y_score > optimal_threshold
    optimal_precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Round to 1 decimal place
    text = f"""
           ROC surface : {round(roc_auc * 100, 1)}%
           Optimal threshold: {round(optimal_threshold * 100, 1)}%
           Sensitivity: {round(optimal_sensitivity * 100, 1)}%
           Specificity: {round(optimal_specificity * 100, 1)}%
           PV: {round(optimal_precision * 100, 1)}%
           F1-score: {round(f1 * 100, 1)}%
           """
    return text


def viz_roc_balanced(fig, axes, model_name, estimators):
    for idxc, ax in zip([0, 1], axes):
        y_pred_prob = []
        fpr_l = []
        tpr_l = []
        thresholds_l = []
        roc_auc_l = []
        sen_l = []
        spec_l = []

        for _, est in enumerate(estimators):
            if idxc == 0:
                cluster = est.ML_saxs
            else:
                cluster = est.ML_waxs

            y_score = cluster.model.predict_proba(cluster.X_test)[:, 1]
            y_pred_prob.append(y_score)
            fpr, tpr, threshold = roc_curve(cluster.y_test, y_score)
            fpr_l.append(fpr)
            tpr_l.append(tpr)

            optimal_idx = np.argmax(tpr - fpr)
            optimal_sensitivity = tpr[optimal_idx]
            optimal_specificity = 1 - fpr[optimal_idx]
            sen_l.append(optimal_sensitivity)
            spec_l.append(optimal_specificity)
            thresholds_l.append(threshold)
            roc_auc_l.append(auc(fpr, tpr))

        min_idx = np.argmin(roc_auc_l)
        max_idx = np.argmax(roc_auc_l)

        tpr_t = []

        for fpr, tpr in zip(fpr_l, tpr_l):
            new_fpr = np.linspace(0, 1, 500)
            new_tpr = np.interp(new_fpr, fpr, tpr)
            tpr_t.append(new_tpr)
        tpra = np.mean(tpr_t, axis=0)
        fpra = new_fpr

        for i, fpr, tpr in zip(range(len(fpr_l)), fpr_l, tpr_l):
            # print(idxc, len(fpr_l))
            if i == min_idx:
                ax.plot(fpr, tpr, color="blue", linewidth=2)
            elif i == max_idx:
                ax.plot(fpr, tpr, color="red", linewidth=2)
            else:
                ax.plot(fpr, tpr, color="gray", alpha=0.1, linewidth=0.5)

        ax.plot(fpra, tpra, color="black", linewidth=2)

        if idxc == 0:
            min_ = estimators[min_idx].ML_saxs
            max_ = estimators[max_idx].ML_saxs
        else:
            min_ = estimators[min_idx].ML_waxs
            max_ = estimators[max_idx].ML_waxs

        text_min = metrics(
            tpr_l[min_idx],
            fpr_l[min_idx],
            thresholds_l[min_idx],
            y_pred_prob[min_idx],
            min_.y_test,
            roc_auc_l[min_idx],
        )
        text_max = metrics(
            tpr_l[max_idx],
            fpr_l[max_idx],
            thresholds_l[max_idx],
            y_pred_prob[max_idx],
            max_.y_test,
            roc_auc_l[max_idx],
        )

        text = f"""
        Min:
        {text_min}
        Max:
        {text_max}
        """

        ax.text(
            0.45, 0.3, text, fontsize=7, color="black", ha="left", va="center"
        )

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.plot(
            [0, 1.0], [0, 1.0], linestyle="--", color="gray"
        )  # , label='Random guess')
        # ax.legend(loc='lower right')

        if idxc == 0:
            ax.set_title("SAXS")
        elif idxc == 1:
            ax.set_title("WAXS")

    # Adjust layout
    plt.tight_layout()
    # plt.savefig('roc_all.png', dpi=400)
    plt.show()


def generate_roc_curve(y_true, y_score):
    RocCurveDisplay.from_predictions(y_true, y_score)
    fig = plt.gcf()
    plt.title("ROC Keele SAXS")
    fig.set_size_inches(4, 4)
    fig.set_dpi(150)
    fig.set_facecolor("white")
    # plt.savefig(f"analysis/fitting_classification/roc/keele_SAXS_ROC.png")
    plt.show()

    # Optimal threshold closest to perfect classifier
    tpr, fpr, optimal_idx, optimal_threshold = calculate_optimal_threshold(
        y_true, y_score
    )

    # Compute optimal sensitivity, specificity, and precision
    optimal_sensitivity = round(tpr[optimal_idx] * 100, 1)
    optimal_specificity = round((1 - fpr[optimal_idx]) * 100, 1)
    y_pred = y_score > optimal_threshold
    optimal_precision = round(precision_score(y_true, y_pred) * 100, 1)

    # Round to 1 decimal place
    print("Best performance closest to ideal classifier:")
    print(f"Sensitivity: {optimal_sensitivity}%")
    print(f"Specificity: {optimal_specificity}%")
    print(f"PPV: {optimal_precision}%")


def calculate_optimal_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold}")
    return tpr, fpr, optimal_idx, optimal_threshold
