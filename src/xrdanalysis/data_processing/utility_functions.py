"""Various utility functions used in different parts of codebase"""

import os
from typing import Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import zscore
from skimage.measure import label, regionprops
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve, precision_score, f1_score, RocCurveDisplay


from xrdanalysis.data_processing.containers import MLCluster


def get_center(data: np.ndarray, threshold=3.0) -> Tuple[float]:
    """
    Takes SAXS data and determines the center of the beam.

    Args:
        data (numpy.ndarray): The input SAXS data.
        threshold (float, optional): The threshold factor for identifying
            the center of the beam. Defaults to 3.0 times the average value
            of the input data.

    Returns:
        tuple: The coordinates of the center of the beam in the input data.
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

    if 'calibration_measurement_id' not in df.columns:
        df['calibration_measurement_id'] = [f"{i + 1}_fake" for i in range(len(df))]

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


def is_nan_pair(pair):
    """
    Check if the input is a tuple of two NaN values.

    Args:
        x (tuple or any): Input to check.

    Returns:
        bool: True if the input is a tuple of two NaN values, False otherwise.
    """
    if isinstance(pair, tuple) and len(pair) == 2:
        return all(np.isnan(x) for x in pair)
    return False


def interpolate_cluster(df, cluster_label, perc_min, perc_max, azimuth):
    """
    Interpolate a cluster of azimuthal integration data.

    Args:
        df (DataFrame): Input DataFrame containing data to be interpolated.
        cluster_label (int): Label of the cluster to be interpolated.
        perc_min (float): The minimum percentage of the maximum
            q-range for interpolation.
        perc_max (float): The maximum percentage of the maximum
            q-range for interpolation.
        azimuth (AzimuthalIntegration transformer): A transformer to
            use for azimuthal integration

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

    dfc = azimuth.transform(dfc)

    return MLCluster(df=dfc, q_cluster=cluster_label, q_range=dfc["q_range"])


def normalize_scale_cluster(
    cluster: MLCluster, normt='l1', norm=True, std=None, do_fit=True
):
    """
    Normalize and scale a cluster of azimuthal integration data.

    Args:
        cluster (MLCluster): The cluster to be normalized and scaled.
        std (StandardScaler or None, optional): StandardScaler instance
            for scaling. If None, a new instance will be created.
            Defaults to None.
        norm (Normalizer or None, optional): Normalizer instance for
            normalization. If None, a new instance will be created.
            Defaults to None.
        do_fit (bool, optional): Whether to fit the scaler. Defaults to True.
    """
    dfc = cluster.df.copy()
    if norm:
        norm = Normalizer(norm=normt)
        dfc["radial_profile_data_norm"] = dfc["radial_profile_data"].apply(
            lambda x: norm.transform([x])[0]
        )
    else:
        dfc["radial_profile_data_norm"] = dfc["radial_profile_data"]

    if not std:
        std = StandardScaler()

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
    Remove outliers from a DataFrame by clustering and z-score thresholding.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the data.
        z_score_threshold (float): The threshold for Z-score based
            outlier removal.
        direction (str, optional): The direction of outlier removal, either
            "both", "positive", or "negative". Defaults to "negative".
        num_clusters (int, optional): The number of clusters to use.
            Defaults to 4.

    Returns:
        pd.DataFrame: A DataFrame with outliers removed based
            on the specified criteria.
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


SCALED_DATA = 'radial_profile_data_norm_scaled'


def prep(df):
    """
    Prepare dataframe to RF learning or other models.
    It uses SCALED_DATA and age of the patient's measurement.
    """
    dfc = df.copy()
    transformed_data = np.vstack(dfc[SCALED_DATA].values)
    counts = dfc.groupby('patient_id').size()
    dfc['entry_count'] = dfc['patient_id'].map(counts)
    entry = dfc['entry_count'].values.reshape(-1, 1)
    if 'age' not in dfc.columns:
        dfc['age'] = [-1] * len(dfc)
    age = dfc['age'].values.reshape(-1, 1)
    return np.concatenate((transformed_data, entry, age), axis=1)


def custom_splitter_balanced(df, split):
    total_measurements = len(df)
    df_cancer = df[df['cancer_diagnosis']]
    df_non_cancer = df[~df['cancer_diagnosis']]
    cancer_fraction = df_cancer.shape[0] / total_measurements
    non_cancer_fraction = df_non_cancer.shape[0] / total_measurements
    train_size = int((1 - split) * total_measurements)
    train_cancer_size = int(train_size * cancer_fraction)
    train_non_cancer_size = int(train_size * non_cancer_fraction)
    patients_in_train_cancer = set()
    patients_in_train_non_cancer = set()
    unique_patient_ids_cancer = df_cancer['patient_id'].unique()
    unique_patient_ids_non_cancer = df_non_cancer['patient_id'].unique()

    np.random.shuffle(unique_patient_ids_cancer)
    np.random.shuffle(unique_patient_ids_non_cancer)

    train_data_cancer = 0
    train_data_non_cancer = 0
    # Iterate over each patient for cancer
    for patient_id in unique_patient_ids_cancer:
        # Get all rows corresponding to the current patient
        patient_rows = df_cancer[df_cancer['patient_id'] == patient_id]
        if train_data_cancer <= train_cancer_size:
            train_data_cancer += len(patient_rows)
            patients_in_train_cancer.add(patient_id)
        else:
            break
    for patient_id in unique_patient_ids_non_cancer:
        # Get all rows corresponding to the current patient
        patient_rows = df_non_cancer[df_non_cancer['patient_id'] == patient_id]
        if train_data_non_cancer <= train_non_cancer_size:
            train_data_non_cancer += len(patient_rows)
            patients_in_train_non_cancer.add(patient_id)
        else:
            break
    unique_patient_ids = df.patient_id.unique()
    patients_in_train = patients_in_train_cancer.union(patients_in_train_non_cancer)
    patients_in_test = set(unique_patient_ids) - patients_in_train
    train = df[df['patient_id'].isin(patients_in_train)].index
    train_idx = [df.index.get_loc(label) for label in train]
    test = df[df['patient_id'].isin(patients_in_test)].index
    test_idx = [df.index.get_loc(label) for label in test]
    return train_idx, test_idx


def random_forest_model_train(self: MLCluster,
                              n_estimators=100, max_depth=10,
                              random_state=32, split=0.4, shuffle=True,
                              split_func=None):
    """
    RF training function, intially developped for Keele Dataset
    """
    dfc = self.df.copy()
    y_data = dfc["cancer_diagnosis"].astype(int).values
    final = prep(dfc)

    # Split the transformed data into training and testing sets
    if split_func is None:
        x_train, x_test, y_train, y_test = train_test_split(final,
                                                            y_data,
                                                            test_size=split,
                                                            random_state=random_state,
                                                            shuffle=shuffle)
    else:
        train_idx, test_idx = custom_splitter_balanced(dfc, split)
        x_train = final[train_idx]
        y_train = y_data[train_idx]
        x_test = final[test_idx]
        y_test = y_data[test_idx]

    # Create and train the Random Forest classifier
    rf_params = {'n_estimators': n_estimators,
                 'max_depth': max_depth,
                 'min_samples_split': 2,
                 'min_samples_leaf': 1,
                 'bootstrap': True,
                 'random_state': random_state
                 }
    # Create RandomForestClassifier with specified parameters
    rf_classifier = RandomForestClassifier(**rf_params)
    rf_classifier.fit(x_train, y_train)

    # Make predictions on the test data
    y_pred = rf_classifier.predict(x_test)
    y_proba = rf_classifier.predict_proba(x_test)
    y_score = y_proba[:, 1]

    # Evaluate the accuracy and ROC_auc
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_score)

    self.model = rf_classifier
    self.accuracy = round(accuracy, 3)
    self.X_train = x_train
    self.y_train = y_train
    self.X_test = x_test
    self.y_test = y_test
    self.roc_auc = round(roc_auc, 3)


def viz_roc(fig, axes, model_name, predictor, text_on=True, legend_on=True):
    def draw_roc(y_score,
                 y_true,
                 title,
                 ax,
                 patients,
                 measurements,
                 cancer_measurements,
                 text_on=True, legend_on=True):

        display = RocCurveDisplay.from_predictions(y_true,
                                                   y_score,
                                                   ax=ax)
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
        y_pred = (y_score > optimal_threshold)
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
            ax.text(0.2, 0.5, text, fontsize=7, color='black', ha='left',
                    va='center')

    clusters = {0: predictor.ML_saxs, 1: predictor.ML_waxs}

    for idx, cluster in clusters.items():
        x_test = cluster.X_test
        y_proba = cluster.model.predict_proba(x_test)
        y_score = y_proba[:, 1]
        y_true = cluster.y_test

        patients = cluster.df["patient_id"].unique().shape[0]
        measurements = cluster.df.shape[0]
        cancer_measurements = cluster.df["cancer_diagnosis"].sum()

        title = f'Cluster {idx}: {model_name}'
        ax = axes[idx]
        draw_roc(y_score, y_true, title, ax, patients, measurements, cancer_measurements,
                 text_on, legend_on)
    plt.show()


def metrics(tpr, fpr, thresholds, y_score, y_true, roc_auc):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    y_pred = (y_score > optimal_threshold)
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

        for idx, est in enumerate(estimators):
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
            optimal_threshold = threshold[optimal_idx]
            optimal_sensitivity = tpr[optimal_idx]
            optimal_specificity = 1 - fpr[optimal_idx]
            sen_l.append(optimal_sensitivity)
            spec_l.append(optimal_specificity)
            thresholds_l.append(threshold)
            roc_auc_l.append(auc(fpr, tpr))

        min_idx = np.argmin(roc_auc_l)
        max_idx = np.argmax(roc_auc_l)

        fpr_t = []
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
                ax.plot(fpr, tpr, color='blue', linewidth=2)
            elif i == max_idx:
                ax.plot(fpr, tpr, color='red', linewidth=2)
            else:
                ax.plot(fpr, tpr, color='gray', alpha=0.1, linewidth=0.5)

        ax.plot(fpra, tpra, color='black', linewidth=2)

        if idxc == 0:
            min_ = estimators[min_idx].ML_saxs
            max_ = estimators[max_idx].ML_saxs
        else:
            min_ = estimators[min_idx].ML_waxs
            max_ = estimators[max_idx].ML_waxs

        text_min = metrics(tpr_l[min_idx], fpr_l[min_idx], thresholds_l[min_idx], y_pred_prob[min_idx], min_.y_test,
                           roc_auc_l[min_idx])
        text_max = metrics(tpr_l[max_idx], fpr_l[max_idx], thresholds_l[max_idx], y_pred_prob[max_idx], max_.y_test,
                           roc_auc_l[max_idx])
        text_a = ""  # metrics(tpra, fpra, thresholdsa, y_scorea, y)

        text = f"""
        Min: 
        {text_min}
        Max: 
        {text_max}
        """

        ax.text(0.45, 0.3, text, fontsize=7, color='black', ha='left', va='center')

        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.plot([0, 1.], [0, 1.], linestyle='--', color='gray')  # , label='Random guess')
        # ax.legend(loc='lower right')

        if idxc == 0:
            ax.set_title(f'SAXS')
        elif idxc == 1:
            ax.set_title(f'WAXS')

    # Adjust layout
    plt.tight_layout()
    # plt.savefig('roc_all.png', dpi=400)
    plt.show()
