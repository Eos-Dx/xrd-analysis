"""Private ROC evaluation helpers behind public compatibility wrappers."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, auc, f1_score, precision_score, roc_curve


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
    patients_in_train = patients_in_train_cancer.union(patients_in_train_non_cancer)
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
        y_pred = y_score >= optimal_threshold
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
    y_pred = y_score >= optimal_threshold
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


def viz_roc_balanced(fig, axes, model_name, estimators):  # noqa: C901
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

        ax.text(0.45, 0.3, text, fontsize=7, color="black", ha="left", va="center")

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


def generate_roc_based_metrics(
    y_true, y_score, show_flag=True, min_sensitivity=None, min_specificity=None
):
    """Generate ROC-based classification metrics.

    Parameters
    ----------
    y_true
        True binary labels.
    y_score
        Positive-class scores or probability estimates.
    show_flag
        Display the ROC curve when true.
    min_sensitivity
        Optional minimum sensitivity used to filter thresholds.
    min_specificity
        Optional minimum specificity used to filter thresholds.

    Returns
    -------
    tuple
        Optimal sensitivity, specificity, precision, balanced accuracy
        (percent), and threshold.
    """
    if show_flag:
        RocCurveDisplay.from_predictions(y_true, y_score)
        fig = plt.gcf()
        plt.title("ROC Curve")
        fig.set_size_inches(4, 4)
        fig.set_dpi(150)
        fig.set_facecolor("white")
        # plt.savefig(f"analysis/fitting_classification/roc/keele_SAXS_ROC.png")
        plt.show()

    # Optimal threshold closest to perfect classifier
    tpr, fpr, optimal_idx, optimal_threshold = calculate_optimal_threshold(
        y_true,
        y_score,
        print_flag=False,
        min_sensitivity=min_sensitivity,
        min_specificity=min_specificity,
    )

    # Compute optimal sensitivity, specificity, and precision
    optimal_sensitivity = round(tpr[optimal_idx] * 100, 1)
    optimal_specificity = round((1 - fpr[optimal_idx]) * 100, 1)
    y_pred = y_score >= optimal_threshold
    optimal_precision = round(precision_score(y_true, y_pred) * 100, 1)

    # Calculate balanced accuracy
    balanced_accuracy = (tpr[optimal_idx] + 1 - fpr[optimal_idx]) / 2
    balanced_accuracy = round(balanced_accuracy * 100, 1)

    return (
        optimal_sensitivity,
        optimal_specificity,
        optimal_precision,
        balanced_accuracy,
        optimal_threshold,
    )


def calculate_optimal_threshold(
    y_true,
    y_score,
    min_sensitivity=None,
    min_specificity=None,
    print_flag=False,
):
    """Calculate the optimal binary ROC threshold.

    Parameters
    ----------
    y_true
        True binary labels.
    y_score
        Positive-class scores or probability estimates.
    min_sensitivity
        Optional minimum sensitivity used to filter thresholds.
    min_specificity
        Optional minimum specificity used to filter thresholds.
    print_flag
        Print the selected threshold when true.

    Returns
    -------
    tuple
        True-positive rates, false-positive rates, selected index, and threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # ROC thresholds use score >= threshold. Apply every requested constraint
    # to the same candidate set so persisted decisions honor reported metrics.
    threshold_mask = np.ones(thresholds.shape, dtype=bool)
    if min_sensitivity is not None:
        threshold_mask &= tpr >= min_sensitivity
    if min_specificity is not None:
        threshold_mask &= (1 - fpr) >= min_specificity
    fpr, tpr, thresholds = (
        fpr[threshold_mask],
        tpr[threshold_mask],
        thresholds[threshold_mask],
    )

    # Find the optimal index and threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    if print_flag:
        print(f"Optimal threshold: {optimal_threshold}")

    return tpr, fpr, optimal_idx, optimal_threshold
