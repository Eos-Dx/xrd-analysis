"""Private grouped-array statistics and plotting implementation."""

from __future__ import annotations

import numpy as np


def compute_group_statistics(df, label_column, array_column):
    """Return per-label mean and standard-deviation arrays."""
    grouped = df.groupby(label_column)[array_column]
    result = grouped.apply(
        lambda arrays: (
            np.mean(np.stack(arrays), axis=0),
            np.std(np.stack(arrays), axis=0),
        )
    )
    result_df = result.reset_index(name="stats")
    result_df["mean"] = result_df["stats"].map(lambda value: value[0])
    result_df["std"] = result_df["stats"].map(lambda value: value[1])
    return result_df.drop(columns=["stats"])


def plot_group_statistics(
    df,
    label_column,
    array_column,
    selected_labels=None,
    *,
    compute_group_statistics,
    plt,
):
    """Plot raw profiles with the public grouped-statistics calculation."""
    stats_df = compute_group_statistics(df, label_column, array_column)
    if selected_labels is None:
        selected_labels = stats_df[label_column].unique()
    else:
        selected_labels = [
            label for label in selected_labels if label in stats_df[label_column].values
        ]

    for label in selected_labels:
        group_data = df[df[label_column] == label][array_column]
        group_stats = stats_df[stats_df[label_column] == label]
        mean = group_stats["mean"].values[0]
        std = group_stats["std"].values[0]

        plt.figure(figsize=(8, 5))
        for array in group_data:
            plt.plot(array, color="royalblue", alpha=0.7, label="_nolegend_")
        plt.plot(mean, label=f"{label} Mean", color="blue", linewidth=2)
        plt.fill_between(
            range(len(mean)),
            mean - std,
            mean + std,
            color="blue",
            alpha=0.3,
            label=f"{label} Std Dev",
        )
        plt.title(f"{label} Group")
        plt.xlabel("q range (nm^-1)")
        plt.ylabel("Intensity (a.u.)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
