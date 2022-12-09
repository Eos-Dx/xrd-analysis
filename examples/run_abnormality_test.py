"""
Runs abnormality test to find optimal parameters
"""
import numpy as np
import pandas as pd

from eosdxanalysis.models.abnormality_test import abnormality_test_batch
from eosdxanalysis.models.utils import metrics_report

# Path to patients database file
patients_db = "patients_database.csv"
# Path to preprocessed data files
source_data_path = "dataset"
# Set pixel brightness treshold range
pixel_threshold_range = np.arange(0.5, 1.0, step=0.1)
# Set image brightness threshold range
image_threshold_range = np.arange(0.00, 0.5, step=0.1)
# Set image area
image_area = 36220
# Set output file
output_filepath = "prediction_table.csv"


def run_abnormality_test(
        patients_db, source_data_path, pixel_threshold_range,
        image_threshold_range, print_metrics=True, return_results=False,
        image_area=36220):
    # Set up results array
    if return_results:
        results_array = np.zeros(
                (pixel_threshold_range.size, image_threshold_range.size),
                    dtype=(object,6))
    # Loop over pixel brightness threshold values
    for idx in range(pixel_threshold_range.size):
        # Loop over image brightness threshold values
        for jdx in range(image_threshold_range.size):
            pixel_brightness_threshold = pixel_threshold_range[idx]
            image_brightness_threshold = image_threshold_range[jdx]
            print(
                "Running abnormality test batch for bright pixel threshold {:0.2f}"
                    " and bright image threshold {:0.2f}".format(
                    pixel_brightness_threshold, image_brightness_threshold))
            predictions = abnormality_test_batch(
                    patients_db=patients_db,
                    source_data_path=source_data_path,
                    pixel_brightness_threshold=pixel_brightness_threshold,
                    image_brightness_threshold=image_brightness_threshold,
                    image_area=image_area)
            if return_results:
                # Store parameter values and predictions
                result = (pixel_brightness_threshold, image_brightness_threshold) + \
                        predictions
                results_array[idx][jdx] = result
            if print_metrics:
                    print(
                        "Running metrics report for bright pixel threshold {:0.2f}"
                        " and bright image threshold {:0.2f}".format(
                            pixel_brightness_threshold, image_brightness_threshold))
                    TP, FP, TN, FN = predictions
                    metrics_report(TP=TP, FP=FP, TN=TN, FN=FN)

    if return_results:
        return results_array

# Generate results
results = run_abnormality_test(
        patients_db, source_data_path, pixel_threshold_range,
        image_threshold_range, print_metrics=False, return_results=True,
        image_area=36220)

# Collect results into dataframe
columns = [
        "pixel_brightness_threshold",
        "image_brightness_threshold",
        "TP",
        "FP",
        "TN",
        "FN",
        ]
df = pd.DataFrame(data=results.reshape(-1,results.shape[-1]), columns=columns)

# Calculate metrics

# Calculate accuracy
df["accuracy"] = (df["TP"] + df["TN"]) / (df["TP"] + df["TN"] + df["FP"] + df["FN"]) 
# Calculate false positive rate
df["FPR"] = df["FP"] / (df["FP"] + df["TN"]) 
# Calculate false negative rate
df["FNR"] = df["FN"] / (df["FN"] + df["TP"]) 
# Calculate precision
df["precision"] = df["TP"] / (df["TP"] + df["FP"])
# Calculate recall (sensitivity)
df["sensitivity"] = df["TP"] / (df["TP"] + df["FN"]) 
# Specificity
df["specificity"] = df["TN"] / (df["TN"] + df["FP"]) 
# Calculate F1 score
df["F1"] = 2 * df["precision"] * df["sensitivity"] / (df["precision"] + df["sensitivity"])

# Save data to file
df.to_csv(output_filepath, index=False)
