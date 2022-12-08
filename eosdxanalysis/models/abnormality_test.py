"""
Code to train abnormality test parameters
"""
import os
import glob
import shutil

import argparse

import pandas as pd
import numpy as np

from eosdxanalysis.models.utils import metrics_report
from eosdxanalysis.preprocessing.image_processing import bright_pixel_count


def abnormality_test(
        masked_image, pixel_brightness_threshold=0.75,
        image_brightness_threshold=0.1, image_area=36220):
    """
    Predicts abnormality of an image based on brightness.

    Parameters
    ----------

    masked_image : ndarray
        X-ray diffraction measurement data file that has been preprocessed
        (centered, rotated, and removal of the beam and outer areas).

    pixel_brightness_threshold : float
        Relative value to define a bright pixel. The lowest intensity pixel is
        is mapped to ``0``, and the highest intensity pixel is mapped to ``1``.
        All pixels that are above the relative threshold are considered bright
        pixels.

    image_brightness_threshold : float
        Relative value to define a bright image. If the ratio of bright pixels
        to the image area is greater than the relative threshold, the image is
        considered bright and labeled abnormal.

    image_area : int
        The number of pixels in the image to analyze.  Default value is
        ``36220`` which corresponds to ``rmin=24``, and ``rmax=110``.

    Returns
    -------

    abnormal : bool
        ``True`` if image is abnormally bright.
    """

    bright_pixels = bright_pixel_count(
            masked_image, qmin=pixel_brightness_threshold)

    abnormal = bright_pixels/image_area > image_brightness_threshold
    return abnormal

def abnormality_test_batch(
        patients_db=None, source_data_path=None, output_filepath=None,
        pixel_brightness_threshold=None, image_brightness_threshold=None,
        image_area=36220):
    """
    Predicts abnormality of a dataset based on image brightness.

    Parameters
    ----------

    patients_db : str
        Path to csv file containing two columns, ``Barcode``, and ``Diagnosis``,
        where ``Barcode`` is a unique patient ID, and ``Diagnosis`` is either
        ``healthy`` or ``cancer``.

    source_data_path : str
        Path to preprocessed measurement files. Files should be centered,
        rotated with beam and outer areas removed.

    pixel_brightness_threshold : float
        Relative value to define a bright pixel. The lowest intensity pixel is
        is mapped to ``0``, and the highest intensity pixel is mapped to ``1``.
        All pixels that are above the relative threshold are considered bright
        pixels.

    image_brightness_threshold : float
        Relative value to define a bright image. If the ratio of bright pixels
        to the image area is greater than the relative threshold, the image is
        considered bright and labeled abnormal.

    image_area : int
        The number of pixels in the image to analyze.  Default value is
        ``36220`` which corresponds to ``rmin=24``, and ``rmax=110``.

    Returns
    -------

    TP, FP, TN, FN : 4-tuple (int)
        True positive, false positive, true negative, and false negative
        counts.

    """
    # Load patient database file
    db = pd.read_csv(patients_db, index_col="Barcode")

    # Create a dataframe corresponding to input data filenames
    filepath_list = glob.glob(os.path.join(source_data_path, "*.txt"))
    filepath_list.sort()

    df = pd.DataFrame(columns={"Abnormal"})

    # Run abnormality test for all data files
    for filepath in filepath_list[:]:
        # Extract the filename
        filename = os.path.basename(filepath)
        # Load the masked image
        masked_image = np.loadtxt(filepath)
        # Run abnormality test
        abnormal = abnormality_test(
                masked_image,
                pixel_brightness_threshold=pixel_brightness_threshold,
                image_brightness_threshold=image_brightness_threshold,
                image_area=image_area)
        # Add row to dataframe
        df.loc[filename] = int(abnormal)

    # Add a Barcode column to the dataframe
    # Note: Issue if the Barcode format changes
    extraction = df.index.str.extractall("CR_([A-Z]{1}).*?([0-9]+)")
    extraction_series = extraction[0] + extraction[1]
    extraction_list = extraction_series.tolist()

    assert(len(extraction_list) == df.shape[0])
    df["Barcode"] = extraction_list

    db_healthy = db[db["Diagnosis"] == "healthy"]
    db_cancer = db[db["Diagnosis"] == "cancer"]

    healthy_barcode_list = db_healthy.index.tolist()
    cancer_barcode_list = db_cancer.index.tolist()

    df_healthy = df[df["Barcode"].isin(healthy_barcode_list)]
    df_cancer = df[df["Barcode"].isin(cancer_barcode_list)]

    ###################
    # Diagnostic Rule #
    ###################

    # Note: This part could be refactored, exact same code is repeated twice

    # If any patient has a measurement that is identified as abnormal,
    # set the Prediction column to 1

    # Get active cancer barcode list
    active_cancer_barcode_list = \
            df_cancer[df_cancer["Barcode"].isin(
                cancer_barcode_list)]["Barcode"].unique().tolist()
    df_cancer_patients = pd.DataFrame(data=active_cancer_barcode_list, columns=["Barcode"])

    # Make predictions
    cancer_prediction_list = []
    for cancer_barcode in active_cancer_barcode_list:
        cancer_barcode_prediction_list = df_cancer[df_cancer["Barcode"] == cancer_barcode]["Abnormal"].tolist()
        if any(cancer_barcode_prediction_list):
            cancer_prediction_list.append(1)
        else:
            cancer_prediction_list.append(0)
            
    # Store predictions
    df_cancer_patients["Prediction"] = cancer_prediction_list

    # Get active healthy barcode list
    active_healthy_barcode_list = \
            df_healthy[df_healthy["Barcode"].isin(
                healthy_barcode_list)]["Barcode"].unique().tolist()
    df_healthy_patients = pd.DataFrame(data=active_healthy_barcode_list, columns=["Barcode"])

    # Make predictions
    healthy_prediction_list = []
    for healthy_barcode in active_healthy_barcode_list:
        healthy_barcode_prediction_list = df_healthy[df_healthy["Barcode"] == healthy_barcode]["Abnormal"].tolist()
        if any(healthy_barcode_prediction_list):
            healthy_prediction_list.append(1)
        else:
            healthy_prediction_list.append(0)

    # Store predictions
    df_healthy_patients["Prediction"] = healthy_prediction_list

    # Calculate true positives, false positives, true negatives,
    # and false negatives
    TP = (df_cancer_patients["Prediction"] == 1).sum()
    FP = (df_healthy_patients["Prediction"] == 1).sum()
    TN = (df_healthy_patients["Prediction"] == 0).sum()
    FN = (df_cancer_patients["Prediction"] == 0).sum()

    # Save the results
    if output_filepath:
        df_healthy_patients["Diagnosis"] = "healthy"
        df_cancer_patients["Diagnosis"] = "cancer"
        df_patients = pd.concat([df_healthy_patients, df_cancer_patients])
        # Re-order columns
        df_patients = df_patients.reindex(columns=["Barcode", "Diagnosis", "Prediction"])
        df_patients.to_csv(output_filepath, index=False)

    return TP, FP, TN, FN


if __name__ == '__main__':
    """
    Performs abnormality detection on a dataset per patient.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser()
    # Set up parser arguments
    parser.add_argument(
            "--patients_db", default=None, required=True,
            help="The patients database file containing diagnosis.")
    parser.add_argument(
            "--source_data_path", default=None, required=True,
            help="The path to preprocessed data files.")
    parser.add_argument(
            "--output_filepath", default=None, required=True,
            help="The path to store patient predictions csv file.")
    parser.add_argument(
            "--pixel_brightness_threshold", default=0.75, required=True,
            help="Relative threshold level to define a bright pixel.")
    parser.add_argument(
            "--image_brightness_threshold", default=0.1, required=True,
            help="Relative threshold to define a bright image.")
    parser.add_argument(
            "--image_area", default=36220, required=False,
            help="Number of pixels in image to analyze.")
    parser.add_argument(
            "--print", action="store_true", help="Flag to print metrics.")

    # Collect arguments
    args = parser.parse_args()

    patients_db = args.patients_db
    source_data_path = args.source_data_path
    output_filepath = args.output_filepath
    pixel_brightness_threshold = np.float64(args.pixel_brightness_threshold)
    image_brightness_threshold = np.float64(args.image_brightness_threshold)
    image_area = np.uint32(args.image_area)
    print_metrics = args.print

    # Run abnormality test on a dataset
    TP, FP, TN, FN = abnormality_test_batch(
            patients_db=patients_db, source_data_path=source_data_path,
            output_filepath=output_filepath,
            pixel_brightness_threshold=pixel_brightness_threshold,
            image_brightness_threshold=image_brightness_threshold,
            image_area=image_area,
            )

    # Calculate performance metrics using per-patient predictions
    metrics_report(TP=TP, FP=FP, TN=TN, FN=FN)
