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
    # Extract the first letter and all numbers in the filename before the subindex
    # E.g., from filename AB12345-01.txt -> A12345 is extracted
    # Note: Issue if the Barcode format changes
    extraction = df.index.str.extractall("CR_([A-Z]{1}).*?([0-9]+)")
    extraction_series = extraction[0] + extraction[1]
    extraction_list = extraction_series.tolist()

    assert(len(extraction_list) == df.shape[0])
    df["Barcode"] = extraction_list

    # Merge the databases
    df_ext = pd.merge(df, db, left_on="Barcode", right_index=True)


    ###################
    # Diagnostic Rule #
    ###################

    # If any patient has a measurement that is identified as abnormal,
    # set the Prediction column to 1

    # Get the list of patient Ids
    patient_id_list = df_ext["Patient_ID"].dropna().unique().tolist()

    # Create an empty patients dataframe
    df_patients = pd.DataFrame(columns={"Prediction"})

    # Fill the patients dataframe with predictions
    for patient_id in patient_id_list:
        patient_slice = df_ext[df_ext["Patient_ID"] == patient_id]
        if any(patient_slice["Abnormal"] == 1):
            df_patients.loc[patient_id] = 1
        else:
            df_patients.loc[patient_id] = 0

    # Get patients and associated diagnosis
    db_patients = db[["Patient_ID", "Diagnosis"]].drop_duplicates()
    # Merge to get patient diagnosis and prediction in the same dataframe
    df_patients_ext = pd.merge(
            df_patients, db_patients, left_index=True, right_on="Patient_ID")
    # Set patient id as index
    df_patients_ext.index = df_patients_ext["Patient_ID"]
    # Extract diagnosis and prediction
    df_patients_ext = df_patients_ext[["Diagnosis", "Prediction"]]

    # Calculate true positives, false positives, true negatives,
    # and false negatives
    TP = (
            (df_patients_ext["Diagnosis"] == "cancer") & \
                    (df_patients_ext["Prediction"] == 1)).sum()
    FP = (
            (df_patients_ext["Diagnosis"] == "healthy") & \
                    (df_patients_ext["Prediction"] == 1)).sum()
    TN = (
            (df_patients_ext["Diagnosis"] == "healthy") & \
                    (df_patients_ext["Prediction"] == 0)).sum()
    FN = (
            (df_patients_ext["Diagnosis"] == "cancer") & \
                    (df_patients_ext["Prediction"] == 0)).sum()

    # Save the results
    if output_filepath:
        df_patients_ext.to_csv(output_filepath)

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
