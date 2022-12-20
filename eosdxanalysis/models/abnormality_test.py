"""
Code to train abnormality test parameters
"""
import os
import glob
import shutil

import argparse

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

from eosdxanalysis.models.utils import metrics_report
from eosdxanalysis.preprocessing.image_processing import bright_pixel_count

def image_brightness(
        masked_image, pixel_brightness_threshold=0.75,
        image_area=36220, image_mask=None):
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

    image_area : int
        The number of pixels in the image to analyze.  Default value is
        ``36220`` which corresponds to ``rmin=24``, and ``rmax=110``.

    image_mask : array-like
        Boolean mask defining areas of the image to analyze.

    Returns
    -------

    brightness : float
        Relative brightness from 0 to 1.
    """
    # If an image mask is provided, set areas outside the mask to zero
    if type(image_mask) is np.ndarray:
        image = masked_image.copy()
        image[~image_mask] = 0
    # Otherwise use the provided image
    else:
        image = masked_image

    # Calculate the bright pixel count
    bright_pixels = bright_pixel_count(
            image, qmin=pixel_brightness_threshold)

    brightness = bright_pixels/image_area
    return brightness

def image_brightness_batch(
        patients_db=None, source_data_path=None,
        pixel_brightness_threshold=None, image_area=36220, image_mask=None):
    """
    Calculates relative image brightness for a dataset.

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

    image_area : int
        The number of pixels in the image to analyze.  Default value is
        ``36220`` which corresponds to ``rmin=24``, and ``rmax=110``.

    image_mask : array-like
        Boolean mask defining areas of the image to analyze.

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

    df = pd.DataFrame(columns={"Image_Brightness"})

    # Run abnormality test for all data files
    for filepath in filepath_list[:]:
        # Extract the filename
        filename = os.path.basename(filepath)
        # Load the masked image
        masked_image = np.loadtxt(filepath)
        # Run abnormality test
        brightness = image_brightness(
                masked_image,
                pixel_brightness_threshold=pixel_brightness_threshold,
                image_area=image_area,
                image_mask=image_mask,
                )
        # Add row to dataframe
        df.loc[filename] = brightness

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
    df_patients = pd.DataFrame(columns={"Max_Image_Brightness"})

    # Fill the patients dataframe with the maximum image brightness seen
    for patient_id in patient_id_list:
        patient_slice = df_ext[df_ext["Patient_ID"] == patient_id]
        max_image_brightness = patient_slice["Image_Brightness"].max()
        df_patients.loc[patient_id] = max_image_brightness

    # Get patients and associated diagnosis
    db_patients = db[["Patient_ID", "Diagnosis"]].drop_duplicates()
    # Merge to get patient diagnosis and prediction in the same dataframe
    df_patients_ext = pd.merge(
            df_patients, db_patients, left_index=True, right_on="Patient_ID")
    # Set patient id as index
    df_patients_ext.index = df_patients_ext["Patient_ID"]
    # Extract diagnosis and prediction
    df_patients_ext = df_patients_ext[["Diagnosis", "Max_Image_Brightness"]]

    # Prepare outputs for ROC analysis
    y_true = df_patients_ext["Diagnosis"].replace("cancer",1).replace("healthy",0).values
    y_score = df_patients_ext["Max_Image_Brightness"].values

    return y_true, y_score

def abnormality_test(
        masked_image, pixel_brightness_threshold=0.75,
        image_brightness_threshold=0.1, image_area=36220, image_mask=None):
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

    image_mask : array-like
        Boolean mask defining areas of the image to analyze.

    Returns
    -------

    abnormal : bool
        ``True`` if image is abnormally bright.
    """

    brightness = image_brightness(
                masked_image,
                pixel_brightness_threshold=pixel_brightness_threshold,
                image_area=image_area,
                image_mask=image_mask,
                )

    abnormal = brightness > image_brightness_threshold
    return abnormal

def abnormality_test_batch(
        patients_db=None, source_data_path=None,
        patient_predictions_filepath=None,
        measurement_predictions_filepath=None,
        pixel_brightness_threshold=None, image_brightness_threshold=None,
        image_area=36220, image_mask=None):
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

    patient_predictions_filepath : str
        Path to store patient predictions csv file.

    measurement_predictions_filepath : str
        Path to store measurement predictions csv file.

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

    image_mask : array-like
        Boolean mask defining areas of the image to analyze.

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
                image_area=image_area,
                image_mask=image_mask,
                )
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
    # Replace cancer diagnosis with 1, healthy diagnosis with 0
    df_patients_ext.loc[df_patients_ext["Diagnosis"] == "cancer",
            "Diagnosis"] = 1
    df_patients_ext.loc[df_patients_ext["Diagnosis"] == "healthy",
            "Diagnosis"] = 0

    # Calculate true positives, false positives, true negatives,
    # and false negatives
    df_patients_with_diagnosis = df_patients_ext[~df_patients_ext["Diagnosis"].isna()]
    y_true = df_patients_with_diagnosis["Diagnosis"].astype(int)
    y_pred = df_patients_with_diagnosis["Prediction"].astype(int)

    TN, FP, FN, TP = confusion_matrix(y_true.values, y_pred.values).ravel()

    # Save the results
    if patient_predictions_filepath:
        df_patients_ext.to_csv(patient_predictions_filepath)
    if measurement_predictions_filepath:
        df_ext.to_csv(measurement_predictions_filepath)

    return TN, FP, FN, TP


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
            "--patient_predictions_filepath", default=None, required=True,
            help="The path to store patient predictions csv file.")
    parser.add_argument(
            "--measurement_predictions_filepath", default=None, required=True,
            help="The path to store measurement predictions csv file.")
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
            "--image_mask", default=36220, required=True,
            help="Mask defining image areas to analyze.")
    parser.add_argument(
            "--print", action="store_true", help="Flag to print metrics.")

    # Collect arguments
    args = parser.parse_args()

    patients_db = args.patients_db
    source_data_path = args.source_data_path
    measurement_predictions_filepath = args.measurement_predictions_filepath
    patient_predictions_filepath = args.patient_predictions_filepath
    pixel_brightness_threshold = np.float64(args.pixel_brightness_threshold)
    image_brightness_threshold = np.float64(args.image_brightness_threshold)
    image_area = np.uint32(args.image_area)
    print_metrics = args.print
    image_mask = np.loadtxt(args.image_mask, dtype=bool)

    # Run abnormality test on a dataset
    TP, FP, TN, FN = abnormality_test_batch(
            patients_db=patients_db, source_data_path=source_data_path,
            patient_predictions_filepath=patient_predictions_filepath,
            measurement_predictions_filepath=measurement_predictions_filepath,
            pixel_brightness_threshold=pixel_brightness_threshold,
            image_brightness_threshold=image_brightness_threshold,
            image_area=image_area, image_mask=image_mask,
            )

    # Calculate performance metrics using per-patient predictions
    metrics_report(TP=TP, FP=FP, TN=TN, FN=FN)
