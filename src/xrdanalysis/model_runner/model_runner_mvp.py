import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, simpledialog
from typing import Dict, List

import h5py
import joblib
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import Normalizer, StandardScaler
from tqdm import tqdm

from xrdanalysis.data_processing.containers import RandomForestClassifier

SCALED_DATA = "radial_profile_data_norm_scaled"


def restore_df_from_hdf5(hdf5_files):
    dataframes = []

    for file in tqdm(hdf5_files, desc="Processing HDF5 files"):
        with h5py.File(file, "r") as h5file:
            patient_id = file.stem  # Extract patient_id from the filename

            for measure_type in ["SAXS", "WAXS"]:
                if measure_type in h5file:
                    grp = h5file[measure_type]

                    # Extract 'data' dataset
                    data_stack = grp["data"][:]
                    N = data_stack.shape[0]  # Number of entries

                    # Extract metadata attributes
                    meta_grp = grp["meta"]
                    cancer_diagnosis = meta_grp.attrs["cancer_diagnosis"]
                    age = meta_grp.attrs["age"]
                    cancer_type = meta_grp.attrs["cancer_type"]
                    grade = meta_grp.attrs["grade"]
                    cohort = meta_grp.attrs["cohort"]

                    # Extract remaining datasets
                    calibration_meas_id = meta_grp[
                        "calibration_measurement_id"
                    ][:]
                    wavelength = meta_grp["wavelength"][:]
                    pixel_size = meta_grp["pixel_size"][:]
                    measurement_orig_file_name = meta_grp[
                        "measurement_orig_file_name"
                    ][:]
                    measurement_date = meta_grp["measurement_date"][:]
                    measurement_date = pd.to_datetime(
                        [date.decode("utf-8") for date in measurement_date]
                    )
                    ponifile = meta_grp["ponifile"][:]

                    # Ensure the length of the extracted metadata matches N
                    if len(wavelength) != N or len(pixel_size) != N:
                        raise ValueError(
                            f"""Mismatch in dimensions for
                            patient {patient_id} in {measure_type}"""
                        )

                    # Construct a DataFrame for the current measure type
                    df = pd.DataFrame(
                        {
                            "patient_id": [patient_id] * N,
                            "calibration_measurement_id": calibration_meas_id,
                            "measurement_data": list(data_stack),
                            "cancer_diagnosis": [cancer_diagnosis] * N,
                            "age": [age] * N,
                            "cancer_type": [cancer_type] * N,
                            "grade": [grade] * N,
                            "cohort": [cohort] * N,
                            "wavelength": wavelength,
                            "pixel_size": pixel_size,
                            "measurement_orig_file_name": [
                                file.decode("utf-8")
                                for file in measurement_orig_file_name
                            ],
                            "measurement_date": measurement_date,
                            "ponifile": [
                                poni.decode("utf-8") for poni in ponifile
                            ],
                            "type_measurement": [measure_type] * N,
                        }
                    )

                    dataframes.append(df)

    # Concatenate all DataFrames into one
    return pd.concat(dataframes, ignore_index=True)


class PredictorRF(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        saxs_models: List[RandomForestClassifier],
        waxs_models: List[RandomForestClassifier],
        main_models: List[RandomForestClassifier],
    ):
        self.saxs_models = saxs_models
        self.waxs_models = waxs_models
        self.main_models = main_models

    @staticmethod
    def flatten(mixed_list):
        flat_list = []
        for item in mixed_list:
            if isinstance(item, np.ndarray):
                flat_list.extend(item)
            elif isinstance(item, list):
                flat_list.extend(PredictorRF.flatten(item))
            else:
                flat_list.append(item)
        return flat_list

    @staticmethod
    def calc_dist(points):
        if points.shape[0] > 1:
            dist_matrix = distance_matrix(points, points)
            # Get the upper triangle of the distance matrix
            # without the diagonal
            upper_triangle = dist_matrix[np.triu_indices(len(points), k=1)]
            m = np.mean(upper_triangle)
        else:
            m = -1
        return m

    def fit(self, X, y):
        # In a more complex model, you would train the model here
        self.mean_class_ = np.mean(y)  # Store the mean of the target
        return self

    def predict(self, df):
        pass

    def predict_proba(self, df):
        patients = df["patient_id"].unique()
        patients_res = []
        for cl_saxs, cl_waxs, cl_main in zip(
            self.saxs_models, self.waxs_models, self.main_models
        ):
            results = []
            for pat_id in patients:
                saxs = df[
                    (df["patient_id"] == pat_id)
                    & (df["type_measurement"] == "SAXS")
                ]
                waxs = df[
                    (df["patient_id"] == pat_id)
                    & (df["type_measurement"] == "WAXS")
                ]
                age = -1
                try:
                    age = int(saxs["age"].iloc[0])
                except Exception:
                    try:
                        age = int(waxs["age"].iloc[0])
                    except Exception:
                        pass

                saxs_pred = [-1, -1]
                waxs_pred = [-1, -1]
                # Ensure transformed data variables are lists
                # or empty lists if None
                dist_saxs = -1
                dist_waxs = -1

                if saxs.shape[0] > 0:
                    transformed_data_saxs = np.vstack(saxs[SCALED_DATA].values)
                    saxs_pred = cl_saxs.predict_proba(transformed_data_saxs)
                    saxs_pred = np.mean(saxs_pred, axis=0)
                    dist_saxs = PredictorRF.calc_dist(transformed_data_saxs)
                if waxs.shape[0] > 0:
                    transformed_data_waxs = np.vstack(waxs[SCALED_DATA].values)
                    waxs_pred = cl_waxs.predict_proba(transformed_data_waxs)
                    waxs_pred = np.mean(waxs_pred, axis=0)
                    dist_waxs = PredictorRF.calc_dist(transformed_data_waxs)

                # Create the list to flatten
                elements_to_flatten = [
                    saxs_pred,
                    waxs_pred,
                    saxs.shape[0],
                    waxs.shape[0],
                    dist_saxs,
                    dist_waxs,
                    age,
                ]

                # Flatten the list
                res = PredictorRF.flatten(elements_to_flatten)
                results.append(res)
            patients_res.append(cl_main.predict_proba(results))
        return np.mean(patients_res, axis=0)


class NormScaler(TransformerMixin):

    def __init__(self, scalers: Dict[str, StandardScaler]):
        self.scalers = scalers

    def fit(self, x: pd.DataFrame, y=None):
        _ = x
        _ = y
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc = df.copy()
        norm = Normalizer("l1")
        dfc["radial_profile_data_norm"] = dfc["radial_profile_data"].apply(
            lambda x: norm.transform([x])[0]
        )

        df_saxs = dfc[dfc["type_measurement"] == "SAXS"].copy()
        df_waxs = dfc[dfc["type_measurement"] == "WAXS"].copy()

        # Apply the scaler for SAXS data
        if not df_saxs.empty:
            matrix_2d_saxs = np.vstack(
                df_saxs["radial_profile_data_norm"].values
            )
            scaled_data_saxs = self.scalers["SAXS"].transform(matrix_2d_saxs)
            df_saxs["radial_profile_data_norm_scaled"] = [
                arr for arr in scaled_data_saxs
            ]

        # Apply the scaler for WAXS data
        if not df_waxs.empty:
            matrix_2d_waxs = np.vstack(
                df_waxs["radial_profile_data_norm"].values
            )
            scaled_data_waxs = self.scalers["WAXS"].transform(matrix_2d_waxs)
            df_waxs["radial_profile_data_norm_scaled"] = [
                arr for arr in scaled_data_waxs
            ]

        # Combine the processed DataFrames back into one
        dfc_processed = pd.concat([df_saxs, df_waxs], ignore_index=True)
        return dfc_processed


class ModelRunner:
    def __init__(self, root: tk.Tk):
        self.root = root
        # PATH TO MODEL, WILL BE A FILE CHOICE OR A DROPDOWN IN THE FUTURE
        self.model = joblib.load(
            "/home/confucii/EOSDX/Repos/tests/Keele playground_SD/pipeline.pkl"
        )
        self.setup_ui()

    def browse(self):
        choice = simpledialog.askstring(
            "Input",
            "Type 'file' to select a file or 'folder' to select a folder:",
        )

        if choice.lower() == "file":
            self.hdf5_path = filedialog.askopenfilename()
            if self.hdf5_path:
                self.file_path_label.config(text=f"File: {self.hdf5_path}")
        elif choice.lower() == "folder":
            self.hdf5_path = filedialog.askdirectory()
            if self.hdf5_path:
                self.file_path_label.config(text=f"Folder: {self.hdf5_path}")
        else:
            tk.messagebox.showerror(
                "Error", "Invalid choice. Please type 'file' or 'folder'."
            )

    def process_h5(self):
        try:
            paths_to_process = []
            if os.path.isfile(self.hdf5_path):
                paths_to_process.append(Path(self.hdf5_path))
            elif os.path.isdir(self.hdf5_path):
                paths_to_process = list(Path(self.hdf5_path).glob("*.h5"))

            if not paths_to_process:
                raise ValueError("No HDF5 files found to process.")

            for path in paths_to_process:
                df = restore_df_from_hdf5([path])
                transformed_data = self.model[:-1].transform(df)
                predictions = self.model.named_steps[
                    "predictor"
                ].predict_proba(transformed_data)
                print(f"Predictions for {path}: {predictions}")

        except Exception as e:
            print(e)

    def setup_ui(self):
        self.root.title("Model Runner MVP")
        self.root.geometry("1000x600")

        plot_frame = tk.Frame(self.root)
        plot_frame.grid(row=0, column=0, padx=10, pady=10)

        browse_button = tk.Button(root, text="Browse", command=self.browse)
        browse_button.place(relx=0.4, rely=0.05, anchor="nw")

        button_explore = tk.Button(
            self.root, text="Process", command=self.process_h5
        )
        button_explore.place(relx=0.6, rely=0.05, anchor="nw")

        self.file_path_label = tk.Label(
            root, text="No file or folder selected"
        )
        self.file_path_label.place(
            relx=0.5, rely=0.01, anchor="nw"
        )  # Add the label to the grid


if __name__ == "__main__":
    root = tk.Tk()

    app = ModelRunner(root)
    root.mainloop()
