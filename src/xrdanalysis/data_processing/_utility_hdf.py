"""Private HDF5-to-DataFrame conversion implementation."""

from __future__ import annotations

import h5py
import pandas as pd


def combine_h5_to_df(file_paths, *, h5_to_df):
    """Combine calibration and measurement tables from several HDF5 files."""
    calibration_df = None
    measurement_df = None

    for file in file_paths:
        cal, meas = h5_to_df(file)
        if calibration_df is not None:
            calibration_df = pd.concat([calibration_df, cal])
            measurement_df = pd.concat([measurement_df, meas])
        else:
            calibration_df = cal
            measurement_df = meas
    return calibration_df, measurement_df


def h5_to_df(file_path):  # noqa: C901
    """Convert supported simple and grouped HDF5 layouts into DataFrames."""
    calibration_data = []
    measurement_data = []

    with h5py.File(file_path, "r") as hdf:
        if "calibrations" in hdf and "measurements" in hdf:
            calibration_group = hdf["calibrations"]
            measurements_group = hdf["measurements"]

            cal_metadata = {
                f"calib_{key}": calibration_group.attrs[key]
                for key in calibration_group.attrs
            }
            meas_metadata = {
                f"{key}": measurements_group.attrs[key]
                for key in measurements_group.attrs
            }

            for ds_name in calibration_group:
                dataset = calibration_group[ds_name]
                if not isinstance(dataset, h5py.Dataset):
                    continue
                cal_data = {f"calib_{key}": dataset.attrs[key] for key in dataset.attrs}
                cal_data["measurement_data"] = dataset[...]
                cal_data["cal_name"] = ds_name
                cal_data["id"] = "root"
                calibration_data.append({**cal_data, **cal_metadata})

            for ds_name in measurements_group:
                dataset = measurements_group[ds_name]
                if not isinstance(dataset, h5py.Dataset):
                    continue
                meas_data = {f"{key}": dataset.attrs[key] for key in dataset.attrs}
                meas_data["measurement_data"] = dataset[...]
                meas_data["meas_name"] = ds_name
                meas_data["id"] = "root"
                measurement_data.append({**meas_data, **meas_metadata, **cal_metadata})
        else:
            for group_name in hdf:
                group = hdf[group_name]
                if "calibrations" not in group or not any(
                    key.startswith("measurements_") for key in group
                ):
                    continue

                calibration_group = group["calibrations"]
                cal_metadata = {
                    f"calib_{key}": calibration_group.attrs[key]
                    for key in calibration_group.attrs
                }
                for ds_name in calibration_group:
                    dataset = calibration_group[ds_name]
                    if not isinstance(dataset, h5py.Dataset):
                        continue
                    cal_data = {
                        f"calib_{key}": dataset.attrs[key] for key in dataset.attrs
                    }
                    cal_data["measurement_data"] = dataset[...]
                    cal_data["cal_name"] = ds_name
                    cal_data["id"] = group_name
                    calibration_data.append({**cal_data, **cal_metadata})

                for key in group:
                    if not key.startswith("measurements_"):
                        continue
                    measurements_group = group[key]
                    meas_metadata = {
                        f"{key}": measurements_group.attrs[key]
                        for key in measurements_group.attrs
                    }
                    for ds_name in measurements_group:
                        dataset = measurements_group[ds_name]
                        if not isinstance(dataset, h5py.Dataset):
                            continue
                        meas_data = {
                            f"{key}": dataset.attrs[key] for key in dataset.attrs
                        }
                        meas_data["measurement_data"] = dataset[...]
                        meas_data["meas_name"] = ds_name
                        meas_data["id"] = group_name
                        measurement_data.append(
                            {**meas_data, **meas_metadata, **cal_metadata}
                        )

    calibration_df = pd.DataFrame(calibration_data)
    measurement_df = pd.DataFrame(measurement_data)
    measurement_df.rename(columns={"calib_ponifile": "ponifile"}, inplace=True)
    calibration_df.rename(columns={"calib_ponifile": "ponifile"}, inplace=True)
    return calibration_df, measurement_df
