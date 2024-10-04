"""
This file includes functions and classes essential for dealing
with EosDX DB for data extraction
"""

import os
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

UNZIP_PATH_DATA = Path("./unzipped_data/")
UNZIP_PATH_BLIND_DATA = Path("./unzipped_blind_data/")
URL_DATA = "https://api.eosdx.com/api/getmultiple"
URL_BLIND_DATA = "https://api.eosdx.com/api/getblinddata"


@dataclass
class RequestDB:
    """
    A dataclass to store essential information for requests to the server.

    :param api_key: The API key used for authentication.
    :type api_key: str
    :param form: A dictionary containing form data for the request.
    :type form: Dict[str, str]
    :param file_name: Name of the file where the data will be downloaded.
    :type file_name: str
    :param url: The URL of the server. Defaults to a predefined URL.
    :type url: str
    :param unzip_path: The path where downloaded files should be unzipped.
        Defaults to a predefined path.
    :type unzip_path: Union[str, Path]
    :param dataset_name: Name to save the dataset as.
    :type dataset_name: str
    """

    api_key: str
    form: Dict[str, str]
    file_name: str
    url: str
    unzip_path: Union[str, Path]
    dataset_name: str = "data"

    def __post_init__(self):
        print("Converting unzip_path, file_name to Path()")
        self.unzip_path = Path(self.unzip_path)
        if not os.path.exists(self.unzip_path):
            os.mkdir(self.unzip_path)
        self.file_name = Path(self.file_name)
        if not self.file_name.is_absolute():
            self.file_name = self.unzip_path / self.file_name
        print(f"File name is set to {self.file_name}")


def monitor_elapsed_time(start_time, stop_event):
    """
    Monitors and prints the elapsed time in a separate thread.

    :param start_time: The start time of the operation.
    :type start_time: float
    :param stop_event: Event object to signal the thread to stop.
    :type stop_event: threading.Event
    """
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        print(
            "Waiting for server to respond... "
            f"Elapsed time: {elapsed_time:.2f} seconds",
            end="\r",
        )
        time.sleep(0.1)


def download_data(
    api_key: str, form: Dict[str, str], url: str, file_name: str
):
    """
    Downloads the required data from the EOSDX DB.

    :param api_key: The API key as a string.
    :type api_key: str

    :param form: JSON-like dictionary representing the request form.
        See API.md for information.

        Example for cancer tissue data:

        .. code-block:: python

            form = {
                'key': 'your-access-key',
                'cancer_tissue': True,
                'measurement_id': '< 3',
                'measurement_date': '2023-04-07'
            }

        Example for blind data:

        .. code-block:: python

            form = {
                'study': '2',  # 1 for California, 2 for Keele, 3 for mice data
                'key': key,
                'machine': '3', # 1 for Cu, 2 for Mo in California, 3 for Keele
                'manual_distance': '160'
            }

    :type form: Dict[str, str]

    :param url: The URL of the server.

        Example URLs:

        - URL_DATA = "https://api.eosdx.com/api/getmultiple"
        - URL_BLIND_DATA = "https://api.eosdx.com/api/getblinddata"

    :type url: str

    :param file_name: Name of the file where the data will be downloaded.
    :type file_name: str
    """
    form["key"] = api_key
    start_time = time.time()

    # Event object to control the monitor thread
    stop_event = threading.Event()

    # Create a separate thread to monitor elapsed time
    monitor_thread = threading.Thread(
        target=monitor_elapsed_time, args=(start_time, stop_event)
    )
    monitor_thread.start()

    response = requests.post(url, form, stream=True)

    # Stop the monitor thread once response is received
    stop_event.set()
    monitor_thread.join()

    print("\nResponse received")

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(file_name, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    print("Download completed")


def unzip_data(file_name: str, unzip_path: str):
    """
    Unzips data downloaded from the server to the specified folder.

    :param unzip_path: The path where the data will be extracted.
        Defaults to UNZIP_PATH if not provided.
    :type unzip_path: Union[str, Path]
    """
    print(f"Unzipping {file_name}...")
    with zipfile.ZipFile(file_name, "r") as zf:
        zf.extractall(unzip_path)
    if os.path.exists(file_name):
        os.remove(file_name)


def form_df(unzip_path=UNZIP_PATH_DATA) -> pd.DataFrame:
    """
    Generates a pandas DataFrame from the data downloaded using the EOSDX API.

    :param unzip_path: The path where the downloaded data is extracted.
        Defaults to UNZIP_PATH if not provided.
    :type unzip_path: Path
    :returns: A pandas DataFrame containing the data.
    :rtype: pd.DataFrame
    """
    print("Forming dataframe...")
    df = pd.read_csv(unzip_path / Path("description.csv"))
    # MANDATORY!!!
    df["measurement_data"] = np.nan
    df["measurement_data"] = df["measurement_data"].astype(object)
    df.set_index("measurement_id", inplace=True)
    # Define the path to the measurements directory as a Path object
    meas_path = unzip_path / Path("measurements")
    # Use apply with a lambda function to load the matrix and assign
    # it to 'measurement_data' column
    df["measurement_data"] = df.index.map(
        lambda idx: np.load(meas_path / Path(f"{idx}.npy"), allow_pickle=True)
    )
    return df


def save_df(df, path, dataset_name):
    """
    Saves the DataFrame as a pickle file.

    :param df: A pandas DataFrame containing the data.
    :type df: pd.DataFrame
    :param path: The path where the file will be saved.
        Defaults to UNZIP_PATH if not provided.
    :type path: Path
    :param dataset_name: Filename to save the file as.
    :type dataset_name: str
    """
    df_file_path = path / Path(dataset_name)
    df.to_pickle(str(df_file_path) + ".pkl")
    print(f"Data frame is formed and saved as data {df_file_path}")


def get_df(request: RequestDB) -> pd.DataFrame:
    """
    Makes a request to the database using the provided `RequestDB` object\
    and returns a pandas DataFrame.

    :param request: A `RequestDB` object containing essential information for\
        the request.
    :type request: RequestDB
    :returns: A pandas DataFrame containing the data retrieved from the\
        database.
    :rtype: pd.DataFrame
    """
    download_data(
        request.api_key,
        request.form,
        request.url,
        request.file_name,
    )
    unzip_data(request.file_name, request.unzip_path)
    df = form_df(request.unzip_path)
    save_df(df, request.unzip_path, request.dataset_name)
    return df
