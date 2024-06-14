"""
This file includes functions and classes essential for dealing
with EosDX DB for data extraction
"""

import zipfile
import time
import threading
import numpy as np
import pandas as pd
import requests
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union
from tqdm import tqdm

UNZIP_PATH_DATA = Path("./unzipped_data/")
UNZIP_PATH_BLIND_DATA = Path("./unzipped_blind_data/")
URL_DATA = "https://api.eosdx.com/api/getmultiple"
URL_BLIND_DATA = "https://api.eosdx.com/api/getblinddata"


@dataclass
class RequestDB:
    """
    RequestDB is a dataclass used to store essential
    info for requests to the server.

    Args:
        api_key (str): The API key used for authentication.
        form (Dict[str, str]): A dictionary containing form data
            for the request.
        file_name (str): Name of the file where the data will be downloaded.
        url (str): The URL of the server. Defaults to a predefined URL.
        unzip_path (Union[str, Path]): The path where downloaded
            files should be unzipped. Defaults to a predefined path.
        dataset_name (str): Name to save the dataset as.
    """

    api_key: str
    form: Dict[str, str]
    file_name: str
    url: str
    unzip_path: Union[str, Path]
    dataset_name: str = "data.json"

    def __post_init__(self):
        print("Converting unzip_path, file_name to Path()")
        self.unzip_path = Path(self.unzip_path)
        self.file_name = Path(self.file_name)
        if not self.file_name.is_absolute():
            self.file_name = self.unzip_path / self.file_name
        print(f'File name is set to {self.file_name}')


def monitor_elapsed_time(start_time, stop_event):
    """
    Function to monitor and print elapsed time in a separate thread.

    Args:
        start_time (float): The start time of the operation.
        stop_event (threading.Event): Event object to signal the thread to stop.
    """
    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        print(f"Waiting for server to respond... Elapsed time: {elapsed_time:.2f} seconds", end="\r")
        time.sleep(0.1)


def download_data(
        api_key: str, form: Dict[str, str], url: str, file_name: str
):
    """
    Download the required data from the EOSDX DB.

    Args:
        api_key (str): The API key as a string.
        form (Dict[str, str]): JSON-like dict request form.
            See API.md description for information.
            Example for cancer tissue data:
                form = {'key': 'your-access-key',
                        'cancer_tissue': True,
                        'measurement_id': '< 3',
                        'measurement_date': '2023-04-07'}
            Example for blind data:
                form = {'study': '2',  # 1 for california, 2 for keele,
                        3 for mice data
                        'key': key,
                        'machine': '3',  # 1 for Cu, 2 for Mo in california,
                        3 for keele
                        'manual_distance': '160'}
        url (str): The URL of the server.
            Example URLs:
                URL_DATA = "https://api.eosdx.com/api/getmultiple"
                URL_BLIND_DATA = "https://api.eosdx.com/api/getblinddata"
        file_name (str): Name of the file where the data will be downloaded.
    """
    form["key"] = api_key
    start_time = time.time()

    # Event object to control the monitor thread
    stop_event = threading.Event()

    # Create a separate thread to monitor elapsed time
    monitor_thread = threading.Thread(target=monitor_elapsed_time, args=(start_time, stop_event))
    monitor_thread.start()

    response = requests.post(url, form, stream=True)

    # Stop the monitor thread once response is received
    stop_event.set()
    monitor_thread.join()

    print("\nResponse received")

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(file_name, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    print('Download completed')


def unzip_data(file_name: str, unzip_path: str):
    """
    Unzip data downloaded from the server to the specified unzip_path folder.

    Args:
        unzip_path (Union[str, Path]): The path where the data will
            be extracted. Defaults to UNZIP_PATH if not provided.
    """
    print(f'Unzipping {file_name}...')
    with zipfile.ZipFile(file_name, "r") as zf:
        zf.extractall(unzip_path)
    os.remove(file_name)


def form_df(
    unzip_path=UNZIP_PATH_DATA, dataset_name="data.json"
) -> pd.DataFrame:
    """
    Generates a pandas DataFrame according to the data downloaded
    from the DB using the EOSDX API. Saves the df as a json file.

    Args:
        unzip_path (Path): The path where the downloaded
            data is extracted. Defaults to UNZIP_PATH if not provided.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data.
    """
    print(f'Forming data frame...')
    df = pd.read_csv(Path(unzip_path) / Path("description.csv"))
    # MANDATORY!!!
    df["measurement_data"] = np.nan
    df["measurement_data"] = df["measurement_data"].astype(object)
    df.set_index("measurement_id", inplace=True)
    # Define the path to the measurements directory as a Path object
    meas_path = Path(unzip_path) / Path("measurements")
    # Use apply with a lambda function to load the matrix and assign
    # it to 'measurement_data' column
    df["measurement_data"] = df.index.map(
        lambda idx: np.load(meas_path / Path(f"{idx}.npy"),
                            allow_pickle=True)
    )
    df_file_path = Path(unzip_path / dataset_name)
    df.to_json(df_file_path)
    print(f'Data frame is formed and saved as data {df_file_path}')
    return df


def get_df(request: RequestDB) -> pd.DataFrame:
    """
    Makes a request to the database using the provided RequestDB
    object and returns a pandas DataFrame.

    Args:
        request (RequestDB): A RequestDB object containing
            essential information for the request.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data
            retrieved from the database.
    """
    download_data(
        request.api_key,
        request.form,
        request.url,
        request.file_name,
    )
    unzip_data(request.file_name, request.unzip_path)
    return form_df(request.unzip_path, request.dataset_name)
