"""
This file includes functions and classes essential for dealing
with EosDX DB for data extraction
"""

import zipfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union
from tqdm import tqdm
import threading
import numpy as np
import pandas as pd
import requests

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
        url (str): The URL of the server. Defaults to a predefined URL.
        unzip_path (Union[str, Path]): The path where downloaded
            files should be unzipped. Defaults to a predefined path.
    """

    api_key: str
    form: Dict[str, str]
    file_name: str
    url: str
    unzip_path: Union[str, Path]

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


def download_data(api_key: str, form: Dict[str, str], url: str, file_name: str):
    """
    Download the required data from the EOSDX DB and print the time taken for the response to arrive in real-time.

    Args:
        api_key (str): The API key as a string.
        form (Dict[str, str]): JSON-like dict request form.
        url (str): The URL of the server.
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
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    with open(file_name, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    print('Download completed')


def unzip_data(file_name: str, unzip_path: str):
    """
    Unzip data downloaded from the server to the specified unzip_path folder.

    Args:
        unzip_path (Union[str, Path]): The path where the data will
            be extracted. Defaults to UNZIP_PATH if not provided.
    """
    print('Data is extracting from zip')
    with zipfile.ZipFile(file_name, "r") as zf:
        zf.extractall(unzip_path)


def form_df(unzip_path=UNZIP_PATH_DATA) -> pd.DataFrame:
    """
    Generates a pandas DataFrame according to the data downloaded
    from the DB using the EOSDX API.

    Args:
        unzip_path (Path): The path where the downloaded
            data is extracted. Defaults to UNZIP_PATH if not provided.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data.
    """
    print('Data frame is forming...')
    df = pd.read_csv(unzip_path / "description.csv")
    # MANDATORY!!!
    df["measurement_data"] = np.nan
    df["measurement_data"] = df["measurement_data"].astype(object)
    df.set_index("measurement_id", inplace=True)
    # Define the path to the measurements directory as a Path object
    meas_path = unzip_path / "measurements"
    # Use apply with a lambda function to load the matrix and assign
    # it to 'measurement_data' column
    df["measurement_data"] = df.index.map(
        lambda idx: np.load(meas_path / f"{idx}.npy", allow_pickle=True)
    )
    print('Data frame is formed')
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
    return form_df(request.unzip_path)
