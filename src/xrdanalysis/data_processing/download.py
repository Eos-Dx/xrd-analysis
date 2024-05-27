"""
This file includes functions and classes essential for dealing
with EosDX DB for data extraction
"""

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

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
    response = requests.post(url, form, stream=True)
    block_size = 1024
    with open(file_name, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)


def unzip_data(file_name: str, unzip_path: str):
    """
    Unzip data downloaded from the server to the specified unzip_path folder.

    Args:
        unzip_path (Union[str, Path]): The path where the data will
            be extracted. Defaults to UNZIP_PATH if not provided.
    """
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
