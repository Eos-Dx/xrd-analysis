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

UNZIP_PATH = Path("./unzipped_data/")
URL = "https://api.eosdx.com/api/getmultiple"


@dataclass
class RequestDB:
    """
    RequestDB is a dataclass used to store essential info for
    requests to the server.

    Attributes:
        api_key (str): The API key used for authentication.
        form (Dict[str, str]): A dictionary containing form data for the
        request.
        url (str): The URL of the server. Defaults to a predefined URL.
        unzip_path (Union[str, Path]): The path where downloaded files should
        be unzipped. Defaults to a predefined path.
    """

    api_key: str
    form: Dict[str, str]
    url: str = URL
    unzip_path: Union[str, Path] = UNZIP_PATH


def download_data(api_key: str, form={}, url=URL, where="data.zip"):
    """
    Download the required data from the EOSDX DB.

    :param api_key: The API key as a string.
    :param form: JSON-like dictionary request form. See API.md for more
    information.
                 Example form: {'key': 'your-access-key',
                                'cancer_tissue': True,
                                'measurement_id': '< 3',
                                'measurement_date': '2023-04-07'}
                 Defaults to an empty dictionary.
    :param url: The URL of the server. Defaults to a predefined URL.
    :param where: The location where the downloaded data will be saved.
    Defaults to "data.zip".
    :return: None
    """
    form["key"] = api_key
    response = requests.post(url, form, stream=True)
    block_size = 1024
    with open(where, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)


def unzip_data(unzip_path=UNZIP_PATH):
    """
    Unzip data downloaded from the server to the specified unzip_path folder.

    :param unzip_path: The path where the data will be extracted.
                       Defaults to UNZIP_PATH if not provided.
    :return: None
    """
    with zipfile.ZipFile("data.zip", "r") as zf:
        zf.extractall(unzip_path)


def form_df(unzip_path=UNZIP_PATH) -> pd.DataFrame:
    """
    Generates a pandas DataFrame according to the data downloaded from the DB
    using the EOSDX API.

    :param unzip_path: The path where the downloaded data is extracted.
    Defaults to UNZIP_PATH if not provided.
    :return: A pandas DataFrame containing the data.
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
    Makes a request to the database using the provided RequestDB object
    and returns a pandas DataFrame.

    :param request: A RequestDB object containing essential information
    for the request.
    :return: A pandas DataFrame containing the data retrieved from
    the database.
    """
    download_data(api_key=request.api_key, form=request.form, url=request.url)
    unzip_data(request.unzip_path)
    return form_df(request.unzip_path)
