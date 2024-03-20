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
    RequestDB is a dataclass used to store essential info for
    requests to the server
    """

    api_key: str
    form: Dict[str, str]
    file_name: str
    url: str
    unzip_path: Union[str, Path]


def download_data(api_key: str, form: Dict[str, str], url: str, file_name: str):
    """
    The function download the required data from eosdx DB
    :param api_key: the API key as string
    :param form: JSON like dict request form, see API.md
    description for information
    form = {'key': 'your-access-key', 'cancer_tissue': True,
     'measurement_id': '< 3', 'measurement_date': '2023-04-07'}

    for blind data
    form = {'study': '2', # 1 for california, 2 for keele, 3 for mice data
           'key': key, #
           'machine': '3', # 1 for Cu, 2 for Mo in california, 3 for keele,
           'manual_distance': '160'}

    :param url:
    URL_DATA = "https://api.eosdx.com/api/getmultiple"
    URL_BLIND_DATA = "https://api.eosdx.com/api/getblinddata"
    :file_name: name of file where the data will be downloaded
    :return: None
    """
    form["key"] = api_key
    response = requests.post(url, form, stream=True)
    block_size = 1024
    with open(file_name, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)


def unzip_data(file_name: str, unzip_path: str):
    """
    unzip data downloaded from the server to unzip_path folder
    """
    with zipfile.ZipFile(file_name, "r") as zf:
        zf.extractall(unzip_path)


def form_df(unzip_path=UNZIP_PATH_DATA) -> pd.DataFrame:
    """
    generates pandas dataframe according to the data downloaded from the DB
    using eosdx API
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
    This is function that makes a requst to the DB and returns
    pandas dataframe
    """
    download_data(api_key=request.api_key, form=request.form,
                  url=request.url, file_name=request.file_name)
    unzip_data(file_name=request.file_name, unzip_path=request.unzip_path)
    return form_df(request.unzip_path)
