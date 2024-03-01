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
    requests to the server
    """

    api_key: str
    form: Dict[str, str]
    url: str = URL
    unzip_path: Union[str, Path] = UNZIP_PATH


def download_data(api_key: str, form={}, url=URL):
    """
    The function download the required data from eosdx DB
    :param api_key: the use API key
    :param form: JSON like dict request form, see API.md
    description for information
    form = {'key': 'your-access-key', 'cancer_tissue': True,
     'measurement_id': '< 3', 'measurement_date': '2023-04-07'}
    :param url: sharepoint
    :return: None
    """
    form["key"] = api_key
    response = requests.post(url, form, stream=True)
    block_size = 1024
    with open("data.zip", "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)


def unzip_data(unzip_path=UNZIP_PATH):
    """
    unzip data downloaded from the server to unzip_path folder
    """
    with zipfile.ZipFile("data.zip", "r") as zf:
        zf.extractall(unzip_path)


def form_df(unzip_path=UNZIP_PATH) -> pd.DataFrame:
    """
    generates pandas dataframe according to the data downloaded from the DB
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
    download_data(api_key=request.api_key, form=request.form, url=request.url)
    unzip_data(request.unzip_path)
    return form_df(request.unzip_path)
