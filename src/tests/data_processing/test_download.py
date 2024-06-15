import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd

from xrdanalysis.data_processing.download import (
    RequestDB,
    download_data,
    form_df,
    get_df,
    unzip_data,
)


# Test for download_data
@patch("requests.post")
def test_download_data(mock_post):
    """Test for data download"""
    # Mock response content
    mock_response = MagicMock()
    mock_response.iter_content = lambda _: [b"data"]
    mock_post.return_value = mock_response

    with patch("builtins.open", mock_open()) as mock_file:
        download_data(
            "api_key",
            {"form_key": "form_value"},
            "https://example.com",
            "file.zip",
        )
        mock_post.assert_called_once_with(
            "https://example.com",
            {"form_key": "form_value", "key": "api_key"},
            stream=True,
        )
        mock_file.assert_called_once_with("file.zip", "wb")
        mock_file().write.assert_called_once_with(b"data")


# Test for unzip_data
@patch("zipfile.ZipFile")
def test_unzip_data(mock_zipfile):
    """Test for unzipping data"""
    mock_zip = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip
    os.mkdir("unzipped_path")

    unzip_data("file.zip", "unzipped_path")

    mock_zipfile.assert_called_once_with("file.zip", "r")
    mock_zip.extractall.assert_called_once_with("unzipped_path")


# Test for form_df
@patch("pandas.read_csv")
@patch("numpy.load")
def test_form_df(mock_load, mock_read_csv):
    """Test for dataframe creation"""
    # Mock DataFrame
    df = pd.DataFrame({"measurement_id": [1, 2, 3]})
    mock_read_csv.return_value = df

    # Mock np.load return value
    mock_load.side_effect = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([7, 8, 9]),
    ]

    result_df = form_df(Path("unzipped_path"))

    mock_read_csv.assert_called_once_with(
        Path("unzipped_path") / "description.csv"
    )
    assert "measurement_data" in result_df.columns
    assert os.path.exists(Path("unzipped_path/data.json"))
    assert np.array_equal(
        result_df.loc[1, "measurement_data"], np.array([1, 2, 3])
    )
    assert np.array_equal(
        result_df.loc[2, "measurement_data"], np.array([4, 5, 6])
    )
    assert np.array_equal(
        result_df.loc[3, "measurement_data"], np.array([7, 8, 9])
    )


# Test for get_df
@patch("xrdanalysis.data_processing.download.download_data")
@patch("xrdanalysis.data_processing.download.unzip_data")
@patch("xrdanalysis.data_processing.download.form_df")
def test_get_df(mock_form_df, mock_unzip_data, mock_download_data):
    """Test for encompassing function"""
    # Mock the DataFrame returned by form_df
    mock_df = pd.DataFrame({"measurement_id": [1, 2, 3]})
    mock_form_df.return_value = mock_df

    request = RequestDB(
        api_key="api_key",
        form={"form_key": "form_value"},
        file_name="file.zip",
        url="https://example.com",
        unzip_path="unzipped_path",
    )

    result_df = get_df(request)

    mock_download_data.assert_called_once_with(
        "api_key",
        {"form_key": "form_value"},
        "https://example.com",
        Path("unzipped_path/file.zip"),
    )
    mock_unzip_data.assert_called_once_with(
        Path("unzipped_path/file.zip"), Path("unzipped_path")
    )
    mock_form_df.assert_called_once_with(Path("unzipped_path"), "data.json")
    assert result_df.equals(mock_df)
    if os.path.exists("unzipped_path"):
        shutil.rmtree("unzipped_path")
