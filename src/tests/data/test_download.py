"""
Test for data module
"""

import numpy as np
import pytest
import requests_mock

from xrdanalysis.data.download import UNZIP_PATH, URL, RequestDB, download_data, form_df


@pytest.fixture
def sample_request_data():
    """
    This fixture returns an instance of RequestDB.
    It's designed to simulate a sample request configuration.

    - `api_key`: A placeholder API key for authentication.
    - `form`: A dictionary containing sample form data to be sent.
    - `url`: The URL of the example API endpoint.

    This setup can be used to test API request construction, parameter passing,
    and URL formatting.
    """
    return RequestDB(
        api_key="your_api_key", form={"key": "value"}, url="http://example.com"
    )


def test_requestdb_instance(sample_request_data):
    """
    Test to ensure the 'sample_request_data' fixture returns
    an instance of RequestDB.
    Verifies that the object created by the fixture is indeed
    of the type RequestDB.
    """
    assert isinstance(sample_request_data, RequestDB)


def test_requestdb_attributes(sample_request_data):
    """
    Test to verify that the 'sample_request_data' fixture has
    the correct attributes set.

    Checks the following attributes of the RequestDB instance:
    - `api_key` matches the expected placeholder value.
    - `form` contains the correct dictionary as initialized.
    - `url` is set to the specified example endpoint.
    - `unzip_path` equals the globally defined UNZIP_PATH.

    This ensures that the RequestDB instance is correctly initialized
    with the expected data.
    """
    assert sample_request_data.api_key == "your_api_key"
    assert sample_request_data.form == {"key": "value"}
    assert sample_request_data.url == "http://example.com"
    assert sample_request_data.unzip_path == UNZIP_PATH


@pytest.fixture
def api_key():
    """
    This fixture returns an mock api_key
    """
    return "12345"


@pytest.fixture
def mock_response():
    """
    This fixture returns a mock response of a data server
    """
    return {"status": "ok", "data": "example data"}


def test_download_data(api_key, mock_response):
    """
    This function tests function download_test
    """
    with requests_mock.Mocker() as m:
        m.post(URL, json=mock_response)

        # Call your function with the mocked request
        download_data(api_key=api_key, url=URL)

        # Verify the request was made with the correct API key
        assert m.last_request.text == "key={}".format(api_key)

        # Assuming your function is supposed to create "data.zip"
        # from the response
        # Check if "data.zip" file was created and is not empty
        try:
            with open("data.zip", "rb") as file:
                content = file.read()
                # Simple check to ensure the file is not empty
                assert content
        finally:
            # Clean up the "data.zip" file after test to avoid cluttering
            import os

            if os.path.exists("data.zip"):
                os.remove("data.zip")


def test_form_df(tmp_path):
    # Setup a dummy CSV file and a .npy file in the temporary directory
    description_csv = tmp_path / "description.csv"
    description_csv.write_text("measurement_id\n1")

    measurements_path = tmp_path / "measurements"
    measurements_path.mkdir()
    np.save(measurements_path / "1.npy", np.array([1, 2, 3]))

    # Call form_df and assert
    df = form_df(tmp_path)

    assert not df.empty
    assert df.loc[1, "measurement_data"].tolist() == [1, 2, 3]
