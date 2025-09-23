"""Data download and dataframe utilities (minimal implementation for tests)."""

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests


@dataclass
class RequestDB:
    api_key: str
    form: dict
    file_name: str
    url: str
    unzip_path: str


def download_data(api_key: str, form: dict, url: str, out_zip_path: Path):
    payload = dict(form)
    payload["key"] = api_key
    resp = requests.post(url, payload, stream=True)
    out_zip_path = Path(out_zip_path)
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_zip_path), "wb") as f:
        for chunk in resp.iter_content(8192):
            if chunk:
                f.write(chunk)


def unzip_data(zip_path: Path, unzip_path: Path):
    zip_path = Path(zip_path)
    unzip_path = Path(unzip_path)
    unzip_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(unzip_path))


def form_df(unzip_path: Path) -> pd.DataFrame:
    unzip_path = Path(unzip_path)
    df = pd.read_csv(unzip_path / "description.csv")
    df = df.set_index("measurement_id", drop=False)

    # Assume measurement IDs in 'measurement_id' and per-id .npy files next to CSV
    def _load_arr(mid):
        return np.load(unzip_path / f"{int(mid)}.npy")

    df = df.copy()
    df["measurement_data"] = df["measurement_id"].apply(_load_arr)
    return df


def save_df(df: pd.DataFrame, out_dir: Path, name: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Minimal persistence (optional for tests)
    df.to_csv(out_dir / f"{name}.csv", index=False)


def get_df(req: RequestDB) -> pd.DataFrame:
    unzip_dir = Path(req.unzip_path)
    zip_path = unzip_dir / req.file_name
    download_data(req.api_key, req.form, req.url, zip_path)
    unzip_data(zip_path, unzip_dir)
    df = form_df(unzip_dir)
    save_df(df, unzip_dir, "data")
    return df
