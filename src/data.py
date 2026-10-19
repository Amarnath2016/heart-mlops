"""Data acquisition and cleaning."""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def download_raw(dest: Path = config.RAW_PATH, url: str = config.DATA_URL) -> Path:
    """Download the Cleveland subset of the UCI Heart Disease dataset."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed https URL
    return dest


def load_raw(path: Path = config.RAW_PATH) -> pd.DataFrame:
    """Read the headerless raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=config.COLUMNS, na_values="?")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy.

    * numeric coercion (guards against stray strings)
    * duplicate rows removed
    * target binarised: 0 = no disease, 1..4 = disease present
    Missing values are kept as NaN on purpose; they are imputed inside the
    modelling pipeline so that imputation statistics are learned on training
    folds only (no leakage).
    """
    out = df.copy()
    for col in config.COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.drop_duplicates().reset_index(drop=True)
    out = out.dropna(subset=[config.TARGET])
    out[config.TARGET] = (out[config.TARGET] > 0).astype(int)
    for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        out[col] = out[col].astype(float)  # keep NaN-capable numeric codes
    return out.replace([np.inf, -np.inf], np.nan)


def prepare(raw_path: Path = config.RAW_PATH, out_path: Path = config.CLEAN_PATH) -> pd.DataFrame:
    df = clean(load_raw(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_clean(path: Path = config.CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
