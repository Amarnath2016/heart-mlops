"""Download the UCI Heart Disease (Cleveland) data and build the cleaned CSV.

    python scripts/download_data.py
If the UCI server is unreachable, download the file manually from
https://archive.ics.uci.edu/dataset/45/heart+disease (processed.cleveland.data)
and place it at data/raw/processed.cleveland.data, then run with --skip-download.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    if not a.skip_download:
        print("Downloaded:", data.download_raw())
    df = data.prepare()
    print(f"Clean dataset: {df.shape} -> {config.CLEAN_PATH}")
    print(df[config.TARGET].value_counts().to_string())
