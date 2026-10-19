"""Shared synthetic data for tests (never used for reported results)."""
import numpy as np
import pandas as pd

from src import config


def synthetic_heart(n=200, seed=0, with_missing=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n), "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n), "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 400, n), "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n), "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n), "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n), "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
    })
    score = (df.cp == 4) * 1.5 + df.exang + df.oldpeak / 2 + df.ca - df.thalach / 100
    df["target"] = (score + rng.normal(0, 0.5, n) > score.median()).astype(int)
    if with_missing:
        df.loc[df.sample(5, random_state=1).index, "ca"] = np.nan
        df.loc[df.sample(3, random_state=2).index, "thal"] = np.nan
    return df[config.COLUMNS]
