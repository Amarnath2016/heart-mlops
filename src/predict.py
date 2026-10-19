"""Inference helpers shared by the API and offline scripts."""
import json
from pathlib import Path

import joblib
import pandas as pd

from src import config

LABELS = {0: "no heart disease", 1: "heart disease"}

# Plausible physiological bounds; guards against garbage input at the API edge.
BOUNDS = {
    "age": (1, 120), "trestbps": (50, 300), "chol": (50, 700),
    "thalach": (40, 260), "oldpeak": (0, 10),
}
ALLOWED = {
    "sex": {0, 1}, "cp": {1, 2, 3, 4}, "fbs": {0, 1}, "restecg": {0, 1, 2},
    "exang": {0, 1}, "slope": {1, 2, 3}, "ca": {0, 1, 2, 3}, "thal": {3, 6, 7},
}
OPTIONAL = {"ca", "thal"}  # missing in some UCI rows; imputed by the pipeline


class ValidationError(ValueError):
    pass


def load_model(path: Path = config.MODEL_PATH):
    return joblib.load(path)


def load_metadata(path: Path = config.METADATA_PATH) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def validate(payload) -> dict:
    """Return a clean {feature: float|None} dict or raise ValidationError."""
    if not isinstance(payload, dict):
        raise ValidationError("Body must be a JSON object")
    missing = [f for f in config.FEATURES if f not in payload and f not in OPTIONAL]
    if missing:
        raise ValidationError(f"Missing fields: {missing}")
    clean = {}
    for f in config.FEATURES:
        v = payload.get(f)
        if v is None:
            if f in OPTIONAL:
                clean[f] = None
                continue
            raise ValidationError(f"Field '{f}' must not be null")
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValidationError(f"Field '{f}' must be numeric") from None
        if f in BOUNDS and not BOUNDS[f][0] <= v <= BOUNDS[f][1]:
            raise ValidationError(f"Field '{f}' out of range {BOUNDS[f]}")
        if f in ALLOWED and v not in ALLOWED[f]:
            raise ValidationError(f"Field '{f}' must be one of {sorted(ALLOWED[f])}")
        clean[f] = v
    return clean


def predict_one(model, payload) -> dict:
    row = validate(payload)
    X = pd.DataFrame([row], columns=config.FEATURES).astype(float)
    proba = model.predict_proba(X)[0]
    pred = int(proba[1] >= 0.5)
    return {
        "prediction": pred,
        "label": LABELS[pred],
        "confidence": round(float(proba[pred]), 4),
        "probability_heart_disease": round(float(proba[1]), 4),
    }
