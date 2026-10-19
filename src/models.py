"""Candidate models and their hyper-parameter grids."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.features import build_preprocessor


def candidates():
    """name -> (pipeline, param_grid). Grid keys address the 'clf' step."""
    lr = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=1)),
    ])
    return {
        "logistic_regression": (lr, {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
        }),
        "random_forest": (rf, {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 5, None],
            "clf__min_samples_leaf": [1, 3, 5],
        }),
    }
