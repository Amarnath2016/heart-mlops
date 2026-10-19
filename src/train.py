"""Train, tune, evaluate and track models; persist the best pipeline.

Usage:  python -m src.train [--tracking-uri file:./mlruns] [--experiment heart-disease]
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)

from src import config  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.models import candidates  # noqa: E402

CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def split(df, test_size=0.2):
    X, y = df[config.FEATURES], df[config.TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=config.RANDOM_STATE)


def evaluate_holdout(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def tune_and_cv(name, pipe, grid, X_train, y_train, cv):
    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=1, refit=True)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    # Cross-validated metrics of the tuned configuration. (Tuning and reporting use the
    # same folds, so these are slightly optimistic; the hold-out test set is the
    # unbiased estimate.)
    cvres = cross_validate(best, X_train, y_train, cv=cv, scoring=CV_SCORING)
    cv_metrics = {f"cv_{m}_mean": float(np.mean(cvres[f"test_{m}"])) for m in CV_SCORING}
    cv_metrics.update({f"cv_{m}_std": float(np.std(cvres[f"test_{m}"])) for m in CV_SCORING})
    return search, best, cv_metrics


def make_plots(model, X_test, y_test, name):
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax1)
    ax1.set_title(f"ROC curve - {name}")
    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax2, cmap="Blues")
    ax2.set_title(f"Confusion matrix - {name}")
    return fig1, fig2


def main(tracking_uri=None, experiment="heart-disease-risk", cv_splits=5):
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(tracking_uri or f"file:{config.ROOT / 'mlruns'}")
    mlflow.set_experiment(experiment)

    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=config.RANDOM_STATE)

    results = {}
    for name, (pipe, grid) in candidates().items():
        with mlflow.start_run(run_name=name):
            search, best, cv_metrics = tune_and_cv(name, pipe, grid, X_train, y_train, cv)
            t_metrics = evaluate_holdout(best, X_test, y_test)
            mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
            mlflow.log_params({"model": name, "cv_splits": cv_splits,
                               "n_train": len(X_train), "n_test": len(X_test),
                               "grid_size": len(search.cv_results_["params"])})
            mlflow.log_metrics({**cv_metrics, **t_metrics,
                                "best_cv_roc_auc_search": float(search.best_score_)})
            roc_fig, cm_fig = make_plots(best, X_test, y_test, name)
            mlflow.log_figure(roc_fig, "roc_curve.png")
            mlflow.log_figure(cm_fig, "confusion_matrix.png")
            plt.close("all")
            mlflow.sklearn.log_model(best, "model")
            results[name] = {"model": best, "params": search.best_params_,
                             "cv": cv_metrics, "test": t_metrics,
                             "run_id": mlflow.active_run().info.run_id}
            print(f"{name}: cv_roc_auc={cv_metrics['cv_roc_auc_mean']:.3f} "
                  f"test_roc_auc={t_metrics['test_roc_auc']:.3f}")

    # Model selection: highest mean CV ROC-AUC (test set is not used for selection).
    winner = max(results, key=lambda n: results[n]["cv"]["cv_roc_auc_mean"])
    final = results[winner]
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final["model"], config.MODEL_PATH)
    meta = {
        "model_name": winner,
        "best_params": {k: str(v) for k, v in final["params"].items()},
        "cv_metrics": final["cv"],
        "test_metrics": final["test"],
        "features": config.FEATURES,
        "mlflow_run_id": final["run_id"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "comparison": {n: {"cv_roc_auc_mean": r["cv"]["cv_roc_auc_mean"],
                           "test_roc_auc": r["test"]["test_roc_auc"],
                           "test_accuracy": r["test"]["test_accuracy"],
                           "test_precision": r["test"]["test_precision"],
                           "test_recall": r["test"]["test_recall"]}
                       for n, r in results.items()},
    }
    config.METADATA_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Selected: {winner} -> {config.MODEL_PATH}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--experiment", default="heart-disease-risk")
    ap.add_argument("--cv-splits", type=int, default=5)
    a = ap.parse_args()
    main(a.tracking_uri, a.experiment, a.cv_splits)
