"""Train RandomForest and GradientBoosting candidates with MLflow tracking.

Every hyperparameter configuration is evaluated with 5-fold stratified CV on the
training split and logged as its own MLflow run. The best run (by validation
macro F1) is registered as ``WineClassifier`` and given the alias ``champion``.
"""
import argparse
import os
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from src.data import SEED, get_train_test_split

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
SKOPS_TRUSTED = [
    "sklearn.tree._tree.Tree",
    "sklearn.tree._classes.DecisionTreeClassifier",
    "sklearn.ensemble._forest.RandomForestClassifier",
    "sklearn.ensemble._gb.GradientBoostingClassifier",
    "numpy.ndarray",
    "numpy.dtype",
]
CV_FOLDS = 5

# Model Family A: RandomForestClassifier, Model Family B: GradientBoostingClassifier
SEARCH_GRID = {
    "RandomForest": [
        {"n_estimators": 25, "max_depth": 3, "min_samples_split": 2},
        {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 4},
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
    ],
    "GradientBoosting": [
        {"n_estimators": 25, "learning_rate": 0.1, "max_depth": 2},
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 2},
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 2},
    ],
}

MODEL_CLASSES = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}


def build_model(family, params):
    """Instantiate a classifier with a fixed seed."""
    return MODEL_CLASSES[family](random_state=SEED, **params)


def _scores(model, X, y):
    proba = model.predict_proba(X)
    pred = model.predict(X)
    return {
        "f1_macro": f1_score(y, pred, average="macro"),
        "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, proba, labels=model.classes_),
    }


def cross_validate_model(family, params, X, y, n_splits=CV_FOLDS):
    """5-fold stratified CV; returns mean train/validation metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for tr_idx, va_idx in skf.split(X, y):
        model = build_model(family, params)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        tr = _scores(model, X.iloc[tr_idx], y.iloc[tr_idx])
        va = _scores(model, X.iloc[va_idx], y.iloc[va_idx])
        row = {f"train_{k}": v for k, v in tr.items()}
        row.update({f"val_{k}": v for k, v in va.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    metrics = df.mean().to_dict()
    metrics["val_f1_macro_std"] = float(df["val_f1_macro"].std())
    return metrics


def run_training(tracking_uri=DEFAULT_TRACKING_URI, experiment_name=EXPERIMENT_NAME):
    """Run all candidate configs, register the best one, return a summary dict."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = get_train_test_split()
    input_example = X_train.head(5)

    results = []
    for family, configs in SEARCH_GRID.items():
        for i, params in enumerate(configs, start=1):
            run_name = f"{family}-cfg{i}"
            with mlflow.start_run(run_name=run_name) as run:
                metrics = cross_validate_model(family, params, X_train, y_train)

                mlflow.log_params({"model_family": family, "cv_folds": CV_FOLDS,
                                   "seed": SEED, **params})
                mlflow.log_metrics(metrics)
                mlflow.set_tags({"model_family": family, "dataset": "sklearn.load_wine",
                                 "validation": f"{CV_FOLDS}-fold-stratified-cv",
                                 "candidate": run_name})

                # Final fit on the full training split for the logged artifact.
                model = build_model(family, params).fit(X_train, y_train)
                signature = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                    model, name="model", signature=signature,
                    input_example=input_example,
                    skops_trusted_types=SKOPS_TRUSTED,
                )
                results.append({"run_id": run.info.run_id, "run_name": run_name,
                                "family": family, **params, **metrics})
                print(f"[{run_name}] val_f1_macro={metrics['val_f1_macro']:.4f} "
                      f"val_acc={metrics['val_accuracy']:.4f} "
                      f"val_logloss={metrics['val_log_loss']:.4f}")

    # Champion selection: compare all runs by validation macro F1.
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    best = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.val_f1_macro DESC", "metrics.val_log_loss ASC"],
        max_results=1,
    ).iloc[0]
    best_run_id = best["run_id"]

    mv = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, mv.version)
    print(f"\nRegistered {MODEL_NAME} v{mv.version} (run {best_run_id}) "
          f"with alias '{CHAMPION_ALIAS}'. val_f1_macro={best['metrics.val_f1_macro']:.4f}")

    table = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    table.to_csv("reports/table1_hyperparameter_results.csv", index=False)
    return {"best_run_id": best_run_id, "model_version": mv.version,
            "best_val_f1_macro": float(best["metrics.val_f1_macro"]), "results": table}


def parse_args():
    p = argparse.ArgumentParser(description="Train wine classifiers with MLflow tracking")
    p.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    p.add_argument("--experiment", default=EXPERIMENT_NAME)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args.tracking_uri, args.experiment)
