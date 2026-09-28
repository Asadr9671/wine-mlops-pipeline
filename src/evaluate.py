"""Inference verification: load the registered champion model and score the test split."""
import argparse

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import get_train_test_split
from src.train import CHAMPION_ALIAS, DEFAULT_TRACKING_URI, MODEL_NAME


def load_champion(tracking_uri=DEFAULT_TRACKING_URI):
    """Load the model registered as WineClassifier@champion."""
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")


def evaluate_champion(tracking_uri=DEFAULT_TRACKING_URI):
    """Compute final metrics of the champion model on the held-out test split."""
    model = load_champion(tracking_uri)
    _, X_test, _, y_test = get_train_test_split()
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    return {
        "test_f1_macro": f1_score(y_test, pred, average="macro"),
        "test_accuracy": accuracy_score(y_test, pred),
        "test_log_loss": log_loss(y_test, proba, labels=model.classes_),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    metrics = evaluate_champion(ap.parse_args().tracking_uri)
    print(f"Champion ({MODEL_NAME}@{CHAMPION_ALIAS}) test-set metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
