"""MLOps quality gate: blocks degrading models from reaching main.

The fixture trains the full pipeline into a throwaway MLflow store, so the gate is
self-contained on CI (mlruns/ and mlflow.db are gitignored).
"""
import statistics
import time

import numpy as np
import pytest

from src.data import get_train_test_split
from src.evaluate import load_champion
from src.train import run_training

F1_THRESHOLD = 0.88
LATENCY_THRESHOLD_MS = 30.0
VALID_CLASSES = {0, 1, 2}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    uri = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'gate.db'}"
    summary = run_training(tracking_uri=uri)
    model = load_champion(uri)
    _, X_test, _, _ = get_train_test_split()
    return summary, model, X_test


def test_metric_threshold_gate(pipeline):
    summary, _, _ = pipeline
    assert summary["best_val_f1_macro"] >= F1_THRESHOLD, (
        f"Validation macro F1 {summary['best_val_f1_macro']:.4f} < {F1_THRESHOLD}"
    )


def test_inference_latency_gate(pipeline):
    _, model, X_test = pipeline
    model.predict(X_test)  # warm-up
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        model.predict(X_test)
        timings.append((time.perf_counter() - start) * 1000)
    median_ms = statistics.median(timings)
    assert median_ms <= LATENCY_THRESHOLD_MS, (
        f"Batch inference {median_ms:.2f} ms > {LATENCY_THRESHOLD_MS} ms"
    )


def test_output_schema_integrity(pipeline):
    _, model, X_test = pipeline
    preds = np.asarray(model.predict(X_test))
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset(VALID_CLASSES)
    assert np.issubdtype(preds.dtype, np.integer)
