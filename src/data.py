"""Data loading, validation and splitting for the Wine dataset."""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

SEED = 42
EXPECTED_N_FEATURES = 13


def load_wine_data():
    """Return the Wine dataset as (X: DataFrame, y: Series)."""
    bunch = load_wine(as_frame=True)
    return bunch.data.copy(), bunch.target.copy()


def validate_data(X, y=None):
    """Validate that there are no nulls and exactly 13 features.

    Raises ValueError on failure and returns True otherwise.
    """
    if X.isnull().any().any():
        raise ValueError("Validation failed: feature matrix contains null values.")
    if X.shape[1] != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Validation failed: expected {EXPECTED_N_FEATURES} features, got {X.shape[1]}."
        )
    if y is not None:
        if pd.Series(y).isnull().any():
            raise ValueError("Validation failed: target contains null values.")
        if len(y) != len(X):
            raise ValueError("Validation failed: X and y lengths differ.")
    return True


def get_train_test_split(test_size=0.2, seed=SEED):
    """Validated, stratified 80/20 split with a fixed random seed."""
    X, y = load_wine_data()
    validate_data(X, y)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
