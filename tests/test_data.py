import pandas as pd
import pytest

from src.data import get_train_test_split, load_wine_data, validate_data


def test_dataset_shape():
    X, y = load_wine_data()
    assert X.shape == (178, 13)
    assert set(y.unique()) == {0, 1, 2}


def test_validation_passes_on_clean_data():
    X, y = load_wine_data()
    assert validate_data(X, y) is True


def test_validation_rejects_nulls():
    X, y = load_wine_data()
    X.iloc[0, 0] = None
    with pytest.raises(ValueError, match="null"):
        validate_data(X, y)


def test_validation_rejects_wrong_feature_count():
    X, y = load_wine_data()
    with pytest.raises(ValueError, match="13 features"):
        validate_data(X.iloc[:, :12], y)
    with pytest.raises(ValueError, match="13 features"):
        validate_data(pd.concat([X, X.iloc[:, :1].add_suffix("_dup")], axis=1), y)


def test_split_is_stratified_80_20_and_reproducible():
    X_tr, X_te, y_tr, y_te = get_train_test_split()
    assert len(X_tr) + len(X_te) == 178
    assert len(X_te) == pytest.approx(178 * 0.2, abs=1)
    _, y = load_wine_data()
    for cls in (0, 1, 2):
        assert (y_te == cls).mean() == pytest.approx((y == cls).mean(), abs=0.03)
    X_tr2, _, _, _ = get_train_test_split()
    pd.testing.assert_frame_equal(X_tr, X_tr2)
