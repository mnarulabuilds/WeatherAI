import numpy as np

from weather_ai.metrics import (
    evaluate_classification,
    evaluate_regression,
    temporal_split_index,
)


def test_evaluate_regression_perfect_match():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    mae, r2, per_feature = evaluate_regression(y, y)
    assert mae == 0.0
    assert r2 == 1.0
    assert per_feature == (0.0, 0.0)


def test_evaluate_classification_perfect_match():
    y = np.array(["1000", "0010", "0100"])
    assert evaluate_classification(y, y) == 1.0


def test_temporal_split_index_respects_holdout():
    assert temporal_split_index(100, 0.2) == 80
    assert temporal_split_index(1, 0.2) == 1
