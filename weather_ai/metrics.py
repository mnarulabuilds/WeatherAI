from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score


@dataclass(frozen=True)
class EvaluationMetrics:
    regression_mae: float
    regression_r2: float
    classification_accuracy: float
    per_feature_mae: tuple[float, ...]

    def summary_lines(self) -> list[str]:
        return [
            f"Regression MAE (all features): {self.regression_mae:.3f}",
            f"Regression R²: {self.regression_r2:.3f}",
            f"Classification accuracy: {self.classification_accuracy * 100:.1f}%",
        ]


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, tuple[float, ...]]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    per_feature = tuple(float(mean_absolute_error(y_true[:, i], y_pred[:, i])) for i in range(y_true.shape[1]))
    return mae, r2, per_feature


def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def temporal_split_index(length: int, holdout_fraction: float) -> int:
    if length < 2:
        return length
    holdout = max(1, int(length * holdout_fraction))
    return max(1, length - holdout)
