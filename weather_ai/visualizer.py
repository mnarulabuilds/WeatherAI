from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from weather_ai.constants import CLASS_CODE_TO_PLOT_VALUE, FEATURE_LABELS, PLOT_VALUE_TO_LABEL


class WeatherVisualizer:
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.feature_labels = list(FEATURE_LABELS)

    def plot_comparison(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        feature_idx: int,
        title: str | None = None,
    ) -> str:
        actual = np.asarray(actual)
        predicted = np.asarray(predicted)
        if actual.shape != predicted.shape:
            raise ValueError("Actual and predicted arrays must share the same shape.")
        if feature_idx < 0 or feature_idx >= actual.shape[1]:
            raise ValueError("feature_idx out of range.")

        plt.figure(figsize=(12, 6))
        days = np.arange(len(actual))
        plt.plot(days, actual[:, feature_idx], "g*", label="Actual", markersize=4)
        plt.plot(days, predicted[:, feature_idx], "ro", label="Predicted", alpha=0.6, markersize=2)

        label = self.feature_labels[feature_idx]
        plt.title(title or f"Comparison: {label}")
        plt.xlabel("Day")
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        filename = f"{label.replace(' ', '_')}.png"
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return path

    def plot_all_features(self, actual: np.ndarray, predicted: np.ndarray) -> list[str]:
        return [self.plot_comparison(actual, predicted, idx) for idx in range(actual.shape[1])]

    def plot_classes(self, actual_codes: np.ndarray, predicted_codes: np.ndarray) -> str:
        y_actual = [CLASS_CODE_TO_PLOT_VALUE.get(str(c), 4) for c in actual_codes]
        y_pred = [CLASS_CODE_TO_PLOT_VALUE.get(str(c), 4) for c in predicted_codes]

        plt.figure(figsize=(12, 6))
        days = np.arange(len(y_actual))
        plt.plot(days, y_actual, "b^", label="Actual", markersize=6)
        plt.plot(days, y_pred, "ro", label="Predicted", alpha=0.6, markersize=3)
        plt.yticks(list(PLOT_VALUE_TO_LABEL.keys()), list(PLOT_VALUE_TO_LABEL.values()))
        plt.title("Weather Class Comparison")
        plt.xlabel("Day")
        plt.ylabel("Event")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        path = os.path.join(self.output_dir, "Classes_Comparison.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return path
