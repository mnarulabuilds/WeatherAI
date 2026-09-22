from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from weather_ai.engine import TrainingResult, WeatherEngine
from weather_ai.metrics import EvaluationMetrics, evaluate_classification, evaluate_regression
from weather_ai.visualizer import WeatherVisualizer


@dataclass(frozen=True)
class PredictionArtifacts:
    metrics: EvaluationMetrics
    class_plot_path: str
    feature_plot_paths: list[str]
    primary_preview_path: str


class AppController:
    """Headless orchestration used by the desktop UI and unit tests."""

    def __init__(self, engine: WeatherEngine, visualizer: WeatherVisualizer):
        self.engine = engine
        self.visualizer = visualizer
        self.training_result: Optional[TrainingResult] = None

    def train(
        self,
        start_year: int,
        end_year: int,
        progress=None,
    ) -> TrainingResult:
        self.training_result = self.engine.run_full_pipeline(
            start_year=start_year,
            end_year=end_year,
            progress=progress,
        )
        return self.training_result

    def run_predictions(self, plot_all_features: bool = True) -> PredictionArtifacts:
        if self.training_result is None:
            raise RuntimeError("Train models before running predictions.")

        df = self.training_result.dataframe
        x_reg = self.training_result.regression_x
        y_reg = self.training_result.regression_y

        pred_features = self.engine.predict_regression(x_reg)
        pred_classes = self.engine.predict_classes_from_features(pred_features)

        offset = self.engine.days_per_year
        actual_classes = df["ClassCode"].to_numpy()[offset:]

        reg_mae, reg_r2, per_feature_mae = evaluate_regression(y_reg, pred_features)
        clf_acc = evaluate_classification(actual_classes, pred_classes)
        metrics = EvaluationMetrics(
            regression_mae=reg_mae,
            regression_r2=reg_r2,
            classification_accuracy=clf_acc,
            per_feature_mae=per_feature_mae,
        )

        class_plot = self.visualizer.plot_classes(actual_classes, pred_classes)
        feature_paths = (
            self.visualizer.plot_all_features(y_reg, pred_features)
            if plot_all_features
            else [self.visualizer.plot_comparison(y_reg, pred_features, 0)]
        )

        return PredictionArtifacts(
            metrics=metrics,
            class_plot_path=class_plot,
            feature_plot_paths=feature_paths,
            primary_preview_path=class_plot,
        )
