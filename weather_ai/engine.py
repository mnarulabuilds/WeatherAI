from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from weather_ai.config import ModelConfig
from weather_ai.constants import CLASS_MAP, FEATURE_COLUMNS
from weather_ai.metrics import (
    EvaluationMetrics,
    evaluate_classification,
    evaluate_regression,
    temporal_split_index,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class TrainingResult:
    dataframe: pd.DataFrame
    regression_x: np.ndarray
    regression_y: np.ndarray
    holdout_metrics: EvaluationMetrics


class WeatherEngine:
    def __init__(
        self,
        data_dir: str = ".",
        config: Optional[ModelConfig] = None,
    ):
        self.data_dir = data_dir
        self.config = config or ModelConfig()
        self.feature_columns = list(FEATURE_COLUMNS)
        self.class_map = dict(CLASS_MAP)
        self.days_per_year = self.config.days_per_year

        self.predictor: Optional[MLPRegressor] = None
        self.classifier: Optional[MLPClassifier] = None
        self.feature_scaler = StandardScaler()
        self.regression_x_scaler = StandardScaler()
        self.regression_y_scaler = StandardScaler()

    def load_data(self, start_year: int = 1997, end_year: int = 2015) -> pd.DataFrame:
        """Load WeatherXXXX.txt files into a single DataFrame."""
        frames: list[pd.DataFrame] = []
        dtype = np.float32 if self.config.use_float32 else np.float64

        for year in range(start_year, end_year + 1):
            file_path = os.path.join(self.data_dir, f"Weather{year}.txt")
            if not os.path.exists(file_path):
                logger.warning("Missing data file: %s", file_path)
                continue

            df = pd.read_csv(
                file_path,
                sep=r"\s+",
                header=None,
                dtype={12: str},
                engine="c",
            )
            df.columns = ["Bias"] + self.feature_columns + ["ClassCode"]
            for col in self.feature_columns:
                df[col] = df[col].astype(dtype)
            df["Year"] = np.int16(year)
            frames.append(df)

        if not frames:
            raise FileNotFoundError("No weather data files found.")

        return pd.concat(frames, ignore_index=True)

    def prepare_regression_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized (day t) -> (day t + days_per_year) feature pairs."""
        features = df[self.feature_columns].to_numpy(copy=False)
        offset = self.days_per_year
        if len(features) <= offset:
            raise ValueError("Not enough rows to build year-over-year regression pairs.")

        x = features[:-offset]
        y = features[offset:]
        return x, y

    def _build_regressor(self) -> MLPRegressor:
        cfg = self.config
        return MLPRegressor(
            hidden_layer_sizes=cfg.reg_hidden_layers,
            activation=cfg.reg_activation,
            solver="adam",
            alpha=cfg.reg_alpha,
            max_iter=cfg.reg_max_iter,
            learning_rate_init=cfg.reg_learning_rate_init,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=cfg.random_state,
        )

    def _build_classifier(self) -> MLPClassifier:
        cfg = self.config
        return MLPClassifier(
            hidden_layer_sizes=cfg.clf_hidden_layers,
            activation=cfg.clf_activation,
            solver="adam",
            alpha=cfg.clf_alpha,
            max_iter=cfg.clf_max_iter,
            learning_rate_init=cfg.clf_learning_rate_init,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=cfg.random_state,
        )

    def train_predictor(self, x: np.ndarray, y: np.ndarray) -> None:
        x_scaled = self.regression_x_scaler.fit_transform(x)
        y_scaled = self.regression_y_scaler.fit_transform(y)
        self.predictor = self._build_regressor()
        self.predictor.fit(x_scaled, y_scaled)
        logger.info("Predictor training complete.")

    def predict_regression(self, x: np.ndarray) -> np.ndarray:
        if self.predictor is None:
            raise ValueError("Predictor is not trained.")
        x_scaled = self.regression_x_scaler.transform(x)
        y_scaled = self.predictor.predict(x_scaled)
        return self.regression_y_scaler.inverse_transform(y_scaled)

    def train_classifier(self, df: pd.DataFrame) -> None:
        x = df[self.feature_columns].to_numpy(copy=False)
        y = df["ClassCode"].to_numpy()
        x_scaled = self.feature_scaler.fit_transform(x)
        self.classifier = self._build_classifier()
        self.classifier.fit(x_scaled, y)
        logger.info("Classifier training complete.")

    def predict_classes_from_features(self, features: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise ValueError("Classifier is not trained.")
        scaled = self.feature_scaler.transform(features)
        return self.classifier.predict(scaled)

    def evaluate_holdout(self, df: pd.DataFrame, x_reg: np.ndarray, y_reg: np.ndarray) -> EvaluationMetrics:
        split = temporal_split_index(len(x_reg), self.config.holdout_fraction)

        x_train, y_train = x_reg[:split], y_reg[:split]
        x_test, y_test = x_reg[split:], y_reg[split:]

        regressor = self._build_regressor()
        x_train_s = self.regression_x_scaler.fit_transform(x_train)
        y_train_s = self.regression_y_scaler.fit_transform(y_train)
        regressor.fit(x_train_s, y_train_s)

        x_test_s = self.regression_x_scaler.transform(x_test)
        y_pred_s = regressor.predict(x_test_s)
        y_pred = self.regression_y_scaler.inverse_transform(y_pred_s)

        reg_mae, reg_r2, per_feature_mae = evaluate_regression(y_test, y_pred)

        clf_split = temporal_split_index(len(df), self.config.holdout_fraction)
        class_x = df[self.feature_columns].to_numpy(copy=False)
        class_y = df["ClassCode"].to_numpy()
        clf_scaler = StandardScaler()
        classifier = self._build_classifier()
        classifier.fit(clf_scaler.fit_transform(class_x[:clf_split]), class_y[:clf_split])
        pred_class = classifier.predict(clf_scaler.transform(class_x[clf_split:]))
        clf_acc = evaluate_classification(class_y[clf_split:], pred_class)

        return EvaluationMetrics(
            regression_mae=reg_mae,
            regression_r2=reg_r2,
            classification_accuracy=clf_acc,
            per_feature_mae=per_feature_mae,
        )

    def run_full_pipeline(
        self,
        start_year: int = 1997,
        end_year: int = 2015,
        progress: Optional[ProgressCallback] = None,
    ) -> TrainingResult:
        def report(message: str, fraction: float) -> None:
            if progress:
                progress(message, fraction)

        report("Loading historical weather data…", 0.05)
        df = self.load_data(start_year=start_year, end_year=end_year)

        report("Building regression dataset…", 0.2)
        x_reg, y_reg = self.prepare_regression_data(df)

        report("Estimating hold-out quality…", 0.35)
        holdout_metrics = self.evaluate_holdout(df, x_reg, y_reg)

        report("Training weather classifier…", 0.55)
        self.train_classifier(df)

        report("Training year-ahead regressor…", 0.75)
        self.train_predictor(x_reg, y_reg)

        report("Models ready.", 1.0)
        return TrainingResult(
            dataframe=df,
            regression_x=x_reg,
            regression_y=y_reg,
            holdout_metrics=holdout_metrics,
        )

    def predict_next_year(self, last_year_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictions = self.predict_regression(last_year_data)
        classes = self.predict_classes_from_features(predictions)
        return predictions, classes
