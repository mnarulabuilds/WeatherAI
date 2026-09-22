import numpy as np
import pandas as pd
import pytest

from weather_ai.engine import WeatherEngine


def test_load_data_missing_dir(tmp_path):
    engine = WeatherEngine(data_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        engine.load_data(1990, 1991)


def test_prepare_regression_vectorized(sample_data_dir, fast_config):
    engine = WeatherEngine(data_dir=str(sample_data_dir), config=fast_config)
    df = engine.load_data(2001, 2003)
    x, y = engine.prepare_regression_data(df)
    assert x.shape == y.shape
    assert x.shape[1] == len(engine.feature_columns)
    assert len(x) == len(df) - engine.days_per_year


def test_run_full_pipeline_and_predict_next_year(sample_data_dir, fast_config):
    engine = WeatherEngine(data_dir=str(sample_data_dir), config=fast_config)
    progress_calls = []

    def progress(message, fraction):
        progress_calls.append((message, fraction))

    result = engine.run_full_pipeline(2001, 2003, progress=progress)
    assert isinstance(result.dataframe, pd.DataFrame)
    assert result.holdout_metrics.classification_accuracy >= 0.0
    assert progress_calls[-1][1] == 1.0

    last_year = result.dataframe[result.dataframe["Year"] == 2003][engine.feature_columns].to_numpy()
    preds, classes = engine.predict_next_year(last_year)
    assert preds.shape == (365, len(engine.feature_columns))
    assert len(classes) == 365


def test_prepare_regression_requires_enough_rows(tmp_path, fast_config):
    engine = WeatherEngine(data_dir=str(tmp_path), config=fast_config)
    from tests.conftest import write_weather_year

    write_weather_year(tmp_path / "Weather2001.txt", 2001, rows=100)
    df = engine.load_data(2001, 2001)
    with pytest.raises(ValueError):
        engine.prepare_regression_data(df)


def test_predict_before_train_raises(sample_data_dir, fast_config):
    engine = WeatherEngine(data_dir=str(sample_data_dir), config=fast_config)
    x = np.zeros((2, len(engine.feature_columns)))
    with pytest.raises(ValueError):
        engine.predict_regression(x)
    with pytest.raises(ValueError):
        engine.predict_classes_from_features(x)


def test_load_data_warns_on_missing_year(sample_data_dir, fast_config, caplog):
    engine = WeatherEngine(data_dir=str(sample_data_dir), config=fast_config)
    import logging

    with caplog.at_level(logging.WARNING):
        engine.load_data(2000, 2003)
    assert any("Missing data file" in record.message for record in caplog.records)
