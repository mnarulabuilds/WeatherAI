import pytest

from weather_ai.app_controller import AppController
from weather_ai.engine import WeatherEngine
from weather_ai.visualizer import WeatherVisualizer


def test_controller_train_and_predict(sample_data_dir, fast_config, tmp_path):
    engine = WeatherEngine(data_dir=str(sample_data_dir), config=fast_config)
    viz = WeatherVisualizer(output_dir=str(tmp_path / "plots"))
    controller = AppController(engine, viz)

    controller.train(2001, 2003)
    artifacts = controller.run_predictions(plot_all_features=False)

    assert artifacts.metrics.regression_mae >= 0.0
    assert artifacts.primary_preview_path.endswith("Classes_Comparison.png")


def test_controller_predict_without_train_raises(sample_data_dir, fast_config, tmp_path):
    controller = AppController(
        WeatherEngine(data_dir=str(sample_data_dir), config=fast_config),
        WeatherVisualizer(output_dir=str(tmp_path / "plots")),
    )
    with pytest.raises(RuntimeError):
        controller.run_predictions()
