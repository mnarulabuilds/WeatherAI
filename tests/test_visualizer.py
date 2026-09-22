import numpy as np
import pytest

from weather_ai.visualizer import WeatherVisualizer


def test_plot_outputs(tmp_path):
    viz = WeatherVisualizer(output_dir=str(tmp_path))
    actual = np.linspace(0, 1, 50)[:, None] * np.ones((50, 11))
    pred = actual + 0.01

    class_path = viz.plot_classes(
        np.array(["1000", "0010"] * 25),
        np.array(["1000", "0100"] * 25),
    )
    feature_path = viz.plot_comparison(actual, pred, 0)
    all_paths = viz.plot_all_features(actual, pred)

    assert class_path.endswith("Classes_Comparison.png")
    assert feature_path.endswith("Max_Temperature.png")
    assert len(all_paths) == 11


def test_plot_comparison_validates_shape(tmp_path):
    viz = WeatherVisualizer(output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        viz.plot_comparison(np.zeros((2, 11)), np.zeros((3, 11)), 0)
    with pytest.raises(ValueError):
        viz.plot_comparison(np.zeros((2, 11)), np.zeros((2, 11)), 99)
