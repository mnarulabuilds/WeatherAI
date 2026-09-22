def test_shim_modules_expose_classes():
    from weather_engine import WeatherEngine as EngineFromShim
    from weather_visualizer import WeatherVisualizer as VizFromShim
    from weather_ai.engine import WeatherEngine
    from weather_ai.visualizer import WeatherVisualizer

    assert EngineFromShim is WeatherEngine
    assert VizFromShim is WeatherVisualizer


def test_desktop_app_entrypoint():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "desktop_app.py").read_text(encoding="utf-8")
    assert "ensure_tkinter_available" in text
