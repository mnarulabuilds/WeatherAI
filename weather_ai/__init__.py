"""WeatherAI — neural weather prediction and classification."""

__all__ = ["WeatherEngine", "TrainingResult", "EvaluationMetrics"]


def __getattr__(name: str):
    if name == "WeatherEngine":
        from weather_ai.engine import WeatherEngine

        return WeatherEngine
    if name == "TrainingResult":
        from weather_ai.engine import TrainingResult

        return TrainingResult
    if name == "EvaluationMetrics":
        from weather_ai.metrics import EvaluationMetrics

        return EvaluationMetrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
