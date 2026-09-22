import weather_ai


def test_lazy_exports():
    assert weather_ai.WeatherEngine.__name__ == "WeatherEngine"
    assert weather_ai.TrainingResult.__name__ == "TrainingResult"
    assert weather_ai.EvaluationMetrics.__name__ == "EvaluationMetrics"
