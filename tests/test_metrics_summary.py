from weather_ai.metrics import EvaluationMetrics


def test_evaluation_metrics_summary():
    metrics = EvaluationMetrics(
        regression_mae=1.5,
        regression_r2=0.8,
        classification_accuracy=0.9,
        per_feature_mae=(1.0, 2.0),
    )
    lines = metrics.summary_lines()
    assert any("MAE" in line for line in lines)
    assert any("accuracy" in line for line in lines)
