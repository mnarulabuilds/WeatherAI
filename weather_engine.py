"""Backward-compatible entry point for the core engine."""

from weather_ai.engine import TrainingResult, WeatherEngine

__all__ = ["WeatherEngine", "TrainingResult"]

if __name__ == "__main__":
    engine = WeatherEngine()
    result = engine.run_full_pipeline()
    df = result.dataframe

    last_year = df[df["Year"] == df["Year"].max()][engine.feature_columns].to_numpy()
    pred_features, pred_classes = engine.predict_next_year(last_year)

    print(f"Predicted {len(pred_features)} days for the upcoming year.")
    print(f"Sample prediction (day 1): class={pred_classes[0]}")
