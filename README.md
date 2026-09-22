# WeatherAI

Production-oriented Python application for weather feature prediction and classification using neural networks. The system learns from daily historical records (1997–2015), forecasts year-over-year feature trajectories, and classifies conditions as Thunderstorm, Rainy, Foggy, or Sunny.

## Highlights

- **Optimized ML pipeline** — vectorized dataset construction, float32 features, scaled regression targets, Adam + early stopping.
- **Honest metrics** — temporal hold-out evaluation before the final fit (MAE, R², classification accuracy).
- **Desktop studio UI** — CustomTkinter dashboard with metrics, activity log, theme toggle, and cross-platform plot folder access.
- **Tested** — pytest suite with ≥90% coverage (`pytest --cov`).

## Requirements

- Python 3.10+
- Dependencies in `requirements.txt`

## Install

```bash
pip install -r requirements.txt
```

## Run

On **macOS with Homebrew Python**, install Tk once (required for the GUI):

```bash
brew install python-tk@3.14   # match your python3 --version minor release
```

```bash
# Desktop app (recommended)
python desktop_app.py

# CLI engine
python weather_engine.py
```

## Tests & coverage

```bash
pytest
bash scripts/run_coverage.sh
```

Coverage is enforced at **≥90%** on the `weather_ai` package (engine, metrics, controller, visualizer). The CustomTkinter layout module (`weather_ai/app.py`) is excluded from the gate because headless CI often lacks `_tkinter`; UI flows are covered indirectly via `AppController` tests.

On Python 3.14, use `scripts/run_coverage.sh` (per-test coverage processes) instead of `pytest --cov`, which can conflict with NumPy’s import hooks.

## Project layout

```
weather_ai/          # Core package (engine, metrics, UI, visualizer)
tests/               # Unit tests
WeatherYYYY.txt      # Training data (1997–2015)
desktop_app.py       # App entry point
weather_engine.py    # CLI entry point (backward compatible)
```

## Dataset

Each `WeatherYYYY.txt` row contains a bias term, 11 numeric weather features, and a 4-character class code (`0001`, `0010`, `0100`, `1000`).
