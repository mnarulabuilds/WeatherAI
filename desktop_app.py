"""Launch the WeatherAI desktop application."""

from weather_ai.runtime_check import ensure_tkinter_available


def main() -> None:
    ensure_tkinter_available()
    from weather_ai.app import main as run_app

    run_app()


if __name__ == "__main__":
    main()
