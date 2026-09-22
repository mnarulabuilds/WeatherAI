FEATURE_COLUMNS = [
    "MaxTemp",
    "MinTemp",
    "MaxDewPoint",
    "MinDewPoint",
    "MaxHumidity",
    "MinHumidity",
    "MaxPressure",
    "MinPressure",
    "MaxVisibility",
    "MinVisibility",
    "MeanWindSpeed",
]

CLASS_MAP = {
    "0001": "Thunderstorm",
    "0010": "Rainy",
    "0100": "Foggy",
    "1000": "Sunny",
}

CLASS_CODE_TO_PLOT_VALUE = {"0001": 1, "0010": 2, "0100": 3, "1000": 4}

PLOT_VALUE_TO_LABEL = {1: "Thunderstorm", 2: "Rainy", 3: "Foggy", 4: "Sunny"}

FEATURE_LABELS = [
    "Max Temperature",
    "Min Temperature",
    "Max DewPoint",
    "Min DewPoint",
    "Max Humidity",
    "Min Humidity",
    "Max Pressure",
    "Min Pressure",
    "Max Visibility",
    "Min Visibility",
    "Mean Wind Speed",
]

DEFAULT_DAYS_PER_YEAR = 365
DEFAULT_START_YEAR = 1997
DEFAULT_END_YEAR = 2015
