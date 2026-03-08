# Weather-Prediction-Using-Neural-Networks

A modern Python-based Machine Learning project for weather feature prediction and classification. The system uses Neural Networks to classify weather into Thunderstorm, Rainy, Foggy, and Sunny categories, and predicts future weather features (temperatures, humidity, pressure, etc.) based on historical data.

## Features
- **Neural Network Prediction**: Predicts future weather parameters using multi-layer perceptrons.
- **Weather Classification**: Categorizes daily weather into four distinct types.
- **Desktop Application**: Modern GUI built with `CustomTkinter` for easy interaction.
- **Data Visualization**: Comparison charts generated with `Matplotlib`.

## Prerequisites
- Python 3.10 or higher
- Pip (Python package installer)

## Installation

1. Clone or download the repository.
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Desktop App (Recommended)
Launch the professional GUI to train models and view predictions:
```bash
python desktop_app.py
```

### Running the Engine (CLI)
You can also interact with the core engine directly:
```bash
python weather_engine.py
```

## Dataset
The project includes historical weather data from 1997 to 2015 (`WeatherXXXX.txt` files) used for training the neural networks.

## Old Project Files
The original Octave and C++ files have been superseded by this Python migration for better performance, maintainability, and user experience.
