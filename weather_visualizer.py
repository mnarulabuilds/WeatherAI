import matplotlib.pyplot as plt
import numpy as np
import os

class WeatherVisualizer:
    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.feature_labels = [
            "Max Temperature", "Min Temperature", "Max DewPoint", "Min DewPoint",
            "Max Humidity", "Min Humidity", "Max Pressure", "Min Pressure",
            "Max Visibility", "Min Visibility", "Mean Wind Speed"
        ]

    def plot_comparison(self, actual, predicted, feature_idx, title=None):
        plt.figure(figsize=(12, 6))
        days = np.arange(len(actual))
        
        plt.plot(days, actual[:, feature_idx], 'g*', label='Actual', markersize=4)
        plt.plot(days, predicted[:, feature_idx], 'r@', label='Predicted', alpha=0.6)
        
        label = self.feature_labels[feature_idx]
        plt.title(title or f"Comparison: {label}")
        plt.xlabel("Day")
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        filename = f"{label.replace(' ', '_')}.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        print(f"Saved plot: {filename}")

    def plot_classes(self, actual_codes, predicted_codes):
        # Map codes to numeric values for plotting
        code_map = {"0001": 1, "0010": 2, "0100": 3, "1000": 4}
        label_map = {1: "Thunderstorm", 2: "Rainy", 3: "Foggy", 4: "Sunny"}
        
        y_actual = [code_map.get(c, 4) for c in actual_codes]
        y_pred = [code_map.get(c, 4) for c in predicted_codes]
        
        plt.figure(figsize=(12, 6))
        days = np.arange(len(y_actual))
        
        plt.plot(days, y_actual, 'b^', label='Actual', markersize=6)
        plt.plot(days, y_pred, 'r@', label='Predicted', alpha=0.6)
        
        plt.yticks(list(label_map.keys()), list(label_map.values()))
        plt.title("Weather Class Comparison")
        plt.xlabel("Day")
        plt.ylabel("Event")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(self.output_dir, "Classes_Comparison.png"))
        plt.close()
        print("Saved plot: Classes_Comparison.png")

if __name__ == "__main__":
    # Mock data for testing
    viz = WeatherVisualizer()
    actual = np.random.rand(100, 11) * 30
    pred = actual + np.random.normal(0, 2, (100, 11))
    viz.plot_comparison(actual, pred, 0)
