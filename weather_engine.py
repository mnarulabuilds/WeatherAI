import numpy as np
import pandas as pd
import os
import glob
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class WeatherEngine:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.feature_columns = [
            "MaxTemp", "MinTemp", "MaxDewPoint", "MinDewPoint",
            "MaxHumidity", "MinHumidity", "MaxPressure", "MinPressure",
            "MaxVisibility", "MinVisibility", "MeanWindSpeed"
        ]
        self.class_map = {
            "0001": "Thunderstorm",
            "0010": "Rainy",
            "0100": "Foggy",
            "1000": "Sunny"
        }
        self.predictor = None
        self.classifier = None
        self.scaler = StandardScaler()

    def load_data(self, start_year=1997, end_year=2015):
        """Loads WeatherXXXX.txt files and combines them into a DataFrame."""
        all_data = []
        for year in range(start_year, end_year + 1):
            file_path = os.path.join(self.data_dir, f"Weather{year}.txt")
            if os.path.exists(file_path):
                # The files are space-separated
                # Columns: 1, Features(11), ClassCode(4-char string)
                df = pd.read_csv(file_path, sep='\s+', header=None, dtype={12: str})
                df.columns = ["Bias"] + self.feature_columns + ["ClassCode"]
                df['Year'] = year
                all_data.append(df)
            else:
                print(f"Warning: {file_path} not found.")
        
        if not all_data:
            raise FileNotFoundError("No weather data files found.")
            
        return pd.concat(all_data, ignore_index=True)

    def prepare_regression_data(self, df):
        """Prepares data for predicting t+365 features from t."""
        # This replicates the logic in Predictor.m: map day N of Year Y to Day N of Year Y+1
        features = df[self.feature_columns].values
        
        X = []
        Y = []
        
        # Group by day and year
        # Assuming each year has 365 days for simplicity as in original code
        # Original code used m=365
        m = 365
        total_days = len(features)
        
        for i in range(total_days - m):
            X.append(features[i])
            Y.append(features[i+m])
            
        return np.array(X), np.array(Y)

    def train_predictor(self, X, y):
        print("Training Predictor (Regression)...")
        # MLPRegressor with 12 hidden units to match original hidden layer
        self.predictor = MLPRegressor(
            hidden_layer_sizes=(12,),
            activation='identity', # Original used no activation? Wait, check.
            solver='lbfgs',
            max_iter=5000,
            alpha=0.01 # regularization
        )
        # Note: Original code used linear-like forward prop for predictor?
        # a2 = [1; z1]; a3 = [1; z2]; where z1 = T1*a1. 
        # This is essentially multiple linear layers or identity activation.
        
        self.predictor.fit(X, y)
        print("Predictor training complete.")

    def train_classifier(self, df):
        print("Training Classifier...")
        X = df[self.feature_columns].values
        y = df['ClassCode'].values
        
        # Scale data for classifier
        X_scaled = self.scaler.fit_transform(X)
        
        # MLPClassifier
        # Original had 2 hidden layers of size K=4
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(10, 10), # Slightly larger for better convergence
            activation='logistic', # sigmoid
            solver='lbfgs',
            max_iter=5000, # Increased significantly
            alpha=0.01
        )
        self.classifier.fit(X_scaled, y)
        print("Classifier training complete.")

    def run_full_pipeline(self):
        df = self.load_data()
        
        # Classification
        self.train_classifier(df)
        
        # Regression
        X_reg, y_reg = self.prepare_regression_data(df)
        self.train_predictor(X_reg, y_reg)
        
        return df, X_reg, y_reg

    def predict_next_year(self, last_year_data):
        """Predicts weather features for the next year based on current data."""
        if self.predictor is None:
            raise ValueError("Model not trained.")
        
        # last_year_data should be (365, 11)
        predictions = self.predictor.predict(last_year_data)
        
        # Classify those predictions
        pred_scaled = self.scaler.transform(predictions)
        classes = self.classifier.predict(pred_scaled)
        
        return predictions, classes

if __name__ == "__main__":
    engine = WeatherEngine()
    df = engine.run_full_pipeline()
    
    # Test on final year
    last_year = df[df['Year'] == 2015][engine.feature_columns].values
    pred_features, pred_classes = engine.predict_next_year(last_year)
    
    print(f"Predicted {len(pred_features)} days for the upcoming year.")
    print(f"Sample prediction (Day 1): Features={pred_features[0]}, Class={pred_classes[0]}")
