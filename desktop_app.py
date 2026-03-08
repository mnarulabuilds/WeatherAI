import customtkinter as ctk
from weather_engine import WeatherEngine
from weather_visualizer import WeatherVisualizer
import threading
import tkinter.messagebox as messagebox
from PIL import Image
import os

class WeatherApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Weather Prediction System")
        self.geometry("1100x700") # Wider for better visualization
        
        # Initialize Engine
        self.engine = WeatherEngine()
        self.viz = WeatherVisualizer(output_dir="app_plots")
        
        self.training_data = None #(df, X_reg, y_reg)

        # UI Layout
        self.setup_ui()

    def setup_ui(self):
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="WeatherAI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(pady=30)
        
        self.train_btn = ctk.CTkButton(self.sidebar, text="Train Models", command=self.start_training)
        self.train_btn.pack(pady=10, padx=20)
        
        self.predict_btn = ctk.CTkButton(self.sidebar, text="Run Predictions", command=self.run_predictions, state="disabled")
        self.predict_btn.pack(pady=10, padx=20)

        self.open_plots_btn = ctk.CTkButton(self.sidebar, text="Open Plots Folder", command=self.open_plot_folder, state="disabled")
        self.open_plots_btn.pack(pady=10, padx=20)

        # Main Content
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        
        self.status_label = ctk.CTkLabel(self.main_frame, text="System Ready", font=ctk.CTkFont(size=16))
        self.status_label.pack(pady=10)
        
        self.progress = ctk.CTkProgressBar(self.main_frame)
        self.progress.pack(pady=5, padx=20, fill="x")
        self.progress.set(0)

        # Dashboard / Image Area
        self.image_container = ctk.CTkFrame(self.main_frame)
        self.image_container.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.image_label = ctk.CTkLabel(self.image_container, text="Graphs will appear here after prediction")
        self.image_label.pack(expand=True)

        self.log_box = ctk.CTkTextbox(self.main_frame, height=150)
        self.log_box.pack(pady=10, padx=20, fill="x")

    def log(self, message):
        self.log_box.insert("end", f"> {message}\n")
        self.log_box.see("end")

    def open_plot_folder(self):
        os.startfile(os.path.abspath(self.viz.output_dir))

    def start_training(self):
        self.train_btn.configure(state="disabled")
        self.status_label.configure(text="Training Neural Networks...")
        self.progress.set(0.1)
        
        thread = threading.Thread(target=self.train_thread)
        thread.start()

    def train_thread(self):
        try:
            self.log("Step 1: Loading 20 years of weather data...")
            self.training_data = self.engine.run_full_pipeline()
            # training_data is (df, X_reg, y_reg)
            self.progress.set(1.0)
            self.log("Step 2: Training both Predictor and Classifier complete.")
            self.after(0, self.training_done)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Training Error", str(e)))
            self.after(0, lambda: self.train_btn.configure(state="normal"))

    def training_done(self):
        self.status_label.configure(text="Model Ready")
        self.predict_btn.configure(state="normal")
        self.train_btn.configure(state="normal")
        self.log("Ready to generate predictions.")

    def run_predictions(self):
        self.status_label.configure(text="Generating Visualizations...")
        self.log("Calculating predictions and generating comparison charts...")
        
        # We'll run visualization on a thread too
        thread = threading.Thread(target=self.predict_thread)
        thread.start()

    def predict_thread(self):
        try:
            df, X_reg, y_reg = self.training_data
            
            # Predict
            pred_features = self.engine.predictor.predict(X_reg)
            pred_classes = self.engine.classifier.predict(self.engine.scaler.transform(pred_features))
            
            actual_classes = df['ClassCode'].values[365:] # Match the offset
            
            # Generate Plots
            # 1. Classes plot
            self.viz.plot_classes(actual_classes, pred_classes)
            
            # 2. Key feature comparison (Max Temp)
            self.viz.plot_comparison(y_reg, pred_features, 0) # Max Temp
            
            self.after(0, self.display_results)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Prediction Error", str(e)))

    def display_results(self):
        # Load the newly created image
        img_path = os.path.join(self.viz.output_dir, "Classes_Comparison.png")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            # Resize image to fit the container
            # Container is roughly 800x400
            img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(750, 350))
            self.image_label.configure(image=img_ctk, text="")
            
        self.status_label.configure(text="Predictions Complete")
        self.open_plots_btn.configure(state="normal")
        self.log("Results visualised. You can also click 'Open Plots Folder' to see detailed charts for all features.")

if __name__ == "__main__":
    app = WeatherApp()
    app.mainloop()
