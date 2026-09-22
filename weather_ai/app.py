from __future__ import annotations

import os
import threading
import tkinter.messagebox as messagebox

import customtkinter as ctk
from PIL import Image

from weather_ai.app_controller import AppController
from weather_ai.constants import DEFAULT_END_YEAR, DEFAULT_START_YEAR
from weather_ai.engine import WeatherEngine
from weather_ai.platform_util import open_path_in_file_manager, parse_year_range
from weather_ai.visualizer import WeatherVisualizer

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class WeatherApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("WeatherAI — Prediction Studio")
        self.geometry("1280x820")
        self.minsize(1024, 680)

        self.engine = WeatherEngine()
        self.visualizer = WeatherVisualizer(output_dir="app_plots")
        self.controller = AppController(self.engine, self.visualizer)
        self._preview_image: ctk.CTkImage | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="WeatherAI",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(pady=(28, 8), padx=20, anchor="w")
        ctk.CTkLabel(
            sidebar,
            text="Neural weather forecasting",
            font=ctk.CTkFont(size=13),
            text_color=("gray40", "gray70"),
        ).pack(pady=(0, 24), padx=20, anchor="w")

        year_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        year_frame.pack(fill="x", padx=20, pady=(0, 12))
        ctk.CTkLabel(year_frame, text="Training years", anchor="w").pack(fill="x")
        row = ctk.CTkFrame(year_frame, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))
        self.start_year_entry = ctk.CTkEntry(row, width=90, placeholder_text=str(DEFAULT_START_YEAR))
        self.start_year_entry.pack(side="left", padx=(0, 8))
        self.end_year_entry = ctk.CTkEntry(row, width=90, placeholder_text=str(DEFAULT_END_YEAR))
        self.end_year_entry.pack(side="left")

        self.train_btn = ctk.CTkButton(sidebar, text="Train models", command=self._start_training)
        self.train_btn.pack(fill="x", padx=20, pady=8)

        self.predict_btn = ctk.CTkButton(
            sidebar,
            text="Run evaluation & plots",
            command=self._run_predictions,
            state="disabled",
        )
        self.predict_btn.pack(fill="x", padx=20, pady=8)

        self.open_plots_btn = ctk.CTkButton(
            sidebar,
            text="Open plots folder",
            command=self._open_plot_folder,
            state="disabled",
        )
        self.open_plots_btn.pack(fill="x", padx=20, pady=8)

        self.appearance_menu = ctk.CTkOptionMenu(
            sidebar,
            values=["System", "Light", "Dark"],
            command=self._change_appearance,
        )
        self.appearance_menu.pack(fill="x", padx=20, pady=(24, 8))
        self.appearance_menu.set("System")

        content = ctk.CTkFrame(self, corner_radius=12)
        content.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(content, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        header.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            header,
            text="Ready to train",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.progress = ctk.CTkProgressBar(header)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.progress.set(0)

        self.tabs = ctk.CTkTabview(content)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=16, pady=8)
        self.tabs.add("Dashboard")
        self.tabs.add("Metrics")
        self.tabs.add("Activity log")

        self.preview_label = ctk.CTkLabel(
            self.tabs.tab("Dashboard"),
            text="Train the models to see evaluation charts here.",
            justify="center",
        )
        self.preview_label.pack(expand=True, fill="both", padx=12, pady=12)

        self.metrics_box = ctk.CTkTextbox(self.tabs.tab("Metrics"), font=ctk.CTkFont(family="Menlo", size=13))
        self.metrics_box.pack(expand=True, fill="both", padx=12, pady=12)
        self.metrics_box.insert("end", "Hold-out and full-run metrics will appear after training.\n")
        self.metrics_box.configure(state="disabled")

        self.log_box = ctk.CTkTextbox(self.tabs.tab("Activity log"), height=200)
        self.log_box.pack(expand=True, fill="both", padx=12, pady=12)

    def _change_appearance(self, mode: str) -> None:
        ctk.set_appearance_mode(mode)

    def _log(self, message: str) -> None:
        self.log_box.insert("end", f"• {message}\n")
        self.log_box.see("end")

    def _set_metrics_text(self, text: str) -> None:
        self.metrics_box.configure(state="normal")
        self.metrics_box.delete("1.0", "end")
        self.metrics_box.insert("end", text)
        self.metrics_box.configure(state="disabled")

    def _open_plot_folder(self) -> None:
        open_path_in_file_manager(self.visualizer.output_dir)

    def _start_training(self) -> None:
        try:
            start_year, end_year = parse_year_range(
                self.start_year_entry.get(),
                self.end_year_entry.get(),
                DEFAULT_START_YEAR,
                DEFAULT_END_YEAR,
            )
        except ValueError as exc:
            messagebox.showerror("Invalid year range", str(exc))
            return

        self.train_btn.configure(state="disabled")
        self.predict_btn.configure(state="disabled")
        self.open_plots_btn.configure(state="disabled")
        self.status_label.configure(text="Training in progress…")
        self.progress.set(0.05)
        self._log(f"Training on years {start_year}–{end_year}…")

        thread = threading.Thread(
            target=self._train_worker,
            args=(start_year, end_year),
            daemon=True,
        )
        thread.start()

    def _train_worker(self, start_year: int, end_year: int) -> None:
        try:

            def progress(message: str, fraction: float) -> None:
                self.after(0, lambda: self.progress.set(fraction))
                self.after(0, lambda m=message: self._log(m))

            result = self.controller.train(start_year, end_year, progress=progress)
            holdout = result.holdout_metrics
            metrics_text = "\n".join(
                [
                    "Hold-out quality (temporal split, not used for final fit):",
                    *holdout.summary_lines(),
                    "",
                    "Per-feature MAE (hold-out):",
                    *[f"  - {label}: {value:.3f}" for label, value in zip(self.engine.feature_columns, holdout.per_feature_mae)],
                ]
            )
            self.after(0, lambda t=metrics_text: self._set_metrics_text(t))
            self.after(0, self._training_finished)
        except Exception as exc:
            message = str(exc)
            self.after(0, lambda m=message: messagebox.showerror("Training error", m))
            self.after(0, lambda: self.train_btn.configure(state="normal"))

    def _training_finished(self) -> None:
        self.status_label.configure(text="Models trained — ready for evaluation")
        self.progress.set(1.0)
        self.train_btn.configure(state="normal")
        self.predict_btn.configure(state="normal")
        self._log("Training complete.")

    def _run_predictions(self) -> None:
        self.status_label.configure(text="Generating predictions and charts…")
        self.predict_btn.configure(state="disabled")
        thread = threading.Thread(target=self._predict_worker, daemon=True)
        thread.start()

    def _predict_worker(self) -> None:
        try:
            artifacts = self.controller.run_predictions(plot_all_features=True)
            lines = [
                "Full evaluation on year-over-year pairs:",
                *artifacts.metrics.summary_lines(),
                "",
                f"Class chart: {artifacts.class_plot_path}",
                f"Feature charts: {len(artifacts.feature_plot_paths)} files in {self.visualizer.output_dir}",
            ]
            text = "\n".join(lines)
            self.after(0, lambda t=text: self._set_metrics_text(t))
            self.after(0, lambda p=artifacts.primary_preview_path: self._show_preview(p))
            self.after(0, self._predictions_finished)
        except Exception as exc:
            message = str(exc)
            self.after(0, lambda m=message: messagebox.showerror("Prediction error", m))
            self.after(0, lambda: self.predict_btn.configure(state="normal"))

    def _show_preview(self, image_path: str) -> None:
        if not os.path.exists(image_path):
            return
        image = Image.open(image_path)
        self._preview_image = ctk.CTkImage(light_image=image, dark_image=image, size=(900, 420))
        self.preview_label.configure(image=self._preview_image, text="")

    def _predictions_finished(self) -> None:
        self.status_label.configure(text="Evaluation complete")
        self.predict_btn.configure(state="normal")
        self.open_plots_btn.configure(state="normal")
        self.tabs.set("Dashboard")
        self._log("Charts exported to app_plots/.")


def main() -> None:
    app = WeatherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
