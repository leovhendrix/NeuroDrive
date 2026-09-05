"""
gui/dashboard.py

Lightweight Tkinter dashboard. Reads state from a thread-safe queue that the
real-time loop pushes updates into, so EEG acquisition/classification is
never blocked waiting on the GUI event loop.
"""
import queue
import tkinter as tk
from tkinter import ttk
from collections import deque

import numpy as np


class Dashboard:
    def __init__(self, update_queue: "queue.Queue", on_emergency_stop):
        self.q = update_queue
        self.on_emergency_stop = on_emergency_stop

        self.root = tk.Tk()
        self.root.title("BCI Robot Control Dashboard")
        self.root.geometry("700x600")
        self.root.bind("<space>", lambda e: self.on_emergency_stop())

        self._build_widgets()
        self._waveform_buffer = deque(maxlen=500)

        self.root.after(100, self._poll_queue)

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 4}

        status_frame = ttk.LabelFrame(self.root, text="Status")
        status_frame.pack(fill="x", **pad)

        self.lbl_eeg_conn = self._row(status_frame, "EEG connection:")
        self.lbl_quality = self._row(status_frame, "Signal quality:")
        self.lbl_command = self._row(status_frame, "Predicted command:")
        self.lbl_confidence = self._row(status_frame, "Confidence:")
        self.lbl_robot_state = self._row(status_frame, "Robot state:")
        self.lbl_estop = self._row(status_frame, "Emergency stop:")

        latency_frame = ttk.LabelFrame(self.root, text="Latency (ms)")
        latency_frame.pack(fill="x", **pad)
        self.lbl_latency = self._row(latency_frame, "Total / acq / proc / clf:")
        self.lbl_fps = self._row(latency_frame, "Sample rate / FPS:")

        wave_frame = ttk.LabelFrame(self.root, text="EEG waveform (most recent channel)")
        wave_frame.pack(fill="both", expand=True, **pad)
        self.canvas = tk.Canvas(wave_frame, bg="black", height=200)
        self.canvas.pack(fill="both", expand=True)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", **pad)
        estop_btn = tk.Button(
            btn_frame, text="EMERGENCY STOP (SPACE)", bg="red", fg="white",
            font=("Arial", 14, "bold"), command=self.on_emergency_stop,
        )
        estop_btn.pack(fill="x", pady=8)

    def _row(self, parent, label_text):
        row = ttk.Frame(parent)
        row.pack(fill="x")
        ttk.Label(row, text=label_text, width=28).pack(side="left")
        value_lbl = ttk.Label(row, text="--")
        value_lbl.pack(side="left")
        return value_lbl

    def _poll_queue(self):
        try:
            while True:
                update = self.q.get_nowait()
                self._apply_update(update)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _apply_update(self, update: dict):
        if "eeg_connected" in update:
            self.lbl_eeg_conn.config(text="CONNECTED" if update["eeg_connected"] else "LOST",
                                      foreground="green" if update["eeg_connected"] else "red")
        if "signal_quality" in update:
            self.lbl_quality.config(text=f"{update['signal_quality']*100:.0f}%")
        if "command" in update:
            self.lbl_command.config(text=update["command"])
        if "confidence" in update:
            self.lbl_confidence.config(text=f"{update['confidence']*100:.0f}%")
        if "robot_state" in update:
            self.lbl_robot_state.config(text=update["robot_state"])
        if "emergency_stop" in update:
            self.lbl_estop.config(text="ACTIVE" if update["emergency_stop"] else "clear",
                                   foreground="red" if update["emergency_stop"] else "green")
        if "latency_ms" in update:
            lat = update["latency_ms"]
            self.lbl_latency.config(
                text=f"{lat.get('total_ms', 0):.0f} / {lat.get('acquisition_ms', 0):.0f} / "
                     f"{lat.get('processing_ms', 0):.0f} / {lat.get('classification_ms', 0):.0f}"
            )
        if "fps" in update:
            self.lbl_fps.config(text=f"{update['fps']:.1f} Hz")
        if "waveform" in update:
            self._draw_waveform(update["waveform"])

    def _draw_waveform(self, samples: np.ndarray):
        self.canvas.delete("wave")
        w = self.canvas.winfo_width() or 600
        h = self.canvas.winfo_height() or 200
        if len(samples) < 2:
            return
        norm = samples - np.mean(samples)
        max_abs = np.max(np.abs(norm)) + 1e-9
        norm = norm / max_abs * (h / 2 - 10)

        points = []
        for i, v in enumerate(norm):
            x = i / len(norm) * w
            y = h / 2 - v
            points.append((x, y))

        for i in range(len(points) - 1):
            self.canvas.create_line(*points[i], *points[i + 1], fill="lime", tags="wave")

    def run(self):
        self.root.mainloop()
