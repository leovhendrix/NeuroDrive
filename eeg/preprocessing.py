"""
eeg/preprocessing.py

Detrend -> band-pass -> notch -> normalize.

Real-time constraint: filtfilt (zero-phase) needs future samples, which
don't exist in a live stream, so real-time mode uses causal `lfilter` with
persistent filter state across calls. Offline/training mode may use
zero-phase filtering since the whole recording is already available.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, lfilter, detrend

from config import PreprocessConfig


class RealtimeFilterState:
    """Holds per-channel filter state (zi) so causal filtering is continuous
    across successive windows instead of resetting to zero each call."""

    def __init__(self):
        self.bandpass_zi: Optional[np.ndarray] = None
        self.notch_zi: Dict[float, np.ndarray] = {}


class Preprocessor:
    def __init__(self, cfg: PreprocessConfig, sampling_rate: int):
        self.cfg = cfg
        self.fs = sampling_rate
        self._design_filters()

    def _design_filters(self):
        nyq = self.fs / 2.0
        low = self.cfg.bandpass_low_hz / nyq
        high = min(self.cfg.bandpass_high_hz / nyq, 0.99)
        self.bp_b, self.bp_a = butter(self.cfg.filter_order, [low, high], btype="band")

        self.notch_filters = []
        for f0 in self.cfg.notch_freqs_hz:
            if f0 >= nyq:
                continue
            b, a = iirnotch(f0, Q=30, fs=self.fs)
            self.notch_filters.append((f0, b, a))

    # -------------------- offline (training) path -----------------------
    def process_offline(self, data: np.ndarray) -> np.ndarray:
        """data: (n_channels, n_samples). Zero-phase filtering is fine here
        because the whole segment is available up front."""
        x = detrend(data, axis=-1, type="linear")
        x = filtfilt(self.bp_b, self.bp_a, x, axis=-1)
        for f0, b, a in self.notch_filters:
            x = filtfilt(b, a, x, axis=-1)
        x = self._normalize(x)
        return x

    # -------------------- real-time (causal) path ------------------------
    def process_realtime(self, data: np.ndarray, state: RealtimeFilterState) -> np.ndarray:
        """data: (n_channels, n_samples). Causal filtering with persistent
        state, so we never depend on samples that haven't arrived yet."""
        x = detrend(data, axis=-1, type="linear")

        n_ch = x.shape[0]
        if state.bandpass_zi is None:
            zi_single = self._lfilter_zi(self.bp_b, self.bp_a)
            state.bandpass_zi = np.tile(zi_single, (n_ch, 1))

        x_filtered = np.zeros_like(x)
        for ch in range(n_ch):
            x_filtered[ch], state.bandpass_zi[ch] = lfilter(
                self.bp_b, self.bp_a, x[ch], zi=state.bandpass_zi[ch]
            )
        x = x_filtered

        for f0, b, a in self.notch_filters:
            if f0 not in state.notch_zi:
                zi_single = self._lfilter_zi(b, a)
                state.notch_zi[f0] = np.tile(zi_single, (n_ch, 1))
            x_filtered = np.zeros_like(x)
            for ch in range(n_ch):
                x_filtered[ch], state.notch_zi[f0][ch] = lfilter(
                    b, a, x[ch], zi=state.notch_zi[f0][ch]
                )
            x = x_filtered

        x = self._normalize(x)
        return x

    @staticmethod
    def _lfilter_zi(b, a):
        from scipy.signal import lfilter_zi
        return lfilter_zi(b, a)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        std[std < 1e-9] = 1e-9
        return (x - mean) / std
