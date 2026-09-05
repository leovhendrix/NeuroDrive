"""
features/bandpower.py

Absolute and relative band-power features per channel per EEG band.
"""
import numpy as np
from scipy.signal import welch

from config import PreprocessConfig


class BandPowerExtractor:
    def __init__(self, cfg: PreprocessConfig, sampling_rate: int):
        self.cfg = cfg
        self.fs = sampling_rate

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # Stateless — nothing to fit — kept for interface consistency.
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: (n_epochs, n_channels, n_samples)
        Returns: (n_epochs, n_channels * n_bands * 2)  [absolute + relative power]
        """
        n_epochs, n_channels, n_samples = X.shape
        band_names = list(self.cfg.bands.keys())
        n_bands = len(band_names)
        feats = np.zeros((n_epochs, n_channels * n_bands * 2))

        for e in range(n_epochs):
            freqs, psd = welch(X[e], fs=self.fs, axis=-1, nperseg=min(n_samples, 256))
            total_power = np.sum(psd, axis=-1) + 1e-12  # (n_channels,)

            col = 0
            for band_name in band_names:
                lo, hi = self.cfg.bands[band_name]
                mask = (freqs >= lo) & (freqs <= hi)
                band_power = np.sum(psd[:, mask], axis=-1)  # (n_channels,)
                rel_power = band_power / total_power

                feats[e, col:col + n_channels] = band_power
                col += n_channels
                feats[e, col:col + n_channels] = rel_power
                col += n_channels

        return feats
