"""
eeg/quality.py

Signal-quality checks. A window is only considered usable for classification
if it passes ALL relevant checks; otherwise it is marked unreliable and the
safety controller must not allow a movement command through.
"""
from dataclasses import dataclass
import numpy as np

from config import QualityConfig


@dataclass
class QualityReport:
    ok: bool
    score: float  # 0..1, overall quality estimate
    reasons: list  # list of strings describing any failures


def check_quality(window: np.ndarray, fs: int, cfg: QualityConfig) -> QualityReport:
    """window: (n_channels, n_samples), assumed already band-passed/notched."""
    reasons = []
    scores = []

    # --- amplitude / saturation ---
    max_abs = np.max(np.abs(window), axis=-1)
    saturated = max_abs > cfg.max_abs_amplitude_uv
    if np.any(saturated):
        reasons.append(f"saturation/large-amplitude on {int(saturated.sum())} channel(s)")
    scores.append(1.0 - saturated.mean())

    # --- flatline / disconnect (near-zero variance sustained) ---
    variance = np.var(window, axis=-1)
    flat = variance < cfg.min_variance_uv2
    if np.any(flat):
        reasons.append(f"possible electrode disconnect on {int(flat.sum())} channel(s)")
    scores.append(1.0 - flat.mean())

    # --- excessive noise (variance way above expected physiological range) ---
    noisy = variance > cfg.max_variance_uv2
    if np.any(noisy):
        reasons.append(f"excessive noise on {int(noisy.sum())} channel(s)")
    scores.append(1.0 - noisy.mean())

    # --- crude eye-blink / low-frequency contamination check ---
    blink_ratio = _low_freq_power_ratio(window, fs, cfg.blink_band_hz)
    blink_bad = blink_ratio > cfg.blink_power_ratio_threshold
    if blink_bad:
        reasons.append(f"possible eye-blink/EOG contamination (low-freq power ratio={blink_ratio:.2f})")
    scores.append(0.0 if blink_bad else 1.0)

    overall = float(np.mean(scores)) if scores else 0.0
    ok = overall >= cfg.quality_pass_threshold and len(reasons) == 0
    return QualityReport(ok=ok, score=overall, reasons=reasons)


def _low_freq_power_ratio(window: np.ndarray, fs: int, band: tuple) -> float:
    from scipy.signal import welch
    freqs, psd = welch(window, fs=fs, axis=-1, nperseg=min(window.shape[-1], 256))
    total_power = np.sum(psd, axis=-1) + 1e-12
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.sum(psd[:, band_mask], axis=-1)
    ratio = band_power / total_power
    return float(np.mean(ratio))
