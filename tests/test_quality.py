import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import QualityConfig
from eeg.quality import check_quality


def test_clean_signal_passes():
    fs = 250
    cfg = QualityConfig()
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 5, size=(8, fs * 2))
    report = check_quality(clean, fs, cfg)
    assert isinstance(report.score, float)


def test_flatline_fails():
    fs = 250
    cfg = QualityConfig()
    flat = np.zeros((8, fs * 2))
    report = check_quality(flat, fs, cfg)
    assert not report.ok
    assert any("disconnect" in r for r in report.reasons)


def test_saturated_signal_fails():
    fs = 250
    cfg = QualityConfig()
    rng = np.random.default_rng(1)
    saturated = rng.normal(0, 5, size=(8, fs * 2))
    saturated[0, :] = 500.0  # way above max_abs_amplitude_uv
    report = check_quality(saturated, fs, cfg)
    assert not report.ok
    assert any("saturation" in r for r in report.reasons)
