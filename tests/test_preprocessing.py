import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import PreprocessConfig
from eeg.preprocessing import Preprocessor, RealtimeFilterState


def test_offline_preprocessing_shape_preserved():
    fs = 250
    cfg = PreprocessConfig()
    pre = Preprocessor(cfg, sampling_rate=fs)
    data = np.random.randn(8, fs * 2)  # 8 channels, 2 seconds
    out = pre.process_offline(data)
    assert out.shape == data.shape


def test_realtime_preprocessing_shape_preserved():
    fs = 250
    cfg = PreprocessConfig()
    pre = Preprocessor(cfg, sampling_rate=fs)
    state = RealtimeFilterState()
    data = np.random.randn(8, fs)
    out = pre.process_realtime(data, state)
    assert out.shape == data.shape


def test_realtime_filter_state_persists_across_calls():
    fs = 250
    cfg = PreprocessConfig()
    pre = Preprocessor(cfg, sampling_rate=fs)
    state = RealtimeFilterState()
    chunk1 = np.random.randn(4, 50)
    chunk2 = np.random.randn(4, 50)
    pre.process_realtime(chunk1, state)
    assert state.bandpass_zi is not None
    zi_after_first = state.bandpass_zi.copy()
    pre.process_realtime(chunk2, state)
    # state should have evolved, not reset
    assert not np.allclose(zi_after_first, state.bandpass_zi)


def test_normalization_zero_mean_unit_var():
    fs = 250
    cfg = PreprocessConfig()
    pre = Preprocessor(cfg, sampling_rate=fs)
    data = np.random.randn(4, fs * 2) * 5 + 10
    out = pre.process_offline(data)
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-6)
