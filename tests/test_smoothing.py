import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DecisionConfig
from realtime.smoothing import TemporalSmoother


def make_cfg():
    return DecisionConfig(min_confidence=0.8, required_consistent_windows=3,
                           prediction_history_len=10, idle_command="STOP")


def test_low_confidence_never_emits_command():
    cfg = make_cfg()
    smoother = TemporalSmoother(cfg)
    decision = smoother.update("LEFT", confidence=0.5)
    assert decision.command == "STOP"


def test_requires_consistent_windows_before_acting():
    cfg = make_cfg()
    smoother = TemporalSmoother(cfg)
    d1 = smoother.update("LEFT", confidence=0.9)
    d2 = smoother.update("LEFT", confidence=0.9)
    assert d1.command == "STOP"
    assert d2.command == "STOP"
    d3 = smoother.update("LEFT", confidence=0.9)
    assert d3.command == "LEFT"


def test_inconsistent_predictions_reset_streak():
    cfg = make_cfg()
    smoother = TemporalSmoother(cfg)
    smoother.update("LEFT", confidence=0.9)
    smoother.update("LEFT", confidence=0.9)
    d = smoother.update("RIGHT", confidence=0.9)  # breaks streak
    assert d.command == "STOP"


def test_low_confidence_clears_history():
    cfg = make_cfg()
    smoother = TemporalSmoother(cfg)
    smoother.update("LEFT", confidence=0.9)
    smoother.update("LEFT", confidence=0.9)
    smoother.update("LEFT", confidence=0.5)  # low confidence clears
    d = smoother.update("LEFT", confidence=0.9)
    assert d.command == "STOP"  # streak restarted, only 1 consistent window so far
