"""
realtime/smoothing.py

Never act on a single window's prediction. Require several consecutive,
consistent, high-confidence predictions before emitting a movement command;
otherwise fall back to the idle command (STOP).

These thresholds are starting points, not guarantees of any particular
accuracy — tune them against your own validation results.
"""
from collections import deque
from dataclasses import dataclass

from config import DecisionConfig


@dataclass
class SmoothedDecision:
    command: str
    confidence: float
    consistent_count: int
    raw_command: str


class TemporalSmoother:
    def __init__(self, cfg: DecisionConfig):
        self.cfg = cfg
        self.history = deque(maxlen=cfg.prediction_history_len)

    def update(self, predicted_command: str, confidence: float) -> SmoothedDecision:
        if confidence < self.cfg.min_confidence:
            self.history.clear()
            return SmoothedDecision(self.cfg.idle_command, confidence, 0, predicted_command)

        self.history.append(predicted_command)

        consistent_count = 0
        for cmd in reversed(self.history):
            if cmd == predicted_command:
                consistent_count += 1
            else:
                break

        if consistent_count >= self.cfg.required_consistent_windows:
            return SmoothedDecision(predicted_command, confidence, consistent_count, predicted_command)

        return SmoothedDecision(self.cfg.idle_command, confidence, consistent_count, predicted_command)

    def reset(self):
        self.history.clear()
