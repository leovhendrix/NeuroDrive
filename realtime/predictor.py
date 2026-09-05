"""
realtime/predictor.py

EEG samples -> rolling buffer -> preprocessing -> epoch -> feature extraction
-> classifier -> probability -> confidence threshold -> temporal smoothing
-> (handed to safety controller, NOT directly to the robot).
"""
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from config import PREPROCESS, DECISION, QUALITY, EEGConfig
from eeg.acquisition import EEGAcquisition, ConnectionLostError
from eeg.preprocessing import Preprocessor, RealtimeFilterState
from eeg.quality import check_quality
from models.model_manager import ModelBundle
from realtime.smoothing import TemporalSmoother

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    command: str
    confidence: float
    timestamp: float
    signal_quality: float
    quality_ok: bool
    raw_command: str
    latency_ms: dict


class RealtimePredictor:
    def __init__(self, eeg: EEGAcquisition, bundle: ModelBundle, eeg_cfg: EEGConfig):
        self.eeg = eeg
        self.bundle = bundle
        self.eeg_cfg = eeg_cfg
        self.pre = Preprocessor(PREPROCESS, sampling_rate=eeg_cfg.sampling_rate)
        self.filter_state = RealtimeFilterState()
        self.smoother = TemporalSmoother(DECISION)

        self.class_labels = list(bundle.class_labels)
        self.class_to_command = bundle.metadata.get("class_to_command", {})

    def step(self) -> Optional[Prediction]:
        """Call repeatedly in the real-time loop. Returns None if not enough
        data is buffered yet for a full window."""
        t0 = time.time()

        try:
            raw_window = self.eeg.get_window(self.eeg_cfg.window_seconds)
        except ConnectionLostError:
            logger.error("EEG connection lost during real-time prediction.")
            return Prediction(
                command="STOP", confidence=0.0, timestamp=time.time(),
                signal_quality=0.0, quality_ok=False, raw_command="STOP",
                latency_ms={"total": 0},
            )

        if raw_window is None:
            return None
        t_acquired = time.time()

        processed = self.pre.process_realtime(raw_window, self.filter_state)
        t_processed = time.time()

        quality = check_quality(processed, self.eeg_cfg.sampling_rate, QUALITY)

        if not quality.ok:
            self.smoother.reset()
            latency = {
                "acquisition_ms": (t_acquired - t0) * 1000,
                "processing_ms": (t_processed - t_acquired) * 1000,
                "classification_ms": 0,
                "total_ms": (time.time() - t0) * 1000,
            }
            return Prediction(
                command=DECISION.idle_command, confidence=0.0, timestamp=time.time(),
                signal_quality=quality.score, quality_ok=False, raw_command="REJECTED",
                latency_ms=latency,
            )

        epoch = processed[np.newaxis, :, :]  # (1, n_channels, n_samples)
        features = self.bundle.feature_extractor.transform(epoch)

        proba = self.bundle.classifier.predict_proba(features)[0]
        best_idx = int(np.argmax(proba))
        predicted_class = self.class_labels[best_idx]
        confidence = float(proba[best_idx])
        t_classified = time.time()

        predicted_command = self.class_to_command.get(str(predicted_class), "STOP")

        decision = self.smoother.update(predicted_command, confidence)

        latency = {
            "acquisition_ms": (t_acquired - t0) * 1000,
            "processing_ms": (t_processed - t_acquired) * 1000,
            "classification_ms": (t_classified - t_processed) * 1000,
            "total_ms": (time.time() - t0) * 1000,
        }

        return Prediction(
            command=decision.command,
            confidence=confidence,
            timestamp=time.time(),
            signal_quality=quality.score,
            quality_ok=True,
            raw_command=decision.raw_command,
            latency_ms=latency,
        )
