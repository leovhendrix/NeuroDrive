"""
eeg/acquisition.py

Real EEG acquisition using BrainFlow. Hardware configuration is isolated here
so nothing else in the codebase needs to know which board is connected.

Simulation mode is provided ONLY for software development/testing and is
implemented via BrainFlow's own SYNTHETIC_BOARD (board_id = -1). That is a
real BrainFlow board type documented for this exact purpose — it is not a
hand-rolled random-number generator pretending to be EEG, and it is clearly
gated behind an explicit `simulation=True` flag that must never be set when
real hardware is intended.
"""
import time
import logging
import threading
from collections import deque
from typing import Optional, List

import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

from config import EEGConfig

logger = logging.getLogger(__name__)


class ConnectionLostError(RuntimeError):
    pass


class EEGAcquisition:
    """
    Wraps a BrainFlow board session and exposes a simple, hardware-agnostic
    interface: connect() -> start() -> get_window() -> stop().

    Usage (real hardware):
        cfg = EEGConfig(board_id=0, serial_port="/dev/ttyUSB0")
        eeg = EEGAcquisition(cfg, simulation=False)

    Usage (software testing only):
        cfg = EEGConfig(board_id=-1)
        eeg = EEGAcquisition(cfg, simulation=True)
    """

    def __init__(self, cfg: EEGConfig, simulation: bool = False):
        self.cfg = cfg
        self.simulation = simulation

        if simulation and cfg.board_id != BoardIds.SYNTHETIC_BOARD.value:
            logger.warning(
                "simulation=True but board_id != SYNTHETIC_BOARD (-1). "
                "Forcing board_id to SYNTHETIC_BOARD for safety."
            )
            cfg.board_id = BoardIds.SYNTHETIC_BOARD.value

        if not simulation and cfg.board_id == BoardIds.SYNTHETIC_BOARD.value:
            raise ValueError(
                "board_id is SYNTHETIC_BOARD but simulation=False. "
                "Refusing to start: this would silently run on fake data "
                "while the rest of the system believes it is on real hardware."
            )

        self._params = BrainFlowInputParams()
        if cfg.serial_port:
            self._params.serial_port = cfg.serial_port
        if cfg.mac_address:
            self._params.mac_address = cfg.mac_address
        for k, v in cfg.other_params.items():
            setattr(self._params, k, v)

        self._board: Optional[BoardShim] = None
        self._eeg_channels: List[int] = []
        self._connected = False
        self._streaming = False
        self._last_sample_time = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def connect(self):
        BoardShim.enable_dev_board_logger() if False else None  # keep default logging quiet
        self._board = BoardShim(self.cfg.board_id, self._params)
        self._board.prepare_session()

        self.cfg.sampling_rate = BoardShim.get_sampling_rate(self.cfg.board_id)
        self._eeg_channels = BoardShim.get_eeg_channels(self.cfg.board_id)
        try:
            names = BoardShim.get_eeg_names(self.cfg.board_id)
            self.cfg.channel_names = list(names)
        except Exception:
            self.cfg.channel_names = [f"ch{i}" for i in range(len(self._eeg_channels))]

        self._connected = True
        logger.info(
            "Connected to board_id=%s, fs=%d Hz, channels=%s (simulation=%s)",
            self.cfg.board_id, self.cfg.sampling_rate, self.cfg.channel_names,
            self.simulation,
        )

    def start(self):
        if not self._connected:
            raise RuntimeError("Call connect() before start().")
        self._board.start_stream()
        self._streaming = True
        self._last_sample_time = time.time()
        logger.info("EEG streaming started.")

    def is_connected(self) -> bool:
        """Best-effort connection-loss detection.

        BrainFlow does not always raise immediately on a physical disconnect,
        so we additionally track whether fresh samples keep arriving.
        """
        if not self._connected or self._board is None:
            return False
        try:
            _ = self._board.get_board_data_count()
            return True
        except Exception:
            self._connected = False
            return False

    def get_window(self, seconds: float) -> Optional[np.ndarray]:
        """
        Returns the most recent `seconds` of EEG data as an array of shape
        (n_channels, n_samples), or None if not enough data is buffered yet.
        Raises ConnectionLostError if the board appears disconnected.
        """
        if not self._streaming:
            raise RuntimeError("Call start() before get_window().")
        if not self.is_connected():
            raise ConnectionLostError("EEG board connection lost.")

        n_samples_needed = int(seconds * self.cfg.sampling_rate)
        available = self._board.get_board_data_count()
        if available < n_samples_needed:
            return None

        data = self._board.get_current_board_data(n_samples_needed)
        window = data[self._eeg_channels, :]

        if self._contains_invalid_samples(window):
            logger.warning("Invalid samples (NaN/Inf) detected in EEG window.")

        self._last_sample_time = time.time()
        return window

    @staticmethod
    def _contains_invalid_samples(window: np.ndarray) -> bool:
        return not np.all(np.isfinite(window))

    def seconds_since_last_sample(self) -> float:
        return time.time() - self._last_sample_time

    def stop(self):
        if self._board is not None and self._streaming:
            try:
                self._board.stop_stream()
            except Exception as e:
                logger.warning("Error stopping stream: %s", e)
        self._streaming = False

    def release(self):
        self.stop()
        if self._board is not None:
            try:
                self._board.release_session()
            except Exception as e:
                logger.warning("Error releasing session: %s", e)
        self._connected = False
        logger.info("EEG session released.")

    def __enter__(self):
        self.connect()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
