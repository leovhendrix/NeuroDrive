"""
Central configuration for the BCI robot-control system.

Nothing in this file is a "guarantee" of accuracy or performance — thresholds
here are starting points you should tune against your own validation results
(see training/evaluate.py output).
"""
from dataclasses import dataclass, field
from typing import List, Dict


# ---------------------------------------------------------------------------
# EEG hardware configuration
# ---------------------------------------------------------------------------
@dataclass
class EEGConfig:
    # BrainFlow board id. See https://brainflow.readthedocs.io/en/stable/SupportedBoards.html
    # Common examples:
    #   -1  -> SYNTHETIC_BOARD   (simulation / software testing ONLY)
    #   0   -> CYTON_BOARD (OpenBCI Cyton, serial)
    #   2   -> CYTON_DAISY_BOARD
    #   38  -> UNICORN_BOARD (g.tec Unicorn)
    #   57  -> MUSE_S_BOARD (via BLED112/BlueMuse bridge, board-id varies by OS/build)
    # Always confirm the exact integer for your hardware/BrainFlow version
    # with `brainflow.board_shim.BoardIds` before relying on it.
    board_id: int = -1  # default: synthetic board, for pipeline development only

    serial_port: str = ""       # e.g. "/dev/ttyUSB0" or "COM3", required for most real boards
    mac_address: str = ""       # required for some BLE boards
    other_params: Dict[str, str] = field(default_factory=dict)

    sampling_rate: int = 250    # overwritten at runtime from BoardShim.get_sampling_rate()
    channel_names: List[str] = field(default_factory=lambda: [])  # filled in at connect time

    # Real-time acquisition
    window_seconds: float = 2.0        # length of one processing epoch
    step_seconds: float = 0.25         # how often we produce a new window (sliding)
    ring_buffer_seconds: float = 30.0  # how much history we keep in memory


# ---------------------------------------------------------------------------
# Preprocessing configuration
# ---------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 40.0
    notch_freqs_hz: List[float] = field(default_factory=lambda: [50.0, 60.0])  # mains hum, both regions
    filter_order: int = 4

    # Real-time filtering must be causal (no zero-phase / filtfilt), since
    # filtfilt requires future samples that don't exist yet in a live stream.
    causal_in_realtime: bool = True

    # EEG frequency bands (Hz) used for band-power features
    bands: Dict[str, tuple] = field(default_factory=lambda: {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    })


# ---------------------------------------------------------------------------
# Signal quality thresholds
# ---------------------------------------------------------------------------
@dataclass
class QualityConfig:
    max_abs_amplitude_uv: float = 150.0     # above this -> likely artifact/saturation
    min_abs_amplitude_uv: float = 0.05      # near-zero variance -> disconnected electrode
    max_flatline_seconds: float = 1.0       # signal not changing -> disconnect
    min_variance_uv2: float = 0.5
    max_variance_uv2: float = 5000.0
    blink_band_hz: tuple = (1.0, 4.0)       # crude low-freq blink/EOG contamination band
    blink_power_ratio_threshold: float = 0.6
    quality_pass_threshold: float = 0.7     # overall quality score in [0,1] required to proceed


# ---------------------------------------------------------------------------
# BCI paradigm / class mapping
# ---------------------------------------------------------------------------
# This mapping is a DEFAULT SUGGESTION for a motor-imagery paradigm.
# It is not a claim that any of these mappings are neurologically fixed;
# the classifier only learns whatever mapping you train it on.
CLASS_TO_COMMAND: Dict[str, str] = {
    "rest": "STOP",
    "left_hand": "LEFT",
    "right_hand": "RIGHT",
    "feet": "FORWARD",
    "tongue": "BACKWARD",   # optional 5th class if you choose to train it
}

COMMAND_SET = ["REST", "FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    method: str = "csp_logvar"   # "csp_logvar" | "bandpower" | "csp+bandpower"
    n_csp_components: int = 6


# ---------------------------------------------------------------------------
# Classifier / training
# ---------------------------------------------------------------------------
@dataclass
class ClassifierConfig:
    candidates: List[str] = field(default_factory=lambda: ["lda", "svm", "logreg", "rf"])
    cv_folds: int = 5
    random_state: int = 42


# ---------------------------------------------------------------------------
# Real-time decision layer
# ---------------------------------------------------------------------------
@dataclass
class DecisionConfig:
    min_confidence: float = 0.80
    required_consistent_windows: int = 3
    prediction_history_len: int = 10
    idle_command: str = "STOP"


# ---------------------------------------------------------------------------
# Robot communication
# ---------------------------------------------------------------------------
@dataclass
class RobotConfig:
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    command_timeout_s: float = 0.5
    ack_timeout_s: float = 0.3
    heartbeat_interval_s: float = 0.5
    max_missed_heartbeats: int = 3
    command_map: Dict[str, str] = field(default_factory=lambda: {
        "FORWARD": "F",
        "BACKWARD": "B",
        "LEFT": "L",
        "RIGHT": "R",
        "STOP": "S",
        "REST": "S",
    })


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
@dataclass
class SafetyConfig:
    require_quality_ok: bool = True
    require_robot_ack: bool = True
    max_command_age_s: float = 1.0     # if a command is older than this, force STOP
    watchdog_interval_s: float = 0.2


# Single import-friendly bundle
EEG = EEGConfig()
PREPROCESS = PreprocessConfig()
QUALITY = QualityConfig()
FEATURES = FeatureConfig()
CLASSIFIER = ClassifierConfig()
DECISION = DecisionConfig()
ROBOT = RobotConfig()
SAFETY = SafetyConfig()
