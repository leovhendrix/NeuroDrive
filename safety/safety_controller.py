"""
safety/safety_controller.py

MANDATORY safety layer. Architecture is:

    BCI classifier -> SafetyController -> RobotController -> Robot

NEVER: BCI classifier -> RobotController directly.

The robot is forced to STOP whenever any of the following is true:
  - EEG signal quality is poor
  - Classifier confidence is too low
  - No prediction has been received recently (stale/missing)
  - EEG is disconnected
  - Robot connection has failed
  - Robot communication has timed out (no ACK/heartbeat)
  - An invalid command was generated
  - Emergency stop has been triggered (always overrides everything)
"""
import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional

from config import SafetyConfig, ROBOT
from robot.controller import RobotController, RobotConnectionError
from realtime.predictor import Prediction

logger = logging.getLogger(__name__)

VALID_COMMANDS = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP", "REST"}


@dataclass
class SafetyState:
    emergency_stop: bool = False
    eeg_connected: bool = True
    robot_connected: bool = True
    last_prediction_time: float = 0.0
    last_command_sent: str = "STOP"
    last_block_reason: Optional[str] = None


class SafetyController:
    def __init__(self, cfg: SafetyConfig, robot: RobotController):
        self.cfg = cfg
        self.robot = robot
        self.state = SafetyState()
        # RLock: process_prediction holds the lock while calling helper
        # methods (_force_stop/_send_command) that also acquire it.
        self._lock = threading.RLock()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    def start_watchdog(self):
        """Background thread that forces STOP if predictions/heartbeats
        go stale, independent of whether the main loop is calling in."""
        self._running = True
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def stop_watchdog(self):
        self._running = False
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=1.0)

    def _watchdog_loop(self):
        while self._running:
            time.sleep(self.cfg.watchdog_interval_s)
            with self._lock:
                stale = (time.time() - self.state.last_prediction_time) > self.cfg.max_command_age_s
            if stale and self.state.last_command_sent != "STOP":
                logger.warning("Watchdog: predictions stale, forcing STOP.")
                self._force_stop("stale_prediction")

            try:
                acked = self.robot.ping()
                with self._lock:
                    self.state.robot_connected = acked
                if not acked:
                    self._force_stop("robot_heartbeat_failed")
            except RobotConnectionError:
                with self._lock:
                    self.state.robot_connected = False
                self._force_stop("robot_connection_lost")

    # ------------------------------------------------------------------
    def trigger_emergency_stop(self):
        logger.critical("EMERGENCY STOP TRIGGERED")
        with self._lock:
            self.state.emergency_stop = True
        self.robot.emergency_stop()
        self._set_last_command("STOP")

    def clear_emergency_stop(self):
        with self._lock:
            self.state.emergency_stop = False
        logger.info("Emergency stop cleared.")

    # ------------------------------------------------------------------
    def process_prediction(self, prediction: Prediction) -> str:
        """
        The single entry point the real-time loop must call. Returns the
        command actually sent to the robot (never the raw classifier output
        directly) and performs the send itself.
        """
        with self._lock:
            self.state.last_prediction_time = time.time()

            if self.state.emergency_stop:
                return self._force_stop("emergency_stop_active")

            if not prediction.quality_ok:
                return self._force_stop(f"poor_signal_quality(score={prediction.signal_quality:.2f})")

            if prediction.command not in VALID_COMMANDS:
                return self._force_stop(f"invalid_command({prediction.command})")

            if self.cfg.require_quality_ok and prediction.confidence < 0:
                # defensive; smoothing already enforces the real threshold
                return self._force_stop("negative_confidence")

            if not self.state.eeg_connected:
                return self._force_stop("eeg_disconnected")

            if not self.state.robot_connected:
                return self._force_stop("robot_disconnected")

        return self._send_command(prediction.command)

    def notify_eeg_disconnected(self):
        with self._lock:
            self.state.eeg_connected = False
        self._force_stop("eeg_disconnected")

    def notify_eeg_connected(self):
        with self._lock:
            self.state.eeg_connected = True

    # ------------------------------------------------------------------
    def _send_command(self, command: str) -> str:
        letter = ROBOT.command_map.get(command, "S")
        try:
            acked = self.robot.send_command(letter)
        except RobotConnectionError:
            return self._force_stop("robot_send_failed")

        if self.cfg.require_robot_ack and not acked:
            return self._force_stop("no_robot_ack")

        self._set_last_command(command)
        with self._lock:
            self.state.last_block_reason = None
        return command

    def _force_stop(self, reason: str) -> str:
        logger.info("Safety override -> STOP (reason: %s)", reason)
        try:
            self.robot.send_command(ROBOT.command_map.get("STOP", "S"))
        except RobotConnectionError:
            pass
        with self._lock:
            self.state.last_block_reason = reason
        self._set_last_command("STOP")
        return "STOP"

    def _set_last_command(self, command: str):
        with self._lock:
            self.state.last_command_sent = command
