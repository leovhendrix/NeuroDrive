"""
robot/controller.py

Serial connection to an Arduino/ESP32-class microcontroller running the
companion firmware in firmware/bci_robot_receiver.ino.

This module ONLY exposes low-level, individually-safe primitives
(send_command, ping, emergency_stop). It does NOT decide whether a command
is safe to send — that decision belongs entirely to safety/safety_controller.py.
"""
import time
import logging
import threading
from typing import Optional

import serial

from config import RobotConfig
from robot.serial_protocol import encode_command, encode_ping, encode_estop, parse_line

logger = logging.getLogger(__name__)


class RobotConnectionError(RuntimeError):
    pass


class RobotController:
    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._last_ack_time = 0.0
        self._connected = False

    def connect(self):
        try:
            self._ser = serial.Serial(
                self.cfg.serial_port, self.cfg.baud_rate,
                timeout=self.cfg.ack_timeout_s,
            )
            time.sleep(2.0)  # allow Arduino auto-reset after serial open
            self._connected = True
            logger.info("Connected to robot on %s @ %d baud", self.cfg.serial_port, self.cfg.baud_rate)
        except serial.SerialException as e:
            self._connected = False
            raise RobotConnectionError(f"Could not open serial port: {e}") from e

    def is_connected(self) -> bool:
        return self._connected and self._ser is not None and self._ser.is_open

    def _send_and_wait_ack(self, payload: bytes, expect_prefix: str) -> bool:
        with self._lock:
            if not self.is_connected():
                raise RobotConnectionError("Robot not connected.")
            try:
                self._ser.reset_input_buffer()
                self._ser.write(payload)
            except serial.SerialException as e:
                self._connected = False
                raise RobotConnectionError(f"Write failed: {e}") from e

            deadline = time.time() + self.cfg.ack_timeout_s
            while time.time() < deadline:
                line = self._ser.readline().decode("ascii", errors="ignore")
                if not line:
                    continue
                msg_type, msg_payload = parse_line(line)
                if msg_type in ("ACK", "PONG"):
                    self._last_ack_time = time.time()
                    return True
                if msg_type == "ERR":
                    logger.warning("Robot reported error: %s", msg_payload)
                    return False
            return False  # timeout, no ack

    def send_command(self, letter: str) -> bool:
        """Returns True if the robot acknowledged the command."""
        payload = encode_command(letter)
        acked = self._send_and_wait_ack(payload, "ACK")
        if not acked:
            logger.warning("No ACK received for command %s within %.2fs", letter, self.cfg.ack_timeout_s)
        return acked

    def ping(self) -> bool:
        return self._send_and_wait_ack(encode_ping(), "PONG")

    def emergency_stop(self):
        """Fire-and-forget: send ESTOP as many times/fast as possible;
        do not block waiting for an ack since this must be immediate."""
        with self._lock:
            if not self.is_connected():
                return
            try:
                for _ in range(3):
                    self._ser.write(encode_estop())
            except serial.SerialException:
                self._connected = False

    def seconds_since_last_ack(self) -> float:
        if self._last_ack_time == 0.0:
            return float("inf")
        return time.time() - self._last_ack_time

    def close(self):
        if self._ser is not None:
            try:
                self.emergency_stop()
                self._ser.close()
            except Exception as e:
                logger.warning("Error closing robot serial port: %s", e)
        self._connected = False
