import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SafetyConfig
from safety.safety_controller import SafetyController
from realtime.predictor import Prediction


class FakeRobot:
    def __init__(self, ack=True):
        self.ack = ack
        self.sent = []
        self.estopped = False

    def send_command(self, letter):
        self.sent.append(letter)
        return self.ack

    def ping(self):
        return self.ack

    def emergency_stop(self):
        self.estopped = True

    def close(self):
        pass


def make_prediction(command="FORWARD", confidence=0.9, quality_ok=True, signal_quality=0.9):
    return Prediction(
        command=command, confidence=confidence, timestamp=time.time(),
        signal_quality=signal_quality, quality_ok=quality_ok, raw_command=command,
        latency_ms={"total_ms": 10},
    )


def make_safety(robot):
    cfg = SafetyConfig(require_quality_ok=True, require_robot_ack=True,
                        max_command_age_s=5.0, watchdog_interval_s=100.0)
    return SafetyController(cfg, robot)


def test_good_prediction_is_forwarded():
    robot = FakeRobot(ack=True)
    safety = make_safety(robot)
    result = safety.process_prediction(make_prediction(command="LEFT"))
    assert result == "LEFT"
    assert "L" in robot.sent


def test_poor_quality_forces_stop():
    robot = FakeRobot(ack=True)
    safety = make_safety(robot)
    result = safety.process_prediction(make_prediction(quality_ok=False))
    assert result == "STOP"


def test_emergency_stop_overrides_everything():
    robot = FakeRobot(ack=True)
    safety = make_safety(robot)
    safety.trigger_emergency_stop()
    result = safety.process_prediction(make_prediction(command="FORWARD", confidence=0.99))
    assert result == "STOP"


def test_no_robot_ack_forces_stop():
    robot = FakeRobot(ack=False)
    safety = make_safety(robot)
    result = safety.process_prediction(make_prediction(command="RIGHT"))
    assert result == "STOP"


def test_invalid_command_forces_stop():
    robot = FakeRobot(ack=True)
    safety = make_safety(robot)
    result = safety.process_prediction(make_prediction(command="SPIN_360"))
    assert result == "STOP"


def test_disconnected_eeg_forces_stop():
    robot = FakeRobot(ack=True)
    safety = make_safety(robot)
    safety.notify_eeg_disconnected()
    result = safety.process_prediction(make_prediction(command="FORWARD"))
    assert result == "STOP"
