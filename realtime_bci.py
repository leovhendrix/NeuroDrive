"""
realtime_bci.py

Full real-time pipeline:

  EEG samples -> RealtimePredictor -> SafetyController -> RobotController -> Robot
                                            ^
                                            |
                                   dashboard reads state (does not gate control)

Run:
  python realtime_bci.py --model trained_models --simulation      # software test, no real robot
  python realtime_bci.py --model trained_models --robot-port COM5 # real robot, real EEG board
"""
import argparse
import logging
import queue
import threading
import time

from config import EEG, ROBOT, SAFETY
from eeg.acquisition import EEGAcquisition, ConnectionLostError
from models.model_manager import load_bundle
from realtime.predictor import RealtimePredictor
from robot.controller import RobotController, RobotConnectionError
from safety.safety_controller import SafetyController
from gui.dashboard import Dashboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def realtime_loop(predictor: RealtimePredictor, eeg: EEGAcquisition,
                   safety: SafetyController, update_queue: "queue.Queue", stop_event: threading.Event):
    frame_times = []
    while not stop_event.is_set():
        loop_start = time.time()

        if not eeg.is_connected():
            safety.notify_eeg_disconnected()
            update_queue.put({"eeg_connected": False})
            time.sleep(0.2)
            continue
        safety.notify_eeg_connected()

        try:
            prediction = predictor.step()
        except ConnectionLostError:
            safety.notify_eeg_disconnected()
            update_queue.put({"eeg_connected": False})
            time.sleep(0.2)
            continue

        if prediction is None:
            time.sleep(0.02)
            continue

        sent_command = safety.process_prediction(prediction)

        frame_times.append(loop_start)
        frame_times = [t for t in frame_times if loop_start - t < 1.0]
        fps = len(frame_times)

        update_queue.put({
            "eeg_connected": True,
            "signal_quality": prediction.signal_quality,
            "command": sent_command,
            "confidence": prediction.confidence,
            "robot_state": sent_command,
            "emergency_stop": safety.state.emergency_stop,
            "latency_ms": prediction.latency_ms,
            "fps": fps,
        })

        time.sleep(max(0.0, EEG.step_seconds - (time.time() - loop_start)))


def main():
    parser = argparse.ArgumentParser(description="Real-time BCI robot control")
    parser.add_argument("--model", default="trained_models")
    parser.add_argument("--simulation", action="store_true",
                         help="EEG via BrainFlow synthetic board (software testing only).")
    parser.add_argument("--robot-port", default=None,
                         help="Serial port for the robot. If omitted, robot commands are logged only (dry run).")
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    logger.info("Loaded model: %s (trained %s)", bundle.metadata.get("model_type"),
                bundle.metadata.get("training_date_utc"))

    eeg = EEGAcquisition(EEG, simulation=args.simulation)
    eeg.connect()
    eeg.start()

    if args.robot_port:
        ROBOT.serial_port = args.robot_port
        robot = RobotController(ROBOT)
        try:
            robot.connect()
        except RobotConnectionError as e:
            logger.error("Could not connect to robot: %s. Exiting.", e)
            eeg.release()
            return
    else:
        logger.warning("No --robot-port given: running in DRY RUN mode (no robot hardware).")
        robot = _DryRunRobot()

    safety = SafetyController(SAFETY, robot)
    safety.start_watchdog()

    predictor = RealtimePredictor(eeg, bundle, EEG)

    update_queue: "queue.Queue" = queue.Queue()
    stop_event = threading.Event()

    def emergency_stop():
        safety.trigger_emergency_stop()

    loop_thread = threading.Thread(
        target=realtime_loop, args=(predictor, eeg, safety, update_queue, stop_event), daemon=True
    )
    loop_thread.start()

    dashboard = Dashboard(update_queue, on_emergency_stop=emergency_stop)
    try:
        dashboard.run()
    finally:
        stop_event.set()
        safety.stop_watchdog()
        robot.close()
        eeg.release()


class _DryRunRobot:
    """Stand-in used when no --robot-port is supplied: logs commands instead
    of sending them, so the full pipeline can be exercised without hardware."""

    def send_command(self, letter: str) -> bool:
        logger.info("[DRY RUN] would send command: %s", letter)
        return True

    def ping(self) -> bool:
        return True

    def emergency_stop(self):
        logger.info("[DRY RUN] emergency stop")

    def close(self):
        pass


if __name__ == "__main__":
    main()
