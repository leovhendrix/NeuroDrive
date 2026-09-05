"""
training/collect_data.py

Runs a calibration session: cues the user through repeated trials of each
class (rest / left_hand / right_hand / feet / ...), records real EEG for
each trial, and saves the raw epochs + labels to disk for train_bci.py.

This is a command-line cueing script. Swap print()+sleep() for the GUI
dashboard's cue display if you want a nicer visual experience.
"""
import argparse
import os
import time
import logging

import numpy as np

from config import EEG, CLASS_TO_COMMAND
from eeg.acquisition import EEGAcquisition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_session(classes, trials_per_class: int, trial_seconds: float,
                 rest_seconds: float, out_path: str, simulation: bool):
    eeg = EEGAcquisition(EEG, simulation=simulation)
    eeg.connect()
    eeg.start()

    trial_order = classes * trials_per_class
    np.random.shuffle(trial_order)

    epochs = []
    labels = []

    try:
        for i, cls in enumerate(trial_order):
            print(f"\n[{i+1}/{len(trial_order)}] REST — relax...")
            _wait_collect(eeg, rest_seconds)  # discard, just a pause cue

            print(f"[{i+1}/{len(trial_order)}] >>> IMAGINE: {cls.upper()} <<<")
            window = _wait_collect(eeg, trial_seconds)
            if window is not None:
                epochs.append(window)
                labels.append(cls)
            else:
                logger.warning("Trial %d for class %s produced no data — skipped.", i, cls)

    finally:
        eeg.release()

    epochs = np.stack(epochs, axis=0)  # (n_epochs, n_channels, n_samples)
    labels = np.array(labels)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, epochs=epochs, labels=labels,
             sampling_rate=eeg.cfg.sampling_rate,
             channel_names=eeg.cfg.channel_names)
    print(f"\nSaved {len(labels)} trials to {out_path}")


def _wait_collect(eeg: EEGAcquisition, seconds: float):
    time.sleep(seconds)
    return eeg.get_window(seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCI calibration data collection")
    parser.add_argument("--classes", nargs="+", default=list(CLASS_TO_COMMAND.keys()))
    parser.add_argument("--trials-per-class", type=int, default=20)
    parser.add_argument("--trial-seconds", type=float, default=4.0)
    parser.add_argument("--rest-seconds", type=float, default=3.0)
    parser.add_argument("--out", default="data/calibration_session.npz")
    parser.add_argument("--simulation", action="store_true",
                         help="Use BrainFlow synthetic board for software testing only.")
    args = parser.parse_args()

    run_session(args.classes, args.trials_per_class, args.trial_seconds,
                args.rest_seconds, args.out, args.simulation)
