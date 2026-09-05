"""
training/train_bci.py

Loads a calibration session recorded by collect_data.py, preprocesses it,
extracts features, cross-validates several classifiers, selects the best
one by validation performance, and saves the full model bundle.
"""
import argparse
import logging

import numpy as np

from config import PREPROCESS, FEATURES, CLASSIFIER, CLASS_TO_COMMAND
from eeg.preprocessing import Preprocessor
from eeg.artifacts import reject_bad_epochs
from features.csp import CSPLogVarExtractor
from features.bandpower import BandPowerExtractor
from models.classifier import select_best_classifier
from models.model_manager import ModelBundle, save_bundle, build_metadata
from training.evaluate import print_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_feature_extractor():
    if FEATURES.method == "csp_logvar":
        return CSPLogVarExtractor(FEATURES)
    if FEATURES.method == "bandpower":
        return BandPowerExtractor(PREPROCESS, sampling_rate=None)  # fs set below
    raise ValueError(f"Unsupported feature method for this script: {FEATURES.method}")


def main(session_path: str, out_dir: str):
    data = np.load(session_path, allow_pickle=True)
    epochs = data["epochs"]          # (n_epochs, n_channels, n_samples)
    labels = data["labels"]
    fs = int(data["sampling_rate"])
    channel_names = list(data["channel_names"])

    logger.info("Loaded %d epochs, fs=%d Hz, %d channels", epochs.shape[0], fs, epochs.shape[1])

    epochs, labels, rejected = reject_bad_epochs(epochs, labels)
    logger.info("Rejected %d artifact-contaminated epochs, %d remain", len(rejected), epochs.shape[0])

    pre = Preprocessor(PREPROCESS, sampling_rate=fs)
    clean_epochs = np.stack([pre.process_offline(e) for e in epochs], axis=0)

    if FEATURES.method == "bandpower":
        extractor = BandPowerExtractor(PREPROCESS, sampling_rate=fs)
    else:
        extractor = build_feature_extractor()

    extractor.fit(clean_epochs, labels)
    X = extractor.transform(clean_epochs)
    y = labels

    best_name, best_model, all_results = select_best_classifier(X, y, CLASSIFIER)
    print_report(best_name, all_results)

    metadata = build_metadata(fs, channel_names, FEATURES, best_name, all_results[best_name])
    metadata["class_to_command"] = CLASS_TO_COMMAND
    metadata["n_trials_used"] = int(epochs.shape[0])
    metadata["n_trials_rejected"] = int(len(rejected))

    bundle = ModelBundle(best_model, extractor, np.unique(y), metadata)
    save_bundle(bundle, out_dir)
    logger.info("Saved model bundle to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the BCI classifier from a calibration session")
    parser.add_argument("--session", default="data/calibration_session.npz")
    parser.add_argument("--out", default="trained_models")
    args = parser.parse_args()
    main(args.session, args.out)
