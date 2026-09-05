"""
models/model_manager.py

Saves/loads the trained classifier + preprocessing pipeline + metadata as a
single versioned bundle, so realtime_bci.py always knows exactly what it is
loading (sampling rate, channel order, feature config, training date, etc.).
"""
import json
import os
from datetime import datetime, timezone
from dataclasses import asdict
from typing import Any, Dict

import joblib


class ModelBundle:
    def __init__(self, classifier, feature_extractor, class_labels, metadata: Dict[str, Any]):
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.class_labels = class_labels
        self.metadata = metadata


def save_bundle(bundle: ModelBundle, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(bundle.classifier, os.path.join(out_dir, "bci_model.joblib"))
    joblib.dump(bundle.feature_extractor, os.path.join(out_dir, "preprocessing.joblib"))

    metadata = dict(bundle.metadata)
    metadata["class_labels"] = list(bundle.class_labels)
    metadata["saved_at_utc"] = datetime.now(timezone.utc).isoformat()

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def load_bundle(in_dir: str) -> ModelBundle:
    classifier = joblib.load(os.path.join(in_dir, "bci_model.joblib"))
    feature_extractor = joblib.load(os.path.join(in_dir, "preprocessing.joblib"))

    with open(os.path.join(in_dir, "metadata.json")) as f:
        metadata = json.load(f)

    class_labels = metadata.get("class_labels", [])
    return ModelBundle(classifier, feature_extractor, class_labels, metadata)


def build_metadata(sampling_rate: int, channel_names: list, feature_cfg,
                    classifier_name: str, validation_metrics: dict) -> Dict[str, Any]:
    return {
        "eeg_sampling_rate": sampling_rate,
        "channel_names": channel_names,
        "feature_config": {
            "method": feature_cfg.method,
            "n_csp_components": feature_cfg.n_csp_components,
        },
        "model_type": classifier_name,
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "validation_metrics": validation_metrics,
    }
