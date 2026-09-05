"""
eeg/artifacts.py

Epoch-level artifact rejection for training data. Used when building the
calibration dataset so contaminated trials don't poison the classifier.
"""
import numpy as np


def reject_bad_epochs(epochs: np.ndarray, labels: np.ndarray, amplitude_threshold_uv: float = 150.0):
    """
    epochs: (n_epochs, n_channels, n_samples)
    labels: (n_epochs,)

    Returns (clean_epochs, clean_labels, rejected_indices)
    """
    n_epochs = epochs.shape[0]
    keep_mask = np.ones(n_epochs, dtype=bool)
    rejected = []

    for i in range(n_epochs):
        epoch = epochs[i]
        max_abs = np.max(np.abs(epoch))
        has_nan = not np.all(np.isfinite(epoch))
        flat = np.any(np.var(epoch, axis=-1) < 1e-6)

        if max_abs > amplitude_threshold_uv or has_nan or flat:
            keep_mask[i] = False
            rejected.append(i)

    return epochs[keep_mask], labels[keep_mask], rejected
