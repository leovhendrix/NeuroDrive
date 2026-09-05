"""
features/csp.py

Common Spatial Patterns + log-variance features, using MNE's CSP
implementation (mne.decoding.CSP), which is the standard well-validated
implementation rather than a hand-rolled version.
"""
import numpy as np
from mne.decoding import CSP

from config import FeatureConfig


class CSPLogVarExtractor:
    """
    Wraps mne.decoding.CSP to fit the FeatureExtractor interface expected
    elsewhere in this codebase (fit/transform on (n_epochs, n_channels,
    n_samples) arrays).

    Note: standard CSP is a binary (2-class) spatial filter. For >2 classes
    we use MNE's built-in one-vs-rest extension (multiclass CSP), which
    trains one CSP per class and concatenates the resulting features.
    """

    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self._csp = None
        self._classes = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._classes = np.unique(y)
        n_components = min(self.cfg.n_csp_components, X.shape[1])

        if len(self._classes) == 2:
            self._csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
            self._csp.fit(X.astype(np.float64), y)
        else:
            # one CSP per class, one-vs-rest
            self._csp = {}
            for c in self._classes:
                y_bin = (y == c).astype(int)
                csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
                csp.fit(X.astype(np.float64), y_bin)
                self._csp[c] = csp
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        if isinstance(self._csp, dict):
            feats = [self._csp[c].transform(X) for c in self._classes]
            return np.concatenate(feats, axis=1)
        return self._csp.transform(X)
