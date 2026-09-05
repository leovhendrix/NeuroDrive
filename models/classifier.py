"""
models/classifier.py

Builds and cross-validates candidate classifiers (LDA / SVM / LogReg / RF)
on top of extracted features, and selects the best one based on validation
(not training) performance. No model is assumed superior a priori.
"""
import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix,
)

from config import ClassifierConfig

logger = logging.getLogger(__name__)


def _build_candidate(name: str, cfg: ClassifierConfig):
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "svm":
        return SVC(kernel="linear", probability=True, random_state=cfg.random_state)
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=cfg.random_state)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=cfg.random_state)
    raise ValueError(f"Unknown classifier candidate: {name}")


def evaluate_candidate(clf, X: np.ndarray, y: np.ndarray, cfg: ClassifierConfig) -> Dict:
    """Subject-specific k-fold cross-validation. Returns metrics computed
    ONLY on held-out folds — never on the training fold itself."""
    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    y_pred = cross_val_predict(clf, X, y, cv=skf)

    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y, y_pred)

    classes = np.unique(y)
    per_class_acc = {}
    for c in classes:
        mask = y == c
        per_class_acc[str(c)] = float(accuracy_score(y[mask], y_pred[mask]))

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_acc,
        "classes": classes.tolist(),
    }


def select_best_classifier(X: np.ndarray, y: np.ndarray, cfg: ClassifierConfig) -> Tuple[str, object, Dict]:
    """
    Cross-validates every candidate in cfg.candidates and returns
    (best_name, fitted_best_model, all_results) where fitted_best_model is
    refit on the FULL dataset only after model selection is complete.
    """
    all_results = {}
    for name in cfg.candidates:
        clf = _build_candidate(name, cfg)
        try:
            metrics = evaluate_candidate(clf, X, y, cfg)
            all_results[name] = metrics
            logger.info(
                "%s: acc=%.3f bal_acc=%.3f f1=%.3f",
                name, metrics["accuracy"], metrics["balanced_accuracy"], metrics["f1"],
            )
        except Exception as e:
            logger.warning("Candidate %s failed: %s", name, e)

    if not all_results:
        raise RuntimeError("No classifier candidate could be evaluated.")

    best_name = max(all_results, key=lambda n: all_results[n]["balanced_accuracy"])
    best_model = _build_candidate(best_name, cfg)
    best_model.fit(X, y)

    return best_name, best_model, all_results
