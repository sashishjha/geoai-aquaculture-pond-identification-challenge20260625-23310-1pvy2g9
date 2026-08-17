"""Evaluation metrics matching the Zindi scoring formula.

Final Score = 0.60 * F1 + 0.40 * ROC-AUC
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def zindi_score(y_true: np.ndarray, y_pred_binary: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Compute the Zindi combined leaderboard score."""
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_proba)
    return 0.60 * f1 + 0.40 * auc


def f1_from_proba(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> float:
    return f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)


def auc_from_proba(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return roc_auc_score(y_true, y_proba)


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, n_steps: int = 200) -> tuple[float, float]:
    """Find threshold that maximises F1 on OOF predictions.

    Returns:
        (best_threshold, best_f1)
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx]), float(f1s[best_idx])


def print_scores(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_bin = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_bin, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    score = 0.60 * f1 + 0.40 * auc
    metrics = {"F1": f1, "ROC-AUC": auc, "Zindi Score": score}
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics
