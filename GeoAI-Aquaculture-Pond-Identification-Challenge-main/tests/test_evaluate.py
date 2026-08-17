"""Unit tests for src/evaluate.py"""

import numpy as np
import pytest

from src.evaluate import auc_from_proba, f1_from_proba, print_scores, zindi_score


def test_zindi_score_perfect():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.1, 0.2])
    score = zindi_score(y, (proba >= 0.5).astype(int), proba)
    assert abs(score - 1.0) < 1e-6


def test_zindi_score_random():
    y = np.array([1, 0, 1, 0, 1, 0])
    proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    score = zindi_score(y, (proba >= 0.5).astype(int), proba)
    assert 0.0 <= score <= 1.0


def test_zindi_score_weights():
    """Score = 0.6*F1 + 0.4*AUC."""
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.1, 0.2])
    f1 = f1_from_proba(y, proba)
    auc = auc_from_proba(y, proba)
    expected = 0.6 * f1 + 0.4 * auc
    score = zindi_score(y, (proba >= 0.5).astype(int), proba)
    assert abs(score - expected) < 1e-9


def test_f1_from_proba_threshold():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.6, 0.4, 0.3, 0.7])
    # threshold=0.5: predicted [1,0,0,1] → TP=1, FP=1, FN=1 → F1=0.5
    f1 = f1_from_proba(y, proba, threshold=0.5)
    assert abs(f1 - 0.5) < 1e-6


def test_f1_all_zeros():
    y = np.array([1, 1, 0])
    proba = np.array([0.1, 0.2, 0.3])
    f1 = f1_from_proba(y, proba)
    assert f1 == 0.0


def test_auc_perfect():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc_from_proba(y, proba) == 1.0


def test_auc_random():
    y = np.array([0, 1])
    proba = np.array([0.5, 0.5])
    assert abs(auc_from_proba(y, proba) - 0.5) < 1e-6


def test_print_scores_returns_dict(capsys):
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.8, 0.2, 0.7, 0.3])
    metrics = print_scores(y, proba)
    assert "F1" in metrics
    assert "ROC-AUC" in metrics
    assert "Zindi Score" in metrics
    captured = capsys.readouterr()
    assert "F1" in captured.out
