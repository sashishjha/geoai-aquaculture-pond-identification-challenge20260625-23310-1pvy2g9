"""
train_baseline.py — §22 reference baseline.

Loads data -> engineers features on the FULL train cube -> validates under BOTH
(a) a conventional stratified split and (b) the robust masking-simulated split
-> applies the mandatory 0.5 threshold -> logs the run to experiment_log.csv.

This is deliberately simple (a single class-weighted model) so every later
experiment has a fixed reference point. Runs in seconds on CPU.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

import common as C
from experiment_logger import log_experiment


def score(y, proba):
    """Official Phase-One objective. Threshold is FIXED at 0.5 (§11)."""
    pred = (proba >= 0.5).astype(int)          # <-- never tuned, never per-fold
    f1 = f1_score(y, pred)
    auc = roc_auc_score(y, proba)
    return f1, auc, 0.60 * f1 + 0.40 * auc


def robust_cv(df, cube, y, n_splits=5, seed=C.SEED, note=""):
    """5-fold CV where each validation fold is masked to look like the TEST set.

    Training features are built on fully-observed months (as train really is);
    validation features are built on masking-simulated months (as test really
    is). This measures generalisation to arbitrary 4-6 month windows.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    oof = np.zeros(len(y))
    f1s, aucs, ws = [], [], []
    for tr_i, va_i in skf.split(cube, y):
        Xtr = C.build_features(cube[tr_i])
        va_masked = C.simulate_test_masking(cube[va_i], rng)
        Xva = C.build_features(va_masked)
        m = HistGradientBoostingClassifier(
            random_state=seed, max_iter=400, learning_rate=0.05,
            l2_regularization=1.0, class_weight="balanced")
        m.fit(Xtr, y[tr_i])
        p = m.predict_proba(Xva)[:, 1]
        oof[va_i] = p
        f1, auc, w = score(y[va_i], p)
        f1s.append(f1); aucs.append(auc); ws.append(w)
    return np.mean(f1s), np.mean(aucs), np.mean(ws), (np.std(ws))


def baseline_cv(cube, y, n_splits=5, seed=C.SEED):
    """Conventional CV — no masking. Reference only (optimistic vs. real test)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = C.build_features(cube)
    f1s, aucs, ws = [], [], []
    for tr_i, va_i in skf.split(X, y):
        m = HistGradientBoostingClassifier(
            random_state=seed, max_iter=400, learning_rate=0.05,
            l2_regularization=1.0, class_weight="balanced")
        m.fit(X.iloc[tr_i], y[tr_i])
        p = m.predict_proba(X.iloc[va_i])[:, 1]
        f1, auc, w = score(y[va_i], p)
        f1s.append(f1); aucs.append(auc); ws.append(w)
    return np.mean(f1s), np.mean(aucs), np.mean(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data_checkpoint/raw/Train.csv")
    ap.add_argument("--test",  default="data_checkpoint/raw/Test.csv")
    ap.add_argument("--log",   default="experiments_checkpoint/experiment_log.csv")
    a = ap.parse_args()

    np.random.seed(C.SEED)
    tr, te = C.load_raw(a.train, a.test)
    cube = C.band_cube(tr)
    y = tr["label"].values

    t0 = time.time()
    bf1, bauc, bw = baseline_cv(cube, y)
    rf1, rauc, rw, rstd = robust_cv(tr, cube, y)
    dt = time.time() - t0

    print(f"[baseline]  conventional CV  F1={bf1:.4f} AUC={bauc:.4f} score={bw:.4f}")
    print(f"[baseline]  ROBUST(masked)CV F1={rf1:.4f} AUC={rauc:.4f} score={rw:.4f} (+/-{rstd:.4f})")
    print(f"[baseline]  elapsed {dt:.1f}s")

    log_experiment(a.log, dict(
        experiment_id="baseline_histgbm",
        model="HistGradientBoostingClassifier",
        features="per-band temporal stats + missingness + spectral indices (valid-month agg)",
        preprocessing="native NaN (no imputation); -9999 treated as missing; dedup train",
        hyperparameters="max_iter=400,lr=0.05,l2=1.0,class_weight=balanced",
        validation_strategy="5-fold conventional + 5-fold robust(masking-simulated valid)",
        seed=C.SEED,
        F1=round(rf1, 4), ROC_AUC=round(rauc, 4), weighted_score=round(rw, 4),
        notes=f"REFERENCE. conv_score={bw:.4f}; robust_std={rstd:.4f}; {dt:.1f}s",
    ))


if __name__ == "__main__":
    main()
