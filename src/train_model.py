"""
train_model.py — candidate models + optional ensemble, with checkpoint/resume.

Implements the §21 model search on top of the robust masking CV from
train_baseline. Tries LightGBM / XGBoost / CatBoost / HistGBM if installed,
each class-weighted, then a simple probability-averaging ensemble of whichever
are available. Saves fitted models to model_checkpoints/ (satisfies both the
Hammerspoon 'checkpoint' naming rule and the SLURM 24h resume requirement).

The 0.5 threshold is applied ONLY for reporting F1; it is never tuned.
Run with:  python train_model.py --model lgbm   (or xgb/cat/hgb/ensemble)
"""
from __future__ import annotations
import argparse, os, time, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
import random

import common as C
from config import RunConfig, CHAMPION_CONFIG
from experiment_logger import log_experiment
from runs import log_run

CKPT_DIR = "model_checkpoints"


def _score(y, proba):
    pred = (proba >= 0.5).astype(int)   # fixed 0.5 rule (§11)
    f1 = f1_score(y, pred); auc = roc_auc_score(y, proba)
    return f1, auc, 0.60 * f1 + 0.40 * auc


def make_model(kind: str, seed: int, pos_weight: float):
    kind = kind.lower()
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            random_state=seed, max_iter=500, learning_rate=0.04,
            l2_regularization=1.0, max_leaf_nodes=31)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            random_state=seed, n_estimators=700, learning_rate=0.03,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, verbosity=-1,
            n_jobs=1)  # n_jobs=1 for determinism
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(
            random_state=seed, n_estimators=700, learning_rate=0.03,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, eval_metric="logloss", tree_method="hist",
            nthread=1)  # nthread=1 for determinism
    if kind == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            random_state=seed, iterations=700, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, verbose=0,
            thread_count=1)  # thread_count=1 for determinism
    raise ValueError(kind)


def available(kind: str) -> bool:
    try:
        make_model(kind, C.SEED, 1.0); return True
    except Exception as e:
        print(f"[skip] {kind} unavailable: {e}"); return False


def robust_cv_model(cube, y, kind, cfg: RunConfig, seed=C.SEED, n_splits=5):
    from sklearn.model_selection import StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    group_id = np.arange(len(y)) // cfg.cv_block_size
    rng = np.random.default_rng(seed)
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    f1s, aucs, ws = [], [], []
    oof_preds = np.zeros(len(y), dtype=np.float64)
    for fold_idx, (tr_i, va_i) in enumerate(skf.split(cube, y, groups=group_id)):
        aug_cube, aug_y = C.augment_training_data(
            cube[tr_i], y[tr_i], n_copies=cfg.n_augmentation_copies, seed=seed + fold_idx)
        Xtr = C.build_features(aug_cube, cfg=cfg)
        Xva = C.build_features(C.simulate_test_masking(cube[va_i], rng), cfg=cfg)
        # AV weights removed — use_av_weights is always False in Phase 3+
        m = make_model(kind, seed, pos_w)
        m.fit(Xtr, aug_y)
        p = m.predict_proba(Xva)[:, 1]
        oof_preds[va_i] = p
        f1, auc, w = _score(y[va_i], p)
        f1s.append(f1); aucs.append(auc); ws.append(w)
        
    os.makedirs("oof_predictions", exist_ok=True)
    np.save(f"oof_predictions/{kind}_{cfg.hash()}_oof.npy", oof_preds)
    
    return np.mean(f1s), np.mean(aucs), np.mean(ws), np.std(ws)


def fit_full(cube, y, kind, cfg: RunConfig, seed=C.SEED):
    """Fit on ALL train rows (full months) and checkpoint the model."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    # Include config hash in checkpoint name so different configs don't share checkpoints
    path = os.path.join(CKPT_DIR, f"{kind}_{cfg.hash()}_checkpoint.pkl")
    if os.path.exists(path):
        print(f"[resume] loading existing {path}")
        return pickle.load(open(path, "rb"))
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    aug_cube, aug_y = C.augment_training_data(cube, y, n_copies=cfg.n_augmentation_copies, seed=seed)
    X = C.build_features(aug_cube, cfg=cfg)
    m = make_model(kind, seed, pos_w)
    m.fit(X, aug_y)  # No sample weights — AV weighting is disabled
    pickle.dump(m, open(path, "wb"))
    print(f"[checkpoint] saved {path}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hgb",
                    help="hgb|lgbm|xgb|cat|ensemble")
    ap.add_argument("--train", default="data_checkpoint/raw/Train.csv")
    ap.add_argument("--test",  default="data_checkpoint/raw/Test.csv")
    ap.add_argument("--log",   default="experiments_checkpoint/experiment_log.csv")
    ap.add_argument("--config", default=None,
                    help="Path to config JSON. Defaults to CHAMPION_CONFIG (Iteration 1).")
    a = ap.parse_args()

    # Load config — default is champion (Iteration 1)
    cfg = RunConfig.load(a.config) if a.config else CHAMPION_CONFIG
    print(f"[config] hash={cfg.hash()} use_absolute={cfg.use_absolute_anchors} "
          f"use_z_shape={cfg.use_z_shape_features} use_rank={cfg.use_rank_features} "
          f"n_aug={cfg.n_augmentation_copies} av_weights={cfg.use_av_weights}")

    # Global seed for reproducibility
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    tr, te = C.load_raw(a.train, a.test)
    cube = C.band_cube(tr); y = tr["label"].values

    # AV weights: disabled. Keep path check for diagnostic use only.
    if cfg.use_av_weights:
        raise RuntimeError(
            "use_av_weights=True is forbidden. "
            "Loss-level reweighting destroys calibration (Iteration 4 failure).")

    kinds = ["hgb", "lgbm", "xgb", "cat"] if a.model == "ensemble" else [a.model]
    kinds = [k for k in kinds if available(k)]
    if not kinds:
        print("No requested model available."); return

    t0 = time.time()
    for k in kinds:
        f1, auc, w, sd = robust_cv_model(cube, y, k, cfg=cfg)
        print(f"[{k}] robust CV  F1={f1:.4f} AUC={auc:.4f} score={w:.4f} (+/-{sd:.4f})")
        fit_full(cube, y, k, cfg=cfg)
        log_experiment(a.log, dict(
            experiment_id=f"model_{k}",
            model=k, features="valid-month temporal+missingness+spectral",
            preprocessing="native NaN; -9999=missing; dedup",
            hyperparameters="see make_model()", validation_strategy="5-fold SGKFold(block=25) robust masking",
            seed=C.SEED, F1=round(f1, 4), ROC_AUC=round(auc, 4),
            weighted_score=round(w, 4), notes=f"robust_std={sd:.4f}"))

    if a.model == "ensemble" and len(kinds) > 1:
        from sklearn.model_selection import StratifiedGroupKFold
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
        group_id = np.arange(len(y)) // cfg.cv_block_size
        rng = np.random.default_rng(C.SEED)
        pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
        f1s, aucs, ws = [], [], []

        oof_preds = np.zeros(len(y))
        oof_masks = np.zeros(len(y), dtype=bool)

        for fold_idx, (tr_i, va_i) in enumerate(skf.split(cube, y, groups=group_id)):
            aug_cube, aug_y = C.augment_training_data(
                cube[tr_i], y[tr_i], n_copies=cfg.n_augmentation_copies, seed=C.SEED + fold_idx)
            Xtr = C.build_features(aug_cube, cfg=cfg)
            Xva = C.build_features(C.simulate_test_masking(cube[va_i], rng), cfg=cfg)
            ps = []
            for k in kinds:
                m = make_model(k, C.SEED, pos_w)
                m.fit(Xtr, aug_y)  # No AV weights
                ps.append(m.predict_proba(Xva)[:, 1])
            p = np.mean(ps, axis=0)

            oof_preds[va_i] = p
            oof_masks[va_i] = True

            f1, auc, wsc = _score(y[va_i], p)
            f1s.append(f1); aucs.append(auc); ws.append(wsc)

        oof_f1   = np.mean(f1s)
        oof_auc  = np.mean(aucs)
        oof_score = np.mean(ws)
        print(f"[ensemble] robust CV F1={oof_f1:.4f} AUC={oof_auc:.4f} score={oof_score:.4f}")

        # Save OOF predictions
        oof_path = f"data_checkpoint/oof_{cfg.hash()}.csv"
        pd.DataFrame({
            'Target': y[oof_masks],
            'pred': oof_preds[oof_masks]
        }).to_csv(oof_path, index=False)
        # Also save as the canonical oof for backward compatibility
        pd.DataFrame({
            'Target': y[oof_masks],
            'pred': oof_preds[oof_masks]
        }).to_csv("data_checkpoint/oof_ensemble.csv", index=False)
        print(f"[ensemble] saved {oof_path}")

        log_experiment(a.log, dict(
            experiment_id=f"ensemble_{cfg.hash()}", model="+".join(kinds),
            features=f"absolute={cfg.use_absolute_anchors},z={cfg.use_z_shape_features},rank={cfg.use_rank_features}",
            preprocessing="native NaN; dedup", hyperparameters=f"aug={cfg.n_augmentation_copies}x",
            validation_strategy="5-fold SGKFold robust masking CV", seed=C.SEED,
            F1=round(oof_f1, 4), ROC_AUC=round(oof_auc, 4),
            weighted_score=round(oof_score, 4), notes=f"config_hash={cfg.hash()}"))

        log_run(
            config_hash=cfg.hash(),
            oof_f1=oof_f1, oof_auc=oof_auc,
            notes=f"model={a.model}",
        )

    print(f"[done] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
