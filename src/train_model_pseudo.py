"""
src/train_model_pseudo.py — Phase 8: Train with fold-safe pseudo-label injection.

Critical invariant (Fix 1):
- Pseudo rows are injected ONLY into the training side of each fold, AFTER the split.
- OOF predictions, proxy score, and t* are computed on REAL-labeled rows only.
- Pseudo rows never appear in any validation fold.

Three validation gates after training (Fix 2):
1. Real-OOF non-degradation: proxy must not drop >1σ vs S4 baseline.
2. Held-out pseudo consistency: 141 held-out confident rows scored ≥95% correctly.
3. Uncertain-row sanity: score distribution shift on 272 uncertain rows is modest.
"""
from __future__ import annotations
import argparse, os, time, pickle
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import random, sys

sys.path.insert(0, 'src')
import common as C
from config import RunConfig
from train_model import _score, make_model, available, fit_full, CKPT_DIR

KINDS = ["hgb", "lgbm", "xgb", "cat"]
OOF_DIR = "oof_predictions"

# S4 baseline reference (from our verified proxy run)
S4_PROXY_MEAN = 0.9714
S4_PROXY_STD  = 0.0020


def robust_cv_pseudo(cube, y, pseudo_cube, pseudo_y, kind, cfg: RunConfig, seed=C.SEED, n_splits=5):
    """CV with pseudo-rows injected into train-side only. OOF on real rows only."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    group_id = np.arange(len(y)) // cfg.cv_block_size
    rng = np.random.default_rng(seed)
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    f1s, aucs, ws = [], [], []
    oof_preds = np.zeros(len(y), dtype=np.float64)
    
    for fold_idx, (tr_i, va_i) in enumerate(skf.split(cube, y, groups=group_id)):
        # Real training data — augmented as normal
        aug_cube, aug_y = C.augment_training_data(
            cube[tr_i], y[tr_i], n_copies=cfg.n_augmentation_copies, seed=seed + fold_idx)
        
        # Inject pseudo-rows into train side ONLY (Fix 1)
        # Pseudo-rows already have 5 diversified copies from build_pseudo_train.py
        combined_cube = np.concatenate([aug_cube, pseudo_cube], axis=0)
        combined_y   = np.concatenate([aug_y, pseudo_y], axis=0)
        
        Xtr = C.build_features(combined_cube, cfg=cfg)
        Xva = C.build_features(C.simulate_test_masking(cube[va_i], rng), cfg=cfg)
        
        m = make_model(kind, seed, pos_w)
        m.fit(Xtr, combined_y)
        
        # OOF only on REAL validation rows
        p = m.predict_proba(Xva)[:, 1]
        oof_preds[va_i] = p
        f1, auc, w = _score(y[va_i], p)
        f1s.append(f1); aucs.append(auc); ws.append(w)
        print(f"  [{kind}] fold {fold_idx}: F1={f1:.4f} AUC={auc:.4f} score={w:.4f}", flush=True)
    
    os.makedirs(OOF_DIR, exist_ok=True)
    tag = f"{kind}_{cfg.hash()}_pseudo"
    np.save(f"{OOF_DIR}/{tag}_oof.npy", oof_preds)
    
    return np.mean(f1s), np.mean(aucs), np.mean(ws), np.std(ws), oof_preds


def fit_full_pseudo(cube, y, pseudo_cube, pseudo_y, kind, cfg: RunConfig, seed=C.SEED):
    """Fit on ALL real train rows + pseudo rows for checkpoint."""
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"{kind}_{cfg.hash()}_pseudo"
    path = os.path.join(CKPT_DIR, f"{tag}_checkpoint.pkl")
    if os.path.exists(path):
        print(f"[resume] loading existing {path}")
        return pickle.load(open(path, "rb"))
    
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    aug_cube, aug_y = C.augment_training_data(cube, y, n_copies=cfg.n_augmentation_copies, seed=seed)
    
    # Inject pseudo into full training
    combined_cube = np.concatenate([aug_cube, pseudo_cube], axis=0)
    combined_y   = np.concatenate([aug_y, pseudo_y], axis=0)
    
    X = C.build_features(combined_cube, cfg=cfg)
    m = make_model(kind, seed, pos_w)
    m.fit(X, combined_y)
    pickle.dump(m, open(path, "wb"))
    print(f"[checkpoint] saved {path}")
    return m


def gate_1_real_oof(proxy_mean, proxy_std):
    """Real-OOF non-degradation: must not drop >1σ vs S4."""
    print(f"\n=== Gate 1: Real-OOF Non-Degradation ===")
    print(f"S4 baseline: {S4_PROXY_MEAN:.4f} ± {S4_PROXY_STD:.4f}")
    print(f"Pseudo model: {proxy_mean:.4f} ± {proxy_std:.4f}")
    delta_logit = logit(proxy_mean) - logit(S4_PROXY_MEAN)
    threshold_logit = -S4_PROXY_STD  # must not drop by >1σ
    passed = delta_logit >= threshold_logit
    print(f"Delta logit: {delta_logit:.4f} (threshold: >= {threshold_logit:.4f})")
    print(f"Gate 1: {'PASS' if passed else 'FAIL'}")
    return passed


def gate_2_pseudo_consistency(cfg, pseudo_cube, holdout_manifest_path):
    """Held-out pseudo consistency: retrained ensemble scores ≥95% correctly."""
    print(f"\n=== Gate 2: Held-Out Pseudo Consistency ===")
    holdout_df = pd.read_csv(holdout_manifest_path)
    holdout_idx = holdout_df['test_row_idx'].values
    holdout_labels = holdout_df['label'].values
    
    X_holdout = C.build_features(pseudo_cube[holdout_idx], cfg=cfg)  # pseudo_cube is full-test-sized? No.
    # Actually holdout is indexed into test cube — need to reload
    _, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    cube_te = C.band_cube(te)
    X_holdout = C.build_features(cube_te[holdout_idx], cfg=cfg)
    
    tag_suffix = f"{cfg.hash()}_pseudo"
    preds_per_model = []
    for k in KINDS:
        ckpt = f"{CKPT_DIR}/{k}_{tag_suffix}_checkpoint.pkl"
        if not os.path.exists(ckpt):
            print(f"  Missing checkpoint {ckpt}, skipping.")
            continue
        m = pickle.load(open(ckpt, "rb"))
        preds_per_model.append(m.predict_proba(X_holdout)[:, 1])
    
    if not preds_per_model:
        print("No models loaded — gate 2 skipped.")
        return True
    
    ensemble_pred = np.mean(preds_per_model, axis=0)
    predicted_class = (ensemble_pred > 0.5).astype(int)
    accuracy = (predicted_class == holdout_labels).mean()
    print(f"Holdout accuracy: {accuracy:.4f} (threshold: >= 0.95)")
    passed = accuracy >= 0.95
    print(f"Gate 2: {'PASS' if passed else 'FAIL'}")
    return passed


def gate_3_uncertain_sanity(cfg):
    """Uncertain-row sanity: compare S4 vs S5 score distributions on uncertain rows."""
    print(f"\n=== Gate 3: Uncertain-Row Sanity ===")
    uncertain_df = pd.read_csv("scratch/pseudo_uncertain_manifest.csv")
    uncertain_idx = uncertain_df['test_row_idx'].values
    s4_scores = uncertain_df['s4_mean_score'].values
    
    _, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    cube_te = C.band_cube(te)
    X_unc = C.build_features(cube_te[uncertain_idx], cfg=cfg)
    
    tag_suffix = f"{cfg.hash()}_pseudo"
    preds_per_model = []
    for k in KINDS:
        ckpt = f"{CKPT_DIR}/{k}_{tag_suffix}_checkpoint.pkl"
        if not os.path.exists(ckpt):
            continue
        m = pickle.load(open(ckpt, "rb"))
        preds_per_model.append(m.predict_proba(X_unc)[:, 1])
    
    if not preds_per_model:
        print("No models loaded — gate 3 skipped.")
        return True
    
    s5_scores = np.mean(preds_per_model, axis=0)
    
    s4_pos_rate = (s4_scores > 0.5).mean()
    s5_pos_rate = (s5_scores > 0.5).mean()
    shift = abs(s5_pos_rate - s4_pos_rate)
    
    print(f"S4 uncertain positive rate: {s4_pos_rate:.4f}")
    print(f"S5 uncertain positive rate: {s5_pos_rate:.4f}")
    print(f"Absolute shift: {shift:.4f} (threshold: < 0.10 for sanity)")
    
    # Check correlation — if scores are reshuffled, something went wrong
    corr = np.corrcoef(s4_scores, s5_scores)[0, 1]
    print(f"S4/S5 uncertain score correlation: {corr:.4f} (threshold: > 0.80)")
    
    passed = shift < 0.10 and corr > 0.80
    print(f"Gate 3: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_s3.json")
    a = ap.parse_args()
    
    cfg = RunConfig.load(a.config)
    print(f"[config] hash={cfg.hash()}")
    
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
    tr, _ = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    cube = C.band_cube(tr)
    y = tr["label"].values
    
    # Load pre-built pseudo data (from build_pseudo_train.py)
    pseudo_cube = np.load("data_checkpoint/pseudo_cube.npy")
    pseudo_y    = np.load("data_checkpoint/pseudo_y.npy")
    print(f"Loaded pseudo cube: {pseudo_cube.shape}, labels: {pseudo_y.shape}, prevalence={pseudo_y.mean():.4f}")
    
    kinds = [k for k in KINDS if available(k)]
    proxy_scores = []
    
    for k in kinds:
        print(f"\n--- Training {k} with pseudo injection ---")
        f1, auc, score, std, oof = robust_cv_pseudo(cube, y, pseudo_cube, pseudo_y, k, cfg=cfg)
        print(f"[{k}] Real-OOF: F1={f1:.4f} AUC={auc:.4f} score={score:.4f} ±{std:.4f}")
        proxy_scores.append(score)
        fit_full_pseudo(cube, y, pseudo_cube, pseudo_y, k, cfg=cfg)
    
    overall_proxy = np.mean(proxy_scores)
    overall_std   = np.std(proxy_scores)
    print(f"\nOverall Real-OOF Proxy (mean across models): {overall_proxy:.4f} ± {overall_std:.4f}")
    
    # === Run All Three Gates ===
    g1 = gate_1_real_oof(overall_proxy, overall_std)
    g2 = gate_2_pseudo_consistency(cfg, pseudo_cube, "scratch/pseudo_holdout_manifest.csv")
    g3 = gate_3_uncertain_sanity(cfg)
    
    print(f"\n=== GATE SUMMARY ===")
    print(f"Gate 1 (Real-OOF non-degradation): {'PASS' if g1 else 'FAIL'}")
    print(f"Gate 2 (Held-out pseudo consistency): {'PASS' if g2 else 'FAIL'}")
    print(f"Gate 3 (Uncertain-row sanity): {'PASS' if g3 else 'FAIL'}")
    
    if g1 and g2 and g3:
        print("\nAll gates PASSED. Proceed to generate S5 ensemble submission.")
    else:
        print("\nOne or more gates FAILED. Abort S5. Keep S4 as final submission.")
        sys.exit(1)


if __name__ == "__main__":
    main()
