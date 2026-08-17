"""
scratch/ensemble_pseudo.py — Phase 8: Generate S5 ensemble from pseudo-trained models.

Uses the same rank-average + logit-shift logic as src/ensemble.py but loads
the pseudo-trained checkpoints (tagged with _pseudo suffix).
"""
import sys, os, pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit

sys.path.insert(0, 'src')
import common as C
from config import RunConfig
from train_model import _score
from submission_writer import write_submission, pre_submission_check

KINDS = ["hgb", "lgbm", "xgb", "cat"]
CKPT_DIR = "model_checkpoints"
OOF_DIR = "oof_predictions"

def to_percentile(arr):
    return (rankdata(arr) - 1) / (len(arr) - 1)

def main():
    cfg = RunConfig.load("config_s3.json")
    cfg_hash = cfg.hash()
    tag = f"{cfg_hash}_pseudo"
    
    tr, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    y = tr["label"].values
    
    # Load pseudo-trained OOF predictions (real rows only — no pseudo contamination)
    oof_preds = {}
    for k in KINDS:
        path = f"{OOF_DIR}/{k}_{tag}_oof.npy"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        oof_preds[k] = np.load(path)
        f1, auc, score = _score(y, oof_preds[k])
        print(f"[oof] {k}: score={score:.4f} (f1={f1:.4f}, auc={auc:.4f})")
    
    # Best single model
    best_k = max(oof_preds, key=lambda k: _score(y, oof_preds[k])[2])
    best_oof = oof_preds[best_k]
    print(f"[ensemble] best single model: {best_k}")
    
    # Rank-average OOF → blended OOF
    oof_ranks = [to_percentile(p) for p in oof_preds.values()]
    avg_rank_oof = np.mean(oof_ranks, axis=0)
    blended_oof = np.quantile(best_oof, avg_rank_oof)
    f1, auc, score = _score(y, blended_oof)
    print(f"[ensemble] Blended OOF: score={score:.4f} (f1={f1:.4f}, auc={auc:.4f})")
    
    # Measure t* on REAL blended OOF → derive logit shift
    t_star = np.quantile(blended_oof, 1 - cfg.pi_t_assumed)
    delta   = -logit(t_star)
    print(f"[calibration] t*={t_star:.4f}, logit shift={delta:.4f} (pi_t={cfg.pi_t_assumed})")
    
    # Load pseudo-trained checkpoints and predict on test
    cube_te = C.band_cube(te)
    X_te = C.build_features(cube_te, cfg=cfg)
    
    test_preds = {}
    for k in KINDS:
        ckpt = f"{CKPT_DIR}/{k}_{tag}_checkpoint.pkl"
        with open(ckpt, "rb") as f:
            m = pickle.load(f)
        test_preds[k] = m.predict_proba(X_te)[:, 1]
    
    # Rank-average test predictions
    te_ranks = [to_percentile(p) for p in test_preds.values()]
    avg_rank_te = np.mean(te_ranks, axis=0)
    blended_te = np.quantile(best_oof, avg_rank_te)
    
    # Apply logit shift
    calibrated_te = expit(logit(blended_te) + delta)
    pos_rate = (calibrated_te > 0.5).mean()
    print(f"[calibration] Test positive rate after shift: {pos_rate:.4f} (target ~{cfg.pi_t_assumed:.4f})")
    
    # Write submission
    out_path = "submissions/S5_pseudo_ensemble.csv"
    sub = pd.DataFrame({
        'ID': te['ID'].values,
        'TargetRAUC': calibrated_te,
        'TargetF1': (calibrated_te > 0.5).astype(int)
    })
    pos_rate_val = float((calibrated_te > 0.5).mean())
    pre_submission_check(pos_rate_val, cfg.pi_t_assumed)
    sub.to_csv(out_path, index=False, float_format="%.17f")
    print(f"[done] Submission written: {out_path}")
    
    # Final comparison
    print(f"\n=== Final Comparison ===")
    print(f"S4 (S3 + rank ensemble):          LB=0.8264")
    print(f"S5 (S4 + pseudo-labeled 2026 data): OOF={score:.4f} — submit to verify")

if __name__ == "__main__":
    main()
