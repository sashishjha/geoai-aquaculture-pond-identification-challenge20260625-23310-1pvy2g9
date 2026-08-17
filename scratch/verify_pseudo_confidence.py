import sys, os, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
import common as C
from config import RunConfig
from scipy.special import logit, expit

KINDS = ["hgb", "lgbm", "xgb", "cat"]
CKPT_DIR = "model_checkpoints"
OOF_DIR = "oof_predictions"

def to_percentile(arr):
    from scipy.stats import rankdata
    return (rankdata(arr) - 1) / (len(arr) - 1)

def main():
    cfg = RunConfig.load("config_s3.json")
    cfg_hash = cfg.hash()
    
    tr, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    y = tr["label"].values
    cube_te = C.band_cube(te)
    X_te = C.build_features(cube_te, cfg=cfg)
    
    # Load all base model test predictions
    test_preds = {}
    for k in KINDS:
        ckpt = f"{CKPT_DIR}/{k}_{cfg_hash}_checkpoint.pkl"
        with open(ckpt, "rb") as f:
            m = pickle.load(f)
        test_preds[k] = m.predict_proba(X_te)[:, 1]
    
    # Recreate calibrated ensemble score (same as S4)
    oof_preds = {k: np.load(f"{OOF_DIR}/{k}_{cfg_hash}_oof.npy") for k in KINDS}
    best_k = max(oof_preds, key=lambda k: np.mean(oof_preds[k]))  # approximate
    # Actually find best by score
    from train_model import _score
    best_k, best_score = None, -1
    for k, p in oof_preds.items():
        f1, auc, score = _score(y, p)
        if score > best_score:
            best_score, best_k = score, k
    best_oof = oof_preds[best_k]
    
    ranks = [to_percentile(p) for p in test_preds.values()]
    avg_rank_te = np.mean(ranks, axis=0)
    ensemble_te = np.quantile(best_oof, avg_rank_te)
    
    # Logit shift same as ensemble.py
    avg_rank_oof = np.mean([to_percentile(p) for p in oof_preds.values()], axis=0)
    blended_oof = np.quantile(best_oof, avg_rank_oof)
    t_star = np.quantile(blended_oof, 1 - cfg.pi_t_assumed)
    delta_f1 = -logit(t_star)
    calibrated_te = expit(logit(ensemble_te) + delta_f1)
    
    df = pd.DataFrame(test_preds)
    df['ensemble_calibrated'] = calibrated_te
    
    pos_mask = (df[KINDS] > 0.95).all(axis=1)
    neg_mask = (df[KINDS] < 0.05).all(axis=1)
    
    print(f"=== Confidence Verification ===")
    print(f"Total Test Rows: {len(df)}")
    print(f"Confident Positives (>0.95 across all 4 base models): {pos_mask.sum()}")
    print(f"Confident Negatives (<0.05 across all 4 base models): {neg_mask.sum()}")
    
    # Cross-check against calibrated ensemble
    pos_ens_agree = (df.loc[pos_mask, 'ensemble_calibrated'] > 0.5).sum()
    neg_ens_agree = (df.loc[neg_mask, 'ensemble_calibrated'] < 0.5).sum()
    print(f"\nEnsemble agreement on base-model positives: {pos_ens_agree}/{pos_mask.sum()} ({pos_ens_agree/pos_mask.sum()*100:.1f}%)")
    print(f"Ensemble agreement on base-model negatives: {neg_ens_agree}/{neg_mask.sum()} ({neg_ens_agree/neg_mask.sum()*100:.1f}%)")
    
    # Check calibrated ensemble also agrees at high threshold
    strong_pos = pos_mask & (df['ensemble_calibrated'] > 0.95)
    strong_neg = neg_mask & (df['ensemble_calibrated'] < 0.05)
    print(f"\nStrong double-gated (base >0.95 AND ensemble >0.95): {strong_pos.sum()}")
    print(f"Strong double-gated (base <0.05 AND ensemble <0.05): {strong_neg.sum()}")
    
    total_final = strong_pos.sum() + strong_neg.sum()
    print(f"Total double-gated rows: {total_final} ({total_final/len(df)*100:.1f}%)")
    if total_final > 0:
        pos_rate = strong_pos.sum() / total_final
        print(f"Prevalence in double-gated set: {pos_rate:.4f} (target pi_t=0.6175)")

if __name__ == "__main__":
    main()
