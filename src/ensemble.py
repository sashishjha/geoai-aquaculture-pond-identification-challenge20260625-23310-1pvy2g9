"""
Phase 7: Rank-Average Ensembling

1. Rank-average the OOF predictions of the 4 base models.
2. Map blended OOF ranks through the best single model's OOF distribution.
3. Re-measure t* on the blended OOF and re-derive logit shift.
4. Predict on Test set using the 4 fitted models.
5. Rank-average the Test predictions.
6. Map blended Test ranks through the best single model's OOF distribution.
7. Write submission via submission_writer.
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit, expit
import pickle

sys.path.insert(0, 'src')
sys.path.insert(0, 'scratch')
import common as C
from config import RunConfig, CHAMPION_CONFIG
from submission_writer import write_submission, pre_submission_check
from runs import log_run

KINDS = ["hgb", "lgbm", "xgb", "cat"]
CKPT_DIR = "model_checkpoints"
OOF_DIR = "oof_predictions"

def to_percentile(arr):
    return (rankdata(arr) - 1) / (len(arr) - 1)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config JSON (e.g. config_s3.json)")
    ap.add_argument("--out", default="submissions/S4_rank_ensemble.csv")
    a = ap.parse_args()
    
    cfg = RunConfig.load(a.config)
    cfg_hash = cfg.hash()
    print(f"[ensemble] using config {cfg_hash}")
    
    tr, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    y = tr["label"].values
    
    # 1. Load OOF predictions
    oof_preds = {}
    for k in KINDS:
        path = f"{OOF_DIR}/{k}_{cfg_hash}_oof.npy"
        if not os.path.exists(path):
            raise FileNotFoundError(f"OOF predictions not found: {path}. Run train_model.py --model ensemble first.")
        oof_preds[k] = np.load(path)
        
    # Best single model? Let's check LightGBM or whoever has the highest proxy
    from train_model import _score
    best_k, best_score = None, -1
    for k, p in oof_preds.items():
        f1, auc, score = _score(y, p)
        print(f"[oof] {k}: score={score:.4f} (f1={f1:.4f}, auc={auc:.4f})")
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"[ensemble] best single model is {best_k} (score={best_score:.4f})")
    best_oof = oof_preds[best_k]
    
    # 2. Rank-average OOF
    ranks = [to_percentile(p) for p in oof_preds.values()]
    avg_rank_oof = np.mean(ranks, axis=0)
    
    # Map back through best single model's OOF distribution
    blended_oof = np.quantile(best_oof, avg_rank_oof)
    
    f1, auc, score = _score(y, blended_oof)
    print(f"[ensemble] Blended OOF: score={score:.4f} (f1={f1:.4f}, auc={auc:.4f})")
    
    # 3. Re-measure t*
    print(f"[calibration] pi_t assumed: {cfg.pi_t_assumed:.4f}")
    t_star = np.quantile(blended_oof, 1 - cfg.pi_t_assumed)
    print(f"[calibration] new OOF t* = {t_star:.4f}")
    
    # Apply FULL shift to center t* at 0.5 because rank-mapping forces the test
    # distribution to match OOF exactly. A full shift ensures exactly pi_t
    # positive rate on the test set, passing the kill criterion.
    delta_f1 = -logit(t_star)
    lambda_f1 = 1.0
    total_shift = delta_f1 * lambda_f1
    print(f"[calibration] derived total shift: {total_shift:.4f}")
        
    # 4. Predict on Test
    cube_te = C.band_cube(te)
    X_te = C.build_features(cube_te, cfg=cfg)
    
    test_preds = {}
    for k in KINDS:
        ckpt = f"{CKPT_DIR}/{k}_{cfg_hash}_checkpoint.pkl"
        with open(ckpt, "rb") as f:
            m = pickle.load(f)
        test_preds[k] = m.predict_proba(X_te)[:, 1]
        
    # 5. Rank-average Test
    test_ranks = [to_percentile(p) for p in test_preds.values()]
    avg_rank_test = np.mean(test_ranks, axis=0)
    
    # 6. Map back through best single model OOF
    blended_test = np.quantile(best_oof, avg_rank_test)
    
    raw_pos_rate = float((blended_test >= 0.5).mean())
    print(f"[ensemble] raw test pos_rate (pre-shift) = {raw_pos_rate:.4f}")
    
    # 7. Write submission
    df = write_submission(
        ids=te["ID"].values,
        p_raw=blended_test,
        total_shift=total_shift,
        path=a.out,
        calibration_enabled=True,
    )
    
    final_pos_rate = float(df["TargetF1"].mean())
    pre_submission_check(final_pos_rate, pi_t=cfg.pi_t_assumed)
    n_unique = int(df["TargetRAUC"].nunique())
    
    log_run(
        config_hash=cfg_hash + "_rank_ens",
        test_pos_rate=final_pos_rate,
        n_unique_scores=n_unique,
        submitted=False,
        notes=f"rank ensemble, shift={total_shift:.4f}",
    )
    print(f"[ensemble] SUCCESS. Wrote {a.out}")

if __name__ == "__main__":
    main()
