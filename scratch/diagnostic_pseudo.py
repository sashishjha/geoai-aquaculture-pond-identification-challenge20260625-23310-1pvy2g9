import sys, os, pickle
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, 'src')
import common as C
from config import RunConfig

KINDS = ["hgb", "lgbm", "xgb", "cat"]
CKPT_DIR = "model_checkpoints"

def main():
    cfg = RunConfig.load("config_s3.json")
    cfg_hash = cfg.hash()
    
    _, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    cube_te = C.band_cube(te)
    X_te = C.build_features(cube_te, cfg=cfg)
    
    test_preds = {}
    for k in KINDS:
        ckpt = f"{CKPT_DIR}/{k}_{cfg_hash}_checkpoint.pkl"
        with open(ckpt, "rb") as f:
            m = pickle.load(f)
        test_preds[k] = m.predict_proba(X_te)[:, 1]
        
    df = pd.DataFrame(test_preds)
    df['id'] = te['ID'].values
    
    # Confident positive: ALL > 0.95
    pos_mask = (df[KINDS] > 0.95).all(axis=1)
    
    # Confident negative: ALL < 0.05
    neg_mask = (df[KINDS] < 0.05).all(axis=1)
    
    print(f"Total Test Rows: {len(df)}")
    print(f"Confident Positives (>0.95 across all): {pos_mask.sum()}")
    print(f"Confident Negatives (<0.05 across all): {neg_mask.sum()}")
    
    total_conf = pos_mask.sum() + neg_mask.sum()
    print(f"Total Pseudo-labels available: {total_conf} ({total_conf/len(df)*100:.1f}%)")
    
    if total_conf > 0:
        pos_rate = pos_mask.sum() / total_conf
        print(f"Prevalence in Pseudo-labels: {pos_rate:.4f} (Target pi_t = 0.6175)")
        
if __name__ == "__main__":
    main()
