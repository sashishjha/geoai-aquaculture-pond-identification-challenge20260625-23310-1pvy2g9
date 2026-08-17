import sys, os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import logit
from scipy.optimize import minimize

sys.path.insert(0, 'src')
import common as C
from config import RunConfig
from train_model import _score

KINDS = ["hgb", "lgbm", "xgb", "cat"]
OOF_DIR = "oof_predictions"

def to_percentile(arr):
    return (rankdata(arr) - 1) / (len(arr) - 1)

def main():
    cfg = RunConfig.load("config_s3.json")
    cfg_hash = cfg.hash()
    
    tr, _ = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    y = tr["label"].values
    
    oof_preds = {}
    for k in KINDS:
        path = f"{OOF_DIR}/{k}_{cfg_hash}_oof.npy"
        oof_preds[k] = np.load(path)
        
    best_k, best_oof = None, None
    best_score = -1
    for k, p in oof_preds.items():
        f1, auc, score = _score(y, p)
        print(f"{k}: score={score:.5f} (f1={f1:.5f}, auc={auc:.5f})")
        if score > best_score:
            best_score = score
            best_k, best_oof = k, p
            
    ranks = [to_percentile(p) for p in oof_preds.values()]
    
    def objective(w):
        w = np.array(w) / np.sum(w)
        avg_rank = np.average(ranks, axis=0, weights=w)
        blended = np.quantile(best_oof, avg_rank)
        f1, auc, score = _score(y, blended)
        return -score
        
    print("\nOptimizing weights...")
    res = minimize(objective, x0=[0.25]*4, bounds=[(0, 1)]*4, method='SLSQP')
    best_w = res.x / np.sum(res.x)
    
    print("\nBest Weights:", dict(zip(KINDS, best_w)))
    print("Best OOF Score:", -res.fun)
    
    avg_rank = np.average(ranks, axis=0, weights=[0.25]*4)
    blended = np.quantile(best_oof, avg_rank)
    f1, auc, base_score = _score(y, blended)
    print("Uniform Weights Score:", base_score)

if __name__ == "__main__":
    main()
