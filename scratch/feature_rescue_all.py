import sys, os
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, 'src')
import common as C
from config import RunConfig, CHAMPION_CONFIG
from proxy_validation import run_proxy

DANGER_FEATS = [
    "sar_ratio_median", "evi_median", "sar_ratio_mean", "mndwi_std", "evi_max", 
    "swir2_std", "bsi_max", "VH_min", "evi_min", "ndwi_min", "sar_ratio_min", 
    "VV_median", "sar_ratio_max", "VH_range", "VH_std", "blue_max", "ndbi_min", 
    "evi_std", "ndbi_max"
]

def main():
    print("Computing S3 Baseline + ALL Rescue Features Proxy...", flush=True)
    cfg_all = RunConfig(
        use_absolute_anchors=True, use_z_shape_features=False, 
        use_rank_features=False, use_av_weights=False, 
        n_augmentation_copies=1, drop_features=DANGER_FEATS,
        use_rescue_features=True
    )
    
    res = run_proxy(cfg_all, pi_t=0.6175, n_seeds=1, kinds=["lgbm"])
    score = res['kfold_score_mean']
    std = res['kfold_score_std']
    score_logit = logit(score)
    
    print(f"\nALL Rescue Features Proxy: Score={score:.4f}±{std:.4f} (Logit: {score_logit:.4f})", flush=True)
    
if __name__ == "__main__":
    main()
