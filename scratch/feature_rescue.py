"""
Evaluates Phase 6.5 Feature Rescue candidates individually.
"""
import sys, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.special import logit

sys.path.insert(0, 'src')
import common as C
from config import RunConfig
from proxy_validation import run_proxy

DANGER_FEATS = [
    "sar_ratio_median", "evi_median", "sar_ratio_mean", "mndwi_std", "evi_max", 
    "swir2_std", "bsi_max", "VH_min", "evi_min", "ndwi_min", "sar_ratio_min", 
    "VV_median", "sar_ratio_max", "VH_range", "VH_std", "blue_max", "ndbi_min", 
    "evi_std", "ndbi_max"
]

def eval_candidate(feature_name: str, X_tr_full, X_te_full, y_tr, base_score_mean, base_score_logit, base_std) -> bool:
    print(f"\n--- Evaluating Candidate: {feature_name} ---", flush=True)
    
    # 1. Adversarial AUC check
    X_tr_adv = X_tr_full[[feature_name]].copy()
    X_tr_adv["is_test"] = 0
    X_te_adv = X_te_full[[feature_name]].copy()
    X_te_adv["is_test"] = 1
    
    df_adv = pd.concat([X_tr_adv, X_te_adv], axis=0).reset_index(drop=True)
    X_a = df_adv[[feature_name]]
    y_a = df_adv["is_test"].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X_a, y_a):
        clf = lgb.LGBMClassifier(random_state=42, n_estimators=50, max_depth=3, verbosity=-1)
        clf.fit(X_a.iloc[train_idx], y_a[train_idx])
        preds = clf.predict_proba(X_a.iloc[val_idx])[:, 1]
        aucs.append(roc_auc_score(y_a[val_idx], preds))
        
    adv_auc = np.mean(aucs)
    print(f"[{feature_name}] Adversarial AUC: {adv_auc:.4f} (must not approach 0.98)", flush=True)
    if adv_auc > 0.85:
        print(f"[{feature_name}] REJECTED: Failed Domain Invariance (Adv AUC too high)")
        return False
        
    # 2. Proxy check
    class SingleFeatureRescueConfig(RunConfig):
        def __init__(self, keep_feat):
            super().__init__(
                use_absolute_anchors=True, use_z_shape_features=False, 
                use_rank_features=False, use_av_weights=False, 
                n_augmentation_copies=1, drop_features=DANGER_FEATS,
                use_rescue_features=True
            )
            self.keep_feat = keep_feat
    
    cfg = SingleFeatureRescueConfig(feature_name)
    
    # Monkeypatch to drop ALL rescue features EXCEPT the candidate
    orig_build = C.build_features
    def custom_build(cube, cfg=None):
        df = orig_build(cube, cfg=cfg)
        rescue_cols = [c for c in df.columns if "_rescue_" in c]
        drops = [c for c in rescue_cols if c != cfg.keep_feat]
        return df.drop(columns=drops)
    
    C.build_features = custom_build
    
    # Evaluate on proxy (1 seed, lgbm only for speed)
    res = run_proxy(cfg, pi_t=0.6175, n_seeds=1, kinds=["lgbm"])
    proxy_score = res['kfold_score_mean']
    proxy_logit = logit(proxy_score)
    
    delta_logit = proxy_logit - base_score_logit
    print(f"[{feature_name}] Proxy Score: {proxy_score:.4f} (Logit: {proxy_logit:.4f})", flush=True)
    print(f"[{feature_name}] Delta vs S3 (logit): {delta_logit:.4f} (threshold: > {base_std:.4f})", flush=True)
    
    C.build_features = orig_build  # restore
    
    if delta_logit > base_std:
        print(f"[{feature_name}] ACCEPTED: Beats S3 by >1 std in logit space.")
        return True
    else:
        print(f"[{feature_name}] REJECTED: Not enough proxy gain.")
        return False

def main():
    # Base S3
    cfg_s3 = RunConfig(
        use_absolute_anchors=True, use_z_shape_features=False, 
        use_rank_features=False, use_av_weights=False, 
        n_augmentation_copies=1, drop_features=DANGER_FEATS,
        use_rescue_features=False
    )
    print("Computing S3 Baseline Proxy (1 seed, lgbm only)...", flush=True)
    base_res = run_proxy(cfg_s3, pi_t=0.6175, n_seeds=1, kinds=["lgbm"])
    s3_score = base_res['kfold_score_mean']
    s3_std = base_res['kfold_score_std']
    # If 1 seed, std might be 0, so set a minimum std of 0.002
    if s3_std < 0.002: s3_std = 0.002
    s3_logit = logit(s3_score)
    s3_std_logit = (logit(s3_score + s3_std) - logit(s3_score - s3_std)) / 2
    
    print(f"S3 Baseline: Score={s3_score:.4f}±{s3_std:.4f} (Logit={s3_logit:.4f}±{s3_std_logit:.4f})", flush=True)
    
    # Build full cube once with all rescue features
    cfg_all = RunConfig(
        use_absolute_anchors=True, use_z_shape_features=False, 
        use_rank_features=False, use_av_weights=False, 
        n_augmentation_copies=1, drop_features=DANGER_FEATS,
        use_rescue_features=True
    )
    
    tr, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv', dedup=True)
    rng = np.random.default_rng(C.SEED)
    tr_cube_masked = C.simulate_test_masking(C.band_cube(tr), rng)
    X_tr_full = C.build_features(tr_cube_masked, cfg=cfg_all)
    X_te_full = C.build_features(C.band_cube(te), cfg=cfg_all)
    
    rescue_candidates = [c for c in X_tr_full.columns if "_rescue_" in c]
    print(f"Found {len(rescue_candidates)} rescue candidates.", flush=True)
    
    accepted = []
    for cand in rescue_candidates:
        if eval_candidate(cand, X_tr_full, X_te_full, tr['label'].values, s3_score, s3_logit, s3_std_logit):
            accepted.append(cand)
        import gc; gc.collect()
    print("\n--- FINAL ACCEPTED FEATURES ---", flush=True)
    print(accepted, flush=True)

if __name__ == "__main__":
    main()
