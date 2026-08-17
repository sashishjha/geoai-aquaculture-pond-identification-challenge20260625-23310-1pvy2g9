"""
adversarial_validation.py — Phase 6 Diagnostic

Trains an adversarial classifier to separate augmented/masked train from real test.
Uses the champion feature configuration.
Sorts features by their power to discriminate between domains.
"""
import sys, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, 'src')
import common as C
from config import CHAMPION_CONFIG

def main():
    print(f"[av] Config: {CHAMPION_CONFIG.hash()}")
    tr, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv', dedup=True)
    
    rng = np.random.default_rng(C.SEED)
    print("[av] Masking train set...")
    tr_cube_masked = C.simulate_test_masking(C.band_cube(tr), rng)
    
    print("[av] Building features...")
    X_tr = C.build_features(tr_cube_masked, cfg=CHAMPION_CONFIG)
    X_te = C.build_features(C.band_cube(te), cfg=CHAMPION_CONFIG)
    
    X_tr["is_test"] = 0
    X_te["is_test"] = 1
    
    df = pd.concat([X_tr, X_te], axis=0).reset_index(drop=True)
    X = df.drop(columns=["is_test"])
    y = df["is_test"].values
    
    print(f"[av] Data shape: {X.shape}, train={len(X_tr)}, test={len(X_te)}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    importances = np.zeros(X.shape[1])
    
    oof_preds = np.zeros(len(df))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        clf = lgb.LGBMClassifier(random_state=42+fold, n_estimators=100, max_depth=6, verbosity=-1)
        clf.fit(X_train, y_train)
        
        preds = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
        oof_preds[val_idx] = preds
        
        importances += clf.feature_importances_ / 5.0
        
    print(f"\n[av] Masked Train vs Test AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    
    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)
    
    print("\n[av] Top 20 Most Discriminative Features (Domain Indicators):")
    print(imp_df.head(20).to_string(index=False))
    
    imp_df.to_csv("data_checkpoint/av_feature_importance.csv", index=False)
    
if __name__ == "__main__":
    main()
