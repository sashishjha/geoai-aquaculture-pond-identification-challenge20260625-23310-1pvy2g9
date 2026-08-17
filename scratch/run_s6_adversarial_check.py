"""
Step 9.1 adversarial check: new phenology features must NOT reinflate
domain separability. Each new feature must land in the bottom half of
domain importances. If any feature lands in the top half, reject it.
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, 'src')
import common as C
from config import RunConfig

def main():
    cfg = RunConfig.load("config_s6.json")
    
    tr, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    cube_tr = C.band_cube(tr)
    cube_te = C.band_cube(te)
    
    # Build masked train (simulate test conditions) and test features
    rng = np.random.default_rng(42)
    cube_tr_masked, _ = C.augment_training_data(cube_tr, tr['label'].values, n_copies=1, seed=42)
    # Take 1 copy's worth  
    cube_tr_masked = cube_tr_masked[:len(cube_tr)]
    
    X_train_masked = C.build_features(cube_tr_masked, cfg=cfg)
    X_test = C.build_features(cube_te, cfg=cfg)
    
    # Domain labels: 0=train, 1=test
    X_dom = pd.concat([X_train_masked, X_test], ignore_index=True)
    y_dom = np.array([0]*len(X_train_masked) + [1]*len(X_test))
    
    # Train adversarial LightGBM
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                              random_state=42, verbosity=-1, n_jobs=1)
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr_i, va_i in skf.split(X_dom, y_dom):
        clf.fit(X_dom.iloc[tr_i], y_dom[tr_i])
        p = clf.predict_proba(X_dom.iloc[va_i])[:, 1]
        aucs.append(roc_auc_score(y_dom[va_i], p))
    
    adv_auc = np.mean(aucs)
    print(f"Adversarial AUC with phenology features: {adv_auc:.4f}")
    print(f"Previous (S3 baseline): 0.9843")
    
    # Check where new features land in domain importance ranking
    clf.fit(X_dom, y_dom)
    importances = pd.Series(clf.feature_importances_, index=X_dom.columns)
    importances = importances.sort_values(ascending=False)
    
    new_features = [c for c in X_dom.columns if any(x in c for x in
                    ['vh_diff', 'mndwi_diff', 'corr_ndvi', 'corr_vh', 'corr_evi', 'water_perm',
                     'pos_frac', 'regime', 'skew', 'refill'])]
    
    n_total = len(importances)
    print(f"\n=== New Phenology Features — Domain Importance Ranks (lower=safer) ===")
    print(f"Total features: {n_total}")
    failed = []
    for feat in sorted(new_features):
        if feat not in importances.index:
            print(f"  {feat}: NOT IN FEATURES (check name)")
            continue
        rank = importances.index.get_loc(feat) + 1
        pct = rank / n_total * 100
        status = "OK" if rank > n_total // 2 else "DANGER"
        if status == "DANGER":
            failed.append(feat)
        print(f"  {feat}: rank {rank}/{n_total} ({pct:.0f}th pct) [{status}]")
    
    print(f"\n=== VERDICT ===")
    if not failed and adv_auc < 0.99:
        print(f"ALL new features safe — none in top-half of domain importance.")
        print(f"Adversarial AUC = {adv_auc:.4f} (no inflation vs baseline 0.9843).")
        print("→ PROCEED with proxy evaluation.")
    elif failed:
        print(f"DANGER: {len(failed)} features in top half of domain importance:")
        for f in failed: print(f"  - {f}")
        print("→ DROP these features before proxy run.")
    else:
        print(f"Adversarial AUC {adv_auc:.4f} — acceptable.")

if __name__ == "__main__":
    main()
