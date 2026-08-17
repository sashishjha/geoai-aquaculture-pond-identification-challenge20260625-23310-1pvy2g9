import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, 'src')
import common as C
from config import RunConfig

def load_oof(path):
    df = pd.read_csv(path)
    if 'TargetRAUC' in df.columns:
        return df['TargetRAUC'].values
    elif 'p_1' in df.columns:
        return df['p_1'].values
    else:
        cols = [c for c in df.columns if c not in ['id', 'target']]
        return df[cols[-1]].values

def apply_shift(p, train_prev=0.404, test_prev=0.6175):
    from scipy.special import logit, expit
    # Ensure p doesn't hit 0 or 1
    p = np.clip(p, 1e-15, 1 - 1e-15)
    # Convert to log-odds
    odds = p / (1 - p)
    # Apply Bayes shift
    shift = (test_prev / (1 - test_prev)) / (train_prev / (1 - train_prev))
    new_odds = odds * shift
    return new_odds / (1 + new_odds)

def main():
    print("Building Domain-Proximal Proxy (DPP)...")
    
    cfg = RunConfig.load("config_s3.json")
    tr, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    cube_tr = C.band_cube(tr)
    cube_te = C.band_cube(te)
    
    rng = np.random.default_rng(42)
    cube_tr_masked, _ = C.augment_training_data(cube_tr, tr['label'].values, n_copies=1, seed=42)
    cube_tr_masked = cube_tr_masked[:len(cube_tr)]
    
    X_train_masked = C.build_features(cube_tr_masked, cfg=cfg)
    X_test = C.build_features(cube_te, cfg=cfg)
    
    print("Training adversarial discriminator...")
    groups = tr.index // 25
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                              random_state=42, verbosity=-1, n_jobs=1)
                              
    p_test = np.zeros(len(tr))
    
    for tr_i, va_i in skf.split(X_train_masked, tr['label'].values, groups=groups):
        X_dom_tr = pd.concat([X_train_masked.iloc[tr_i], X_test], ignore_index=True)
        y_dom_tr = np.array([0]*len(tr_i) + [1]*len(X_test))
        
        clf.fit(X_dom_tr, y_dom_tr)
        p_test[va_i] = clf.predict_proba(X_train_masked.iloc[va_i])[:, 1]
        
    print(f"Mean p(test|x) on train OOF: {np.mean(p_test):.4f}")
    
    cutoff = np.percentile(p_test, 75)
    test_like_idx = np.where(p_test >= cutoff)[0]
    print(f"Top 25% cutoff: {cutoff:.4f}. Number of test-like rows: {len(test_like_idx)}")
    
    np.save('data_checkpoint/dpp_test_like_idx.npy', test_like_idx)
    print("Saved test-like indices to data_checkpoint/dpp_test_like_idx.npy")
    
    y_true = tr['label'].values
    
    # Let's read experiment_log.csv to dynamically find the hashes
    log = pd.read_csv('experiments_checkpoint/experiment_log.csv')
    
    # We want:
    # S3: "valid-month temporal+missingness+spectral", no phenology, config ebf7b651583b
    # S4: rank ensemble, e5c8fdf4e68b
    # S5: pseudo ensemble, 1de72f8b7995
    # S6: phenology ensemble, 690022d2dae8
    
    oofs = {
        "S3_baseline": "data_checkpoint/oof_ebf7b651583b.csv",
        "S4_rank_ens": "data_checkpoint/oof_e5c8fdf4e68b.csv", 
        "S5_pseudo": "data_checkpoint/oof_1de72f8b7995.csv",
        "S6_phenology": "data_checkpoint/oof_690022d2dae8.csv",
        "S7_phenology_w2": "data_checkpoint/oof_1fad1f7a0efc.csv"
    }
    
    def score_dpp(p, idx):
        y_t = y_true[idx]
        p_t = p[idx]
        pred_binary = (p_t >= 0.5).astype(int)
        
        if len(np.unique(y_t)) < 2: return 0, 0, 0
        f1 = f1_score(y_t, pred_binary)
        auc = roc_auc_score(y_t, p_t)
        score = 0.6 * f1 + 0.4 * auc
        return score, f1, auc

    print("\n--- DPP Evaluation (Top 25% Test-Like) ---")
    for name, path in oofs.items():
        try:
            p = load_oof(path)
            if np.min(p) < 0 or np.max(p) > 1:
                from scipy.special import expit
                p = expit(p)
                
            p_calib = apply_shift(p, 0.404, 0.6175)
            
            score, f1, auc = score_dpp(p_calib, test_like_idx)
            print(f"{name:>15}: DPP Score = {score:.5f}  (F1={f1:.4f}, AUC={auc:.4f})")
            
            # also score on the flat OOF for reference
            flat_s, flat_f, flat_a = score_dpp(p_calib, np.arange(len(p)))
            print(f"{name:>15} [FLAT] : Score = {flat_s:.5f}  (F1={flat_f:.4f}, AUC={flat_a:.4f})")
            
        except FileNotFoundError:
            print(f"{name}: File not found ({path})")

if __name__ == "__main__":
    main()
