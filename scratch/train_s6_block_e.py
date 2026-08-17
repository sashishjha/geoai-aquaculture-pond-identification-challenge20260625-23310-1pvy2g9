import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy.optimize import minimize
from scipy.special import expit, logit

sys.path.append('src')
import common as C
from train_model import make_model, _score
from config import RunConfig

# S2 Cloud Rates for Test (empirical) -> average is roughly 7% overall
S2_IDX = [C.ALL_BANDS.index(b) for b in C.S2_BANDS]

def apply_s2_dropout(cube, rng, dropout_rate=0.07):
    """Applies additional random S2 dropout to an already-masked cube."""
    out = cube.copy()
    N, M, _ = out.shape
    
    # We only apply dropout where there is currently an observation
    valid = C._valid_mask(out)
    
    # dropout rate applies uniformly across all S2 bands for a given (row, month)
    # i.e., cloud covers all optical bands
    drop_mask = rng.random((N, M)) < dropout_rate
    
    for i in range(N):
        for m in range(M):
            if drop_mask[i, m]:
                out[i, m, S2_IDX] = C.SENTINEL
                
    return out

def get_tta_views(cube, rng, n_views=8):
    """
    Returns n_views of the cube:
    - View 0: original
    - Views 1 to n_views-1: extra S2 dropout
    """
    views = [cube]
    for _ in range(n_views - 1):
        views.append(apply_s2_dropout(cube, rng))
    return views

def find_optimal_t(y_true, y_prob):
    """Find threshold maximizing S = 0.6*F1 + 0.4*AUC."""
    auc = roc_auc_score(y_true, y_prob)
    
    def obj(t):
        p_bin = (y_prob >= t[0]).astype(int)
        f1 = f1_score(y_true, p_bin)
        return -(0.6 * f1 + 0.4 * auc)
        
    res = minimize(obj, [0.5], bounds=[(0.01, 0.99)], method='Nelder-Mead')
    best_t = res.x[0]
    best_f1 = f1_score(y_true, (y_prob >= best_t).astype(int))
    best_score = 0.6 * best_f1 + 0.4 * auc
    return best_t, best_score, best_f1, auc

def find_global_logit_shift(y_true, y_prob, t_star, target_t=0.5):
    """Finds logit shift so that target_t matches the performance of t_star."""
    log_odds = logit(y_prob.clip(1e-7, 1-1e-7))
    
    # We want expit(log_odds + shift) >= target_t to be exactly y_prob >= t_star
    # log_odds + shift = logit(target_t) when y_prob = t_star
    # shift = logit(target_t) - logit(t_star)
    shift = logit(target_t) - logit(t_star)
    return shift

def main():
    print("Loading data...")
    cfg = RunConfig.load('config_s6.json')
    tr_df, te_df = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    
    cube_tr = C.band_cube(tr_df)
    y_tr = tr_df['label'].values
    
    cube_te = C.band_cube(te_df)
    
    # 5-fold CV setup
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    group_id = np.arange(len(y_tr)) // cfg.cv_block_size
    rng = np.random.default_rng(C.SEED)
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    
    kinds = ["lgbm", "xgb", "hgb", "cat"]
    seeds = [42, 43, 44]
    
    oof_preds = np.zeros(len(y_tr))
    models_list = [] # List of tuples (model, kind, seed)
    
    print("\n--- Training & OOF TTA ---")
    t0 = time.time()
    
    for fold_idx, (tr_i, va_i) in enumerate(skf.split(cube_tr, y_tr, groups=group_id)):
        print(f"Fold {fold_idx+1}/5")
        
        # 1. Augment Training Data
        aug_cube, aug_y = C.augment_training_data(
            cube_tr[tr_i], y_tr[tr_i], n_copies=cfg.n_augmentation_copies, seed=C.SEED + fold_idx
        )
        Xtr = C.build_features(aug_cube, cfg=cfg)
        
        # 2. Train Models (Bagging over seeds)
        fold_models = []
        for s in seeds:
            for k in kinds:
                m = make_model(k, s, pos_w)
                m.fit(Xtr, aug_y)
                fold_models.append((m, k, s))
                models_list.append((m, k, s))
                
        # 3. Predict on Validation with TTA
        # Base validation set simulates the Test domain masking once
        base_va_cube = C.simulate_test_masking(cube_tr[va_i], rng)
        
        # Generate 8 views
        va_views = get_tta_views(base_va_cube, rng, n_views=8)
        
        # Predict with all models on all views
        all_logits = []
        for v_cube in va_views:
            Xva = C.build_features(v_cube, cfg=cfg)
            for m, k, s in fold_models:
                p = m.predict_proba(Xva)[:, 1]
                l = logit(p.clip(1e-7, 1-1e-7))
                all_logits.append(l)
                
        # Average logits
        avg_logits = np.mean(all_logits, axis=0)
        p_avg = expit(avg_logits)
        oof_preds[va_i] = p_avg
        
    print(f"Training completed in {time.time()-t0:.1f}s")
    
    # OOF Evaluation & Calibration
    f1, auc, wsc = _score(y_tr, oof_preds)
    print(f"\n[S6 Block E] Default (t=0.5) CV: F1={f1:.4f} AUC={auc:.4f} score={wsc:.4f}")
    
    t_star, best_score, best_f1, best_auc = find_optimal_t(y_tr, oof_preds)
    print(f"[S6 Block E] Optimal t*={t_star:.4f} -> F1={best_f1:.4f} AUC={best_auc:.4f} score={best_score:.4f}")
    
    shift = find_global_logit_shift(y_tr, oof_preds, t_star, target_t=0.5)
    print(f"[S6 Block E] Logit shift applied: {shift:.4f}")
    
    # Apply shift to OOF and re-evaluate at t=0.5 to verify
    shifted_oof_logits = logit(oof_preds.clip(1e-7, 1-1e-7)) + shift
    shifted_oof_preds = expit(shifted_oof_logits)
    sf1, sauc, swsc = _score(y_tr, shifted_oof_preds)
    print(f"[S6 Block E] Shifted CV at t=0.5: F1={sf1:.4f} AUC={sauc:.4f} score={swsc:.4f}")
    
    # Save OOF
    pd.DataFrame({'Target': y_tr, 'pred': shifted_oof_preds}).to_csv("data_checkpoint/oof_s6_block_e.csv", index=False)
    
    print("\n--- Test Prediction with TTA ---")
    te_views = get_tta_views(cube_te, rng, n_views=8)
    
    all_te_logits = []
    # Predict all 60 models (5 folds * 12 models) on all 8 views
    # Wait, actually we have 60 models total, and 8 views. That's 480 predictions per row!
    for v_cube in te_views:
        Xte = C.build_features(v_cube, cfg=cfg)
        for m, k, s in models_list:
            p = m.predict_proba(Xte)[:, 1]
            l = logit(p.clip(1e-7, 1-1e-7))
            all_te_logits.append(l)
            
    avg_te_logits = np.mean(all_te_logits, axis=0)
    shifted_te_logits = avg_te_logits + shift
    final_te_preds = expit(shifted_te_logits)
    
    sub = pd.DataFrame({'ID': te_df['ID']})
    sub['TargetRAUC'] = final_te_preds
    sub['TargetF1'] = (final_te_preds >= 0.5).astype(int)
    
    sub_path = "submissions/s6_block_e_submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved {sub_path}")
    print(f"Positive rate: {sub['TargetF1'].mean():.4f} (pi_t=0.6175)")
    
if __name__ == '__main__':
    main()
