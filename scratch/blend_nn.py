import pandas as pd
import numpy as np
from scipy.special import logit, expit
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score
import sys
sys.path.append('src')
import common as C

def empirical_threshold(y_true, y_prob):
    # Find t* that maximizes F1
    best_f1 = 0
    best_t = 0.5
    for t in np.linspace(0.1, 0.9, 161):
        f1 = f1_score(y_true, y_prob >= t)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

def find_logit_shift(y_prob, target_t):
    # we want expit(logit(t*) + shift) = 0.5
    # shift = logit(0.5) - logit(t*) = -logit(t*)
    return -logit(target_t)

def apply_logit_shift(y_prob, shift):
    return expit(logit(np.clip(y_prob, 1e-15, 1-1e-15)) + shift)

def compute_proxy(y_true, y_prob_calib):
    auc = roc_auc_score(y_true, y_prob_calib)
    f1 = f1_score(y_true, y_prob_calib >= 0.5)
    return 0.6 * f1 + 0.4 * auc

def main():
    # Load GBM OOFs
    tr, _ = C.load_raw("data_checkpoint/raw/Train.csv", "data_checkpoint/raw/Test.csv")
    y = tr['label'].values
    hgb = np.load("oof_predictions/hgb_690022d2dae8_oof.npy")
    lgbm = np.load("oof_predictions/lgbm_690022d2dae8_oof.npy")
    xgb = np.load("oof_predictions/xgb_690022d2dae8_oof.npy")
    cat = np.load("oof_predictions/cat_690022d2dae8_oof.npy")
    
    gbm_blend_ranks = (rankdata(hgb) + rankdata(lgbm) + rankdata(xgb) + rankdata(cat)) / 4.0
    gbm_blend_probs = gbm_blend_ranks / len(y)
    
    nn_logits = np.load("model_checkpoints/gru_oof_preds.npy")
    nn_probs = expit(nn_logits)
    nn_ranks = rankdata(nn_probs) / len(y)
    
    # Base GBM Proxy
    s6_sorted = np.sort(gbm_blend_probs)
    base_t_star, base_f1 = empirical_threshold(y, gbm_blend_probs)
    base_shift = find_logit_shift(gbm_blend_probs, base_t_star)
    base_calib = apply_logit_shift(gbm_blend_probs, base_shift)
    base_proxy = compute_proxy(y, base_calib)
    print(f"Base GBM Proxy: {base_proxy:.6f} (Logit: {logit(base_proxy):.4f})")
    
    # 9.2c - Admission Gates
    gbm_auc = roc_auc_score(y, gbm_blend_probs)
    nn_auc = roc_auc_score(y, nn_probs)
    print(f"GBM Blend AUC: {gbm_auc:.4f}")
    print(f"NN Solo AUC:   {nn_auc:.4f}")
    
    if abs(gbm_auc - nn_auc) > 0.03:
        print("FAIL GATE 1: NN Solo OOF AUC not within 0.03 of GBM blend.")
        # We might still want to proceed to see
    
    spearman = np.corrcoef(gbm_blend_ranks, nn_ranks)[0, 1]
    print(f"Spearman rank corr (GBM vs NN): {spearman:.4f}")
    
    if spearman >= 0.85:
        print("FAIL GATE 2: NN predictions too correlated with GBM blend (>0.85).")
        
    print("\n--- Sweeping NN weight ---")
    best_proxy = 0
    best_w = 0
    
    # We map the blended ranks to the original S3 (or S6) score distribution
    # Actually, we can map them to the S6 distribution.
    # To do that, we sort S6 probs and apply them to the blended ranks.
    s6_sorted = np.sort(gbm_blend_probs)
    
    for w in np.linspace(0.15, 0.30, 16):
        blend_ranks = rankdata((1 - w) * gbm_blend_ranks + w * nn_ranks)
        
        # Map to reference distribution (S6's probability distribution shape)
        mapped_probs = np.zeros(len(y))
        for i in range(len(y)):
            mapped_probs[i] = s6_sorted[int(blend_ranks[i]) - 1]
            
        t_star, f1 = empirical_threshold(y, mapped_probs)
        shift = find_logit_shift(mapped_probs, t_star)
        calib_probs = apply_logit_shift(mapped_probs, shift)
        
        proxy = compute_proxy(y, calib_probs)
        
        print(f"Weight: {w:.2f} | t*: {t_star:.3f} | shift: {shift:+.3f} | Proxy: {proxy:.6f} (Logit: {logit(proxy):.4f})")
        
        if proxy > best_proxy:
            best_proxy = proxy
            best_w = w
            
    print(f"\nBest Weight: {best_w:.2f} | Best Proxy: {best_proxy:.4f} (Logit: {logit(best_proxy):.4f})")
    
if __name__ == "__main__":
    main()
