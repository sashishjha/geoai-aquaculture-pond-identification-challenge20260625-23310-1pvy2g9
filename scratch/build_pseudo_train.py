"""
scratch/build_pseudo_train.py — Phase 8: Build balanced pseudo-augmented training cube.

Rules (per AI expert review):
1. Balance pseudo-labels to match pi_t=0.6175 (438 pos, 271 neg from base model gate).
2. Hold out 20% of confident rows (~140) for the consistency gate — never trained on.
3. Diversify the 5 copies with fresh ~7% S2 cloud dropout (S1 untouched).
4. Pseudo-rows MUST NOT enter validation folds (handled in train_model_pseudo.py).
5. Save a manifest so validation gates can be run later.
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, 'src')
import common as C
from config import RunConfig

KINDS = ["hgb", "lgbm", "xgb", "cat"]
CKPT_DIR = "model_checkpoints"
PI_T = 0.6175

def s2_only_cloud_dropout(cube: np.ndarray, rng: np.random.Generator, rate: float = 0.07) -> np.ndarray:
    """Apply fresh S2-only cloud dropout at a fixed rate per copy. S1 untouched."""
    out = cube.copy()
    N = out.shape[0]
    s2_idx = [C.ALL_BANDS.index(b) for b in C.S2_BANDS]
    # Get months that are currently valid (not SENTINEL) for each row
    for i in range(N):
        for m in range(12):
            # Only drop S2 if this month is already observed (S1 is non-sentinel)
            if out[i, m, 0] != C.SENTINEL:  # S1 channel is present
                if rng.random() < rate:
                    out[i, m, s2_idx] = C.SENTINEL
    return out


def main():
    cfg = RunConfig.load("config_s3.json")
    cfg_hash = cfg.hash()
    
    _, te = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    cube_te = C.band_cube(te)
    test_ids = te['ID'].values
    
    # Load all base model test predictions (from checkpoints — uncalibrated)
    test_preds = {}
    for k in KINDS:
        ckpt = f"{CKPT_DIR}/{k}_{cfg_hash}_checkpoint.pkl"
        with open(ckpt, "rb") as f:
            m = pickle.load(f)
        test_preds[k] = m.predict_proba(C.build_features(cube_te, cfg=cfg))[:, 1]
    
    df = pd.DataFrame(test_preds)
    df['id'] = test_ids
    
    # Gate: all 4 models must agree
    pos_mask = (df[KINDS] > 0.95).all(axis=1)
    neg_mask = (df[KINDS] < 0.05).all(axis=1)
    
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]
    
    print(f"Confident positives: {len(pos_idx)}")
    print(f"Confident negatives: {len(neg_idx)}")
    
    # Balance to pi_t: keep all 438 positives, subsample negatives
    n_pos = len(pos_idx)
    n_neg_target = round(n_pos * (1 - PI_T) / PI_T)  # 438 * 0.3825 / 0.6175 = 271
    
    rng_balance = np.random.default_rng(42)
    # Sort by most confident (lowest mean) to keep the most certain negatives
    neg_confidences = df.loc[neg_mask, KINDS].mean(axis=1).values
    neg_order = np.argsort(neg_confidences)  # ascending = most negative first
    neg_idx_sorted = neg_idx[neg_order]
    neg_idx_selected = neg_idx_sorted[:n_neg_target]
    
    print(f"Balanced negatives selected: {len(neg_idx_selected)} (target pi_t={PI_T:.4f})")
    print(f"Actual prevalence: {n_pos/(n_pos+len(neg_idx_selected)):.4f}")
    
    # Hold out 20% of the confident set for the consistency gate
    all_confident_idx = np.concatenate([pos_idx, neg_idx_selected])
    all_confident_labels = np.concatenate([np.ones(n_pos), np.zeros(len(neg_idx_selected))])
    
    shuffle_order = rng_balance.permutation(len(all_confident_idx))
    all_confident_idx = all_confident_idx[shuffle_order]
    all_confident_labels = all_confident_labels[shuffle_order]
    
    n_holdout = int(0.20 * len(all_confident_idx))
    holdout_idx = all_confident_idx[:n_holdout]
    holdout_labels = all_confident_labels[:n_holdout]
    train_pseudo_idx = all_confident_idx[n_holdout:]
    train_pseudo_labels = all_confident_labels[n_holdout:]
    
    print(f"\nTrainable pseudo rows: {len(train_pseudo_idx)}")
    print(f"Held-out (gate only): {len(holdout_idx)}")
    
    # Save holdout manifest for the consistency gate
    holdout_df = pd.DataFrame({
        'test_row_idx': holdout_idx,
        'label': holdout_labels.astype(int),
        'id': test_ids[holdout_idx]
    })
    holdout_df.to_csv("scratch/pseudo_holdout_manifest.csv", index=False)
    print("Saved holdout manifest -> scratch/pseudo_holdout_manifest.csv")
    
    # Build 5 diversified copies of the trainable pseudo-rows
    rng_aug = np.random.default_rng(123)
    pseudo_cubes = []
    pseudo_labels_all = []
    for copy_i in range(5):
        base_cube = cube_te[train_pseudo_idx]  # raw test cube (no masking yet)
        # Apply fresh S2 dropout to diversify
        augmented_cube = s2_only_cloud_dropout(base_cube, rng_aug, rate=0.07)
        pseudo_cubes.append(augmented_cube)
        pseudo_labels_all.append(train_pseudo_labels)
    
    pseudo_cube = np.concatenate(pseudo_cubes, axis=0)
    pseudo_y = np.concatenate(pseudo_labels_all, axis=0).astype(int)
    
    print(f"\nFinal pseudo cube shape: {pseudo_cube.shape}")
    print(f"Final pseudo labels shape: {pseudo_y.shape}")
    print(f"Pseudo label prevalence: {pseudo_y.mean():.4f}")
    
    # Save to disk for injection during training
    np.save("data_checkpoint/pseudo_cube.npy", pseudo_cube)
    np.save("data_checkpoint/pseudo_y.npy", pseudo_y)
    print("Saved pseudo_cube.npy and pseudo_y.npy -> data_checkpoint/")
    
    # Also save the uncertain rows (not confident) for the sanity gate
    uncertain_mask = ~pos_mask & ~neg_mask
    uncertain_idx = np.where(uncertain_mask)[0]
    uncertain_scores = df.loc[uncertain_mask, KINDS].mean(axis=1).values
    uncertain_df = pd.DataFrame({
        'test_row_idx': uncertain_idx,
        'id': test_ids[uncertain_idx],
        's4_mean_score': uncertain_scores
    })
    uncertain_df.to_csv("scratch/pseudo_uncertain_manifest.csv", index=False)
    print(f"Saved {len(uncertain_idx)} uncertain rows -> scratch/pseudo_uncertain_manifest.csv")


if __name__ == "__main__":
    main()
