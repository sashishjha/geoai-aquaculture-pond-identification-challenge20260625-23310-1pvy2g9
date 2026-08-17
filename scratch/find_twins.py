import sys
import numpy as np
import pandas as pd
from scipy.stats import iqr

sys.path.append('src')
import common as C

# Constants
S1_BANDS = ['VV', 'VH']
S2_BANDS = ['blue', 'green', 'red', 'rededge1', 'rededge2', 'rededge3', 'nir', 'nir08', 'swir1', 'swir2']
ALL_BANDS = S1_BANDS + S2_BANDS
S1_IDX = [0, 1]
S2_IDX = list(range(2, 12))

def build_q_cube(cube, valid):
    """Domain-specific quantile normalization (View 1)."""
    N, M, B = cube.shape
    q_cube = np.full_like(cube, np.nan, dtype=float)
    
    for m in range(M):
        for b in range(B):
            v = valid[:, m, b]
            if v.sum() > 0:
                vals = cube[v, m, b]
                # Map to uniform [0, 1] using argsort ranks
                ranks = np.empty_like(vals, dtype=float)
                # handle ties by averaging ranks
                from scipy.stats import rankdata
                ranks = rankdata(vals) / len(vals)
                q_cube[v, m, b] = ranks
                
    return q_cube

def vectorized_match_query(query_cube, query_q, query_valid, cand_cube, cand_q, cand_valid):
    """
    Match a single query (12, 12) against all candidates (N, 12, 12).
    query_valid specifies which (month, band) pairs we MUST match on.
    """
    N = cand_cube.shape[0]
    cand_has_all = (cand_valid & query_valid).sum(axis=(1, 2)) == query_valid.sum()
    
    valid_cands = np.where(cand_has_all)[0]
    if len(valid_cands) == 0:
        return None, np.inf, np.inf
        
    # View 1 (Quantile MAE)
    q_diff = np.abs(cand_q[valid_cands][:, query_valid] - query_q[query_valid])
    v1_dist = q_diff.mean(axis=1)
    
    # View 2 (Shape MAE)
    # We must z-score the candidate over ONLY the query_valid months!
    query_z = np.full_like(query_cube, np.nan, dtype=float)
    cand_z = np.full((len(valid_cands), 12, 12), np.nan, dtype=float)
    
    for b in range(12):
        v_b = query_valid[:, b]
        if v_b.sum() > 0:
            # Query
            qb_vals = query_cube[v_b, b]
            if len(qb_vals) > 1:
                iqr_q = iqr(qb_vals)
                if iqr_q > 1e-6:
                    query_z[v_b, b] = (qb_vals - np.median(qb_vals)) / iqr_q
                else:
                    query_z[v_b, b] = 0.0
            else:
                query_z[v_b, b] = 0.0
                
            # Cands
            cb_vals = cand_cube[valid_cands][:, v_b, b] # (num_cands, num_valid_months)
            if cb_vals.shape[1] > 1:
                # np.percentile over axis=1
                q75, q25 = np.percentile(cb_vals, [75, 25], axis=1)
                iqr_c = q75 - q25
                med_c = np.median(cb_vals, axis=1)
                
                valid_iqr = iqr_c > 1e-6
                
                # Expand dims for broadcasting
                cb_z = np.zeros_like(cb_vals)
                cb_z[valid_iqr] = (cb_vals[valid_iqr] - med_c[valid_iqr, None]) / iqr_c[valid_iqr, None]
                cand_z[:, v_b, b] = cb_z
            else:
                cand_z[:, v_b, b] = 0.0
                
    v2_diff = np.abs(cand_z[:, query_valid] - query_z[query_valid])
    v2_dist = v2_diff.mean(axis=1)
        
    best_idx = np.argmin(v1_dist)
    return valid_cands[best_idx], v1_dist[best_idx], v2_dist[best_idx]

def simulate_query_degradation(cube, y, rng):
    """
    Takes full train rows and degrades them to test-like queries.
    - Mask to 4-6 month window
    - 7% S2 dropout
    - Affine jitter (per-channel)
    """
    N, M, B = cube.shape
    q_cube = cube.copy()
    
    # Apply affine jitter (gain drift) to simulate 2024->2026 sensor shift
    gain = rng.uniform(0.9, 1.1, size=(N, 1, B))
    bias = rng.normal(0, 0.05, size=(N, 1, B))
    q_cube = q_cube * gain + bias
    
    # Apply test masking
    q_cube = C.simulate_test_masking(q_cube, rng)
    
    q_valid = C._valid_mask(q_cube)
    return q_cube, q_valid

def main():
    print("Loading data...")
    tr_df, te_df = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    # train has 1817 rows (duplicates removed)
    
    train_cube = C.band_cube(tr_df)
    test_cube = C.band_cube(te_df)
    
    train_valid = C._valid_mask(train_cube)
    test_valid = C._valid_mask(test_cube)
    
    print("Computing Q-cubes (Domain Quantile Normalization)...")
    q_train = build_q_cube(train_cube, train_valid)
    q_test = build_q_cube(test_cube, test_valid)
    
    # Train-to-Train Validation (with degradation)
    print("\n--- Validating on Degraded Train ---")
    rng = np.random.default_rng(42)
    deg_train_cube, deg_train_valid = simulate_query_degradation(train_cube, tr_df['label'].values, rng)
    
    # Note: deg_train_cube is functionally a "test set" from a new domain due to jitter.
    # We must Q-normalize it separately!
    q_deg_train = build_q_cube(deg_train_cube, deg_train_valid)
    
    # We use 5-fold CV to match against out-of-fold candidates
    from sklearn.model_selection import StratifiedGroupKFold
    groups = tr_df.index // 25
    y = tr_df['label'].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
    
    v1_dists = []
    v2_dists = []
    acc_matches = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(train_cube, y, groups)):
        cand_cube = train_cube[train_idx]
        cand_q = q_train[train_idx]
        cand_val = train_valid[train_idx]
        cand_y = y[train_idx]
        
        for v_i in val_idx:
            best_c, d1, d2 = vectorized_match_query(
                deg_train_cube[v_i], q_deg_train[v_i], deg_train_valid[v_i],
                cand_cube, cand_q, cand_val
            )
            if best_c is not None:
                v1_dists.append(d1)
                v2_dists.append(d2)
                acc_matches.append(cand_y[best_c] == y[v_i])
            else:
                v1_dists.append(np.inf)
                v2_dists.append(np.inf)
                acc_matches.append(False)
                
    v1_dists = np.array(v1_dists)
    v2_dists = np.array(v2_dists)
    acc_matches = np.array(acc_matches)
    
    print("Validation Results:")
    print(f"Total evaluated: {len(v1_dists)}")
    print(f"Total matched (any): {(v1_dists < np.inf).sum()}")
    
    # Find thresholds for 98% and 95% accuracy
    # We require both v1 and v2 to be below threshold (dual-view agreement)
    for pct in [5, 10, 20, 30, 40, 50, 60, 70]:
        t1 = np.percentile(v1_dists[v1_dists < np.inf], pct)
        t2 = np.percentile(v2_dists[v2_dists < np.inf], pct)
        mask = (v1_dists <= t1) & (v2_dists <= t2)
        if mask.sum() > 0:
            acc = acc_matches[mask].mean()
            print(f"Top {pct}% (v1<={t1:.4f}, v2<={t2:.4f}) -> Acc: {acc:.4f} (N={mask.sum()})")
            
    print("\n--- Running on Test Set ---")
    te_v1_dists = []
    te_v2_dists = []
    te_nn_idx = []
    
    for i in range(len(test_cube)):
        best_c, d1, d2 = vectorized_match_query(
            test_cube[i], q_test[i], test_valid[i],
            train_cube, q_train, train_valid
        )
        te_v1_dists.append(d1)
        te_v2_dists.append(d2)
        te_nn_idx.append(best_c)
        
    te_v1_dists = np.array(te_v1_dists, dtype=float)
    te_v2_dists = np.array(te_v2_dists, dtype=float)
    
    np.save('scratch/twins_test_v1_dists.npy', te_v1_dists)
    np.save('scratch/twins_test_v2_dists.npy', te_v2_dists)
    np.save('scratch/twins_test_nn_idx.npy', te_nn_idx)
    
    print("Test distances saved.")
    
    # Print Test percentiles
    print("\nTest Distance Percentiles (v1):")
    valid_v1 = te_v1_dists[te_v1_dists < np.inf]
    for pct in [1, 5, 10, 25, 50]:
        print(f"  {pct}%: {np.percentile(valid_v1, pct):.4f}")

if __name__ == '__main__':
    main()
