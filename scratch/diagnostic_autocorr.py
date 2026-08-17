import pandas as pd
import numpy as np

def compute_autocorrelation(series, max_lag=25):
    mean = np.mean(series)
    var = np.var(series)
    n = len(series)
    autocorr = {}
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        cov = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / n
        autocorr[lag] = cov / var
    return autocorr

def main():
    df = pd.read_csv("submissions/S4_rank_ensemble.csv")
    scores = df['TargetRAUC'].values
    
    print(f"Loaded {len(scores)} test scores.")
    
    # 1. Autocorrelation
    ac = compute_autocorrelation(scores, max_lag=25)
    print("\n--- Autocorrelation (Lag 1-25) ---")
    for lag in range(1, 6):
        print(f"Lag {lag:2d}: {ac.get(lag, 0):.4f}")
    print("...")
    for lag in range(20, 26):
        print(f"Lag {lag:2d}: {ac.get(lag, 0):.4f}")
        
    # 2. Block Variance
    block_size = 25
    blocks = [scores[i:i+block_size] for i in range(0, len(scores), block_size)]
    
    within_vars = [np.var(b) for b in blocks if len(b) == block_size]
    mean_within_var = np.mean(within_vars)
    
    block_means = [np.mean(b) for b in blocks if len(b) == block_size]
    across_var = np.var(block_means)
    
    global_var = np.var(scores)
    
    print("\n--- Block Structure (size=25) ---")
    print(f"Global Variance:        {global_var:.4f}")
    print(f"Mean Within-Block Var:  {mean_within_var:.4f}")
    print(f"Variance Across Blocks: {across_var:.4f}")
    
if __name__ == "__main__":
    main()
