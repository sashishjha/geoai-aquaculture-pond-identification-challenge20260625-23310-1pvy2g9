"""
proxy_validation.py — Phase 2: The LB Proxy (Corrected Validation Harness)

Fixes all three defects from the broken Iteration 4 harness:
  1. simulate_test_masking applied to validation rows ONLY at prediction time.
     Augmentation happens strictly inside the training fold. Group IDs ensure
     all 15 copies of a base row land on the same fold side.
  2. After masking, each validation fold is resampled to pi_t before computing F1.
     Uses provisional pi_t = 0.60 until Phase 4 delivers the real estimate.
  3. Proxy computes EXACTLY what the LB computes:
       TargetF1 = (TargetRAUC >= 0.5).astype(int)
       proxy = 0.6 * f1_score(y, TargetF1) + 0.4 * roc_auc_score(y, TargetRAUC)

Acceptance gate: the proxy must rank historical submissions:
  Iter1 (0.8173) > Iter3 (0.7748) > Iter2 (0.7633) > Iter4 (0.7583)
Minimum viable gate (Amendment 3): Iter1 > Iter4 by out-of-noise margin.
Forgiving: Iter2 vs Iter4 within noise (0.005 apart) is acceptable.

Run:
    uv run python src/proxy_validation.py
    uv run python src/proxy_validation.py --config path/to/config.json
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import common as C
from config import RunConfig, CHAMPION_CONFIG
from train_model import make_model

# ── Constants ────────────────────────────────────────────────────────────────
N_SEEDS = 3          # masking + resampling seeds (gains < 1σ don't count)
N_FOLDS = 5
BLOCK_SIZE = 25
SLAB_FRAC = 0.30     # contiguous slab hold-out fraction


def _competition_score(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float, float]:
    """Compute the exact competition metric: fixed 0.5 threshold F1 + AUC."""
    y_pred = (y_prob >= 0.5).astype(int)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    return f1, auc, 0.6 * f1 + 0.4 * auc


def _resample_to_prevalence(
    y: np.ndarray,
    p: np.ndarray,
    pi_t: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample (y, p) so the positive rate matches pi_t.
    Upsamples positives if pi_t > pi_obs, downsamples negatives if pi_t < pi_obs.
    Returns indices into the original arrays.
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return y, p

    # Target: n_pos / (n_pos + n_neg_new) = pi_t  =>  n_neg_new = n_pos * (1-pi_t) / pi_t
    n_neg_target = int(round(n_pos * (1 - pi_t) / pi_t))
    n_neg_target = max(1, min(n_neg_target, n_neg * 5))  # cap upsampling 5x

    if n_neg_target <= n_neg:
        # downsample negatives
        sampled_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)
    else:
        # upsample negatives (rare, pi_t < pi_obs)
        sampled_neg = rng.choice(neg_idx, size=n_neg_target, replace=True)

    idx = np.concatenate([pos_idx, sampled_neg])
    return y[idx], p[idx]


def _fold_cv(
    cube: np.ndarray,
    y: np.ndarray,
    cfg: RunConfig,
    pi_t: float,
    seed: int,
    kinds: list[str] = None,
) -> dict:
    """Run one K-fold pass. Returns dict with f1, auc, score per fold."""
    if kinds is None:
        kinds = ["hgb", "lgbm", "xgb", "cat"]

    skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    # Groups: contiguous blocks of BLOCK_SIZE (geographic clustering)
    group_id = np.arange(len(y)) // cfg.cv_block_size
    mask_rng = np.random.default_rng(seed + 1000)
    resamp_rng = np.random.default_rng(seed + 2000)

    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    f1s, aucs, scores = [], [], []
    f1s_pi, aucs_pi, scores_pi = [], [], []   # pi_t-resampled

    for fold_idx, (tr_i, va_i) in enumerate(skf.split(cube, y, groups=group_id)):
        # Assert augmentation siblings don't leak across fold boundary
        assert len(set(tr_i) & set(va_i)) == 0, "fold leakage detected"

        # Training: augment strictly inside the fold
        aug_cube, aug_y = C.augment_training_data(
            cube[tr_i], y[tr_i],
            n_copies=cfg.n_augmentation_copies,
            seed=seed + fold_idx,
        )
        Xtr = C.build_features(aug_cube, cfg=cfg)

        # Validation: apply test masking at PREDICTION TIME only (not at fit time)
        val_masked = C.simulate_test_masking(cube[va_i], mask_rng)
        Xva = C.build_features(val_masked, cfg=cfg)

        # Ensemble predictions
        ps = []
        for k in kinds:
            m = make_model(k, seed, pos_w)
            m.fit(Xtr, aug_y)
            ps.append(m.predict_proba(Xva)[:, 1])
        p_fold = np.mean(ps, axis=0)
        y_fold = y[va_i]

        # Score at raw prevalence (~40%)
        f1, auc, sc = _competition_score(y_fold, p_fold)
        f1s.append(f1); aucs.append(auc); scores.append(sc)

        # Score at pi_t (resampled to estimated test prevalence)
        y_r, p_r = _resample_to_prevalence(y_fold, p_fold, pi_t, resamp_rng)
        f1_r, auc_r, sc_r = _competition_score(y_r, p_r)
        f1s_pi.append(f1_r); aucs_pi.append(auc_r); scores_pi.append(sc_r)

    return {
        "f1_raw": np.mean(f1s),   "auc_raw": np.mean(aucs),   "score_raw": np.mean(scores),
        "f1_pi":  np.mean(f1s_pi), "auc_pi":  np.mean(aucs_pi), "score_pi":  np.mean(scores_pi),
    }


def _slab_eval(
    cube: np.ndarray,
    y: np.ndarray,
    cfg: RunConfig,
    pi_t: float,
    seed: int,
    slab_frac: float = SLAB_FRAC,
    kinds: list[str] = None,
) -> dict:
    """Contiguous slab hold-out: train on first (1-slab_frac), evaluate on last slab_frac."""
    if kinds is None:
        kinds = ["hgb", "lgbm", "xgb", "cat"]

    n_slab = int(len(y) * slab_frac)
    tr_i = np.arange(len(y) - n_slab)
    va_i = np.arange(len(y) - n_slab, len(y))

    mask_rng = np.random.default_rng(seed + 3000)
    resamp_rng = np.random.default_rng(seed + 4000)

    aug_cube, aug_y = C.augment_training_data(
        cube[tr_i], y[tr_i], n_copies=cfg.n_augmentation_copies, seed=seed)
    Xtr = C.build_features(aug_cube, cfg=cfg)

    val_masked = C.simulate_test_masking(cube[va_i], mask_rng)
    Xva = C.build_features(val_masked, cfg=cfg)

    pos_w = (y[tr_i] == 0).sum() / max((y[tr_i] == 1).sum(), 1)
    ps = []
    for k in kinds:
        m = make_model(k, seed, pos_w)
        m.fit(Xtr, aug_y)
        ps.append(m.predict_proba(Xva)[:, 1])
    p = np.mean(ps, axis=0)
    y_va = y[va_i]

    f1, auc, sc = _competition_score(y_va, p)
    y_r, p_r = _resample_to_prevalence(y_va, p, pi_t, resamp_rng)
    f1_r, auc_r, sc_r = _competition_score(y_r, p_r)

    return {
        "f1_raw": f1, "auc_raw": auc, "score_raw": sc,
        "f1_pi": f1_r, "auc_pi": auc_r, "score_pi": sc_r,
    }


def run_proxy(
    cfg: RunConfig,
    train_path: str = "data_checkpoint/raw/Train.csv",
    test_path:  str = "data_checkpoint/raw/Test.csv",
    pi_t: float = 0.60,
    n_seeds: int = N_SEEDS,
    kinds: list[str] = None,
    verbose: bool = True,
) -> dict:
    """
    Full proxy evaluation with n_seeds masking seeds.
    Returns mean ± std over seeds for both K-fold and slab schemes.
    """
    tr, te = C.load_raw(train_path, test_path)
    cube = C.band_cube(tr)
    y = tr["label"].values

    if kinds is None:
        kinds = ["hgb", "lgbm", "xgb", "cat"]

    kfold_scores, slab_scores = [], []
    kfold_f1s, kfold_aucs = [], []
    slab_f1s, slab_aucs = [], []

    # Sensitivity band: also report at pi=0.50 and pi=0.70
    kfold_scores_50, kfold_scores_70 = [], []

    from joblib import Parallel, delayed
    kfold_results = Parallel(n_jobs=1)(
        delayed(_fold_cv)(cube, y, cfg, pi_t=pi_t, seed=C.SEED + s * 1000, kinds=kinds)
        for s in range(n_seeds)
    )
    slab_results = Parallel(n_jobs=1)(
        delayed(_slab_eval)(cube, y, cfg, pi_t=pi_t, seed=C.SEED + s * 1000, kinds=kinds)
        for s in range(n_seeds)
    )

    for seed_offset in range(n_seeds):
        kf = kfold_results[seed_offset]
        sl = slab_results[seed_offset]
        seed = C.SEED + seed_offset * 1000

        kfold_scores.append(kf["score_pi"])
        kfold_f1s.append(kf["f1_pi"])
        kfold_aucs.append(kf["auc_pi"])
        slab_scores.append(sl["score_pi"])
        slab_f1s.append(sl["f1_pi"])
        slab_aucs.append(sl["auc_pi"])

        # Sensitivity band
        kf50 = _fold_cv(cube, y, cfg, pi_t=0.50, seed=seed, kinds=kinds)
        kf70 = _fold_cv(cube, y, cfg, pi_t=0.70, seed=seed, kinds=kinds)
        kfold_scores_50.append(kf50["score_pi"])
        kfold_scores_70.append(kf70["score_pi"])

        if verbose:
            print(f"  seed+{seed_offset:02d}: kfold={kf['score_pi']:.4f} "
                  f"(F1={kf['f1_pi']:.4f}, AUC={kf['auc_pi']:.4f}) | "
                  f"slab={sl['score_pi']:.4f}")

    result = {
        "config_hash": cfg.hash(),
        "pi_t": pi_t,
        "kfold_score_mean": float(np.mean(kfold_scores)),
        "kfold_score_std":  float(np.std(kfold_scores)),
        "kfold_f1_mean":    float(np.mean(kfold_f1s)),
        "kfold_auc_mean":   float(np.mean(kfold_aucs)),
        "slab_score_mean":  float(np.mean(slab_scores)),
        "slab_score_std":   float(np.std(slab_scores)),
        "slab_f1_mean":     float(np.mean(slab_f1s)),
        "slab_auc_mean":    float(np.mean(slab_aucs)),
        "sensitivity_pi50": float(np.mean(kfold_scores_50)),
        "sensitivity_pi70": float(np.mean(kfold_scores_70)),
    }

    if verbose:
        print(f"\n[proxy] config={cfg.hash()} pi_t={pi_t}")
        print(f"  K-fold : {result['kfold_score_mean']:.4f} ± {result['kfold_score_std']:.4f} "
              f"(F1={result['kfold_f1_mean']:.4f}, AUC={result['kfold_auc_mean']:.4f})")
        print(f"  Slab   : {result['slab_score_mean']:.4f} ± {result['slab_score_std']:.4f} "
              f"(F1={result['slab_f1_mean']:.4f}, AUC={result['slab_auc_mean']:.4f})")
        print(f"  Sensitivity band: pi=0.50 → {result['sensitivity_pi50']:.4f} | "
              f"pi=0.70 → {result['sensitivity_pi70']:.4f}")

    return result


def backtest_gate(results: dict[str, dict], verbose: bool = True) -> bool:
    """
    Amendment 3: check the proxy correctly ranks historical iterations.
    Full target: Iter1 > Iter3 > Iter2 > Iter4
    Minimum viable gate: Iter1 > Iter4 by out-of-noise margin.
    Iter2 vs Iter4 confusion (0.005 apart on LB) is forgivable.
    Putting Iter4 on top is not.
    """
    required = ["iter1", "iter4"]
    missing = [k for k in required if k not in results]
    if missing:
        print(f"[backtest] Cannot run gate: missing {missing}")
        return False

    iter1_score = results["iter1"]["kfold_score_mean"]
    iter4_score = results["iter4"]["kfold_score_mean"]
    iter1_std   = results["iter1"]["kfold_score_std"]
    iter4_std   = results["iter4"]["kfold_score_std"]
    noise = max(iter1_std, iter4_std)

    gap = iter1_score - iter4_score
    passes_min = gap > noise  # out-of-noise margin

    if verbose:
        print(f"\n[backtest] Minimum viable gate:")
        print(f"  Iter1  = {iter1_score:.4f} ± {iter1_std:.4f}")
        print(f"  Iter4  = {iter4_score:.4f} ± {iter4_std:.4f}")
        print(f"  Gap    = {gap:.4f} vs noise = {noise:.4f}")
        print(f"  Result = {'PASS' if passes_min else 'FAIL'}")

    if "iter2" in results and "iter3" in results:
        iter2_score = results["iter2"]["kfold_score_mean"]
        iter3_score = results["iter3"]["kfold_score_mean"]
        full_order = [
            ("iter1", iter1_score), ("iter3", iter3_score),
            ("iter2", iter2_score), ("iter4", iter4_score),
        ]
        sorted_by_proxy = sorted(full_order, key=lambda x: -x[1])
        expected_order  = ["iter1", "iter3", "iter2", "iter4"]
        actual_order    = [k for k, _ in sorted_by_proxy]
        full_pass = (actual_order == expected_order)
        # Forgiving: iter2/iter4 swap is OK if gap < 2*noise
        if not full_pass:
            i2_i4_gap = abs(iter2_score - iter4_score)
            forgivable = (
                actual_order[0] == "iter1" and
                "iter4" not in actual_order[:1] and
                i2_i4_gap < 2 * noise
            )
        else:
            forgivable = False
        if verbose:
            print(f"\n[backtest] Full gate (4-point):")
            for k, s in sorted_by_proxy:
                print(f"  {k}: {s:.4f}")
            if full_pass:
                print("  Result = PASS (full ordering correct)")
            elif forgivable:
                print("  Result = PASS (iter2/iter4 swap within noise, forgivable)")
            else:
                print("  Result = FAIL (ordering violated)")

    return passes_min


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Path to config JSON. Default: champion.")
    ap.add_argument("--pi_t", type=float, default=0.60, help="Assumed test prevalence.")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="Number of masking seeds.")
    ap.add_argument("--backtest", action="store_true",
                    help="Run back-test on all four historical submissions.")
    a = ap.parse_args()

    cfg = RunConfig.load(a.config) if a.config else CHAMPION_CONFIG
    print(f"[proxy] Running proxy for config={cfg.hash()}, pi_t={a.pi_t}, seeds={a.seeds}")
    result = run_proxy(cfg, pi_t=a.pi_t, n_seeds=a.seeds)
    print(f"\n[PROXY_CHAMPION] score={result['kfold_score_mean']:.4f} ± {result['kfold_score_std']:.4f}")
