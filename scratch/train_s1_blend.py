"""
train_s1_blend.py — S1-Only Decorrelation Member + Rank-Blend into S6 Block E

Design:
  - Trains a model on ONLY S1 (VH, VV) features + their temporal stats + SAR ratio.
  - S1 is never cloud-masked → this model has uncorrelated errors on the 273
    cloud-affected rows where S6's S2-heavy features are weakest.
  - Same seed-bagging (3 seeds) and predict-time TTA as Block E (8 views).
  - For TTA on S1-only model, TTA does NOT apply S2 dropout (S1 is never masked),
    so all 8 views are identical → TTA degenerates to a single predict (free, no noise).
  - Rank-blend weight sweep [0.0, 0.05, 0.10, ..., 0.40] of S1-OOF into S6-OOF.
  - Gate 1: DPP AUC improvement > 0 (must not regress on test-like rows).
  - Gate 2: Slab holdout AUC must not regress vs. plain S6 Block E.
  - If both gates pass: write final blended submission.
  - If either gate fails: report why, do not write submission.
"""
import sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

sys.path.append('src')
import common as C
from config import RunConfig
from train_model import make_model, _score

# ─── S1-ONLY FEATURE EXTRACTOR ───────────────────────────────────────────────
S1_BAND_IDX = [C.ALL_BANDS.index(b) for b in C.S1_BANDS]  # [0, 1] for VH, VV

def build_s1_features(cube: np.ndarray) -> pd.DataFrame:
    """
    Extract features using ONLY S1 bands (VH, VV) — structurally immune to S2 masking.
    
    Features per band:
      - mean, median, std, min, max, range, slope, first, last (absolute anchors)
      - z_min, z_max, z_slope (drift-immune z-shape)
      - rank_first, rank_last, sign_changes (nonparametric rank)
    Plus the SAR ratio (VH - VV in dB, affine-invariant).
    Plus S1-specific valid month count and window_length.
    """
    valid = C._valid_mask(cube)  # (N, 12, 12)
    N = cube.shape[0]
    feats: dict[str, np.ndarray] = {}
    
    # Per S1 band stats
    for k_idx, b in zip(S1_BAND_IDX, C.S1_BANDS):
        vals = cube[:, :, k_idx]
        vmask = valid[:, :, k_idx]
        s = C._masked_stats(vals, vmask)
        
        # Absolute anchors
        feats[f"{b}_mean"]   = s["mean"]
        feats[f"{b}_median"] = s["median"]
        feats[f"{b}_min"]    = s["min"]
        feats[f"{b}_max"]    = s["max"]
        feats[f"{b}_first"]  = s["first"]
        feats[f"{b}_last"]   = s["last"]
        feats[f"{b}_std"]    = s["std"]
        feats[f"{b}_range"]  = s["max"] - s["min"]
        feats[f"{b}_slope"]  = s["slope"]
        
        # Z-shape (drift-immune)
        with np.errstate(invalid="ignore"):
            global_std_reg = np.nanmedian(s["std"]) * 0.1
            reg_std = s["std"] + global_std_reg + 1e-6
            feats[f"{b}_z_min"]   = (s["min"] - s["mean"]) / reg_std
            feats[f"{b}_z_max"]   = (s["max"] - s["mean"]) / reg_std
            feats[f"{b}_z_first"] = (s["first"] - s["mean"]) / reg_std
            feats[f"{b}_z_last"]  = (s["last"] - s["mean"]) / reg_std
            feats[f"{b}_z_slope"] = s["slope"] / reg_std
        
        # Rank features (nonparametric)
        p_first = np.full(N, np.nan)
        p_last  = np.full(N, np.nan)
        sign_changes = np.full(N, np.nan)
        for i in range(N):
            v = vals[i][vmask[i]]
            if v.size > 0:
                ranks = np.argsort(np.argsort(v)) / max(1, len(v) - 1)
                p_first[i] = ranks[0]
                p_last[i]  = ranks[-1]
                sign_changes[i] = np.sum(
                    np.diff(np.sign(v - s["median"][i])) != 0
                ) if v.size > 1 else 0
        feats[f"{b}_rank_first"]   = p_first
        feats[f"{b}_rank_last"]    = p_last
        feats[f"{b}_sign_changes"] = sign_changes
        
        # First-difference dynamics (harvest-cycle: drain-refill)
        max_abs_diff = np.full(N, np.nan)
        diff_skew    = np.full(N, np.nan)
        drain_refill = np.full(N, np.nan)
        from scipy.stats import skew as _skew
        for i in range(N):
            v = vals[i][vmask[i]]
            if v.size >= 2:
                d = np.diff(v)
                abs_d = np.abs(d)
                max_abs_diff[i] = abs_d.max()
                if v.size >= 3 and np.std(d) > 1e-9:
                    diff_skew[i] = float(_skew(d))
                drain_refill[i] = d.max() / (np.abs(d.min()) + 1e-9)
        feats[f"{b}_max_abs_diff"] = max_abs_diff
        feats[f"{b}_diff_skew"]    = diff_skew
        feats[f"{b}_drain_refill"] = drain_refill
    
    # SAR ratio VH - VV (dB difference = log-domain ratio, affine-drift-immune)
    vh = cube[:, :, S1_BAND_IDX[0]]; vv = cube[:, :, S1_BAND_IDX[1]]
    vh_v = valid[:, :, S1_BAND_IDX[0]]; vv_v = valid[:, :, S1_BAND_IDX[1]]
    sar_ratio = vh - vv
    sar_valid  = vh_v & vv_v
    sr_stats = C._masked_stats(sar_ratio, sar_valid)
    feats["sar_ratio_mean"]   = sr_stats["mean"]
    feats["sar_ratio_median"] = sr_stats["median"]
    feats["sar_ratio_std"]    = sr_stats["std"]
    feats["sar_ratio_min"]    = sr_stats["min"]
    feats["sar_ratio_max"]    = sr_stats["max"]
    feats["sar_ratio_slope"]  = sr_stats["slope"]
    
    # Missingness features
    s1_month_valid = valid[:, :, S1_BAND_IDX].all(axis=2)   # (N,12): months where BOTH S1 bands valid
    feats["window_length"]     = s1_month_valid.sum(axis=1).astype(float)
    feats["n_valid_months_s1"] = s1_month_valid.sum(axis=1).astype(float)
    
    return pd.DataFrame(feats)


def find_optimal_t(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    def obj(t):
        p_bin = (y_prob >= t[0]).astype(int)
        f1 = f1_score(y_true, p_bin, zero_division=0)
        return -(0.6 * f1 + 0.4 * auc)
    res = minimize(obj, [0.5], bounds=[(0.01, 0.99)], method='Nelder-Mead')
    t_star = res.x[0]
    best_f1 = f1_score(y_true, (y_prob >= t_star).astype(int), zero_division=0)
    best_score = 0.6 * best_f1 + 0.4 * auc
    return t_star, best_score, best_f1, auc


def apply_logit_shift(preds, t_star, target_t=0.5):
    shift = logit(target_t) - logit(np.clip(t_star, 1e-7, 1 - 1e-7))
    log_odds = logit(np.clip(preds, 1e-7, 1 - 1e-7)) + shift
    return expit(log_odds)


def rank_blend(p_s6, p_s1, w_s1):
    """Rank-average: (1-w)*rank(S6) + w*rank(S1), then re-normalise to [0,1]."""
    if w_s1 == 0.0:
        return p_s6.copy()
    N = len(p_s6)
    r6 = rankdata(p_s6) / N
    r1 = rankdata(p_s1) / N
    blended = (1 - w_s1) * r6 + w_s1 * r1
    return blended / blended.max()  # keep [0, 1]


def main():
    t_total = time.time()
    print("=" * 60)
    print("S1-Only Decorrelation Blend")
    print("=" * 60)
    
    # ─── Load ───
    cfg = RunConfig.load('config_s6.json')
    tr_df, te_df = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    cube_tr = C.band_cube(tr_df)
    y_tr    = tr_df['label'].values
    cube_te = C.band_cube(te_df)
    
    # ─── DPP and slab indices ───
    dpp_idx = np.load('data_checkpoint/dpp_test_like_idx.npy')
    # Slab holdout: use the LAST fold (fold 4) as an out-of-time slab
    # We'll identify it from the CV split below and store it for gate checking
    
    # ─── Load S6 Block E OOF ───
    oof_s6_df = pd.read_csv('data_checkpoint/oof_s6_block_e.csv')
    oof_s6 = oof_s6_df['pred'].values
    y_s6   = oof_s6_df['Target'].values  # should match y_tr order
    assert len(oof_s6) == len(y_tr), "OOF length mismatch"
    assert np.allclose(y_s6, y_tr), "OOF target mismatch"
    
    # ─── Cross-Validation Setup ───
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    group_id = np.arange(len(y_tr)) // cfg.cv_block_size
    rng = np.random.default_rng(C.SEED)
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    
    kinds  = ["lgbm", "xgb", "hgb", "cat"]
    seeds  = [42, 43, 44]
    
    s1_oof_preds = np.zeros(len(y_tr))
    all_s1_models = []   # (model, fold_idx)
    slab_idx = None      # last fold's val indices
    
    print("\n--- Training S1-Only Models ---")
    for fold_idx, (tr_i, va_i) in enumerate(skf.split(cube_tr, y_tr, groups=group_id)):
        print(f"  Fold {fold_idx+1}/5")
        if fold_idx == 4:
            slab_idx = va_i  # save last fold as slab holdout
        
        # Augment training
        aug_cube, aug_y = C.augment_training_data(
            cube_tr[tr_i], y_tr[tr_i], n_copies=cfg.n_augmentation_copies, seed=C.SEED + fold_idx
        )
        Xtr = build_s1_features(aug_cube)
        
        # Train with seed-bagging
        fold_models = []
        for s in seeds:
            for k in kinds:
                m = make_model(k, s, pos_w)
                m.fit(Xtr, aug_y)
                fold_models.append(m)
                all_s1_models.append((m, fold_idx))
        
        # Predict on validation (S1 is never cloud-masked, so TTA is identity)
        # Apply simulate_test_masking to be consistent with the S6 Block E validation path,
        # but the S1 features are extracted from S1 bands only — the S2 dropout doesn't affect them.
        base_va_cube = C.simulate_test_masking(cube_tr[va_i], rng)
        Xva = build_s1_features(base_va_cube)
        
        # Average logits over all fold models
        fold_logits = []
        for m in fold_models:
            p = m.predict_proba(Xva)[:, 1]
            fold_logits.append(logit(np.clip(p, 1e-7, 1-1e-7)))
        s1_oof_preds[va_i] = expit(np.mean(fold_logits, axis=0))
    
    print(f"\nS1-Only OOF AUC: {roc_auc_score(y_tr, s1_oof_preds):.4f}")
    f1_s1, auc_s1, wsc_s1 = _score(y_tr, s1_oof_preds)
    print(f"S1-Only OOF: F1={f1_s1:.4f} AUC={auc_s1:.4f} Score={wsc_s1:.4f}")
    
    # ─── DPP Scores ───
    s6_dpp_auc = roc_auc_score(y_tr[dpp_idx], oof_s6[dpp_idx])
    s1_dpp_auc = roc_auc_score(y_tr[dpp_idx], s1_oof_preds[dpp_idx])
    print(f"\nDPP AUC — S6 Block E: {s6_dpp_auc:.4f} | S1-Only: {s1_dpp_auc:.4f}")
    
    # ─── Slab Holdout Scores ───
    s6_slab_auc = roc_auc_score(y_tr[slab_idx], oof_s6[slab_idx])
    s1_slab_auc = roc_auc_score(y_tr[slab_idx], s1_oof_preds[slab_idx])
    print(f"Slab AUC — S6 Block E: {s6_slab_auc:.4f} | S1-Only: {s1_slab_auc:.4f}")
    
    # ─── Blend Weight Sweep ───
    print("\n--- Blend Weight Sweep (rank-blend S1 into S6) ---")
    weights = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    results = []
    for w in weights:
        blended_oof = rank_blend(oof_s6, s1_oof_preds, w)
        overall_auc = roc_auc_score(y_tr, blended_oof)
        dpp_auc     = roc_auc_score(y_tr[dpp_idx], blended_oof[dpp_idx])
        slab_auc    = roc_auc_score(y_tr[slab_idx], blended_oof[slab_idx])
        results.append({
            "w_s1": w, "overall_auc": overall_auc,
            "dpp_auc": dpp_auc, "slab_auc": slab_auc
        })
        print(f"  w={w:.2f} | Overall={overall_auc:.4f} | DPP={dpp_auc:.4f} | Slab={slab_auc:.4f}")
    
    # ─── Select Best Weight by DPP AUC ───
    # Gate: DPP must improve, slab must not drop vs. w=0.0 baseline
    baseline_dpp  = results[0]["dpp_auc"]
    baseline_slab = results[0]["slab_auc"]
    
    best_w = 0.0
    best_dpp = baseline_dpp
    gate_passed = False
    
    for r in results[1:]:  # skip w=0.0
        dpp_delta  = r["dpp_auc"]  - baseline_dpp
        slab_delta = r["slab_auc"] - baseline_slab
        if dpp_delta > 0 and slab_delta >= -0.001 and r["dpp_auc"] > best_dpp:
            best_dpp = r["dpp_auc"]
            best_w = r["w_s1"]
            gate_passed = True
    
    print(f"\n--- Gate Assessment ---")
    print(f"Baseline (w=0): DPP={baseline_dpp:.4f}, Slab={baseline_slab:.4f}")
    print(f"Best w_s1={best_w:.2f}: DPP={best_dpp:.4f} (delta={best_dpp - baseline_dpp:+.4f})")
    
    if not gate_passed:
        print("\n❌ GATE FAILED: S1-only member did not improve DPP without slab regression.")
        print("   No new submission generated. S6 Block E remains the champion.")
        return
    
    print(f"\n✅ GATE PASSED: Blend weight w_s1={best_w:.2f}")
    
    # ─── Final Blended OOF: report metrics only (no logit shift needed for test) ───
    blended_oof = rank_blend(oof_s6, s1_oof_preds, best_w)
    t_star, best_score, best_f1, best_auc = find_optimal_t(y_tr, blended_oof)
    print(f"OOF t*={t_star:.4f} | F1={best_f1:.4f} | AUC={best_auc:.4f} | Score={best_score:.4f}")
    print(f"NOTE: Logit shift NOT applied to test set (rank-space calibration doesn't transfer).")
    print(f"      TargetF1 on test will be thresholded at (1 - pi_t) percentile of test RAUC.")
    
    # ─── Generate Test Predictions ───
    print("\n--- Generating Test Predictions ---")
    
    # S6 Block E test preds: undo the +0.0500 logit shift to recover raw ordering
    s6_sub = pd.read_csv('submissions/s6_block_e_submission.csv')
    s6_block_e_shift = 0.0500  # recorded from train_s6_block_e.py output
    s6_te_rauc = s6_sub['TargetRAUC'].values
    s6_te_logits_raw = logit(np.clip(s6_te_rauc, 1e-7, 1-1e-7)) - s6_block_e_shift
    s6_te_preds_raw = expit(s6_te_logits_raw)
    
    # S1-only test predictions (compute once, average logits over all models)
    Xte = build_s1_features(cube_te)  # S1 features don't change across TTA
    s1_te_logits = []
    for m, _ in all_s1_models:
        p = m.predict_proba(Xte)[:, 1]
        s1_te_logits.append(logit(np.clip(p, 1e-7, 1-1e-7)))
    s1_te_preds = expit(np.mean(s1_te_logits, axis=0))
    
    # Rank-blend test predictions
    blended_te = rank_blend(s6_te_preds_raw, s1_te_preds, best_w)
    
    # Break any rank ties with epsilon noise (preserves ordering, makes all unique)
    rng_tie = np.random.default_rng(seed=7777)
    blended_te = blended_te + rng_tie.uniform(-1e-9, 1e-9, size=len(blended_te))
    blended_te = np.clip(blended_te, 0.0, 1.0)
    
    # TargetF1: threshold at (1 - pi_t) percentile of test RAUC.
    # This classifies exactly round(pi_t * N) rows as positive — correct for any rank-space.
    # Do NOT use OOF t_star (it's in OOF rank-space, not test rank-space).
    pi_t = 0.6175
    f1_threshold = np.percentile(blended_te, (1 - pi_t) * 100)
    target_f1 = (blended_te >= f1_threshold).astype(int)
    
    # ─── Pre-flight Checks ───
    sub = pd.DataFrame({
        'ID': te_df['ID'],
        'TargetRAUC': blended_te,
        'TargetF1': target_f1
    })
    
    s4_sub = pd.read_csv('submissions/S4_rank_ensemble.csv')
    
    pos_rate = sub['TargetF1'].mean()
    pos_count_blend = sub['TargetF1'].sum()
    pos_count_s4 = s4_sub['TargetF1'].sum()
    spearman = sub['TargetRAUC'].corr(s4_sub['TargetRAUC'], method='spearman')
    unique_rauc = len(sub['TargetRAUC'].unique())
    
    print(f"\n--- Pre-flight Assertions ---")
    print(f"Rows: {len(sub)} (expected 1030)")
    print(f"Positive rate: {pos_rate:.4f} (target ~{pi_t:.4f})")
    print(f"Positive count — Blend: {pos_count_blend} | S4: {pos_count_s4} | Diff: {pos_count_blend - pos_count_s4}")
    print(f"Spearman corr with S4: {spearman:.4f}")
    print(f"Unique RAUC values: {unique_rauc} (should equal {len(sub)})")
    
    assert len(sub) == 1030
    assert unique_rauc == len(sub), f"Still have ties: {unique_rauc} unique vs {len(sub)} rows"
    assert 0.58 <= pos_rate <= 0.65, f"Positive rate {pos_rate:.4f} out of expected range"
    
    # ─── Save ───
    sub_path = f'submissions/s6_s1_blend_w{int(best_w*100):02d}.csv'
    sub.to_csv(sub_path, index=False)
    print(f"\n✅ Saved: {sub_path}")
    print(f"Total time: {time.time() - t_total:.1f}s")


if __name__ == '__main__':
    main()
