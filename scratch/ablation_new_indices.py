"""
ablation_new_indices.py — AWEI-nsh / NDRE / CIG ablation

Three new spectral indices computed from existing S2 bands:
  AWEI-nsh = 4*(green - swir1) - (0.25*nir + 2.75*swir2)
  NDRE     = (nir - re1) / (nir + re1)
  CIG      = (nir / green) - 1

CRITICAL: All three are S2 (optical, drifted, cloud-masked) features.
  - We compute ONLY drift-robust representations: z-shape + rank (NO absolute anchors)
  - Adversarial importance gate: drop any new feature that lands in top-half
    of train-vs-test discriminator importance (same protocol that vetted phenology block)
  - DPP gate: delta > 0.0059 (actual 1-sigma on 455-row DPP subset)
  - Slab gate: no >0.001 drop vs. S6 Block E baseline

Gate pass → write submission. Gate fail → report and stop.
"""
import sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

sys.path.append('src')
import common as C
from config import RunConfig
from train_model import make_model, _score

DPP_SIGMA = 0.00590   # Hanley-McNeil SE on 455-row DPP, computed in prior step

# ─── New index computation ────────────────────────────────────────────────────
def compute_new_indices(cube: np.ndarray) -> dict:
    """
    Compute AWEI-nsh, NDRE, CIG time series from the full (N,12,12) cube.
    Returns dict of (series, valid_mask) arrays, each (N,12).
    All S2 bands: blue(2), green(1), nir(2)=idx8, nira=9, re1=4, re2=5, re3=6,
    red=7, swir1=10, swir2=11. Use C.ALL_BANDS index.
    """
    idxmap = {b: C.ALL_BANDS.index(b) for b in C.ALL_BANDS}
    valid = C._valid_mask(cube)

    def band(b):
        return cube[:, :, idxmap[b]]
    def bv(b):
        return valid[:, :, idxmap[b]]

    indices = {}

    # AWEI-nsh: no-shadow automated water extraction index
    # = 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2)
    green = band("green"); swir1 = band("swir1")
    nir   = band("nir");   swir2 = band("swir2")
    gv = bv("green"); sv1 = bv("swir1"); nv = bv("nir"); sv2 = bv("swir2")
    m_awei = gv & sv1 & nv & sv2
    with np.errstate(invalid="ignore"):
        s_awei = 4.0 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    s_awei = np.where(m_awei, s_awei, np.nan)
    indices["awei_nsh"] = (s_awei, m_awei)

    # NDRE: Normalized Difference Red Edge
    # = (NIR - re1) / (NIR + re1)
    re1 = band("re1"); rv1 = bv("re1")
    m_ndre = nv & rv1
    with np.errstate(divide="ignore", invalid="ignore"):
        s_ndre = (nir - re1) / (nir + re1 + 1e-9)
    s_ndre = np.where(np.isfinite(s_ndre) & m_ndre, s_ndre, np.nan)
    indices["ndre"] = (s_ndre, m_ndre)

    # CIG: Chlorophyll Index Green
    # = (NIR / Green) - 1
    m_cig = nv & gv
    with np.errstate(divide="ignore", invalid="ignore"):
        s_cig = (nir / (green + 1e-9)) - 1.0
    s_cig = np.where(np.isfinite(s_cig) & m_cig, s_cig, np.nan)
    indices["cig"] = (s_cig, m_cig)

    return indices


def extract_driftrobust_features(cube: np.ndarray) -> pd.DataFrame:
    """
    Extract ONLY z-shape and rank features for the three new indices.
    NO absolute anchors (mean, median, min, max, first, last) — those are S2-drifted.
    """
    indices = compute_new_indices(cube)
    N = cube.shape[0]
    feats = {}

    for name, (s, m) in indices.items():
        # s is (N, 12) with NaN for invalid months
        s_stats = C._masked_stats(s, m)

        # Z-shape (drift-immune: normalizes by own std)
        with np.errstate(invalid="ignore"):
            global_std_reg = np.nanmedian(s_stats["std"]) * 0.1
            reg_std = s_stats["std"] + global_std_reg + 1e-6
            feats[f"{name}_z_min"]   = (s_stats["min"] - s_stats["mean"]) / reg_std
            feats[f"{name}_z_max"]   = (s_stats["max"] - s_stats["mean"]) / reg_std
            feats[f"{name}_z_first"] = (s_stats["first"] - s_stats["mean"]) / reg_std
            feats[f"{name}_z_last"]  = (s_stats["last"] - s_stats["mean"]) / reg_std
            feats[f"{name}_z_slope"] = s_stats["slope"] / reg_std

        # Rank features (nonparametric, affine-invariant)
        p_first = np.full(N, np.nan)
        p_last  = np.full(N, np.nan)
        sign_changes = np.full(N, np.nan)
        for i in range(N):
            v = s[i][m[i]]
            if v.size > 0:
                ranks = np.argsort(np.argsort(v)) / max(1, len(v) - 1)
                p_first[i] = ranks[0]
                p_last[i]  = ranks[-1]
                sign_changes[i] = np.sum(
                    np.diff(np.sign(v - s_stats["median"][i])) != 0
                ) if v.size > 1 else 0
        feats[f"{name}_rank_first"]   = p_first
        feats[f"{name}_rank_last"]    = p_last
        feats[f"{name}_sign_changes"] = sign_changes

    return pd.DataFrame(feats)


def adversarial_gate(new_feats_tr: pd.DataFrame,
                     new_feats_te: pd.DataFrame) -> list[str]:
    """
    Train a LightGBM to classify train vs test rows using only the new features.
    Return the list of features whose importance is BELOW the median (safe to use).
    Features in the top half (high domain discriminability) are dropped.
    """
    print("\n--- Adversarial Importance Gate ---")
    N_tr = len(new_feats_tr)
    N_te = len(new_feats_te)

    X_av = pd.concat([new_feats_tr, new_feats_te], ignore_index=True).fillna(-1)
    y_av = np.array([0]*N_tr + [1]*N_te)

    clf = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                         random_state=42, verbose=-1)
    clf.fit(X_av, y_av)
    av_auc = roc_auc_score(y_av, clf.predict_proba(X_av)[:, 1])
    print(f"  Train-vs-test AUC on NEW features only: {av_auc:.4f}")

    importance = dict(zip(X_av.columns, clf.feature_importances_))
    median_imp = np.median(list(importance.values()))
    print(f"  Median importance: {median_imp:.1f}")

    safe_features = [f for f, imp in importance.items() if imp <= median_imp]
    dropped = [f for f, imp in importance.items() if imp > median_imp]
    print(f"  Dropped (top-half AV importance): {dropped}")
    print(f"  Kept (bottom-half, domain-safe):  {safe_features}")
    return safe_features


def find_optimal_t(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    def obj(t):
        p_bin = (y_prob >= t[0]).astype(int)
        f1 = f1_score(y_true, p_bin, zero_division=0)
        return -(0.6 * f1 + 0.4 * auc)
    res = minimize(obj, [0.5], bounds=[(0.01, 0.99)], method='Nelder-Mead')
    t_star = res.x[0]
    best_f1 = f1_score(y_true, (y_prob >= t_star).astype(int), zero_division=0)
    return t_star, 0.6 * best_f1 + 0.4 * auc, best_f1, auc


def apply_s2_dropout(cube, rng, dropout_rate=0.07):
    """Applies additional random S2 dropout to an already-masked cube."""
    out = cube.copy()
    N, M, _ = out.shape
    valid = C._valid_mask(out)
    s2_idx = [C.ALL_BANDS.index(b) for b in C.S2_BANDS]
    drop_mask = rng.random((N, M)) < dropout_rate
    for i in range(N):
        for m_idx in range(M):
            if drop_mask[i, m_idx]:
                out[i, m_idx, s2_idx] = C.SENTINEL
    return out


def main():
    t_total = time.time()
    print("=" * 60)
    print("AWEI-nsh / NDRE / CIG Ablation")
    print("=" * 60)

    # ─── Load ───
    cfg = RunConfig.load('config_s6.json')
    tr_df, te_df = C.load_raw('data_checkpoint/raw/Train.csv',
                               'data_checkpoint/raw/Test.csv')
    cube_tr = C.band_cube(tr_df)
    y_tr    = tr_df['label'].values
    cube_te = C.band_cube(te_df)

    dpp_idx = np.load('data_checkpoint/dpp_test_like_idx.npy')

    # ─── Extract new drift-robust features ───
    print("\n[1/5] Extracting drift-robust new indices...")
    new_tr = extract_driftrobust_features(cube_tr)
    new_te = extract_driftrobust_features(cube_te)
    print(f"  New features: {list(new_tr.columns)}")

    # ─── Adversarial gate ───
    safe_feats = adversarial_gate(new_tr, new_te)
    if not safe_feats:
        print("\n❌ ALL new features are domain-discriminating. Ablation stops.")
        return

    new_tr_safe = new_tr[safe_feats]
    new_te_safe = new_te[safe_feats]
    print(f"\n  Using {len(safe_feats)} safe new features.")

    # ─── S6 Block E OOF baseline (for DPP and slab gates) ───
    oof_s6_df = pd.read_csv('data_checkpoint/oof_s6_block_e.csv')
    oof_s6 = oof_s6_df['pred'].values
    dpp_auc_baseline = roc_auc_score(y_tr[dpp_idx], oof_s6[dpp_idx])

    # ─── 5-fold CV with augmented features ───
    print("\n[2/5] Training with new features appended to S6 feature set...")
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=C.SEED)
    group_id = np.arange(len(y_tr)) // cfg.cv_block_size
    rng = np.random.default_rng(C.SEED)
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    kinds  = ["lgbm", "xgb", "hgb", "cat"]
    seeds  = [42, 43, 44]

    oof_new = np.zeros(len(y_tr))
    all_models = []
    slab_idx = None

    for fold_idx, (tr_i, va_i) in enumerate(skf.split(cube_tr, y_tr, groups=group_id)):
        print(f"  Fold {fold_idx+1}/5")
        if fold_idx == 4:
            slab_idx = va_i

        # Augment training
        aug_cube, aug_y = C.augment_training_data(
            cube_tr[tr_i], y_tr[tr_i],
            n_copies=cfg.n_augmentation_copies, seed=C.SEED + fold_idx
        )
        Xtr_s6 = C.build_features(aug_cube, cfg=cfg)
        # Compute new features on aug_cube — augmentation adds masking but
        # we compute from the cube directly (masking already applied)
        new_feats_aug = extract_driftrobust_features(aug_cube)[safe_feats]
        Xtr = pd.concat([Xtr_s6.reset_index(drop=True),
                         new_feats_aug.reset_index(drop=True)], axis=1)

        # Seed-bagging
        fold_models = []
        for s in seeds:
            for k in kinds:
                m = make_model(k, s, pos_w)
                m.fit(Xtr, aug_y)
                fold_models.append(m)
                all_models.append(m)

        # TTA validation (1 original + 7 S2 dropout views)
        base_va_cube = C.simulate_test_masking(cube_tr[va_i], rng)
        views = [base_va_cube] + [apply_s2_dropout(base_va_cube, rng) for _ in range(7)]
        all_logits = []
        for v_cube in views:
            Xva_s6 = C.build_features(v_cube, cfg=cfg)
            Xva_new = extract_driftrobust_features(v_cube)[safe_feats]
            Xva = pd.concat([Xva_s6.reset_index(drop=True),
                             Xva_new.reset_index(drop=True)], axis=1)
            for m in fold_models:
                p = m.predict_proba(Xva)[:, 1]
                all_logits.append(logit(np.clip(p, 1e-7, 1-1e-7)))
        oof_new[va_i] = expit(np.mean(all_logits, axis=0))

    # ─── OOF Evaluation ───
    f1_new, auc_new, wsc_new = _score(y_tr, oof_new)
    dpp_auc_new  = roc_auc_score(y_tr[dpp_idx], oof_new[dpp_idx])
    slab_auc_new = roc_auc_score(y_tr[slab_idx], oof_new[slab_idx])
    slab_auc_s6  = roc_auc_score(y_tr[slab_idx], oof_s6[slab_idx])

    print(f"\n[3/5] OOF Results:")
    print(f"  S6 Block E baseline — DPP={dpp_auc_baseline:.4f}")
    print(f"  New ablation        — Overall F1={f1_new:.4f} AUC={auc_new:.4f} Score={wsc_new:.4f}")
    print(f"  New ablation        — DPP={dpp_auc_new:.4f} (delta={dpp_auc_new-dpp_auc_baseline:+.4f})")
    print(f"  New ablation        — Slab={slab_auc_new:.4f} (S6 baseline={slab_auc_s6:.4f})")

    # ─── Gates ───
    print(f"\n[4/5] Gate Assessment:")
    dpp_delta   = dpp_auc_new - dpp_auc_baseline
    slab_delta  = slab_auc_new - slab_auc_s6
    gate_dpp    = dpp_delta > DPP_SIGMA
    gate_slab   = slab_delta >= -0.001

    print(f"  DPP gate  (delta > {DPP_SIGMA:.4f}): {dpp_delta:+.4f} → {'✅ PASS' if gate_dpp else '❌ FAIL'}")
    print(f"  Slab gate (delta >= -0.001):      {slab_delta:+.4f} → {'✅ PASS' if gate_slab else '❌ FAIL'}")

    if not (gate_dpp and gate_slab):
        print("\n❌ Gate(s) failed. New indices do not improve domain-proximal score.")
        print("   No submission generated. S6 Block E remains the champion.")
        return

    # ─── Generate Submission ───
    print("\n[5/5] Generating submission...")
    t_star, best_score, best_f1, best_auc = find_optimal_t(y_tr, oof_new)
    shift = logit(0.5) - logit(np.clip(t_star, 1e-7, 1-1e-7))
    print(f"  t*={t_star:.4f}  shift={shift:.4f}  Score={best_score:.4f}")

    # Test predictions — TTA on cube_te
    all_te_logits = []
    Xte_s6 = C.build_features(cube_te, cfg=cfg)
    Xte_new = new_te_safe.reset_index(drop=True)
    te_views = [cube_te] + [apply_s2_dropout(cube_te, rng) for _ in range(7)]
    for v_cube in te_views:
        Xv_s6  = C.build_features(v_cube, cfg=cfg)
        Xv_new = extract_driftrobust_features(v_cube)[safe_feats]
        Xv = pd.concat([Xv_s6.reset_index(drop=True),
                        Xv_new.reset_index(drop=True)], axis=1)
        for m in all_models:
            p = m.predict_proba(Xv)[:, 1]
            all_te_logits.append(logit(np.clip(p, 1e-7, 1-1e-7)))
    te_preds = expit(np.mean(all_te_logits, axis=0) + shift)

    # Pre-flight
    sub = pd.DataFrame({
        'ID': te_df['ID'],
        'TargetRAUC': te_preds,
        'TargetF1': (te_preds >= 0.5).astype(int)
    })
    s4_sub = pd.read_csv('submissions/S4_rank_ensemble.csv')
    pos_rate  = sub['TargetF1'].mean()
    spearman  = sub['TargetRAUC'].corr(s4_sub['TargetRAUC'], method='spearman')
    n_unique  = len(sub['TargetRAUC'].unique())

    print(f"\n  Positive rate: {pos_rate:.4f} (target ~0.6175)")
    print(f"  Spearman corr with S4: {spearman:.4f}")
    print(f"  Unique RAUC values: {n_unique}/{len(sub)}")

    assert len(sub) == 1030
    assert n_unique == len(sub), "Unique value constraint violated"

    sub_path = 'submissions/s6_newidx_submission.csv'
    sub.to_csv(sub_path, index=False)
    print(f"\n✅ Saved: {sub_path}")
    print(f"Total time: {time.time()-t_total:.1f}s")


if __name__ == '__main__':
    main()
