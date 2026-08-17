"""
common.py — shared config, IO, masking simulator, and temporal feature engineering.

Zindi GeoAI Aquaculture Pond Identification Challenge (FAO x ITU).

Design constraints baked in here (see docs_checkpoint/data_audit.md):
  * Train rows are FULLY observed (all 12 months). Test rows show only a
    CONSECUTIVE window of 4/5/6 months (defined by Sentinel-1 presence), with
    everything outside the window set to the -9999 sentinel.
  * WITHIN the observed window, Sentinel-2 (optical) can be additionally masked
    (-9999) on individual months due to cloud cover, while Sentinel-1 (SAR,
    VH/VV) is always present in the window (cloud-independent).
  * Therefore every feature is computed by aggregating ONLY over valid months,
    PER SENSOR, and missingness itself is encoded as features. Nothing here
    assumes a specific calendar month exists.

All randomness is seeded. Nothing tunes a decision threshold — the 0.5 rule is
enforced exclusively at submission time (see generate_submission.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

SEED = 42
SENTINEL = -9999.0

MONTHS = [f"{m:02d}" for m in range(1, 13)]
S1_BANDS = ["VH", "VV"]                      # SAR, cloud-independent
S2_BANDS = ["blue", "green", "nir", "nira", "re1", "re2", "re3",
            "red", "swir1", "swir2"]         # optical
ALL_BANDS = S1_BANDS + S2_BANDS
FEATURE_COLS = [f"{b}_{mm}" for mm in MONTHS for b in ALL_BANDS]


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_raw(train_path: str, test_path: str, dedup: bool = True):
    tr = pd.read_csv(train_path)
    te = pd.read_csv(test_path)
    if dedup:
        # 4 exact duplicate feature-pairs exist in train, all label==1 (no label
        # conflict). Drop to avoid the same observation landing in two CV folds.
        before = len(tr)
        tr = tr.drop_duplicates(subset=FEATURE_COLS, keep="first").reset_index(drop=True)
        if len(tr) != before:
            print(f"[load_raw] dropped {before - len(tr)} duplicate train rows")
    return tr, te


def band_cube(df: pd.DataFrame) -> np.ndarray:
    """Return array of shape (N, 12 months, 12 bands) in ALL_BANDS order."""
    N = len(df)
    cube = np.empty((N, 12, len(ALL_BANDS)), dtype=np.float64)
    for j, mm in enumerate(MONTHS):
        for k, b in enumerate(ALL_BANDS):
            cube[:, j, k] = df[f"{b}_{mm}"].values
    return cube


# --------------------------------------------------------------------------- #
# Masking simulator — the heart of the robust validation scheme (§17)
# --------------------------------------------------------------------------- #
def simulate_test_masking(cube: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = cube.copy()
    N = out.shape[0]
    s2_idx = [ALL_BANDS.index(b) for b in S2_BANDS]
    
    # Empirical distributions extracted from Test.csv
    START_PROBS = np.array([0.125243, 0.127184, 0.126214, 0.125243, 0.126214, 0.126214, 0.126214, 0.079612, 0.037864])
    START_PROBS = START_PROBS / START_PROBS.sum()
    S2_CLOUD_RATES = np.array([0.008, 0.146, 0.008, 0.002, 0.002, 0.114, 0.002, 0.015, 0.006, 0.463, 0.027, 0.000])

    for i in range(N):
        L = int(rng.integers(4, 7))            # 4,5,6 inclusive (uniform, matches test)
        
        # 7c Backport: The empirical joint (window_len, start_month) of the Test set is uniform.
        start = int(rng.integers(0, 13 - L))
            
        window = set(range(start, start + L))
        for m in range(12):
            if m not in window:
                out[i, m, :] = SENTINEL
            else:
                # cloud-mask S2 within window based on exact empirical per-month rate
                if rng.random() < S2_CLOUD_RATES[m]:
                    out[i, m, s2_idx] = SENTINEL
    return out


def augment_training_data(cube: np.ndarray, y: np.ndarray, n_copies: int = 15, seed: int = SEED):
    """
    Augments the training data by generating multiple random test-like maskings.
    This multiplies the dataset by `n_copies` while perfectly simulating the test domain.
    """
    rng = np.random.default_rng(seed)
    augmented_cubes = []
    augmented_y = []
    for _ in range(n_copies):
        augmented_cubes.append(simulate_test_masking(cube, rng))
        augmented_y.append(y)
    return np.concatenate(augmented_cubes, axis=0), np.concatenate(augmented_y, axis=0)

# --------------------------------------------------------------------------- #
# Feature engineering — all robust to arbitrary 4-6 month windows (§19)
# --------------------------------------------------------------------------- #
def _valid_mask(cube: np.ndarray) -> np.ndarray:
    """Boolean (N,12,12): True where a band-cell is a real observation."""
    return cube != SENTINEL


def _masked_stats(values: np.ndarray, valid: np.ndarray):
    """Vectorised per-row stats over valid months only.

    Returns dict with mean, median, std, min, max, first, last, slope; NaN where
    a row has no valid month. Uses numpy masked arrays (no per-row Python loop
    for the moment-style stats) and a vectorised least-squares slope.
    """
    N, Mo = values.shape
    m = np.ma.array(values, mask=~valid)
    cnt = valid.sum(axis=1)
    has = cnt > 0
    def _fill(a):
        a = np.asarray(a, dtype=float)
        a[~has] = np.nan
        return a
    mean = _fill(m.mean(axis=1).filled(np.nan))
    std = _fill(m.std(axis=1).filled(np.nan))
    mn = _fill(m.min(axis=1).filled(np.nan))
    mx = _fill(m.max(axis=1).filled(np.nan))
    # median needs care with masked arrays -> use filled with nan + nanmedian
    vals_nan = np.where(valid, values, np.nan)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(vals_nan, axis=1)
    median = _fill(median)
    # first / last valid observation
    idx = np.arange(Mo)[None, :]
    first_idx = np.where(valid, idx, Mo).min(axis=1)
    last_idx = np.where(valid, idx, -1).max(axis=1)
    rows = np.arange(N)
    first = np.where(has, values[rows, np.clip(first_idx, 0, Mo - 1)], np.nan)
    last = np.where(has, values[rows, np.clip(last_idx, 0, Mo - 1)], np.nan)
    # vectorised slope: cov(t,x)/var(t) over valid months per row
    t = np.where(valid, idx.astype(float), np.nan)
    x = vals_nan
    with np.errstate(invalid="ignore", divide="ignore"):
        tbar = np.nanmean(t, axis=1, keepdims=True)
        xbar = np.nanmean(x, axis=1, keepdims=True)
        num = np.nansum((t - tbar) * (x - xbar), axis=1)
        den = np.nansum((t - tbar) ** 2, axis=1)
        slope = np.where(den > 0, num / den, 0.0)
    slope = np.where(has, slope, np.nan)
    return dict(mean=mean, median=median, std=std, min=mn, max=mx,
                first=first, last=last, slope=slope)


def _agg_over_valid(values: np.ndarray, valid: np.ndarray, func) -> np.ndarray:
    """Apply func over the month axis using only valid cells; NaN where none.

    Retained for spectral-index aggregation; vectorised for mean/median/std/min/
    max, falls back to a per-row loop only for uncommon custom funcs.
    """
    vals_nan = np.where(valid, values, np.nan)
    has = valid.sum(axis=1) > 0
    with np.errstate(invalid="ignore"):
        if func is np.mean:
            out = np.nanmean(vals_nan, axis=1)
        elif func is np.median:
            out = np.nanmedian(vals_nan, axis=1)
        elif func is np.min:
            out = np.nanmin(vals_nan, axis=1)
        elif func is np.max:
            out = np.nanmax(vals_nan, axis=1)
        else:  # e.g. std lambda
            N = values.shape[0]
            out = np.full(N, np.nan)
            for i in range(N):
                v = values[i][valid[i]]
                if v.size:
                    out[i] = func(v)
            return out
    return np.where(has, out, np.nan)


def build_features(cube: np.ndarray, cfg=None) -> pd.DataFrame:
    """Compute the temporal / sensor-fusion / missingness feature table.

    Per band we aggregate over VALID months only: mean, median, std, min, max,
    range, first-valid, last-valid, and slope (trend vs. valid month index).
    Plus per-sensor valid-month counts, S2 cloud-within-window count, window
    length, and a set of spectral indices (NDVI/NDWI/MNDWI/NDBI + SAR ratio)
    aggregated the same way. None of these require a specific month to exist.

    cfg : RunConfig or None. If None, uses Iteration 1 champion defaults
          (absolute anchors ON, z-shape OFF, rank features OFF).
    """
    # Feature flags — default to Iteration 1 champion config
    if cfg is None:
        use_absolute = True
        use_z_shape  = False
        use_rank     = False
    else:
        use_absolute = cfg.use_absolute_anchors
        use_z_shape  = cfg.use_z_shape_features
        use_rank     = cfg.use_rank_features

    valid = _valid_mask(cube)
    N = cube.shape[0]
    feats: dict[str, np.ndarray] = {}

    # per-band temporal stats over valid months (vectorised)
    for k, b in enumerate(ALL_BANDS):
        vals = cube[:, :, k]
        vmask = valid[:, :, k]
        s = _masked_stats(vals, vmask)

        # Iteration 1 champion features — absolute anchors
        if use_absolute:
            feats[f"{b}_mean"]   = s["mean"]
            feats[f"{b}_median"] = s["median"]
            feats[f"{b}_min"]    = s["min"]
            feats[f"{b}_max"]    = s["max"]
            feats[f"{b}_first"]  = s["first"]
            feats[f"{b}_last"]   = s["last"]
        feats[f"{b}_std"]    = s["std"]
        feats[f"{b}_range"]  = s["max"] - s["min"]
        feats[f"{b}_slope"]  = s["slope"]

        # Phase 2 z-scored shape features (additive, not substitutive)
        if use_z_shape:
            with np.errstate(invalid="ignore"):
                global_std_reg = np.nanmedian(s["std"]) * 0.1
                reg_std = s["std"] + global_std_reg + 1e-6
                feats[f"{b}_z_min"]   = (s["min"] - s["mean"]) / reg_std
                feats[f"{b}_z_max"]   = (s["max"] - s["mean"]) / reg_std
                feats[f"{b}_z_first"] = (s["first"] - s["mean"]) / reg_std
                feats[f"{b}_z_last"]  = (s["last"] - s["mean"]) / reg_std
                feats[f"{b}_z_slope"] = s["slope"] / reg_std

        # Nonparametric rank-based features
        if use_rank:
            with np.errstate(invalid="ignore"):
                Nn = vals.shape[0]
                p_min   = np.full(Nn, np.nan)
                p_max   = np.full(Nn, np.nan)
                p_first = np.full(Nn, np.nan)
                p_last  = np.full(Nn, np.nan)
                sign_changes = np.full(Nn, np.nan)
                for i in range(Nn):
                    v = vals[i][vmask[i]]
                    if v.size > 0:
                        ranks = np.argsort(np.argsort(v)) / max(1, len(v) - 1)
                        p_min[i]   = ranks.min()
                        p_max[i]   = ranks.max()
                        p_first[i] = ranks[0]
                        p_last[i]  = ranks[-1]
                        sign_changes[i] = np.sum(
                            np.diff(np.sign(v - s["median"][i])) != 0
                        ) if v.size > 1 else 0
                feats[f"{b}_rank_first"]   = p_first
                feats[f"{b}_rank_last"]    = p_last
                feats[f"{b}_sign_changes"] = sign_changes

        # Phase 6.5 Feature Rescue for Base Bands (SAR / Optical)
        if getattr(cfg, "use_rescue_features", False):
            with np.errstate(invalid="ignore", divide="ignore"):
                Nn = vals.shape[0]
                if b in ("VH", "VV"):
                    diff_std = np.full(Nn, np.nan)
                    diff_iqr = np.full(Nn, np.nan)
                    for i in range(Nn):
                        v = vals[i][vmask[i]]
                        if v.size > 1:
                            d = np.diff(v)
                            diff_std[i] = np.nanstd(d)
                            diff_iqr[i] = np.nanpercentile(d, 75) - np.nanpercentile(d, 25)
                    feats[f"{b}_rescue_diff_std"] = diff_std
                    feats[f"{b}_rescue_diff_iqr"] = diff_iqr
                elif b in ("blue", "green", "red", "nir", "swir1", "swir2", "re1", "re2", "re3"):
                    # Relative amplitude / CV
                    feats[f"{b}_rescue_cv"] = s["std"] / (np.abs(s["median"]) + 1e-6)

    s1_idx = [ALL_BANDS.index(b) for b in S1_BANDS]
    s2_idx = [ALL_BANDS.index(b) for b in S2_BANDS]
    s1_month_valid = valid[:, :, s1_idx].all(axis=2)   # (N,12)
    s2_month_valid = valid[:, :, s2_idx].all(axis=2)
    feats["n_valid_months_s1"] = s1_month_valid.sum(axis=1).astype(float)  # window length
    feats["n_valid_months_s2"] = s2_month_valid.sum(axis=1).astype(float)
    feats["window_length"]     = s1_month_valid.sum(axis=1).astype(float)
    feats["s2_cloud_in_window"] = (s1_month_valid & ~s2_month_valid).sum(axis=1).astype(float)
    feats["s2_avail_ratio"] = np.where(
        feats["n_valid_months_s1"] > 0,
        feats["n_valid_months_s2"] / np.maximum(feats["n_valid_months_s1"], 1), 0.0)

    # ---- spectral indices, aggregated over valid months (§19) ----
    def band(k):  # month series for band index k
        return cube[:, :, k]
    def bvalid(k):
        return valid[:, :, k]
    idxmap = {b: ALL_BANDS.index(b) for b in ALL_BANDS}

    def index_series(num_b, den_b):
        a = band(idxmap[num_b]); c = band(idxmap[den_b])
        m = bvalid(idxmap[num_b]) & bvalid(idxmap[den_b])
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (a - c) / (a + c)
        s = np.where(np.isfinite(s), s, np.nan)
        return s, m

    spectral = {
        "ndvi":  ("nir", "red"),      # vegetation
        "ndwi":  ("green", "nir"),    # open water (McFeeters)
        "mndwi": ("green", "swir1"),  # modified water index — pond signal
        "ndbi":  ("swir1", "nir"),    # built-up
        "sar_ratio": None,            # handled below
        "evi": None,
        "awei": None,
        "bsi": None,
    }
    for name, pair in spectral.items():
        if name == "sar_ratio":
            vh = band(idxmap["VH"]); vv = band(idxmap["VV"])
            m = bvalid(idxmap["VH"]) & bvalid(idxmap["VV"])
            s = vh - vv  # dB difference (log-domain ratio)
        elif name == "evi":
            nir = band(idxmap["nir"]); red = band(idxmap["red"]); blue = band(idxmap["blue"])
            m = bvalid(idxmap["nir"]) & bvalid(idxmap["red"]) & bvalid(idxmap["blue"])
            with np.errstate(divide="ignore", invalid="ignore"):
                s = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0)
            s = np.where(np.isfinite(s), s, np.nan)
        elif name == "awei":
            blue = band(idxmap["blue"]); green = band(idxmap["green"])
            nir = band(idxmap["nir"]); swir1 = band(idxmap["swir1"]); swir2 = band(idxmap["swir2"])
            m = bvalid(idxmap["blue"]) & bvalid(idxmap["green"]) & bvalid(idxmap["nir"]) & bvalid(idxmap["swir1"]) & bvalid(idxmap["swir2"])
            s = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
        elif name == "bsi":
            swir1 = band(idxmap["swir1"]); red = band(idxmap["red"])
            nir = band(idxmap["nir"]); blue = band(idxmap["blue"])
            m = bvalid(idxmap["swir1"]) & bvalid(idxmap["red"]) & bvalid(idxmap["nir"]) & bvalid(idxmap["blue"])
            with np.errstate(divide="ignore", invalid="ignore"):
                s = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
            s = np.where(np.isfinite(s), s, np.nan)
        else:
            s, m = index_series(*pair)
            
        s_stats = _masked_stats(s, m)

        # Iteration 1 champion features — absolute anchors for spectral indices
        if use_absolute:
            feats[f"{name}_mean"]   = s_stats["mean"]
            feats[f"{name}_median"] = s_stats["median"]
            feats[f"{name}_min"]    = s_stats["min"]
            feats[f"{name}_max"]    = s_stats["max"]
            feats[f"{name}_first"]  = s_stats["first"]
            feats[f"{name}_last"]   = s_stats["last"]
        feats[f"{name}_std"]    = s_stats["std"]
        feats[f"{name}_slope"]  = s_stats["slope"]

        # Phase 2 z-scored shape features for spectral indices
        if use_z_shape:
            with np.errstate(invalid="ignore"):
                global_std_reg = np.nanmedian(s_stats["std"]) * 0.1
                reg_std = s_stats["std"] + global_std_reg + 1e-6
                feats[f"{name}_z_min"]   = (s_stats["min"] - s_stats["mean"]) / reg_std
                feats[f"{name}_z_max"]   = (s_stats["max"] - s_stats["mean"]) / reg_std
                feats[f"{name}_z_first"] = (s_stats["first"] - s_stats["mean"]) / reg_std
                feats[f"{name}_z_last"]  = (s_stats["last"] - s_stats["mean"]) / reg_std
                feats[f"{name}_z_slope"] = s_stats["slope"] / reg_std

        # Nonparametric rank-based features for spectral indices
        if use_rank:
            with np.errstate(invalid="ignore"):
                Nn = s.shape[0]
                p_first = np.full(Nn, np.nan)
                p_last  = np.full(Nn, np.nan)
                sign_changes = np.full(Nn, np.nan)
                for i in range(Nn):
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

        # Phase 6.5 Feature Rescue for Spectral Indices
        if getattr(cfg, "use_rescue_features", False):
            with np.errstate(invalid="ignore", divide="ignore"):
                Nn = s.shape[0]
                if name == "sar_ratio":
                    diff_std = np.full(Nn, np.nan)
                    diff_iqr = np.full(Nn, np.nan)
                    for i in range(Nn):
                        v = s[i][m[i]]
                        if v.size > 1:
                            d = np.diff(v)
                            diff_std[i] = np.nanstd(d)
                            diff_iqr[i] = np.nanpercentile(d, 75) - np.nanpercentile(d, 25)
                    feats[f"{name}_rescue_diff_std"] = diff_std
                    feats[f"{name}_rescue_diff_iqr"] = diff_iqr
                else:
                    feats[f"{name}_rescue_cv"] = s_stats["std"] / (np.abs(s_stats["median"]) + 1e-6)

    # ── Phase 9.1: Drift-Immune Phenology Features ────────────────────────────
    # Family A: First-difference / transition dynamics (harvest-cycle signature)
    # Family B: Within-row cross-band temporal correlation (phenology-coupling)
    # Both are affine-drift-immune: dB differences cancel gain shift exactly,
    # and cross-band correlations are invariant to any per-band affine transform.
    if getattr(cfg, "use_phenology_features", False):
        Nn = cube.shape[0]
        idxmap_ph = {b: ALL_BANDS.index(b) for b in ALL_BANDS}

        # ── Family A: First-difference dynamics ──────────────────────────────
        # VH and MNDWI (both reliable SAR and water signals)
        vh_series   = cube[:, :, idxmap_ph["VH"]]
        vv_series   = cube[:, :, idxmap_ph["VV"]]
        vh_valid    = valid[:, :, idxmap_ph["VH"]]

        green_s = cube[:, :, idxmap_ph["green"]]; swir1_s = cube[:, :, idxmap_ph["swir1"]]
        gv = valid[:, :, idxmap_ph["green"]]; sv = valid[:, :, idxmap_ph["swir1"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            mndwi_series = np.where(gv & sv, (green_s - swir1_s) / (green_s + swir1_s + 1e-9), np.nan)
        mndwi_valid = gv & sv

        for sig_name, sig, sig_v in [("vh_diff", vh_series, vh_valid),
                                      ("mndwi_diff", mndwi_series, mndwi_valid)]:
            max_abs_diff = np.full(Nn, np.nan)
            p90_abs_diff = np.full(Nn, np.nan)
            n_large_sign_changes = np.full(Nn, np.nan)
            n_median_crossings = np.full(Nn, np.nan)

            for i in range(Nn):
                v = sig[i][sig_v[i]]  # valid observations only
                if v.size < 2:
                    continue
                d = np.diff(v)
                abs_d = np.abs(d)
                max_abs_diff[i] = abs_d.max()
                p90_abs_diff[i] = np.percentile(abs_d, 90)

                # Sign changes whose magnitude exceeds 0.5 × row IQR (row-scaled threshold)
                row_iqr = np.percentile(abs_d, 75) - np.percentile(abs_d, 25) if abs_d.size >= 4 else abs_d.mean()
                threshold = 0.5 * row_iqr
                large_changes = abs_d > threshold
                sign_changes_arr = np.diff(np.sign(d)) != 0
                n_large_sign_changes[i] = float((sign_changes_arr & large_changes[:-1]).sum()) if d.size > 1 else 0.0

                # Crossings of the row's own median
                row_median = np.median(v)
                crossings = np.diff(np.sign(v - row_median))
                n_median_crossings[i] = float((crossings != 0).sum())

            feats[f"{sig_name}_max_abs"]   = max_abs_diff
            feats[f"{sig_name}_p90_abs"]   = p90_abs_diff
            feats[f"{sig_name}_large_sc"]  = n_large_sign_changes
            feats[f"{sig_name}_med_cross"] = n_median_crossings

        # ── Family B: Within-row cross-band temporal correlation ─────────────
        # Spearman corr(NDVI, MNDWI), corr(VH, MNDWI), corr(EVI, NDWI) per row
        # Requires ≥4 common valid months, else NaN. GBMs route around NaN natively.
        from scipy.stats import spearmanr as _spearmanr

        nir_s  = cube[:, :, idxmap_ph["nir"]];  red_s  = cube[:, :, idxmap_ph["red"]]
        niv    = valid[:, :, idxmap_ph["nir"]];  rdv    = valid[:, :, idxmap_ph["red"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi_s = np.where(niv & rdv, (nir_s - red_s) / (nir_s + red_s + 1e-9), np.nan)
        ndvi_valid = niv & rdv

        nir_s2 = cube[:, :, idxmap_ph["nir"]]
        grn_s  = cube[:, :, idxmap_ph["green"]]
        niv2   = valid[:, :, idxmap_ph["nir"]]; gnv = valid[:, :, idxmap_ph["green"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi_s = np.where(niv2 & gnv, (grn_s - nir_s2) / (grn_s + nir_s2 + 1e-9), np.nan)
        ndwi_valid = niv2 & gnv

        # EVI (reuse existing bands)
        blue_s2 = cube[:, :, idxmap_ph["blue"]]; blv = valid[:, :, idxmap_ph["blue"]]
        with np.errstate(divide="ignore", invalid="ignore"):
            evi_s = np.where(niv & rdv & blv,
                             2.5 * (nir_s - red_s) / (nir_s + 6.0 * red_s - 7.5 * blue_s2 + 1.0),
                             np.nan)
        evi_valid = niv & rdv & blv

        # Skip expensive per-row Spearman if all three corr features will be dropped anyway
        _drop_set = set(getattr(cfg, "drop_features", []) or [])
        _corr_feats = {"corr_ndvi_mndwi", "corr_vh_mndwi", "corr_evi_ndwi"}
        if not _corr_feats.issubset(_drop_set):
            def _row_spearman(a_ser, a_v, b_ser, b_v):
                """Per-row Spearman corr, NaN if <4 valid common months."""
                result = np.full(Nn, np.nan)
                both_v = a_v & b_v
                for i in range(Nn):
                    idx = both_v[i]
                    if idx.sum() < 4:
                        continue
                    a_vals = a_ser[i][idx]
                    b_vals = b_ser[i][idx]
                    if np.nanstd(a_vals) < 1e-10 or np.nanstd(b_vals) < 1e-10:
                        continue  # constant series → corr undefined
                    rho, _ = _spearmanr(a_vals, b_vals)
                    result[i] = rho
                return result

            feats["corr_ndvi_mndwi"] = _row_spearman(ndvi_s, ndvi_valid, mndwi_series, mndwi_valid)
            feats["corr_vh_mndwi"]   = _row_spearman(vh_series, vh_valid, mndwi_series, mndwi_valid)
            feats["corr_evi_ndwi"]   = _row_spearman(evi_s, evi_valid, ndwi_s, ndwi_valid)

        # Water permanence fraction: share of observed months with MNDWI above row's own median
        water_perm = np.full(Nn, np.nan)
        for i in range(Nn):
            v = mndwi_series[i][mndwi_valid[i]]
            if v.size >= 2:
                water_perm[i] = float((v > np.median(v)).mean())
        feats["water_permanence_frac"] = water_perm

        # ── Family C: Within-row Temporal Ranks & Position ─────────────────
        vh_rank_min_pos = np.full(Nn, np.nan)
        vh_rank_max_pos = np.full(Nn, np.nan)
        mndwi_rank_min_pos = np.full(Nn, np.nan)
        mndwi_rank_max_pos = np.full(Nn, np.nan)
        
        # ── Family D: Regime-conditional contrast ────────────────────────────
        vh_regime_mndwi_contrast = np.full(Nn, np.nan)
        
        # ── Family E: Drain-Refill Asymmetry ─────────────────────────────────
        vh_diff_skew = np.full(Nn, np.nan)
        vh_drain_refill_ratio = np.full(Nn, np.nan)
        
        from scipy.stats import skew as _skew
        
        for i in range(Nn):
            v_vh = vh_series[i][vh_valid[i]]
            v_mndwi = mndwi_series[i][mndwi_valid[i]]
            
            if v_vh.size > 0:
                vh_rank_min_pos[i] = np.argmin(v_vh) / v_vh.size
                vh_rank_max_pos[i] = np.argmax(v_vh) / v_vh.size
                
                if v_vh.size >= 3:
                    d_vh = np.diff(v_vh)
                    if np.std(d_vh) > 1e-9:
                        vh_diff_skew[i] = float(_skew(d_vh))
                    vh_drain_refill_ratio[i] = d_vh.max() / (np.abs(d_vh.min()) + 1e-9)
            
            if v_mndwi.size > 0:
                mndwi_rank_min_pos[i] = np.argmin(v_mndwi) / v_mndwi.size
                mndwi_rank_max_pos[i] = np.argmax(v_mndwi) / v_mndwi.size
                
            both_v = vh_valid[i] & mndwi_valid[i]
            if both_v.sum() >= 4:
                vh_c = vh_series[i][both_v]
                mndwi_c = mndwi_series[i][both_v]
                vh_med = np.median(vh_c)
                mask_low = vh_c < vh_med
                mask_high = vh_c > vh_med
                if mask_low.sum() > 0 and mask_high.sum() > 0:
                    vh_regime_mndwi_contrast[i] = mndwi_c[mask_low].mean() - mndwi_c[mask_high].mean()

        feats["vh_min_pos_frac"] = vh_rank_min_pos
        feats["vh_max_pos_frac"] = vh_rank_max_pos
        feats["mndwi_min_pos_frac"] = mndwi_rank_min_pos
        feats["mndwi_max_pos_frac"] = mndwi_rank_max_pos
        feats["vh_regime_mndwi_contrast"] = vh_regime_mndwi_contrast
        feats["vh_diff_skew"] = vh_diff_skew
        feats["vh_drain_refill_ratio"] = vh_drain_refill_ratio

    out = pd.DataFrame(feats)

    if cfg is not None and getattr(cfg, "drop_features", None):
        drops = [f for f in cfg.drop_features if f in out.columns]
        if drops:
            out = out.drop(columns=drops)
    # tree models handle NaN natively; keep NaN as an informative signal.
    return out


def make_xy(df: pd.DataFrame, cube: np.ndarray, cfg=None):
    X = build_features(cube, cfg=cfg)
    y = df["label"].values if "label" in df.columns else None
    return X, y
