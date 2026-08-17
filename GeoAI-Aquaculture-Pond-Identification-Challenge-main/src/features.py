"""Spectral index feature engineering for the aquaculture pond challenge.

Indices are computed from:
  - Sentinel-1 SAR: VH, VV  (dB values, typically -35 to -5)
  - Sentinel-2 optical: blue, green, red, re1, re2, re3, nir, nira, swir1, swir2
    (integer surface reflectance ×10000, range ~0-10000)

References: https://github.com/awesome-spectral-indices/awesome-spectral-indices
"""

import numpy as np
import pandas as pd

from src.preprocessing import (
    N_TIMESTEPS,
    get_band_matrix,
    replace_missing,
)

EPS = 1e-6


# ---------------------------------------------------------------------------
# Per-timestep spectral index functions
# ---------------------------------------------------------------------------

def _ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red)  — low for water bodies."""
    return (nir - red) / (nir + red + EPS)


def _ndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """NDWI = (Green - SWIR1) / (Green + SWIR1)  — high for water."""
    return (green - swir1) / (green + swir1 + EPS)


def _mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """MNDWI shares formula with NDWI for Sentinel-2 band mapping."""
    return _ndwi(green, swir1)


def _awei_nsh(green: np.ndarray, swir1: np.ndarray, nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """AWEInsh = 4*(G - S1) - 0.25*N + 2.75*S2  (no-shadow variant)."""
    return 4.0 * (green - swir1) - 0.25 * nir + 2.75 * swir2


def _awei_sh(blue: np.ndarray, green: np.ndarray, nir: np.ndarray, swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """AWEIsh = B + 2.5*G - 1.5*(N + S1) - 0.25*S2  (shadow-elimination variant)."""
    return blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2


def _evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """EVI = 2.5*(N-R)/(N + 6R - 7.5B + 10000)  — enhanced vegetation index."""
    return 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 10000 + EPS)


def _savi(nir: np.ndarray, red: np.ndarray, L_add: float = 0.5, L_scaled: float = 5000.0) -> np.ndarray:
    """SAVI = (1+0.5)*(N-R)/(N+R+5000)  — L=0.5, denominator offset scaled to ×10000 reflectance."""
    return (1 + L_add) * (nir - red) / (nir + red + L_scaled + EPS)


def _bsi(swir1: np.ndarray, red: np.ndarray, nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """BSI = ((S1+R)-(N+B)) / ((S1+R)+(N+B))  — bare soil index, positive for bare land."""
    num = (swir1 + red) - (nir + blue)
    den = (swir1 + red) + (nir + blue)
    return num / (den + EPS)


def _ndre(nir: np.ndarray, re1: np.ndarray) -> np.ndarray:
    """NDRE = (NIR - RE1) / (NIR + RE1)  — red-edge chlorophyll."""
    return (nir - re1) / (nir + re1 + EPS)


def _cig(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    """CIG = NIR/Green - 1  — chlorophyll index green."""
    return (nir / (green + EPS)) - 1.0


def _sar_ratio(vh: np.ndarray, vv: np.ndarray) -> np.ndarray:
    """VH-VV difference in dB (ratio in linear scale)."""
    return vh - vv


def _sar_rvi(vh: np.ndarray, vv: np.ndarray) -> np.ndarray:
    """Radar Vegetation Index (RVI): 4*VH_lin/(VV_lin+VH_lin). Clips to realistic SAR dB range first."""
    vh_c = np.clip(vh, -40.0, 0.0)
    vv_c = np.clip(vv, -40.0, 0.0)
    vh_lin = np.power(10.0, vh_c / 10.0)
    vv_lin = np.power(10.0, vv_c / 10.0)
    return 4.0 * vh_lin / (vv_lin + vh_lin + EPS)


# ---------------------------------------------------------------------------
# Temporal aggregation
# ---------------------------------------------------------------------------

def _temporal_trend(mat: np.ndarray) -> np.ndarray:
    """Linear slope of index values over the valid timesteps.

    Positive = signal increasing over the year, negative = decreasing.
    Returns NaN for samples with fewer than 3 valid observations.
    """
    n_samples, n_times = mat.shape
    t = np.arange(1, n_times + 1, dtype=float)
    slopes = np.full(n_samples, np.nan)
    for i in range(n_samples):
        valid = ~np.isnan(mat[i])
        n = int(valid.sum())
        if n >= 3:
            x = t[valid]
            y = mat[i, valid]
            xm = x.mean()
            ym = y.mean()
            denom = float(np.sum((x - xm) ** 2))
            if denom > EPS:
                slopes[i] = float(np.sum((x - xm) * (y - ym)) / denom)
    return slopes


def _temporal_stats(mat: np.ndarray, prefix: str) -> dict:
    """Compute mean, std, min, max, range over valid (non-NaN) timesteps."""
    n_valid = np.sum(~np.isnan(mat), axis=1)
    with np.errstate(all="ignore"):
        mean = np.nanmean(mat, axis=1)
        std = np.nanstd(mat, axis=1)
        mn = np.nanmin(mat, axis=1)
        mx = np.nanmax(mat, axis=1)
    mean[n_valid == 0] = np.nan
    std[n_valid == 0] = np.nan
    mn[n_valid == 0] = np.nan
    mx[n_valid == 0] = np.nan
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": mn,
        f"{prefix}_max": mx,
        f"{prefix}_range": mx - mn,
        f"{prefix}_n_valid": n_valid.astype(float),
    }


# ---------------------------------------------------------------------------
# Main feature engineering pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix from raw band columns.

    Args:
        df: raw DataFrame (missing values already replaced with NaN)

    Returns:
        Feature DataFrame indexed the same as df.
    """
    features: dict[str, np.ndarray] = {}

    # --- retrieve band matrices (n_samples, 12) ---
    blue  = get_band_matrix(df, "blue")
    green = get_band_matrix(df, "green")
    red   = get_band_matrix(df, "red")
    nir   = get_band_matrix(df, "nir")
    nira  = get_band_matrix(df, "nira")
    re1   = get_band_matrix(df, "re1")
    re2   = get_band_matrix(df, "re2")
    re3   = get_band_matrix(df, "re3")
    swir1 = get_band_matrix(df, "swir1")
    swir2 = get_band_matrix(df, "swir2")
    vh    = get_band_matrix(df, "VH")
    vv    = get_band_matrix(df, "VV")

    # --- per-timestep index matrices ---
    indices = {
        "ndvi":     _ndvi(nir, red),
        "ndwi":     _ndwi(green, swir1),
        "awei_nsh": _awei_nsh(green, swir1, nir, swir2),
        "awei_sh":  _awei_sh(blue, green, nir, swir1, swir2),
        "evi":      _evi(nir, red, blue),
        "savi":     _savi(nir, red),
        "bsi":      _bsi(swir1, red, nir, blue),
        "ndre":     _ndre(nir, re1),
        "cig":      _cig(nir, green),
        "sar_diff": _sar_ratio(vh, vv),
        "sar_rvi":  _sar_rvi(vh, vv),
    }

    # raw band temporal stats
    raw_bands = {
        "vh": vh, "vv": vv,
        "blue": blue, "green": green, "red": red,
        "nir": nir, "nira": nira,
        "re1": re1, "re2": re2, "re3": re3,
        "swir1": swir1, "swir2": swir2,
    }
    for name, mat in raw_bands.items():
        features.update(_temporal_stats(mat, name))

    # spectral index temporal stats
    for name, mat in indices.items():
        features.update(_temporal_stats(mat, name))

    # temporal trend (slope) for key indices
    for name in ["ndwi", "ndvi", "awei_sh", "sar_diff"]:
        features[f"{name}_trend"] = _temporal_trend(indices[name])
    features["vh_trend"] = _temporal_trend(vh)
    features["vv_trend"] = _temporal_trend(vv)

    # coefficient of variation — permanent water has low CV
    _cv_sources = {**raw_bands, **indices}  # indices overwrite raw_bands keys if same name
    for name in ["ndwi", "ndvi", "awei_sh", "vh", "vv"]:
        if name in _cv_sources:
            with np.errstate(all="ignore"):
                mn = np.nanmean(_cv_sources[name], axis=1)
                sd = np.nanstd(_cv_sources[name], axis=1)
            features[f"{name}_cv"] = sd / (np.abs(mn) + EPS)

    # interaction: high NDWI + low NDVI → permanent open water
    ndwi_mean = features["ndwi_mean"]
    ndvi_mean = features["ndvi_mean"]
    features["water_score"]  = ndwi_mean - ndvi_mean          # high for ponds
    features["pond_signal"]  = ndwi_mean * (1.0 - ndvi_mean)  # product interaction
    features["vh_ndwi"]      = features["vh_mean"] * ndwi_mean  # SAR × water index

    # per-timestep values for raw SAR (always available, critical for model)
    for t in range(N_TIMESTEPS):
        features[f"vh_t{t+1:02d}"] = vh[:, t]
        features[f"vv_t{t+1:02d}"] = vv[:, t]

    # global valid count (how many months have optical data)
    opt_valid = ~np.isnan(nir)
    features["n_opt_months"] = opt_valid.sum(axis=1).astype(float)

    return pd.DataFrame(features, index=df.index)


def prepare_Xy(
    train_path: str,
    test_path: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Full pipeline: load → clean → engineer features.

    Returns:
        X_train, y_train, X_test
    """
    from src.preprocessing import load_train, load_test

    train_raw, y = load_train(train_path)
    test_raw = load_test(test_path)

    train_clean = replace_missing(train_raw)
    test_clean = replace_missing(test_raw)

    X_train = build_features(train_clean)
    X_test = build_features(test_clean)

    return X_train, y, X_test
