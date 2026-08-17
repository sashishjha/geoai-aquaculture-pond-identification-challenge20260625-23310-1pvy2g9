"""Unit tests for src/features.py"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import ALL_BANDS, N_TIMESTEPS, replace_missing
from src.features import (
    build_features,
    _ndvi,
    _ndwi,
    _awei_nsh,
    _awei_sh,
    _evi,
    _savi,
    _bsi,
    _ndre,
    _cig,
    _sar_ratio,
    _sar_rvi,
    _temporal_stats,
)


def _make_clean_df(n: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = {
        f"{b}_{t:02d}": rng.uniform(500, 5000, n).astype(float)
        for b in ALL_BANDS
        for t in range(1, N_TIMESTEPS + 1)
    }
    return replace_missing(pd.DataFrame(cols))


# ---------------------------------------------------------------------------
# Spectral index range / sign tests
# ---------------------------------------------------------------------------

def test_ndvi_range():
    nir = np.array([3000.0, 2000.0])
    red = np.array([1000.0, 3000.0])
    result = _ndvi(nir, red)
    assert (result >= -1).all() and (result <= 1).all()


def test_ndvi_water_like():
    """Water has NIR < Red → NDVI negative."""
    nir = np.array([500.0])
    red = np.array([1500.0])
    assert _ndvi(nir, red)[0] < 0


def test_ndwi_water_positive():
    """Open water: Green > SWIR1 → NDWI > 0."""
    green = np.array([2000.0])
    swir1 = np.array([500.0])
    assert _ndwi(green, swir1)[0] > 0


def test_ndwi_land_negative():
    green = np.array([1000.0])
    swir1 = np.array([2500.0])
    assert _ndwi(green, swir1)[0] < 0


def test_awei_nsh_water_positive():
    """AWEInsh should be positive for water pixels."""
    green = np.array([2000.0])
    swir1 = np.array([400.0])
    nir = np.array([500.0])
    swir2 = np.array([200.0])
    assert _awei_nsh(green, swir1, nir, swir2)[0] > 0


def test_awei_sh_water_positive():
    blue = np.array([1000.0])
    green = np.array([2000.0])
    nir = np.array([500.0])
    swir1 = np.array([400.0])
    swir2 = np.array([300.0])
    assert _awei_sh(blue, green, nir, swir1, swir2)[0] > 0


def test_evi_bounded():
    nir = np.array([3000.0])
    red = np.array([1000.0])
    blue = np.array([500.0])
    result = _evi(nir, red, blue)
    assert result[0] > 0  # vegetation


def test_savi_range():
    nir = np.array([3000.0, 500.0])
    red = np.array([1000.0, 3000.0])
    result = _savi(nir, red)
    # SAVI with L=0.5 produces values in (-1.5, 1.5) for valid reflectance
    assert (result >= -1.5).all() and (result <= 1.5).all()


def test_bsi_bare_soil_positive():
    """High SWIR1+Red and low NIR+Blue → positive BSI."""
    swir1 = np.array([3000.0])
    red = np.array([2500.0])
    nir = np.array([1000.0])
    blue = np.array([500.0])
    assert _bsi(swir1, red, nir, blue)[0] > 0


def test_ndre_range():
    nir = np.array([3000.0])
    re1 = np.array([2000.0])
    result = _ndre(nir, re1)
    assert -1 <= result[0] <= 1


def test_cig_positive_vegetation():
    nir = np.array([3000.0])
    green = np.array([1000.0])
    assert _cig(nir, green)[0] > 0


def test_sar_ratio_shape():
    vh = np.array([-20.0, -25.0])
    vv = np.array([-10.0, -15.0])
    result = _sar_ratio(vh, vv)
    assert result.shape == (2,)
    assert (result < 0).all()  # VH < VV → negative dB difference


def test_sar_rvi_bounded():
    vh = np.array([-20.0])
    vv = np.array([-10.0])
    result = _sar_rvi(vh, vv)
    assert 0 <= result[0] <= 1


# ---------------------------------------------------------------------------
# Temporal stats
# ---------------------------------------------------------------------------

def test_temporal_stats_all_valid():
    mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    stats = _temporal_stats(mat, "x")
    np.testing.assert_allclose(stats["x_mean"], [2.0, 5.0])
    np.testing.assert_allclose(stats["x_n_valid"], [3.0, 3.0])
    np.testing.assert_allclose(stats["x_min"], [1.0, 4.0])
    np.testing.assert_allclose(stats["x_max"], [3.0, 6.0])
    np.testing.assert_allclose(stats["x_range"], [2.0, 2.0])


def test_temporal_stats_with_nan():
    mat = np.array([[1.0, np.nan, 3.0]])
    stats = _temporal_stats(mat, "y")
    assert stats["y_n_valid"][0] == 2
    np.testing.assert_allclose(stats["y_mean"], [2.0])


def test_temporal_stats_all_nan():
    mat = np.array([[np.nan, np.nan]])
    stats = _temporal_stats(mat, "z")
    assert stats["z_n_valid"][0] == 0
    assert np.isnan(stats["z_mean"][0])


# ---------------------------------------------------------------------------
# build_features integration test
# ---------------------------------------------------------------------------

def test_build_features_shape():
    df = _make_clean_df(n=10)
    feat = build_features(df)
    assert len(feat) == 10
    assert feat.shape[1] > 50  # should have many engineered features


def test_build_features_no_inf():
    df = _make_clean_df(n=20)
    feat = build_features(df)
    assert not np.isinf(feat.values).any(), "build_features produced inf values"


def test_build_features_index_preserved():
    df = _make_clean_df(n=5)
    df.index = [f"ID_{i}" for i in range(5)]
    feat = build_features(df)
    assert list(feat.index) == list(df.index)


def test_build_features_with_missing():
    df = _make_clean_df(n=3)
    # knock out all optical for first sample first month
    from src.preprocessing import OPT_BANDS
    for b in OPT_BANDS:
        df.loc[0, f"{b}_01"] = np.nan
    feat = build_features(df)
    # n_opt_months for first sample should be 11
    assert feat.loc[0, "n_opt_months"] == N_TIMESTEPS - 1
