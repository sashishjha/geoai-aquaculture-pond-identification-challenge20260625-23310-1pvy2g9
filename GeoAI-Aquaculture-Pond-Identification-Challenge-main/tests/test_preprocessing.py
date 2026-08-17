"""Unit tests for src/preprocessing.py"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    ALL_BANDS,
    N_TIMESTEPS,
    OPT_BANDS,
    SAR_BANDS,
    count_valid_timesteps,
    get_band_matrix,
    replace_missing,
)


def _make_df(n=5):
    """Create a minimal raw DataFrame with all expected columns."""
    cols = {f"{b}_{t:02d}": np.random.uniform(500, 5000, n) for b in ALL_BANDS for t in range(1, N_TIMESTEPS + 1)}
    return pd.DataFrame(cols)


def test_replace_missing():
    df = pd.DataFrame({"a": [-9999.0, 1.0, 2.0], "b": [3.0, -9999.0, 5.0]})
    result = replace_missing(df)
    assert np.isnan(result.loc[0, "a"])
    assert np.isnan(result.loc[1, "b"])
    assert result.loc[1, "a"] == 1.0


def test_replace_missing_no_side_effects():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    result = replace_missing(df)
    assert result is not df  # returns a new object (replace returns copy)


def test_get_band_matrix_shape():
    df = _make_df(n=10)
    mat = get_band_matrix(df, "VH")
    assert mat.shape == (10, N_TIMESTEPS)


def test_get_band_matrix_nan_propagation():
    df = _make_df(n=3)
    df = replace_missing(df)
    df["VH_01"] = np.nan
    mat = get_band_matrix(df, "VH")
    assert np.isnan(mat[:, 0]).all()
    assert not np.isnan(mat[:, 1]).any()


def test_count_valid_timesteps():
    df = _make_df(n=3)
    df = replace_missing(df)
    result = count_valid_timesteps(df)
    assert len(result) == 3
    assert (result == N_TIMESTEPS).all()


def test_count_valid_timesteps_with_missing():
    df = _make_df(n=2)
    df = replace_missing(df)
    df.loc[0, "VH_01"] = np.nan
    df.loc[0, "VH_02"] = np.nan
    result = count_valid_timesteps(df)
    assert result.iloc[0] == N_TIMESTEPS - 2
    assert result.iloc[1] == N_TIMESTEPS


def test_all_bands_constant():
    assert "VH" in SAR_BANDS
    assert "VV" in SAR_BANDS
    assert "nir" in OPT_BANDS
    assert "swir1" in OPT_BANDS
    assert len(ALL_BANDS) == len(SAR_BANDS) + len(OPT_BANDS)
