"""Data loading and preprocessing for the aquaculture pond challenge."""

import numpy as np
import pandas as pd

MISSING = -9999
N_TIMESTEPS = 12

SAR_BANDS = ["VH", "VV"]
OPT_BANDS = ["blue", "green", "nir", "nira", "re1", "re2", "re3", "red", "swir1", "swir2"]
ALL_BANDS = SAR_BANDS + OPT_BANDS


def _band_col(band: str, t: int) -> str:
    return f"{band}_{t:02d}"


def load_train(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, index_col="ID")
    y = df.pop("label")
    return df, y


def load_test(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col="ID")


def replace_missing(df: pd.DataFrame, missing_val: float = MISSING) -> pd.DataFrame:
    """Replace sentinel missing values with NaN."""
    return df.replace(missing_val, np.nan)


def get_band_matrix(df: pd.DataFrame, band: str) -> np.ndarray:
    """Return (n_samples, N_TIMESTEPS) array for one band, NaN where missing."""
    cols = [_band_col(band, t) for t in range(1, N_TIMESTEPS + 1)]
    return df[cols].values.astype(float)


def count_valid_timesteps(df: pd.DataFrame) -> pd.Series:
    """Per-sample count of timesteps with at least one valid SAR observation."""
    sar_cols = [_band_col(b, t) for b in SAR_BANDS for t in range(1, N_TIMESTEPS + 1)]
    valid = ~df[sar_cols].isna()
    sar_valid = valid[[_band_col("VH", t) for t in range(1, N_TIMESTEPS + 1)]]
    return sar_valid.sum(axis=1).rename("n_valid_timesteps")
