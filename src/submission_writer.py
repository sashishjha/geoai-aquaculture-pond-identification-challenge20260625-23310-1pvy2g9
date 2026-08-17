"""
submission_writer.py — Numerically safe, full-precision, assertion-guarded
submission writer. Every submission routes through write_submission().

Implements Phase 5, Step 3 of the Iteration 5 Roadmap:
  - float64 throughout, never saturates to exactly 0/1
  - Clips BEFORE logit, never after sigmoid
  - Asserts monotonicity, no tie creation, no saturation
  - Round-trip verifies the CSV (catches locale/dtype quirks in to_csv)
  - Config flag: calibration_enabled=False passes probabilities through untouched
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def write_submission(
    ids: np.ndarray,
    p_raw: np.ndarray,
    total_shift: float,
    path: str,
    calibration_enabled: bool = True,
) -> pd.DataFrame:
    """
    Write a competition-ready submission CSV.

    Parameters
    ----------
    ids            : array of ID strings, length N
    p_raw          : raw model probabilities, length N, any float dtype
    total_shift    : total logit shift to apply (0.0 = no calibration)
    path           : output CSV path
    calibration_enabled : if False, shift is forced to 0.0 regardless of total_shift
                          (config flag for rule-compliance contingency)

    Returns
    -------
    DataFrame with columns [ID, TargetRAUC, TargetF1]
    """
    # ── Config flag: disable calibration with one switch ─────────────────────
    if not calibration_enabled:
        total_shift = 0.0

    p = np.asarray(p_raw, dtype=np.float64)
    n = len(p)

    # Clip BEFORE logit, never after sigmoid — clipping after sigmoid creates
    # saturation at exactly 0 or 1 which destroys AUC via tied pairs.
    eps = 1e-12
    p_clipped = np.clip(p, eps, 1.0 - eps)

    # Apply shift in log-odds space (constant additive offset → strictly monotone)
    z = np.log(p_clipped / (1.0 - p_clipped)) + float(total_shift)
    out = 1.0 / (1.0 + np.exp(-z))  # float64, no saturation

    # ── Assertions: fail loudly before writing ────────────────────────────────
    assert out.dtype == np.float64, f"dtype is {out.dtype}, expected float64"
    assert np.all(np.isfinite(out)), "non-finite values in output"
    assert out.min() > 0.0 and out.max() < 1.0, \
        f"saturation detected: min={out.min()}, max={out.max()}"

    # ── Build DataFrame in required column order ──────────────────────────────
    df = pd.DataFrame({
        "ID": ids,
        "TargetRAUC": out,
        "TargetF1": (out >= 0.5).astype(int),
    })
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, float_format="%.17f")

    # ── Round-trip verification ───────────────────────────────────────────────
    rt = pd.read_csv(path)
    rt_rauc = rt["TargetRAUC"].values

    spear2 = spearmanr(out, rt_rauc).statistic
    assert abs(spear2 - 1.0) < 1e-7, \
        f"CSV round-trip monotonicity broken: Spearman = {spear2:.10f}"

    # ── Pre-submission report ─────────────────────────────────────────────────
    pos_rate = df["TargetF1"].mean()
    print(f"[submission] written: {path}")
    print(f"  rows={n}, pos_rate={pos_rate:.4f}, shift={total_shift:.6f}")
    print(f"  n_unique_scores={len(np.unique(out))}/{n}")
    print(f"  min={out.min():.8f}, max={out.max():.8f}")
    print(f"  round-trip Spearman={spear2:.10f} ✓")

    return df


def pre_submission_check(pos_rate: float, pi_t: float, tol: float = 0.05) -> None:
    """Kill criterion: abort if predicted positive rate is outside [pi_t ± tol]."""
    if abs(pos_rate - pi_t) > tol:
        raise RuntimeError(
            f"[KILL CRITERION] Predicted positive rate {pos_rate:.4f} deviates from "
            f"assumed pi_t={pi_t:.4f} by {abs(pos_rate - pi_t):.4f} > tol={tol}. "
            f"Do not submit — diagnose the score distribution first."
        )
    print(f"[pre-submission check] pos_rate={pos_rate:.4f} within ±{tol} of pi_t={pi_t:.4f} ✓")
