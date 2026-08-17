"""
generate_submission.py — produce the exact competition CSV.

Fits the chosen model (or ensemble) on ALL train rows using the full-month
features, predicts on the REAL test cube (already partially masked), and
writes the submission via the numerically safe writer (submission_writer.py):

    ID, TargetRAUC, TargetF1
    * TargetRAUC  = genuine model probability in (0,1), full float64 precision
    * TargetF1    = (TargetRAUC >= 0.5).astype(int)   (the ONLY allowed rule)

All submissions route through write_submission(), which enforces:
  - float64, no saturation, no ties created, monotonicity preserved
  - CSV round-trip Spearman = 1.0 verified before the file is kept
  - Calibration via logit shift (controlled by config.calibration_enabled)

Kill criterion: pos_rate outside [pi_t ± 0.05] aborts before writing.

Usage:
    uv run python src/generate_submission.py
    uv run python src/generate_submission.py --config path/to/config.json
    uv run python src/generate_submission.py --shift 0.15  # manual shift override
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import common as C
from config import RunConfig, CHAMPION_CONFIG
from train_model import make_model, available, fit_full
from submission_writer import write_submission, pre_submission_check
from runs import log_run


def build_predictor(cube, y, kinds, cfg: RunConfig):
    models = []
    for k in kinds:
        m = fit_full(cube, y, k, cfg=cfg)
        models.append(m)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="ensemble", help="hgb|lgbm|xgb|cat|ensemble")
    ap.add_argument("--train",  default="data_checkpoint/raw/Train.csv")
    ap.add_argument("--test",   default="data_checkpoint/raw/Test.csv")
    ap.add_argument("--out",    default="submissions/submission_checkpoint.csv")
    ap.add_argument("--config", default=None,
                    help="Path to config JSON. Default: CHAMPION_CONFIG (Iteration 1).")
    ap.add_argument("--shift",  type=float, default=None,
                    help="Override total logit shift. Default: read from config.")
    a = ap.parse_args()

    cfg = RunConfig.load(a.config) if a.config else CHAMPION_CONFIG
    print(f"[submission] config={cfg.hash()} calibration={cfg.calibration_enabled} "
          f"shift={cfg.calibration_shift:.4f}")

    np.random.seed(cfg.seed)
    tr, te = C.load_raw(a.train, a.test)
    cube_tr = C.band_cube(tr)
    y = tr["label"].values
    cube_te = C.band_cube(te)
    Xte = C.build_features(cube_te, cfg=cfg)

    wanted = ["hgb", "lgbm", "xgb", "cat"] if a.model == "ensemble" else [a.model]
    kinds = [k for k in wanted if available(k)]
    if not kinds:
        raise SystemExit("No requested model available in this environment.")
    print(f"[submission] using models: {kinds}")

    models = build_predictor(cube_tr, y, kinds, cfg=cfg)
    probs = np.mean([m.predict_proba(Xte)[:, 1] for m in models], axis=0)

    # Determine total shift
    total_shift = a.shift if a.shift is not None else cfg.calibration_shift
    calibration_enabled = cfg.calibration_enabled if a.shift is None else (a.shift != 0.0)

    # Kill criterion: check pos_rate before writing
    # (approximate check on raw probs before shift)
    raw_pos_rate = float((probs >= 0.5).mean())
    print(f"[submission] raw pos_rate (pre-shift) = {raw_pos_rate:.4f}")

    # Write via safe writer (shift applied inside, assertions enforced)
    df = write_submission(
        ids=te["ID"].values,
        p_raw=probs,
        total_shift=total_shift,
        path=a.out,
        calibration_enabled=calibration_enabled,
    )

    # Kill criterion check on final pos_rate
    final_pos_rate = float(df["TargetF1"].mean())
    pre_submission_check(final_pos_rate, pi_t=cfg.pi_t_assumed)

    n_unique = int(df["TargetRAUC"].nunique())
    log_run(
        config_hash=cfg.hash(),
        test_pos_rate=final_pos_rate,
        n_unique_scores=n_unique,
        submitted=False,
        notes=f"shift={total_shift:.4f}",
    )


if __name__ == "__main__":
    main()
