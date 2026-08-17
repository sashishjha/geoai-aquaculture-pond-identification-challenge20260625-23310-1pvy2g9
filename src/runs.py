"""
runs.py — Run registry. Appends one row per experiment to runs.csv.
Every run is identified by its config hash and timestamp.
"""
from __future__ import annotations
import csv, os, time
from typing import Optional


RUNS_PATH = "experiments_checkpoint/runs.csv"

COLS = [
    "run_id", "config_hash", "timestamp",
    "proxy_score", "proxy_f1", "proxy_auc",
    "oof_auc", "oof_f1",
    "test_pos_rate", "n_unique_scores",
    "submitted", "lb_score", "lb_auc", "lb_f1",
    "notes",
]


def log_run(
    config_hash: str,
    proxy_score: Optional[float] = None,
    proxy_f1: Optional[float] = None,
    proxy_auc: Optional[float] = None,
    oof_auc: Optional[float] = None,
    oof_f1: Optional[float] = None,
    test_pos_rate: Optional[float] = None,
    n_unique_scores: Optional[int] = None,
    submitted: bool = False,
    lb_score: Optional[float] = None,
    lb_auc: Optional[float] = None,
    lb_f1: Optional[float] = None,
    notes: str = "",
) -> str:
    """Append one row to runs.csv. Returns the run_id."""
    os.makedirs(os.path.dirname(RUNS_PATH), exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_id = f"{config_hash[:8]}_{ts}"

    write_header = not os.path.exists(RUNS_PATH)
    with open(RUNS_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if write_header:
            w.writeheader()
        w.writerow({
            "run_id": run_id,
            "config_hash": config_hash,
            "timestamp": ts,
            "proxy_score": round(proxy_score, 6) if proxy_score is not None else "",
            "proxy_f1": round(proxy_f1, 6) if proxy_f1 is not None else "",
            "proxy_auc": round(proxy_auc, 6) if proxy_auc is not None else "",
            "oof_auc": round(oof_auc, 6) if oof_auc is not None else "",
            "oof_f1": round(oof_f1, 6) if oof_f1 is not None else "",
            "test_pos_rate": round(test_pos_rate, 6) if test_pos_rate is not None else "",
            "n_unique_scores": n_unique_scores if n_unique_scores is not None else "",
            "submitted": int(submitted),
            "lb_score": round(lb_score, 6) if lb_score is not None else "",
            "lb_auc": round(lb_auc, 6) if lb_auc is not None else "",
            "lb_f1": round(lb_f1, 6) if lb_f1 is not None else "",
            "notes": notes,
        })
    print(f"[runs] logged run_id={run_id}")
    return run_id
