"""experiment_logger.py — append one row per run to experiment_log.csv (§23)."""
from __future__ import annotations
import os, csv

COLUMNS = ["experiment_id", "model", "features", "preprocessing", "hyperparameters",
           "validation_strategy", "seed", "F1", "ROC_AUC", "weighted_score", "notes"]


def log_experiment(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})
    print(f"[log] appended '{row.get('experiment_id')}' -> {path}")
