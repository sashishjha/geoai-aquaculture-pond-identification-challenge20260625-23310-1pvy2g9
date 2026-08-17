"""Generate submission file from trained models."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR   = Path(__file__).parent.parent / "data" / "raw"
MODEL_DIR  = Path(__file__).parent.parent / "models"
SUBMIT_DIR = Path(__file__).parent.parent / "data" / "submissions"


def load_models(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba(models: list, X: pd.DataFrame) -> np.ndarray:
    preds = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    return preds.mean(axis=1)


def make_submission(
    test_ids: pd.Index,
    proba: np.ndarray,
    threshold: float = 0.5,
    tag: str = "submission",
) -> pd.DataFrame:
    sub = pd.DataFrame({
        "ID": test_ids,
        "TargetF1": (proba >= threshold).astype(int),
        "TargetRAUC": proba,
    })
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBMIT_DIR / f"{tag}.csv"
    sub.to_csv(path, index=False)
    print(f"Submission saved → {path}  ({sub['TargetF1'].sum()} positives / {len(sub)} total)")
    return sub


def load_threshold(key: str = "blend", default: float = 0.5) -> float:
    """Load the OOF-optimised threshold saved by train.py."""
    path = MODEL_DIR / "thresholds.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        thr = float(data.get(key, default))
        print(f"Loaded threshold [{key}] = {thr:.4f}")
        return thr
    print(f"No thresholds.json found, using default={default}")
    return default


def generate_all_submissions(
    X_test: "pd.DataFrame",
    tag: str = "v2",
) -> dict:
    """Load all saved models, generate per-model and blended submissions.

    Produces up to 5 CSV files:
        {tag}_lgb.csv, {tag}_xgb.csv, {tag}_rf.csv,
        {tag}_cb.csv  (if available),
        {tag}_blend.csv
    Returns a dict of {name: proba_array}.
    """
    from src.train import blend

    model_files = {
        "lgb": MODEL_DIR / "lgb_models.pkl",
        "xgb": MODEL_DIR / "xgb_models.pkl",
        "rf":  MODEL_DIR / "rf_models.pkl",
        "cb":  MODEL_DIR / "cb_models.pkl",
    }

    probas: dict = {}
    for name, path in model_files.items():
        if path.exists():
            models = load_models(path)
            proba  = predict_proba(models, X_test)
            probas[name] = proba
            thr = load_threshold(name)
            make_submission(X_test.index, proba, threshold=thr, tag=f"{tag}_{name}")
        else:
            print(f"  [skip] {name}: {path.name} not found")

    if len(probas) < 2:
        print("Not enough models to blend.")
        return probas

    # equal-weight blend of whatever models are available
    names = list(probas.keys())
    weights_map = {"lgb": 3, "xgb": 2, "rf": 2, "cb": 2}
    weights = [weights_map.get(n, 1) for n in names]
    blended = blend([probas[n] for n in names], weights=weights)
    thr = load_threshold("blend")
    make_submission(X_test.index, blended, threshold=thr, tag=f"{tag}_blend")
    probas["blend"] = blended

    return probas


if __name__ == "__main__":
    from src.features import prepare_Xy

    print("Engineering features …")
    _, _, X_test = prepare_Xy(
        str(DATA_DIR / "Train.csv"),
        str(DATA_DIR / "Test.csv"),
    )

    print("\nGenerating submissions for all saved models …")
    generate_all_submissions(X_test, tag="v2")
    print("Done.")
