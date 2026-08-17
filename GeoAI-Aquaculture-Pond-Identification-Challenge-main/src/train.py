"""Training pipeline with LightGBM, XGBoost (+ optional CatBoost) and ensemble blending."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from src.evaluate import find_best_threshold, print_scores, zindi_score

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
MODEL_DIR = Path(__file__).parent.parent / "models"


# ---------------------------------------------------------------------------
# Default hyperparameters (tuned baseline)
# ---------------------------------------------------------------------------

LGB_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 2000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1086 / 735,  # neg/pos ratio from training data
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

CB_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.02,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "bagging_temperature": 0.5,
    "random_strength": 1.0,
    "border_count": 128,
    "auto_class_weights": "Balanced",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1,
}

N_SPLITS = 5
EARLY_STOPPING = 100


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "lgb",
    n_splits: int = N_SPLITS,
    params: dict | None = None,
) -> tuple[np.ndarray, list]:
    """Stratified K-Fold cross-validation.

    Returns:
        oof_proba: out-of-fold probabilities (n_samples,)
        models: list of trained model objects
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    models = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = _fit(X_tr, y_tr, X_val, y_val, model_type, params)
        oof[val_idx] = _predict_proba(model, X_val, model_type)
        score = zindi_score(y_val.values, (oof[val_idx] >= 0.5).astype(int), oof[val_idx])
        print(f"  Fold {fold}/{n_splits}  Zindi={score:.4f}")
        models.append(model)

    best_thr, best_f1 = find_best_threshold(y.values, oof)
    print(f"\nOOF (threshold=0.5):")
    print_scores(y.values, oof)
    print(f"\nOOF (optimal threshold={best_thr:.3f}):")
    print_scores(y.values, oof, threshold=best_thr)
    return oof, models, best_thr


def _fit(X_tr, y_tr, X_val, y_val, model_type: str, params: dict | None):
    p = (params or {})
    if model_type == "lgb":
        p = {**LGB_PARAMS, **p}
        model = lgb.LGBMClassifier(**p)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
    elif model_type == "xgb":
        p = {**XGB_PARAMS, **p}
        model = xgb.XGBClassifier(**p)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif model_type == "rf":
        p = {**RF_PARAMS, **p}
        # RF does not handle NaN → median imputation via Pipeline
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(**p)),
        ])
        model.fit(X_tr, y_tr)
    elif model_type == "cb":
        if not HAS_CATBOOST:
            raise ImportError("catboost not installed — run: uv add catboost")
        p = {**CB_PARAMS, **p}
        pool_tr  = cb.Pool(X_tr,  y_tr,  feature_names=list(X_tr.columns))
        pool_val = cb.Pool(X_val, y_val, feature_names=list(X_val.columns))
        model = cb.CatBoostClassifier(**p)
        model.fit(pool_tr, eval_set=pool_val, early_stopping_rounds=EARLY_STOPPING)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


def _predict_proba(model, X: pd.DataFrame, model_type: str) -> np.ndarray:
    if model_type == "cb":
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


def ensemble_predict(models: list, X: pd.DataFrame, model_type: str) -> np.ndarray:
    """Average predictions from all fold models."""
    preds = np.stack([_predict_proba(m, X, model_type) for m in models], axis=1)
    return preds.mean(axis=1)


def blend(probas: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted blend of multiple probability arrays."""
    if weights is None:
        weights = [1.0 / len(probas)] * len(probas)
    w = np.array(weights) / sum(weights)
    return sum(p * wi for p, wi in zip(probas, w))


def save_models(models: list, model_type: str, path: Path | None = None) -> Path:
    path = path or (MODEL_DIR / f"{model_type}_models.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)
    print(f"Saved {len(models)} models → {path}")
    return path


def load_models(path: Path) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import json
    from src.features import prepare_Xy

    print("Loading and engineering features …")
    X_train, y_train, X_test = prepare_Xy(
        str(DATA_DIR / "Train.csv"),
        str(DATA_DIR / "Test.csv"),
    )
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("\n=== LightGBM ===")
    lgb_oof, lgb_models, lgb_thr = cross_validate(X_train, y_train, model_type="lgb")

    print("\n=== XGBoost ===")
    xgb_oof, xgb_models, xgb_thr = cross_validate(X_train, y_train, model_type="xgb")

    print("\n=== Random Forest ===")
    rf_oof, rf_models, rf_thr = cross_validate(X_train, y_train, model_type="rf")

    # blend OOF of all three with weights; RF contributes diversity
    oof_list    = [lgb_oof, xgb_oof, rf_oof]
    w_list      = [3, 2, 2]
    blended_oof = blend(oof_list, weights=w_list)
    best_thr, best_f1 = find_best_threshold(y_train.values, blended_oof)
    print(f"\n=== Blended OOF LGB+XGB+RF (threshold=0.5) ===")
    print_scores(y_train.values, blended_oof)
    print(f"\n=== Blended OOF LGB+XGB+RF (optimal threshold={best_thr:.3f}) ===")
    print_scores(y_train.values, blended_oof, threshold=best_thr)

    save_models(lgb_models, "lgb")
    save_models(xgb_models, "xgb")
    save_models(rf_models,  "rf")

    # persist optimal thresholds for predict.py
    thr_path = MODEL_DIR / "thresholds.json"
    thr_path.parent.mkdir(parents=True, exist_ok=True)
    with open(thr_path, "w") as f:
        json.dump({
            "blend": best_thr,
            "lgb": lgb_thr,
            "xgb": xgb_thr,
            "rf":  rf_thr,
        }, f, indent=2)
    print(f"Thresholds saved → {thr_path}")
    print("Done.")
