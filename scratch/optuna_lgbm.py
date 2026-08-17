import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, 'src')
import common as C
from config import RunConfig

def apply_shift(p, train_prev=0.404, test_prev=0.6175):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    odds = p / (1 - p)
    shift = (test_prev / (1 - test_prev)) / (train_prev / (1 - train_prev))
    new_odds = odds * shift
    return new_odds / (1 + new_odds)

def main():
    print("Loading data...")
    cfg = RunConfig.load("config_s6.json")
    tr, _ = C.load_raw('data_checkpoint/raw/Train.csv', 'data_checkpoint/raw/Test.csv')
    cube_tr = C.band_cube(tr)
    
    rng = np.random.default_rng(42)
    cube_tr_masked, y_tr = C.augment_training_data(cube_tr, tr['label'].values, n_copies=1, seed=42)
    cube_tr_masked = cube_tr_masked[:len(cube_tr)]
    y_tr = y_tr[:len(cube_tr)]
    
    print("Building features...")
    X_train = C.build_features(cube_tr_masked, cfg=cfg)
    
    # We must enforce test_like_idx strictly
    test_like_idx = np.load('data_checkpoint/dpp_test_like_idx.npy')
    
    groups = tr.index // 25
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'none',
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': pos_w,
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': 4
        }
        
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y_tr))
        
        for tr_i, va_i in skf.split(X_train, y_tr, groups=groups):
            X_tr, y_tr_fold = X_train.iloc[tr_i], y_tr[tr_i]
            X_va = X_train.iloc[va_i]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr_fold)
            
            oof_preds[va_i] = model.predict_proba(X_va)[:, 1]
            
        p_calib = apply_shift(oof_preds, 0.404, 0.6175)
        
        y_test_like = y_tr[test_like_idx]
        p_test_like = p_calib[test_like_idx]
        
        pred_binary = (p_test_like >= 0.5).astype(int)
        
        if len(np.unique(y_test_like)) < 2: 
            return 0.0
            
        f1 = f1_score(y_test_like, pred_binary)
        auc = roc_auc_score(y_test_like, p_test_like)
        
        dpp_score = 0.6 * f1 + 0.4 * auc
        return dpp_score

    print("Starting bounded Optuna tune (20 trials max) on DPP Metric...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    print("\n=== OPTUNA TUNE COMPLETE ===")
    print(f"Best DPP Score: {study.best_value:.5f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
if __name__ == "__main__":
    main()
