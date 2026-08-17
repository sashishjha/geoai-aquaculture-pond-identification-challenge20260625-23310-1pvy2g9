# GeoAI Aquaculture Pond Identification Challenge by FAO and ITU
## Steps
### Import Libs
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

### Load Train Data Set
    train_df = pd.read_csv('Train.csv')

### Define features
    features = ['VH_01', 'VV_01', 'blue_01', 'green_01', 'nir_01', 'nira_01', 're1_01', 're2_01']

### Define X, Y sets
    X = train_df[features]
    y = train_df['label']

### Split Data for validating the prediction
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

## We are using Random Forest and Gradient Boosting
# Hyperparameter Tuning
## Random Forest
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    }

## Gradient Boost
    param_grid_gb = {
        'n_estimators': [100, 150],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }

# Calibrating and Training the models
## Random Forest
    rf = RandomForestClassifier(random_state=42)
    grid_rf = GridSearchCV(rf, param_grid_rf, scoring='roc_auc', cv=3, n_jobs=-1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    cal_rf = CalibratedClassifierCV(best_rf, method='sigmoid')
    cal_rf.fit(X_train, y_train)

## Gradient Boost
    gb = GradientBoostingClassifier(random_state=42)
    grid_gb = GridSearchCV(gb, param_grid_gb, scoring='roc_auc', cv=3, n_jobs=-1)
    grid_gb.fit(X_train, y_train)
    best_gb = grid_gb.best_estimator_

### Get Probability on Validating Sets
    prob_rf = cal_rf.predict_proba(X_val)[:, 1]
    prob_gb = best_gb.predict_proba(X_val)[:, 1]
    ensemble_probs = (prob_rf + prob_gb) / 2

### Optimize Threshold
    thresholds = np.linspace(0, 1, 101)
    best_thresh = 0.5
    best_auc = 0
    for t in thresholds:
        preds = (ensemble_probs >= t).astype(int)
        score = roc_auc_score(y_val, preds)
        if score > best_auc:
            best_auc = score
            best_thresh = t

    print(f"Optimal threshold: {best_thresh:.2f} with validation ROC-AUC: {best_auc:.4f}")

### Retrain the data with full data
    cal_rf.fit(X, y)
    best_gb.fit(X, y)

### Load Test Data set
    test_df = pd.read_csv('Test.csv')

### Define feature
    X_test = test_df[features]

### Get test Probabilities
    prob_rf_test = cal_rf.predict_proba(X_test)[:, 1]
    prob_gb_test = best_gb.predict_proba(X_test)[:, 1]
    ensemble_test_probs = (prob_rf_test + prob_gb_test) / 2

### FInal Prediction
    final_preds = (ensemble_test_probs >= best_thresh).astype(int)

### Save Submission
    sample_sub = pd.read_csv('SampleSubmission.csv')
    submission = pd.DataFrame({
        'ID': sample_sub['ID'],
        'TargetF1': final_preds,  # Binary decision
        'TargetRAUC': ensemble_test_probs  # Full probability scores for maximum power
    })

    submission.to_csv('submission_powerful_prob_scores.csv', index=False)

    print("Powerful submission created: 'submission_powerful_prob_scores.csv'")
