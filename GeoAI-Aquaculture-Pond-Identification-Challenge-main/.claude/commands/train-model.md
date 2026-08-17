# Train Model

Run the full training pipeline: feature engineering → 5-fold LightGBM + XGBoost → save models.

```bash
cd /Users/clementkm/Documents/PROJECTS/Challenge/GeoAI_Aquaculture_Pond_Identification_Challenge_by_FAO_and_ITU && PYTHONPATH=. python3 src/train.py
```

Reports per-fold and overall OOF Zindi Score (0.60 × F1 + 0.40 × ROC-AUC).  
Models saved to `models/lgb_models.pkl` and `models/xgb_models.pkl`.
