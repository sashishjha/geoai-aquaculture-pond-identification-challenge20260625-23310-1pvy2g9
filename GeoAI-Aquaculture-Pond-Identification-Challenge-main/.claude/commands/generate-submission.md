# Generate Submission

Load saved models and generate the Zindi submission CSV.

```bash
cd /Users/clementkm/Documents/PROJECTS/Challenge/GeoAI_Aquaculture_Pond_Identification_Challenge_by_FAO_and_ITU && PYTHONPATH=. python3 src/predict.py
```

Saves to `data/submissions/lgb_xgb_blend.csv` with columns `ID`, `TargetF1` (binary), `TargetRAUC` (probability).

**Prerequisites**: run `/train-model` first to generate the model files in `models/`.
