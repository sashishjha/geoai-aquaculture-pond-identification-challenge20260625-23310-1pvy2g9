# CLAUDE.md — GeoAI Aquaculture Pond Identification Challenge

## Project Overview

Zindi competition to classify aquaculture ponds from Sentinel-1/2 satellite time series.  
**Score = 0.60 × F1 + 0.40 × ROC-AUC**  
Deadline: **2026-08-16**

## Key Commands

```bash
# Run all tests
PYTHONPATH=. python3 -m pytest tests/ -v

# Train models (LGB + XGB, saves to models/)
PYTHONPATH=. python3 src/train.py

# Generate submission (reads models/, writes data/submissions/)
PYTHONPATH=. python3 src/predict.py
```

## Architecture

```
src/preprocessing.py  → load CSV, replace -9999 → NaN, band matrix utilities
src/features.py       → compute spectral indices per timestep, aggregate over months
src/evaluate.py       → zindi_score(), print_scores()
src/train.py          → cross_validate(), blend(), save_models()
src/predict.py        → ensemble_predict(), make_submission()
```

## Data

- Train: 1821 rows, 144 features (12 months × 12 bands) + `label`
- Test: 1031 rows, same features, no `label`, some with only 4-6 valid months
- Missing = -9999 → replaced with NaN, then handled by LightGBM natively
- Positive class (pond): 40.4% of train

## Feature Engineering (`src/features.py`)

Indices computed per timestep (n_samples × 12 matrix), then mean/std/min/max/range extracted:
- Water: NDWI, AWEInsh, AWEIsh
- Vegetation: NDVI, EVI, SAVI
- Soil/RE: BSI, NDRE, CIG
- SAR: VH-VV diff, SAR RVI
- Raw bands: all 12 bands with temporal stats
- Per-timestep VH/VV values directly

## Submission Format

```csv
ID,TargetF1,TargetRAUC
ID_TS_NEW_XXX,0,0.12     # TargetF1=binary, TargetRAUC=probability
```

## Available Skills

- `/train-model` — run cross-validation and save models
- `/generate-submission` — load models and predict on test set
- `/run-eda` — launch EDA notebook

## Coding Conventions

- All code in `src/` uses type hints
- Missing values: always replace -9999 with `np.nan` via `replace_missing()`
- Tests in `tests/`, run with `PYTHONPATH=. python3 -m pytest tests/`
- Never commit `models/`, `data/processed/`, `data/submissions/`
