# GeoAI Aquaculture Pond Identification Challenge
**Zindi · FAO · ITU** — June–August 2026

Identify aquaculture ponds from multi-temporal satellite imagery.  
Final Score = **0.60 × F1** + **0.40 × ROC-AUC**

## Problem

Classify 10 m × 10 m ground patches as aquaculture ponds (1) or other land cover (0).  
Data are 12 monthly composites of Sentinel-1 (SAR) and Sentinel-2 (optical) bands.  
Models trained on one time period must generalise to a different temporal window — temporal robustness is critical.

## Data

| File | Rows | Description |
|------|------|-------------|
| `data/raw/Train.csv` | 1 821 | Training samples with `label` column |
| `data/raw/Test.csv` | 1 031 | Test samples (missing `label`) |
| `data/raw/SampleSubmission.csv` | 1 031 | Expected submission format |

**Columns**: `ID`, `label` (train only), then `{band}_{t:02d}` for `t` ∈ 1..12.

| Band group | Bands | Sensor |
|------------|-------|--------|
| SAR | `VH`, `VV` | Sentinel-1 (dB) |
| Optical | `blue`, `green`, `red`, `re1`, `re2`, `re3`, `nir`, `nira`, `swir1`, `swir2` | Sentinel-2 (×10 000) |

Missing values encoded as **-9999** (cloud cover / no acquisition).

## Project Structure

```
.
├── data/
│   ├── raw/            # Train.csv, Test.csv, SampleSubmission.csv
│   ├── processed/      # Engineered features (gitignored)
│   └── submissions/    # Generated CSVs (gitignored)
├── notebook/
│   ├── 01_EDA.ipynb               # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Spectral indices + importance
│   └── 03_modeling.ipynb          # LightGBM, XGBoost, SHAP, submission
├── src/
│   ├── preprocessing.py  # Data loading, NaN handling
│   ├── features.py       # Spectral index computation & temporal aggregation
│   ├── evaluate.py       # Zindi scoring metric
│   ├── train.py          # Cross-validation pipeline
│   └── predict.py        # Submission generation
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_evaluate.py
├── .claude/commands/     # Claude Code skills
└── pyproject.toml
```

## Feature Engineering

Spectral indices computed from [awesome-spectral-indices](https://github.com/awesome-spectral-indices/awesome-spectral-indices), per time step then aggregated (mean / std / min / max / range):

| Index | Formula | Signal |
|-------|---------|--------|
| NDWI | (Green − SWIR1) / (Green + SWIR1) | Open water → positive |
| AWEInsh | 4(G−S1) − 0.25·N + 2.75·S2 | Water extraction |
| AWEIsh | B + 2.5G − 1.5(N+S1) − 0.25S2 | Water + shadow removal |
| NDVI | (NIR − Red) / (NIR + Red) | Vegetation → negative for water |
| EVI | 2.5·(N−R)/(N+6R−7.5B+10000) | Enhanced vegetation |
| SAVI | 1.5·(N−R)/(N+R+5000) | Soil-adjusted vegetation |
| BSI | ((S1+R)−(N+B))/((S1+R)+(N+B)) | Bare soil |
| NDRE | (NIR − RE1) / (NIR + RE1) | Red-edge chlorophyll |
| CIG | NIR/Green − 1 | Chlorophyll green |
| SAR diff | VH − VV | Polarisation ratio (dB) |
| SAR RVI | 4·VH_lin/(VV_lin+VH_lin) | Radar vegetation index |

## Quickstart

```bash
# Install dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/ -v

# Train models (from project root)
python src/train.py

# Generate submission
python src/predict.py
```

## Evaluation

```python
from src.evaluate import zindi_score
score = zindi_score(y_true, y_pred_binary, y_pred_proba)
# score = 0.60 * F1 + 0.40 * ROC-AUC
```

## Trustworthiness (for top-10 submission)

See the notebook `03_modeling.ipynb` — Trustworthiness Notes section — covering:
1. **Data & Model Bias**: class imbalance handling, regional/temporal bias discussion
2. **Model Transparency**: SHAP analysis, key feature insights
3. **Approach Reusability**: band-agnostic pipeline adaptable to other tasks
4. **Sustainability**: lightweight gradient boosting, CodeCarbon integration

## Timeline

| Date | Milestone |
|------|-----------|
| 2026-06-08 | Competition start |
| 2026-08-07 | Enrollment closes |
| 2026-08-16 | **Challenge closes** |
