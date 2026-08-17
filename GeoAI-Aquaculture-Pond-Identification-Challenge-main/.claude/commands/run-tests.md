# Run Unit Tests

Run the full test suite (35 tests across preprocessing, features, evaluation).

```bash
cd /Users/clementkm/Documents/PROJECTS/Challenge/GeoAI_Aquaculture_Pond_Identification_Challenge_by_FAO_and_ITU && PYTHONPATH=. python3 -m pytest tests/ -v
```

Test coverage:
- `tests/test_preprocessing.py` — data loading, NaN replacement, band matrix extraction
- `tests/test_features.py` — spectral index formulas, temporal stats, build_features()
- `tests/test_evaluate.py` — Zindi score formula, F1, ROC-AUC
