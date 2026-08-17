# Decisions Log (condensed — one line per major decision)

- **Treat `-9999` as missing sentinel, never a value** — verified train has 0,
  test has 89,756; no real band value can equal −9999.
- **Window length defined by S1 presence; S2 can be cloud-masked within window**
  — 320 S2-only masked cells found; drives per-sensor aggregation design.
- **Aggregate every feature over VALID months only, per sensor** — guarantees
  invariance to arbitrary 4–6 month windows (§6/§19).
- **Encode missingness as features** — window_length, s2_cloud_in_window,
  per-sensor valid counts, s2_avail_ratio (missingness carries signal, §7).
- **Robust validation via masking simulation** — each CV valid fold is masked to
  4/5/6 consecutive months + ~7% S2 cloud, matching the real test regime (§17).
- **Keep a conventional CV too** — as an (optimistic) reference point only.
- **Tree models with native NaN handling** (HistGBM/LightGBM/XGBoost/CatBoost);
  no mean imputation that would erase the missingness signal (§21).
- **Class imbalance via class weighting / scale_pos_weight only** — never via
  threshold tuning (§9/§11).
- **0.5 threshold hard-locked** — `TargetF1 = (proba>=0.5).astype(int)`, applied
  only at submission time; thresholds explored for diagnostics never feed the
  submission (§11). NON-NEGOTIABLE.
- **Deduplicate train** — dropped 4 exact positive-label duplicate pairs to avoid
  cross-fold leakage. IDs: YWUURQ8CBZ, R54MVWFXYV, ZL3STYKTP5, X36W5R66JK,
  U49A985LTS, NCN3CCHMF7, 5MHPUMP3S9, MFUA7NQNDC.
- **No external data, no lat/long reconstruction, no old test set** (§2/§8/§15).
- **Seeds fixed everywhere (SEED=42)** — Python/NumPy/model/split/masking-rng.
- **CatBoost optional** — code degrades gracefully if not installed; ensemble
  uses whichever of hgb/lgbm/xgb/cat are available.
- **"EU Adjustments" not found** — flagged, not coded against (§26).
