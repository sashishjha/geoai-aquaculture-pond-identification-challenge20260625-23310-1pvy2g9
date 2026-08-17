# handoff.md — Aquaculture Pond Identification (single consolidated resume file)

## Current §27 workflow step
**Steps 1–11 complete** (inspect → audit → masking understood → baseline +
candidate models → robust validation → submission generated → compliance audit).
**Step 12 (documentation)** drafted in `docs_checkpoint/phase2_writeup.md`.
Next action for the human: **run the real training on your machine/cluster and
submit `submissions/submission_checkpoint.csv` to Zindi** (or regenerate it with
CatBoost added to the ensemble, which was unavailable in the build sandbox).

## What's been verified (from the real files, not assumed)
- Train 1,817 rows after dedup (1,821 − 4 dup positive pairs), 138 engineered
  features; Test 1,030 rows.
- Train fully observed (0 × −9999). Test = consecutive **4/5/6-month** windows
  (345/343/342). Windows defined by **S1 presence**; **S2 additionally
  cloud-masked** on 320 in-window months (S1 never masked with S2 present).
- No train/test exact-row overlap; no duplicate IDs; no lat/long; no region col.
- **Real CV numbers (LightGBM, seed 42):**
  - Conventional 5-fold: F1 0.9759 / AUC 0.9969 / **score 0.9843** (optimistic).
  - Robust masking 5-fold: F1 0.9008 / AUC 0.9748 / **score 0.9304** (±0.0028) —
    the realistic private-test estimate.
- Submission generated (hgb+lgbm+xgb ensemble): 1,030 rows, cols exactly
  `ID,TargetF1,TargetRAUC`, TargetF1∈{0,1} via 0.5, RAUC∈[0,1], 0 NaN,
  test pos_rate 0.476 (no threshold tuning).

## What's still open
- CatBoost not installed in sandbox → ensemble used hgb+lgbm+xgb. Add catboost
  and rerun `generate_submission.py --model ensemble` if desired.
- `train_baseline.py` / `train_model.py` will populate the blank experiment-log
  rows with exact HistGBM/ensemble numbers when you run them locally.
- Optional: SHAP figures for the Phase-Two transparency section.

## Exact next action
1. `pip install -r requirements.txt` (add `catboost` for the 4th member).
2. `python src/train_baseline.py` then `python src/train_model.py --model ensemble`.
3. `python src/generate_submission.py --model ensemble --out submissions/submission_checkpoint.csv`.
4. Submit to Zindi (5/day, 100 total). Select 2 for private LB before close.

## Condensed decisions log (see docs_checkpoint/decisions_log.md for full)
- −9999 = missing sentinel, never a value.
- Window length set by S1; S2 cloud-masked within window → per-sensor aggregation.
- All features aggregate over VALID months only → window-invariant (4/5/6 safe).
- Missingness encoded as features (window_length, s2_cloud_in_window, counts, ratio).
- Robust validation = simulate test masking on train valid folds. Keep conventional as reference.
- Tree models, native NaN, class weighting. **0.5 threshold hard-locked** (§11).
- Dedup 4 positive pairs. Seeds=42 everywhere. No external data / no lat-long / no old test set.
- "EU Adjustments" not found → flagged, not coded.

## Sync reminder
Before/after any cluster run or checkpoint export, trigger **Ctrl+Cmd+S** to
sync to `sashishj@10.72.30.28:~/projects/...`. All persisted folders already use
`*checkpoint*` / `submissions/` / `outputs/` / `logs/` names (rsync-safe).
