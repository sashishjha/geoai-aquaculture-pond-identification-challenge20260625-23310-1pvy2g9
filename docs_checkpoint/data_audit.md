# Data Audit — GeoAI Aquaculture Pond Identification (FAO × ITU)

All findings below were produced by inspecting the **actual** supplied
`Train.csv` / `Test.csv` (not assumed from the brief). Verified 2026-08-10.

## 1. Shape & schema
| File | Rows | Cols | Notes |
|------|------|------|-------|
| Train.csv | 1,821 | 146 | `ID`, `label`, then 144 band-month cols |
| Test.csv  | 1,030 | 145 | `ID`, then 144 band-month cols (no `label`) |

- **ID column:** `ID` (train IDs `ID_TR_NEW_*`, test IDs `ID_TS_NEW_*`).
- **Target:** `label` ∈ {0,1}. Present in train only.
- **Feature naming:** `{band}_{MM}`, `MM` ∈ `01..12`.
- **Bands (12):** S1/SAR = `VH, VV`; S2/optical = `blue, green, nir, nira,
  re1, re2, re3, red, swir1, swir2` (`nira` = narrow NIR). 12 months × 12 bands
  = 144 columns. Confirmed exactly as the brief's §5 schema.

## 2. Target distribution
- Train: **label=0 → 1,086 (59.6%)**, **label=1 → 735 (40.4%)**. ~40% positive.
- Test labels absent by design. Class balance handled via class weighting only;
  **the 0.5 decision threshold is never tuned** (competition rule, §11).

## 3. Missingness / `-9999` sentinel  — KEY FINDINGS
- **Train has ZERO `-9999` cells.** Every train row is fully observed across all
  12 months. (Total `-9999` in train = 0.)
- **Test has 89,756 `-9999` cells.** Per-row the observed months form a
  **consecutive window of exactly 4, 5, or 6 months** (dist: 4→345, 5→343,
  6→342). All windows verified contiguous (0 non-consecutive rows).
- **Refinement of the brief's §5 "12-column block" assumption (FLAGGED):** masking
  is *not* purely whole-month blocks. There are **320 (row,month) cells where
  Sentinel-2 is masked while Sentinel-1 (VH/VV) is present** — i.e. cloud cover
  removes optical data while radar survives. There are **0** cases of S1 masked
  with S2 present. So:
  - The **window length (4/5/6) is defined by Sentinel-1 presence.**
  - **Within** the window, S2 can be additionally cloud-masked on some months.
  - Per-sensor valid-month counts differ: S1 = {4,5,6}; S2 = {2,3,4,5,6}.
  This validates the S1/S2 complementarity in §4/§7/§20 and is the single most
  important structural fact for feature engineering.

## 4. Train/test distribution shift
- The core shift is **observation coverage**, not the band value ranges:
  train = full 12-month history; test = a short arbitrary window. A model that
  leans on any specific month (e.g. summer) will fail on windows that exclude it
  (the challenge's explicit intent, §6).
- **Mitigation:** all features aggregate over *valid months only* (per sensor),
  and validation **simulates the test masking on train rows** (see below).

## 5. Duplicates & leakage
- **Duplicate feature-rows in train:** 8 rows = 4 exact pairs, **all label=1**
  (no label conflict). Dropped `keep=first` to prevent the same observation
  appearing in two CV folds. IDs recorded in `decisions_log.md`.
- **Duplicate IDs:** none in train or test.
- **Train/Test exact overlap:** 0 rows share an identical full feature vector →
  no train/test duplication leakage to exploit (and we would flag, not exploit,
  any such leakage per §15).

## 6. Value ranges (sanity)
- S1 dB values negative (e.g. `VH` ≈ −48…−9), S2 reflectance positive integers
  (e.g. `blue` ≈ 983…4277). No plausible real value equals −9999, confirming it
  is a pure sentinel.

## 7. Region / lat-long
- No lat/long columns (removed by organisers, §2/§8). No explicit region ID
  column present. We do **not** reconstruct coordinates or region from any
  external source. Any region-like structure is only used if it emerges from the
  supplied features themselves.

## 8. Terminology check
- No "EU Adjustments" mechanism appears anywhere in the supplied files or the
  Trustworthiness PDF. Flagged for human review; not coded against (§26).

## Implications for modelling
1. Features must be **window-invariant**: aggregate over valid months only.
2. Encode **missingness as signal**: window length, S2-cloud-in-window count,
   per-sensor valid counts, S2 availability ratio.
3. **Validate with masking simulation**, not a plain random split, so CV
   reflects the real 4–6 month test regime.
4. Tree models with native NaN handling (HistGBM / LightGBM / XGBoost) fit the
   ragged, partially-missing structure without destructive imputation.
