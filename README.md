# GeoAI Aquaculture Pond Identification (FAO × ITU) — Solution Package

Reproducible, rule-compliant pipeline for classifying whether a 10 m × 10 m
Sentinel-1/2 patch is an aquaculture pond (`0`=other, `1`=pond).

## Quick start
```bash
pip install -r requirements.txt
python src/train_baseline.py                       # §22 reference + robust CV
python src/train_model.py --model ensemble         # candidate models + checkpoints
python src/generate_submission.py --model ensemble \
       --out submissions/submission_checkpoint.csv # exact Zindi CSV (§12)
```
All seeds fixed (SEED=42). CPU-only, runs in seconds.

## What's inside
```
src/common.py              data IO, masking simulator, temporal feature engineering
src/experiment_logger.py   experiment_log.csv writer (§23)
src/train_baseline.py      §22 baseline: conventional + robust masking CV
src/train_model.py         LightGBM/XGBoost/CatBoost/HistGBM + ensemble, checkpoint/resume
src/generate_submission.py fits on full train, predicts masked test, writes submission
slurm/train_checkpoint.slurm  RBCCPS/ROCm job (24h checkpoint-resume, HIP_VISIBLE_DEVICES)
docs_checkpoint/           data_audit.md, decisions_log.md, phase2_writeup.md, progress.md
experiments_checkpoint/    experiment_log.csv
submissions/               submission_checkpoint.csv
data_checkpoint/raw/       Train.csv, Test.csv, Trustworthiness_Evaluation.pdf
handoff.md                 single-file resume point for a new session/account
```

## Key results (LightGBM, seed 42, real CV)
| Validation | F1 | ROC-AUC | Weighted (0.6F1+0.4AUC) |
|---|---|---|---|
| Conventional 5-fold (optimistic) | 0.9759 | 0.9969 | **0.9843** |
| **Robust masking 5-fold (realistic)** | 0.9008 | 0.9748 | **0.9304** ±0.003 |

The ~5-point gap is the price of the 4–6 month masking shift; the robust number
is the honest private-test estimate.

## Design in one paragraph
Train rows are fully observed; test rows expose only a consecutive **4/5/6-month
window** (defined by Sentinel-1 presence), with Sentinel-2 additionally
cloud-masked within the window. Every feature therefore aggregates over **valid
months only, per sensor** (window-invariant), and missingness is encoded as
signal (window length, S2-cloud count, per-sensor valid counts). Validation
**simulates the test masking on train** so scores reflect the real challenge.
Tree models with native NaN handling, class weighting for imbalance, and the
**fixed 0.5 threshold** (never tuned) for `TargetF1`.

## Compliance (see §28 audit in decisions_log / phase2_writeup)
Only official Train/Test used · no external data/labels/geo · no lat-long
reconstruction · −9999 treated as missing · 4/5/6 windows + S1/S2 missingness
handled · seeds fixed · no AutoML/paid services/custom packages · no leakage ·
**no custom threshold, no probability manipulation** · submission format verified.
