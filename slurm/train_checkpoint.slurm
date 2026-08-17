#!/bin/bash
#SBATCH --job-name=pond_aqua
#SBATCH --partition=debug
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1              # <=3 GPUs per job (non-PhD). Trees use CPU; keep 1 for safety.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/pond_%j.out
#SBATCH --error=logs/pond_%j.err

# RBCCPS AMD MI300X / ROCm. Tree models are CPU-bound; GPU optional.
module load rocm/6.2.4 || echo "adjust ROCm module version as required"

# ROCm uses HIP_VISIBLE_DEVICES; keep CUDA var too for portability.
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}

source .venv/bin/activate 2>/dev/null || true

# checkpoint/resume: train_model.py loads model_checkpoints/*.pkl if present,
# so a run killed at the 24h wall limit continues instead of restarting.
python src/train_baseline.py
python src/train_model.py --model ensemble
python src/generate_submission.py --model ensemble \
       --out submissions/submission_checkpoint.csv

echo "Done. Remember to Ctrl+Cmd+S sync back to sashishj@10.72.30.28"
