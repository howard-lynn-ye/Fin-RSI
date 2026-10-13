#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=fin-structured-small15
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-structured-small15-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
S=$B/reader-contract-20261013-candidate19/source
test -f "$S/../passed.txt"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
cd "$S"
$B/v7-models-v1/env/bin/python "$M/fin-context-access-matrix-r15.py" qwen-general-7b olmo3-7b 7b qwen3-4b granite-8b phi4-mini smollm3-3b mistral12b falcon3-10b
