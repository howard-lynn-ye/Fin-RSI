#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=fin-access-ledger-export
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-access-ledger-export-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
S=$B/v9-access-20261013-candidate4/source
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$S"

$B/v7-models-v1/env/bin/python "$M/fin-context-audit-access-matrix3.py" "$S" "$B/v9-access-matrix-20261013-r4/pairs" "$B/v9-access-public-20261013-snapshot3" --families 7b qwen-general-7b qwen3-4b granite-8b phi4-mini smollm3-3b mistral12b falcon3-10b olmo3-7b 14b 32b mistral-small-24b deepseek-r1-distill-14b

$B/v7-models-v1/env/bin/python "$M/fin-context-access-failure-audit3.py"
