#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=04:00:00
#SBATCH --job-name=fin-discovery-canary20
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-discovery-canary20-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
S=$B/algorithm-discovery-20261013-candidate20/source
test -f "$B/algorithm-discovery-20261013-candidate20/passed.txt"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$S"
$B/v7-models-v1/env/bin/python "$M/fin-context-discovery-canary20.py"
