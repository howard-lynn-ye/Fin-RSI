#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=fin-context-audit
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-context-audit-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
S=$B/v9-context-20261013-candidate6/source
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export TMPDIR=$B/v9-context-model-preflight-20261013-r6/audit-tmp
mkdir -p "$TMPDIR"
cd "$S"
$B/v7-models-v1/env/bin/python "$M/fin-context-audit-preflight6.py" "$B/v9-context-model-preflight-20261013-r6"
