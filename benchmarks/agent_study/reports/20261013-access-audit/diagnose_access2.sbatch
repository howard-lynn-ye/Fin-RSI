#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=fin-access-diagnosis
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-access-diagnosis-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
export PYTHONPATH=$B/v9-context-20261013-candidate6/source:$B/schema-decoder-20261013/site
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub
export TMPDIR=$B/v9-context-model-preflight-20261013-r6/tmp-diagnosis
export HF_HOME=$TMPDIR/hf XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p "$TMPDIR"
cd "$B/v9-context-20261013-candidate6/source"
$B/v7-models-v1/env/bin/python "$M/fin-context-diagnose-access2.py"
