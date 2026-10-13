#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=fin-schema-tokens
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-schema-tokens-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
R=$B/v9-context-20261013-candidate6
S=$R/source
P=$B/v7-models-v1/env/bin/python
test -f "$R/passed.txt"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=$R/tmp XDG_CACHE_HOME=$R/cache HF_HOME=$R/hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$S"
$P "$M/fin-context-qualify-schema.py" "$R/tokenizer-qualification.json"
