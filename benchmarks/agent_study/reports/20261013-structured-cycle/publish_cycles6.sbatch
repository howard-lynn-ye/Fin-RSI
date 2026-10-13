#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=fin-cycle-publication
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-cycle-publication-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
R=$B/v9-structured-cycle-publication-20261013
S=$R/source
P=$B/v7-models-v1/env/bin/python
test ! -e "$S"
mkdir -p "$R/tmp"
cp -a "$B/v9-context-20261013-candidate6/source" "$S"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$R/tmp XDG_CACHE_HOME=$R/cache MPLCONFIGDIR=$R/matplotlib
cd "$S"
$P scripts/build_index.py
$P scripts/build_package.py
$P scripts/validate.py
$P "$M/fin-context-publish-cycles6.py" "$B/v9-structured-cycle-public-20261013" "$B/v9-context-model-preflight-20261013-r6"
