#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=fin-oct07-export
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/oct07-export-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
P=$B/v7-models-v1/env/bin/python
R=$B/publication-20261007-v1
test ! -e "$R"
mkdir "$R"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$R"
$P "$M/fin-oct07-response-diagnostics-v1.py"
$P "$M/fin-oct07-publish-snapshot-v1.py" --base "$B" \
  --verification "$B/verified-status-1961154.json" \
  --diagnostics "$B/response-diagnostics-$SLURM_JOB_ID.json" \
  --output "$R/snapshot.json" --private-errors "$R/private-errors.json"
printf 'EXPORT_COMPLETE\n'
