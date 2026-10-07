#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=fin-oct07-validate
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/oct07-validate-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
P=$B/v7-models-v1/env/bin/python
R=$B/publication-20261007-validation-v1
A=$M/fin-oct07-publication-v1.zip
test ! -e "$R"
mkdir -p "$R/source"
$P -m zipfile -e "$A" "$R/source"
cd "$R/source"
export PYTHONPATH=$R/source PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
$P scripts/build_index.py
$P scripts/build_package.py
$P scripts/validate.py
$P -m pytest -q tests/test_multisource_snapshot_report.py
$P benchmarks/agent_study/reports/20261007-multisource/validate_publication.py \
  "$A" "$R/reexport"
printf 'PUBLICATION_VALIDATION_COMPLETE\n'
