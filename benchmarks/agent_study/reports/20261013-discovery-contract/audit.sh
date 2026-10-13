#!/bin/bash
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
P=$B/v7-models-v1/env/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=$B/schema-decoder-20261013/site
$P "$M/fin-context-access-audit-matrix4.py" "$B/probe-20261013-candidate11/source" "$B/v9-access-matrix-20261013-r8/pairs" "$B/cycle-public-20261013-progress4-original" --families phi4-mini mistral12b
$P "$M/fin-context-access-audit-matrix4.py" "$B/research-json-20261013-candidate16/source" "$B/v9-access-matrix-20261013-r12/pairs" "$B/cycle-public-20261013-progress4-structured-general" --families qwen-general-7b
$P "$M/fin-context-access-audit-matrix4.py" "$B/research-json-20261013-candidate16/source" "$B/v9-access-matrix-20261013-r13/pairs" "$B/cycle-public-20261013-progress4-structured-other" --families 7b qwen3-4b
