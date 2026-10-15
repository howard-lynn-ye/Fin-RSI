#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=08:00:00
#SBATCH --job-name=fin-32b-cycle33
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-32b-cycle33-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
S=$B/library-contract-audit-20261013-candidate29/source
R=$B/research-canary-20261014-r30/pairs/32b-11
O=$B/full32-cycle-20261015-r33
P=$B/v7-models-v1/env/bin/python
test -f "$B/library-contract-audit-20261013-candidate29/cpu-passed.txt"
test -f "$R/research-readiness.json"
test ! -e "$R/inference-started.json"
test ! -e "$O"
mkdir -p "$O/tmp"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_CACHE=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub
export TMPDIR=$O/tmp HF_HOME=$O/hf XDG_CACHE_HOME=$O/xdg MPLCONFIGDIR=$O/matplotlib TORCH_HOME=$O/torch
export TORCHINDUCTOR_CACHE_DIR=$O/inductor TRITON_CACHE_DIR=$O/triton CUDA_CACHE_PATH=$O/cuda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$S"
"$P" "$M/fin-context-cycle32-start33.py"
"$P" -m benchmarks.agent_study.multisource_model_study run "$R"
"$P" -m benchmarks.agent_study.multisource_model_study score "$R"
"$P" "$M/fin-context-audit-cycle32-33.py" "$S" "$B/research-canary-20261014-r30/pairs" "$O/public"
printf 'Complete paired cycle, score and independent full-ledger export passed.\n' > "$O/completed.txt"
