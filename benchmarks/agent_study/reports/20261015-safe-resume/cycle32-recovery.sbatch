#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --job-name=fin-32b-recovery34
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-32b-recovery34-%j.log
# Supply the reviewed compatible GPU request at submission; no assumed GPU default.
# Never submit while ANY other fin-skills job is queued/running.
set -euo pipefail
previous_job=${1:?Pass the ended original or recorded recovery job ID}
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
S=$B/library-contract-audit-20261013-candidate29/source
O=$B/full32-cycle-20261015-r33
P=$B/v7-models-v1/env/bin/python
test ! -f /beacon-projects/radfm/wy891/BEACON_CONNECTION_PAUSED.md
test -f "$B/library-contract-audit-20261013-candidate29/cpu-passed.txt"
mkdir -p "$O/tmp"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_CACHE=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub
export TMPDIR=$O/tmp HF_HOME=$O/hf XDG_CACHE_HOME=$O/xdg MPLCONFIGDIR=$O/matplotlib TORCH_HOME=$O/torch
export TORCHINDUCTOR_CACHE_DIR=$O/inductor TRITON_CACHE_DIR=$O/triton CUDA_CACHE_PATH=$O/cuda
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$S"
"$P" "$M/fin-context-safe-resume34.py" --config "$M/fin-context-safe-resume34.json" --previous-job "$previous_job"
