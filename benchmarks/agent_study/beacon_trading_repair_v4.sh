#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm required}" "${SLURM_ARRAY_TASK_ID:?Array index required}"
r="${1:?staging root}"
family="${2:?model family}"
seeds=(11 23 37)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
cd "$r/source"
export PYTHONPATH="$r/source" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export HF_HUB_CACHE=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
jobroot="$r/batch/$family-$seed"
export TMPDIR="$jobroot/tmp" HF_HOME="$r/cache/hf" XDG_CACHE_HOME="$r/cache/xdg"
export MPLCONFIGDIR="$jobroot/tmp/matplotlib" TORCH_HOME="$r/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$jobroot/tmp/inductor" TRITON_CACHE_DIR="$jobroot/tmp/triton"
export CUDA_CACHE_PATH="$jobroot/tmp/cuda"
mkdir -p "$TMPDIR"
p=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python
"$p" -B -m benchmarks.agent_study.trading_study_v4 run "$jobroot"
"$p" -B -m benchmarks.agent_study.trading_study_v4 score "$jobroot"
"$p" -B -m benchmarks.agent_study.trading_study_v4 aggregate "$r/batch"
