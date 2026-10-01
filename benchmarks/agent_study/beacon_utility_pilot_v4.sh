#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Slurm allocation required}"
r="${1:?experiment staging root}"
cd "$r/source"
export PYTHONPATH="$r/source" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export HF_HUB_CACHE=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="$r/tmp" HF_HOME="$r/cache/hf" XDG_CACHE_HOME="$r/cache/xdg"
export MPLCONFIGDIR="$r/cache/matplotlib" TORCH_HOME="$r/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$r/cache/inductor" TRITON_CACHE_DIR="$r/cache/triton"
export CUDA_CACHE_PATH="$r/cache/cuda"
p=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python
"$p" -B -m benchmarks.agent_study.library_utility_pilot_v4 run "$r/pilot"
"$p" -B -m benchmarks.agent_study.library_utility_pilot_v4 score "$r/pilot"
printf 'completed\n' > "$r/pilot/completion.txt"
