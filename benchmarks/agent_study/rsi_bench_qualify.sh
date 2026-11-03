#!/usr/bin/env bash
# RSI benchmark, stage 1: GPU model qualification on dev tasks (benchmarks/agent_study/rsi_bench.py).
#
#   sbatch --account=angliece --partition=beacon --qos=medium --nodes=1 --ntasks=1 \
#          --cpus-per-task=4 --mem=64G --gres=gpu:<N or type:N> --time=06:00:00 \
#          --job-name=fin-rsi-qualify rsi_bench_qualify.sh ROOT MODEL_IDS
#
# ROOT holds source/ (a git checkout of this repository at a recorded commit). MODEL_IDS is a
# comma-separated list of inventory ids. Environment and caches follow codegen-enforced-v2.
set -euo pipefail
: "${SLURM_JOB_ID:?run under Slurm}"
r="${1:?ROOT}"
models="${2:?MODEL_IDS}"
B=/beacon-projects/radfm/wy891
mkdir -p "$r/tmp" "$r/cache" "$r/logs"
export TMPDIR="$r/tmp" TMP="$r/tmp" TEMP="$r/tmp"
export XDG_CACHE_HOME="$r/cache/xdg" XDG_CONFIG_HOME="$r/cache/config" XDG_DATA_HOME="$r/cache/data"
export XDG_STATE_HOME="$r/cache/state" HF_HOME="$r/cache/hf" HF_MODULES_CACHE="$r/cache/modules"
export TORCHINDUCTOR_CACHE_DIR="$r/cache/inductor" TRITON_CACHE_DIR="$r/cache/triton"
export CUDA_CACHE_PATH="$r/cache/cuda" MPLCONFIGDIR="$r/cache/matplotlib" NUMBA_CACHE_DIR="$r/cache/numba"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$r/source"
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export FIN_TASKS_READONLY="$FIN_STUDY_SHARED_RUNTIME:$r/source"
export FIN_TASKS_CONFINE=benchmarks.agent_study.linux_sandbox:confine
PY="$B/fin-skills-audit-20260921/env/bin/python"
INV="$B/fin-multisource-models-20261005/free-matrix-v11-20261017/allocations/2366254/cache-inventory-private.json"
cd "$r/source"
git rev-parse HEAD > "$r/source-commit.txt" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total --format=csv > "$r/logs/gpu-$SLURM_JOB_ID.txt" || true
"$PY" -B -m benchmarks.agent_study.rsi_bench qualify "$r" --inventory "$INV" --models "$models"
date -u +%FT%TZ > "$r/completion-$SLURM_JOB_ID.txt"
