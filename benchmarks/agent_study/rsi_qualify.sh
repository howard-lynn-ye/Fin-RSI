#!/usr/bin/env bash
# Resource requirements are chosen at submission; no preferred GPU type.
set -euo pipefail
: "${SLURM_JOB_ID:?run under Slurm}"
r="${1:?fresh release root}"
phase="${2:?cpu or model}"
shift 2
mkdir -p "$r/tmp" "$r/cache" "$r/logs"
export TMPDIR="$r/tmp" TMP="$r/tmp" TEMP="$r/tmp"
export XDG_CACHE_HOME="$r/cache/xdg" XDG_CONFIG_HOME="$r/cache/config"
export XDG_DATA_HOME="$r/cache/data" XDG_STATE_HOME="$r/cache/state"
export HF_HOME="$r/cache/hf" HF_MODULES_CACHE="$r/cache/modules"
export TORCHINDUCTOR_CACHE_DIR="$r/cache/inductor" TRITON_CACHE_DIR="$r/cache/triton"
export CUDA_CACHE_PATH="$r/cache/cuda" MPLCONFIGDIR="$r/cache/matplotlib"
export NUMBA_CACHE_DIR="$r/cache/numba" TORCH_HOME="$r/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$r/source"
if test -d "$r/data-tools"; then
  export PYTHONPATH="$r/data-tools:$PYTHONPATH"
fi
export FIN_TASKS_CONFINE=benchmarks.agent_study.linux_sandbox:confine
export FIN_TASKS_REQUIRE_CONFINEMENT=1
PY=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python
cd "$r/source"
exec "$PY" -B -m benchmarks.agent_study.rsi_qualify "$phase" "$r/$phase-$SLURM_JOB_ID" "$@"
