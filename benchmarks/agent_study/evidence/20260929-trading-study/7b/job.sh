#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?}"
r="${1:?}"
cd "$r"
export TMPDIR="$r/tmp" TMP="$r/tmp" TEMP="$r/tmp"
export XDG_CACHE_HOME="$r/cache/xdg" XDG_CONFIG_HOME="$r/cache/config" XDG_DATA_HOME="$r/cache/data" XDG_STATE_HOME="$r/cache/state"
export HF_HOME="$r/cache/hf" HF_DATASETS_CACHE="$r/cache/datasets" HF_MODULES_CACHE="$r/cache/modules"
export PIP_CACHE_DIR="$r/cache/pip" MPLCONFIGDIR="$r/cache/matplotlib" TORCH_HOME="$r/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$r/cache/inductor" TRITON_CACHE_DIR="$r/cache/triton" CUDA_CACHE_PATH="$r/cache/cuda" NUMBA_CACHE_DIR="$r/cache/numba"
export LLAMA_INDEX_CACHE_DIR="$r/cache/llamaindex" TIKTOKEN_CACHE_DIR="$r/cache/tiktoken" NLTK_DATA="$r/cache/nltk"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

mkdir "$r/work"
tar -xzf "$r/source/library-and-runner.tar.gz" -C "$r/work"
cp "$r/source/inputs.json" "$r/source/protocol.json" "$r/"
cp -r "$r/source/data" "$r/data"
export PYTHONPATH="$r/work"
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export HF_HUB_CACHE=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$r/work"
/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python -B -m benchmarks.agent_study.trading_study qualify "$r"
/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python -B -m benchmarks.agent_study.trading_study run "$r" --families 7b
/beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python -B -m benchmarks.agent_study.trading_study score "$r"
date -u +%FT%TZ > "$r/completion.txt"
