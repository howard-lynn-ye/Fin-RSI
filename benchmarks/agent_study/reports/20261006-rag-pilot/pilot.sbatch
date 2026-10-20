#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=fin-v9-rag-pilot
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/rag-v9-pilot-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
S=$B/rag-v9-candidate3-step
R=$B/rag-v9-pilot-20261006
P=$B/v7-models-v1/env/bin/python
test -f "$S/passed.txt"
test -f "$M/rag-v9-release-validated.txt"
test ! -e "$R"
mkdir -p "$R/source" "$R/tmp" "$R/cache"
$P -m zipfile -e "$M/fin-rag-v9-release-ad5a9c7.zip" "$R/source"
cd "$R/source"
mkdir -p benchmarks/agent_study/data
ln -s "$B/r2/source/benchmarks/agent_study/data/market-us-20260929" benchmarks/agent_study/data/market-us-20260929
export PYTHONPATH=$R/source PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=$R/cache/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
for family in qwen3-4b granite-8b smollm3-3b; do
  export HF_HUB_CACHE=$B/matrix-weights-v2/cache/hf/hub
  if [ "$family" = smollm3-3b ] || [ "$family" = granite-8b ]; then export HF_HUB_CACHE=$B/matrix-weights-v1/cache/hf/hub; fi
  if [ "$family" = 14b ]; then export HF_HUB_CACHE=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub; fi
  PAIR=$R/pairs/$family-11
  export TMPDIR=$R/tmp XDG_CACHE_HOME=$R/cache/xdg MPLCONFIGDIR=$R/tmp/matplotlib
  export TORCH_HOME=$R/cache/torch TORCHINDUCTOR_CACHE_DIR=$R/tmp/inductor
  export TRITON_CACHE_DIR=$R/tmp/triton CUDA_CACHE_PATH=$R/tmp/cuda
  $P -m benchmarks.agent_study.multisource_model_study freeze "$PAIR" --batch matrix_hf --family "$family" --seed 11 --evidence "$M/evidence.json" --allow-retrospective --interface v9
  $P -m benchmarks.agent_study.multisource_model_study qualify "$PAIR"
  $P "$M/fin-rag-v9-model-pilot.py" "$PAIR"
done
printf 'V9_RAG_USABILITY_PILOT_COMPLETE\n' > "$R/completed.txt"
