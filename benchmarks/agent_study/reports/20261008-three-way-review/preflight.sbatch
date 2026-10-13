#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%3
#SBATCH --job-name=fin-review-model-preflight
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-review-preflight-%A_%a.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
P=$B/v7-models-v1/env/bin/python
S=$B/v9-review-20261012-candidate4/source
test -f "$B/v9-review-20261012-candidate4/passed.txt"
sha256sum -c "$B/v9-review-20261012-candidate4/validated-patch.sha256"
case "$SLURM_ARRAY_TASK_ID" in
  0) FAMILY=7b; CACHE=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub ;;
  1) FAMILY=granite-8b; CACHE=$B/matrix-weights-v1/cache/hf/hub ;;
  2) FAMILY=qwen3-4b; CACHE=$B/matrix-weights-v2/cache/hf/hub ;;
  *) exit 2 ;;
esac
PAIR=$B/v9-review-model-preflight-20261012/$FAMILY-11
test ! -e "$PAIR"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_CACHE=$CACHE
export TMPDIR=$B/v9-review-model-preflight-20261012/tmp-$FAMILY
export HF_HOME=$TMPDIR/hf XDG_CACHE_HOME=$TMPDIR/xdg MPLCONFIGDIR=$TMPDIR/matplotlib
export TORCH_HOME=$TMPDIR/torch TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export TRITON_CACHE_DIR=$TMPDIR/triton CUDA_CACHE_PATH=$TMPDIR/cuda
mkdir -p "$TMPDIR"
cd "$S"
$P -m benchmarks.agent_study.multisource_model_study freeze "$PAIR" --batch matrix_hf \
  --family "$FAMILY" --seed 11 --evidence "$M/evidence.json" --allow-retrospective --interface v9
$P -m benchmarks.agent_study.multisource_model_study qualify "$PAIR"
$P "$M/fin-model-preflight-20261012.py" "$PAIR"
