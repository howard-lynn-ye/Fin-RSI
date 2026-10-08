#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%2
#SBATCH --job-name=fin-allocation-full-cycle
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-allocation-full-%A_%a.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
R=$B/v9-allocation-full-cycle-20261008
P=$B/v7-models-v1/env/bin/python
test -f "$M/fin-allocation-pilot-audit-passed-20261008.txt"
sha256sum -c "$B/v9-allocation-20261008-candidate1/validated-patch.sha256"
S=$B/v9-allocation-20261008-candidate1/source
mkdir -p "$R/tmp"
cd "$S"
export PYTHONPATH=$S PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
INVENTORY=$R/tmp/shard-$SLURM_ARRAY_TASK_ID.tsv
$P - "$SLURM_ARRAY_TASK_ID" > "$INVENTORY" <<'PY'
import json, sys
m = json.load(open('/beacon-projects/radfm/wy891/fin-multisource-models-20261005/v9-allocation-full-cycle-20261008/launcher/manifest.json'))
shard = int(sys.argv[1])
assert shard in (0, 1)
for i, row in enumerate(m['models']):
    if i % 2 == shard:
        for seed in m['seeds']:
            print(row['family'], seed, row['cache'])
PY
FAILED=0
BLOCKED_FAMILY=''
while read -r FAMILY SEED CACHE; do
if [ "$FAMILY" = "$BLOCKED_FAMILY" ]; then
  printf 'INFRASTRUCTURE_BLOCKED_NOT_RUN %s %s\n' "$FAMILY" "$SEED"
  continue
fi
case "$CACHE" in
  coder) export HF_HUB_CACHE=/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub ;;
  matrix-v1) export HF_HUB_CACHE=$B/matrix-weights-v1/cache/hf/hub ;;
  matrix-v2) export HF_HUB_CACHE=$B/matrix-weights-v2/cache/hf/hub ;;
  *) exit 2 ;;
esac
PAIR=$R/pairs/$FAMILY-$SEED
export TMPDIR=$R/tmp/$FAMILY-$SEED
export HF_HOME=$TMPDIR/hf XDG_CACHE_HOME=$TMPDIR/xdg MPLCONFIGDIR=$TMPDIR/matplotlib
export TORCH_HOME=$TMPDIR/torch TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
export TRITON_CACHE_DIR=$TMPDIR/triton CUDA_CACHE_PATH=$TMPDIR/cuda
mkdir -p "$TMPDIR"
if [ ! -e "$PAIR" ]; then
  $P -m benchmarks.agent_study.multisource_model_study freeze "$PAIR" --batch matrix_hf \
    --family "$FAMILY" --seed "$SEED" --evidence "$M/evidence.json" --allow-retrospective --interface v9
  $P -m benchmarks.agent_study.multisource_model_study qualify "$PAIR"
fi
if ! $P "$R/launcher/run_pair.py" "$PAIR"; then
  printf 'INCOMPLETE_PAIR_PRESERVED %s %s\n' "$FAMILY" "$SEED"
  BLOCKED_FAMILY=$FAMILY
  FAILED=1
fi
done < "$INVENTORY"
exit "$FAILED"
