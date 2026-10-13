#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=fin-native-sol-gemini
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-native-combined-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
P=$B/v7-models-v1/env/bin/python
SOL=$B/research-json-20261013-candidate16/source
GEMINI=$B/native-input-20261013-candidate17/source
test -f "$SOL/../passed.txt"
test -f "$GEMINI/../passed.txt"
test -f "$B/native-matrix-20261013-r7/relay-allowlist.json"
test -f "$B/native-matrix-20261013-r8/relay-allowlist.json"
test ! -e "$B/native-matrix-20261013-r7/run-started.json"
test ! -e "$B/native-matrix-20261013-r8/run-started.json"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
env PYTHONPATH="$SOL:$B/schema-decoder-20261013/site" "$P" "$M/fin-context-native-campaign7.py" run > "$M/fin-native-sol-combined-${SLURM_JOB_ID}.log" 2>&1 &
sol_pid=$!
env PYTHONPATH="$GEMINI:$B/schema-decoder-20261013/site" "$P" "$M/fin-context-native-campaign8.py" run > "$M/fin-native-gemini-combined-${SLURM_JOB_ID}.log" 2>&1 &
gemini_pid=$!
sol_code=0
gemini_code=0
wait "$sol_pid" || sol_code=$?
wait "$gemini_pid" || gemini_code=$?
printf 'Driver exits: sol=%s gemini=%s; inspect preserved campaign-status files, not just this launcher exit.\n' "$sol_code" "$gemini_code"
test "$sol_code" = 0 && test "$gemini_code" = 0
