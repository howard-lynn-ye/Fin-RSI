#!/bin/bash
# Lightweight, one-shot submission guard. Never cancels jobs or changes accounts.
set -euo pipefail
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
MODE=${1:?Use prepare, run0, or run1}
case "$MODE" in prepare|run0|run1) ;; *) exit 2 ;; esac
test ! -f /beacon-projects/radfm/wy891/BEACON_CONNECTION_PAUSED.md
RECEIPT=$M/v9-full-$MODE-submission.txt
if [ -f "$RECEIPT" ]; then
  cat "$RECEIPT"
  printf 'Already submitted; inspect this job instead of creating a duplicate.\n'
  exit 0
fi
POLICY=$(scontrol show assoc_mgr flags=assoc | awk '
  /^ClusterName=/{keep=($0 ~ /Account=angliece / && $0 ~ /UserName=wy891\(/ && $0 ~ /Partition=beacon /)}
  keep && /MaxSubmitJobs=/{print}')
if [[ ! "$POLICY" =~ MaxSubmitJobs=([0-9]+)\(([0-9]+)\) ]]; then
  printf 'Cannot verify current submission quota; no submission attempted.\n' >&2
  exit 2
fi
LIMIT=${BASH_REMATCH[1]}
USED=${BASH_REMATCH[2]}
if (( USED >= LIMIT )); then
  printf 'No submission slot: %s/%s occupied. No sbatch attempted.\n' "$USED" "$LIMIT"
  exit 75
fi
cd "$M"
if [ "$MODE" = prepare ]; then
  JOB=$(sbatch --parsable "$M/fin-v9-full-cycle-prepare-release1.sbatch")
else
  PREP=$(cat "$M/v9-full-prepare-submission.txt")
  [[ "$PREP" =~ ^[0-9]+$ ]]
  SHARD=${MODE#run}
  JOB=$(sbatch --parsable --array="$SHARD%1" --dependency="afterok:$PREP" \
    "$M/fin-v9-full-cycle-run-release1.sbatch")
fi
printf '%s\n' "$JOB" > "$RECEIPT"
printf '%s submitted as %s\n' "$MODE" "$JOB"
