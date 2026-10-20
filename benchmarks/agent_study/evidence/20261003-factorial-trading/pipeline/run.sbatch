#!/bin/bash
#SBATCH --job-name=fin-factorial-pair
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003/logs/pair-%A_%a.out
#SBATCH --error=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003/logs/pair-%A_%a.err
set -euo pipefail
root=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003
source "$root/scripts/gpu-env.sh"
cd "$root/source"
exec /beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python "$root/scripts/run.py" "$SLURM_ARRAY_TASK_ID"
