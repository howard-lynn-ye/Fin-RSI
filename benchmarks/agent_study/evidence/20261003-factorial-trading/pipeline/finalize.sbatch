#!/bin/bash
#SBATCH --job-name=fin-factorial-score
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003/logs/score-%j.out
#SBATCH --error=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003/logs/score-%j.err
set -euo pipefail
root=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003
source "$root/scripts/gpu-env.sh"
cd "$root/source"
exec /beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python "$root/scripts/finalize.py"
