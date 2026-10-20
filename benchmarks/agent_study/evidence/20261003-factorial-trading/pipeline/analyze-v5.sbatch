#!/bin/bash
#SBATCH --job-name=fin-v5-posthoc
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003/logs/v5-analysis-%j.out
#SBATCH --error=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003/logs/v5-analysis-%j.err
set -euo pipefail
root=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003
source "$root/scripts/gpu-env.sh"
cd "$root/source"
exec /beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python "$root/scripts/analyze-v5.py"
