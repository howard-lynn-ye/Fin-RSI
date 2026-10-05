#!/bin/bash
#SBATCH --job-name=fin-factorial-freeze
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003/logs/freeze-%j.out
#SBATCH --error=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003/logs/freeze-%j.err
set -euo pipefail
root=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003
source "$root/scripts/gpu-env.sh"
cd "$root/source"
exec /beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python "$root/scripts/freeze-pipeline.py"
