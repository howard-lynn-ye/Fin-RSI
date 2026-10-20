#!/bin/bash
#SBATCH --job-name=fin-factorial-prepare
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --output=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003/logs/prepare-%j.out
#SBATCH --error=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003/logs/prepare-%j.err
set -euo pipefail
root=/beacon-projects/radfm/wy891/fin-skills-factorial-20261003
mkdir "$root/source"
tar -xf /beacon-projects/radfm/wy891/fin-skills-publish-20261002/source.tar -C "$root/source"
cp "$root"/scripts/factorial_*.py "$root/source/benchmarks/agent_study/"
cd "$root/source"
export PYTHONPATH="$root/source" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export TMPDIR="$root/tmp" MPLCONFIGDIR="$root/tmp/matplotlib" XDG_CACHE_HOME="$root/tmp/xdg"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /beacon-projects/radfm/wy891/fin-skills-audit-20260921/env/bin/python "$root/scripts/prepare.py"
