#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --job-name=fin-native-recovery-validate
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-native-recovery-validate-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
R=$B/native-recovery-20261013-candidate6
S=$R/source
P=$B/v7-models-v1/env/bin/python
test -f "$B/v9-decoder-20261013-candidate5/passed.txt"
test ! -e "$S"
mkdir -p "$R"
cp -a "$B/v9-decoder-20261013-candidate5/source" "$S"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$R/tmp XDG_CACHE_HOME=$R/cache MPLCONFIGDIR=$R/matplotlib
mkdir -p "$TMPDIR"
cd "$S"
git apply "$M/fin-context-native-recovery-candidate6.patch"
$P scripts/build_index.py
$P scripts/build_package.py
$P scripts/validate.py
$P -m pytest -q tests/test_reasoning_model_track.py tests/test_sampling_failure.py tests/test_native_chat.py tests/test_rag_paging_qualification.py tests/test_trading_replay_availability_v9.py tests/test_trading_context_repair_v8.py tests/test_trading_access_v9.py tests/test_trading_preflight_v8.py tests/test_trading_schema_v8.py tests/test_trading_schema_review_v8.py tests/test_model_matrix_qualification.py tests/test_trading_rag_v9.py tests/test_trading_terminal_v8.py
$P "$M/fin-context-native-recovery-actual.py"
printf 'Candidate6 regression/build/actual-event replay passed\n' > "$R/passed.txt"