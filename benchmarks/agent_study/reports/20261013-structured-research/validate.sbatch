#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --job-name=fin-single-gpu-validate
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-single-gpu-validate-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
R=$B/research-json-20261013-candidate16
S=$R/source
P=$B/v7-models-v1/env/bin/python
test -f "$B/probe-20261013-candidate11/passed.txt"
test ! -e "$S"
mkdir -p "$R"
cp -a "$B/probe-20261013-candidate11/source" "$S"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$R/tmp XDG_CACHE_HOME=$R/cache MPLCONFIGDIR=$R/matplotlib
mkdir -p "$TMPDIR"
cd "$S"
git apply "$M/fin-context-research-candidate16.patch"
cp "$M/fin-context-test-trading-python-label12.py" tests/test_trading_python_label.py
cp "$M/fin-context-trading-research15.py" benchmarks/agent_study/trading_research_v9.py
cp "$M/fin-context-test-trading-research16.py" tests/test_trading_research_v9.py

$P scripts/build_index.py
$P scripts/build_package.py
$P scripts/validate.py
$P -m pytest -q tests/test_trading_research_v9.py tests/test_trading_python_label.py tests/test_trading_submission_v8.py tests/test_native_response_scope.py tests/test_trading_transport_feedback.py tests/test_multisource_model_study.py tests/test_reasoning_model_track.py tests/test_sampling_failure.py tests/test_native_chat.py tests/test_rag_paging_qualification.py tests/test_trading_replay_availability_v9.py tests/test_trading_context_repair_v8.py tests/test_trading_access_v9.py tests/test_trading_preflight_v8.py tests/test_trading_schema_v8.py tests/test_trading_schema_review_v8.py tests/test_model_matrix_qualification.py tests/test_trading_rag_v9.py tests/test_trading_terminal_v8.py

printf 'Candidate16 structured research regression/build validation passed\n' > "$R/passed.txt"
