#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=none
#SBATCH --time=00:30:00
#SBATCH --job-name=fin-contract29-cpu
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/fin-contract29-cpu-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
R=$B/library-contract-audit-20261013-candidate29
S=$R/source
P=$B/v7-models-v1/env/bin/python
test ! -e "$S"
mkdir -p "$R/tmp"
cp -a "$B/research-readiness-final-20261013-candidate28/source" "$S"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$R/tmp XDG_CACHE_HOME=$R/cache MPLCONFIGDIR=$R/matplotlib
cd "$S"
cp "$M/fin-context-test-nested-tool29.py" tests/test_nested_tool_delivery.py
set +e
$P -m pytest -q tests/test_nested_tool_delivery.py > "$R/baseline-regression.log" 2>&1
baseline_code=$?
set -e
test "$baseline_code" -eq 1
tail -n 16 "$R/baseline-regression.log"
git apply "$M/fin-context-contract29.patch"
$P scripts/build_index.py
$P scripts/build_package.py
$P scripts/validate.py
$P -m pytest -q tests/test_nested_tool_delivery.py tests/test_reader_contract_v8.py tests/test_trading_preflight_v8.py tests/test_trading_terminal_v8.py tests/test_trading_context_repair_v8.py tests/test_trading_access_v9.py tests/test_trading_capabilities.py tests/test_trading_rag_v9.py tests/test_rag_paging_qualification.py tests/test_research_context.py tests/test_rag.py tests/test_multisource_model_study.py
printf 'Reproduced baseline regressions; fixed focused CPU tests and builds passed. No model inference.\n' > "$R/cpu-passed.txt"
