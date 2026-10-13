#!/bin/bash
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
R=$B/algorithm-discovery-20261013-candidate20
S=$R/source
P=$B/v7-models-v1/env/bin/python
test -f "$B/reader-contract-20261013-candidate19/passed.txt"
test ! -e "$S"
mkdir -p "$R"
cp -a "$B/reader-contract-20261013-candidate19/source" "$S"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$R/tmp XDG_CACHE_HOME=$R/cache MPLCONFIGDIR=$R/matplotlib
mkdir -p "$TMPDIR"
cd "$S"
git apply "$M/fin-context-algorithm-discovery20.patch"
$P scripts/build_index.py
$P scripts/build_package.py
$P scripts/validate.py
$P -m pytest -q tests/test_algorithm_discovery_contract.py tests/test_algorithms.py tests/test_reader_contract_v8.py tests/test_trading_context_repair_v8.py tests/test_trading_research_v9.py tests/test_trading_terminal_v8.py tests/test_trading_rag_v9.py tests/test_trading_replay_availability_v9.py tests/test_trading_access_v9.py tests/test_trading_preflight_v8.py tests/test_multisource_model_study.py
printf 'Candidate20 discovery contracts: build and regression validation passed\n' > "$R/passed.txt"
