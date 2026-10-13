#!/bin/bash
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
R=$B/reader-recovery-20261013-candidate24
S=$R/source
P=$B/v7-models-v1/env/bin/python
test -f "$B/control-filter-20261013-candidate21/passed.txt"
test ! -e "$S"
mkdir -p "$R/tmp"
cp -a "$B/control-filter-20261013-candidate21/source" "$S"
export PYTHONPATH=$S:$B/schema-decoder-20261013/site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=$R/tmp XDG_CACHE_HOME=$R/cache MPLCONFIGDIR=$R/matplotlib
cd "$S"
git apply "$M/fin-context-reader-recovery24.patch"
$P scripts/build_index.py
$P scripts/build_package.py
$P scripts/validate.py
$P -m pytest -q tests/test_reader_contract_v8.py tests/test_model_matrix_qualification.py tests/test_multisource_model_study.py tests/test_reasoning_model_track.py tests/test_native_chat.py tests/test_trading_research_v9.py tests/test_trading_terminal_v8.py tests/test_trading_context_repair_v8.py tests/test_trading_access_v9.py tests/test_trading_preflight_v8.py tests/test_decoder_control_filter.py
printf 'Reader recovery and free-only campaign regression/build validation passed\n' > "$R/passed.txt"
