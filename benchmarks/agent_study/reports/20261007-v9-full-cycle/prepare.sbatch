#!/bin/bash
#SBATCH --account=angliece
#SBATCH --partition=beacon
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=fin-v9-full-prepare
#SBATCH --output=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005/v9-full-prepare-%j.log
set -euo pipefail
B=/beacon-projects/radfm/wy891/fin-multisource-models-20261005
M=/beacon-projects/radfm/wy891/fin-manual-multisource-20261005
R=$B/v9-full-cycle-20261007
P=$B/v7-models-v1/env/bin/python
A=$M/fin-v9-full-cycle-release-20261007.zip
test ! -e "$R"
mkdir -p "$R/source" "$R/tmp"
$P -m zipfile -e "$A" "$R/source"
cd "$R/source"
test -f benchmarks/agent_study/data/market-us-20260929/yahoo_raw_20260929.json
export PYTHONPATH=$R/source PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=$R/tmp
export FIN_STUDY_SHARED_RUNTIME=/beacon-projects/radfm/envs/rfj-gpu
$P scripts/build_index.py
$P scripts/build_package.py
$P scripts/validate.py
$P -m pytest -q tests/test_trading_terminal_v8.py tests/test_trading_rag_v9.py \
  tests/test_model_matrix_qualification.py tests/test_trading_capabilities.py tests/test_personal_chat.py
$P -m py_compile benchmarks/agent_study/reports/20261007-v9-full-cycle/run_pair.py
$P - "$A" <<'PY'
import pathlib, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    for name in archive.namelist():
        if name.endswith('/'):
            continue
        if name in ('README.md', 'catalog/index.json') or name.startswith('fin_skills/'):
            assert pathlib.Path(name).read_text() == archive.read(name).decode('utf-8').replace('\r\n', '\n'), name
print('GENERATED_CONTENT_UNCHANGED')
PY
sha256sum "$A" > "$R/validated-archive.sha256"
printf 'V9_FULL_CYCLE_PREPARATION_PASSED\n' > "$R/passed.txt"
cat "$R/passed.txt"
