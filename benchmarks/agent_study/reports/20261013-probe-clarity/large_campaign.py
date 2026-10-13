"""Frozen HF campaign driver. Preserve failures, stop that model, continue the roster."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.agent_study import multisource_model_study as study

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
M = Path('/beacon-projects/radfm/wy891/fin-manual-multisource-20261005')
R = B / 'v9-access-matrix-20261013-r9'
S = B / 'probe-20261013-candidate11/source'
families = sys.argv[1:]
assert families and all(f in study.BATCHES['matrix_hf']+study.BATCHES['matrix_reasoning'] for f in families)
assert Path(study.__file__).resolve().is_relative_to(S)
assert (S.parent/'passed.txt').exists()
(R/'events').mkdir(parents=True, exist_ok=True)
for family in families:
    batch = 'matrix_reasoning' if family in study.REASONING_FAMILIES else 'matrix_hf'
    if family in ('7b', '14b', '32b'):
        cache = Path('/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub')
    elif family == 'mistral12b':
        cache = Path('/beacon-projects/radfm/wy891/fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub')
    elif family in ('phi4-mini', 'granite-8b', 'smollm3-3b'):
        cache = B/'matrix-weights-v1/cache/hf/hub'
    else:
        cache = B/'matrix-weights-v2/cache/hf/hub'
    for seed in (11, 23, 37):
        root = R/'pairs'/f'{family}-{seed}'
        if root.exists():
            raise FileExistsError(f'Preserve existing run; inspect before any resume: {root}')
        tmp = R/'tmp'/f'{family}-{seed}'
        tmp.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HF_HUB_CACHE=str(cache), TMPDIR=str(tmp),
                   HF_HOME=str(tmp/'hf'), XDG_CACHE_HOME=str(tmp/'xdg'),
                   MPLCONFIGDIR=str(tmp/'matplotlib'), TORCH_HOME=str(tmp/'torch'),
                   TORCHINDUCTOR_CACHE_DIR=str(tmp/'inductor'), TRITON_CACHE_DIR=str(tmp/'triton'),
                   CUDA_CACHE_PATH=str(tmp/'cuda'))
        base = [sys.executable, '-m', 'benchmarks.agent_study.multisource_model_study']
        phases = [('freeze', base+['freeze', str(root), '--batch', batch, '--family', family,
                    '--seed', str(seed), '--evidence', str(M/'evidence.json'), '--allow-retrospective', '--interface', 'v9']),
                  ('qualify', base+['qualify', str(root)]),
                  ('model_preflight', [sys.executable, str(M/'fin-context-access-matrix-preflight.py'), str(root)]),
                  ('run', base+['run', str(root)]), ('score', base+['score', str(root)])]
        for phase, command in phases:
            result = subprocess.run(command, cwd=S, env=env)
            row = dict(family=family, seed=seed, phase=phase, exit_code=result.returncode,
                time_utc=datetime.now(timezone.utc).isoformat(), job=os.environ['SLURM_JOB_ID'],
                driver_sha256=study.old.sha(Path(__file__)), root=str(root))
            study.old.write(R/'events'/f'{family}-{seed}-{phase}.json', row)
            print(json.dumps(row), flush=True)
            if result.returncode:
                print(f'{family}: stopped after {phase}; failure preserved, no replacement decisions or silent retry.', flush=True)
                break
        if result.returncode:
            break
print('Assigned roster processed; inspect every event. This is not an all-model success flag.', flush=True)
