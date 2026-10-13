"""Two-model control-filter canary. Never runs or scores the investment cycle."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.agent_study import multisource_model_study as study

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
M = Path('/beacon-projects/radfm/wy891/fin-manual-multisource-20261005')
R = B/'control-canary-20261013-r21'
S = B/'control-filter-20261013-candidate21/source'
assert Path(study.__file__).resolve().is_relative_to(S)
assert (S.parent/'passed.txt').exists()
assert study.old.load(S.parent/'tokenizer-qualification.json')['passed']
assert not (R/'started.json').exists(), 'Preserve every original attempt; no automatic replay'
R.mkdir(parents=True, exist_ok=True)
study.old.write(R/'started.json', dict(time_utc=datetime.now(timezone.utc).isoformat(),
    job=os.environ['SLURM_JOB_ID'], families=['mistral12b', 'falcon3-10b'], seed=11,
    phase='interface_only', no_full_cycle=True, external_model_calls=False))
for family, cache in (
    ('mistral12b', Path('/beacon-projects/radfm/wy891/fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub')),
    ('falcon3-10b', B/'matrix-weights-v2/cache/hf/hub')):
    root = R/'pairs'/f'{family}-11'
    assert not root.exists()
    tmp = R/'tmp'/family
    tmp.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HUB_CACHE=str(cache), TMPDIR=str(tmp),
        HF_HOME=str(tmp/'hf'), XDG_CACHE_HOME=str(tmp/'xdg'),
        MPLCONFIGDIR=str(tmp/'matplotlib'), TORCH_HOME=str(tmp/'torch'),
        TORCHINDUCTOR_CACHE_DIR=str(tmp/'inductor'), TRITON_CACHE_DIR=str(tmp/'triton'),
        CUDA_CACHE_PATH=str(tmp/'cuda'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    base = [sys.executable, '-m', 'benchmarks.agent_study.multisource_model_study']
    for phase, command in (
        ('freeze', base+['freeze', str(root), '--batch', 'matrix_hf', '--family', family,
            '--seed', '11', '--evidence', str(M/'evidence.json'), '--allow-retrospective', '--interface', 'v9']),
        ('qualify', base+['qualify', str(root)]),
        ('model_preflight', [sys.executable, str(M/'fin-context-control-preflight21.py'), str(root)])):
        result = subprocess.run(command, cwd=S, env=env)
        event = dict(family=family, seed=11, phase=phase, exit_code=result.returncode,
            time_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
            driver_sha256=study.old.sha(Path(__file__)), job=os.environ['SLURM_JOB_ID'])
        study.old.write(R/f'{family}-{phase}.json', event)
        print(json.dumps(event), flush=True)
        if result.returncode:
            break
print('Interface canary finished. Inspect errors and tool receipts, not only completed decisions.', flush=True)
